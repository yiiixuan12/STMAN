#metric.py
# ----------------------------
# 损失函数 (Huber + 注意力熵稀疏 + 方差下界 + 可选对称正则)
# ----------------------------
import torch  # <- 补上这一行
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

class STFractalLoss(nn.Module):
    """
    多任务损失函数：
      L_total = α * Huber + β * (entropy(S_s2t)+entropy(S_t2s)) + φ * VarLowerBound + η * ||B_st - B_ts||_F^2
    """
    def __init__(
        self,
        huber_delta: float = 1.0,
        alpha: float = 1.0,   # pred
        beta: float = 0.1,    # sparse
        phi: float = 0.1,     # var
        gamma: float = 0.1,   # var lower bound
        eta_sym: float = 0.0, # 对称性软正则权重（B_st ~ B_ts）
        diversity_weight: float = 0.0,     # 时间多样性损失权重 (防坍缩)
        diversity_threshold: float = 0.3,  # pred_std/true_std 最低比例
        vmg_weight: float = 0.0,           # variance matching gradient weight (smooth anti-collapse)
        qml_weight: float = 0.0,           # quantile matching loss weight (IQR + tail matching)
        horizon_weight_mode: str = "uniform",  # uniform | linear | exp | final
        horizon_last_weight: float = 1.0,
        delta_weight: float = 0.0,         # trend/delta matching loss weight
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.gamma = gamma
        self.eta_sym = eta_sym
        self.diversity_weight = diversity_weight
        self.diversity_threshold = diversity_threshold
        self.vmg_weight = vmg_weight
        self.qml_weight = qml_weight
        self.horizon_weight_mode = horizon_weight_mode
        self.horizon_last_weight = float(horizon_last_weight)
        self.delta_weight = float(delta_weight)

    def _horizon_weights(self, y_true: torch.Tensor) -> Optional[torch.Tensor]:
        mode = str(self.horizon_weight_mode or "uniform").lower()
        last_weight = max(float(self.horizon_last_weight), 1e-6)
        steps = int(y_true.shape[1])
        if mode == "uniform" or steps <= 1 or abs(last_weight - 1.0) < 1e-8:
            return None
        if mode == "linear":
            weights = torch.linspace(1.0, last_weight, steps, device=y_true.device, dtype=y_true.dtype)
        elif mode == "exp":
            weights = torch.exp(torch.linspace(0.0, math.log(last_weight), steps, device=y_true.device, dtype=y_true.dtype))
        elif mode == "final":
            weights = torch.ones(steps, device=y_true.device, dtype=y_true.dtype)
            weights[-1] = last_weight
        else:
            raise ValueError(f"unsupported horizon_weight_mode={self.horizon_weight_mode!r}")
        weights = weights / weights.mean().clamp_min(1e-8)
        return weights.view(1, steps, 1, 1)

    def masked_mae_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
        """
        Masked MAE loss: ignore positions where y_true == null_val (sensor failures).
        When operating in normalized (scaled) space, null_val=0.0 masking is effectively
        a no-op since original zeros are shifted. This is standard practice -- the real
        zero-masking happens in the evaluation metrics on the original scale.
        """
        if math.isnan(null_val):
            mask = ~torch.isnan(y_true)
        else:
            eps = 5e-5
            mask = torch.abs(y_true - null_val) > eps
        mask = mask.float()
        num_valid = torch.sum(mask)
        if num_valid == 0:
            return torch.zeros(1, device=y_true.device, requires_grad=True).squeeze()
        weights = self._horizon_weights(y_true)
        loss = torch.abs(y_pred - y_true) * mask
        if weights is not None:
            weighted_mask = mask * weights
            return torch.sum(loss * weights) / torch.sum(weighted_mask).clamp_min(1.0)
        return torch.sum(loss) / num_valid

    def huber_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        absd = diff.abs()
        cond = absd <= self.huber_delta
        sq = 0.5 * diff.pow(2)
        lin = self.huber_delta * absd - 0.5 * self.huber_delta ** 2
        return torch.where(cond, sq, lin).mean()

    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = (p + eps).clamp_max(1.0)
        return -(p * torch.log(p)).sum(dim=-1)

    def sparse_loss(self, S_s2t: torch.Tensor, S_t2s: torch.Tensor) -> torch.Tensor:
        return self._entropy(S_s2t).mean() + self._entropy(S_t2s).mean()

    def variance_loss(self, H_node: torch.Tensor) -> torch.Tensor:
        var_per_dim = torch.var(H_node, dim=0, unbiased=False)  # [d_model]
        safe_std = torch.sqrt(var_per_dim.clamp_min(1e-8))
        penalty = torch.clamp(self.gamma - safe_std, min=0.0)
        return penalty.mean()

    def symmetry_loss(self, B_st: torch.Tensor, B_ts: torch.Tensor) -> torch.Tensor:
        if self.eta_sym <= 0.0:
            return torch.zeros((), device=B_st.device)
        return self.eta_sym * torch.mean((B_st - B_ts) ** 2)

    def variance_matching_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Match time-std of prediction to time-std of truth (differentiable).
        Unlike diversity_loss (hard threshold), this provides smooth gradient pushing
        pred_std toward true_std. Rewards bidirectional matching — penalizes both
        under-variance (smooth/collapsed) and over-variance (noisy).
        y_true, y_pred: [B, T, N, out_dim]
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        pred_std = torch.std(y_pred, dim=1).clamp_min(1e-8)
        true_std = torch.std(y_true, dim=1).clamp_min(1e-8)
        return (pred_std - true_std).abs().mean()

    def quantile_matching_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Match distribution shape via multiple quantiles (IQR + tails).
        Extends VMG: instead of just std, match q10/q25/q50/q75/q90 across time.
        Forces pred to have similar distributional shape as truth, not just mean/std.
        Higher ROI than VMG alone for non-Gaussian traffic data.
        y_true, y_pred: [B, T, N, out_dim], aggregate over T dim."""
        y_pred = y_pred.float()
        y_true = y_true.float()
        loss = y_pred.new_zeros(())
        quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
        for q in quantiles:
            pq = torch.quantile(y_pred, q, dim=1)
            tq = torch.quantile(y_true, q, dim=1)
            loss = loss + (pq - tq).abs().mean()
        return loss / len(quantiles)

    def temporal_diversity_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Anti-collapse: penalize when prediction temporal variance is much lower than target.
        y_true, y_pred: [B, T, N, out_dim]
        Penalty = max(0, ratio_threshold - pred_std/true_std) averaged over nodes.
        Only activates when pred is significantly less varied than truth."""
        # std across time dimension for each (batch, node, channel)
        y_pred = y_pred.float()
        y_true = y_true.float()
        pred_std = torch.std(y_pred, dim=1).clamp_min(1e-8)  # [B, N, out_dim]
        true_std = torch.std(y_true, dim=1).clamp_min(1e-8)  # [B, N, out_dim]
        ratio = pred_std / true_std  # should be ~1.0 for good predictions
        # Penalize ratio below threshold (e.g. 0.3 means pred varies at least 30% as much as truth)
        penalty = torch.clamp(self.diversity_threshold - ratio, min=0.0)
        return penalty.mean()

    def delta_matching_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Match short-term traffic changes to reduce overly smooth peak/valley forecasts."""
        if y_true.shape[1] < 2:
            return torch.zeros((), device=y_true.device)
        true_delta = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        pred_delta = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        weights = self._horizon_weights(y_true)
        loss = (pred_delta - true_delta).abs()
        if weights is not None:
            loss = loss * weights[:, 1:, :, :]
        return loss.mean()

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        aux: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        L_pred = self.masked_mae_loss(y_true, y_pred, null_val=0.0)
        L_sparse = self.sparse_loss(aux["S_s2t"], aux["S_t2s"])
        L_var = self.variance_loss(aux["H_st_node"])
        L_sym = self.symmetry_loss(aux["B_st"], aux["B_ts"])
        L_div = self.diversity_weight * self.temporal_diversity_loss(y_true, y_pred) if self.diversity_weight > 0 else torch.zeros((), device=y_true.device)
        L_vmg = self.vmg_weight * self.variance_matching_loss(y_true, y_pred) if self.vmg_weight > 0 else torch.zeros((), device=y_true.device)
        L_qml = self.qml_weight * self.quantile_matching_loss(y_true, y_pred) if self.qml_weight > 0 else torch.zeros((), device=y_true.device)
        L_delta = self.delta_weight * self.delta_matching_loss(y_true, y_pred) if self.delta_weight > 0 else torch.zeros((), device=y_true.device)

        L_total = self.alpha * L_pred + self.beta * L_sparse + self.phi * L_var + L_sym + L_div + L_vmg + L_qml + L_delta
        loss_dict = {"total": L_total, "pred": L_pred, "sparse": L_sparse, "var": L_var, "sym": L_sym,
                     "div": L_div if isinstance(L_div, torch.Tensor) else torch.tensor(0.0),
                     "vmg": L_vmg if isinstance(L_vmg, torch.Tensor) else torch.tensor(0.0),
                     "qml": L_qml if isinstance(L_qml, torch.Tensor) else torch.tensor(0.0),
                     "delta": L_delta if isinstance(L_delta, torch.Tensor) else torch.tensor(0.0)}
        return L_total, loss_dict
