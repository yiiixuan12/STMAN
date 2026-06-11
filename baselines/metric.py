# metric.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class HuberRegLoss(nn.Module):
    """
    仅预测任务的 Huber 损失:
      loss = { 0.5*(y-y_hat)^2, |y-y_hat| <= delta
             { delta*|y-y_hat| - 0.5*delta^2, otherwise
    """
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        absd = diff.abs()
        sq = 0.5 * diff.pow(2)
        lin = self.delta * absd - 0.5 * (self.delta ** 2)
        loss = torch.where(absd <= self.delta, sq, lin)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
