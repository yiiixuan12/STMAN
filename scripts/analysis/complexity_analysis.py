#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Complexity Analysis: STMAN (STFractalTransformer) vs Baselines
Computes total trainable parameters and model size in MB (float32).
"""
import sys
import os
import traceback
import importlib

import torch
import torch.nn as nn
import numpy as np

# ---- paths ----
ROOT = "/mnt/nfs/yxwang/code/multifract"
BASELINE_DIR = os.path.join(ROOT, "baseline")
BASELINE_MODEL_DIR = os.path.join(BASELINE_DIR, "model")

sys.path.insert(0, ROOT)
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, BASELINE_MODEL_DIR)

# ---- helper ----
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)

# ---- dataset configs ----
# We use METR-LA as the reference dataset
N_METR = 207
N_BAY = 325
N_PEMS03 = 358
N_PEMS07 = 883
N_PEMS08 = 170

# Use METR-LA as primary comparison
N = N_METR
IN_DIM = 1
SEQ_LEN = 12
PRED_LEN = 12
SF_DIM = 10
TF_DIM = 7

results = []

# =========================================
# 1. STMAN (STFractalTransformer)
# =========================================
print("=" * 70)
print("Instantiating STMAN (STFractalTransformer)...")
try:
    from model import STFractalTransformer
    model = STFractalTransformer(
        n_nodes=N,
        in_dim=IN_DIM,
        d_model=32,
        n_heads=1,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        sf_dim=SF_DIM,
        tf_dim=TF_DIM,
        num_encoder_layers=2,
        num_decoder_layers=1,
        ff_multiplier=4,
        dropout=0.1,
        use_bias_zscore=True,
        residual_delta=True,
    )
    total, trainable = count_params(model)
    results.append(("STMAN (Ours)", total, trainable))
    print(f"  Total: {total:,}  Trainable: {trainable:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# =========================================
# 2. STGCN
# =========================================
print("\nInstantiating STGCN...")
try:
    from types import SimpleNamespace
    # Import STGCN directly from file to avoid model.py shadowing
    spec = importlib.util.spec_from_file_location("STGCN", os.path.join(BASELINE_MODEL_DIR, "STGCN.py"))
    stgcn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stgcn_mod)
    STGCNChebGraphConv = stgcn_mod.STGCNChebGraphConv

    # Build GSO (Chebyshev scaled Laplacian) manually
    adj_np = np.random.rand(N, N).astype(np.float32)
    adj_np = (adj_np + adj_np.T) / 2
    adj_np[adj_np < 0.7] = 0  # sparse
    np.fill_diagonal(adj_np, 0)
    adj = torch.from_numpy(adj_np)

    # Build scaled Laplacian for Cheb manually (avoid importing baseline utils which needs pandas)
    def _build_gso_cheb(adj_tensor, add_self_loops=True):
        A = adj_tensor.clone().float()
        N_ = A.size(0)
        if add_self_loops:
            A = A + torch.eye(N_)
        d = A.sum(dim=1)
        d_inv_sqrt = torch.pow(d.clamp_min(1e-12), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = torch.eye(N_) - D_inv_sqrt @ A @ D_inv_sqrt
        # Scale to [-1, 1]
        try:
            evals = torch.linalg.eigvalsh(L)
            lambda_max = evals[-1].item()
        except:
            lambda_max = 2.0
        L_scaled = (2.0 / max(lambda_max, 1e-6)) * L - torch.eye(N_)
        return L_scaled

    gso = _build_gso_cheb(adj)

    end_channel = PRED_LEN * 1  # out_dim=1
    blocks = [
        [IN_DIM],
        [64, 16, 64],
        [64, 16, 64],
        [64, 128],
        [end_channel],
    ]
    args_ns = SimpleNamespace(
        n_his=SEQ_LEN,
        Kt=3,
        Ks=3,
        act_func="glu",
        graph_conv_type="cheb_graph_conv",
        gso=gso,
        enable_bias=True,
        droprate=0.0,
    )
    model = STGCNChebGraphConv(args_ns, blocks, N)
    total, trainable = count_params(model)
    results.append(("STGCN", total, trainable))
    print(f"  Total: {total:,}  Trainable: {trainable:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# =========================================
# 3. TGCN
# =========================================
print("\nInstantiating TGCN...")
try:
    spec = importlib.util.spec_from_file_location("TGCN", os.path.join(BASELINE_MODEL_DIR, "TGCN.py"))
    tgcn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgcn_mod)
    TGCNForecaster = tgcn_mod.TGCNForecaster
    adj_tgcn = torch.rand(N, N)
    adj_tgcn = (adj_tgcn + adj_tgcn.T) / 2
    adj_tgcn[adj_tgcn < 0.7] = 0
    adj_tgcn.fill_diagonal_(0)

    model = TGCNForecaster(
        adj=adj_tgcn,
        num_nodes=N,
        input_dim=IN_DIM,
        hidden_dim=64,
        out_dim=1,
        horizon=PRED_LEN,
        add_self_loops=True,
        dropout=0.0,
        non_autoregressive=True,
    )
    total, trainable = count_params(model)
    results.append(("TGCN", total, trainable))
    print(f"  Total: {total:,}  Trainable: {trainable:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# =========================================
# 4. STAEformer
# =========================================
print("\nInstantiating STAEformer...")
try:
    # Read STAEformer source and remove torchinfo import
    stae_src_path = os.path.join(BASELINE_MODEL_DIR, "STAEformer.py")
    with open(stae_src_path, 'r') as f:
        stae_src = f.read()
    stae_src = stae_src.replace("from torchinfo import summary", "# from torchinfo import summary")
    # Compile and exec in a temporary module
    import types as _types
    stae_mod = _types.ModuleType("STAEformer")
    stae_mod.__file__ = stae_src_path
    stae_mod.torch = torch
    stae_mod.nn = nn
    exec(compile(stae_src, stae_src_path, 'exec'), stae_mod.__dict__)
    STAEformer = stae_mod.STAEformer
    model = STAEformer(
        num_nodes=N,
        in_steps=SEQ_LEN,
        out_steps=PRED_LEN,
        steps_per_day=288,
        input_dim=3,        # value + tod + dow
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    )
    total, trainable = count_params(model)
    results.append(("STAEformer", total, trainable))
    print(f"  Total: {total:,}  Trainable: {trainable:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# =========================================
# 5. PDFormer
# =========================================
print("\nInstantiating PDFormer...")
try:
    spec = importlib.util.spec_from_file_location("PDFormer", os.path.join(BASELINE_MODEL_DIR, "PDFormer.py"))
    pdf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdf_mod)
    PDFormer = pdf_mod.PDFormer

    adj_np = np.random.rand(N, N).astype(np.float32)
    adj_np = (adj_np + adj_np.T) / 2
    adj_np[adj_np < 0.7] = 0
    np.fill_diagonal(adj_np, 0)

    # Compute hop distance
    G = (adj_np > 0).astype(np.float32)
    INF = 1e12
    D = np.where(G > 0, 1.0, INF)
    np.fill_diagonal(D, 0.0)
    for k in range(N):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    D[D >= INF/2] = N + 1
    sh_mx = D.astype(np.float32)

    # DTW matrix (use hop dist as proxy)
    dtw_matrix = sh_mx.copy()

    # Pattern keys
    s_attn_size = 3
    out_dim_pdf = 1
    pattern_keys = np.zeros((SEQ_LEN, N, s_attn_size, out_dim_pdf), dtype=np.float32)

    data_feature = dict(
        num_nodes=N,
        feature_dim=IN_DIM,
        ext_dim=0,
        num_batches=100,
        dtw_matrix=dtw_matrix,
        adj_mx=None,
        sd_mx=sh_mx,
        sh_mx=sh_mx,
        pattern_keys=pattern_keys,
        scaler=None,
    )

    config = dict(
        embed_dim=64,
        skip_dim=128,
        lape_dim=8,
        geo_num_heads=4,
        sem_num_heads=2,
        t_num_heads=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        s_attn_size=s_attn_size,
        t_attn_size=3,
        enc_depth=3,
        type_ln="pre",
        type_short_path="hop",
        output_dim=out_dim_pdf,
        input_window=SEQ_LEN,
        output_window=PRED_LEN,
        add_time_in_day=False,
        add_day_in_week=False,
        device=torch.device('cpu'),
        world_size=1,
        huber_delta=1.0,
        quan_delta=0.25,
        far_mask_delta=5.0,
        dtw_delta=3,
        use_curriculum_learning=False,
        step_size=2500,
        max_epoch=200,
        task_level=PRED_LEN,
    )

    model = PDFormer(config, data_feature)
    total, trainable = count_params(model)
    results.append(("PDFormer", total, trainable))
    print(f"  Total: {total:,}  Trainable: {trainable:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# =========================================
# 6. LSTNN (MultiscaleMLP)
# =========================================
print("\nInstantiating LSTNN (MultiscaleMLP)...")
try:
    spec = importlib.util.spec_from_file_location("LSTNN", os.path.join(BASELINE_MODEL_DIR, "LSTNN.py"))
    lstnn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstnn_mod)
    MultiscaleMLP = lstnn_mod.MultiscaleMLP
    lstnn_args = dict(
        node_size=N,
        input_len=SEQ_LEN,
        input_dim=3,  # value + tod + dow
        output_len=PRED_LEN,
        patch_len=4,     # for seq_len=12
        stride=4,
        td_size=288,
        dw_size=7,
        d_td=8,
        d_dw=4,
        d_d=32,
        d_spa=8,
        if_time_in_day=True,
        if_day_in_week=True,
        if_spatial=True,
        num_layer=2,
    )
    model = MultiscaleMLP(**lstnn_args)

    # LSTNN uses Conv1x1Auto (lazy init), so we need a forward pass to initialize
    total, trainable = count_params(model)

    # Check for lazy modules
    lazy_count = 0
    for name, m in model.named_modules():
        if hasattr(m, 'conv') and m.conv is None:
            lazy_count += 1

    if lazy_count > 0:
        # Run a dummy forward with properly ranged data to initialize lazy modules
        print(f"  ({lazy_count} lazy Conv1x1Auto modules, running dummy forward...)")
        B_dummy = 2
        dummy_x = torch.zeros(B_dummy, SEQ_LEN, N, 3)
        dummy_x[..., 0] = torch.randn(B_dummy, SEQ_LEN, N)       # values
        dummy_x[..., 1] = torch.rand(B_dummy, SEQ_LEN, N) * 0.99  # time_in_day in [0, 1)
        dummy_x[..., 2] = torch.randint(0, 7, (B_dummy, SEQ_LEN, N)).float()  # day_of_week
        try:
            with torch.no_grad():
                _ = model(dummy_x)
            total, trainable = count_params(model)
        except Exception as e2:
            print(f"  Warning: forward pass failed ({e2}), counts may be incomplete")

    results.append(("LSTNN", total, trainable))
    print(f"  Total: {total:,}  Trainable: {trainable:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# =========================================
# Print comparison table
# =========================================
print("\n")
print("=" * 70)
print("MODEL COMPLEXITY COMPARISON (METR-LA, N=207, seq_len=12, pred_len=12)")
print("=" * 70)
print(f"{'Model':<20} {'Parameters':>15} {'Trainable':>15} {'Size(MB)':>10}")
print("-" * 70)
for name, total, trainable in results:
    size_mb = trainable * 4 / 1024 / 1024
    print(f"{name:<20} {total:>15,} {trainable:>15,} {size_mb:>10.3f}")
print("-" * 70)
print("Note: Size(MB) = trainable_params x 4 bytes (float32)")

# =========================================
# Also check training times from logs
# =========================================
print("\n")
print("=" * 70)
print("TRAINING TIME PER EPOCH (from logs, METR-LA, seq_len=12)")
print("=" * 70)
log_dir = os.path.join(BASELINE_DIR, "logs")
log_files = {
    "STGCN": "METR-LA_stgcn_12.txt",
    "STAEformer": "METR-LA_staeformer_12.txt",
    "LSTNN": "metr_lstnn_12.txt",
}
# Also check PDFormer logs
import glob
pdf_logs = glob.glob(os.path.join(log_dir, "*pdformer*12*"))
if pdf_logs:
    log_files["PDFormer"] = os.path.basename(pdf_logs[0])

for model_name, fname in log_files.items():
    fpath = os.path.join(log_dir, fname)
    if os.path.exists(fpath):
        times = []
        with open(fpath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 11:
                    try:
                        t = float(parts[10])
                        times.append(t)
                    except ValueError:
                        pass
        if times:
            avg_t = sum(times) / len(times)
            print(f"{model_name:<20} avg={avg_t:.1f}s/epoch  (over {len(times)} epochs)")
        else:
            print(f"{model_name:<20} no timing data found in {fname}")
    else:
        print(f"{model_name:<20} log file not found: {fname}")

# Also check STMAN logs
stman_log_dir = os.path.join(ROOT, "checkpoints")
if os.path.isdir(stman_log_dir):
    stman_logs = glob.glob(os.path.join(stman_log_dir, "**/*metr*12*.txt"), recursive=True)
    if not stman_logs:
        stman_logs = glob.glob(os.path.join(stman_log_dir, "**/*.txt"), recursive=True)
    if stman_logs:
        print(f"\nSTMAN log files found: {stman_logs[:5]}")

# Also check other pred_len for more data
print("\n")
print("=" * 70)
print("TRAINING TIME PER EPOCH (from logs, various datasets/pred_lens)")
print("=" * 70)
for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.txt'):
        continue
    fpath = os.path.join(log_dir, fname)
    times = []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 11:
                try:
                    t = float(parts[10])
                    times.append(t)
                except ValueError:
                    pass
    if times:
        avg_t = sum(times) / len(times)
        print(f"  {fname:<45} avg={avg_t:>8.1f}s/epoch  ({len(times):>3} epochs)")


# =========================================
# Multi-dataset parameter comparison
# =========================================
print("\n")
print("=" * 70)
print("STMAN PARAMETER COUNTS ACROSS DATASETS")
print("=" * 70)
datasets = {
    "METR-LA": (207, 10, 7),
    "PEMS-BAY": (325, 10, 7),
    "PEMS03": (358, 10, 7),
    "PEMS07": (883, 10, 7),
    "PEMS08": (170, 10, 7),
}
# Load actual fractal dims
for ds_name in datasets:
    sf_path = os.path.join(ROOT, f"spatial_fractal_vectors_{ds_name}.npy")
    tf_path = os.path.join(ROOT, f"train_timefractals_{ds_name}.npy")
    if os.path.exists(sf_path) and os.path.exists(tf_path):
        sf = np.load(sf_path)
        tf = np.load(tf_path)
        datasets[ds_name] = (sf.shape[0], sf.shape[1], tf.shape[1])
        print(f"  {ds_name}: N={sf.shape[0]}, sf_dim={sf.shape[1]}, tf_dim={tf.shape[1]}")

print(f"\n{'Dataset':<15} {'N':>5} {'sf_dim':>7} {'tf_dim':>7} {'Params':>12} {'Size(MB)':>10}")
print("-" * 60)
for ds_name, (n_nodes, sf_d, tf_d) in datasets.items():
    try:
        m = STFractalTransformer(
            n_nodes=n_nodes, in_dim=1, d_model=32, n_heads=1,
            seq_len=12, pred_len=12,
            sf_dim=sf_d, tf_dim=tf_d,
            num_encoder_layers=2, num_decoder_layers=1,
            ff_multiplier=4, dropout=0.1,
        )
        t, tr = count_params(m)
        size_mb = tr * 4 / 1024 / 1024
        print(f"{ds_name:<15} {n_nodes:>5} {sf_d:>7} {tf_d:>7} {tr:>12,} {size_mb:>10.3f}")
    except Exception as e:
        print(f"{ds_name:<15} ERROR: {e}")
