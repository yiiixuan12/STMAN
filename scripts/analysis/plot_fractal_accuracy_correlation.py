#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate fractal feature vs prediction accuracy correlation figure.
Produces a 1x2 subplot:
  (a) Delta-alpha vs STMAN per-node MAE  (positive r: higher complexity -> higher error)
  (b) Spatial fractal dimension D vs STMAN per-node MAE, colored by h(2)

Uses METR-LA, 288-step prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports ──
sys.path.insert(0, "/mnt/nfs/yxwang/code/multifract")
from model import STFractalTransformer
from utils import (
    load_adjacency_csv,
    TrafficDataset,
    StandardScaler,
    MinMaxScaler,
)
from train import inverse_transform_batch

# ── matplotlib (non-interactive) ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

# ══════════════════════════════════════════════════════════════
# 0.  Configuration
# ══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda:0")  # will be remapped by CUDA_VISIBLE_DEVICES
PROJECT = "/mnt/nfs/yxwang/code/multifract"
CKPT_PATH = os.path.join(PROJECT, "checkpoints", "METR-LA_288_best.pt")
DATA_CSV = os.path.join(PROJECT, "Datasets", "METR-LA", "METR-LA.csv")
ADJ_CSV = os.path.join(PROJECT, "Datasets", "METR-LA", "adj_matrix.csv")
SF_NPY = os.path.join(PROJECT, "spatial_fractal_vectors_METR-LA.npy")
TF_NPY = os.path.join(PROJECT, "train_timefractals_METR-LA.npy")
OUT_PDF = "/mnt/nfs/yxwang/paper/traffic/figures/fractal_accuracy_correlation.pdf"

SPLIT_RATE = 0.6
VAL_RATE = 0.2
SEQ_LEN = 288
PRED_LEN = 288
BATCH_SIZE = 4  # small to fit GPU memory for long sequences

# Best baseline MAE at METR-LA 288-step (PDFormer from Table III)
BEST_BASELINE_MAE = 10.42

# ══════════════════════════════════════════════════════════════
# 1.  Load data
# ══════════════════════════════════════════════════════════════
print("[1/5] Loading data ...")
raw = pd.read_csv(DATA_CSV).values                       # (T, 207)
X = np.expand_dims(raw, axis=-1).astype(np.float32)       # (T, 207, 1)
T_total, N, F_dim = X.shape

n_train = int(T_total * SPLIT_RATE)
n_val = int(T_total * VAL_RATE)
X_train = X[:n_train]
X_val = X[n_train : n_train + n_val]
X_test = X[n_train + n_val:]

# Fit scaler on training set (minmax, matching training config)
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

test_dataset = TrafficDataset(X_test_sc, SEQ_LEN, PRED_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)
print(f"  Test samples: {len(test_dataset)}, batches: {len(test_loader)}")

# ══════════════════════════════════════════════════════════════
# 2.  Load fractal features & adjacency
# ══════════════════════════════════════════════════════════════
print("[2/5] Loading fractal features ...")
SF = np.load(SF_NPY)   # (207, 10)
TF = np.load(TF_NPY)   # (207, 7)
print(f"  SF shape: {SF.shape},  TF shape: {TF.shape}")

# Extract key features
# TF vector: [h(-4), h(0), h(2), h(4), h(5), delta_alpha, asymmetry]
delta_alpha = TF[:, -2]       # spectrum width
fractal_dim_D = SF[:, -2]     # spatial fractal dimension
h2 = TF[:, 2]                 # h(q=2), the 3rd element (q_targets=[-4, 0, 2, 4, 5])

SF_t = torch.tensor(SF, dtype=torch.float32)
TF_t = torch.tensor(TF, dtype=torch.float32)
A = load_adjacency_csv(ADJ_CSV, n_nodes=N, add_self_loops=True,
                       self_loop_weight=1.0, normalize=None)

# Compute node degree from original adjacency (before self-loop addition)
adj_raw = pd.read_csv(ADJ_CSV, header=None).values
adj_binary = (adj_raw > 0).astype(float)
np.fill_diagonal(adj_binary, 0)
node_degree = adj_binary.sum(axis=1)

# ══════════════════════════════════════════════════════════════
# 3.  Build model & run inference  ->  per-node MAE
# ══════════════════════════════════════════════════════════════
print("[3/5] Loading model & running inference ...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg = ckpt["config"]

model = STFractalTransformer(
    n_nodes=cfg["n_nodes"],
    in_dim=cfg["in_dim"],
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    seq_len=cfg["seq_len"],
    pred_len=cfg["pred_len"],
    sf_dim=cfg["sf_dim"],
    tf_dim=cfg["tf_dim"],
    num_encoder_layers=cfg["enc_layers"],
    num_decoder_layers=cfg["dec_layers"],
    ff_multiplier=cfg["ff_mult"],
    dropout=cfg["dropout"],
    use_bias_zscore=True,
    residual_delta=True,
)
model.load_state_dict(ckpt["model"], strict=True)
model.to(DEVICE).eval()

A_dev = A.to(DEVICE)
SF_dev = SF_t.to(DEVICE)
TF_dev = TF_t.to(DEVICE)

# Accumulate per-node absolute errors
node_abs_errors = np.zeros(N, dtype=np.float64)   # sum of |y - y_hat|
node_counts = np.zeros(N, dtype=np.float64)        # number of (time_step * sample) per node

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        y_hat, _ = model(x_batch, A_dev, SF_dev, TF_dev, y_in=None)

        # Inverse-transform to original scale
        y_true_np = inverse_transform_batch(y_batch, scaler)   # (B, T_out, N, 1)
        y_pred_np = inverse_transform_batch(y_hat, scaler)     # (B, T_out, N, 1)

        abs_err = np.abs(y_true_np - y_pred_np)                # (B, T_out, N, 1)
        # Sum over batch & time & feature dims -> per node
        node_abs_errors += abs_err.sum(axis=(0, 1, 3))         # (N,)
        node_counts += abs_err.shape[0] * abs_err.shape[1] * abs_err.shape[3]

per_node_mae = node_abs_errors / node_counts
overall_mae = per_node_mae.mean()
print(f"  STMAN overall MAE = {overall_mae:.4f}")
print(f"  Per-node MAE range: [{per_node_mae.min():.4f}, {per_node_mae.max():.4f}]")

# ══════════════════════════════════════════════════════════════
# 4.  Compute derived quantities
# ══════════════════════════════════════════════════════════════
print("[4/5] Computing correlations ...")

# (a) delta_alpha vs STMAN per-node MAE
r_a, p_a = stats.pearsonr(delta_alpha, per_node_mae)
print(f"  (a) Pearson r(delta_alpha, node_MAE) = {r_a:.4f},  p = {p_a:.2e}")

# (b) fractal dim vs node MAE (both Pearson and Spearman due to skewed D)
r_b, p_b = stats.pearsonr(fractal_dim_D, per_node_mae)
rho_b, p_rho_b = stats.spearmanr(fractal_dim_D, per_node_mae)
print(f"  (b) Pearson  r(D, node_MAE) = {r_b:.4f},  p = {p_b:.2e}")
print(f"  (b) Spearman rho(D, node_MAE) = {rho_b:.4f},  p = {p_rho_b:.2e}")

# Extra correlations for completeness
r_h2, p_h2 = stats.pearsonr(h2, per_node_mae)
print(f"  Extra: Pearson r(h(2), node_MAE) = {r_h2:.4f},  p = {p_h2:.2e}")
r_deg, p_deg = stats.pearsonr(node_degree, per_node_mae)
print(f"  Extra: Pearson r(degree, node_MAE) = {r_deg:.4f},  p = {p_deg:.2e}")

# ══════════════════════════════════════════════════════════════
# 5.  Plot
# ══════════════════════════════════════════════════════════════
print("[5/5] Generating figure ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ── colour palette ──
DOT_ALPHA = 0.6
DOT_SIZE = 22
REG_COLOR = "#d62728"
REG_LW = 2.0

# ────────────────────────────── (a) ──────────────────────────
ax = axes[0]
ax.scatter(delta_alpha, per_node_mae, s=DOT_SIZE, alpha=DOT_ALPHA,
           edgecolors="none", c="#1f77b4", zorder=2)

# Linear regression line
slope_a, intercept_a = np.polyfit(delta_alpha, per_node_mae, 1)
x_line = np.linspace(delta_alpha.min(), delta_alpha.max(), 200)
ax.plot(x_line, slope_a * x_line + intercept_a, color=REG_COLOR, lw=REG_LW,
        zorder=3)

# Annotate Pearson r & p
p_str_a = f"{p_a:.2e}" if p_a < 0.01 else f"{p_a:.3f}"
ax.text(0.05, 0.95,
        f"$r$ = {r_a:.3f}\n$p$ = {p_str_a}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="gray", linewidth=0.5))

ax.set_xlabel(r"Multifractal spectrum width $\Delta\alpha$", fontsize=12)
ax.set_ylabel("STMAN per-node MAE (mph)", fontsize=12)
ax.set_title("(a)", fontsize=13, fontweight="bold", loc="left")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=11)

# ────────────────────────────── (b) ──────────────────────────
ax = axes[1]
sc = ax.scatter(fractal_dim_D, per_node_mae, s=DOT_SIZE, alpha=DOT_ALPHA,
                c=h2, cmap="viridis", edgecolors="none", zorder=2)

# Linear regression line
slope_b, intercept_b = np.polyfit(fractal_dim_D, per_node_mae, 1)
x_line_b = np.linspace(fractal_dim_D.min(), fractal_dim_D.max(), 200)
ax.plot(x_line_b, slope_b * x_line_b + intercept_b, color=REG_COLOR, lw=REG_LW,
        zorder=3)

# Annotate both Pearson & Spearman
p_str_b = f"{p_b:.2e}" if p_b < 0.01 else f"{p_b:.3f}"
p_str_rho = f"{p_rho_b:.2e}" if p_rho_b < 0.01 else f"{p_rho_b:.3f}"
ax.text(0.05, 0.95,
        f"$r$ = {r_b:.3f}, $p$ = {p_str_b}\n"
        f"$\\rho$ = {rho_b:.3f}, $p$ = {p_str_rho}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="gray", linewidth=0.5))

cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$h(2)$", fontsize=11)
cbar.ax.tick_params(labelsize=10)

ax.set_xlabel(r"Spatial fractal dimension $D$", fontsize=12)
ax.set_ylabel("STMAN per-node MAE (mph)", fontsize=12)
ax.set_title("(b)", fontsize=13, fontweight="bold", loc="left")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=11)

# ── finalise ──
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
print(f"  Saved -> {OUT_PDF}")
plt.close(fig)
print("Done.")
