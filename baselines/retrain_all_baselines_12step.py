#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用标准超参数重训所有旧 baseline 的 12步预测。
确保结果可复现，与 Table III 对应。
统一评估协议：minmax scaler, 6:2:2 split, unmasked MAE。
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import build_gso_for_stgcn
from train import train_model

# 标准超参数（来自各论文的推荐配置）
BASELINE_CONFIGS = {
    "tgcn": {
        "model_kind": "tgcn",
        "hidden_dim": 64,
        "dropout": 0.0,
        "epochs": 100,
        "lr": 1e-3,
        "patience": 20,
    },
    "stgcn": {
        "model_kind": "stgcn",
        "hidden_dim": 64,
        "stgcn_Kt": 3,
        "stgcn_Ks": 3,
        "stgcn_blocks": [[1, 64, 16, 64], [64, 64, 16, 64]],
        "stgcn_droprate": 0.0,
        "dropout": 0.0,
        "epochs": 100,
        "lr": 1e-3,
        "patience": 20,
    },
    "lstnn": {
        "model_kind": "lstnn",
        "hidden_dim": 64,
        "msmlp_d_d": 64,
        "msmlp_d_spa": 16,
        "msmlp_num_layer": 2,
        "dropout": 0.0,
        "epochs": 100,
        "lr": 1e-3,
        "patience": 20,
    },
    "staeformer": {
        "model_kind": "staeformer",
        "hidden_dim": 64,
        "stf_input_embedding_dim": 24,
        "stf_tod_embedding_dim": 24,
        "stf_dow_embedding_dim": 24,
        "stf_spatial_embedding_dim": 0,
        "stf_adaptive_embedding_dim": 80,
        "stf_feed_forward_dim": 256,
        "stf_num_heads": 4,
        "stf_num_layers": 3,
        "stf_dropout": 0.1,
        "dropout": 0.0,
        "epochs": 100,
        "lr": 1e-3,
        "patience": 20,
    },
    "pdformer": {
        "model_kind": "pdformer",
        "hidden_dim": 64,
        "pdf_embed_dim": 64,
        "pdf_skip_dim": 128,
        "pdf_lape_dim": 8,
        "pdf_geo_heads": 4,
        "pdf_sem_heads": 2,
        "pdf_time_heads": 2,
        "pdf_enc_depth": 6,
        "pdf_drop": 0.0,
        "pdf_attn_drop": 0.0,
        "pdf_drop_path": 0.1,
        "dropout": 0.0,
        "epochs": 100,
        "lr": 1e-3,
        "patience": 20,
    },
}

# 数据集配置
DATASET_INFO = {
    "METR-LA": {"n_nodes": 207, "csv": "METR-LA.csv", "h5": "metr-la.h5", "adj": "adj_matrix.csv"},
    "PEMS-BAY": {"n_nodes": 325, "csv": None, "h5": "pems-bay.h5", "adj": "adj_matrix.csv"},
    "PEMS03": {"n_nodes": 358, "csv": "PEMS03.csv", "h5": None, "adj": "adj_matrix.csv"},
    "PEMS07": {"n_nodes": 883, "csv": "PEMS07.csv", "h5": None, "adj": "adj_matrix.csv"},
    "PEMS08": {"n_nodes": 170, "csv": "PEMS08.csv", "h5": None, "adj": "adj_matrix.csv"},
}


def load_data(dataset_name, project_dir):
    """Load data with minmax scaler, 6:2:2 split"""
    sys.path.insert(0, project_dir)
    from utils import load_adjacency_csv, create_data_loaders

    info = DATASET_INFO[dataset_name]
    data_dir = os.path.join(project_dir, "Datasets", dataset_name)

    # Load adjacency
    adj = load_adjacency_csv(
        os.path.join(data_dir, info["adj"]),
        normalize="sym", add_self_loops=True
    )

    # Load traffic data
    csv_path = os.path.join(data_dir, info["csv"]) if info["csv"] else None
    h5_path = os.path.join(data_dir, info["h5"]) if info["h5"] else None

    if csv_path and os.path.exists(csv_path):
        raw = pd.read_csv(csv_path).values.astype(np.float32)
    elif h5_path and os.path.exists(h5_path):
        raw = pd.read_hdf(h5_path).values.astype(np.float32)
    else:
        raise FileNotFoundError(f"No data for {dataset_name}")

    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]

    T = raw.shape[0]
    n_train = int(T * 0.6)
    n_val = int(T * 0.2)

    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        raw[:n_train], raw[n_train:n_train + n_val], raw[n_train + n_val:],
        seq_len=12, pred_len=12, batch_size=32, scaler_type='minmax'
    )
    return adj, train_loader, val_loader, test_loader, scaler, info["n_nodes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["tgcn", "stgcn", "lstnn", "staeformer", "pdformer"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Retraining {args.model} on {args.dataset} (12-step, standard config)")
    print(f"{'='*60}\n")

    # Load data
    adj, train_loader, val_loader, test_loader, scaler, n_nodes = load_data(args.dataset, project_dir)
    adj = adj.to(device)

    # Get config
    cfg = BASELINE_CONFIGS[args.model].copy()
    model_kind = cfg.pop("model_kind")
    epochs = cfg.pop("epochs")
    lr = cfg.pop("lr")
    patience_val = cfg.pop("patience")

    # Checkpoint paths
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_v2")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_v2")
    os.makedirs(log_dir, exist_ok=True)

    best_ckpt = os.path.join(ckpt_dir, f"{args.dataset}_{args.model}_12_best.pt")
    log_path = os.path.join(log_dir, f"{args.dataset}_{args.model}_12.txt")

    # Train
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        adj=adj,
        n_nodes=n_nodes,
        in_dim=1,
        seq_len=12,
        pred_len=12,
        scaler_obj=scaler,
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience_val,
        model_kind=model_kind,
        best_ckpt_path=best_ckpt,
        log_txt_path=log_path,
        **cfg
    )

    print(f"\nDone! Best checkpoint: {best_ckpt}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
