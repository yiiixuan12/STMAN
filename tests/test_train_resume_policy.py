#!/usr/bin/env python
import math
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train


def _write_ckpt(path, val_loss):
    torch.save(
        {
            "model": {},
            "epoch": 0,
            "val_loss": float(val_loss),
            "config": {},
        },
        path,
    )


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        suspect_best = os.path.join(ckpt_dir, "PEMS08_48_v6_best.pt")
        suspect_last = os.path.join(ckpt_dir, "PEMS08_48_v6_last.pt")
        suspect_log = os.path.join(log_dir, "pems08_48_v6_progress.log")
        _write_ckpt(suspect_best, 30.0708)
        _write_ckpt(suspect_last, 18.8677)
        with open(suspect_log, "w", encoding="utf-8") as fh:
            fh.write("epoch=259 MAE=18.8677 best_MAE=18.4730 div=0.82 pers=58.98 vsp=0.32 acc15=0.79\n")

        aligned_best = os.path.join(ckpt_dir, "PEMS03_288_v13dp2rp1_best.pt")
        aligned_last = os.path.join(ckpt_dir, "PEMS03_288_v13dp2rp1_last.pt")
        aligned_log = os.path.join(log_dir, "pems03_288_v13dp2rp1_progress.log")
        _write_ckpt(aligned_best, 20.8207)
        _write_ckpt(aligned_last, 28.8472)
        with open(aligned_log, "w", encoding="utf-8") as fh:
            fh.write("epoch=215 MAE=28.8472 best_MAE=20.8207 div=0.92 pers=111.25 vsp=0.26 acc15=0.51\n")

        suspect_plan = train.resolve_resume_checkpoint_policy(
            suspect_best,
            resume_states=False,
            resume_best_mae=None,
            workdir=tmpdir,
        )
        assert suspect_plan["path"] == suspect_last, suspect_plan
        assert abs(suspect_plan["best_mae_seed"] - 18.4730) < 1e-6, suspect_plan
        assert suspect_plan["reason"] == "prefer_last_over_loss_selected_best", suspect_plan

        aligned_plan = train.resolve_resume_checkpoint_policy(
            aligned_best,
            resume_states=False,
            resume_best_mae=None,
            workdir=tmpdir,
        )
        assert aligned_plan["path"] == aligned_best, aligned_plan
        assert abs(aligned_plan["best_mae_seed"] - 20.8207) < 1e-6, aligned_plan
        assert aligned_plan["reason"] is None, aligned_plan

        best_val, best_mae = train.resolve_resume_monitor_state(
            best_val_loaded=30.0708,
            resume_states=False,
            resume_best_mae=18.4730,
        )
        assert abs(best_val - 18.4730) < 1e-6, (best_val, best_mae)
        assert abs(best_mae - 18.4730) < 1e-6, (best_val, best_mae)

        best_val, best_mae = train.resolve_resume_monitor_state(
            best_val_loaded=20.8207,
            resume_states=True,
            resume_best_mae=None,
        )
        assert abs(best_val - 20.8207) < 1e-6, (best_val, best_mae)
        assert abs(best_mae - 20.8207) < 1e-6, (best_val, best_mae)

    print("PASS test_train_resume_policy")


if __name__ == "__main__":
    main()
