#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-PEMS03 PEMS07 PEMS08 PEMS-BAY METR-LA}"
HORIZONS="${HORIZONS:-12 48 96 288}"
SEQ_LEN="${SEQ_LEN:-96}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TAG="${TAG:-public}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "${ROOT}"
mkdir -p checkpoints/stman logs

for dataset in ${DATASETS}; do
  for horizon in ${HORIZONS}; do
    echo "[$(date '+%F %T')] STMAN dataset=${dataset} horizon=${horizon}"
    "${PYTHON_BIN}" train_stman.py \
      --dataset "${dataset}" \
      --seq_len "${SEQ_LEN}" \
      --pred_len "${horizon}" \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --tag "${TAG}" \
      --checkpoint_dir checkpoints/stman \
      2>&1 | tee "logs/stman_${dataset}_${horizon}_${TAG}.log"
  done
done
