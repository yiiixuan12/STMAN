#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BASELINE_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
GPU="${GPU:-0}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <graphwavenet|dmfgcrn|mtegcrn> <METRLA|PEMSBAY|PEMS08> [seed]" >&2
  exit 2
fi

MODEL="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
DATASET_RAW="$(echo "$2" | tr '[:lower:]' '[:upper:]' | tr -d -- '-_')"
SEED="${3:-42}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export TABLE4_CUSTOM_DATA_ROOT="${TABLE4_CUSTOM_DATA_ROOT:-${REPO_ROOT}/Datasets}"

case "${DATASET_RAW}" in
  METRLA) DATASET="METRLA" ;;
  PEMSBAY) DATASET="PEMSBAY" ;;
  PEMS08|PEMSD8) DATASET="PEMSD8" ;;
  *)
    echo "Unsupported dataset: $2" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  graphwavenet|gwn|graph-wavenet)
    cd "${BASELINE_ROOT}/Graph-WaveNet"
    case "${DATASET}" in
      METRLA)
        stream_source="${REPO_ROOT}/Datasets/METR-LA/METR-LA.npz"
        adj_path="${REPO_ROOT}/Datasets/METR-LA/adj_mat.pkl"
        nodes=207
        save_prefix="./garage/public_metrla"
        ;;
      PEMSBAY)
        stream_source="${REPO_ROOT}/Datasets/PEMS-BAY/PEMS-BAY.npz"
        adj_path="${REPO_ROOT}/Datasets/PEMS-BAY/adj_mx_bay.pkl"
        nodes=325
        save_prefix="./garage/public_pemsbay"
        ;;
      *)
        echo "GraphWaveNet launcher supports METRLA and PEMSBAY." >&2
        exit 2
        ;;
    esac
    mkdir -p garage
    exec "${PYTHON_BIN}" -u train.py \
      --device cuda:0 \
      --adjdata "${adj_path}" \
      --adjtype doubletransition \
      --stream_source "${stream_source}" \
      --stream_seq_length_x 12 \
      --gcn_bool \
      --addaptadj \
      --randomadj \
      --seq_length 12 \
      --nhid 32 \
      --in_dim 2 \
      --num_nodes "${nodes}" \
      --batch_size 64 \
      --learning_rate 0.001 \
      --dropout 0.3 \
      --weight_decay 0.0001 \
      --epochs 100 \
      --print_every 50 \
      --save "${save_prefix}" \
      --expid "${SEED}"
    ;;
  dmfgcrn)
    cd "${BASELINE_ROOT}/DMFGCRN"
    exec "${PYTHON_BIN}" -u run_single.py \
      --dataset "${DATASET}" \
      --model DMFGCRN \
      --device cuda:0 \
      --seed "${SEED}"
    ;;
  mtegcrn)
    cd "${BASELINE_ROOT}/MTEGCRN"
    exec "${PYTHON_BIN}" -u run_single.py \
      --dataset "${DATASET}" \
      --model MTEGCRN \
      --device cuda:0 \
      --seed "${SEED}"
    ;;
  *)
    echo "Unsupported model: $1" >&2
    exit 2
    ;;
esac
