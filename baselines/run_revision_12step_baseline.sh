#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nfs/yxwang/code/multifract/baseline"
PYTHON="${PYTHON:-/mnt/nfs/yxwang/env/fract_env/bin/python}"

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <graphwavenet|dmfgcrn|mtegcrn> <METRLA|PEMSBAY|PEMSD8> <gpu> [seed]" >&2
  exit 2
fi

MODEL="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
DATASET_RAW="$(echo "$2" | tr '[:lower:]' '[:upper:]' | tr -d -- '-_')"
GPU="$3"
SEED="${4:-42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-7000}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-120}"

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${GPU}"

case "${DATASET_RAW}" in
  METRLA) DATASET="METRLA" ;;
  PEMSBAY) DATASET="PEMSBAY" ;;
  PEMSD8|PEMS08) DATASET="PEMSD8" ;;
  *)
    echo "Unsupported dataset: $2" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T')] model=${MODEL} dataset=${DATASET} physical_gpu=${GPU} visible_device=cuda:0 seed=${SEED}"
echo "python=$("${PYTHON}" -c 'import sys; print(sys.executable)')"

if [[ "${WAIT_FOR_GPU}" == "1" ]]; then
  while true; do
    free_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {print int($1)}')"
    if [[ "${free_mb}" -ge "${MIN_FREE_MB}" ]]; then
      echo "[$(date '+%F %T')] gpu=${GPU} free_mb=${free_mb} >= ${MIN_FREE_MB}; starting."
      break
    fi
    echo "[$(date '+%F %T')] gpu=${GPU} free_mb=${free_mb} < ${MIN_FREE_MB}; waiting ${WAIT_INTERVAL_SEC}s."
    sleep "${WAIT_INTERVAL_SEC}"
  done
fi

case "${MODEL}" in
  graphwavenet|gwn|graph-wavenet)
    cd "${ROOT}/Graph-WaveNet"
    mkdir -p garage/revision
    case "${DATASET}" in
      METRLA)
        data_dir="data/METR-LA"
        adj_path="data/sensor_graph/adj_mx.pkl"
        nodes=207
        save_prefix="./garage/revision/${STAMP}_gwn_metrla"
        ;;
      PEMSBAY)
        data_dir="data/PEMS-BAY"
        adj_path="data/sensor_graph/adj_mx_bay.pkl"
        nodes=325
        save_prefix="./garage/revision/${STAMP}_gwn_pemsbay"
        ;;
      *)
        echo "Graph WaveNet launcher only supports METRLA/PEMSBAY for revision baselines." >&2
        exit 2
        ;;
    esac
    test -d "${data_dir}"
    test -f "${adj_path}"
    exec "${PYTHON}" -u train.py \
      --device cuda:0 \
      --data "${data_dir}" \
      --adjdata "${adj_path}" \
      --adjtype doubletransition \
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
    cd "${ROOT}/DMFGCRN"
    test -f "config_file/${DATASET}_DMFGCRN.conf"
    exec "${PYTHON}" -u run_single.py \
      --dataset "${DATASET}" \
      --model DMFGCRN \
      --device cuda:0 \
      --seed "${SEED}"
    ;;
  mtegcrn)
    cd "${ROOT}/MTEGCRN"
    test -f "config_file/${DATASET}_MTEGCRN.conf"
    exec "${PYTHON}" -u run_single.py \
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
