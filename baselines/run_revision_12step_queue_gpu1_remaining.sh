#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nfs/yxwang/code/multifract/baseline"
LOG_DIR="${ROOT}/revision_logs"
LAUNCHER="${ROOT}/run_revision_12step_baseline.sh"
GPU="${GPU:-1}"

mkdir -p "${LOG_DIR}"

run_one() {
  local model="$1"
  local dataset="$2"
  local seed="$3"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log="${LOG_DIR}/${stamp}_${model}_${dataset}_12step_gpu${GPU}_seed${seed}.log"
  echo "[$(date '+%F %T')] START ${model} ${dataset} seed=${seed} gpu=${GPU} log=${log}"
  bash "${LAUNCHER}" "${model}" "${dataset}" "${GPU}" "${seed}" > "${log}" 2>&1
  echo "[$(date '+%F %T')] DONE ${model} ${dataset} seed=${seed} gpu=${GPU} log=${log}"
}

run_one mtegcrn METRLA 42
run_one dmfgcrn PEMSBAY 42
run_one mtegcrn PEMSBAY 42
