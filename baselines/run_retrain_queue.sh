#!/bin/bash
# ================================================================
# 旧 Baseline 12步标准配置重训队列
# 用标准超参数确保可复现性
# ================================================================
export LD_LIBRARY_PATH=/home/yxwang/anaconda3/envs/geo_env/lib:$LD_LIBRARY_PATH
export MPLBACKEND=Agg
PYTHON=/home/yxwang/anaconda3/envs/geo_env/bin/python
SCRIPT=/mnt/nfs/yxwang/code/multifract/baseline/retrain_all_baselines_12step.py

GPU=$1
if [ -z "$GPU" ]; then
    echo "Usage: bash run_retrain_queue.sh <GPU_ID>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU

cd /mnt/nfs/yxwang/code/multifract/baseline

echo "Starting baseline retraining queue on GPU $GPU"
echo "Start: $(date)"

# 优先级：先跑 METR-LA（审稿人最关注），然后 PEMS08，然后其他
# 每个方法约 100 epochs × ~30-60s/epoch ≈ 1-2 小时

# === METR-LA ===
for model in staeformer stgcn tgcn lstnn pdformer; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Training $model on METR-LA"
    echo "=========================================="
    $PYTHON $SCRIPT --model $model --dataset METR-LA --device cuda:0
done

# === PEMS08 ===
for model in staeformer stgcn tgcn lstnn pdformer; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Training $model on PEMS08"
    echo "=========================================="
    $PYTHON $SCRIPT --model $model --dataset PEMS08 --device cuda:0
done

# === PEMS-BAY ===
for model in staeformer stgcn tgcn lstnn pdformer; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Training $model on PEMS-BAY"
    echo "=========================================="
    $PYTHON $SCRIPT --model $model --dataset PEMS-BAY --device cuda:0
done

# === PEMS03 ===
for model in staeformer stgcn tgcn lstnn pdformer; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Training $model on PEMS03"
    echo "=========================================="
    $PYTHON $SCRIPT --model $model --dataset PEMS03 --device cuda:0
done

# === PEMS07 ===
for model in staeformer stgcn tgcn lstnn pdformer; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Training $model on PEMS07"
    echo "=========================================="
    $PYTHON $SCRIPT --model $model --dataset PEMS07 --device cuda:0
done

echo ""
echo "All baseline retraining completed: $(date)"
