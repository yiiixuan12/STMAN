#!/bin/bash
# ================================================================
# STMAN v2 全实验矩阵：5数据集 × 6步长
# 按数据集大小排序，小数据集先跑
# ================================================================

export LD_LIBRARY_PATH=/home/yxwang/anaconda3/envs/geo_env/lib:$LD_LIBRARY_PATH
export MPLBACKEND=Agg
PYTHON=/home/yxwang/anaconda3/envs/geo_env/bin/python
SCRIPT=/mnt/nfs/yxwang/code/multifract/train_metrla_optimized.py
cd /mnt/nfs/yxwang/code/multifract

# GPU 选择（通过 CUDA_VISIBLE_DEVICES 控制）
GPU=$1
if [ -z "$GPU" ]; then
    echo "Usage: bash run_all_experiments.sh <GPU_ID>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU

# ================================================================
# 批大小策略：根据序列长度调整，避免OOM（24GB GPU）
# ================================================================
get_batch_size() {
    local pred=$1
    if [ $pred -le 12 ]; then echo 64
    elif [ $pred -le 48 ]; then echo 32
    elif [ $pred -le 96 ]; then echo 16
    elif [ $pred -le 288 ]; then echo 8
    elif [ $pred -le 864 ]; then echo 4
    else echo 2  # 2016步
    fi
}

# d_model策略：长序列用较小维度避免OOM
get_d_model() {
    local pred=$1
    if [ $pred -le 96 ]; then echo 64
    elif [ $pred -le 288 ]; then echo 48
    else echo 32  # 864/2016用32
    fi
}

# ================================================================
# 实验执行函数
# ================================================================
run_exp() {
    local dataset=$1
    local pred_len=$2
    local split=$3

    local bs=$(get_batch_size $pred_len)
    local dm=$(get_d_model $pred_len)
    local tag="v2"
    local logfile="logs/${dataset}_${pred_len}_${tag}.log"

    mkdir -p logs

    echo "=============================================="
    echo "[$(date)] Starting: $dataset pred=$pred_len bs=$bs d_model=$dm"
    echo "=============================================="

    $PYTHON $SCRIPT \
        --dataset $dataset \
        --seq_len $pred_len \
        --pred_len $pred_len \
        --d_model $dm \
        --n_heads 4 \
        --enc_layers 3 \
        --dec_layers 2 \
        --batch_size $bs \
        --epochs 200 \
        --patience 30 \
        --lr 1e-3 \
        --scaler standard \
        --split_rate $split \
        --tag $tag \
        2>&1 | tee $logfile

    echo "[$(date)] Finished: $dataset pred=$pred_len" >> logs/experiment_log.txt
}

# ================================================================
# 实验矩阵
# PEMS03/07/08: split=0.6  PEMS-BAY/METR-LA: split=0.6 (先用6:2:2)
# ================================================================

echo "Starting full experiment matrix on GPU $GPU"
echo "Start time: $(date)" > logs/experiment_log.txt

# 第一优先级：3个审稿人要求对比的数据集，先跑短步长
for pred in 12 48 96 288; do
    run_exp "PEMS08" $pred 0.6
done

for pred in 12 48 96 288; do
    run_exp "METR-LA" $pred 0.6
done

for pred in 12 48 96 288; do
    run_exp "PEMS-BAY" $pred 0.6
done

# 第二优先级：长步长
for pred in 864 2016; do
    run_exp "PEMS08" $pred 0.6  # PEMS08没有2016，会自动跳过
done

for pred in 864 2016; do
    run_exp "METR-LA" $pred 0.6
done

for pred in 864 2016; do
    run_exp "PEMS-BAY" $pred 0.6
done

# 第三优先级：补全其他数据集
for pred in 12 48 96 288 864 2016; do
    run_exp "PEMS03" $pred 0.6
done

for pred in 12 48 96 288 864 2016; do
    run_exp "PEMS07" $pred 0.6
done

echo "All experiments completed: $(date)" >> logs/experiment_log.txt
echo "DONE!"
