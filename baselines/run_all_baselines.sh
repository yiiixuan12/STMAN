#!/bin/bash
# ================================================================
# 新 Baseline 全实验矩阵
# DMFGCRN / MTEGCRN / Graph-WaveNet
# 数据集：METR-LA, PEMS-BAY, PEMS08
# 步长：12, 48, 96, 288 (长步长可选)
# ================================================================

export LD_LIBRARY_PATH=/home/yxwang/anaconda3/envs/geo_env/lib:$LD_LIBRARY_PATH
export MPLBACKEND=Agg
PYTHON=/home/yxwang/anaconda3/envs/geo_env/bin/python

GPU=$1
if [ -z "$GPU" ]; then
    echo "Usage: bash run_all_baselines.sh <GPU_ID>"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p /mnt/nfs/yxwang/code/multifract/baseline/results

# ================================================================
# DMFGCRN 实验
# ================================================================
run_dmfgcrn() {
    local dataset=$1   # PEMSD8, METRLA, PEMSBAY
    local horizon=$2
    local nodes=$3

    cd /mnt/nfs/yxwang/code/multifract/baseline/DMFGCRN

    # 动态生成 config
    local conf="config_file/${dataset}_DMFGCRN_h${horizon}.conf"
    local base_conf="config_file/${dataset}_DMFGCRN.conf"

    if [ ! -f "$base_conf" ]; then
        echo "Base config not found: $base_conf, skipping"
        return
    fi

    # 复制基础 config 并修改 horizon 和 lag
    cp $base_conf $conf
    sed -i "s/^lag = .*/lag = ${horizon}/" $conf
    sed -i "s/^horizon = .*/horizon = ${horizon}/" $conf

    # 调整 batch_size
    local bs=64
    if [ $horizon -ge 96 ]; then bs=32; fi
    if [ $horizon -ge 288 ]; then bs=16; fi
    sed -i "s/^batch_size = .*/batch_size = ${bs}/" $conf

    echo "[$(date)] DMFGCRN: $dataset horizon=$horizon bs=$bs"

    $PYTHON run_single.py \
        --dataset=$dataset \
        --device=cuda:0 \
        --seed=42 \
        2>&1 | tee "results/dmfgcrn_${dataset}_h${horizon}.log"

    # 恢复原始 config
    rm -f $conf
}

# ================================================================
# MTEGCRN 实验
# ================================================================
run_mtegcrn() {
    local dataset=$1
    local horizon=$2

    cd /mnt/nfs/yxwang/code/multifract/baseline/MTEGCRN

    local conf="config_file/${dataset}_MTEGCRN_h${horizon}.conf"
    local base_conf="config_file/${dataset}_MTEGCRN.conf"

    if [ ! -f "$base_conf" ]; then
        echo "Base config not found: $base_conf, skipping"
        return
    fi

    cp $base_conf $conf
    sed -i "s/^lag = .*/lag = ${horizon}/" $conf
    sed -i "s/^horizon = .*/horizon = ${horizon}/" $conf

    local bs=64
    if [ $horizon -ge 96 ]; then bs=32; fi
    if [ $horizon -ge 288 ]; then bs=16; fi
    sed -i "s/^batch_size = .*/batch_size = ${bs}/" $conf

    echo "[$(date)] MTEGCRN: $dataset horizon=$horizon bs=$bs"

    $PYTHON run_single.py \
        --dataset=$dataset \
        --device=cuda:0 \
        --seed=42 \
        2>&1 | tee "results/mtegcrn_${dataset}_h${horizon}.log"

    rm -f $conf
}

# ================================================================
# Graph-WaveNet 实验
# 注意：GWN 固定输出 12 步，长步长需要改代码
# 先跑 12 步的 PEMS-BAY，其他步长后续处理
# ================================================================
run_gwn() {
    local dataset=$1   # METR-LA or PEMS-BAY
    local data_dir=$2
    local adj_file=$3
    local num_nodes=$4
    local save_prefix=$5

    cd /mnt/nfs/yxwang/code/multifract/baseline/Graph-WaveNet

    echo "[$(date)] Graph-WaveNet: $dataset"

    $PYTHON -u train.py \
        --device cuda:0 \
        --data "data/${data_dir}" \
        --adjdata "data/sensor_graph/${adj_file}" \
        --adjtype doubletransition \
        --gcn_bool --addaptadj --randomadj \
        --seq_length 12 \
        --nhid 32 \
        --in_dim 2 \
        --num_nodes $num_nodes \
        --batch_size 64 \
        --learning_rate 0.001 \
        --dropout 0.3 \
        --weight_decay 0.0001 \
        --epochs 100 \
        --print_every 50 \
        --save "./garage/${save_prefix}" \
        --expid 1 \
        2>&1 | tee "results/gwn_${dataset}.log"
}

# ================================================================
# 执行顺序（按优先级）
# ================================================================

echo "Starting baseline experiments on GPU $GPU"
echo "Start time: $(date)" > /mnt/nfs/yxwang/code/multifract/baseline/results/baseline_log.txt

# 1. DMFGCRN: PEMS08 多步长
for h in 12 48 96 288; do
    run_dmfgcrn "PEMSD8" $h 170
done

# 2. DMFGCRN: METR-LA 多步长
for h in 12 48 96 288; do
    run_dmfgcrn "METRLA" $h 207
done

# 3. Graph-WaveNet: PEMS-BAY (12步)
run_gwn "PEMS-BAY" "PEMS-BAY" "adj_mx_bay.pkl" 325 "bay"

# 4. MTEGCRN: PEMS08 多步长 (12已经在跑)
for h in 48 96 288; do
    run_mtegcrn "PEMSD8" $h
done

# 5. MTEGCRN: METR-LA 多步长
for h in 12 48 96 288; do
    run_mtegcrn "METRLA" $h
done

echo "All baseline experiments completed: $(date)" >> /mnt/nfs/yxwang/code/multifract/baseline/results/baseline_log.txt
echo "DONE!"
