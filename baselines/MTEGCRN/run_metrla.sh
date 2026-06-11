#!/bin/bash
# Run MTEGCRN training on METR-LA dataset
# METR-LA: 207 nodes, speed data, 5-min intervals

export CUDA_VISIBLE_DEVICES=1
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/yxwang/anaconda3/envs/geo_env/lib:$LD_LIBRARY_PATH

cd /mnt/nfs/yxwang/code/multifract/baseline/MTEGCRN

/home/yxwang/anaconda3/envs/geo_env/bin/python run_single.py \
    --dataset METRLA \
    --mode train \
    --device cuda:0 \
    --debug False \
    --seed 12 \
    2>&1 | tee /mnt/nfs/yxwang/code/multifract/baseline/MTEGCRN/train_metrla.log
