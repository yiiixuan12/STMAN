#!/bin/bash
# Run DMFGCRN training on PEMS-BAY dataset using GPU 2
# PEMS-BAY: 325 nodes, speed data, 5-min intervals

export CUDA_VISIBLE_DEVICES=2
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/yxwang/anaconda3/envs/geo_env/lib:$LD_LIBRARY_PATH

cd /mnt/nfs/yxwang/code/multifract/baseline/DMFGCRN

/home/yxwang/anaconda3/envs/geo_env/bin/python run_single.py \
    --dataset PEMSBAY \
    --mode train \
    --device cuda:0 \
    --debug False \
    --seed 12 \
    --batch_size 64 \
    --epochs 300 \
    2>&1 | tee /mnt/nfs/yxwang/code/multifract/baseline/DMFGCRN/train_pemsbay.log
