#!/bin/bash
# Graph WaveNet training script for PEMS-BAY
# 325 nodes, input=12, output=12, distance-based adj + adaptive adj

export CUDA_VISIBLE_DEVICES=2
export LD_PRELOAD=/home/yxwang/anaconda3/envs/geo_env/lib/libstdc++.so.6

cd /mnt/nfs/yxwang/code/multifract/baseline/Graph-WaveNet

/home/yxwang/anaconda3/envs/geo_env/bin/python -u train.py \
    --device cuda:0 \
    --data data/PEMS-BAY \
    --adjdata data/sensor_graph/adj_mx_bay.pkl \
    --adjtype doubletransition \
    --gcn_bool \
    --addaptadj \
    --randomadj \
    --seq_length 12 \
    --nhid 32 \
    --in_dim 2 \
    --num_nodes 325 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --dropout 0.3 \
    --weight_decay 0.0001 \
    --epochs 100 \
    --print_every 50 \
    --save ./garage/pems \
    --expid 1 \
    2>&1 | tee train_pemsbay.log
