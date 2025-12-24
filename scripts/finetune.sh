#!/bin/bash
cd "$(dirname "$0")/.."

python -W ignore main.py \
    --upscale 5 \
    --lr_slice_patch 4 \
    --ckpt_dir HX_x5_fintune_new \
    --batch_size 4 \
    --gpu_id '0' \
    --model 'i3net' \
    --num_workers 4 \
    --gpu_id 0 \
    --resume True \
    --ckpt /home/user/XueYangWang/I3Net/experiments/i3net/Pretrain_x5/pth/0900.pth \
    --train_mode 'paired'