#!/bin/bash
cd "$(dirname "$0")/.."

python main.py \
    --model vfimamba \
    --upscale 5 \
    --lr_slice_patch 4 \
    --gpu_id '0' \
    --batch_size 4 \
    --one_batch_n_sample 1 \
    --num_workers 4 \
    --max_epoch 1500 \
    --lr 1e-4 \
    --ckpt_dir 'VFIMamba_x5_pretrain' \
    --vfimamba_ckpt '' \
    --vfimamba_F 16 \
    --vfimamba_depth '[2,2,2,3,3]' \
    --vfimamba_M False \
    --vfimamba_local 2 \
    --resume False \
    --amp True
