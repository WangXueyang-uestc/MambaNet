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
    --train_mode 'paired' \
    --hr_data_path '/home/user/XueYangWang/HX_data2/data_processed_slice_hd_hx_train' \
    --lr_data_path '/home/user/XueYangWang/HX_data2/data_processed_slice_lung_hx_train' \
    --ckpt_dir 'VFIMamba_x5_endpoints' \
    --vfimamba_ckpt '' \
    --vfimamba_F 16 \
    --vfimamba_depth '[2,2,2,3,3]' \
    --vfimamba_M False \
    --vfimamba_local 2 \
    --resume True \
    --ckpt /home/user/XueYangWang/I3Net/experiments/vfimamba/VFIMamba_x5_endpoints/pth/0900.pth \
    --amp True
