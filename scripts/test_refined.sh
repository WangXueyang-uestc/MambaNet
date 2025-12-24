#!/bin/bash
cd "$(dirname "$0")/.."

# 测试改进的 RefinedI3Net 模型

python test.py \
    --upscale 5 \
    --lr_slice_patch 4 \
    --gpu_id '0' \
    --model i3net \
    --use_refined_model True \
    --n_refinement_blocks 2 \
    --ckpt /home/user/XueYangWang/I3Net/experiments/i3net/refined_i3net/pth/0600.pth \
    --ckpt_dir refined_i3net_test \
    --num_workers 4 \
    --train_mode 'paired' \
    --test_hr_data_path /home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_gz_test \
    --test_lr_data_path /home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx_gz_test 
