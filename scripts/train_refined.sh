#!/bin/bash
cd "$(dirname "$0")/.."

# 训练改进的 RefinedI3Net 模型
# 联合插值与增强框架

python -W ignore main.py \
    --upscale 5 \
    --lr_slice_patch 4 \
    --ckpt_dir refined_i3net_finetune \
    --batch_size 4 \
    --model 'i3net' \
    --use_refined_model True \
    --n_refinement_blocks 2 \
    --num_workers 4 \
    --gpu_id 0 \
    --train_mode 'paired' \
    --resume True \
    --ckpt '/home/user/XueYangWang/I3Net/experiments/i3net/refined_i3net_finetune/pth/1500.pth' \
    --hr_data_path '/home/user/XueYangWang/HX_data2/data_processed_slice_hd_hx_train' \
    --lr_data_path '/home/user/XueYangWang/HX_data2/data_processed_slice_lung_hx_train'