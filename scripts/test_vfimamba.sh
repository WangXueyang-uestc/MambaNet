#!/bin/bash
cd "$(dirname "$0")/.."

# 使用训练好的 VFIMamba 模型进行测试
# 请修改 --vfimamba_ckpt 指向你训练好的模型路径

python test.py \
    --model vfimamba \
    --upscale 5 \
    --lr_slice_patch 4 \
    --gpu_id '0' \
    --num_workers 4 \
    --train_mode 'paired' \
    --test_hr_data_path '/home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_test' \
    --test_lr_data_path '/home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx_test' \
    --ckpt_dir 'VFIMamba_x5_test' \
    --vfimamba_ckpt 'experiments/vfimamba/VFIMamba_x5/pth/1000.pth' \
    --vfimamba_F 16 \
    --vfimamba_depth '[2,2,2,3,3]' \
    --vfimamba_M False \
    --vfimamba_local 2
