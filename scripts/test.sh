#!/bin/bash
cd "$(dirname "$0")/.."
python test.py \
    --upscale 5 \
    --lr_slice_patch 4 \
    --gpu_id '0' \
    --model i3net \
    --ckpt /Data2/XueYangWang/Medical_SR/I3Net/experiments/i3net/HX_x5_finetune/pth/1500.pth \
    --test_hr_data_path /home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_gz_test \
    --test_lr_data_path /home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx_gz_test 
    --num_workers 4 \
    --train_mode 'paired' \
    --test_hr_data_path /home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_gz_test \
    --test_lr_data_path /home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx_gz_test 