python -W ignore main.py \
    --upscale 5 \
    --lr_slice_patch 4 \
    --ckpt_dir HX_x5 \
    --batch_size 6 \
    --model 'i3net' \
    --num_workers 4 \
    --gpu_id 1 \
    --train_mode 'paired' \