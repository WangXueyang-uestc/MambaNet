#!/bin/bash

# 批量处理MSD数据集所有任务的imagesTr文件
# 用法: bash process_all_msd_tasks.sh

# MSD数据集根目录
MSD_ROOT="/Data2/XueYangWang/Medical_SR/data/MSD"
# 输出目录
OUTPUT_ROOT="/Data2/XueYangWang/Medical_SR/data_processed"
# Python脚本路径
PYTHON_SCRIPT="/Data2/XueYangWang/Medical_SR/I3Net/data_prepare/nii_preprocess_step1.py"

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 获取所有Task文件夹
TASKS=$(find "$MSD_ROOT" -maxdepth 1 -type d -name "Task*" | sort)

echo "=========================================="
echo "开始处理MSD数据集"
echo "=========================================="
echo ""

# 循环处理每个任务
for task_dir in $TASKS; do
    # 获取任务名称
    task_name=$(basename "$task_dir")
    
    # 检查imagesTr文件夹是否存在
    images_dir="${task_dir}/imagesTr"
    if [ ! -d "$images_dir" ]; then
        echo "⚠️  跳过 ${task_name}: imagesTr文件夹不存在"
        continue
    fi
    
    # 检查imagesTr文件夹是否为空
    if [ -z "$(ls -A "$images_dir")" ]; then
        echo "⚠️  跳过 ${task_name}: imagesTr文件夹为空"
        continue
    fi
    
    echo "=========================================="
    echo "正在处理: ${task_name}"
    echo "源路径: ${images_dir}"
    echo "=========================================="
    
    # 创建任务特定的输出目录
    task_output_dir="${OUTPUT_ROOT}"
    mkdir -p "$task_output_dir"
    
    # 使用Python执行预处理
    # 通过临时修改Python脚本中的路径来处理不同任务
    # 或者直接在Python中传递参数
    python3 << EOF
import os
import sys
import numpy as np
from medpy.io import load
import pickle

src_path = '${images_dir}'
tgt_path = '${task_output_dir}'

os.makedirs(tgt_path, exist_ok=True)

patients = os.listdir(src_path)
print(f"找到 {len(patients)} 个文件")

for id, patient in enumerate(patients):
    if not '._' in patient and patient.endswith('.nii.gz'):
        try:
            img, header = load(os.path.join(src_path, patient))
            spacing = header.get_voxel_spacing()
            
            # 预处理，调整像素到0，2^15
            img = np.clip(img, -1024, img.max())
            img = img - img.min()
            img = np.clip(img, 0, 4095)
            img = img.astype("uint16")
            
            # 另存，保留图像和spacing
            data = {'image': img, 'spacing': spacing}
            pickle.dump(data, open(os.path.join(tgt_path, patient.replace('.nii.gz', '.pt')), 'wb'))
            print(f"{id+1}/{len(patients)}: volume finished, {patient}, shape: {img.shape}")
        except Exception as e:
            print(f"❌ 处理 {patient} 时出错: {str(e)}")
            continue

print('${task_name} 处理完成!')
EOF
    
    if [ $? -eq 0 ]; then
        echo "✅ ${task_name} 处理完成"
    else
        echo "❌ ${task_name} 处理失败"
    fi
    echo ""
done

echo "=========================================="
echo "所有任务处理完成！"
echo "输出目录: ${OUTPUT_ROOT}"
echo "=========================================="
