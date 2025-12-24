#!/usr/bin/env python3
"""
将指定目录中带有'compare'的图片序列制作成GIF动图
注意图片的序列顺序
"""

import os
import sys
from pathlib import Path
from PIL import Image
import re

def extract_number(filename):
    """从文件名中提取数字，用于排序"""
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def create_compare_gif(input_dir, output_path, duration=100):
    """
    从目录中提取所有带'compare'的图片，按序号排序后生成GIF
    
    Args:
        input_dir: 输入目录路径
        output_path: 输出GIF文件路径
        duration: 每帧的持续时间，单位毫秒，默认100ms
    """
    
    # 验证目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：目录不存在 - {input_dir}")
        sys.exit(1)
    
    # 获取所有带'compare'的png文件
    compare_files = []
    for filename in os.listdir(input_dir):
        if 'compare' in filename.lower() and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(input_dir, filename)
            if os.path.isfile(full_path):
                compare_files.append(filename)
    
    if not compare_files:
        print(f"错误：未找到任何带'compare'的图片文件")
        sys.exit(1)
    
    # 按数字顺序排序
    compare_files.sort(key=lambda x: extract_number(x))
    
    print(f"找到 {len(compare_files)} 张compare图片")
    print(f"首张图片: {compare_files[0]}")
    print(f"末张图片: {compare_files[-1]}")
    
    # 打开所有图片
    images = []
    print("加载图片中...")
    for i, filename in enumerate(compare_files):
        if i % 5 == 0:
            continue
        if (i + 1) % 50 == 0:
            print(f"  已加载 {i + 1}/{len(compare_files)}")
        
        full_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(full_path)
            images.append(img)
        except Exception as e:
            print(f"警告：无法加载 {filename}: {e}")
    
    if not images:
        print("错误：无法加载任何图片")
        sys.exit(1)
    
    # 生成GIF
    print(f"生成GIF到: {output_path}")
    print(f"每帧延迟: {duration}ms")
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # 无限循环
        optimize=False
    )
    
    print(f"✓ GIF生成成功！")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # 定义路径
    input_dir = "/home/user/XueYangWang/I3Net/experiments/i3net/refined_i3net_finetune_test/visual_results/20251104_倪誉茹_CT481124_154804_1.0 x 1.0_HD_203"
    output_path = "/home/user/XueYangWang/I3Net/experiments/i3net/refined_i3net_finetune_test/visual_results/20251104_倪誉茹_CT481124_154804_compare.gif"
    
    # 可选：从命令行参数指定输出路径
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    # 可选：从命令行参数指定每帧延迟（毫秒）
    duration = 100
    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
        except ValueError:
            print(f"警告：无效的duration参数 '{sys.argv[2]}'，使用默认值100ms")
    
    create_compare_gif(input_dir, output_path, duration)
