#!/usr/bin/env python3
"""
将GT和SR图片并排显示，并改善对比度以便观察
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def normalize_image(img, method='percentile', lower_percentile=1, upper_percentile=99):
    """
    归一化图像以改善对比度
    
    Args:
        img: 输入图像 (numpy array)
        method: 归一化方法
            - 'percentile': 使用百分位数裁剪然后归一化到0-255（推荐，适合医学图像）
            - 'minmax': 最小-最大归一化
            - 'clahe': 对比度受限的自适应直方图均衡化
            - 'stretch': 线性拉伸到0-255（适合值范围很小的图像）
        lower_percentile: 下百分位数（用于percentile方法）
        upper_percentile: 上百分位数（用于percentile方法）
    
    Returns:
        归一化后的图像 (uint8, 0-255)
    """
    if img.dtype != np.uint8:
        # 如果图像不是uint8，先转换
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    if method == 'percentile':
        # 使用百分位数裁剪，去除极端值
        lower = np.percentile(img, lower_percentile)
        upper = np.percentile(img, upper_percentile)
        img_clipped = np.clip(img, lower, upper)
        # 归一化到0-255
        if upper > lower:
            img_norm = ((img_clipped - lower) / (upper - lower) * 255).astype(np.uint8)
        else:
            img_norm = img_clipped.astype(np.uint8)
        return img_norm
    
    elif method == 'stretch':
        # 线性拉伸：直接将当前值范围映射到0-255
        # 特别适合值范围很小的图像（如0-50）
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img_norm = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = img.astype(np.uint8)
        return img_norm
    
    elif method == 'minmax':
        # 最小-最大归一化
        if img.max() > img.min():
            img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img_norm = img.astype(np.uint8)
        return img_norm
    
    elif method == 'clahe':
        # 对比度受限的自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_norm = clahe.apply(img)
        return img_norm
    
    else:
        return img


def visualize_comparison(gt_path, sr_path, output_path=None, method='percentile', 
                         layout='horizontal', figsize=(16, 8)):
    """
    将GT和SR图片并排显示
    
    Args:
        gt_path: GT图片路径
        sr_path: SR图片路径
        output_path: 输出图片路径（如果为None，则在输入目录下生成）
        method: 归一化方法
        layout: 布局方式 'horizontal'（左右）或 'vertical'（上下）
        figsize: 图片大小
    """
    # 读取图片
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    sr = cv2.imread(sr_path, cv2.IMREAD_GRAYSCALE)
    
    if gt is None:
        raise ValueError(f"无法读取GT图片: {gt_path}")
    if sr is None:
        raise ValueError(f"无法读取SR图片: {sr_path}")
    
    print(f"GT图片尺寸: {gt.shape}, 数据类型: {gt.dtype}, 值范围: [{gt.min()}, {gt.max()}]")
    print(f"SR图片尺寸: {sr.shape}, 数据类型: {sr.dtype}, 值范围: [{sr.min()}, {sr.max()}]")
    
    # 归一化图片
    # 对于值范围很小的图像（如0-50），使用stretch方法效果更好
    if gt.max() < 100 or sr.max() < 100:
        print("检测到图像值范围较小，使用stretch方法进行归一化")
        gt_norm = normalize_image(gt, method='stretch')
        sr_norm = normalize_image(sr, method='stretch')
    else:
        gt_norm = normalize_image(gt, method=method)
        sr_norm = normalize_image(sr, method=method)
    
    # 创建对比图
    if layout == 'horizontal':
        # 左右排列
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(gt_norm, cmap='gray')
        axes[0].set_title('Ground Truth (GT)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(sr_norm, cmap='gray')
        axes[1].set_title('Super-Resolution (SR)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    else:
        # 上下排列
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        axes[0].imshow(gt_norm, cmap='gray')
        axes[0].set_title('Ground Truth (GT)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(sr_norm, cmap='gray')
        axes[1].set_title('Super-Resolution (SR)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        # 在GT图片同目录下生成
        base_dir = os.path.dirname(gt_path)
        output_path = os.path.join(base_dir, 'comparison.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {output_path}")
    
    # 也创建一个并排拼接的版本（直接用OpenCV）
    if layout == 'horizontal':
        combined = np.hstack([gt_norm, sr_norm])
    else:
        combined = np.vstack([gt_norm, sr_norm])
    
    # 添加分隔线
    if layout == 'horizontal':
        combined[:, gt_norm.shape[1]:gt_norm.shape[1]+2] = 255  # 白色分隔线
    else:
        combined[gt_norm.shape[0]:gt_norm.shape[0]+2, :] = 255  # 白色分隔线
    
    # 添加文字标签
    combined_labeled = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)  # 白色文字
    
    if layout == 'horizontal':
        cv2.putText(combined_labeled, 'GT', (10, 30), font, font_scale, color, thickness)
        cv2.putText(combined_labeled, 'SR', (gt_norm.shape[1] + 10, 30), font, font_scale, color, thickness)
    else:
        cv2.putText(combined_labeled, 'GT', (10, 30), font, font_scale, color, thickness)
        cv2.putText(combined_labeled, 'SR', (10, gt_norm.shape[0] + 30), font, font_scale, color, thickness)
    
    # 保存OpenCV版本
    cv2_output_path = output_path.replace('.png', '_opencv.png')
    cv2.imwrite(cv2_output_path, combined_labeled)
    print(f"OpenCV拼接版本已保存到: {cv2_output_path}")
    
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("用法: python visualize_comparison.py <GT图片路径> <SR图片路径> [输出路径] [归一化方法] [布局]")
        print("归一化方法: percentile (默认), minmax, clahe")
        print("布局: horizontal (默认, 左右), vertical (上下)")
        print("\n示例:")
        print("  python visualize_comparison.py gt.png sr.png")
        print("  python visualize_comparison.py gt.png sr.png output.png percentile horizontal")
        return
    
    gt_path = sys.argv[1]
    sr_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    method = sys.argv[4] if len(sys.argv) > 4 else 'percentile'
    layout = sys.argv[5] if len(sys.argv) > 5 else 'horizontal'
    
    visualize_comparison(gt_path, sr_path, output_path, method, layout)


if __name__ == '__main__':
    # 如果直接运行，使用示例路径
    if len(sys.argv) == 1:
        # 使用用户提供的路径
        gt_path = '/Data2/XueYangWang/Medical_SR/I3Net/experiments/i3net/Pretrain_x5_finetune_hx_paired/visual_results/20251104_倪誉茹_CT481124_154804_1.0 x 1.0_HD_203/0006_GT.png'
        sr_path = '/Data2/XueYangWang/Medical_SR/I3Net/experiments/i3net/Pretrain_x5_finetune_hx_paired/visual_results/20251104_倪誉茹_CT481124_154804_1.0 x 1.0_HD_203/0006_SR.png'
        visualize_comparison(gt_path, sr_path, method='percentile', layout='horizontal')
    else:
        main()

