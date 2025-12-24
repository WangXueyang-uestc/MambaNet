import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dicom(file_path):
    """加载 DICOM 文件并返回像素数组和像素值范围"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array
    
    # 获取像素值范围
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)
    mean_val = np.mean(pixel_array)
    
    return pixel_array, min_val, max_val, mean_val

def visualize_dicom_pair(file1_path, file2_path, save_path=None):
    """加载两个 DICOM 文件并并排显示"""
    # 加载两个 DICOM 文件
    img1, min1, max1, mean1 = load_dicom(file1_path)
    img2, min2, max2, mean2 = load_dicom(file2_path)
    
    # 创建图形，并排显示两张图片
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 显示第一张图片
    im1 = axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f'{os.path.basename(os.path.dirname(file1_path))}\n'
                     f'像素值范围: [{min1:.1f}, {max1:.1f}]\n'
                     f'平均值: {mean1:.1f}', fontsize=10)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 显示第二张图片
    im2 = axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'{os.path.basename(os.path.dirname(file2_path))}\n'
                     f'像素值范围: [{min2:.1f}, {max2:.1f}]\n'
                     f'平均值: {mean2:.1f}', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()
    
    # 打印详细信息
    print(f"\n文件1: {file1_path}")
    print(f"  形状: {img1.shape}")
    print(f"  像素值范围: [{min1:.2f}, {max1:.2f}]")
    print(f"  平均值: {mean1:.2f}")
    print(f"  数据类型: {img1.dtype}")
    
    print(f"\n文件2: {file2_path}")
    print(f"  形状: {img2.shape}")
    print(f"  像素值范围: [{min2:.2f}, {max2:.2f}]")
    print(f"  平均值: {mean2:.2f}")
    print(f"  数据类型: {img2.dtype}")

if __name__ == "__main__":
    # 文件路径
    file1 = "/Data2/XueYangWang/Medical_SR/medical_data/20251105/曾雯琳_CT481214_095653/1.0 x 1.0_HD_203/00000001.dcm"
    file2 = "/Data2/XueYangWang/Medical_SR/medical_data/20251105/曾雯琳_CT481214_095653/5.0 x 5.0_LUNG_201/00000001.dcm"
    save_path = "/Data2/XueYangWang/Medical_SR/I3Net/output.png"
    # 可视化（可以指定保存路径，不指定则显示）
    visualize_dicom_pair(file1, file2, save_path=save_path)  # 设置为 None 显示，或指定路径如 "output.png" 保存

