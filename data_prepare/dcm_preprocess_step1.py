import pydicom
import os
import pickle
import numpy as np
from glob import glob
import cv2

src_path = '/home/user/XueYangWang/HX_data2'
tgt_path_hd = '/home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx'
tgt_path_lung = '/home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx'

# 创建输出文件夹
os.makedirs(tgt_path_hd, exist_ok=True)
os.makedirs(tgt_path_lung, exist_ok=True)

# 要处理的日期文件夹
date_folders = ['20251104', '20251105', '20251107']

def load_dicom_series(folder_path):
    """加载一个文件夹中的所有 DICOM 文件，按文件名排序并堆叠成 3D volume"""
    dcm_files = glob(os.path.join(folder_path, '*.dcm'))
    dcm_files.sort()  # 按文件名排序
    
    if len(dcm_files) == 0:
        return None, None
    
    slices = []
    spacing = None
    first_slice = None
    
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            pixel_array = ds.pixel_array.astype(np.float32)

            # 如果分辨率为 1024x1024，则下采样为 512x512（双线性插值）
            downsample_factor = 1
            if pixel_array.shape[0] == 1024 and pixel_array.shape[1] == 1024:
                # cv2.resize 的 size 参数是 (宽, 高)
                pixel_array = cv2.resize(
                    pixel_array,
                    (512, 512),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)
                downsample_factor = 2

            slices.append(pixel_array)
            
            # 获取 spacing 信息（从第一个文件获取）
            if first_slice is None:
                first_slice = ds
                # 尝试获取像素间距和层厚
                if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
                    pixel_spacing = [float(x) * downsample_factor for x in ds.PixelSpacing]
                else:
                    pixel_spacing = [1.0 * downsample_factor, 1.0 * downsample_factor]  # 默认值
                
                if hasattr(ds, 'SliceThickness') and ds.SliceThickness:
                    slice_thickness = float(ds.SliceThickness)
                elif hasattr(ds, 'SpacingBetweenSlices') and ds.SpacingBetweenSlices:
                    slice_thickness = float(ds.SpacingBetweenSlices)
                else:
                    slice_thickness = 1.0  # 默认值
                
                spacing = tuple(pixel_spacing + [slice_thickness])
        except Exception as e:
            print(f"  警告: 无法读取文件 {dcm_file}: {e}")
            continue
    
    if len(slices) == 0:
        return None, None
    
    # 堆叠成 3D volume [H, W, S]
    volume = np.stack(slices, axis=-1)
    
    return volume, spacing

def process_volume_hd(img, spacing):
    min_val = np.min(img)
    max_val = np.max(img)
    mean_val = np.mean(img)
    print(f"  原始值范围: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
    img = np.clip(img, 0, 4095)
    img = img.astype("uint16")

    return img

def process_volume_lung(img, spacing):
    min_val = np.min(img)
    max_val = np.max(img)
    mean_val = np.mean(img)
    print(f"  原始值范围: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
    img = np.clip(img, 0, 4095)
    img = img.astype("uint16")
    
    return img

def process_patient_folders(date_folder):
    """处理一个日期文件夹下的所有患者"""
    date_path = os.path.join(src_path, date_folder)
    if not os.path.exists(date_path):
        print(f"日期文件夹不存在: {date_path}")
        return
    
    patients = [p for p in os.listdir(date_path) 
                if os.path.isdir(os.path.join(date_path, p)) and not p.startswith('.')]
    
    print(f"\n处理日期文件夹: {date_folder}, 共 {len(patients)} 个患者")
    
    for patient_idx, patient in enumerate(patients):
        patient_path = os.path.join(date_path, patient)
        print(f"\n[{patient_idx+1}/{len(patients)}] 处理患者: {patient}")
        
        # 查找 HD 和 LUNG 文件夹
        subfolders = [f for f in os.listdir(patient_path) 
                     if os.path.isdir(os.path.join(patient_path, f))]
        
        # 查找 HD 文件夹（包含 "HD" 或 "hd"）
        hd_folders = [f for f in subfolders if 'HD' in f.upper() or 'hd' in f]
        # 查找 LUNG 文件夹（包含 "LUNG" 或 "lung"）
        lung_folders = [f for f in subfolders if 'LUNG' in f.upper() or 'lung' in f]
        
        # 处理 HD 文件夹，并记录是否至少成功处理了一个 HD 序列
        hd_processed = False
        for hd_folder in hd_folders:
            hd_folder_path = os.path.join(patient_path, hd_folder)
            print(f"  处理 HD 文件夹: {hd_folder}")
            
            volume, spacing = load_dicom_series(hd_folder_path)
            if volume is None:
                print(f"    跳过: 无法加载 DICOM 文件")
                continue
            
            print(f"    加载成功: shape={volume.shape}")
            if spacing:
                print(f"    spacing={spacing}")
            
            # 应用预处理
            processed_volume = process_volume_hd(volume, spacing)
            
            # 保存文件
            save_name = f"{date_folder}_{patient}_{hd_folder}.pt"
            save_path = os.path.join(tgt_path_hd, save_name)
            data = {'image': processed_volume, 'spacing': spacing if spacing else (1.0, 1.0, 1.0)}
            pickle.dump(data, open(save_path, 'wb'))
            print(f"    保存到: {save_path}, shape={processed_volume.shape}")
            hd_processed = True

        # 如果没有成功处理任何 HD 序列，则跳过该患者的 LUNG 处理
        if not hd_processed:
            if lung_folders:
                print("  未找到有效的 HD 序列，跳过该患者的 LUNG 序列处理")
            continue

        # 处理 LUNG 文件夹
        for lung_folder in lung_folders:
            lung_folder_path = os.path.join(patient_path, lung_folder)
            print(f"  处理 LUNG 文件夹: {lung_folder}")
            
            volume, spacing = load_dicom_series(lung_folder_path)
            if volume is None:
                print(f"    跳过: 无法加载 DICOM 文件")
                continue
            
            print(f"    加载成功: shape={volume.shape}")
            if spacing:
                print(f"    spacing={spacing}")
            
            # 应用预处理
            processed_volume = process_volume_lung(volume, spacing)
            
            # 保存文件
            save_name = f"{date_folder}_{patient}_{lung_folder}.pt"
            save_path = os.path.join(tgt_path_lung, save_name)
            data = {'image': processed_volume, 'spacing': spacing if spacing else (1.0, 1.0, 1.0)}
            pickle.dump(data, open(save_path, 'wb'))
            print(f"    保存到: {save_path}, shape={processed_volume.shape}")

if __name__ == "__main__":
    print("开始处理 DICOM 文件...")
    print(f"源路径: {src_path}")
    print(f"HD 输出路径: {tgt_path_hd}")
    print(f"LUNG 输出路径: {tgt_path_lung}")
    for date_folder in date_folders:
        process_patient_folders(date_folder)
    
    print("\n处理完成！")

