# random

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import nibabel as nib
import random 
import SimpleITK as sitk

import pickle
import util


class trainSet(Dataset):
    def __init__(self, data_root, args=None,
                random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.args = args
        self.data_root = data_root
        
        # 过滤掉切片数量不足的文件夹
        all_folders = [(data_root + '/' + f) for f in os.listdir(data_root)]
        self.folder_list = []
        for folder in all_folders:
            if os.path.isdir(folder):
                if len(os.listdir(folder)) >= self.args.hr_slice_patch:
                    self.folder_list.append(folder)
        
        if len(self.folder_list) == 0:
            print(f"Warning: No valid folders found in {data_root} with at least {self.args.hr_slice_patch} slices.")
        else:
            print(f"Found {len(self.folder_list)} valid folders out of {len(all_folders)} in {data_root}")

        random.shuffle(self.folder_list)
        self.file_len = len(self.folder_list)

        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t
        

    def __getitem__(self, index):
        volumepath = self.folder_list[index]
        slice_list = [(volumepath+'/'+f) for f in os.listdir(volumepath)]
        slice_list.sort()
        
        def getAVol(slice_list):
            id = random.randint(0,len(slice_list) - self.args.hr_slice_patch)
            volume = []
            for i in range(self.args.hr_slice_patch):
                with open(slice_list[id+i], 'rb') as _f: img = pickle.load(_f)
                volume.append(img['image'])
                # img = np.load(slice_list[id+i])
                # volume.append(img)

            volume = np.array(volume,dtype=np.float32).transpose(1,2,0) 
            volume = util.normalize(volume) # [0-4095]->[0-1]
            if random.random() >= 0.5:
                volume = volume[:,::-1,:].copy()
            if random.random() >= 0.5:
                volume = volume[::-1,:,:].copy()

            # 使用随机裁剪替代中心裁剪，让整个图像区域都能得到训练
            # 如果图像尺寸大于256，则随机裁剪；否则使用中心裁剪
            h, w, c = volume.shape
            crop_size = 256
            if h > crop_size and w > crop_size:
                # 随机裁剪：从 [0, h-crop_size] 和 [0, w-crop_size] 范围内随机选择起始位置
                start_h = random.randint(0, h - crop_size)
                start_w = random.randint(0, w - crop_size)
                volume = volume[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
            else:
                # 如果图像小于256，使用中心裁剪（保持原有逻辑）
                volume = util.crop_center(volume, crop_size, crop_size)
            volume=torch.from_numpy(volume)
            return volume

        volume = []
        for i in range(self.args.one_batch_n_sample):
            volume.append(getAVol(slice_list))
        volume = torch.stack(volume,0)
        
        return volume

    def __len__(self):
        return self.file_len


class trainSetPaired(Dataset):
    """
    使用真实配对的高分辨率-低分辨率数据进行训练
    """
    def __init__(self, hr_data_root, lr_data_root, args=None,
                random_crop=None, resize=None, augment_s=True, augment_t=True):

        self.args = args
        self.hr_data_root = hr_data_root
        self.lr_data_root = lr_data_root
        
        # 获取高分辨率文件夹列表
        hr_folders = [f for f in os.listdir(hr_data_root)]
        # 获取对应的低分辨率文件夹（通过替换后缀 _1 -> _5）
        self.paired_folders = []
        for hr_folder in hr_folders:
            # 假设高分辨率文件夹以 _1 结尾，低分辨率以 _5 结尾
            if hr_folder.endswith('_1'):
                lr_folder = hr_folder[:-2] + '_5'
                hr_path = os.path.join(hr_data_root, hr_folder)
                lr_path = os.path.join(lr_data_root, lr_folder)
                if os.path.exists(lr_path):
                    # 检查切片数量是否足够
                    hr_slices = [f for f in os.listdir(hr_path) if f.endswith('.pt')]
                    lr_slices = [f for f in os.listdir(lr_path) if f.endswith('.pt')]
                    if len(lr_slices) >= self.args.lr_slice_patch:
                        self.paired_folders.append((hr_path, lr_path))
        
        if len(self.paired_folders) == 0:
            print(f"Warning: No valid paired folders found with at least {self.args.lr_slice_patch} LR slices.")
        else:
            print(f"Found {len(self.paired_folders)} valid paired folders out of {len(hr_folders)} HR folders.")

        random.shuffle(self.paired_folders)
        self.file_len = len(self.paired_folders)
        
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

    def _adjust_slice_count(self, hr_slice_list, lr_slice_list):
        """
        调整 HR 和 LR 的 slice 数量，使其满足 upscale 关系
        关系：HR_slice_count = (LR_slice_count - 1) * upscale + 1
        如果不满足，删除 corner slices
        """
        hr_count = len(hr_slice_list)
        lr_count = len(lr_slice_list)
        
        # 计算期望的 slice 数量关系
        # HR_slice_count = (LR_slice_count - 1) * upscale + 1
        # 或者：LR_slice_count = (HR_slice_count - 1) / upscale + 1
        
        # 根据 LR 数量计算期望的 HR 数量
        expected_hr_count = (lr_count - 1) * self.args.upscale + 1
        
        # 根据 HR 数量计算期望的 LR 数量
        expected_lr_count = (hr_count - 1) // self.args.upscale + 1
        
        # 如果 HR 数量过多，删除末尾的 corner slices
        if hr_count > expected_hr_count:
            excess = hr_count - expected_hr_count
            # 删除末尾的 slices
            hr_slice_list = hr_slice_list[:-excess]
            hr_count = len(hr_slice_list)
        
        # 如果 LR 数量过多，删除末尾的 corner slices
        if lr_count > expected_lr_count:
            excess = lr_count - expected_lr_count
            # 删除末尾的 slices
            lr_slice_list = lr_slice_list[:-excess]
            lr_count = len(lr_slice_list)
        
        # 再次检查是否满足关系
        expected_hr_count = (lr_count - 1) * self.args.upscale + 1
        expected_lr_count = (hr_count - 1) // self.args.upscale + 1
        
        # 如果仍然不满足，可能需要删除开头的 slices
        if hr_count != expected_hr_count:
            # 调整 HR 使其满足关系
            hr_slice_list = hr_slice_list[:expected_hr_count]
        if lr_count != expected_lr_count:
            # 调整 LR 使其满足关系
            lr_slice_list = lr_slice_list[:expected_lr_count]
        
        return hr_slice_list, lr_slice_list

    def __getitem__(self, index):
        hr_volumepath, lr_volumepath = self.paired_folders[index]
        # 获取高分辨率和低分辨率的slice列表
        hr_slice_list = [os.path.join(hr_volumepath, f) for f in os.listdir(hr_volumepath) if f.endswith('.pt')]
        lr_slice_list = [os.path.join(lr_volumepath, f) for f in os.listdir(lr_volumepath) if f.endswith('.pt')]
        hr_slice_list.sort()
        lr_slice_list.sort()
        
        # 调整 slice 数量以满足 upscale 关系
        hr_slice_list, lr_slice_list = self._adjust_slice_count(hr_slice_list, lr_slice_list)
        
        def getAVolPaired(hr_slice_list, lr_slice_list):
            max_lr_start = len(lr_slice_list) - self.args.lr_slice_patch
            if max_lr_start < 0:
                raise ValueError(f"Not enough LR slices: LR={len(lr_slice_list)}, need {self.args.lr_slice_patch}")
            
            # 随机选择 LR 的起始索引
            lr_id = random.randint(0, max_lr_start)
            
            # 计算对应的 HR 起始索引
            # LR 的第 lr_id 张对应 HR 的第 lr_id*upscale 张
            hr_id_start = lr_id * self.args.upscale
            hr_id_end = hr_id_start + self.args.hr_slice_patch
            
            # 检查 HR 是否有足够的 slice
            if hr_id_end > len(hr_slice_list):
                raise ValueError(f"Not enough HR slices: HR={len(hr_slice_list)}, need {hr_id_end} (lr_id={lr_id}, upscale={self.args.upscale}, hr_slice_patch={self.args.hr_slice_patch})")
            
            # 加载低分辨率volume（从 lr_id 开始取 lr_slice_patch 个）
            lr_volume = []
            for i in range(self.args.lr_slice_patch):
                with open(lr_slice_list[lr_id + i], 'rb') as _f: 
                    img = pickle.load(_f)
                lr_volume.append(img['image'])
            
            # 加载高分辨率volume（从 hr_id_start 开始取 hr_slice_patch 个）
            hr_volume = []
            for i in range(self.args.hr_slice_patch):
                with open(hr_slice_list[hr_id_start + i], 'rb') as _f: 
                    img = pickle.load(_f)
                hr_volume.append(img['image'])
            
            hr_volume = np.array(hr_volume, dtype=np.float32).transpose(1,2,0)  # [h,w,slice]
            lr_volume = np.array(lr_volume, dtype=np.float32).transpose(1,2,0)  # [h,w,slice]
            
            # 归一化
            hr_volume = util.normalize(hr_volume)  # [0-4095]->[0-1]
            lr_volume = util.normalize(lr_volume)  # [0-4095]->[0-1]
            
            # # 数据增强（对HR和LR同时应用，保持一致性）
            # if random.random() >= 0.5:
            #     hr_volume = hr_volume[:,::-1,:].copy()
            #     lr_volume = lr_volume[:,::-1,:].copy()
            # if random.random() >= 0.5:
            #     hr_volume = hr_volume[::-1,:,:].copy()
            #     lr_volume = lr_volume[::-1,:,:].copy()
            
            # 随机裁剪（对HR和LR使用相同的裁剪位置和尺寸）
            # 假设HR和LR都是相同的像素尺寸（如512x512），都裁剪到256x256
            h_hr, w_hr, c_hr = hr_volume.shape
            h_lr, w_lr, c_lr = lr_volume.shape
            crop_size = 256
            
            # 使用相同的随机裁剪位置，确保HR和LR的空间对应关系
            if h_hr > crop_size and w_hr > crop_size and h_lr > crop_size and w_lr > crop_size:
                # 随机选择裁剪位置（使用HR的尺寸范围，但确保LR也能裁剪）
                max_h = min(h_hr - crop_size, h_lr - crop_size)
                max_w = min(w_hr - crop_size, w_lr - crop_size)
                if max_h >= 0 and max_w >= 0:
                    start_h = random.randint(0, max_h)
                    start_w = random.randint(0, max_w)
                    hr_volume = hr_volume[start_h:start_h+crop_size, 
                                          start_w:start_w+crop_size, :]
                    lr_volume = lr_volume[start_h:start_h+crop_size,
                                         start_w:start_w+crop_size, :]
                else:
                    # 如果无法同时满足，使用中心裁剪
                    hr_volume = util.crop_center(hr_volume, crop_size, crop_size)
                    lr_volume = util.crop_center(lr_volume, crop_size, crop_size)
            else:
                # 如果图像小于裁剪尺寸，使用中心裁剪
                hr_volume = util.crop_center(hr_volume, crop_size, crop_size)
                lr_volume = util.crop_center(lr_volume, crop_size, crop_size)
            
            hr_volume = torch.from_numpy(hr_volume)
            lr_volume = torch.from_numpy(lr_volume)
            return hr_volume, lr_volume

        volumes_hr = []
        volumes_lr = []
        for i in range(self.args.one_batch_n_sample):
            hr_vol, lr_vol = getAVolPaired(hr_slice_list, lr_slice_list)
            volumes_hr.append(hr_vol)
            volumes_lr.append(lr_vol)
        
        volumes_hr = torch.stack(volumes_hr, 0)
        volumes_lr = torch.stack(volumes_lr, 0)
        
        return volumes_hr, volumes_lr

    def __len__(self):
        return self.file_len


class testSet(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.trainlist = [(data_root + '/' + f) for f in os.listdir(data_root)]

        self.file_len = len(self.trainlist)
         
    def __getitem__(self, index):
        volumepath = self.trainlist[index]
        with open(volumepath, 'rb') as _f: volumeIn = pickle.load(_f)
        volumeIn = volumeIn['image'] #[h,w,s] [0,4095]
        volumeIn = util.crop_center(volumeIn,256,256)
        volumeIn = util.normalize(volumeIn).astype(np.float32)
        volumeIn=torch.from_numpy(volumeIn) # w,h,s
        
        name = volumepath.split('/')[-1].split('.')[0]
        return name,volumeIn # [h,w,slice]

    def __len__(self):
        return self.file_len


class testSetPaired(Dataset):
    """
    使用真实配对的高分辨率-低分辨率数据进行测试
    注意：测试使用的是 volume 文件（一个文件包含整个 volume），而不是 slice 文件夹
    """
    def __init__(self, hr_data_root, lr_data_root, args=None):
        self.args = args
        self.hr_data_root = hr_data_root
        self.lr_data_root = lr_data_root
        
        # 获取高分辨率 volume 文件列表
        hr_files = [f for f in os.listdir(hr_data_root) if f.endswith('.pt')]
        hr_files.sort()
        
        # 获取低分辨率文件列表
        lr_files = [f for f in os.listdir(lr_data_root) if f.endswith('.pt')]
        lr_files_dict = {f: f for f in lr_files}  # 用于快速查找
        
        # 匹配配对的低分辨率文件
        # 文件名格式：20251104_陈春蓉_CT480969_094317_1.0 x 1.0_HD_203.pt
        # 对应的LR：20251104_陈春蓉_CT480969_094317_5.0 x 5.0_LUNG_201.pt
        # 注意：末尾数字可能不同（203 vs 201），所以需要提取基础名称进行匹配
        self.paired_files = []
        for hr_file in hr_files:
            # 提取基础名称（到 _1.0 x 1.0_HD 之前的部分）
            if '_1.0 x 1.0_HD' in hr_file:
                base_name = hr_file.split('_1.0 x 1.0_HD')[0]
            elif '_1.0 x 1.0' in hr_file:
                base_name = hr_file.split('_1.0 x 1.0')[0]
            elif '_HD' in hr_file:
                # 如果格式不同，尝试提取到 _HD 之前
                base_name = hr_file.split('_HD')[0]
                # 去掉可能的 _1.0 后缀
                if base_name.endswith('_1.0'):
                    base_name = base_name[:-4]
            else:
                # 如果都没有，尝试提取到最后一个数字之前
                base_name = hr_file.rsplit('_', 2)[0] if hr_file.count('_') >= 2 else hr_file.rsplit('.', 1)[0]
            
            # 在 LR 文件中查找匹配的（以基础名称开头，包含 _5.0 x 5.0_LUNG）
            lr_candidates = [f for f in lr_files 
                           if f.startswith(base_name + '_') and '_5.0 x 5.0_LUNG' in f]
            
            if lr_candidates:
                # 如果找到多个候选，选择第一个（通常只有一个）
                lr_file = lr_candidates[0]
                hr_path = os.path.join(hr_data_root, hr_file)
                lr_path = os.path.join(lr_data_root, lr_file)
                # 提取 volume 名称（去掉扩展名）
                volume_name = hr_file.rsplit('.', 1)[0]
                self.paired_files.append((hr_path, lr_path, volume_name))
            else:
                # 调试信息：如果没找到匹配
                if len(self.paired_files) < 3:  # 只打印前3个未匹配的，避免输出太多
                    print(f"警告: 未找到 HR 文件 {hr_file} 的配对 LR 文件（基础名称: {base_name}）")
        
        self.file_len = len(self.paired_files)
        if self.file_len == 0:
            print(f"错误: 未找到任何配对的测试 volume 文件！")
            print(f"HR 目录: {hr_data_root}，找到 {len(hr_files)} 个文件")
            print(f"LR 目录: {lr_data_root}，找到 {len(lr_files)} 个文件")
            if len(hr_files) > 0:
                print(f"HR 文件示例: {hr_files[0]}")
            if len(lr_files) > 0:
                print(f"LR 文件示例: {lr_files[0]}")
        else:
            print(f"找到 {self.file_len} 对配对的测试 volume 文件")
    
    def _adjust_slice_count(self, hr_volume, lr_volume):
        """
        调整 HR 和 LR volume 的 slice 数量（第3维），使其满足 upscale 关系
        关系：HR_slice_count = (LR_slice_count - 1) * upscale + 1
        如果不满足，删除末尾的 corner slices
        输入：hr_volume [h,w,s], lr_volume [h,w,s]
        输出：调整后的 hr_volume 和 lr_volume
        """
        hr_slice_count = hr_volume.shape[2]
        lr_slice_count = lr_volume.shape[2]
        
        # 计算期望的 slice 数量关系
        expected_hr_count = (lr_slice_count - 1) * self.args.upscale + 1
        expected_lr_count = (hr_slice_count - 1) // self.args.upscale + 1
        
        # 如果 HR slice 数量过多，删除末尾的 corner slices
        if hr_slice_count > expected_hr_count:
            excess = hr_slice_count - expected_hr_count
            hr_volume = hr_volume[:, :, :-excess]
            hr_slice_count = hr_volume.shape[2]
        
        # 如果 LR slice 数量过多，删除末尾的 corner slices
        if lr_slice_count > expected_lr_count:
            excess = lr_slice_count - expected_lr_count
            lr_volume = lr_volume[:, :, :-excess]
            lr_slice_count = lr_volume.shape[2]
        
        # 再次检查并调整到满足关系
        expected_hr_count = (lr_slice_count - 1) * self.args.upscale + 1
        expected_lr_count = (hr_slice_count - 1) // self.args.upscale + 1
        
        if hr_slice_count != expected_hr_count:
            hr_volume = hr_volume[:, :, :expected_hr_count]
        if lr_slice_count != expected_lr_count:
            lr_volume = lr_volume[:, :, :expected_lr_count]
        
        return hr_volume, lr_volume
         
    def __getitem__(self, index):
        hr_filepath, lr_filepath, volume_name = self.paired_files[index]
        
        # 加载高分辨率 volume（一个文件包含整个 volume）
        with open(hr_filepath, 'rb') as _f: 
            hr_data = pickle.load(_f)
        hr_volume = hr_data['image']  # [h,w,s] [0,4095]
        
        # 加载低分辨率 volume
        with open(lr_filepath, 'rb') as _f: 
            lr_data = pickle.load(_f)
        lr_volume = lr_data['image']  # [h,w,s] [0,4095]
        
        # 调整 slice 数量以满足 upscale 关系
        hr_volume, lr_volume = self._adjust_slice_count(hr_volume, lr_volume)
        
        # 归一化
        hr_volume = util.normalize(hr_volume).astype(np.float32)  # [0-4095]->[0-1]
        lr_volume = util.normalize(lr_volume).astype(np.float32)  # [0-4095]->[0-1]
        
        # 中心裁剪
        hr_volume = util.crop_center(hr_volume, 256, 256)
        lr_volume = util.crop_center(lr_volume, 256, 256)
        
        hr_volume = torch.from_numpy(hr_volume)
        lr_volume = torch.from_numpy(lr_volume)
        
        return volume_name, hr_volume, lr_volume  # [h,w,slice]

    def __len__(self):
        return self.file_len

# 读取数据的代码，数据是由原始的dicom数据直接转化为nii.gz文件，没有经过预处理
class testHuaxi_sitk(Dataset):
    def __init__(self, data_root, gt_root):
        self.data_root = data_root
        self.gt_root = gt_root
        self.trainlist = []
        for f in os.listdir(data_root):
            if 'LUNG' in f:
                self.trainlist.append(self.data_root+'/'+f)
        self.gtlist = []
        for f in os.listdir(gt_root):
            self.gtlist.append(self.gt_root+'/'+f)
        self.trainlist.sort()
        self.gtlist.sort()
        #print(self.gtlist)
        self.file_len = len(self.trainlist)
         
    def __getitem__(self, index):
        volumepath = self.trainlist[index]
        data = sitk.ReadImage(volumepath)
        
        meta = {
        'spacing': data.GetSpacing(),
        'origin': data.GetOrigin(),
        'direction': data.GetDirection()
        }
        volumeIn = sitk.GetArrayFromImage(data)
        #print(volumeIn.shape)
        volumeIn = np.clip(volumeIn, -1024, volumeIn.max())
        volumeIn = volumeIn - volumeIn.min()
        volumeIn = np.clip(volumeIn, 0, 4095)
        volumeIn = volumeIn.astype(np.float32)
        volumeIn = volumeIn / 4095.0
        volumeIn = np.transpose(volumeIn,(1,2,0))

        volumeIn = util.crop_center(volumeIn, 256, 256)

        volumeIn=torch.from_numpy(volumeIn) # w,h,s
        
        name = volumepath.split('/')[-1].split('.')[0]
        return name,volumeIn,meta

    def __len__(self):
        return self.file_len


