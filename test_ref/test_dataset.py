# random

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import nibabel as nib
from medpy.io import load
import random 
import SimpleITK as sitk

import pickle
import util

import cv2

# 读取数据的代码，数据是由原始的dicom数据直接转化为nii.gz文件，没有经过预处理
class testHuaxi_sitk(Dataset):
    def __init__(self, data_root, gt_root):
        self.data_root = data_root
        self.gt_root = gt_root
        self.trainlist = []
        for f in os.listdir(data_root):
            if 'lung' in f:
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
        volumeIn = np.clip(volumeIn,-1024,volumeIn.max())
        volumeIn = volumeIn - volumeIn.min()
        volumeIn = np.clip(volumeIn,0,4096)
        volumeIn = volumeIn.astype(np.float32)
        volumeIn = volumeIn/4096
        volumeIn = np.transpose(volumeIn,(1,2,0))
        volumeIn=torch.from_numpy(volumeIn) # w,h,s
        
        name = volumepath.split('/')[-1].split('.')[0]
        return name,volumeIn,meta

    def __len__(self):
        return self.file_len
