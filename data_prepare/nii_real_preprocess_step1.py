# following SAINT https://github.com/cpeng93/SAINT, split train/test volume
# save volume.pt: {'image':image_h_w_s.dtype('uint16'),'spacing':(x,y,z)}, image_scale=0-4095

from medpy.io import load
import os, pickle
import numpy as np

#src_path = 'xxx/Task03_Liver'
# 原图文件地址
src_path = '/home/wangwu/wangwu/3DCT/RPLHR-CT-main/RPLHR-CT-tiny/test/1mm'
#tgt_path = 'xxx/data_volume/Task03_Liver'
#可以生成的新地址
tgt_path = '/home/wangwu/wangwu/3DCT/RPLHR-CT-main/RPLHR-CT-tiny/test/preprocess/1mm'

os.makedirs(tgt_path,exist_ok=True)

patients = os.listdir(src_path)
print(patients)
for id,patient in enumerate(patients):
        if not '._' in patient:
            img, header = load(os.path.join(src_path, patient))
            print(img.max(),img.min())
            spacing = header.get_voxel_spacing()
            # 预处理，调整像素到0，2^15
            #img = np.clip(img,-1024,img.max())
            #img = img - img.min()
            #img = np.clip(img,0,4095)
            img = img*4096
            img = img.astype("uint16")
            # 另存，保留图像和spacing
            data = {'image': img, 'spacing': spacing}
            pickle.dump(data, open(os.path.join(tgt_path, patient.replace('.nii.gz','.pt')), 'wb'))
            print(f"{id}/{len(patients)}:volume finished, " + patient, img.shape)

print('done')