# following SAINT https://github.com/cpeng93/SAINT, split train/test volume
# save volume.pt: {'image':image_h_w_s.dtype('uint16'),'spacing':(x,y,z)}, image_scale=0-4095

from medpy.io import load
import os, pickle
import nibabel as nib
import numpy as np


datapath = '/home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_train/'
volume_list = os.listdir(datapath)
savepath = '/home/user/XueYangWang/HX_data2/data_processed_slice_hd_hx_train/'
os.makedirs(savepath,exist_ok=True)

for i,volumename in enumerate(volume_list):
    volume_path = datapath + volumename

    data = pickle.load(open(volume_path,'rb'))
    volnp = data['image'].astype("uint16")
    spacing = data['spacing']

    savefile = savepath + volumename.split('.')[0]
    os.makedirs(savefile,exist_ok=True)

    for frame in range(volnp.shape[2]):
        vol_slice = volnp[:,:,frame]
        savename = savefile + '/' + volumename.split('.')[0] + '_slice_' + str(frame).zfill(3) + '.pt'
        pickle.dump({'image':vol_slice,'spacing':spacing},open(savename, 'wb'))

    print(f'{i} / {len(volume_list)}  ' + savefile)

print('done')
