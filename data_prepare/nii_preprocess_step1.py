from medpy.io import load
import os, pickle
import numpy as np

src_path = '/Data2/XueYangWang/Medical_SR/data/MSD/Task10_Colon/imagesTs'
tgt_path = '/Data2/XueYangWang/Medical_SR/data_test_test_clone'

os.makedirs(tgt_path,exist_ok=True)

patients = os.listdir(src_path)
print(patients)
for id,patient in enumerate(patients):
        if not '._' in patient:
            img, header = load(os.path.join(src_path, patient))
            min_val = np.min(img)
            max_val = np.max(img)
            mean_val = np.mean(img)
            print(f"min_val: {min_val}, max_val: {max_val}, mean_val: {mean_val}")
            spacing = header.get_voxel_spacing()
            img = np.clip(img,-1024,img.max())
            img = img - img.min()
            img = np.clip(img,0,4095)
            img = img.astype("uint16")
            data = {'image': img, 'spacing': spacing}
            pickle.dump(data, open(os.path.join(tgt_path, patient.replace('.nii.gz','.pt')), 'wb'))
            print(f"{id}/{len(patients)}:volume finished, " + patient, img.shape)

print('done')