import os 
import config
args, unparsed = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from medpy.io import load
import os, pickle
import SimpleITK as sitk

import torch
from data import testHuaxi_sitk as testRealSet
from util_evaluation import calc_psnr,calc_ssim
from select_model import select_model
import cv2
import numpy as np
from my_model.smoe import UTaMoE_Prompt

def main():
    args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        s = "\n\n\n\n\nSTART EXPERIMENT\n"
        f.write(s)
        f.write('testdata:'+args.testdata_path+'\n')
        f.write('checkpoint:'+args.ckpt+'\n')

    #model = select_model(args)
    model = UTaMoE_Prompt(base_filter=64, num_spectral=4, num_head=4, moe_factor=12, num_experts=4)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load:{args.ckpt}')
    model = model.cuda()
    model.eval()
    # 加载数据
    testset = testRealSet(data_root=args.testdata_path, gt_root = args.testdata_gt_path)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=4, pin_memory=False)

    for id, (name,volumeIn,meta) in enumerate(dataloader):
        os.makedirs('./result/'+name[0]+'/', exist_ok=True)
        os.makedirs('./label/'+name[0]+'/', exist_ok=True)

        lr = volumeIn.squeeze(0)
        #print(type(meta['spacing']))
        meta['spacing'][-1] = 1.0
        # 用来划窗
        print(lr.shape)
        sr = torch.zeros(lr.shape[0],lr.shape[1],lr.shape[-1]*5-4)
        #print(sr.shape)
        sr_cnt = torch.zeros(lr.shape[0],lr.shape[1],lr.shape[-1]*5-4)

        for tmp_s in range(lr.shape[2]-args.lr_slice_patch+1):
            #取输入图像中的4张图像
            tmp_lr = lr[...,tmp_s:tmp_s+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).cuda() #[1,s,h,w]
            # 模型预测
            with torch.no_grad():
                _,tmp_sr,_ = model(tmp_lr)

            tmp_sr = torch.clamp(tmp_sr.squeeze(0),0,1).cpu()
            #保存模型预测结果
            sr[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += tmp_sr
            #计算输入图像的每个切片在滑窗的过程中重复计算的次数
            sr_cnt[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += 1

        sr = sr[...,args.upscale : -1*args.upscale]
        sr_cnt = sr_cnt[...,args.upscale : -1*args.upscale]
        # 最终预测结果
        sr /= sr_cnt #[h,w,s]

        # print(sr.shape) # h w s
        sr = sr.cuda()
        sr = sr.cpu().data.squeeze().clamp(0, 1).numpy().transpose(2,0,1)
        #print(sr.shape)
        img = 4096*sr
        img = img.astype("uint16")
        new_img = sitk.GetImageFromArray(img)
        
        new_img.SetOrigin(tuple(float(o) for o in meta['origin']))
        new_img.SetSpacing(tuple(float(s) for s in meta['spacing']))
        new_img.SetDirection(tuple(float(d) for d in meta['direction']))
        # new_img.SetOrigin(meta['origin'])
        # new_img.SetSpacing(meta['spacing'])
        # new_img.SetDirection(meta['direction'])
        #new_img.CopyInformation(data)
        output_path = './result/' + name[0]+'.nii.gz'
        sitk.WriteImage(new_img, output_path)
        #pickle.dump({'image':img,'spacing':spacing},open(savefile, 'wb'))

        
if __name__ == "__main__":
    main()
