import os 
import config
import numpy as np # 引入 numpy
import cv2 # 引入 cv2 用于保存图片
args, unparsed = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
from data import testSet, testSetPaired
from util_evaluation import calc_psnr,calc_ssim
from select_model import select_model

def normalize_image(img, method='stretch'):
    """
    归一化图像以改善对比度，方便观察
    
    Args:
        img: 输入图像 (numpy array, 0-1 或 0-255)
        method: 归一化方法
            - 'stretch': 线性拉伸到0-255（适合值范围小的图像，默认）
            - 'percentile': 使用百分位数裁剪然后归一化到0-255
            - 'minmax': 最小-最大归一化
    
    Returns:
        归一化后的图像 (uint8, 0-255)
    """
    # 如果输入是0-1范围，先转换到0-255
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    if method == 'stretch':
        # 线性拉伸：直接将当前值范围映射到0-255
        # 特别适合值范围很小的图像（如0-50）
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img_norm = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = img.astype(np.uint8)
        return img_norm
    
    elif method == 'percentile':
        # 使用百分位数裁剪，去除极端值
        lower = np.percentile(img, 1)
        upper = np.percentile(img, 99)
        img_clipped = np.clip(img, lower, upper)
        # 归一化到0-255
        if upper > lower:
            img_norm = ((img_clipped - lower) / (upper - lower) * 255).astype(np.uint8)
        else:
            img_norm = img_clipped.astype(np.uint8)
        return img_norm
    
    elif method == 'minmax':
        # 最小-最大归一化
        if img.max() > img.min():
            img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img_norm = img.astype(np.uint8)
        return img_norm
    
    else:
        return img


# 定义一个保存图片的辅助函数，避免主循环太乱
def save_slices(gt_vol, sr_vol, volume_name, save_root, normalize=True):
    """
    保存GT和SR的切片图片
    
    Args:
        gt_vol: Ground Truth volume [H, W, S], 数值范围 0-1
        sr_vol: Super-Resolution volume [H, W, S], 数值范围 0-1
        volume_name: volume名称
        save_root: 保存根目录
        normalize: 是否进行归一化处理以改善对比度（默认True）
    """
    # 确保保存路径存在: experiments/.../visual_results/volume_name/
    save_dir = os.path.join(save_root, 'visual_results', volume_name)
    os.makedirs(save_dir, exist_ok=True)

    # 将 Tensor 转为 Numpy，并确保在 CPU 上
    # 假设输入是 [H, W, S]，数值范围 0-1
    gt_np = gt_vol.cpu().numpy()
    sr_np = sr_vol.cpu().numpy()

    # 遍历每一个切片 (假设第3维是深度/切片维)
    depth = gt_np.shape[2]
    for z in range(depth):
        # 取出切片
        img_gt = gt_np[:, :, z]
        img_sr = sr_np[:, :, z]

        # 归一化处理以改善对比度
        if normalize:
            # 对于值范围很小的图像，使用stretch方法效果更好
            # 先转换到0-255范围
            img_gt_uint8 = (img_gt * 255.0).clip(0, 255).astype(np.uint8)
            img_sr_uint8 = (img_sr * 255.0).clip(0, 255).astype(np.uint8)
            
            # 如果值范围很小（最大值小于100），使用stretch方法
            if img_gt_uint8.max() < 100 or img_sr_uint8.max() < 100:
                img_gt = normalize_image(img_gt, method='stretch')
                img_sr = normalize_image(img_sr, method='stretch')
            else:
                # 否则使用percentile方法（适合医学图像）
                img_gt = normalize_image(img_gt, method='percentile')
                img_sr = normalize_image(img_sr, method='percentile')
        else:
            # 不归一化，直接映射到0-255
            img_gt = (img_gt * 255.0).clip(0, 255).astype(np.uint8)
            img_sr = (img_sr * 255.0).clip(0, 255).astype(np.uint8)

        # 保存单独的GT和SR图片
        cv2.imwrite(os.path.join(save_dir, f'{z:04d}_GT.png'), img_gt)
        cv2.imwrite(os.path.join(save_dir, f'{z:04d}_SR.png'), img_sr)
        
        # 同时保存并排对比图，方便直接观察
        # 添加分隔线（2像素宽的白色线）
        separator = np.ones((img_gt.shape[0], 2), dtype=np.uint8) * 255
        cat_img = np.concatenate((img_gt, separator, img_sr), axis=1)  # 左右拼接
        
        # 添加文字标签
        cat_img_bgr = cv2.cvtColor(cat_img, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (255, 255, 255)  # 白色文字
        cv2.putText(cat_img_bgr, 'GT', (5, 20), font, font_scale, color, thickness)
        cv2.putText(cat_img_bgr, 'SR', (img_gt.shape[1] + 5, 20), font, font_scale, color, thickness)
        
        cv2.imwrite(os.path.join(save_dir, f'{z:04d}_compare.png'), cat_img_bgr)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
    
    # 确保日志文件夹存在
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        s = "\n\n\n\n\nSTART EXPERIMENT\n"
        f.write(s)
        f.write('testdata:'+args.testdata_path+'\n')
        f.write('checkpoint:'+args.ckpt+'\n')

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load:{args.ckpt}')
    model = model.to(device)
    model.eval()

    # 根据训练模式选择测试集
    if args.train_mode == 'paired':
        # 使用配对数据模式
        # 优先使用指定的测试路径，否则从训练路径推导
        if hasattr(args, 'test_hr_data_path') and args.test_hr_data_path:
            test_hr_path = args.test_hr_data_path
        else:
            test_hr_path = args.hr_data_path.replace('train', 'test') if 'train' in args.hr_data_path else args.hr_data_path
        
        if hasattr(args, 'test_lr_data_path') and args.test_lr_data_path:
            test_lr_path = args.test_lr_data_path
        else:
            test_lr_path = args.lr_data_path.replace('train', 'test') if 'train' in args.lr_data_path else args.lr_data_path
        
        print(f"使用配对数据测试模式: HR={test_hr_path}, LR={test_lr_path}")
        testset = testSetPaired(hr_data_root=test_hr_path,
                               lr_data_root=test_lr_path,
                               args=args)
    else:
        # 使用下采样模式（原有方式）
        print(f"使用下采样测试模式: {args.testdata_path}")
        testset = testSet(data_root=args.testdata_path)
    
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=getattr(args, 'num_workers', 4), pin_memory=True)

    average_psnr=0
    total_x_y_ssim=0
    total_x_z_ssim=0
    total_y_z_ssim=0

    if len(dataloader) == 0:
        print("错误: 测试数据集为空，无法进行测试！")
        return

    for id, data in enumerate(dataloader):
        if args.train_mode == 'paired':
            # 配对数据模式：直接使用加载的HR和LR
            name, gt, lr = data
            gt = gt.squeeze(0)  # [h,w,s]
            lr = lr.squeeze(0)  # [h,w,s]
        else:
            # 下采样模式：原有逻辑
            name, volume = data
            gt = volume.squeeze(0)  # [h,w,s]
            m = (gt.shape[2]-1) % args.upscale 
            if m != 0:
                gt = gt[...,:-m]
            lr = gt[...,::args.upscale]
        
        psnr = 0
        x_y_ssim=0 
        x_z_ssim=0
        y_z_ssim=0

        sr = torch.zeros_like(gt, device=device)
        sr_cnt = torch.zeros_like(gt, device=device)

        for tmp_s in range(lr.shape[2]-args.lr_slice_patch+1):
            tmp_lr = lr[...,tmp_s:tmp_s+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).to(device, non_blocking=True) #[1,s,h,w]
        
            with torch.no_grad():
                tmp_sr = model(tmp_lr)

            tmp_sr = torch.clamp(tmp_sr.squeeze(0),0,1)

            sr[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += tmp_sr
            sr_cnt[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += 1

        sr = sr[...,args.upscale : -1*args.upscale]
        sr_cnt = sr_cnt[...,args.upscale : -1*args.upscale]
        gt = gt[...,args.upscale : -1*args.upscale]

        sr /= sr_cnt #[h,w,s]

        # ---------------------------------------------------------
        # 【修改区域】在此处保存图片
        # ---------------------------------------------------------
        # 注意：name 在 dataloader 中通常是一个 tuple ('filename',)，取第一个元素
        vol_name_str = str(name[0]) 
        print(f"Saving images for volume: {vol_name_str} ...")
        save_slices(gt, sr, vol_name_str, args.ckpt_dir)
        # ---------------------------------------------------------

        # print(sr.shape) # h w s
        gt = gt.to(device, non_blocking=True)
        psnr = calc_psnr(sr,gt).item()
        average_psnr += psnr

        for i in range(gt.shape[2]):
            ssim = calc_ssim(gt[:,:,i],sr[:,:,i])
            x_y_ssim += ssim
        x_y_ssim /= (i+1)
        for i in range(gt.shape[0]):
            ssim = calc_ssim(gt[i,:,:],sr[i,:,:])
            x_z_ssim += ssim
        x_z_ssim /= (i+1)
        for i in range(gt.shape[1]):
            ssim = calc_ssim(gt[:,i,:],sr[:,i,:])
            y_z_ssim += ssim
        y_z_ssim /= (i+1)
        log = r"[{} / {}] NAME:{} PSNR:{} x_y_ssim:{:.4f} x_z_ssim:{:.4f} y_z_ssim:{:.4f} "\
            .format(id+1,dataloader.__len__(),name,psnr,x_y_ssim, x_z_ssim,y_z_ssim)
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 

        total_x_y_ssim+=x_y_ssim
        total_x_z_ssim+=x_z_ssim
        total_y_z_ssim+=y_z_ssim

    average_psnr /= (id+1)
    total_x_y_ssim /= (id+1)
    total_x_z_ssim /= (id+1)
    total_y_z_ssim /= (id+1)
    print("average_psnr:",average_psnr) 
    print("average_x_y_ssim:",total_x_y_ssim) 
    print("average_x_z_ssim:",total_x_z_ssim) 
    print("average_y_z_ssim:",total_y_z_ssim) 

    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        log = r"PSNR: {} x_y_ssim: {:.6f} x_z_ssim: {:.6f} y_z_ssim: {:.6f}".format(average_psnr,total_x_y_ssim,total_x_z_ssim,total_y_z_ssim)
        f.write(log)


if __name__ == "__main__":
    main()