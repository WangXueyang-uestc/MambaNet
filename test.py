import os 
import config
import numpy as np # 引入 numpy
import cv2 # 引入 cv2 用于保存图片
args, unparsed = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
import SimpleITK as sitk
from data import testSet, testSetPaired, testHuaxi_sitk
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
        f.write('checkpoint:'+args.ckpt+'\n')

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load:{args.ckpt}')
    model = model.to(device)
    model.eval()

    # 使用 testHuaxi_sitk 加载数据
    # 优先使用 test_lr_data_path 作为数据根目录
    if hasattr(args, 'test_lr_data_path') and args.test_lr_data_path:
        data_root = args.test_lr_data_path
    else:
        data_root = args.testdata_path
        
    # gt_root 只是为了满足 __init__，实际不使用
    gt_root = data_root 
    
    print(f"使用 testHuaxi_sitk 加载数据: {data_root}")
    testset = testHuaxi_sitk(data_root=data_root, gt_root=gt_root)
    
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=getattr(args, 'num_workers', 4), pin_memory=True)

    if len(dataloader) == 0:
        print("错误: 测试数据集为空，无法进行测试！")
        return

    # 创建结果保存目录
    result_dir = os.path.join(args.ckpt_dir, 'result_nii')
    os.makedirs(result_dir, exist_ok=True)

    for id, (name, volumeIn, meta) in enumerate(dataloader):
        vol_name = name[0]
        print(f"Processing {vol_name}...")
        
        lr = volumeIn.squeeze(0) # [H, W, S]
        
        # 计算输出尺寸
        h, w, s_in = lr.shape
        s_out = (s_in - 1) * args.upscale + 1
        
        sr = torch.zeros(h, w, s_out, device=device)
        sr_cnt = torch.zeros(h, w, s_out, device=device)
        
        # 滑窗预测
        for tmp_s in range(s_in - args.lr_slice_patch + 1):
            tmp_lr = lr[..., tmp_s : tmp_s + args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).to(device) # [1, H, W, patch_s]
            
            with torch.no_grad():
                tmp_sr = model(tmp_lr)
            
            tmp_sr = torch.clamp(tmp_sr.squeeze(0), 0, 1)
            
            start = tmp_s * args.upscale
            end = start + ((args.lr_slice_patch - 1) * args.upscale + 1)
            
            sr[..., start:end] += tmp_sr
            sr_cnt[..., start:end] += 1
            
        # 裁剪边界
        sr = sr[..., args.upscale : -args.upscale]
        sr_cnt = sr_cnt[..., args.upscale : -args.upscale]
        
        sr /= sr_cnt
        
        # 保存结果
        sr = sr.cpu().numpy().transpose(2, 0, 1) # [S, H, W] for sitk
        
        # 还原数值范围 (0-1 -> 0-4095)
        img = 4095 * sr
        img = img.astype("uint16")
        
        new_img = sitk.GetImageFromArray(img)
        
        # 设置元数据
        new_img.SetOrigin(tuple(float(o) for o in meta['origin']))
        
        # 调整 Z 轴 spacing
        spacing = [float(s) for s in meta['spacing']]
        spacing[2] = spacing[2] / args.upscale
        new_img.SetSpacing(tuple(spacing))
        
        new_img.SetDirection(tuple(float(d) for d in meta['direction']))
        
        output_path = os.path.join(result_dir, vol_name + '.nii.gz')
        sitk.WriteImage(new_img, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()