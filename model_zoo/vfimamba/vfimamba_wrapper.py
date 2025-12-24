"""
VFIMamba Wrapper for I3Net
将 VFIMamba 视频帧插值模型适配到 I3Net 的医学图像切片插值任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .feature_extractor import feature_extractor as mamba_extractor
from .flow_estimation import MultiScaleFlow as mamba_estimation


def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4], M=False):
    '''Initialize model configuration'''
    return { 
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'mlp_ratios':[4 for i in range(len(depth)-3)],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W for i in range(len(depth)-3)],
        'conv_stages':3
    }, {
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'window_sizes':[W, W],
        'scales':[4*(2**i) for i in range(len(depth)-2)],
        'hidden_dims':[4*F for i in range(len(depth)-3)],
        'c':F,
        'M':M,
        'local_hidden_dims':4*F,
        'local_num':2
    }


class VFIMambaInterpolator(nn.Module):
    """
    将 VFIMamba 适配到 I3Net 的切片插值任务
    输入: [B, H, W, S] 其中 S 是切片数 (通常是 lr_slice_patch)
    输出: [B, H, W, S_out] 其中 S_out = (S-1)*upscale + 1
    
    使用递归二分法效率更高，每次调用网络只处理两张切片对
    """
    def __init__(self, F=16, depth=[2, 2, 2, 3, 3], M=False, local=2, upscale=5):
        super().__init__()
        backbonecfg, multiscalecfg = init_model_config(F=F, depth=depth, M=M)
        self.net = mamba_estimation(mamba_extractor(**backbonecfg), **multiscalecfg)
        self.local = local
        self.upscale = upscale
        
    def infer_pair(self, slice0, slice1, timestep=0.5):
        """
        在两个切片之间插值生成一张中间切片，并返回重建的端点切片
        slice0, slice1: [B, H, W] -> 转换为 [B, 3, H, W]
        timestep: 插值时间步（0 < timestep < 1）
        返回: (pred_t, pred_0, pred_1) 均为 [B, H, W]
        """
        B, H, W = slice0.shape
        
        # 将单通道图像转换为3通道 (VFIMamba 期望RGB输入)
        slice0_rgb = slice0.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
        slice1_rgb = slice1.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        imgs = torch.cat((slice0_rgb, slice1_rgb), 1)  # [B, 6, H, W]
        
        # 调用网络进行插值
        # 现在 net 返回 (flow_list, mask_list, merged, pred, pred0, pred1)
        res = self.net(imgs, timestep=timestep, scale=0, local=self.local)
        pred, pred0, pred1 = res[3], res[4], res[5]
        
        # 转换回单通道 (取RGB的平均)
        pred_gray = pred.mean(dim=1)
        pred0_gray = pred0.mean(dim=1)
        pred1_gray = pred1.mean(dim=1)
        
        return pred_gray, pred0_gray, pred1_gray
    
    def _fill_intermediate(self, slices_dict, slice0, slice1, start_idx, end_idx):
        """
        递归填充两个索引之间的中间切片
        """
        if end_idx - start_idx <= 1:
            return
        
        mid_idx = (start_idx + end_idx) // 2
        timestep = (mid_idx - start_idx) / (end_idx - start_idx)
        
        # 插值生成中间帧
        mid_slice, _, _ = self.infer_pair(slice0, slice1, timestep=timestep)
        slices_dict[mid_idx] = mid_slice
        
        # 递归处理左右两半
        self._fill_intermediate(slices_dict, slice0, mid_slice, start_idx, mid_idx)
        self._fill_intermediate(slices_dict, mid_slice, slice1, mid_idx, end_idx)

    def forward(self, x):
        """
        x: [B, H, W, S] 其中 S 是输入切片数
        返回: [B, H, W, S_out] 其中 S_out = (S-1)*upscale + 1
        """
        B, H, W, S = x.shape
        S_out = (S - 1) * self.upscale + 1
        
        all_slices_dict = {}
        
        for pair_idx in range(S - 1):
            s0 = x[:, :, :, pair_idx]
            s1 = x[:, :, :, pair_idx + 1]
            
            # 1. 获取重建的关键帧和中间帧
            # 我们使用 timestep=0.5 来获取重建的端点
            # 注意：这里的 mid_slice 只是为了获取 rec0 和 rec1，真正的中间帧由递归生成
            _, rec0, rec1 = self.infer_pair(s0, s1, timestep=0.5)
            
            idx0 = pair_idx * self.upscale
            idx1 = (pair_idx + 1) * self.upscale
            
            # 存储重建的关键帧（如果是中间的关键帧，则取平均）
            if idx0 not in all_slices_dict:
                all_slices_dict[idx0] = rec0
            else:
                all_slices_dict[idx0] = (all_slices_dict[idx0] + rec0) / 2.0
                
            if idx1 not in all_slices_dict:
                all_slices_dict[idx1] = rec1
            else:
                all_slices_dict[idx1] = (all_slices_dict[idx1] + rec1) / 2.0
        
        # 2. 递归填充所有中间切片
        # 为了保证一致性，我们使用重建后的关键帧作为递归的起点
        for pair_idx in range(S - 1):
            idx0 = pair_idx * self.upscale
            idx1 = (pair_idx + 1) * self.upscale
            rec0 = all_slices_dict[idx0]
            rec1 = all_slices_dict[idx1]
            
            self._fill_intermediate(all_slices_dict, rec0, rec1, idx0, idx1)
        
        # 3. 排序并堆叠
        sorted_indices = sorted(all_slices_dict.keys())
        output = torch.stack([all_slices_dict[idx] for idx in sorted_indices], dim=3)
        
        return output


def convert(param):
    """转换checkpoint的key，去除 'module.' 前缀"""
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k and 'attn_mask' not in k and 'HW' not in k
    }


def load_vfimamba_model(checkpoint_path, F=16, depth=[2, 2, 2, 3, 3], M=False, local=2, upscale=5):
    """
    加载训练好的 VFIMamba 模型
    """
    model = VFIMambaInterpolator(F=F, depth=depth, M=M, local=local, upscale=upscale)
    
    if checkpoint_path:
        print(f"Loading VFIMamba checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 如果 checkpoint 是字典且包含 'state_dict' key
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 转换 key
        state_dict = convert(state_dict)
        
        # 加载权重
        # 使用 strict=False 因为我们修改了 unet.conv 的输出通道数 (从 3 变为 9)
        model.net.load_state_dict(state_dict, strict=False)
        print("VFIMamba model loaded successfully (strict=False)!")
    
    return model
