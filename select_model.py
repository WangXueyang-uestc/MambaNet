from opt import *
import torch.nn as nn
import json
from importlib import import_module

def args_add_additinoal_attr(args,json_path):
    dic = json.load(open(json_path,'r',))
    for key,value in dic.items():
        if key == '//':
            continue
        setattr(args,key,value)

def select_model(args):
    # 如果选择 VFIMamba 模型
    if args.model.lower() == 'vfimamba':
        from model_zoo.vfimamba.vfimamba_wrapper import load_vfimamba_model
        import json
        
        # VFIMamba 的配置
        F = getattr(args, 'vfimamba_F', 16)
        
        # 解析 depth 参数（可能是字符串或列表）
        depth_raw = getattr(args, 'vfimamba_depth', [2, 2, 2, 3, 3])
        if isinstance(depth_raw, str):
            # 如果是字符串，解析为列表
            depth = json.loads(depth_raw)
        else:
            depth = depth_raw
        
        M = getattr(args, 'vfimamba_M', False)
        local = getattr(args, 'vfimamba_local', 2)
        upscale = getattr(args, 'upscale', 5)
        
        # 如果提供了checkpoint路径，加载预训练模型
        checkpoint_path = getattr(args, 'vfimamba_ckpt', None)
        if checkpoint_path == '':
            checkpoint_path = None
        model = load_vfimamba_model(checkpoint_path, F=F, depth=depth, M=M, local=local, upscale=upscale)
        return model
    
    # 原有的 I3Net 模型选择逻辑
    opt_path = f'opt/{args.model}.json'
    args_add_additinoal_attr(args, opt_path)
    module = import_module(f'model_zoo.{args.model.lower()}.basic_model')
    
    # 支持两种模式：原始 I3Net 和改进的 RefinedI3Net
    if hasattr(args, 'use_refined_model') and args.use_refined_model:
        # 使用改进的联合插值与增强框架
        model = module.RefinedI3Net(args)
    else:
        # 使用原始的插值优先策略
        model = module.make_model(args)
    
    return model

