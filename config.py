import argparse
import os

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--data_type', type=str, default='direct')
data_arg.add_argument('--train_mode', type=str, default='downsample', choices=['downsample', 'paired'], 
                      help='训练模式: downsample=高分辨率下采样, paired=真实配对数据')
data_arg.add_argument('--lr_slice_patch', type=int, default=4, help='每个lr样本的slice个数,插值为中间3个slice')
data_arg.add_argument('--traindata_path', type=str, default='/home/user/XueYangWang/public_data/data_processed_slice')
data_arg.add_argument('--testdata_path', type=str, default='/data/shf/IXI/IXI_T1_HH_volume/test')
# 配对数据路径（当 train_mode='paired' 时使用）
data_arg.add_argument('--hr_data_path', type=str, default='/Data2/XueYangWang/Medical_SR/HX_data2/data_processed_slice_hd_hx_train',
                      help='高分辨率数据路径（配对模式训练）')
data_arg.add_argument('--lr_data_path', type=str, default='/Data2/XueYangWang/Medical_SR/HX_data2/data_processed_slice_lung_hx_train',
                      help='低分辨率数据路径（配对模式训练）')
data_arg.add_argument('--test_hr_data_path', type=str, default='',
                      help='高分辨率测试数据路径（配对模式测试，为空则使用hr_data_path替换train为test）')
data_arg.add_argument('--test_lr_data_path', type=str, default='',
                      help='低分辨率测试数据路径（配对模式测试，为空则使用lr_data_path替换train为test）')


# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--model', type=str, default='', help='select model')
model_arg.add_argument('--upscale', type=int, default=2, help='scale_factor')
model_arg.add_argument("--resume", type=bool, default=False, help='run resume or not')
model_arg.add_argument('--ckpt', type=str, default='', help='pretrained model path')
# 新增：改进模型参数
model_arg.add_argument('--use_refined_model', type=str2bool, default='False', 
                      help='是否使用改进的 RefinedI3Net（联合插值与增强）')
model_arg.add_argument('--n_refinement_blocks', type=int, default=2,
                      help='关键帧优化分支中的 refinement blocks 数量')
# VFIMamba 模型参数
model_arg.add_argument('--vfimamba_ckpt', type=str, default='',
                      help='VFIMamba 预训练模型路径 (.pkl 或 .pth)')
model_arg.add_argument('--vfimamba_F', type=int, default=16,
                      help='VFIMamba 模型特征通道基数')
model_arg.add_argument('--vfimamba_depth', type=str, default='[2,2,2,3,3]',
                      help='VFIMamba 模型深度配置（JSON格式）')
model_arg.add_argument('--vfimamba_M', type=str2bool, default='False',
                      help='VFIMamba 模型 M 参数')
model_arg.add_argument('--vfimamba_local', type=int, default=2,
                      help='VFIMamba 局部refinement参数')


# Training / test parameters
learn_arg = add_argument_group('Learning')
#### optim ####
learn_arg.add_argument('--optim', type=str, default='Adam') 
learn_arg.add_argument('--lr', type=float, default=(3e-4)) # 0.0003
learn_arg.add_argument('--wd', type=float, default=(1e-4), help='weight decay')
learn_arg.add_argument('--beta1', type=float, default=0.9, help='Adam-beta1')
learn_arg.add_argument('--beta2', type=float, default=0.999, help='Adam-beta2')
learn_arg.add_argument('--eps', type=float, default=1e-08)
learn_arg.add_argument('--flood', type=bool, default=False)
#### schedule ####
learn_arg.add_argument('--schedule', type=str, default='cos_lr', help='step/cos_lr/Tmax/Tmin') 
learn_arg.add_argument('--lr_decay', type=int, default=400)
learn_arg.add_argument('--gamma', type=float, default='0.5', help='下降速度')
#### epoch/bs ####
learn_arg.add_argument('--batch_size', type=int, default=6)
learn_arg.add_argument('--one_batch_n_sample', type=int, default=1, help='smapling n times of each volume')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=1500)
learn_arg.add_argument('--warmup_epoch', type=float, default=0.05, help='warm up epoch ratio')


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--ckpt_dir', type=str, default='save_name',help='saved filename')
misc_arg.add_argument('--gpu_id', type=str, default='0')
misc_arg.add_argument('--num_workers', type=int, default=2)
misc_arg.add_argument('--parallel', type=bool, default=True, help="parallel training")
misc_arg.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
misc_arg.add_argument("--amp", default=True, type=bool, help='autocast')

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if len(args.gpu_id.split(',')) > 0:   
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    
    args.hr_slice_patch = args.upscale * (args.lr_slice_patch - 1) + 1
    return args, unparsed

