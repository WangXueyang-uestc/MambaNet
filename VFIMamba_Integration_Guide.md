# VFIMamba 集成到 I3Net 使用说明

## 概述
VFIMamba 是一个基于 Mamba 架构的视频帧插值模型，现已集成到 I3Net 框架中用于医学图像的切片插值任务。

## 文件结构
```
I3Net/
├── model_zoo/
│   └── vfimamba/           # VFIMamba 模型文件
│       ├── __init__.py
│       ├── feature_extractor.py
│       ├── flow_estimation.py
│       ├── warplayer.py
│       ├── refine.py
│       ├── loss.py
│       └── vfimamba_wrapper.py  # I3Net 适配层
├── opt/
│   └── vfimamba.json       # VFIMamba 配置文件
├── scripts/
│   ├── train_vfimamba.sh   # 训练脚本
│   └── test_vfimamba.sh    # 测试脚本
└── select_model.py         # 已更新以支持 VFIMamba
```

## 使用步骤

### 1. 准备预训练模型（如果有）
如果你已经训练了一个 VFIMamba 的 x5 模型，请将 checkpoint 文件（.pkl 或 .pth）放到合适的位置，例如：
```bash
mkdir -p /home/user/XueYangWang/I3Net/experiments/vfimamba/checkpoints
cp your_vfimamba_model.pkl /home/user/XueYangWang/I3Net/experiments/vfimamba/checkpoints/
```

### 2. 训练 VFIMamba 模型
编辑 `scripts/train_vfimamba.sh`，根据需要调整参数：
- `--vfimamba_ckpt`: 预训练模型路径（可选）
- `--hr_data_path`: 高分辨率训练数据路径
- `--lr_data_path`: 低分辨率训练数据路径
- `--upscale`: 插值倍数（你的是 5）

然后运行：
```bash
bash scripts/train_vfimamba.sh
```

### 3. 测试 VFIMamba 模型
编辑 `scripts/test_vfimamba.sh`，设置：
- `--vfimamba_ckpt`: 训练好的模型路径（必须）
- `--test_hr_data_path`: 高分辨率测试数据路径
- `--test_lr_data_path`: 低分辨率测试数据路径

然后运行：
```bash
bash scripts/test_vfimamba.sh
```

## 模型参数说明

### 基本参数
- `--model vfimamba`: 选择 VFIMamba 模型
- `--upscale 5`: 插值倍数（你训练的是 x5）
- `--lr_slice_patch 4`: 输入低分辨率切片数量

### VFIMamba 特定参数
- `--vfimamba_ckpt`: 预训练权重路径
- `--vfimamba_F 16`: 特征通道基数（默认 16）
- `--vfimamba_depth '[2,2,2,3,3]'`: 网络深度配置
- `--vfimamba_M False`: M 参数（通常为 False）
- `--vfimamba_local 2`: 局部 refinement 参数

## 模型架构说明

### VFIMambaInterpolator
- **输入**: `[B, H, W, S]` - S 个低分辨率切片
- **输出**: `[B, H, W, S_out]` - 其中 `S_out = (S-1)*upscale + 1`

### 工作流程
1. 保留原始关键帧切片
2. 在每对相邻切片之间使用 VFIMamba 进行插值
3. 插值时将灰度图像转换为 RGB（复制 3 通道）
4. 使用不同的 timestep 生成中间帧
5. 将 RGB 结果转回灰度图

## 注意事项

1. **数据格式**: VFIMamba 原本设计用于视频帧插值（RGB 图像），现在适配到医学图像（灰度图），通过通道复制实现兼容。

2. **内存占用**: VFIMamba 相比 I3Net 可能需要更多显存，建议：
   - 减小 `batch_size`
   - 减小 `lr_slice_patch`
   - 使用梯度累积

3. **训练时长**: VFIMamba 包含注意力机制和光流估计，训练可能较慢。

4. **Checkpoint 格式**: 支持两种格式：
   - 直接的 state_dict
   - 包含 'state_dict' key 的字典

## 示例命令

### 从头训练
```bash
python main.py \
    --model vfimamba \
    --upscale 5 \
    --lr_slice_patch 4 \
    --batch_size 2 \
    --max_epoch 500 \
    --train_mode paired \
    --ckpt_dir VFIMamba_x5_exp1
```

### 从预训练模型继续训练
```bash
python main.py \
    --model vfimamba \
    --upscale 5 \
    --resume True \
    --vfimamba_ckpt experiments/vfimamba/checkpoints/model.pkl \
    --ckpt_dir VFIMamba_x5_exp2
```

### 测试
```bash
python test.py \
    --model vfimamba \
    --upscale 5 \
    --vfimamba_ckpt experiments/vfimamba/VFIMamba_x5_exp1/model_best.pth \
    --ckpt_dir VFIMamba_x5_test
```

## 常见问题

**Q: 如何找到我训练好的 VFIMamba 模型？**
A: 通常在 `/home/user/XueYangWang/VFIMamba/ckpt/` 目录下，或者查找 `.pkl` 或 `.pth` 文件。

**Q: 可以和 RefinedI3Net 一起使用吗？**
A: VFIMamba 是独立的模型，不需要与 RefinedI3Net 结合。选择其中一个使用即可。

**Q: 为什么训练很慢？**
A: VFIMamba 使用了复杂的光流估计和注意力机制，建议使用更强的 GPU 或减小 batch size。

## 后续优化建议

1. **混合精度训练**: 启用 `--amp True` 加速训练
2. **数据增强**: 在 `data.py` 中添加更多数据增强策略
3. **学习率调度**: 尝试不同的学习率策略
4. **模型集成**: 可以尝试将 VFIMamba 和 I3Net 的优势结合
