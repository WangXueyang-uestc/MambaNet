#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 .pt 体数据渲染为 PNG 图片保存。

兼容两种保存方式：
- pickle.dump({'image': np.ndarray, 'spacing': tuple}) 保存的 .pt（本仓库 data_processed 下的 liver_*.pt 即为此格式）
- torch.save(...) 保存的 .pt（若本机安装了 PyTorch，则也会尝试 torch.load）

功能特性：
- 支持输入单个文件或目录（自动批量处理 .pt）
- 支持切片模式：middle / all / indices（逗号分隔）/ mip（最大强度投影）
- 支持选择切片轴：0/1/2（默认 2，即沿最后一维切片）
- 支持窗口化（--window --level）或百分位拉伸（--pmin --pmax）
- 自动归一化为 0-255 并以灰度保存为 PNG

示例：
1) 渲染单个文件中间切片（默认轴=2）：
   python pt_to_png.py -i /Data2/.../data_processed/liver_3.pt -o ./vis --mode middle

2) 批量渲染整个目录，所有切片，沿轴 2：
   python pt_to_png.py -i /Data2/.../data_processed -o ./vis --mode all --axis 2

3) 指定切片索引：
   python pt_to_png.py -i /Data2/.../data_processed/liver_3.pt -o ./vis --mode indices --indices 10,20,30

4) 最大强度投影：
   python pt_to_png.py -i /Data2/.../data_processed/liver_3.pt -o ./vis --mode mip --axis 2

5) 使用窗口化（例如 CT 常用 WL=40, WW=400）：
   python pt_to_png.py -i /Data2/.../data_processed/liver_3.pt -o ./vis --mode middle --window 400 --level 40

6) 使用百分位拉伸（去除极端值）：
   python pt_to_png.py -i /Data2/.../data_processed/liver_3.pt -o ./vis --mode middle --pmin 1 --pmax 99
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def try_load_pt(path: str) -> Dict[str, Any]:
    """加载 .pt 文件，先尝试 pickle，再尝试 torch.load（若可用）。

    返回一个 dict，至少包含 'image'（3D numpy.ndarray）。可能包含 'spacing'（tuple）。
    """
    # 先尝试 pickle
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and 'image' in obj:
            return obj
    except Exception:
        pass

    # 再尝试 torch.load（若安装了 torch）
    try:
        import torch  # type: ignore

        obj = torch.load(path, map_location='cpu')
        # 可能是 Tensor、dict 或其他结构
        if isinstance(obj, dict) and 'image' in obj:
            data = obj
        elif hasattr(obj, 'numpy'):
            data = {'image': obj.numpy()}
        else:
            # 尝试常见键
            for key in ('data', 'volume', 'img', 'image'):
                if key in obj:
                    val = obj[key]
                    if hasattr(val, 'numpy'):
                        val = val.numpy()
                    data = {'image': val}
                    break
            else:
                raise ValueError("Unsupported .pt content structure for torch.load")

        return data  # type: ignore[return-value]
    except Exception as e:
        raise RuntimeError(f"无法加载 {path}，既不是pickle可读，也不是torch可读: {e}")


def apply_windowing(
    vol: np.ndarray,
    window: Optional[float] = None,
    level: Optional[float] = None,
    pmin: Optional[float] = None,
    pmax: Optional[float] = None,
) -> np.ndarray:
    """对体数据进行窗口化或百分位拉伸，并归一化到 [0, 255] uint8。

    规则：
    - 若提供 window 和 level，则优先使用 WL/WW：
        vmin = level - window/2, vmax = level + window/2
    - 否则若提供 pmin/pmax（百分位），使用对应分位数作为最小/最大
    - 否则使用整个体数据的最小/最大
    """
    vol = vol.astype(np.float32, copy=False)

    if window is not None and level is not None:
        vmin = level - window / 2.0
        vmax = level + window / 2.0
    elif pmin is not None or pmax is not None:
        lo = 0.0 if pmin is None else float(pmin)
        hi = 100.0 if pmax is None else float(pmax)
        lo = max(0.0, min(100.0, lo))
        hi = max(0.0, min(100.0, hi))
        if hi <= lo:
            hi = lo + 1.0
        vmin = np.percentile(vol, lo)
        vmax = np.percentile(vol, hi)
    else:
        vmin = float(np.min(vol))
        vmax = float(np.max(vol))

    if vmax <= vmin:
        vmax = vmin + 1.0

    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin) * 255.0
    vol = np.clip(vol, 0, 255).astype(np.uint8)
    return vol


def save_slice_png(arr2d: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr2d, mode='L').save(out_path)


def ensure_3d(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"期望3D体数据，实际维度: {image.shape}")
    return image


def slice_indices_for_mode(mode: str, axis: int, shape: Tuple[int, int, int], indices: Optional[Sequence[int]]) -> Sequence[int]:
    d = shape[axis]
    if mode == 'middle':
        return [d // 2]
    elif mode == 'all':
        return list(range(d))
    elif mode == 'indices':
        if not indices:
            raise ValueError("--mode indices 需要使用 --indices 指定切片索引，例如: --indices 10,20,30")
        # 过滤合法范围
        return [i for i in indices if 0 <= i < d]
    else:
        raise ValueError(f"未知模式: {mode}")


def process_file(
    in_path: str,
    out_dir: str,
    axis: int = 2,
    mode: str = 'middle',
    indices: Optional[Sequence[int]] = None,
    window: Optional[float] = None,
    level: Optional[float] = None,
    pmin: Optional[float] = None,
    pmax: Optional[float] = None,
    use_subdir: bool = True,
) -> None:
    data = try_load_pt(in_path)
    image = data.get('image')
    if isinstance(image, np.ndarray) is False:
        # 兼容 torch.Tensor 或 list
        if hasattr(image, 'numpy'):
            image = image.numpy()
        else:
            image = np.asarray(image)

    image = ensure_3d(image)  # (H, W, S) 或 (Z, Y, X)，按 axis 切片

    # 对整个体数据进行一次窗口化/拉伸，避免每片重复计算分位数
    norm_vol = apply_windowing(image, window=window, level=level, pmin=pmin, pmax=pmax)

    base = os.path.splitext(os.path.basename(in_path))[0]
    if base.endswith('.nii'):
        base = os.path.splitext(base)[0]

    save_root = os.path.join(out_dir, base) if use_subdir else out_dir
    os.makedirs(save_root, exist_ok=True)

    if mode == 'mip':
        # 最大强度投影
        mip = np.max(norm_vol, axis=axis)
        out_path = os.path.join(save_root, f"{base}_mip_axis{axis}.png")
        save_slice_png(mip, out_path)
        print(f"保存: {out_path}")
        return

    # 普通切片
    idx_list = slice_indices_for_mode(mode, axis, norm_vol.shape, indices)

    for i in idx_list:
        if axis == 0:
            sl = norm_vol[i, :, :]
        elif axis == 1:
            sl = norm_vol[:, i, :]
        else:
            sl = norm_vol[:, :, i]

        out_path = os.path.join(save_root, f"{base}_axis{axis}_slice{i:04d}.png")
        save_slice_png(sl, out_path)
        # 降低输出频率：全量时每隔若干打印一次
        if mode != 'all' or (mode == 'all' and i % 20 == 0):
            print(f"保存: {out_path}")


def iter_pt_files(input_path: str) -> Sequence[str]:
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for name in sorted(os.listdir(input_path)):
        if name.startswith('._'):
            continue
        if name.lower().endswith('.pt'):
            files.append(os.path.join(input_path, name))
    return files


def parse_indices(indices_str: Optional[str]) -> Optional[Sequence[int]]:
    if not indices_str:
        return None
    vals = []
    for tok in indices_str.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            raise ValueError(f"非法切片索引: {tok}")
    return vals


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将 .pt 体数据渲染为 PNG 图片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input', required=True, help='输入文件或目录（.pt）')
    p.add_argument('-o', '--output', required=True, help='输出目录')
    p.add_argument('--axis', type=int, default=2, choices=[0, 1, 2], help='切片轴（沿该轴取索引）')
    p.add_argument('--mode', type=str, default='middle', choices=['middle', 'all', 'indices', 'mip'], help='渲染模式')
    p.add_argument('--indices', type=str, default=None, help='当 mode=indices 时，逗号分隔的切片序号，如 10,20,30')
    p.add_argument('--window', type=float, default=None, help='窗口宽度（WW）')
    p.add_argument('--level', type=float, default=None, help='窗口中心（WL）')
    p.add_argument('--pmin', type=float, default=None, help='归一化下百分位（0-100）')
    p.add_argument('--pmax', type=float, default=None, help='归一化上百分位（0-100）')
    p.add_argument('--no-subdir', action='store_true', help='不为每个输入文件创建子目录，全部输出到 --output 根目录')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    in_path = args.input
    out_dir = args.output
    axis = int(args.axis)
    mode = str(args.mode)
    indices = parse_indices(args.indices)
    window = args.window
    level = args.level
    pmin = args.pmin
    pmax = args.pmax
    use_subdir = not args.no_subdir

    os.makedirs(out_dir, exist_ok=True)

    files = iter_pt_files(in_path)
    if not files:
        print(f"未找到待处理的 .pt 文件: {in_path}")
        return 2

    print(f"共 {len(files)} 个文件，模式={mode}，轴={axis}")
    for k, f in enumerate(files, 1):
        try:
            print(f"[{k}/{len(files)}] 处理 {os.path.basename(f)} ...")
            process_file(
                f,
                out_dir,
                axis=axis,
                mode=mode,
                indices=indices,
                window=window,
                level=level,
                pmin=pmin,
                pmax=pmax,
                use_subdir=use_subdir,
            )
        except Exception as e:
            print(f"❌ 处理 {f} 出错: {e}")

    print("完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())
