import os
import pickle
import numpy as np


def load_pt_volume(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = pickle.load(open(file_path, "rb"))

    # 兼容不同 key 命名：优先使用 'image'
    if isinstance(data, dict):
        if "image" in data:
            img = data["image"]
        else:
            # 如果不是字典或没有 image，就直接当成数组用
            img = np.array(data)
    else:
        img = np.array(data)

    img = np.asarray(img)

    min_val = float(img.min())
    max_val = float(img.max())
    mean_val = float(img.mean())

    print(f"文件: {file_path}")
    print(f"  形状: {img.shape}")
    print(f"  数据类型: {img.dtype}")
    print(f"  像素最小值: {min_val}")
    print(f"  像素最大值: {max_val}")
    print(f"  像素平均值: {mean_val}")


if __name__ == "__main__":
    pt_path = "/Data2/XueYangWang/Medical_SR/data_processed_volume/BRATS_001.pt"
    load_pt_volume(pt_path)


