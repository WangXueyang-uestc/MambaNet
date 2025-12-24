import os
import shutil
import random
from typing import Dict, List, Tuple


HD_SRC_DIR = "/home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx"
LUNG_SRC_DIR = "/home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx"

HD_TRAIN_DIR = "/home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_train"
HD_TEST_DIR = "/home/user/XueYangWang/HX_data2/data_processed_volume_hd_hx_test"
LUNG_TRAIN_DIR = "/home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx_train"
LUNG_TEST_DIR = "/home/user/XueYangWang/HX_data2/data_processed_volume_lung_hx_test"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_pt_files(directory: str) -> List[str]:
    return [
        f for f in os.listdir(directory)
        if f.endswith(".pt") and os.path.isfile(os.path.join(directory, f))
    ]


def extract_patient_key(filename: str) -> str:
    """
    根据保存命名规则：{date}_{patient}_{subfolder}.pt
    其中 subfolder 拆成三个 token（例如：1.0 x 1.0, HD, 203）。
    我们取除去最后三个 token 的部分作为 key，以保证 HD / LUNG 成对。
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) <= 3:
        raise ValueError(f"文件命名无法提取 key: {filename}")
    # 去掉最后三个 token（子文件夹信息）
    key = "_".join(parts[:-3])
    if not key:
        raise ValueError(f"提取到空 key: {filename}")
    return key


def build_map(directory: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for fname in list_pt_files(directory):
        try:
            key = extract_patient_key(fname)
        except ValueError as exc:
            print(f"[WARN] {exc}, 跳过该文件")
            continue
        mapping.setdefault(key, []).append(os.path.join(directory, fname))
    return mapping


def split_pairs(pairs: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """返回 train_pairs, test_pairs"""
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    split_idx = min(max(split_idx, 1), len(pairs) - 1) if len(pairs) > 1 else len(pairs)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    return train_pairs, test_pairs


def copy_pair(pair_list: List[Tuple[str, str, str]], hd_dest: str, lung_dest: str):
    ensure_dir(hd_dest)
    ensure_dir(lung_dest)

    for key, hd_path, lung_path in pair_list:
        hd_fname = os.path.basename(hd_path)
        lung_fname = os.path.basename(lung_path)

        shutil.copy2(hd_path, os.path.join(hd_dest, hd_fname))
        shutil.copy2(lung_path, os.path.join(lung_dest, lung_fname))
        print(f"[COPY] {key} -> {hd_dest}/{hd_fname}, {lung_dest}/{lung_fname}")


def main():
    hd_map = build_map(HD_SRC_DIR)
    lung_map = build_map(LUNG_SRC_DIR)

    shared_keys = sorted(set(hd_map.keys()) & set(lung_map.keys()))
    if not shared_keys:
        print("没有找到 HD 与 LUNG 的匹配文件，退出。")
        return

    pairs: List[Tuple[str, str, str]] = []
    skipped_keys = 0
    for key in shared_keys:
        hd_files = hd_map[key]
        lung_files = lung_map[key]

        if len(hd_files) != 1 or len(lung_files) != 1:
            print(f"[WARN] {key} 对应 {len(hd_files)} 个 HD 或 {len(lung_files)} 个 LUNG，跳过该 key")
            skipped_keys += 1
            continue

        pairs.append((key, hd_files[0], lung_files[0]))

    if not pairs:
        print("匹配的 HD-LUNG 对为空，退出。")
        return

    train_pairs, test_pairs = split_pairs(pairs)

    print(f"总共匹配到 {len(pairs)} 对（跳过 {skipped_keys} 对），"
          f"训练集 {len(train_pairs)} 对，测试集 {len(test_pairs)} 对。")

    copy_pair(train_pairs, HD_TRAIN_DIR, LUNG_TRAIN_DIR)
    copy_pair(test_pairs, HD_TEST_DIR, LUNG_TEST_DIR)

    print("划分完成。")


if __name__ == "__main__":
    main()


