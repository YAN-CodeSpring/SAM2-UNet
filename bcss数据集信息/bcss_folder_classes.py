#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# BCSS 数据集 22 个类别的完整映射字典 (0-21)
BCSS_CLASSES = {
    0: "Outside_ROI (非感兴趣区域)",
    1: "Tumor (肿瘤)",
    2: "Stroma (基质)",
    3: "Lymphocytic_infiltrate (淋巴细胞浸润)",
    4: "Necrosis_or_Debris (坏死或碎片)",
    5: "Glandular_secretions (腺体分泌物)",
    6: "Blood (血液)",
    7: "Exclude (排除区域)",
    8: "Metaplasia_NOS (化生)",
    9: "Fat (脂肪)",
    10: "Plasma_cells (浆细胞)",
    11: "Other_immune_infiltrate (其他免疫细胞浸润)",
    12: "Mucoid_material (粘液性物质)",
    13: "Normal_acinus_or_duct (正常腺泡或导管)",
    14: "Lymphatics (淋巴管)",
    15: "Undetermined (未定区域)",
    16: "Nerve (神经)",
    17: "Skin_adnexa (皮肤附件)",
    18: "Blood_vessel (血管)",
    19: "Angioinvasion (血管浸润)",
    20: "DCIS (导管原位癌)",
    21: "Other (其他)"
}

def check_folder_classes(mask_path):
    if not os.path.exists(mask_path):
        print(f"❌ 错误：掩码路径不存在 -> {mask_path}")
        return

    mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
    
    if not mask_files:
        print(f"⚠️ 警告：在 {mask_path} 下没有找到 PNG 文件。")
        return

    print(f"🚀 开始扫描 {len(mask_files)} 张掩码图片...")

    # 使用 set 来存储全局唯一的类别值
    global_unique_classes = set()

    # 遍历所有图片
    for img_name in tqdm(mask_files, desc="Scanning Images"):
        img_path = os.path.join(mask_path, img_name)
        
        # 读取掩码并提取当前图片的唯一值
        mask_img = Image.open(img_path).convert("L")
        mask_arr = np.array(mask_img)
        unique_in_img = np.unique(mask_arr)
        
        # 将当前图片的类别更新到全局集合中
        global_unique_classes.update(unique_in_img)

    # 排序结果以便于阅读
    sorted_classes = sorted(list(global_unique_classes))

    # --- 打印最终汇总结果 ---
    print("\n" + "="*50)
    print(f"🎯 文件夹扫描结果汇总: {mask_path}")
    print("="*50)
    print(f"共发现 {len(sorted_classes)} 个不同的类别:")
    
    for cls_idx in sorted_classes:
        # 如果像素值超过21（比如255），提示可能是未正确处理的背景或无效值
        cls_name = BCSS_CLASSES.get(cls_idx, f"⚠️ 未知/异常类别 (可能需要检查数据)")
        print(f"  • [ID: {cls_idx:2d}] - {cls_name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan folder for all unique BCSS mask classes")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask directory")
    
    args = parser.parse_args()
    
    check_folder_classes(args.mask_path)