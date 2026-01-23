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

def analyze_masks(mask_path, output_txt):
    # 检查路径有效性
    if not os.path.exists(mask_path):
        print(f"❌ 错误：掩码路径不存在 -> {mask_path}")
        return
    
    # 获取所有的 PNG 掩码图片并排序
    mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
    mask_files.sort()
    
    if not mask_files:
        print(f"⚠️ 警告：在 {mask_path} 下没有找到 PNG 文件。")
        return

    print(f"✅ 找到 {len(mask_files)} 张掩码图片，开始统计类别分布...")

    # 打开输出文件进行写入
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("BCSS 验证集掩码类别统计 (Val Mask Class Stats)\n")
        f.write("=" * 60 + "\n\n")

        # 使用 tqdm 遍历所有图片并显示进度
        for img_name in tqdm(mask_files, desc="Processing Masks"):
            img_path = os.path.join(mask_path, img_name)
            
            # 使用 PIL 读取图片并转换为灰度数组
            # 这里的类别索引正好对应像素值 (0-21)
            mask_img = Image.open(img_path).convert("L")
            mask_arr = np.array(mask_img)
            
            # 获取当前图片中出现的所有唯一像素值（即类别索引）
            unique_classes = np.unique(mask_arr)
            
            # 格式化输出字符串
            class_info = []
            for cls_idx in unique_classes:
                cls_name = BCSS_CLASSES.get(cls_idx, f"Unknown_Class_{cls_idx}")
                class_info.append(f"{cls_idx}: {cls_name}")
            
            # 写入结果
            f.write(f"图片名称: {img_name}\n")
            f.write(f"包含类别: {',  '.join(class_info)}\n")
            f.write("-" * 40 + "\n")

    print(f"🎉 统计完成！结果已成功保存至: {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCSS Mask Class Statistics")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the validation mask directory")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to save the output text file")
    
    args = parser.parse_args()
    
    analyze_masks(args.mask_path, args.output_txt)