#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def analyze_mask_combinations(mask_path, output_txt):
    # 检查路径有效性
    if not os.path.exists(mask_path):
        print(f"❌ 错误：掩码路径不存在 -> {mask_path}")
        return
    
    # 获取所有的 PNG 掩码图片
    mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
    total_images = len(mask_files)
    
    if total_images == 0:
        print(f"⚠️ 警告：在 {mask_path} 下没有找到 PNG 文件。")
        return

    print(f"✅ 找到 {total_images} 张掩码图片，开始统计类别排列组合...")

    # 初始化所有可能组合的计数器
    # 键是包含类别的元组，值是图片计数
    combination_counts = {
        (0,): 0,
        (1,): 0,
        (2,): 0,
        (0, 1): 0,
        (0, 2): 0,
        (1, 2): 0,
        (0, 1, 2): 0,
        # 兜底项：以防出现全黑/空图，或其他意外类别
        "other": 0 
    }

    # 遍历所有图片进行统计
    for img_name in tqdm(mask_files, desc="Processing Masks"):
        img_path = os.path.join(mask_path, img_name)
        
        # 读取图片并获取唯一像素值
        mask_img = Image.open(img_path).convert("L")
        mask_arr = np.array(mask_img)
        unique_classes = np.unique(mask_arr)
        
        # 将当前图片的类别转换为标准的排序元组，以便匹配字典的键
        # 例如：[2, 0] 会被转换成 (0, 2)
        current_combination = tuple(sorted([int(c) for c in unique_classes if c in [0, 1, 2]]))
        
        # 更新计数器
        if current_combination in combination_counts:
            combination_counts[current_combination] += 1
        else:
            combination_counts["other"] += 1

    # --- 格式化输出结果 ---
    output_lines = [
        "BCSS 验证集掩码类别【组合】统计",
        "=" * 40,
        f"有且仅有[0]类的图片张数：{combination_counts[(0,)]}张",
        f"有且仅有[1]类的图片张数：{combination_counts[(1,)]}张",
        f"有且仅有[2]类的图片张数：{combination_counts[(2,)]}张",
        f"有且仅有[0,1]类的图片张数：{combination_counts[(0, 1)]}张",
        f"有且仅有[0,2]类的图片张数：{combination_counts[(0, 2)]}张",
        f"有且仅有[1,2]类的图片张数：{combination_counts[(1, 2)]}张",
        f"有且仅有[0,1,2]类的图片张数：{combination_counts[(0, 1, 2)]}张",
        "-" * 40,
        f"异常或空图片张数：{combination_counts['other']}张",
        f"总图片数量：{total_images}张",
        "=" * 40
    ]

    # 将结果打印到终端
    print("\n" + "\n".join(output_lines) + "\n")

    # 将结果写入 TXT 文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"🎉 统计完成！结果已成功保存至: {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCSS Mask Class Combination Statistics")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask directory")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to save the output text file")
    
    args = parser.parse_args()
    
    analyze_mask_combinations(args.mask_path, args.output_txt)