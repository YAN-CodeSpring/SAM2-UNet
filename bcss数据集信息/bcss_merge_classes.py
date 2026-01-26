#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def merge_and_save_masks(input_dir, output_dir):
    # 检查输入路径是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 错误：输入路径不存在 -> {input_dir}")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 PNG 掩码图片
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    total_images = len(mask_files)

    if total_images == 0:
        print(f"⚠️ 警告：在 {input_dir} 下没有找到 PNG 文件。")
        return

    print(f"🚀 开始转换：将 [0, 2] 合并为背景(0)，[1] 设为肿瘤(1)")
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")

    # 统计信息器
    success_count = 0

    # 使用 tqdm 显示进度条
    for img_name in tqdm(mask_files, desc="Merging & Saving"):
        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        try:
            # 1. 读取原始掩码为 numpy 数组
            mask_img = Image.open(img_path).convert("L")
            mask_arr = np.array(mask_img)

            # 2. 核心合并逻辑：只保留像素值为 1（肿瘤）的区域，其他全部归 0
            # 这样原本的 0 和 2 就自动变成 0 了
            binary_mask = np.where(mask_arr == 1, 1, 0).astype(np.uint8)

            # 3. 将新的二值数组转换回 PIL Image 并保存
            new_img = Image.fromarray(binary_mask, mode='L')
            new_img.save(out_path)
            success_count += 1
            
        except Exception as e:
            print(f"处理图片 {img_name} 时出错: {e}")

    print(f"✅ 转换完成！成功处理 {success_count}/{total_images} 张图片。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge BCSS Classes 0 and 2 into Background, 1 into Tumor")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original mask directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save merged masks")
    
    args = parser.parse_args()
    
    merge_and_save_masks(args.input_dir, args.output_dir)