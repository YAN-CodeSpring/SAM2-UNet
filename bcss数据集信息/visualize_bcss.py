#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

def visualize_samples(img_dir, mask_dir, output_path):
    # 检查路径
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print("❌ 错误：原图或掩码目录不存在。请检查路径。")
        return

    # 获取目录下所有图片名称（假设原图和掩码名字完全一致）
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    if len(img_files) < 5:
        print(f"⚠️ 警告：目录 {img_dir} 下的图片少于 5 张，无法生成 2x5 的网格。")
        return

    # 随机抽取 5 张图片
    sampled_files = random.sample(img_files, 5)
    print(f"🎲 成功随机抽取 5 张图片，开始生成可视化图像...")

    # 创建 2行 5列 的大图，设置总尺寸
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # 定义配色方案
    # 0: 透明 (背景), 1: 红色 (肿瘤), 2: 绿色 (基质)
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.6), (0, 1, 0, 0.6)] 
    cmap_2class = ListedColormap(colors[:2]) # 用于 2 分类
    cmap_3class = ListedColormap(colors[:3]) # 用于 3 分类

    for i, img_name in enumerate(sampled_files):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        if not os.path.exists(mask_path):
            print(f"⚠️ 警告：找不到对应的掩码文件 {img_name}")
            continue

        # 读取原图和掩码
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L"))

        # 动态判断是 2 类还是 3 类
        max_class = np.max(mask)
        is_3class = (max_class == 2)
        current_cmap = cmap_3class if is_3class else cmap_2class

        # --- 第一行：原图 ---
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original Image {i+1}", fontsize=14, fontweight='bold')
        axes[0, i].axis('off')

        # --- 第二行：原图叠加掩码 ---
        axes[1, i].imshow(img)
        # vmax 设置为 cmap 的最大索引，确保颜色映射准确
        axes[1, i].imshow(mask, cmap=current_cmap, vmin=0, vmax=2 if is_3class else 1, interpolation='nearest')
        
        # 根据类别生成标题
        class_str = "2-Class (Tumor=Red)" if not is_3class else "3-Class (Tumor=Red, Stroma=Green)"
        axes[1, i].set_title(f"Overlay: {class_str}", fontsize=12)
        axes[1, i].axis('off')

    # 调整布局以减少白边
    plt.tight_layout()

    # 保存大图
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 可视化完成！大图已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BCSS Images and Masks")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the original images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the masks")
    parser.add_argument("--output_path", type=str, required=True, help="Full path to save the output image (e.g., /path/viz.png)")
    
    args = parser.parse_args()
    
    visualize_samples(args.img_dir, args.mask_dir, args.output_path)