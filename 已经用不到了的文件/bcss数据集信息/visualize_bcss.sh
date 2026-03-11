#!/bin/bash

echo "🎨 启动 BCSS 数据集随机可视化 (支持2类/3类自适应)..."

# ==========================================
# 在这里修改你的原图路径、掩码路径和输出路径
# ==========================================

# 1. 原图所在的文件夹
IMG_DIR="/root/autodl-tmp/BCSS/BCSS_224/val"

# 2. 对应的掩码所在的文件夹（可以是原版3类的，也可以是你刚才生成的binary的）
MASK_DIR="/root/autodl-tmp/BCSS/BCSS_224/val_mask"

# 3. 输出大图的完整保存路径（请精确到 .png 或 .jpg）
OUTPUT_PATH="/root/SAM2-UNet-main/random_3_samples_visualization.png"

# 运行 Python 脚本
python visualize_bcss.py \
    --img_dir "$IMG_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_path "$OUTPUT_PATH"