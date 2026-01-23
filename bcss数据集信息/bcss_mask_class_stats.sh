#!/bin/bash

# 打印提示信息
echo "🚀 开始运行 BCSS 掩码类别统计..."

# 定义输入掩码路径和输出 TXT 路径
MASK_PATH="/root/autodl-tmp/BCSS/BCSS_512/val_mask_512"
OUTPUT_TXT="/root/SAM2-UNet-main/val_512_mask_class_stats.txt"

# 运行 Python 脚本
python bcss_mask_class_stats.py \
    --mask_path "$MASK_PATH" \
    --output_txt "$OUTPUT_TXT"

echo "✅ 任务结束。"