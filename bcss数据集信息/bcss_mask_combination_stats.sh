#!/bin/bash

# 打印提示信息
echo "🚀 开始运行 BCSS 掩码类别组合统计..."

# ==========================================
# 在这里修改你的输入文件夹和输出 TXT 路径
# ==========================================
MASK_DIR="/root/autodl-tmp/BCSS/BCSS_224/train_mask"
OUTPUT_FILE="/root/SAM2-UNet-main/mask_combination_stats_224.txt"

# 运行 Python 脚本
python bcss_mask_combination_stats.py \
    --mask_path "$MASK_DIR" \
    --output_txt "$OUTPUT_FILE"

echo "✅ 任务结束。"