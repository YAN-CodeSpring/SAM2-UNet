#!/bin/bash

# 打印提示信息
echo "🚀 启动 BCSS 掩码二值化合并处理..."

# ==========================================
# 在这里修改你的输入和输出路径
# ==========================================
# 示例：处理验证集
# INPUT_DIR="/root/autodl-tmp/BCSS/BCSS_224/val_mask"
# OUTPUT_DIR="/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary"

# 示例：如果要处理训练集，只需取消下面两行的注释并修改路径
INPUT_DIR="/root/autodl-tmp/BCSS/BCSS_224/train_mask"
OUTPUT_DIR="/root/autodl-tmp/BCSS/BCSS_224/train_mask_binary"

# 运行 Python 脚本
python bcss_merge_classes.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "🎉 任务结束。"