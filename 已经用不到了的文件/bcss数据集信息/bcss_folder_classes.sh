#!/bin/bash

# 你的验证集掩码路径
MASK_PATH="/root/autodl-tmp/BCSS/BCSS_512/val_mask_512"

# 运行 Python 脚本
python bcss_folder_classes.py --mask_path "$MASK_PATH"