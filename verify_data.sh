#!/bin/bash

# ================= 配置区域 =================

# 请在这里修改你的实际路径
# 注意：路径如果包含空格，请用引号括起来
TRAIN_IMG_PATH="/root/autodl-fs/QaTa-COV19-v2/Train Set/Images"
TRAIN_MASK_PATH="/root/autodl-fs/QaTa-COV19-v2/Train Set/Ground-truths"

# 输出结果保存的txt文件名
OUTPUT_FILE="train_set_check_result.txt"

# ===========================================

echo "正在启动数据集校验程序..."
echo "Python脚本: verify_data.py"

# 调用 Python 脚本
python verify_data.py \
    --img_dir "$TRAIN_IMG_PATH" \
    --mask_dir "$TRAIN_MASK_PATH" \
    --output "$OUTPUT_FILE"

# 提示用户
echo "脚本运行结束。"
echo "请打开 $OUTPUT_FILE 查看详细结果。"