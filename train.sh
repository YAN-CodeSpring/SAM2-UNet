#!/bin/bash

# 请先确保运行过 wandb login

HIERA_PATH="/root/SAM2-UNet-main/sam2_hiera_large.pt"

# 训练集路径
TRAIN_IMG="/root/autodl-fs/QaTa-COV19-v2/Train Set/Images"
TRAIN_MASK="/root/autodl-fs/QaTa-COV19-v2/Train Set/Ground-truths"

# 验证集路径
VAL_IMG="/root/autodl-fs/QaTa-COV19-v2/Test Set/Images"
VAL_MASK="/root/autodl-fs/QaTa-COV19-v2/Test Set/Ground-truths"

SAVE_PATH="/root/SAM2-UNet-main/output1_checkpoints"
# [恢复] CSV 日志保存路径
LOG_PATH="/root/SAM2-UNet-main/output1_checkpoints/train_log.csv"

CUDA_VISIBLE_DEVICES="0" \
python train.py \
    --hiera_path "$HIERA_PATH" \
    --train_image_path "$TRAIN_IMG" \
    --train_mask_path "$TRAIN_MASK" \
    --val_image_path "$VAL_IMG" \
    --val_mask_path "$VAL_MASK" \
    --save_path "$SAVE_PATH" \
    --log_path "$LOG_PATH" \
    --epoch 25 \
    --lr 0.0001 \
    --batch_size 64 \
    --wandb_project "Covid-19"