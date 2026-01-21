#!/bin/bash
# 可视化训练曲线
python plot_curves.py \
--log_path "/root/SAM2-UNet-main/output4_checkpoints/train_log.csv" \
--save_path "/root/SAM2-UNet-main/output4_checkpoints/training_curves.png"