#!/bin/bash
# 可视化训练曲线
python plot_curves.py \
--log_path "/root/SAM2-UNet-main/output_bcss_512_checkpoints1/train_log.csv" \
--save_path "/root/SAM2-UNet-main/output_bcss_512_checkpoints1/training_curves.png"