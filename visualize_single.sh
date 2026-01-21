#!/bin/bash
# 单张图像分割结果可视化
CUDA_VISIBLE_DEVICES="0" \
python visualize_single_image.py \
--target_img_name "malignant (53).png" \
--test_image_path "/root/autodl-tmp/busi/images" \
--test_gt_path "/root/autodl-tmp/busi/masks" \
--predict_results_path "/root/autodl-tmp/busi/predict_results4" \
--save_vis_path "/root/SAM2-UNet-main/output4_checkpoints"