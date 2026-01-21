#!/bin/bash
CUDA_VISIBLE_DEVICES="0" \
python bcss_train.py \
--hiera_path "/root/SAM2-UNet-main/sam2_hiera_large.pt" \
--dataset_type "bcss" \
--size 224 \
--train_image_path "/root/autodl-tmp/BCSS/BCSS_224/train" \
--train_mask_path "/root/autodl-tmp/BCSS/BCSS_224/train_mask" \
--val_image_path "/root/autodl-tmp/BCSS/BCSS_224/val" \
--val_mask_path "/root/autodl-tmp/BCSS/BCSS_224/val_mask" \
--save_path "/root/SAM2-UNet-main/output_bcss_checkpoints1" \
--log_path "/root/SAM2-UNet-main/output_bcss_checkpoints1/train_log.csv" \
--epoch 25 \
--lr 0.0002 \
--batch_size 64 \
--weight_decay 5e-4