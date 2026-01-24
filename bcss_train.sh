#!/bin/bash
CUDA_VISIBLE_DEVICES="0" \
python bcss_train.py \
--hiera_path "/root/SAM2-UNet-main/sam2_hiera_large.pt" \
--dataset_type "bcss" \
--size 512 \
--train_image_path "/root/autodl-tmp/BCSS/BCSS_512/train_512" \
--train_mask_path "/root/autodl-tmp/BCSS/BCSS_512/train_mask_512" \
--val_image_path "/root/autodl-tmp/BCSS/BCSS_512/val_512" \
--val_mask_path "/root/autodl-tmp/BCSS/BCSS_512/val_mask_512" \
--save_path "/root/SAM2-UNet-main/output_bcss_512_checkpoints1" \
--log_path "/root/SAM2-UNet-main/output_bcss_512_checkpoints1/train_log.csv" \
--epoch 25 \
--lr 0.0002 \
--batch_size 12 \
--weight_decay 5e-4