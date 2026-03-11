CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/root/SAM2-UNet-main/sam2_hiera_large.pt" \
--train_image_path "/root/autodl-fs/New_data/train/images/" \
--train_mask_path "/root/autodl-fs/New_data/train/masks/" \
--val_image_path "/root/autodl-fs/New_data/valid/images/" \
--val_mask_path "/root/autodl-fs/New_data/valid/masks/" \
--save_path "/root/SAM2-UNet-main/output2_checkpoints" \
--log_path "/root/SAM2-UNet-main/output2_checkpoints/train_log.csv" \
--epoch 20 \
--lr 0.001 \
--batch_size 12