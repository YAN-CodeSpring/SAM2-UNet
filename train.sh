CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/root/SAM2-UNet-main/sam2_hiera_large.pt" \
--train_image_path "/root/autodl-tmp/busi/images" \
--train_mask_path "/root/autodl-tmp/busi/masks" \
--save_path "/root/SAM2-UNet-main/output4_checkpoints" \
--log_path "/root/SAM2-UNet-main/output4_checkpoints/train_log.csv" \
--train_ratio 0.8 \
--epoch 25 \
--lr 0.001 \
--batch_size 12
