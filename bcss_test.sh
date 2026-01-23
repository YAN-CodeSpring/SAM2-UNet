python bcss_test.py \
--checkpoint "/root/SAM2-UNet-main/output_bcss_checkpoints1/SAM2-UNet-25.pth" \
--val_image_path "/root/autodl-tmp/BCSS/BCSS_224/val" \
--val_mask_path "/root/autodl-tmp/BCSS/BCSS_224/val_mask" \
--save_path "/root/autodl-tmp/BCSS/BCSS_224/test1_pred" \
--sample_num 1000 \
--size 224 \
--seed 1024