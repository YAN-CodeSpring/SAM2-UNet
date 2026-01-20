CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/root/SAM2-UNet-main/sam2_hiera_small.pt" \
--test_image_path "/root/autodl-tmp/busi/images" \
--test_gt_path "/root/autodl-tmp/busi/masks" \
--save_path "/root/SAM2-UNet-main/test_busi_small"