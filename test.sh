CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "xxxx" \
--test_image_path "/root/autodl-tmp/busi/images" \
--test_gt_path "/root/autodl-tmp/busi/masks" \
--save_path "/root/SAM2-UNet-main/test_busi_small"