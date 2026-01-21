CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/root/SAM2-UNet-main/output4_checkpoints/SAM2-UNet-25.pth" \
--test_image_path "/root/autodl-tmp/busi/images" \
--test_gt_path "/root/autodl-tmp/busi/masks" \
--save_path "/root/autodl-tmp/busi/predict_results4" \