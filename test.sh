CUDA_VISIBLE_DEVICES="0" \
python test.py \
--test_image_path "/root/autodl-fs/New_data/test/images/" \
--test_gt_path "/root/autodl-fs/New_data/test/masks/" \
--checkpoint "/root/SAM2-UNet-main/output5_checkpoints/best_model.pth" \
--save_path "/root/SAM2-UNet-main/output5_checkpoints/test_results_visual"