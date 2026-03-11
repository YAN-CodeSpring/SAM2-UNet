CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/root/SAM2-UNet-main/output1_checkpoints/best_model.pth" \
--test_image_path "/root/autodl-fs/New_data/test/images/" \
--test_gt_path "/root/autodl-fs/New_data/test/masks/" \
--save_path "/root/SAM2-UNet-main/output1_checkpoints/test_results_visual"