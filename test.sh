CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/root/SAM2-UNet-main/output5_checkpoints/best_model.pth" \
--test_image_path "/root/autodl-tmp/BCSS/BCSS_224/val" \
--test_gt_path "/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary" \
--save_path "/root/autodl-tmp/BCSS/output5_predict_results/" \