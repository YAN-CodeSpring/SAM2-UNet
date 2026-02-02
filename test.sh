CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/root/SAM2-UNet-main/output5_checkpoints/best_model.pth" \
--test_image_path "/root/autodl-fs/QaTa-COV19-v2/Test Set/Images" \
--test_gt_path "/root/autodl-fs/QaTa-COV19-v2/Test Set/Ground-truths" \
--save_path "/root/autodl-tmp/COVID/output5_predict_results/" \