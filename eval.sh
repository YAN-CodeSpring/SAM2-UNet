python eval.py \
--dataset_name "bcss_224_output5" \
--pred_path "/root/autodl-tmp/BCSS/output5_predict_results/" \
--gt_path "/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary"


ls /root/autodl-tmp/BCSS/output5_predict_results/*.{png,jpg,jpeg,bmp} | head -10