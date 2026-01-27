import cv2
import numpy as np

# 替换成你的文件路径
mask_path = "/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0_0_2016_size224.png"
pred_path = "/root/autodl-tmp/BCSS/output5_predict_results/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0_0_2016_size224.png"

# 读取并二值化
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 0).astype(np.uint8) * 255
pred = (pred > 0).astype(np.uint8) * 255

# 拼接可视化（mask左，pred右）
vis = np.hstack([mask, pred])
# 保存到本地，下载查看
cv2.imwrite("vis_mask_pred.png", vis)
print(f"可视化图已保存：vis_mask_pred.png")
print(f"mask非零像素：{np.count_nonzero(mask)}，pred非零像素：{np.count_nonzero(pred)}")