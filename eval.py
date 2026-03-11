import os
import cv2
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(conf_matrix):
    """
    根据混淆矩阵计算每一类的 Dice, IoU, Precision, Recall
    """
    # conf_matrix: [n_classes, n_classes]
    # 行代表真值 (GT)，列代表预测 (Pred)
    n_classes = conf_matrix.shape[0]
    ious = []
    dices = []
    precisions = []
    recalls = []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-6)
        # Dice = 2 * TP / (2 * TP + FP + FN)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-6)
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-6)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)

    return ious, dices, precisions, recalls

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--pred_path", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
args = parser.parse_args()

# 三分类映射映射 (必须与 dataset.py 一致)
# 0: 背景, 1: 良性 (128), 2: 恶性 (255)
label_values = [0, 128, 255]
class_names = ["Background", "Benign", "Malignant"]

mask_name_list = sorted([f for f in os.listdir(args.gt_path) if f.endswith(('.png', '.jpg'))])

# 初始化全局混淆矩阵
total_conf_matrix = np.zeros((3, 3))

print(f"=======开始多分类评估: {args.dataset_name}=======")

for mask_name in mask_name_list:
    mask_path = os.path.join(args.gt_path, mask_name)
    # 自动匹配预测文件名 (假设 test.py 保存的文件名与 mask 一致)
    pred_path = os.path.join(args.pred_path, mask_name)
    
    if not os.path.exists(pred_path):
        continue

    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred_raw = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # 尺寸对齐
    if mask_raw.shape != pred_raw.shape:
        pred_raw = cv2.resize(pred_raw, (mask_raw.shape[1], mask_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 将像素值映射为类别索引 0, 1, 2
    # 逻辑：最接近哪个值就归为哪一类
    def quantize_labels(img):
        output = np.zeros_like(img)
        output[img < 64] = 0
        output[(img >= 64) & (img < 192)] = 1
        output[img >= 192] = 2
        return output

    mask_idx = quantize_labels(mask_raw).flatten()
    pred_idx = quantize_labels(pred_raw).flatten()

    # 计算单张图的混淆矩阵并累加
    cm = confusion_matrix(mask_idx, pred_idx, labels=[0, 1, 2])
    total_conf_matrix += cm

# 计算最终指标
ious, dices, precisions, recalls = calculate_metrics(total_conf_matrix)

print(f"\n{'='*20} 最终评估结果 (3-Class) {'='*20}")
print(f"{'Class':<15} | {'IoU':<10} | {'Dice':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 65)

for i in range(3):
    print(f"{class_names[i]:<15} | {ious[i]:.4f}     | {dices[i]:.4f}     | {precisions[i]:.4f}        | {recalls[i]:.4f}")

print("-" * 65)
# 通常医学影像中，我们更关心肿瘤类（1和2）的平均值
mIoU = np.mean(ious[1:]) 
mDice = np.mean(dices[1:])
print(f"Tumor Mean IoU (Benign+Malignant): {mIoU:.4f}")
print(f"Tumor Mean Dice (Benign+Malignant): {mDice:.4f}")
print(f"{'='*60}\n")