import os
import cv2
import py_sod_metrics
import argparse

# ========== 适配旧版 py_sod_metrics（核心修改） ==========
# 旧版本无需V2，直接初始化（警告可忽略，不影响运行）
FM = py_sod_metrics.Fmeasure()  # 保留旧类，警告不影响结果
WFM = py_sod_metrics.WeightedFmeasure()  # 旧版本无参数，直接初始化
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()  # 旧版本无需传参
MAE = py_sod_metrics.MAE()
# 修复MSIoU：旧版本可能不需要参数，若仍报错可注释掉（你没用到其结果）
try:
    MSIOU = py_sod_metrics.MSIoU()  # 先尝试无参数初始化
except TypeError:
    MSIOU = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True)  # 兼容部分版本

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--pred_path", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--gt_path", type=str, required=True,
                    help="path to the ground truth masks")
args = parser.parse_args()

# ========== 保留原有FMv2配置（评估核心，不影响运行） ==========
sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
        "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
        "rec": py_sod_metrics.RecallHandler(**sample_gray),
        "fpr": py_sod_metrics.FPRHandler(**sample_gray),
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
        "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
        "ber": py_sod_metrics.BERHandler(**sample_gray),
        "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
        "kappa": py_sod_metrics.KappaHandler(**sample_gray),
        "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
        "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
        "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
        "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
        "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
        "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
        "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
        "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
        "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
        "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
        "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
        "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
        "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
        "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
        "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
        "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
        "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
        "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
        "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
        "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
        "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
        "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
    }
)

pred_root = args.pred_path
mask_root = args.gt_path

# ========== 保留适配mask子文件夹的逻辑（核心） ==========
mask_name_list = []
mask_path_list = []
# 遍历masks下的0/1/2子文件夹
for class_dir in os.listdir(mask_root):
    class_dir_path = os.path.join(mask_root, class_dir)
    if not os.path.isdir(class_dir_path):
        continue
    # 遍历子文件夹下的mask文件（_mask.png后缀）
    for mask_name in os.listdir(class_dir_path):
        if mask_name.endswith('_mask.png'):
            mask_name_list.append(mask_name)
            mask_path_list.append(os.path.join(class_dir_path, mask_name))

# 按mask名称排序（保证和预测结果一一对应）
sorted_pairs = sorted(zip(mask_name_list, mask_path_list))
mask_name_list, mask_path_list = zip(*sorted_pairs)
mask_name_list = list(mask_name_list)
mask_path_list = list(mask_path_list)

# 遍历计算指标
for i, (mask_name, mask_path) in enumerate(zip(mask_name_list, mask_path_list)):
    print(f"[{i}] Processing {mask_name}...")
    # 读取真实mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 读取预测mask（替换_mask后缀）
    pred_name = mask_name.replace('_mask.png', '.png')
    pred_path = os.path.join(pred_root, pred_name)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # 检查预测mask是否存在
    if pred is None:
        print(f"警告：未找到预测文件 {pred_path}，跳过该样本！")
        continue

    # ========== 用旧版API调用step（无V2） ==========
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)
    FMv2.step(pred=pred, gt=mask)
    # 可选：如果MSIoU仍报错，注释下面这行（你没用到其结果）
    # MSIOU.step(pred=pred, gt=mask)

# ========== 用旧版API获取结果 ==========
fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
fmv2 = FMv2.get_results()

# 整理评估结果（和原逻辑一致）
curr_results = {
    "meandice": fmv2["dice"]["dynamic"].mean(),
    "meaniou": fmv2["iou"]["dynamic"].mean(),
    'Smeasure': sm,
    "wFmeasure": wfm,
    "adpFm": fm["adp"],  # 旧版Fmeasure的adp指标
    "meanEm": em["curve"].mean(),
    "MAE": mae,
}

# 打印评估结果
print("\n" + "="*50)
print(f"数据集：{args.dataset_name}")
print("="*50)
print("mDice:       ", format(curr_results['meandice'], '.3f'))
print("mIoU:        ", format(curr_results['meaniou'], '.3f'))
print("S_{alpha}:   ", format(curr_results['Smeasure'], '.3f'))
print("F^{w}_{beta}:", format(curr_results['wFmeasure'], '.3f'))
print("F_{beta}:    ", format(curr_results['adpFm'], '.3f'))
print("E_{phi}:     ", format(curr_results['meanEm'], '.3f'))
print("MAE:         ", format(curr_results['MAE'], '.3f'))
print("="*50)