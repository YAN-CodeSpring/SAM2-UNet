import os
import cv2
import numpy as np
import argparse

# ========== 多分类指标计算核心函数 ==========
def calculate_dice(gt, pred, class_id):
    gt_class = (gt == class_id).astype(np.float32)
    pred_class = (pred == class_id).astype(np.float32)
    
    if np.sum(gt_class) == 0 and np.sum(pred_class) == 0:
        return 1.0
    elif np.sum(gt_class) == 0 or np.sum(pred_class) == 0:
        return 0.0
    
    intersection = np.sum(gt_class * pred_class)
    union = np.sum(gt_class) + np.sum(pred_class)
    dice = 2 * intersection / (union + 1e-8)
    return dice

def calculate_iou(gt, pred, class_id):
    gt_class = (gt == class_id).astype(np.float32)
    pred_class = (pred == class_id).astype(np.float32)
    
    if np.sum(gt_class) == 0 and np.sum(pred_class) == 0:
        return 1.0
    elif np.sum(gt_class) == 0 or np.sum(pred_class) == 0:
        return 0.0
    
    intersection = np.sum(gt_class * pred_class)
    union = np.sum(np.maximum(gt_class, pred_class))
    iou = intersection / (union + 1e-8)
    return iou

def calculate_mae(gt, pred):
    mae = np.mean(np.abs(gt - pred))
    return mae

# ========== 加载抽样列表 ==========
def load_sampled_list(sample_list_path):
    if not os.path.exists(sample_list_path):
        raise FileNotFoundError(f"抽样列表文件不存在：{sample_list_path}")
    with open(sample_list_path, 'r', encoding='utf-8') as f:
        sample_names = [line.strip() for line in f if line.strip()]
    return sample_names

# ========== 关键修改：自动匹配mask文件名（支持多种规则） ==========
def find_mask_path(gt_root, img_name):
    """
    自动匹配mask文件名，支持多种常见规则
    :param gt_root: mask文件夹路径
    :param img_name: 图像名（如xxx.png）
    :return: 找到的mask路径，None表示未找到
    """
    # 提取图像名前缀（去掉.png）
    img_prefix = img_name[:-4] if img_name.endswith('.png') else img_name
    
    # 定义常见的mask命名规则（按优先级排序）
    mask_candidates = [
        # 规则1：mask名和图像名完全一致
        os.path.join(gt_root, img_name),
        # 规则2：_mask后缀（xxx_mask.png）
        # os.path.join(gt_root, f"{img_prefix}_mask.png"),
        # 规则3：_label后缀（xxx_label.png）
        # os.path.join(gt_root, f"{img_prefix}_label.png"),
        # 规则4：mask前缀（mask_xxx.png）
        # os.path.join(gt_root, f"mask_{img_name}")
    ]
    
    # 遍历候选路径，返回第一个存在的
    for candidate in mask_candidates:
        if os.path.exists(candidate):
            return candidate
    return None

# ========== 主评估逻辑 ==========
def main(args):
    # 1. 初始化指标容器
    class_dice = {cid: [] for cid in range(22)}
    class_iou = {cid: [] for cid in range(22)}
    all_mae = []
    valid_samples = 0
    missing_samples = 0
    
    # 2. 加载抽样列表
    print(f"加载抽样列表：{args.sample_list_path}")
    sample_names = load_sampled_list(args.sample_list_path)
    print(f"共加载 {len(sample_names)} 个抽样样本")
    
    # 3. 遍历抽样样本计算指标
    for idx, img_name in enumerate(sample_names):
        print(f"[{idx+1}/{len(sample_names)}] 处理 {img_name}...")
        
        # 构建预测mask路径
        pred_prefix = img_name[:-4] if img_name.endswith('.png') else img_name
        pred_path = os.path.join(args.pred_path, f"{pred_prefix}_pred.png")
        
        # 自动查找真实mask路径（核心修改）
        gt_path = find_mask_path(args.gt_path, img_name)
        
        # 检查文件是否存在
        if gt_path is None:
            missing_samples += 1
            print(f"警告：未找到真实mask（尝试多种规则），跳过 {img_name}")
            continue
        if not os.path.exists(pred_path):
            missing_samples += 1
            print(f"警告：预测mask不存在 {pred_path}，跳过 {img_name}")
            continue
        
        # 读取mask
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        if gt is None or pred is None:
            missing_samples += 1
            print(f"警告：mask读取失败 {img_name}，跳过")
            continue
        
        # 统一尺寸
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 计算指标
        for cid in range(22):
            dice = calculate_dice(gt, pred, cid)
            iou = calculate_iou(gt, pred, cid)
            class_dice[cid].append(dice)
            class_iou[cid].append(iou)
        
        mae = calculate_mae(gt, pred)
        all_mae.append(mae)
        
        valid_samples += 1
    
    # 4. 容错：无有效样本时提示，而非直接报错
    if valid_samples == 0:
        print(f"\n错误：无有效样本参与评估！共缺失 {missing_samples} 个样本的mask/预测文件")
        print("请检查：")
        print("1. mask文件夹路径是否正确")
        print("2. mask文件名是否符合常见规则（如xxx.png/xxx_mask.png/xxx_label.png）")
        print("3. 预测mask是否生成成功（xxx_pred.png）")
        return  # 退出，不报错
    
    # 5. 计算最终指标
    mDice = np.mean([np.mean(class_dice[cid]) for cid in range(22)])
    mIoU = np.mean([np.mean(class_iou[cid]) for cid in range(22)])
    avg_MAE = np.mean(all_mae)
    
    # 6. 打印结果
    print("\n" + "="*60)
    print(f"BCSS多分类评估结果（有效样本：{valid_samples} / 缺失样本：{missing_samples}）")
    print("="*60)
    print(f"mDice (平均Dice):       {mDice:.3f}")
    print(f"mIoU (平均IoU):        {mIoU:.3f}")
    print(f"MAE (平均绝对误差):     {avg_MAE:.3f}")
    print("="*60)
    
    # 可选：打印每类指标
    if args.print_class_metrics:
        print("\n" + "="*60)
        print("每类详细指标（Dice/IoU）")
        print("="*60)
        for cid in range(22):
            dice = np.mean(class_dice[cid])
            iou = np.mean(class_iou[cid])
            print(f"类别 {cid:2d}: Dice={dice:.3f}, IoU={iou:.3f}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BCSS多分类语义分割评估脚本（适配任意mask命名）")
    # 核心参数
    parser.add_argument("--pred_path", type=str, required=True,
                        help="test阶段保存的预测mask路径（xxx_pred.png）")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="BCSS val集真实mask路径")
    parser.add_argument("--sample_list_path", type=str, required=True,
                        help="test阶段保存的抽样列表路径（sampled_val_list.txt）")
    # 可选参数
    parser.add_argument("--print_class_metrics", action="store_true",
                        help="是否打印每类的详细Dice/IoU（默认不打印）")
    
    args = parser.parse_args()
    main(args)