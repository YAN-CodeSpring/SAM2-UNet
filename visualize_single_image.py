import os
import argparse
import warnings
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ========== 配置可视化样式 ==========
def setup_plot():
    # 关闭坐标轴、设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300

# ========== 加载并预处理图像/mask ==========
def load_image(path):
    """加载图像（RGB格式）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件不存在：{path}")
    img = Image.open(path).convert('RGB')
    return np.array(img)

def load_mask(path, target_size):
    """加载分割mask并调整尺寸（转单通道）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask文件不存在：{path}")
    mask = Image.open(path).convert('L')  # 转单通道
    mask = mask.resize(target_size, Image.Resampling.BICUBIC)  # 匹配原图尺寸
    mask = np.array(mask)
    # 二值化（确保mask是0/255）
    mask = (mask > 127).astype(np.uint8) * 255
    return mask

def mask_to_color(mask, color=(255, 0, 0), alpha=0.5):
    """单通道mask转彩色图（红色，透明度可调）"""
    # 创建彩色mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 255] = color
    # 转PIL图像（方便叠加）
    color_mask_pil = Image.fromarray(color_mask)
    return color_mask_pil, alpha

def overlay_mask_on_image(img, mask, color=(255, 0, 0), alpha=0.5):
    """将彩色mask叠加到原图上"""
    img_pil = Image.fromarray(img)
    mask_pil, alpha = mask_to_color(mask, color, alpha)
    # 叠加
    overlay = Image.blend(img_pil, mask_pil, alpha)
    return np.array(overlay)

# ========== 拼接并保存可视化图像 ==========
def visualize_single_sample(args):
    # 1. 初始化样式
    setup_plot()
    
    # 2. 解析目标图像名，匹配各文件路径
    img_name = args.target_img_name  # 如 "normal (27).png"
    img_base = os.path.splitext(img_name)[0]  # 提取文件名（无后缀）："normal (27)"
    
    # 原图路径
    img_path = os.path.join(args.test_image_path, img_name)
    # 真实mask路径（遍历mask子文件夹找对应_mask.png）
    gt_mask_path = None
    for cls_dir in os.listdir(args.test_gt_path):
        cls_path = os.path.join(args.test_gt_path, cls_dir)
        if os.path.isdir(cls_path):
            candidate = os.path.join(cls_path, f"{img_base}_mask.png")
            if os.path.exists(candidate):
                gt_mask_path = candidate
                break
    if gt_mask_path is None:
        raise FileNotFoundError(f"未找到{img_name}对应的真实mask（格式：{img_base}_mask.png）")
    
    # 预测mask路径（test.py生成的，通常命名为{img_base}_pred.png）##################################
    pred_mask_path = os.path.join(args.predict_results_path, f"{img_base}.png")
    if not os.path.exists(pred_mask_path):
        raise FileNotFoundError(f"未找到{img_name}对应的预测mask：{pred_mask_path}")
    
    # 3. 加载并预处理
    img = load_image(img_path)
    target_size = (img.shape[1], img.shape[0])  # (宽, 高)
    gt_mask = load_mask(gt_mask_path, target_size)
    pred_mask = load_mask(pred_mask_path, target_size)
    
    # 4. 生成可视化图
    # 真实mask彩色图（红色）
    gt_mask_color = mask_to_color(gt_mask)[0]
    # 预测+真值叠加图（预测=红色，真值=绿色，叠加到原图）
    img_overlay_pred = overlay_mask_on_image(img, pred_mask, color=(255, 0, 0), alpha=0.4)  # 预测红
    img_overlay_gt = overlay_mask_on_image(img_overlay_pred, gt_mask, color=(0, 255, 0), alpha=0.3)  # 真值绿
    
    # 5. 拼接三张图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SAM2-UNet Segmentation Result: {img_name}", fontsize=16, fontweight='bold')
    
    # 子图1：原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 子图2：真实分割图
    axes[1].imshow(gt_mask_color)
    axes[1].set_title("Ground Truth Mask", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 子图3：预测+真值叠加图
    axes[2].imshow(img_overlay_gt)
    axes[2].set_title("Pred Mask (Red) + GT Mask (Green)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # 6. 保存拼接图
    os.makedirs(args.save_vis_path, exist_ok=True)
    save_path = os.path.join(args.save_vis_path, f"{img_base}_vis.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ 可视化图像已保存至：{save_path}")
    print(f"📌 图例说明：红色=预测分割 | 绿色=真实分割 | 叠加=原图+双mask")

# ========== 命令行参数 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single Image Segmentation Visualization")
    # 必选参数
    parser.add_argument("--target_img_name", type=str, required=True,
                        help="目标图像名（如：normal (27).png）")
    parser.add_argument("--test_image_path", type=str, required=True,
                        help="测试原图目录（如：/root/autodl-tmp/busi/images）")
    parser.add_argument("--test_gt_path", type=str, required=True,
                        help="真实mask目录（如：/root/autodl-tmp/busi/masks）")
    parser.add_argument("--predict_results_path", type=str, required=True,
                        help="预测mask目录（test.py生成的，如：/root/autodl-tmp/busi/predict_results5）")
    parser.add_argument("--save_vis_path", type=str, required=True,
                        help="可视化结果保存目录（自定义，如：/root/autodl-tmp/busi/vis_results5）")
    
    args = parser.parse_args()
    
    # 执行可视化
    try:
        visualize_single_sample(args)
    except Exception as e:
        print(f"❌ 可视化失败：{str(e)}")
        exit(1)