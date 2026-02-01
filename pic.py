import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 配置参数（可根据需要修改） --------------------------
# 文件夹路径（替换为你的实际路径）
img_path = "/root/autodl-tmp/BCSS/BCSS_224/val"    # 原图路径
gt_path = "/root/autodl-tmp/BCSS/BCSS_224/val_mask" # 真实掩码路径
pred_path = "/root/autodl-tmp/BCSS/output13_predict_results/" # 预测掩码路径

# 可视化配置
mask_color = (255, 0, 0)  # 掩码叠加时的颜色（红色）
alpha = 0.5               # 掩码叠加透明度（0-1，越大掩码越明显）
save_path = "segmentation_vis_9.png"  # 最终可视化结果保存路径
figsize = (20, 4)         # 拼接图尺寸（宽，高），适合PPT展示

# -------------------------- 核心函数 --------------------------
def get_matching_mask(img_name, mask_folder):
    """
    根据原图文件名，找到对应的掩码文件（假设文件名前缀一致，后缀为.png/.jpg等）
    """
    # 获取原图文件名（去掉后缀）
    img_prefix = os.path.splitext(img_name)[0]
    # 遍历掩码文件夹找匹配的文件
    for mask_file in os.listdir(mask_folder):
        if os.path.splitext(mask_file)[0] == img_prefix:
            return os.path.join(mask_folder, mask_file)
    raise FileNotFoundError(f"未找到{img_name}对应的掩码文件")

def overlay_mask_on_image(image, mask, color=mask_color, alpha=alpha):
    """
    将二值掩码叠加到原图上（掩码1的区域显示指定颜色，带透明度）
    """
    # 确保mask是二值图（0/1），转换为uint8
    mask = (mask > 0).astype(np.uint8)
    # 创建彩色掩码层
    mask_color = np.zeros_like(image)
    mask_color[mask == 1] = color
    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)
    return overlay

def visualize_segmentation():
    # 1. 随机选择一张原图
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        raise ValueError("原图文件夹为空！")
    selected_img = random.choice(img_files)
    print(f"随机选中的图片：{selected_img}")

    # 2. 读取原图、真实掩码、预测掩码
    img_full_path = os.path.join(img_path, selected_img)
    gt_full_path = get_matching_mask(selected_img, gt_path)
    pred_full_path = get_matching_mask(selected_img, pred_path)

    # 读取图片（OpenCV默认BGR，转换为RGB）
    img = cv2.imread(img_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 读取掩码（二值图，单通道）
    gt_mask = cv2.imread(gt_full_path, 0)  # 0表示灰度模式读取
    pred_mask = cv2.imread(pred_full_path, 0)

    # 确保掩码是二值的（0/1）
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    # 3. 生成叠加图
    img_gt_overlay = overlay_mask_on_image(img, gt_mask)
    img_pred_overlay = overlay_mask_on_image(img, pred_mask)

    # 4. 绘制一行五列的拼接图
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 支持英文标签

    # 子图1：原图
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')  # 关闭坐标轴

    # 子图2：真实掩码
    plt.subplot(1, 5, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth Mask', fontsize=12)
    plt.axis('off')

    # 子图3：预测掩码
    plt.subplot(1, 5, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask', fontsize=12)
    plt.axis('off')

    # 子图4：原图+真实掩码
    plt.subplot(1, 5, 4)
    plt.imshow(img_gt_overlay)
    plt.title('Original + GT Mask', fontsize=12)
    plt.axis('off')

    # 子图5：原图+预测掩码
    plt.subplot(1, 5, 5)
    plt.imshow(img_pred_overlay)
    plt.title('Original + Pred Mask', fontsize=12)
    plt.axis('off')

    # 调整子图间距，避免重叠
    plt.tight_layout()
    # 保存图片（高分辨率，适合PPT）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"可视化结果已保存至：{os.path.abspath(save_path)}")

# -------------------------- 执行可视化 --------------------------
if __name__ == "__main__":
    try:
        visualize_segmentation()
    except Exception as e:
        print(f"执行出错：{e}")