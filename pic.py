#### 随机可视化一张图片

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 配置参数 --------------------------
# 文件夹路径
img_path = "/root/autodl-fs/QaTa-COV19-v2/Test Set/Images"    # 原图路径
gt_path = "/root/autodl-fs/QaTa-COV19-v2/Test Set/Ground-truths" # 真实掩码路径
pred_path = "/root/autodl-tmp/COVID/output1_predict_results/" # 预测掩码路径

# 可视化配置
mask_color = (255, 0, 0)  # 掩码叠加颜色（红色）
alpha = 0.5               # 透明度
save_path = "segmentation_vis_random.png"  # 保存路径
figsize = (20, 4)         # 图片尺寸

# -------------------------- 核心函数 --------------------------
def get_matching_mask(img_name, mask_folder):
    """根据原图文件名找到对应的掩码文件"""
    img_prefix = os.path.splitext(img_name)[0]
    for mask_file in os.listdir(mask_folder):
        if os.path.splitext(mask_file)[0] == img_prefix:
            return os.path.join(mask_folder, mask_file)
    # 如果找不到，返回 None，避免直接报错崩溃，方便调试
    print(f"⚠️ 警告：在 {mask_folder} 中未找到 {img_name} 的掩码")
    return None

def overlay_mask_on_image(image, mask, color=mask_color, alpha=alpha):
    """
    将掩码叠加到原图上。
    逻辑：mask > 0 的区域被视为前景，叠加颜色。
    """
    # 1. 转为标准的 0/1 二值掩码
    binary_mask = (mask > 0).astype(np.uint8)

    # 2. 创建彩色层
    mask_layer = np.zeros_like(image)
    mask_layer[binary_mask == 1] = color

    # 3. 叠加逻辑 (仅混合前景区域)
    output = image.copy()
    idx = (binary_mask == 1)
    
    # 只有当掩码里有内容时才进行混合操作
    if idx.any():
        # 只取需要混合的像素区域进行计算，效率更高且不出错
        output[idx] = cv2.addWeighted(image[idx], 1 - alpha, mask_layer[idx], alpha, 0)

    return output

def visualize_segmentation():
    # 1. 随机选择一张原图
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        raise ValueError("原图文件夹为空！")
    
    selected_img = random.choice(img_files)
    print(f"🎲 随机选中的图片：{selected_img}")

    # 2. 获取路径
    img_full_path = os.path.join(img_path, selected_img)
    gt_full_path = get_matching_mask(selected_img, gt_path)
    pred_full_path = get_matching_mask(selected_img, pred_path)

    # 3. 读取并预处理
    # 读取原图
    img = cv2.imread(img_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取掩码 (如果找不到文件，生成全黑掩码代替，防止代码报错中断)
    if gt_full_path:
        gt_mask = cv2.imread(gt_full_path, 0)
    else:
        gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if pred_full_path:
        pred_mask = cv2.imread(pred_full_path, 0)
    else:
        pred_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # ★关键步骤★：统一将掩码转为 0 和 1
    # 这样无论原始数据是 [0, 1] 还是 [0, 255]，处理后都一样
    gt_mask_binary = (gt_mask > 0).astype(np.uint8)
    pred_mask_binary = (pred_mask > 0).astype(np.uint8)

    # 4. 生成叠加图
    img_gt_overlay = overlay_mask_on_image(img, gt_mask_binary)
    img_pred_overlay = overlay_mask_on_image(img, pred_mask_binary)

    # 5. 绘图
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    titles = ['Original Image', 'GT Mask (Binary)', 'Pred Mask (Binary)', 'Original + GT', 'Original + Pred']
    # 这里的列表里放的是我们处理好的二值掩码
    images_to_show = [img, gt_mask_binary, pred_mask_binary, img_gt_overlay, img_pred_overlay]

    for i in range(5):
        plt.subplot(1, 5, i+1)
        
        # 针对掩码图（索引1和2）进行特殊显示设置
        if i in [1, 2]:
            # ★★★ 核心修复 ★★★
            # 强制指定 vmin=0 (黑) 和 vmax=1 (白)
            # 这样即使整张图都是 1，它也会显示为全白，而不是被自动缩放成全黑
            plt.imshow(images_to_show[i], cmap='gray', vmin=0, vmax=1)
            plt.title(f"{titles[i]}\nUnique: {np.unique(images_to_show[i])}", fontsize=10) # 标题里顺便显示数值，方便检查
        else:
            plt.imshow(images_to_show[i])
            plt.title(titles[i], fontsize=12)
            
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至：{os.path.abspath(save_path)}")

# -------------------------- 执行 --------------------------
if __name__ == "__main__":
    try:
        visualize_segmentation()
    except Exception as e:
        print(f"❌ 执行出错：{e}")
        import traceback
        traceback.print_exc()