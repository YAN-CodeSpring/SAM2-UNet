###### 指定文件名字进行可视化

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 配置参数 --------------------------
# 路径配置
img_path = "/root/autodl-tmp/BCSS/BCSS_224/val"    
gt_path = "/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary" 
pred_path = "/root/autodl-tmp/BCSS/output14_predict_results/" 

# ★★★ 在这里修改你要查看的具体文件名 ★★★
target_img_name = "TCGA-D8-A13Z-DX1_xmin30985_ymin49337_MPP-0_1120_1344_size224.png"

# 可视化配置
mask_color = (255, 0, 0)  # 红色 (R, G, B)
alpha = 0.4               # 透明度 (0.4 比较适中，不会完全遮挡原图细节)
save_path = "specific_vis_result.png"
figsize = (20, 4)

# -------------------------- 核心函数 --------------------------
def get_matching_mask(img_name, mask_folder):
    """根据文件名找掩码"""
    img_prefix = os.path.splitext(img_name)[0]
    for mask_file in os.listdir(mask_folder):
        if os.path.splitext(mask_file)[0] == img_prefix:
            return os.path.join(mask_folder, mask_file)
    print(f"⚠️ 警告: 在 {mask_folder} 中未找到 {img_name} 对应的文件")
    return None

def overlay_mask_on_image(image, mask, color=mask_color, alpha=alpha):
    """
    将掩码叠加到原图。
    逻辑：只要 mask > 0 (非背景)，就认为是目标区域，叠加颜色。
    """
    # 1. 归一化为 0/1 二值图
    # 这里 mask > 0 非常重要，因为它能把类别 2, 3, 4 等都变成 1 (即显示为红色)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 2. 准备彩色遮罩
    mask_layer = np.zeros_like(image)
    mask_layer[binary_mask == 1] = color
    
    # 3. 叠加 (仅在掩码区域进行混合，非掩码区域保持原图)
    # 这种写法比 cv2.addWeighted 全图混合更清晰，背景不会变暗
    output = image.copy()
    # 获取需要上色的区域索引
    idx = (binary_mask == 1)
    # 仅对该区域进行加权混合
    output[idx] = cv2.addWeighted(image[idx], 1 - alpha, mask_layer[idx], alpha, 0)
    
    return output

def visualize_specific_image():
    print(f"正在处理图片: {target_img_name} ...")
    
    # 1. 获取路径
    img_full_path = os.path.join(img_path, target_img_name)
    gt_full_path = get_matching_mask(target_img_name, gt_path)
    pred_full_path = get_matching_mask(target_img_name, pred_path)

    if not os.path.exists(img_full_path):
        raise FileNotFoundError(f"原图不存在: {img_full_path}")

    # 2. 读取数据
    img = cv2.imread(img_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转 RGB
    
    # 读取掩码 (保持原值，不要一开始就转二值，方便看类别)
    gt_mask = cv2.imread(gt_full_path, 0) if gt_full_path else np.zeros(img.shape[:2], dtype=np.uint8)
    pred_mask = cv2.imread(pred_full_path, 0) if pred_full_path else np.zeros(img.shape[:2], dtype=np.uint8)

    # ★ 打印当前数据的数值分布，确认逻辑 ★
    print(f"GT 掩码数值: {np.unique(gt_mask)} (若包含 >0 的值，对应区域会变红)")
    print(f"Pred 掩码数值: {np.unique(pred_mask)} (若全为 0，则无红色叠加)")

    # 3. 生成叠加图
    img_gt_overlay = overlay_mask_on_image(img, gt_mask)
    img_pred_overlay = overlay_mask_on_image(img, pred_mask)

    # 4. 绘图
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    titles = ['Original Image', 'GT Mask (Raw)', 'Pred Mask (Raw)', 'Original + GT', 'Original + Pred']
    images = [img, gt_mask, pred_mask, img_gt_overlay, img_pred_overlay]

    for i in range(5):
        plt.subplot(1, 5, i+1)
        # 如果是掩码(索引1和2)，用 jet 色图显示，方便看清楚 0, 1, 2 的区别
        if i in [1, 2]:
            plt.imshow(images[i], cmap='jet', vmin=0, vmax=3) # 固定范围，方便对比
            plt.colorbar(fraction=0.046, pad=0.04) # 加个色条
        else:
            plt.imshow(images[i])
        
        plt.title(titles[i], fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    # plt.show() # 如果在服务器上运行，通常不需要show，直接看保存的文件

# -------------------------- 执行 --------------------------
if __name__ == "__main__":
    visualize_specific_image()