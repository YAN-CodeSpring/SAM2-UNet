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
gt_color = (0, 255, 0)    # GT掩码叠加颜色（绿色，RGB格式）
pred_color = (255, 0, 0)  # Pred掩码叠加颜色（红色，RGB格式）
alpha = 0.5               # 透明度（半透明）
save_path = "segmentation_vis_random.png"  # 保存路径
figsize = (24, 6)         # 调整尺寸适配4列布局

# -------------------------- 核心函数 --------------------------
def get_matching_mask(img_name, mask_folder, is_gt=False):
    """
    根据原图文件名找到对应的掩码文件
    :param img_name: 原图文件名
    :param mask_folder: 掩码文件夹路径
    :param is_gt: 是否为GT掩码（GT需要mask_前缀）
    :return: 掩码文件路径，找不到返回None
    """
    img_prefix = os.path.splitext(img_name)[0]
    # 针对GT掩码：拼接mask_前缀
    target_prefix = f"mask_{img_prefix}" if is_gt else img_prefix
    
    for mask_file in os.listdir(mask_folder):
        mask_prefix = os.path.splitext(mask_file)[0]
        if mask_prefix == target_prefix:
            return os.path.join(mask_folder, mask_file)
    
    # 如果找不到，返回 None，避免直接报错崩溃，方便调试
    print(f"⚠️ 警告：在 {mask_folder} 中未找到 {img_name} 对应的掩码（目标前缀：{target_prefix}）")
    return None

def overlay_mask_on_image(image, mask, color, alpha=alpha):
    """
    将掩码叠加到原图上（支持自定义颜色和透明度）
    :param image: 原图（RGB格式）
    :param mask: 二值掩码（0/1）
    :param color: 叠加颜色（RGB元组）
    :param alpha: 透明度（0-1）
    :return: 叠加后的图片
    """
    # 1. 确保掩码是二值的（0/1）
    binary_mask = (mask > 0).astype(np.uint8)

    # 2. 创建彩色层
    mask_layer = np.zeros_like(image)
    mask_layer[binary_mask == 1] = color

    # 3. 叠加逻辑 (仅混合前景区域，效率更高)
    output = image.copy()
    idx = (binary_mask == 1)
    
    if idx.any():
        output[idx] = cv2.addWeighted(image[idx], 1 - alpha, mask_layer[idx], alpha, 0)

    return output

def overlay_double_masks(image, gt_mask, pred_mask, gt_color=gt_color, pred_color=pred_color, alpha=alpha):
    """
    叠加GT（绿）和Pred（红）双掩码到原图
    :param image: 原图（RGB）
    :param gt_mask: GT二值掩码（0/1）
    :param pred_mask: Pred二值掩码（0/1）
    :return: 双掩码叠加后的图片
    """
    # 先叠加GT掩码（绿色）
    img_with_gt = overlay_mask_on_image(image, gt_mask, gt_color, alpha)
    # 再叠加Pred掩码（红色）
    img_with_both = overlay_mask_on_image(img_with_gt, pred_mask, pred_color, alpha)
    return img_with_both

def visualize_segmentation():
    # 1. 随机选择一张原图
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        raise ValueError("原图文件夹为空！")
    
    selected_img = random.choice(img_files)
    print(f"🎲 随机选中的图片：{selected_img}")

    # 2. 获取路径（GT掩码需要指定is_gt=True）
    img_full_path = os.path.join(img_path, selected_img)
    gt_full_path = get_matching_mask(selected_img, gt_path, is_gt=True)  # 修改点：GT加mask_前缀
    pred_full_path = get_matching_mask(selected_img, pred_path)

    # 3. 读取并预处理
    # 读取原图（转RGB，适配matplotlib）
    img = cv2.imread(img_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取掩码 (找不到则生成全黑掩码，防止报错)
    if gt_full_path:
        gt_mask = cv2.imread(gt_full_path, 0)
    else:
        gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if pred_full_path:
        pred_mask = cv2.imread(pred_full_path, 0)
    else:
        pred_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 统一转为0/1二值掩码（兼容0-1或0-255的原始掩码）
    gt_mask_binary = (gt_mask > 0).astype(np.uint8)
    pred_mask_binary = (pred_mask > 0).astype(np.uint8)

    # 4. 生成四张可视化图
    img_original = img  # 1. 原图
    img_with_gt = overlay_mask_on_image(img, gt_mask_binary, gt_color, alpha)  # 2. 原图+绿色GT
    img_with_pred = overlay_mask_on_image(img, pred_mask_binary, pred_color, alpha)  # 3. 原图+红色Pred
    img_with_both = overlay_double_masks(img, gt_mask_binary, pred_mask_binary)  # 4. 原图+双掩码

    # 5. 绘图（一行四列）
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    # 定义子图标题和要显示的图片
    titles = [
        'Original Image', 
        'Original + GT (Green)', 
        'Original + Pred (Red)', 
        'Original + GT + Pred'
    ]
    images_to_show = [img_original, img_with_gt, img_with_pred, img_with_both]

    # 绘制子图
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images_to_show[i])
        plt.title(titles[i], fontsize=12)
        plt.axis('off')  # 关闭坐标轴

    # 在整张大图下方添加原图名称
    plt.figtext(0.5, 0.02, f"Image Name: {selected_img}", ha='center', fontsize=14, weight='bold')

    # 调整布局（避免标题/文本重叠）
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # rect留出底部文本空间
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