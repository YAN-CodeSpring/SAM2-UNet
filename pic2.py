#### 指定文件名字进行像素数值验证

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 配置参数 --------------------------
# 这里的路径请保持和你之前的代码一致
gt_path = "/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary"       # 真实掩码文件夹
pred_path = "/root/autodl-tmp/BCSS/output14_predict_results/" # 预测掩码文件夹

# 指定你要检查的文件名 (带不带后缀都可以，代码会自动匹配)
target_name = "TCGA-E2-A1LI-DX1_xmin44715_ymin18434_MPP-0_896_4480_size224.png"

# -------------------------- 辅助函数 --------------------------
def find_file(name_prefix, folder):
    """在文件夹中查找匹配的文件路径"""
    name_prefix = os.path.splitext(name_prefix)[0] # 去掉后缀以防万一
    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == name_prefix:
            return os.path.join(folder, f)
    return None

def analyze_mask(path, label):
    """核心分析函数"""
    if path is None:
        print(f"❌ [错误] 在 {label} 路径下未找到该文件！")
        return None
    
    # 以“原样”模式读取（避免OpenCV自动转彩色或压缩）
    # flags=-1 (IMREAD_UNCHANGED) 能读取原本的位深
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"❌ [错误] 无法读取文件: {path}")
        return None

    print(f"\n--- 分析 {label} ---")
    print(f"文件路径: {os.path.basename(path)}")
    print(f"图像形状: {img.shape}")
    print(f"数据类型: {img.dtype}")
    
    # 获取唯一值
    unique_vals = np.unique(img)
    print(f"★ 包含的唯一像素值 (Unique Values): {unique_vals}")
    
    # 统计每个值的像素数量
    for val in unique_vals:
        count = np.sum(img == val)
        ratio = count / img.size * 100
        print(f"  - 值 {val}: {count} 个像素 (占 {ratio:.2f}%)")
        
    return img

# -------------------------- 执行检查 --------------------------
if __name__ == "__main__":
    # 1. 查找文件
    gt_full_path = find_file(target_name, gt_path)
    pred_full_path = find_file(target_name, pred_path)

    # 2. 分析数值
    gt_img = analyze_mask(gt_full_path, "真实掩码 (GT)")
    pred_img = analyze_mask(pred_full_path, "预测掩码 (Pred)")

    # 3. 可视化对比 (如果有读取到图片)
    if gt_img is not None or pred_img is not None:
        plt.figure(figsize=(12, 5))
        
        # 显示 GT
        if gt_img is not None:
            plt.subplot(1, 2, 1)
            # 使用 jet 或 viridis 色图，这样 0, 1, 255 会显示完全不同的颜色
            plt.imshow(gt_img, cmap='jet', interpolation='nearest') 
            plt.colorbar(label='Pixel Value')
            plt.title(f"GT Mask Values\n{np.unique(gt_img)}")
            plt.axis('off')

        # 显示 Pred
        if pred_img is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(pred_img, cmap='jet', interpolation='nearest')
            plt.colorbar(label='Pixel Value')
            plt.title(f"Pred Mask Values\n{np.unique(pred_img)}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        print("\n✅ 分析完成，请查看上方图表和控制台输出。")