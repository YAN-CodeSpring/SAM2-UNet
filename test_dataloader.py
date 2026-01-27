import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import FullDataset  # 导入刚才修改的 dataset.py

def denormalize(img_tensor):
    """把归一化的图片还原回正常颜色"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def test_bcss_loader():
    # ========== ⚠️ 请在这里修改为你的本地路径 ==========
    IMG_DIR = "/root/autodl-tmp/BCSS/BCSS_224/val"
    MASK_DIR = "/root/autodl-tmp/BCSS/BCSS_224/val_mask_binary" # 使用你之前生成的二值掩码

    # 实例化 Dataset
    dataset = FullDataset(image_root=IMG_DIR, gt_root=MASK_DIR, size=224, mode='train')

    # 创建 DataLoader，随机打乱方便抽样
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 获取一个 Batch
    batch = next(iter(loader))
    images, masks = batch['image'], batch['label']

    print("\n" + "="*40)
    print("🛠️ 维度检查:")
    print(f"图像 Tensor 形状: {images.shape}")
    print(f"Mask Tensor 形状: {masks.shape}")
    
    # 关键检查：查看 Mask 里面到底包含哪些值
    unique_vals = torch.unique(masks).numpy()
    print(f"Mask 包含的唯一值: {unique_vals} (如果是 [0. 1.] 则完美二值化成功！)")
    print("="*40 + "\n")

    # 可视化前 3 张图
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    for i in range(3):
        img_show = denormalize(images[i])
        mask_show = masks[i].squeeze().numpy()

        axes[i, 0].imshow(img_show)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_show, cmap='gray')
        axes[i, 1].set_title(f"Binary Mask {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("dataloader_test.png", dpi=150)
    print("✅ 可视化结果已保存至 dataloader_test.png")

if __name__ == "__main__":
    test_bcss_loader()