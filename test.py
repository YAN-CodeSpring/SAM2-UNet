import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset

# 1. 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="path to your best_model.pth")
parser.add_argument("--test_image_path", type=str, required=True)
parser.add_argument("--test_gt_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【适配 1】尺寸统一为 512
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 512)

# 【适配 2】模型加载（不再需要 SAM2 原始权重路径，直接初始化结构）
model = SAM2UNet().to(device)

if os.path.exists(args.checkpoint):
    # 加载你自己训练出的 best_model.pth
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ 成功加载自定义训练权重: {args.checkpoint}")
else:
    raise FileNotFoundError(f"❌ 找不到权重文件: {args.checkpoint}")

model.eval()
os.makedirs(args.save_path, exist_ok=True)

print(f"🚀 开始推理... 总计: {test_loader.len} 张图片")

# 【适配 3】三分类像素映射表
# 索引 0 (背景) -> 0
# 索引 1 (良性) -> 128
# 索引 2 (恶性) -> 255
class_to_pixel = {0: 0, 1: 128, 2: 255}

for i in range(test_loader.len):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        
        # gt 仅用于获取原始尺寸进行插值回传
        orig_h, orig_w = gt.shape
        image = image.to(device)
        
        # 推理得到三个尺度，我们取最终输出 res
        res, _, _ = model(image)
        
        # 【核心修改】多分类处理逻辑
        # 1. 插值回到原始图像尺寸
        res = F.interpolate(res, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        # 2. 从 [1, 3, H, W] 中提取概率最大的类别索引 [H, W]
        res = torch.argmax(res, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # 3. 将索引 (0, 1, 2) 映射回可视化像素 (0, 128, 255)
        res_colored = np.zeros_like(res, dtype=np.uint8)
        for cls_idx, pixel_val in class_to_pixel.items():
            res_colored[res == cls_idx] = pixel_val
        
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{test_loader.len}] 正在处理: {name}...")
        
        save_name = os.path.splitext(name)[0] + ".png"
        imageio.imsave(os.path.join(args.save_path, save_name), res_colored)

print(f"🎉 推理完成！预测结果（三色图）已保存至: {args.save_path}")