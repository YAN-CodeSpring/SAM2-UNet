import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

# 二值化阈值（0.5对应255的127.5）
BIN_THRESHOLD = 0.5

for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res, _, _ = model(image)
        
        # 1. 上采样到gt尺寸（保持和原代码一致）
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        
        # 2. sigmoid归一化到0-1（避免重复sigmoid）
        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        
        # 3. 强制二值化（0或1，和mask尺度对齐）
        res = (res >= BIN_THRESHOLD).astype(np.uint8)
        
        # 4. 转换为0-255的二值图（方便保存为png，和评估时的mask统一尺度）
        res = res * 255
        
        print(f"Saving {name} | pred像素值范围：[{res.min()}, {res.max()}]")
        # 保存为png（仅0/255二值图）
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)