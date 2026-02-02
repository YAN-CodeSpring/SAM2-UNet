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
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                help="path to save the predicted masks")
args = parser.parse_args()

# 2. 设备与数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这里输入 224，TestDataset 会把 self.size 设为 224
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 224)

# 3. 模型加载
model = SAM2UNet().to(device)

if os.path.exists(args.checkpoint):
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ 成功加载权重: {args.checkpoint}")
else:
    raise FileNotFoundError(f"❌ 找不到权重文件: {args.checkpoint}")

model.eval()
os.makedirs(args.save_path, exist_ok=True)

# 二值化阈值
BIN_THRESHOLD = 0.5

# 【修改点 1】这里要用 .len (图片数量)，不要用 .size (分辨率)
print(f"🚀 开始测试... 总计发现 {test_loader.len} 张待测试图片")

# 【修改点 2】循环次数也要改用 .len
for i in range(test_loader.len):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        
        res, _, _ = model(image)
        
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        res = (res >= BIN_THRESHOLD).astype(np.uint8)
        res = res * 255
        
        if (i + 1) % 50 == 0:
            # 【修改点 3】进度条分母也修正为 .len
            print(f"[{i+1}/{test_loader.len}] Processing {name}...")
        
        save_name = os.path.splitext(name)[0] + ".png"
        imageio.imsave(os.path.join(args.save_path, save_name), res)

print(f"🎉 测试完成！结果已保存在: {args.save_path}")