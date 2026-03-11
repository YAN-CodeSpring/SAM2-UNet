import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ==================== 1. 数据增强与变换模块 ====================

class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        
        # 图片转 Tensor (归一化到 0-1)
        image_tensor = F.to_tensor(image)
        
        # Label 处理：将原始像素值映射为类别索引
        # 0 -> 0 (背景), 128 -> 1 (良性), 255 -> 2 (恶性)
        label_np = np.array(label)
        target = np.zeros_like(label_np, dtype=np.float32)
        target[(label_np >= 100) & (label_np <= 150)] = 1.0  # 宽容度映射良性
        target[label_np > 200] = 2.0                         # 宽容度映射恶性
        
        # 转为 Tensor，增加通道维度 (1, H, W)
        label_tensor = torch.from_numpy(target).unsqueeze(0)
        
        return {'image': image_tensor, 'label': label_tensor}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像使用双线性插值
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        # Mask 必须使用最近邻插值，保护 0, 128, 255 像素值的纯净
        label = F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        return {'image': image, 'label': label}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}
        return {'image': image, 'label': label}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}
        return {'image': image, 'label': label}
    
class RandomRotation(object): # 添加了一个随机旋转15度的数据增强处理方法。
    def __init__(self, degree=15):
        self.degree = degree

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < 0.5:
            angle = random.uniform(-self.degree, self.degree)
            # 旋转也要保持插值模式一致
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            label = F.rotate(label, angle, interpolation=InterpolationMode.NEAREST)
        return {'image': image, 'label': label}
    
class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

# ==================== 2. 数据集加载器 ====================

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode='train'):
        self.images = []
        self.gts = []
        self.size = size
        self.mode = mode

        # 支持的图片扩展名
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # 扫描图片文件夹
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"图片路径不存在: {image_root}")
        
        img_files = sorted([f for f in os.listdir(image_root) if Path(f).suffix.lower() in valid_exts])
        
        for img_name in img_files:
            img_path = os.path.join(image_root, img_name)
            mask_path = os.path.join(gt_root, img_name) # 1. 名字完全一样，直接拼接
            
            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.gts.append(mask_path)
            else:
                # print(f"⚠️ 跳过: 找不到对应 Mask: {img_name}")
                pass

        print(f"✅ [{mode.upper()}] 加载 {len(self.images)} 对数据 (Size: {size}x{size})")

        # 变换流程
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)), # 3. 统一 Resize 到指定大小
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotation(degree=15), # 这个地方应用了随机旋转15度
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # 保持原始像素值，用于 ToTensor 中的逻辑映射
            return img.convert('L')

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = []
        self.gts = []
        self.size = size
        self.index = 0

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        img_files = sorted([f for f in os.listdir(image_root) if Path(f).suffix.lower() in valid_exts])
        
        for img_name in img_files:
            self.images.append(os.path.join(image_root, img_name))
            self.gts.append(os.path.join(gt_root, img_name))

        self.len = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_data(self):
        if self.index >= self.len:
            return None
        
        image_path = self.images[self.index]
        gt_path = self.gts[self.index]
        
        image = self.rgb_loader(image_path)
        image_tensor = self.transform(image).unsqueeze(0)
        
        gt = self.binary_loader(gt_path)
        gt = np.array(gt)
        # 测试集通常用于评估，这里根据你的阈值判定
        # 如果需要区分良恶性，可能需要根据具体指标函数修改
        gt = np.array(gt, dtype=np.float32) # 去掉 > 0 的判断 ###### 这里改动了
        
        name = Path(image_path).name
        self.index += 1
        return image_tensor, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

__all__ = ['FullDataset', 'TestDataset']