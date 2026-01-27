import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ==================== 1. 数据增强与变换模块 ====================
# (保持不变，非常标准的变换流程)
class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label) * 255}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像使用双三次插值，Mask使用最近邻插值(避免产生0和1之外的浮点数)
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)}

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

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

# ==================== 2. BCSS 适配的数据集加载器 ====================

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = []
        self.gts = []

        # 获取并排序所有图片，确保原图和Mask严格对应
        img_names = sorted([f for f in os.listdir(image_root) if f.endswith('.png')])
        
        for img_name in img_names:
            img_path = os.path.join(image_root, img_name)
            mask_path = os.path.join(gt_root, img_name) # BCSS中，Mask和原图同名

            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.gts.append(mask_path)
            else:
                print(f"⚠️ 警告：找不到图像 {img_name} 对应的 Mask，已跳过。")

        # 数据增强组合
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        
        print(f"✅ [{mode.upper()}] 成功加载并匹配 {len(self.images)} 对图像与掩码。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') # 灰度图读取

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = []
        self.gts = []

        img_names = sorted([f for f in os.listdir(image_root) if f.endswith('.png')])
        
        for img_name in img_names:
            img_path = os.path.join(image_root, img_name)
            mask_path = os.path.join(gt_root, img_name)

            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.gts.append(mask_path)
            else:
                raise FileNotFoundError(f"测试集：未找到图像 {img_name} 对应的mask文件！")

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        if self.index >= self.size:
            raise StopIteration("测试集数据已遍历完毕！")
        
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt) # 测试时返回原尺寸 Numpy 数组，方便计算指标

        name = self.images[self.index].split('/')[-1]
        self.index += 1
        
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

__all__ = ['FullDataset', 'TestDataset', 'Resize', 'ToTensor', 'Normalize', 'RandomHorizontalFlip', 'RandomVerticalFlip']