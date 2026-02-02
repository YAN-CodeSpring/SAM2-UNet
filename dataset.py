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
        # 图片转 Tensor (0-1), Label 转 Tensor (0-255)
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像使用双三次插值，Mask使用最近邻插值(避免产生0和1之外的浮点数)
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
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

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


# ==================== 2. QaTa-COV19 适配的数据集加载器 ====================

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode='train'):
        self.images = []
        self.gts = []
        self.size = size
        self.mode = mode

        # 支持的图片扩展名
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # 1. 获取所有Mask文件，建立映射表 (核心文件名 -> 完整路径)
        # 这里的核心文件名就是去掉了 'mask_' 前缀的名字
        mask_map = {}
        if not os.path.exists(gt_root):
            raise FileNotFoundError(f"Mask路径不存在: {gt_root}")
            
        for f in os.listdir(gt_root):
            if Path(f).suffix.lower() in valid_exts:
                stem = Path(f).stem
                # 【关键逻辑】去除 'mask_' 前缀
                if stem.startswith("mask_"):
                    core_name = stem[5:]
                else:
                    core_name = stem
                mask_map[core_name] = os.path.join(gt_root, f)

        # 2. 扫描图片文件夹进行匹配
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"图片路径不存在: {image_root}")

        img_files = sorted([f for f in os.listdir(image_root) if Path(f).suffix.lower() in valid_exts])
        
        for img_name in img_files:
            img_path = os.path.join(image_root, img_name)
            core_name = Path(img_name).stem  # 图片的文件名就是核心名 (例如 covid_1)
            
            # 尝试在 mask_map 中找对应的 mask
            if core_name in mask_map:
                self.images.append(img_path)
                self.gts.append(mask_map[core_name])
            else:
                # 仅在调试模式下打印，防止刷屏
                # print(f"⚠️ 跳过: 找不到图片 {img_name} 对应的 Mask")
                pass

        print(f"✅ [{mode.upper()}] 成功加载 {len(self.images)} 对数据 (Images: {image_root})")

        # 定义变换流程
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            # 验证/测试模式
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        # 读取图片和Mask
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        
        # 应用变换
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
            # 确保读入的是单通道 Mask (0=背景, 255=前景)
            return img.convert('L')


class TestDataset:
    """
    专门用于测试/推理阶段的数据加载器 (无数据增强，逐张返回)
    """
    def __init__(self, image_root, gt_root, size):
        self.images = []
        self.gts = []
        self.size = size
        self.index = 0

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # 同样建立 Mask 映射表
        mask_map = {}
        for f in os.listdir(gt_root):
            if Path(f).suffix.lower() in valid_exts:
                stem = Path(f).stem
                if stem.startswith("mask_"):
                    core_name = stem[5:]
                else:
                    core_name = stem
                mask_map[core_name] = os.path.join(gt_root, f)

        # 扫描图片
        img_files = sorted([f for f in os.listdir(image_root) if Path(f).suffix.lower() in valid_exts])
        
        for img_name in img_files:
            img_path = os.path.join(image_root, img_name)
            core_name = Path(img_name).stem
            
            if core_name in mask_map:
                self.images.append(img_path)
                self.gts.append(mask_map[core_name])
            else:
                print(f"⚠️ 测试集警告: 图片 {img_name} 缺失 Mask")

        self.len = len(self.images)
        print(f"✅ [TEST] 准备就绪，共 {self.len} 张测试图片。")

        # 测试时的预处理 (仅缩放+归一化)
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_data(self):
        if self.index >= self.len:
            raise StopIteration("测试集数据已遍历完毕！")
        
        # 读取原始图片 (用于可视化)
        image_path = self.images[self.index]
        gt_path = self.gts[self.index]
        
        image = self.rgb_loader(image_path)
        
        # 预处理图片 (用于送入模型)
        image_tensor = self.transform(image).unsqueeze(0) # 增加 Batch 维度 -> (1, 3, H, W)

        # 读取 GT Mask (不缩放，保持原尺寸以便计算准确指标)
        gt = self.binary_loader(gt_path)
        gt = np.array(gt)
        # 将 GT 二值化 (0 和 255 -> 0 和 1)
        gt = (gt > 128).astype(np.float32)

        name = Path(image_path).name # 获取文件名 (如 covid_1.png)
        
        self.index += 1
        
        return image_tensor, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

__all__ = ['FullDataset', 'TestDataset', 'Resize', 'ToTensor', 'Normalize', 'RandomHorizontalFlip', 'RandomVerticalFlip']