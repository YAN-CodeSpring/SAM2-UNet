import torch
import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ========== 1. 多分类兼容的Transform类（核心修改） ==========
class ToTensor(object):
    """支持二值/多分类mask的ToTensor转换：
       - 图像：转成0-1浮点数
       - 多分类mask：保留原始整数标签（0-21），转为long类型
    """
    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像转成0-1浮点数
        image_tensor = F.to_tensor(image)
        # 多分类mask转成long类型tensor（保留原始标签值）
        label_tensor = torch.tensor(np.array(label), dtype=torch.long)
        return {'image': image_tensor, 'label': label_tensor}

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像用默认插值，mask用最近邻插值（避免标签值被平滑）
        return {
            'image': F.resize(image, self.size), 
            'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        }

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

# ========== 2. 原有BUSI二值分割类（保留，兼容多分类Transform） ==========
class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
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
            return img.convert('L')

# ========== 3. 原有BUSI测试类（保留） ==========
class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)
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

# ========== 4. 多分类专用：BCSS数据集类（核心修正） ==========
class BCSSFullDataset(Dataset):
    def __init__(self, image_root, mask_root, size, mode):
        """
        BCSS多分类语义分割数据集加载
        :param image_root: 图像目录（如/root/autodl-tmp/BCSS/BCSS_224/train）
        :param mask_root: mask目录（如/root/autodl-tmp/BCSS/BCSS_224/train_mask）
        :param size: 图像resize尺寸（如224/512）
        :param mode: train/val（控制数据增强）
        """
        # 加载图像和mask（同名匹配）
        self.image_names = [f for f in os.listdir(image_root) if f.endswith('.png')]
        self.image_paths = [os.path.join(image_root, f) for f in self.image_names]
        self.mask_paths = [os.path.join(mask_root, f) for f in self.image_names]
        
        # 校验数据完整性
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"图像数({len(self.image_paths)})与mask数({len(self.mask_paths)})不匹配！")
        missing_masks = [p for p in self.mask_paths if not os.path.exists(p)]
        if missing_masks:
            raise FileNotFoundError(f"缺失mask文件：{missing_masks[:3]}...（共{len(missing_masks)}个）")
        
        # 数据增强（多分类兼容：mask用最近邻插值）
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),  # 多分类兼容的ToTensor
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        
        print(f"✅ BCSS {mode}集加载完成：{len(self.image_paths)} 张图像 | 22类多分类")

    def __getitem__(self, idx):
        # 加载图像
        image = self.rgb_loader(self.image_paths[idx])
        # 加载多分类mask（保留原始标签值0-21，不做二值化）
        mask = self.multi_class_mask_loader(self.mask_paths[idx])
        # 应用transform
        data = {'image': image, 'label': mask}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.image_paths)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def multi_class_mask_loader(self, path):
        """加载BCSS多分类mask（保留原始标签值0-21）"""
        with open(path, 'rb') as f:
            img = Image.open(f).convert('L')  # 单通道加载
        # 验证标签值范围（0-21）
        mask_np = np.array(img)
        if not (np.all((mask_np >= 0) & (mask_np <= 21))):
            raise ValueError(f"mask标签值超出范围0-21：{np.unique(mask_np)}")
        return Image.fromarray(mask_np)

# ========== 5. 多分类专用：BCSS测试类 ==========
class BCSSTestDataset:
    def __init__(self, image_root, mask_root, size):
        self.image_names = [f for f in os.listdir(image_root) if f.endswith('.png')]
        self.image_paths = [os.path.join(image_root, f) for f in self.image_names]
        self.has_mask = mask_root is not None
        
        if self.has_mask:
            self.mask_paths = [os.path.join(mask_root, f) for f in self.image_names]
            missing_masks = [p for p in self.mask_paths if not os.path.exists(p)]
            if missing_masks:
                raise FileNotFoundError(f"缺失测试mask：{missing_masks[:3]}...")
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.size = len(self.image_paths)
        self.index = 0
        
        print(f"✅ BCSS测试集加载完成：{len(self.image_paths)} 张图像")

    def load_data(self):
        if self.index >= self.size:
            raise StopIteration("BCSS测试集已遍历完毕！")
        
        image_path = self.image_paths[self.index]
        image = self.rgb_loader(image_path)
        image = self.transform(image).unsqueeze(0)
        
        gt = None
        if self.has_mask:
            mask = self.multi_class_mask_loader(self.mask_paths[self.index])
            gt = np.array(mask)
        
        name = image_path.split('/')[-1]
        self.index += 1
        
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def multi_class_mask_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('L')
        mask_np = np.array(img)
        if not (np.all((mask_np >= 0) & (mask_np <= 21))):
            raise ValueError(f"测试mask标签值超出范围0-21：{np.unique(mask_np)}")
        return Image.fromarray(mask_np)

# ========== 6. 导出所有类 ==========
__all__ = [
    'FullDataset', 'TestDataset', 
    'BCSSFullDataset', 'BCSSTestDataset',
    'Resize', 'ToTensor', 'Normalize', 'RandomHorizontalFlip', 'RandomVerticalFlip'
]