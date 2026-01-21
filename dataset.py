import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


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
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        # ========== 1. 加载图像路径并按图像名排序 ==========
        self.images = []
        img_name_list = []  # 存储纯图像名（用于匹配mask）
        for f in os.listdir(image_root):
            if f.endswith('.jpg') or f.endswith('.png'):
                self.images.append(os.path.join(image_root, f))
                img_name_list.append(f.replace('.png', '').replace('.jpg', ''))
        # 按图像名排序（保证顺序一致）
        sorted_pairs = sorted(zip(img_name_list, self.images))
        img_name_list, self.images = zip(*sorted_pairs)
        self.images = list(self.images)
        img_name_list = list(img_name_list)

        # ========== 2. 构建图像名→mask路径的映射 ==========
        img_name_to_mask = {}
        for class_dir in os.listdir(gt_root):
            class_dir_path = os.path.join(gt_root, class_dir)
            if not os.path.isdir(class_dir_path):
                continue
            for f in os.listdir(class_dir_path):
                if f.endswith('_mask.png'):
                    img_base_name = f.replace('_mask.png', '')
                    mask_path = os.path.join(class_dir_path, f)
                    img_name_to_mask[img_base_name] = mask_path

        # ========== 3. 按排序后的图像名，逐个匹配mask ==========
        self.gts = []
        for img_name in img_name_list:
            if img_name in img_name_to_mask:
                self.gts.append(img_name_to_mask[img_name])
            else:
                raise FileNotFoundError(f"未找到图像 {img_name} 对应的mask文件！")

        # ========== 保留原有transform逻辑 ==========
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
        
        # ========== 打印验证 ==========
        print(f"成功加载 {len(self.images)} 张图像")
        print(f"成功匹配 {len(self.gts)} 个mask文件")
        if len(self.images) > 0:
            print(f"示例图像路径：{self.images[700]}")
            print(f"示例图像名：{img_name_list[700]}")
        if len(self.gts) > 0:
            print(f"示例mask路径：{self.gts[700]}")

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    # ========== 关键：确保__len__方法缩进正确（和__init__/__getitem__同级） ==========
    def __len__(self):
        # 返回数据集的总长度（图像数量）
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, gt_root, size):
        # ========== 1. 加载图像路径并按图像名排序（和FullDataset一致） ==========
        self.images = []
        img_name_list = []
        for f in os.listdir(image_root):
            if f.endswith('.jpg') or f.endswith('.png'):
                self.images.append(os.path.join(image_root, f))
                img_name_list.append(f.replace('.png', '').replace('.jpg', ''))
        # 按图像名排序，保证和训练集顺序一致
        sorted_pairs = sorted(zip(img_name_list, self.images))
        img_name_list, self.images = zip(*sorted_pairs)
        self.images = list(self.images)
        img_name_list = list(img_name_list)

        # ========== 2. 匹配mask（适配0/1/2子文件夹 + _mask后缀） ==========
        self.gts = []
        img_name_to_mask = {}
        for class_dir in os.listdir(gt_root):
            class_dir_path = os.path.join(gt_root, class_dir)
            if not os.path.isdir(class_dir_path):
                continue
            for f in os.listdir(class_dir_path):
                if f.endswith('_mask.png'):
                    img_base_name = f.replace('_mask.png', '')
                    mask_path = os.path.join(class_dir_path, f)
                    img_name_to_mask[img_base_name] = mask_path

        for img_name in img_name_list:
            if img_name in img_name_to_mask:
                self.gts.append(img_name_to_mask[img_name])
            else:
                raise FileNotFoundError(f"测试集：未找到图像 {img_name} 对应的mask文件！")

        # ========== 保留原有transform逻辑 ==========
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)  # 测试集总数量
        self.index = 0  # 遍历索引，初始为0

    # ========== 补全缺失的 load_data 方法（核心） ==========
    def load_data(self):
        # 检查索引是否越界（避免测试集遍历完后报错）
        if self.index >= self.size:
            raise StopIteration("测试集数据已遍历完毕！")
        
        # 加载当前索引的图像
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)  # 增加batch维度 (1, 3, H, W)

        # 加载当前索引的mask
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)  # 转为numpy数组，方便后续评估

        # 提取图像名称（用于保存结果）
        name = self.images[self.index].split('/')[-1]

        # 索引+1，准备加载下一个样本
        self.index += 1
        
        return image, gt, name

    # ========== 补全缺失的 rgb_loader 方法 ==========
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # ========== 补全缺失的 binary_loader 方法 ==========
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

# 在dataset.py末尾添加（如果没有）
__all__ = ['FullDataset', 'TestDataset', 'Resize', 'ToTensor', 'Normalize', 'RandomHorizontalFlip', 'RandomVerticalFlip']
