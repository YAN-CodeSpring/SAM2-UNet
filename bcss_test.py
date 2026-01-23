import argparse
import os
import random
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from bcss_SAM2UNet import SAM2UNet
from bcss_dataset import BCSSTestDataset  # 直接导入你已有的BCSSTestDataset类

# ========== 固定随机种子（保证抽样可复现） ==========
def seed_everything(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========== 抽样函数：筛选出要测试的文件名列表 ==========
def sample_val_files(image_root, sample_num=1000, seed=1024):
    """
    从val集抽取指定数量的文件名，返回抽样后的文件名列表
    :param image_root: val图像路径
    :param sample_num: 抽取样本数
    :param seed: 随机种子
    :return: 抽样后的文件名列表（仅包含.png文件）
    """
    # 获取val集所有png文件
    all_names = [f for f in os.listdir(image_root) if f.endswith('.png')]
    # 抽样（若总数<sample_num则取全部）
    random.seed(seed)
    sample_names = random.sample(all_names, min(sample_num, len(all_names)))
    return sample_names

# ========== 重写BCSSTestDataset的文件加载逻辑（仅加载抽样文件） ==========
def get_sampled_test_dataset(image_root, mask_root, size, sample_names):
    """
    创建仅加载抽样文件的BCSSTestDataset实例
    :param image_root: val图像路径
    :param mask_root: val mask路径
    :param size: 图像尺寸
    :param sample_names: 抽样后的文件名列表
    :return: 定制化的BCSSTestDataset
    """
    # 初始化原始类
    test_dataset = BCSSTestDataset(image_root, mask_root, size)
    # 筛选仅保留抽样的文件
    test_dataset.image_names = [name for name in test_dataset.image_names if name in sample_names]
    test_dataset.image_paths = [os.path.join(image_root, name) for name in test_dataset.image_names]
    if test_dataset.has_mask:
        test_dataset.mask_paths = [os.path.join(mask_root, name) for name in test_dataset.image_names]
    # 更新数据集大小
    test_dataset.size = len(test_dataset.image_paths)
    test_dataset.index = 0  # 重置索引
    print(f"✅ 抽样后BCSS测试集加载完成：{test_dataset.size} 张图像")
    return test_dataset

# ========== 主测试逻辑 ==========
def main(args):
    # 1. 初始化设备和随机种子
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)
    
    # 2. 从val集抽样文件名（核心：控制测试样本数量）
    print(f"开始从val集抽取 {args.sample_num} 张样本...")
    sample_names = sample_val_files(args.val_image_path, args.sample_num, args.seed)
    # 保存抽样列表（方便后续eval复用）
    sample_list_path = os.path.join(args.save_path, "sampled_val_list.txt")
    with open(sample_list_path, 'w', encoding='utf-8') as f:
        for name in sample_names:
            f.write(f"{name}\n")
    print(f"✅ 抽样完成，共抽取 {len(sample_names)} 张样本，列表保存至：{sample_list_path}")
    
    # 3. 加载抽样后的测试集（复用你的BCSSTestDataset类）
    test_dataset = get_sampled_test_dataset(
        args.val_image_path, args.val_mask_path, args.size, sample_names
    )
    
    # 4. 加载模型（适配22类多分类）
    print(f"加载模型权重：{args.checkpoint}")
    model = SAM2UNet(checkpoint_path=None, num_classes=22).to(device)  # 22类多分类
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()
    
    # 5. 逐样本推理并保存预测mask
    print("开始推理并保存预测结果...")
    processed_num = 0
    while True:
        try:
            # 复用你的load_data方法加载数据
            image, gt, name = test_dataset.load_data()
            processed_num += 1
            
            # 模型推理（多分类逻辑）
            with torch.no_grad():
                image = image.to(device)  # image已由load_data处理为[1,3,H,W]
                pred, _, _ = model(image)  # 输出：[1,22,H,W]
                
                # 上采样到原始mask尺寸（gt是np.array，shape为(H,W)）
                if gt is not None:
                    pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)
                # 多分类：argmax取每个像素的类别（0-21）
                pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()  # [H,W]，值为0-21
            
            # 保存预测mask（单通道，像素值0-21，兼容多分类指标计算）
            save_name = name[:-4] + "_pred.png"  # 命名：xxx_pred.png
            save_path = os.path.join(args.save_path, save_name)
            imageio.imsave(save_path, pred_mask.astype(np.uint8))  # 保存为uint8（0-21）
            
            # 打印进度
            if processed_num % 50 == 0:
                print(f"已处理 {processed_num}/{test_dataset.size} 张，当前保存：{save_name}")
        
        except StopIteration:
            # 遍历完所有样本
            break
    
    print(f"\n🎉 测试完成！")
    print(f"📁 预测mask保存路径：{args.save_path}")
    print(f"📜 抽样列表路径：{sample_list_path}")
    print(f"📊 共处理样本数：{processed_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BCSS SAM2-UNet Test (多分类) - 复用BCSSTestDataset")
    # 核心参数
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重文件路径（.pth）")
    parser.add_argument("--val_image_path", type=str, required=True,
                        help="BCSS val集图像路径（用于抽样，替代无mask的test集）")
    parser.add_argument("--val_mask_path", type=str, required=True,
                        help="BCSS val集mask路径（用于抽样）")
    parser.add_argument("--save_path", type=str, required=True,
                        help="预测mask保存路径")
    # 抽样参数
    parser.add_argument("--sample_num", type=int, default=1000,
                        help="从val集抽取的样本数（默认1000）")
    parser.add_argument("--seed", type=int, default=1024,
                        help="随机种子（保证抽样可复现）")
    # 模型适配参数
    parser.add_argument("--size", type=int, default=224,
                        help="图像输入尺寸（BCSS=224）")
    
    args = parser.parse_args()
    main(args)