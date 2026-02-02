import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from bcss_dataset import FullDataset, BCSSFullDataset  # 新增BCSS数据集类
from bcss_SAM2UNet import SAM2UNet ###这里改动了
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from dataset import Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip

# ========== 新增：多分类指标计算（兼容二值） ==========
def calculate_metrics(pred, target, dataset_type="busi"):
    """
    兼容二值/多分类的IoU/Dice计算
    :param pred: 模型输出（busi: [B,1,H,W] | bcss: [B,22,H,W]）
    :param target: 标签（busi: float [0/1] | bcss: long [0-21]）
    :param dataset_type: busi/bcss
    :return: 平均IoU、平均Dice
    """
    if dataset_type == "busi":
        # 原有二值分割逻辑（完全保留）
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        target = target.float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)
        
    elif dataset_type == "bcss":
        # BCSS多分类逻辑（22类）
        pred = torch.argmax(pred, dim=1)  # [B,H,W]（预测类别）
        target = target.squeeze(1)        # [B,H,W]（原始标签）
        
        iou_per_class = []
        dice_per_class = []
        # 遍历22类（0-21），跳过无样本的类别
        for cls in range(22):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
            
            # 仅计算有样本的类别
            if union.sum() > 0:
                iou = (intersection + 1e-6) / (union + 1e-6)
                dice = (2 * intersection + 1e-6) / (pred_cls.sum(dim=(1,2)) + target_cls.sum(dim=(1,2)) + 1e-6)
                iou_per_class.append(iou.mean().item())
                dice_per_class.append(dice.mean().item())
        
        # 宏观平均IoU/Dice
        iou = np.mean(iou_per_class) if iou_per_class else 0.0
        dice = np.mean(dice_per_class) if dice_per_class else 0.0
    
    return iou, dice

# ========== 原有损失函数（保留，BUSI专用） ==========
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

# ========== 验证函数（兼容多分类） ==========
@torch.no_grad()
def validate(model, val_dataloader, criterion, device, dataset_type="busi"):
    model.eval()
    val_loss_total = 0.0
    val_iou_total = 0.0
    val_dice_total = 0.0
    val_pbar = tqdm(val_dataloader, desc='Validating', leave=False)
    
    for batch in val_pbar:
        x = batch['image'].to(device)
        target = batch['label'].to(device)
        
        pred0, pred1, pred2 = model(x)
        
        # 损失计算分支
        if dataset_type == "busi":
            loss = criterion(pred0, target) + criterion(pred1, target) + criterion(pred2, target)
        elif dataset_type == "bcss":
            # 多分类损失：target需为[B,H,W] long类型
            target = target.squeeze(1)  # 去掉channel维度 [B,1,H,W] → [B,H,W]
            loss = criterion(pred0, target) + criterion(pred1, target) + criterion(pred2, target)
        
        # 计算指标
        iou, dice = calculate_metrics(pred0, target, dataset_type)
        
        val_loss_total += loss.item()
        val_iou_total += iou
        val_dice_total += dice
        
        val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}', 'Val IoU': f'{iou:.4f}'})
    
    val_loss_avg = val_loss_total / len(val_dataloader)
    val_iou_avg = val_iou_total / len(val_dataloader)
    val_dice_avg = val_dice_total / len(val_dataloader)
    
    model.train()
    return val_loss_avg, val_iou_avg, val_dice_avg

# ========== CSV初始化/写入（保留） ==========
def init_csv(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice'])

def write_csv(log_path, row_data):
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

# ========== 主函数（核心逻辑保留，新增BCSS分支） ==========
def main(args):    
    # ===== 1. 数据集加载分支（BUSI/BCSS） =====
    if args.dataset_type == "busi":
        # 原有BUSI逻辑（完全保留）
        full_dataset = FullDataset(
            image_root=args.train_image_path,
            gt_root=args.train_mask_path,
            size=args.size, 
            mode='train'
        )
        # 划分训练/验证集
        train_size = int(args.train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(1024)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        # 验证集取消增强
        val_dataset.dataset.transform = transforms.Compose([
            Resize((args.size, args.size)),
            ToTensor(),
            Normalize()
        ])
        # 损失函数：原有structure_loss
        criterion = structure_loss
        # 模型输出通道数：1（二值）
        num_classes = 1
        
    elif args.dataset_type == "bcss":
        # BCSS多分类逻辑（新增）
        # 加载BCSS训练集（无需划分，用独立val集）
        train_dataset = BCSSFullDataset(
            image_root=args.train_image_path,
            mask_root=args.train_mask_path,
            size=args.size,
            mode='train'
        )
        # 加载BCSS独立验证集（新增参数）
        val_dataset = BCSSFullDataset(
            image_root=args.val_image_path,
            mask_root=args.val_mask_path,
            size=args.size,
            mode='val'
        )
        # 损失函数：多分类交叉熵
        criterion = torch.nn.CrossEntropyLoss()
        # 模型输出通道数：22（多分类）
        num_classes = 22
    
    # ===== 2. 数据加载（原有逻辑） =====
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # ===== 3. 模型初始化（适配输出通道数） =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 假设SAM2UNet支持num_classes参数（若不支持，需修改SAM2UNet最后一层输出通道）
    model = SAM2UNet(args.hiera_path, num_classes=num_classes)
    model.to(device)
    
    # ===== 4. 优化器/调度器（原有逻辑） =====
    optim = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], 
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    
    # ===== 5. 日志初始化（原有逻辑） =====
    init_csv(args.log_path)
    os.makedirs(args.save_path, exist_ok=True)
    
    # ===== 6. 训练循环（核心逻辑保留，兼容多分类） =====
    for epoch in range(args.epoch):
        model.train()
        train_loss_total = 0.0
        train_iou_total = 0.0
        train_dice_total = 0.0
        pbar = tqdm(enumerate(train_dataloader), 
                    total=len(train_dataloader), 
                    desc=f'Epoch [{epoch+1}/{args.epoch}]',
                    leave=True, ncols=120)
        
        for i, batch in pbar:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            
            # 损失计算分支
            if args.dataset_type == "busi":
                loss0 = criterion(pred0, target)
                loss1 = criterion(pred1, target)
                loss2 = criterion(pred2, target)
            elif args.dataset_type == "bcss":
                # 多分类：target需为[B,H,W] long类型
                target = target.squeeze(1)  # [B,1,H,W] → [B,H,W]
                loss0 = criterion(pred0, target)
                loss1 = criterion(pred1, target)
                loss2 = criterion(pred2, target)
            
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            
            # 计算指标
            iou, dice = calculate_metrics(pred0, target, args.dataset_type)
            
            # 累计指标
            train_loss_total += loss.item()
            train_iou_total += iou
            train_dice_total += dice
            
            # 更新进度条
            current_lr = optim.param_groups[0]['lr']
            avg_loss = train_loss_total / (i + 1)
            avg_iou = train_iou_total / (i + 1)
            avg_dice = train_dice_total / (i + 1)
            pbar.set_postfix({
                'LR': f'{current_lr:.7f}',
                'Cur Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{avg_loss:.6f}',
                'Avg IoU': f'{avg_iou:.4f}',
                'Avg Dice': f'{avg_dice:.4f}'
            })
        
        # 本轮训练指标
        train_loss_avg = train_loss_total / len(train_dataloader)
        train_iou_avg = train_iou_total / len(train_dataloader)
        train_dice_avg = train_dice_total / len(train_dataloader)
        current_lr = optim.param_groups[0]['lr']
        
        # 验证集评估
        val_loss_avg, val_iou_avg, val_dice_avg = validate(model, val_dataloader, criterion, device, args.dataset_type)
        
        # 控制台输出（原有格式）
        print(f'\n===== Epoch {epoch+1} Summary =====')
        print(f'Train Loss: {train_loss_avg:.4f} | Train IoU: {train_iou_avg:.4f} | Train Dice: {train_dice_avg:.4f}')
        print(f'Val Loss:   {val_loss_avg:.4f} | Val IoU:   {val_iou_avg:.4f} | Val Dice: {val_dice_avg:.4f}')
        print(f'Current LR: {current_lr:.7f}')
        print('='*60)
        
        # 写入CSV（原有逻辑）
        row_data = [
            epoch+1,                # epoch
            round(current_lr, 7),   # lr
            round(train_loss_avg, 6),# train_loss
            round(train_iou_avg, 6), # train_iou
            round(train_dice_avg, 6),# train_dice
            round(val_loss_avg, 6),  # val_loss
            round(val_iou_avg, 6),   # val_iou
            round(val_dice_avg, 6)   # val_dice
        ]
        write_csv(args.log_path, row_data)
        
        # 学习率调度/模型保存（原有逻辑）
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            ckpt_path = os.path.join(args.save_path, f'SAM2-UNet-{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'\n[Saving Snapshot:] {ckpt_path}')

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ========== 命令行参数（新增BCSS相关参数） ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM2-UNet")
    # 原有参数（保留）
    parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
    parser.add_argument("--train_image_path", type=str, required=True, help="path to train images")
    parser.add_argument("--train_mask_path", type=str, required=True, help="path to train masks")
    parser.add_argument('--save_path', type=str, required=True, help="path to store the checkpoint")
    parser.add_argument('--log_path', type=str, required=True, help="path to save log.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="ratio of training set (for BUSI)")
    parser.add_argument("--epoch", type=int, default=20, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    
    # 新增参数（BCSS适配）
    parser.add_argument("--dataset_type", type=str, default="busi", choices=["busi", "bcss"], help="dataset type (busi/bcss)")
    parser.add_argument("--size", type=int, default=352, help="image resize size (BCSS=224, BUSI=352)")
    parser.add_argument("--val_image_path", type=str, default=None, help="BCSS val images path")
    parser.add_argument("--val_mask_path", type=str, default=None, help="BCSS val masks path")
    
    args = parser.parse_args()
    seed_torch(1024)
    main(args)