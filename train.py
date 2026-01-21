import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split  # 新增：random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset  # 原有类完全不动
from SAM2UNet import SAM2UNet
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')
# 新增：导入torchvision的transforms（解决Compose未定义）
from torchvision import transforms
# 确保导入dataset中的自定义transform类
from dataset import Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                        help="path to the sam2 pretrained hiera")
# 关键修改：合并为一个图像/mask路径（全部数据）
parser.add_argument("--train_image_path", type=str, required=True, 
                        help="path to ALL images (train+val)")
parser.add_argument("--train_mask_path", type=str, required=True,
                        help="path to ALL masks (train+val)")
parser.add_argument('--save_path', type=str, required=True,
                        help="path to store the checkpoint")
parser.add_argument('--log_path', type=str, required=True,
                        help="path to save log.csv (e.g., ./logs/train_log.csv)")
    # 新增：训练集比例（默认0.8）
parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="ratio of training set (0.8 = 80% train, 20% val)")
    # 原有参数完全不动
parser.add_argument("--epoch", type=int, default=20, 
                        help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


# ========== 1. 指标计算函数（补充train_dice，原有逻辑不变） ==========
def calculate_metrics(pred, target):
    """计算单批次的IoU和Dice（训练/验证通用）"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()
    
    # IoU
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice（补充：训练集也要算dice）
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)
    
    return iou.mean().item(), dice.mean().item()

# ========== 2. 原有损失函数（完全不动） ==========
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

# ========== 3. 验证函数（补充val_dice，原有逻辑不变） ==========
@torch.no_grad()
def validate(model, val_dataloader, criterion, device):
    model.eval()
    val_loss_total = 0.0
    val_iou_total = 0.0
    val_dice_total = 0.0
    val_pbar = tqdm(val_dataloader, desc='Validating', leave=False)
    
    for batch in val_pbar:
        x = batch['image'].to(device)
        target = batch['label'].to(device)
        
        pred0, pred1, pred2 = model(x)
        loss = criterion(pred0, target) + criterion(pred1, target) + criterion(pred2, target)
        
        # 计算val_iou + val_dice
        iou, dice = calculate_metrics(pred0, target)
        
        val_loss_total += loss.item()
        val_iou_total += iou
        val_dice_total += dice
        
        val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}', 'Val IoU': f'{iou:.4f}'})
    
    val_loss_avg = val_loss_total / len(val_dataloader)
    val_iou_avg = val_iou_total / len(val_dataloader)
    val_dice_avg = val_dice_total / len(val_dataloader)
    
    model.train()
    return val_loss_avg, val_iou_avg, val_dice_avg

# ========== 4. CSV初始化/写入（调整表头为你要求的字段） ==========
def init_csv(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 表头：严格按你要求的顺序
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice'])

def write_csv(log_path, row_data):
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

# ========== 5. 主函数（核心修改：自动划分训练/验证集，原有训练逻辑完全不动） ==========
def main(args):    
    # ===== 关键修改1：加载完整数据集，自动划分 =====
    # 加载全部数据（不手动划分，用random_split自动分）
    full_dataset = FullDataset(
        image_root=args.train_image_path,       # 原训练集路径（现在是全部数据）
        gt_root=args.train_mask_path,           # 原mask路径（现在是全部数据）
        size=352, 
        mode='train'                      # 训练集用train增强，验证集后续改mode
    )
    
    # 按比例划分：train_ratio=0.8（可通过参数调整）
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # 固定随机种子，划分结果可复现
    generator = torch.Generator().manual_seed(1024)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # 关键：验证集修改transform（取消数据增强）
    val_dataset.dataset.transform = transforms.Compose([
        Resize((352, 352)),
        ToTensor(),
        Normalize()
    ])
    
    # ===== 数据加载（原有逻辑不变） =====
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # ===== 模型/优化器（原有逻辑完全不动） =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet(args.hiera_path)
    model.to(device)
    
    optim = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], 
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    
    # ===== 初始化日志（原有逻辑） =====
    init_csv(args.log_path)
    os.makedirs(args.save_path, exist_ok=True)
    
    # ===== 训练循环（核心逻辑完全不动，仅补充train_dice计算） =====
    for epoch in range(args.epoch):
        model.train()
        train_loss_total = 0.0
        train_iou_total = 0.0
        train_dice_total = 0.0  # 新增：累计train_dice
        pbar = tqdm(enumerate(train_dataloader), 
                    total=len(train_dataloader), 
                    desc=f'Epoch [{epoch+1}/{args.epoch}]',
                    leave=True, ncols=120)
        
        for i, batch in pbar:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            
            loss.backward()
            optim.step()
            
            # 计算train_iou + train_dice（补充train_dice）
            iou, dice = calculate_metrics(pred0, target)
            
            # 累计训练集指标
            train_loss_total += loss.item()
            train_iou_total += iou
            train_dice_total += dice
            
            # 更新进度条
            current_lr = optim.param_groups[0]['lr']
            avg_loss = train_loss_total / (i + 1)
            avg_iou = train_iou_total / (i + 1)
            avg_dice = train_dice_total / (i + 1)  # 新增：train_dice均值
            pbar.set_postfix({
                'LR': f'{current_lr:.7f}',
                'Cur Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{avg_loss:.6f}',
                'Avg IoU': f'{avg_iou:.4f}',
                'Avg Dice': f'{avg_dice:.4f}'  # 新增：显示train_dice
            })
        
        # 本轮训练集指标汇总
        train_loss_avg = train_loss_total / len(train_dataloader)
        train_iou_avg = train_iou_total / len(train_dataloader)
        train_dice_avg = train_dice_total / len(train_dataloader)  # 新增：train_dice均值
        current_lr = optim.param_groups[0]['lr']
        
        # 验证集评估（原有逻辑）
        val_loss_avg, val_iou_avg, val_dice_avg = validate(model, val_dataloader, structure_loss, device)
        
        # ===== 控制台输出（按你要求的字段） =====
        print(f'\n===== Epoch {epoch+1} Summary =====')
        print(f'Train Loss: {train_loss_avg:.4f} | Train IoU: {train_iou_avg:.4f} | Train Dice: {train_dice_avg:.4f}')
        print(f'Val Loss:   {val_loss_avg:.4f} | Val IoU:   {val_iou_avg:.4f} | Val Dice: {val_dice_avg:.4f}')
        print(f'Current LR: {current_lr:.7f}')
        print('='*60)
        
        # ===== 写入CSV（严格按你要求的字段顺序） =====
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
        
        # ===== 学习率调度/模型保存（原有逻辑完全不动） =====
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

# ========== 6. 命令行参数（简化：去掉手动验证集路径，新增划分比例） ==========
if __name__ == "__main__":
    seed_torch(1024)
    main(args)