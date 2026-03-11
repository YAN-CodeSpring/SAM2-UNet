import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from dataset import FullDataset
from SAM2UNet import SAM2UNet

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser("SAM2-UNet-3Class-Improved")
parser.add_argument("--hiera_path", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True)
parser.add_argument("--train_mask_path", type=str, required=True)
parser.add_argument("--val_image_path", type=str, required=True)
parser.add_argument("--val_mask_path", type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--log_path', type=str, default="train_log.csv")
parser.add_argument("--epoch", type=int, default=25) 
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--wandb_project", type=str, default="Breast-Cancer-3Class")
args = parser.parse_args()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    return torch.clamp(img_tensor * std + mean, 0, 1)

# ==================== 指标计算 ====================
def calculate_dice(pred, mask, num_classes=3):
    """
    计算良性(1)和恶性(2)的 Dice，返回 (dice_benign, dice_malignant)
    """
    pred_softmax = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred_softmax, dim=1)
    target = mask.squeeze(1).long()
    
    dices = []
    for cls in [1, 2]: # 只计算良性和恶性
        p = (pred_labels == cls).float()
        g = (target == cls).float()
        intersection = (p * g).sum()
        union = p.sum() + g.sum()
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        dices.append(dice.item())
    return dices[0], dices[1]

# ==================== 复合损失函数 ====================
def optimized_structure_loss(pred, mask):
    target = mask.squeeze(1).long()
    weights = torch.tensor([0.2, 1.0, 1.2]).to(pred.device)
    ce_loss = F.cross_entropy(pred, target, weight=weights)

    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    
    dice_loss = 0
    foreground_classes = [1, 2] 
    for cls in foreground_classes:
        p = pred_soft[:, cls, :, :]
        g = target_one_hot[:, cls, :, :]
        intersection = (p * g).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        dice = (2. * intersection + 1.0) / (union + 1.0 + 1e-7)
        dice_loss += (1 - dice).mean()
    
    return ce_loss + (dice_loss / len(foreground_classes))

# ==================== 主训练逻辑 ====================
def main(args):    
    set_random_seed(42)
    wandb.init(project=args.wandb_project, config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(FullDataset(args.train_image_path, args.train_mask_path, 512, mode='train'), 
                              batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(FullDataset(args.val_image_path, args.val_mask_path, 512, mode='val'), 
                            batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    model = SAM2UNet(args.hiera_path).to(device)
    
    optimizer = opt.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # 扩展 CSV 表头
    with open(args.log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Train_Mean_Dice,Val_Mean_Dice,Val_Benign_Dice,Val_Malignant_Dice,LR\n")
    
    best_dice = 0.0

    for epoch in range(args.epoch):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_dices = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Train]")
        for batch in train_pbar:
            x, target = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out, out1, out2 = model(x)
            
            loss = optimized_structure_loss(out, target) + \
                   0.4 * optimized_structure_loss(out1, target) + \
                   0.4 * optimized_structure_loss(out2, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 计算训练集 Dice (只取主输出 out)
            d1, d2 = calculate_dice(out, target)
            train_dices.append((d1 + d2) / 2)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_benign_dices = []
        val_malignant_dices = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                x, target = batch['image'].to(device), batch['label'].to(device)
                out, out1, out2 = model(x)
                
                v_loss = optimized_structure_loss(out, target) + \
                         0.4 * optimized_structure_loss(out1, target) + \
                         0.4 * optimized_structure_loss(out2, target)
                val_loss += v_loss.item()
                
                # 计算各类别 Dice
                d1, d2 = calculate_dice(out, target)
                val_benign_dices.append(d1)
                val_malignant_dices.append(d2)
                val_pbar.set_postfix({'v_loss': f"{v_loss.item():.4f}"})

        # 计算平均值
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = np.mean(train_dices)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_benign = np.mean(val_benign_dices)
        avg_val_malignant = np.mean(val_malignant_dices)
        avg_val_mean_dice = (avg_val_benign + avg_val_malignant) / 2
        
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"\n>>> Epoch [{epoch+1}/{args.epoch}] Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    Val Mean Dice: {avg_val_mean_dice:.4f} (Benign: {avg_val_benign:.4f}, Malignant: {avg_val_malignant:.4f})")

        # Wandb 记录更多指标
        with torch.no_grad():
            vis_img = denormalize(x[0]).cpu().permute(1, 2, 0).numpy()
            vis_gt = target[0].cpu().squeeze().numpy().astype(np.uint8)
            vis_pred = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.uint8)
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mean_dice": avg_train_dice,
                "val_mean_dice": avg_val_mean_dice,
                "val_benign_dice": avg_val_benign,
                "val_malignant_dice": avg_val_malignant,
                "lr": curr_lr,
                "Visuals": wandb.Image(vis_img, masks={
                    "pred": {"mask_data": vis_pred, "class_labels": {0: "BG", 1: "Benign", 2: "Malignant"}},
                    "gt": {"mask_data": vis_gt, "class_labels": {0: "BG", 1: "Benign", 2: "Malignant"}}
                })
            })

        # 核心修改：以验证集平均 Dice 为依据更新模型
        if avg_val_mean_dice > best_dice:
            best_dice = avg_val_mean_dice
            save_name = os.path.join(args.save_path, 'best_model.pth')
            torch.save(model.state_dict(), save_name)
            print(f"🌟 Best model updated (Val Mean Dice: {best_dice:.4f})")
        
        # 写入 CSV 日志
        with open(args.log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{avg_train_dice:.6f},"
                    f"{avg_val_mean_dice:.6f},{avg_val_benign:.6f},{avg_val_malignant:.6f},{curr_lr:.6f}\n")

    wandb.finish()

if __name__ == "__main__":
    main(args)