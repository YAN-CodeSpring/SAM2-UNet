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
parser = argparse.ArgumentParser("SAM2-UNet-3Class-Baseline")
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
    with open(args.log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,LR\n")
    
    best_loss = float('inf')

    for epoch in range(args.epoch):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
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
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                x, target = batch['image'].to(device), batch['label'].to(device)
                out, out1, out2 = model(x)
                
                v_loss = optimized_structure_loss(out, target) + \
                         0.4 * optimized_structure_loss(out1, target) + \
                         0.4 * optimized_structure_loss(out2, target)
                val_loss += v_loss.item()
                val_pbar.set_postfix({'v_loss': f"{v_loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 终端打印总结，让你更安心
        print(f"\n>>> Epoch [{epoch+1}/{args.epoch}] Summary: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {curr_lr:.6f}")

        # Wandb 可视化
        with torch.no_grad():
            vis_img = denormalize(x[0]).cpu().permute(1, 2, 0).numpy()
            vis_gt = target[0].cpu().squeeze().numpy().astype(np.uint8)
            vis_pred = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.uint8)
            
            wandb.log({
                "train_loss": avg_train, "val_loss": avg_val, "lr": curr_lr,
                "Visuals": wandb.Image(vis_img, masks={
                    "pred": {"mask_data": vis_pred, "class_labels": {0: "BG", 1: "Benign", 2: "Malignant"}},
                    "gt": {"mask_data": vis_gt, "class_labels": {0: "BG", 1: "Benign", 2: "Malignant"}}
                })
            })

        if avg_val < best_loss:
            best_loss = avg_val
            save_name = os.path.join(args.save_path, 'best_model.pth')
            torch.save(model.state_dict(), save_name)
            print(f"🏆 Best model updated and saved to {save_name} (Val Loss: {best_loss:.4f})")
        
        with open(args.log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train:.6f},{avg_val:.6f},{curr_lr:.6f}\n")

    wandb.finish()

if __name__ == "__main__":
    main(args)