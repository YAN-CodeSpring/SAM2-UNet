import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from dataset import FullDataset
from SAM2UNet import SAM2UNet

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True)
parser.add_argument("--train_mask_path", type=str, required=True)
parser.add_argument("--val_image_path", type=str, required=True)
parser.add_argument("--val_mask_path", type=str, required=True)

parser.add_argument('--save_path', type=str, required=True)
# [恢复] 用于保存 CSV 日志的路径
parser.add_argument('--log_path', type=str, required=True, help="path to the csv log file")

parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--wandb_project", type=str, default="BCSS-SAM2-UNet-Train")
args = parser.parse_args()


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    # [修复] 加上 1e-6 防止分母为0导致 Loss 爆炸
    wiou = 1 - (inter + 1)/(union - inter + 1 + 1e-6)
    return (wbce + wiou).mean()


def main(args):    
    wandb.init(project=args.wandb_project, config=args)
    device = torch.device("cuda")
    
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 224, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 224, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    model = SAM2UNet(args.hiera_path).to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # [恢复] 初始化 CSV 文件并写入表头
    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(args.log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Learning_Rate\n")
    
    print(f"🚀 开始训练! 训练集: {len(train_dataset)}张, 验证集: {len(val_dataset)}张")
    print(f"📊 日志将保存至: {args.log_path}")

    best_val_loss = float('inf') 

    for epoch in range(args.epoch):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss_epoch = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Train]")
        for batch in train_pbar:
            x, target = batch['image'].to(device), batch['label'].to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            
            train_loss_epoch += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss_epoch / len(train_loader)
        current_lr = optim.param_groups[0]['lr']
        scheduler.step()

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss_epoch = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Val  ]")
        with torch.no_grad():
            for batch in val_pbar:
                x, target = batch['image'].to(device), batch['label'].to(device)
                pred0, pred1, pred2 = model(x)
                loss0 = structure_loss(pred0, target)
                loss1 = structure_loss(pred1, target)
                loss2 = structure_loss(pred2, target)
                loss = loss0 + loss1 + loss2
                
                val_loss_epoch += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss_epoch / len(val_loader)
        print(f"✅ Epoch {epoch+1} 结束: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # ==================== 日志与保存权重 ====================
        # [恢复] 将数据写入 CSV 文件
        with open(args.log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{current_lr:.6f}\n")

        # 记录到 wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "learning_rate": current_lr
        })

        # 可视化验证集的一张图片作为预览
        with torch.no_grad():
            vis_img = denormalize(x[0]).cpu().permute(1, 2, 0).numpy()
            vis_gt = target[0].cpu().squeeze().numpy()
            vis_pred = (torch.sigmoid(pred0[0]).cpu().squeeze().numpy() > 0.5).astype(np.uint8)
            wandb.log({
                "Validation Visuals": wandb.Image(vis_img, masks={
                    "predictions": {"mask_data": vis_pred, "class_labels": {0: "BG", 1: "Tumor"}},
                    "ground_truth": {"mask_data": vis_gt.astype(np.uint8), "class_labels": {0: "BG", 1: "Tumor"}}
                })
            })

        # 保存 Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.save_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 新的最佳验证集 Loss: {best_val_loss:.4f}, 已保存为 best_model.pth")

        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))

    wandb.finish()


if __name__ == "__main__":
    main(args)