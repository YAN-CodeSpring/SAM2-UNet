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

parser = argparse.ArgumentParser("SAM2-UNet-3Class")
parser.add_argument("--hiera_path", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True)
parser.add_argument("--train_mask_path", type=str, required=True)
parser.add_argument("--val_image_path", type=str, required=True)
parser.add_argument("--val_mask_path", type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--log_path', type=str, default="train_log.csv")
parser.add_argument("--epoch", type=int, default=25) # 建议25
parser.add_argument("--lr", type=float, default=0.001)
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

def structure_loss_multiclass(pred, mask):
    """
    针对三分类的深度学习损失函数
    pred: [B, 3, 512, 512]
    mask: [B, 1, 512, 512] 映射值为 0, 1, 2
    """
    target = mask.squeeze(1).long()
    
    # 1. CrossEntropy Loss
    ce_loss = F.cross_entropy(pred, target)
    
    # 2. Multiclass IoU Loss
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    
    inter = (pred_soft * target_one_hot).sum(dim=(2, 3))
    union = (pred_soft + target_one_hot).sum(dim=(2, 3))
    iou_loss = 1 - (inter + 1.0) / (union - inter + 1.0 + 1e-6)
    
    return ce_loss + iou_loss.mean()

def main(args):    
    set_random_seed(42)
    wandb.init(project=args.wandb_project, config=args)
    device = torch.device("cuda")
    
    # 加载数据集，尺寸改为 512
    train_loader = DataLoader(FullDataset(args.train_image_path, args.train_mask_path, 512, mode='train'), 
                              batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(FullDataset(args.val_image_path, args.val_mask_path, 512, mode='val'), 
                            batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    model = SAM2UNet(args.hiera_path).to(device)
    
    # 只为带有 Adapter 的层和 Decoder 开启梯度 (遵循原模型冻结 Encoder Trunk 的逻辑)
    optimizer = opt.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    os.makedirs(args.save_path, exist_ok=True)
    with open(args.log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,LR\n")
    
    best_loss = float('inf')

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Train]")
        for batch in pbar:
            x, target = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            
            # 模型返回 out, out1, out2
            out, out1, out2 = model(x)
            
            # 多尺度深监督损失
            loss = structure_loss_multiclass(out, target) + \
                   structure_loss_multiclass(out1, target) + \
                   structure_loss_multiclass(out2, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, target = batch['image'].to(device), batch['label'].to(device)
                out, out1, out2 = model(x)
                loss = structure_loss_multiclass(out, target) + \
                       structure_loss_multiclass(out1, target) + \
                       structure_loss_multiclass(out2, target)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Wandb 可视化逻辑
        with torch.no_grad():
            vis_img = denormalize(x[0]).cpu().permute(1, 2, 0).numpy()
            vis_gt = target[0].cpu().squeeze().numpy().astype(np.uint8)
            # 使用 argmax 获得预测索引 (0, 1, 2)
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
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
        
        with open(args.log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train:.6f},{avg_val:.6f},{curr_lr:.6f}\n")

    wandb.finish()

if __name__ == "__main__":
    main(args)