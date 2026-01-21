import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet
from tqdm import tqdm  # 已导入，保留

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # 修复弃用警告：把 reduce='none' 改成 reduction='none'
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    # 可选：num_workers=8 若出现卡顿，可临时改为 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path)
    model.to(device)
    # 修复笔误：initia_lr → initial_lr（不影响运行，但修正更规范）
    optim = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], 
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    
    # 训练循环：加入tqdm进度条
    for epoch in range(args.epoch):
        model.train()  # 显式设置训练模式（避免模型意外处于eval模式）
        total_loss = 0.0  # 累计本轮总loss
        # 创建epoch级进度条：显示当前epoch、总epoch、batch进度
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader), 
                    desc=f'Epoch [{epoch+1}/{args.epoch}]',
                    leave=True,  # 保留进度条，不刷屏
                    ncols=120)   # 进度条宽度，适配终端
        
        for i, batch in pbar:
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            
            loss.backward()
            optim.step()
            
            # 累计loss，计算平均loss
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            
            # 更新进度条右侧的实时信息
            pbar.set_postfix({
                'Batch': f'{i+1}/{len(dataloader)}',
                'Cur Loss': f'{loss.item():.6f}',  # 当前batch的loss
                'Avg Loss': f'{avg_loss:.6f}',     # 本轮平均loss
                'LR': f'{optim.param_groups[0]["lr"]:.7f}'  # 当前学习率（可选）
            })
            
            # 保留原有每50个batch打印的逻辑（可选，进度条已显示，也可删除）
            if i % 50 == 0:
                print(f"\nepoch:{epoch+1}-{i+1}: loss:{loss.item():.6f}")
        
        # 本轮epoch结束，更新学习率
        scheduler.step()
        
        # 保存模型（保留原有逻辑）
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            ckpt_path = os.path.join(args.save_path, f'SAM2-UNet-{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'\n[Saving Snapshot:] {ckpt_path}')
        # 打印本轮epoch的平均loss
        print(f'Epoch [{epoch+1}/{args.epoch}] 平均Loss: {total_loss/len(dataloader):.6f}')


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(1024)
    main(args)