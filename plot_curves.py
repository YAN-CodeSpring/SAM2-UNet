import os
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 忽略无关警告
warnings.filterwarnings('ignore')

# ========== 绘图样式配置（美观+清晰） ==========
def setup_plot_style():
    # 设置中文字体（避免乱码，若无中文字体可注释）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 设置全局样式
    plt.style.use('seaborn-v0_8-whitegrid')
    # 颜色配置（区分train/val）
    colors = {
        'train': '#1f77b4',    # 蓝色（train）
        'val': '#ff7f0e',      # 橙色（val）
        'lr': '#2ca02c'        # 绿色（lr）
    }
    # 线条样式
    linestyles = {
        'loss': '-',           # 实线（loss）
        'iou': '--',           # 虚线（iou）
        'dice': '-.',          # 点划线（dice）
        'lr': ':'              # 点线（lr）
    }
    return colors, linestyles

# ========== 核心绘图函数 ==========
def plot_training_curves(log_path, save_path):
    # 1. 加载并验证数据
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log文件不存在：{log_path}")
    
    # 读取csv
    df = pd.read_csv(log_path)
    # 验证必要列是否存在
    required_cols = ['epoch', 'lr', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Log文件缺少必要列：{missing_cols}")
    
    # 数据类型转换（确保数值正确）
    df['epoch'] = df['epoch'].astype(int)
    for col in required_cols[1:]:
        df[col] = df[col].astype(float)
    
    # 2. 初始化绘图样式
    colors, linestyles = setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2行2列子图
    fig.suptitle('SAM2-UNet Training Curves (BUSI Dataset)', fontsize=18, fontweight='bold')
    
    # 提取x轴（epoch）
    epochs = df['epoch'].values
    
    # 3. 子图1：学习率曲线（左上）
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['lr'].values, color=colors['lr'], linestyle=linestyles['lr'], 
             linewidth=2, marker='o', markersize=4, label='Learning Rate')
    ax1.set_title('Learning Rate vs Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('LR', fontsize=12)
    ax1.set_yscale('log')  # 对数刻度，更易观察lr变化
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='both', labelsize=10)
    
    # 4. 子图2：损失曲线（右上）
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['train_loss'].values, color=colors['train'], linestyle=linestyles['loss'],
             linewidth=2, marker='s', markersize=4, label='Train Loss')
    ax2.plot(epochs, df['val_loss'].values, color=colors['val'], linestyle=linestyles['loss'],
             linewidth=2, marker='s', markersize=4, label='Val Loss')
    ax2.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    # 标注最低验证损失
    min_val_loss_idx = df['val_loss'].idxmin()
    ax2.annotate(f'Min: {df["val_loss"].min():.4f}', 
                 xy=(df['epoch'][min_val_loss_idx], df['val_loss'].min()),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 fontsize=10)
    
    # 5. 子图3：IoU曲线（左下）
    ax3 = axes[1, 0]
    ax3.plot(epochs, df['train_iou'].values, color=colors['train'], linestyle=linestyles['iou'],
             linewidth=2, marker='^', markersize=4, label='Train IoU')
    ax3.plot(epochs, df['val_iou'].values, color=colors['val'], linestyle=linestyles['iou'],
             linewidth=2, marker='^', markersize=4, label='Val IoU')
    ax3.set_title('IoU vs Epoch', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('IoU', fontsize=12)
    ax3.set_ylim(0, 1)  # IoU范围0-1
    ax3.legend(fontsize=10)
    ax3.tick_params(axis='both', labelsize=10)
    # 标注最高验证IoU
    max_val_iou_idx = df['val_iou'].idxmax()
    ax3.annotate(f'Max: {df["val_iou"].max():.4f}', 
                 xy=(df['epoch'][max_val_iou_idx], df['val_iou'].max()),
                 xytext=(10, -10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.5),
                 fontsize=10)
    
    # 6. 子图4：Dice曲线（右下）
    ax4 = axes[1, 1]
    ax4.plot(epochs, df['train_dice'].values, color=colors['train'], linestyle=linestyles['dice'],
             linewidth=2, marker='*', markersize=4, label='Train Dice')
    ax4.plot(epochs, df['val_dice'].values, color=colors['val'], linestyle=linestyles['dice'],
             linewidth=2, marker='*', markersize=4, label='Val Dice')
    ax4.set_title('Dice vs Epoch', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Dice', fontsize=12)
    ax4.set_ylim(0, 1)  # Dice范围0-1
    ax4.legend(fontsize=10)
    ax4.tick_params(axis='both', labelsize=10)
    # 标注最高验证Dice
    max_val_dice_idx = df['val_dice'].idxmax()
    ax4.annotate(f'Max: {df["val_dice"].max():.4f}', 
                 xy=(df['epoch'][max_val_dice_idx], df['val_dice'].max()),
                 xytext=(10, -10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.5),
                 fontsize=10)
    
    # 7. 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出标题空间
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 高清保存（dpi=300）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化图像已保存至：{save_path}")

# ========== 命令行参数解析 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Curves Visualization")
    parser.add_argument("--log_path", type=str, required=True,
                        help="Path to train_log.csv (e.g., ./output_checkpoints/train_log.csv)")
    parser.add_argument("--save_path", type=str, default="./output_checkpoints/training_curves.png",
                        help="Path to save the plot (e.g., ./output_checkpoints/training_curves.png)")
    args = parser.parse_args()
    
    # 执行绘图
    try:
        plot_training_curves(args.log_path, args.save_path)
    except Exception as e:
        print(f"❌ 绘图失败：{str(e)}")
        exit(1)