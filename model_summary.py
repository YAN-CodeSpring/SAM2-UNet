import torch
from torchinfo import summary
from bcss_SAM2UNet import SAM2UNet  # 导入你的模型结构
# from SAM2UNet import SAM2UNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前设备: {device}")

    # ==========================
    # 👇 这里切换你的任务 (1, 2, 3, 4, 5)
    task_id = 5
    # ==========================

    # 预设的四个任务配置
    tasks = {
        1: {"desc": "512x512图像, 22分类", "size": 512, "classes": 22},
        2: {"desc": "512x512图像,  5分类", "size": 512, "classes": 5},
        3: {"desc": "224x224图像,  3分类", "size": 224, "classes": 22}, # 这里也有可能是 22
        4: {"desc": "224x224图像,  2分类", "size": 224, "classes": 2}, # 1, 或改成2，取决于你的损失函数
        5: {"desc": "BUSI 二值分割 (训练尺寸 352x352)", "size": 352, "classes": 2}, # 这就是BUSI数据集的
    }
    
    config = tasks[task_id]
    print(f"\n🚀 正在评估任务 {task_id}: {config['desc']}")

    # 1. 实例化对应任务的模型
    model = SAM2UNet(checkpoint_path=None, num_classes=config['classes']).to(device)

    # 2. 打印详细的 Summary 表格
    print("="*80)
    summary(
        model, 
        input_size=(1, 3, config['size'], config['size']),  # (Batch, Channel, H, W)
        col_names=["input_size", "output_size", "num_params", "mult_adds"], # 新增 mult_adds 查看计算量(FLOPs)
        col_width=18,
        row_settings=["var_names"]
    )

if __name__ == "__main__":
    main()