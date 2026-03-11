import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 依赖模块 (SwinBlock) - 为了让CATM能跑起来
# ==========================================
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_norm = self.norm(x_flat)
        # 注意：这里的attn其实是简化的Self-Attention，源码中并未实现完整的Swin Window Shift机制
        # 但作为特征提取模块，标准的MultiheadAttention已经足够强力
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(x_flat)
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        return x

# ==========================================
# 2. 核心模块: CATM (Cross-Attention Transformer Module)
# ==========================================
class CATM(nn.Module):
    """
    Cross-Attention Transformer Module
    用于替代 Skip Connection 中的简单 Concat。
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        # 对 Decoder 特征进行增强
        self.swin = SwinTransformerBlock(dim, num_heads)
        
        # 投影层，用于计算 Query, Key, Value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 核心：交叉注意力
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # 空间注意力融合 (SharedSA)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),  # 融合 原始+平均池化+最大池化
            nn.Sigmoid()  # 生成空间权重图
        )

    def forward_shared_sa(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared Spatial Attention: 结合 AvgPool 和 MaxPool 增强特征
        """
        avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, 1)).expand_as(x)
        max_pool = F.adaptive_max_pool2d(x, output_size=(1, 1))[0].expand_as(x) # [0] because max_pool returns (val, idx)
        # 拼接三个维度的特征
        pooled = torch.cat([x, avg_pool, max_pool], dim=1)
        # 生成注意力权重
        attention = self.fuse(pooled)
        return attention * x

    def forward(self, x_skip: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_skip: 来自 SAM2 Encoder 的特征 (Skip Connection) -> [B, C, H, W]
            x_decoder: 来自 Decoder 上一层的特征 -> [B, C, H, W]
        """
        b, c, h, w = x_skip.shape
        
        # 1. 先对 Decoder 特征做一次 Swin Block 增强
        dec_feat = self.swin(x_decoder)
        
        # 2. 展平以进行 Attention 计算
        skip_seq = x_skip.flatten(2).transpose(1, 2)    # (B, HW, C)
        dec_seq = dec_feat.flatten(2).transpose(1, 2)   # (B, HW, C)

        # 3. 生成 Q, K, V
        # Q 来自 Encoder (Skip) -- 这里的逻辑是：我想看看 Encoder 里有哪些东西能匹配上 Decoder
        # *注：原代码逻辑是用 Skip 做 Q，Decoder 做 K/V。
        # 但在一般 Cross-Attention 逻辑中，通常用 Decoder 做 Q (查询)，Encoder 做 K/V (库)。
        # 不过根据原代码逻辑，它是用 Skip (Q) 去找 Decoder (K) 的关联。
        # 既然是拿来主义，我们先保持原代码逻辑，或者你可以尝试反过来。
        q = self.q_proj(skip_seq) 
        k = self.k_proj(dec_seq)
        v = self.v_proj(dec_seq)

        # 4. Cross Attention
        attn_out, _ = self.cross_attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        
        # 5. 残差连接 + 空间注意力融合
        # 将 Attention 的结果加回到 Skip 特征上，然后通过 SA 模块筛选重要区域
        fused = self.forward_shared_sa(attn_out + x_skip)
        
        return fused

# ==========================================
# 3. 测试脚本 (模拟你的 SAM2-UNet 场景)
# ==========================================
if __name__ == "__main__":
    # 模拟设置
    BATCH_SIZE = 64
    IMG_SIZE = 224
    
    # 假设我们在 Decoder 的第 2 层进行融合
    # 此时特征图大小可能是输入尺寸的 1/4 (56x56) 或 1/8 (28x28)
    H, W = 56, 56 
    CHANNELS = 128 # 假设这一层的通道数
    
    # 1. 创建模块
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catm_module = CATM(dim=CHANNELS, num_heads=4).to(device)
    
    print(f"Running test on device: {device}")
    
    # 2. 模拟输入数据
    # x_skip: 来自 SAM2 Encoder 的特征
    x_skip = torch.randn(BATCH_SIZE, CHANNELS, H, W).to(device)
    
    # x_decoder: 来自 Decoder 下层上采样上来的特征
    # (通常 Decoder 特征会先 Upsample 到和 Skip 一样的 H, W)
    x_decoder = torch.randn(BATCH_SIZE, CHANNELS, H, W).to(device)
    
    print(f"Input Skip Shape:    {x_skip.shape}")
    print(f"Input Decoder Shape: {x_decoder.shape}")
    
    # 3. 前向传播
    try:
        output = catm_module(x_skip, x_decoder)
        print("-" * 30)
        print(f"Output Shape:        {output.shape}")
        print("-" * 30)
        
        if output.shape == x_skip.shape:
            print("✅ 测试成功！CATM 模块输出尺寸正确，可以直接替换 Concat。")
        else:
            print("❌ 尺寸不匹配，请检查代码。")
            
        # 4. 显存占用检查 (大概)
        if torch.cuda.is_available():
            print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")

    # ==========================================
    # 4. 真实场景模拟：通道对齐 (重要提示)
    # ==========================================
    print("\n[进阶提示] 如果 SAM2 Encoder 输出通道是 256，而 Decoder 是 128 怎么办？")
    
    sam2_skip_channels = 256
    decoder_channels = 128
    
    # 模拟不匹配的输入
    real_x_skip = torch.randn(BATCH_SIZE, sam2_skip_channels, H, W).to(device)
    real_x_decoder = torch.randn(BATCH_SIZE, decoder_channels, H, W).to(device)
    
    # 你需要加一个 1x1 卷积层来对齐通道，才能喂给 CATM
    # 这个 align_conv 应该定义在你的网络 `__init__` 里
    align_conv = nn.Conv2d(sam2_skip_channels, decoder_channels, kernel_size=1).to(device)
    
    # 前向传播流程
    aligned_skip = align_conv(real_x_skip) # 256 -> 128
    final_out = catm_module(aligned_skip, real_x_decoder)
    
    print(f"SAM2 Skip ({sam2_skip_channels}) -> Aligned ({decoder_channels}) + Decoder ({decoder_channels}) -> CATM Out: {final_out.shape}")






    #########################################
    #########################################

    ## 这是加入CATM的完整代码。

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

# ==========================================
# 1. 新增模块: SwinBlock 和 CATM
# ==========================================

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(x_flat)
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        return x

class CATM(nn.Module):
    """ Cross-Attention Transformer Module: 替代 Skip Connection 中的 Concat """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.swin = SwinTransformerBlock(dim, num_heads)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward_shared_sa(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, 1)).expand_as(x)
        max_pool = F.adaptive_max_pool2d(x, output_size=(1, 1))[0].expand_as(x)
        pooled = torch.cat([x, avg_pool, max_pool], dim=1)
        attention = self.fuse(pooled)
        return attention * x

    def forward(self, x_skip: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
        # x_skip: 来自 Encoder 的特征 (Q)
        # x_decoder: 来自 Decoder 上采样的特征 (K, V)
        b, c, h, w = x_skip.shape
        dec_feat = self.swin(x_decoder)
        
        skip_seq = x_skip.flatten(2).transpose(1, 2)
        dec_seq = dec_feat.flatten(2).transpose(1, 2)

        q = self.q_proj(skip_seq)
        k = self.k_proj(dec_seq)
        v = self.v_proj(dec_seq)

        attn_out, _ = self.cross_attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        
        fused = self.forward_shared_sa(attn_out + x_skip)
        return fused

# ==========================================
# 2. 原有辅助模块 (DoubleConv, Adapter, RFB)
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# [修改点] 将原来的 Up 类替换为整合了 CATM 的 Up_CATM
class Up_CATM(nn.Module):
    """Upscaling then CATM fusion then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels: 这里指单个特征图的通道数 (因为CATM不增加通道，融合后还是64)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 核心修改：使用 CATM 替代 concat
        self.catm = CATM(dim=in_channels, num_heads=4)
        
        # 核心修改：DoubleConv 的输入不再是 2*in_channels，而是 in_channels
        # 因为 CATM 融合后的维度没有变大
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        # x1: deep feature (需要上采样)
        # x2: skip feature (来自 encoder)
        
        x1 = self.up(x1)
        
        # 处理 Padding (以防尺寸不匹配)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # [修改点] 不再是 torch.cat，而是 CATM
        # 注意: x2 是 skip, x1 是 decoder upsampled
        x = self.catm(x2, x1)
        
        return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

# ==========================================
# 3. 主网络结构 SAM2UNet
# ==========================================

class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
            
        # 清理不需要的头
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
            
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        
        # RFB 模块将所有层级特征压缩到 64 通道
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        
        # [修改点] 实例化 Up_CATM
        # 注意: 参数变成了 (64, 64) 而不是原来的 (128, 64)
        # 因为 CATM 输入是 64+64，输出融合后是 64，后续卷积也是 64->64
        self.up1 = (Up_CATM(64, 64)) 
        self.up2 = (Up_CATM(64, 64))
        self.up3 = (Up_CATM(64, 64))
        
        # Up4 通常是最底层的处理，这里看起来你代码里是把 Up 用作上采样融合
        # 但你原来的 self.up4 = (Up(128, 64)) 在 forward 里好像没用到？
        # 为了保持一致性，如果用到的话也改掉
        self.up4 = (Up_CATM(64, 64)) 
        
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder (SAM2)
        x1, x2, x3, x4 = self.encoder(x)
        
        # Neck (RFB) - 全部变成 64 通道
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        
        # Decoder (Using Up_CATM)
        # x4 (64) 和 x3 (64) 融合 -> 64
        x = self.up1(x4, x3) 
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        
        # x (64) 和 x2 (64) 融合 -> 64
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        
        # x (64) 和 x1 (64) 融合 -> 64
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        
        return out, out1, out2

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet().cuda()
        # 注意：这里模拟输入时，记得 SAM2 要求输入尺寸通常是 1024 或 类似，
        # 如果你训练用 224，确保 SAM2 config 支持或能适应
        x = torch.randn(2, 3, 224, 224).cuda() # 显存有限测试时调小 Batch Size
        out, out1, out2 = model(x)
        print(f"Output shapes: {out.shape}, {out1.shape}, {out2.shape}")