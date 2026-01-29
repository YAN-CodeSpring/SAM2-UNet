import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2



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

class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.67):
        super(MultiResBlock, self).__init__()
        
        # 计算各分支的通道数 (参考原论文公式)
        W = out_channels * alpha
        # 原论文近似分配比例: 1/6, 1/3, 1/2
        self.out_c1 = int(W * 0.167)
        self.out_c2 = int(W * 0.333)
        self.out_c3 = int(W * 0.5)
        
        # 修正总通道数误差（确保拼接后等于 out_channels 附近，这里我们最后会接个conv调整）
        self.total_c = self.out_c1 + self.out_c2 + self.out_c3

        # 分支1: 3x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_c1),
            nn.ReLU(inplace=True)
        )
        
        # 分支2: 3x3 -> 3x3 (等效 5x5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_c1, self.out_c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_c2),
            nn.ReLU(inplace=True)
        )
        
        # 分支3: 3x3 -> 3x3 -> 3x3 (等效 7x7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.out_c2, self.out_c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_c3),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接 (Shortcut)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, self.total_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.total_c)
        )
        
        # 最后的融合层 (确保输出通道数对齐)
        self.final_bn = nn.BatchNorm2d(self.total_c)
        self.final_relu = nn.ReLU(inplace=True)
        
        # 如果计算出的通道数和目标不一致，用1x1卷积修正
        self.adjust_conv = None
        if self.total_c != out_channels:
             self.adjust_conv = nn.Conv2d(self.total_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Shortcut
        res = self.shortcut(x)
        
        # Multi-scale features
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # 拼接
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.final_bn(out)
        
        # 残差相加
        out = out + res
        out = self.final_relu(out)
        
        # 通道对齐
        if self.adjust_conv:
            out = self.adjust_conv(out)
            
        return out
    

class MultiResUp(nn.Module):
    """Upscaling -> Concat -> MultiResBlock (代替原来的DoubleConv)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 这里你可以选择是否保留之前的 CoordAttention，如果想纯粹测MultiRes，可以注释掉下面这行
        # self.ca = CoordAtt(out_channels, out_channels) 
        
        # 【核心替换】：用 MultiResBlock 替换 DoubleConv
        # 输入通道是 in_channels (拼接后的), 输出是 out_channels
        self.conv = MultiResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        
        # 多尺度卷积提取
        x = self.conv(x)
        
        # 如果你保留了 CA，就在这里用
        # x = self.ca(x)
        
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

### 下面这个是新加的
# [新增模块] 带有 Attention 的上采样模块
class AttentionUp(nn.Module):
    """Upscaling -> Attention Gate -> Concat -> Double Conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 保持和你原来的 Up 一样的上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 计算 Attention 参数
        # in_channels 是拼接后的总通道数 (例如 128)
        # 所以 gate (深层) 和 x (跳跃连接) 各占一半
        F_g = in_channels // 2
        F_l = in_channels // 2
        F_int = F_g // 2  # 中间层通道数，通常减半
        
        # 实例化 AG
        self.ag = Attention_block(F_g, F_l, F_int)
        
        # 保持和你原来的 Up 一样的卷积
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        # x1: 深层特征 (需要上采样)
        # x2: Skip Connection (浅层特征，包含背景噪声)
        
        # 1. 上采样
        x1 = self.up(x1)
        
        # 2. 处理尺寸不匹配 (完全复制你原来的 padding 逻辑)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 3. [核心修改] 使用 AG 过滤 x2
        # 用 x1 (Gating Signal) 去看 x2
        x2 = self.ag(g=x1, x=x2)
        
        # 4. 拼接 (此时 x2 已经被净化了)
        x = torch.cat([x2, x1], dim=1)
        
        # 5. 卷积输出
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


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
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
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        # ======= 只需要改这里 =======
        # 使用 MultiResUp
        self.up1 = (MultiResUp(128, 64)) 
        self.up2 = (MultiResUp(128, 64))
        self.up3 = (MultiResUp(128, 64))
        
        # 如果需要 up4
        self.up4 = (MultiResUp(128, 64))
        
        # 这里的up123被替换：
        # self.up1 = (Up(128, 64))
        # self.up2 = (Up(128, 64))
        # self.up3 = (Up(128, 64))
        
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet().cuda()
        x = torch.randn(1, 3, 224, 224).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)