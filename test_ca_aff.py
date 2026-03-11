import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 模拟依赖模块 ====================

class CoordAtt(nn.Module):
    """模拟CA模块"""
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # 或者ReLU
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class AFF(nn.Module):
    """模拟AFF模块"""
    def __init__(self, channels):
        super().__init__()
        self.local_att = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        xa = x1 + x2  # 或者cat，取决于AFF具体实现
        xl = self.local_att(torch.cat([x1, x2], dim=1))
        xg = self.global_att(torch.cat([x1, x2], dim=1))
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x1 * wei + x2 * (1 - wei)
        return xo

class DoubleConv(nn.Module):
    """模拟DoubleConv"""
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

# ==================== 你的模块（修正版） ====================

class UpWithCA_AFF(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = CoordAtt(deep_channels, deep_channels)
        
        # 通道对齐
        if deep_channels != skip_channels:
            self.skip_conv = nn.Conv2d(skip_channels, deep_channels, 1)
            aff_channels = deep_channels
        else:
            self.skip_conv = None
            aff_channels = deep_channels
            
        self.aff = AFF(channels=aff_channels)
        self.conv = DoubleConv(aff_channels, out_channels, aff_channels//2)

    def forward(self, x_deep, x_skip):
        print(f"  Input deep: {x_deep.shape}, skip: {x_skip.shape}")
        
        x_deep = self.ca(x_deep)
        print(f"  After CA: {x_deep.shape}")
        
        x_deep = self.up(x_deep)
        print(f"  After Up: {x_deep.shape}")
        
        if x_deep.shape != x_skip.shape:
            diffY = x_skip.size(2) - x_deep.size(2)
            diffX = x_skip.size(3) - x_deep.size(3)
            x_deep = F.pad(x_deep, [diffX//2, diffX-diffX//2,
                                    diffY//2, diffY-diffY//2])
            print(f"  After Pad: {x_deep.shape}")
        
        if self.skip_conv is not None:
            x_skip = self.skip_conv(x_skip)
            print(f"  After skip_conv: {x_skip.shape}")
        
        x = self.aff(x_deep, x_skip)
        print(f"  After AFF: {x.shape}")
        
        out = self.conv(x)
        print(f"  Output: {out.shape}")
        return out

# ==================== 测试 ====================

def test_module():
    print("=" * 60)
    print("测试1: 标准UNet场景 (deep=512, skip=256, out=256)")
    print("=" * 60)
    up1 = UpWithCA_AFF(deep_channels=512, skip_channels=256, out_channels=256)
    x_deep = torch.randn(2, 512, 14, 14)   # 深层特征
    x_skip = torch.randn(2, 256, 28, 28)   # skip connection
    out1 = up1(x_deep, x_skip)
    assert out1.shape == (2, 256, 28, 28), f"形状错误: {out1.shape}"
    print("✅ 测试1通过\n")
    
    print("=" * 60)
    print("测试2: 通道相同场景 (deep=256, skip=256, out=128)")
    print("=" * 60)
    up2 = UpWithCA_AFF(deep_channels=256, skip_channels=256, out_channels=128)
    x_deep = torch.randn(2, 256, 28, 28)
    x_skip = torch.randn(2, 256, 56, 56)
    out2 = up2(x_deep, x_skip)
    assert out2.shape == (2, 128, 56, 56), f"形状错误: {out2.shape}"
    print("✅ 测试2通过\n")
    
    print("=" * 60)
    print("测试3: 梯度传播测试")
    print("=" * 60)
    up3 = UpWithCA_AFF(deep_channels=128, skip_channels=64, out_channels=64)
    x_deep = torch.randn(2, 128, 7, 7, requires_grad=True)
    x_skip = torch.randn(2, 64, 14, 14, requires_grad=True)
    out3 = up3(x_deep, x_skip)
    loss = out3.sum()
    loss.backward()
    assert x_deep.grad is not None, "x_deep梯度丢失"
    assert x_skip.grad is not None, "x_skip梯度丢失"
    print(f"  x_deep grad norm: {x_deep.grad.norm().item():.4f}")
    print(f"  x_skip grad norm: {x_skip.grad.norm().item():.4f}")
    print("✅ 测试3通过\n")
    
    print("=" * 60)
    print("测试4: 数值合理性")
    print("=" * 60)
    up4 = UpWithCA_AFF(deep_channels=64, skip_channels=64, out_channels=32)
    x_deep = torch.randn(4, 64, 16, 16)
    x_skip = torch.randn(4, 64, 32, 32)
    out4 = up4(x_deep, x_skip)
    print(f"  输出范围: [{out4.min():.4f}, {out4.max():.4f}]")
    print(f"  输出均值: {out4.mean():.4f}")
    assert not torch.isnan(out4).any(), "有NaN"
    assert not torch.isinf(out4).any(), "有Inf"
    print("✅ 测试4通过\n")
    
    print("=" * 60)
    print("测试5: 即插即用验证 - 替换UNet的Up层")
    print("=" * 60)
    # 模拟一个简化UNet的前向
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
            self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 112
            self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # 56
            self.enc4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # 28
            
            # 用你的模块替换标准Up
            self.up1 = UpWithCA_AFF(512, 256, 256)   # 28->56
            self.up2 = UpWithCA_AFF(256, 128, 128)   # 56->112
            self.up3 = UpWithCA_AFF(128, 64, 64)     # 112->224
            
            self.final = nn.Conv2d(64, 1, 1)
            
        def forward(self, x):
            print(f"Input: {x.shape}")
            e1 = self.enc1(x)      # 64, 224
            e2 = self.enc2(e1)     # 128, 112
            e3 = self.enc3(e2)     # 256, 56
            e4 = self.enc4(e3)     # 512, 28
            
            d1 = self.up1(e4, e3)  # 256, 56
            d2 = self.up2(d1, e2)  # 128, 112
            d3 = self.up3(d2, e1)  # 64, 224
            
            return self.final(d3)  # 1, 224
    
    model = SimpleUNet()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1, 224, 224), f"最终输出错误: {out.shape}"
    print(f"✅ 测试5通过: UNet输出 {out.shape}")

if __name__ == "__main__":
    test_module()