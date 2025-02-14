import torch
import torch.nn as nn
import torch.nn.functional as F

def getPadding(kernel_size, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

# 定义 Conv + GroupNorm + ReLU 模块
class ConvGroupNormReLU(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 norm_layer=nn.GroupNorm,
                 groups=8,
                 act_layer=nn.ReLU,
                 bias=False):
        super(ConvGroupNormReLU, self).__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      bias=bias,
                      dilation=dilation,
                      stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2
                      ), norm_layer(groups, out_channels), act_layer())

# 定义 SE 注意力模块
class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: 全局平均池化
        y = F.adaptive_avg_pool2d(x, (1, 1))
        # Excitation: 全连接层
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        # 激活函数
        y = self.sigmoid(y)
        return x * y  # 通过注意力加权原始特征

class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 7, 11], groups=8, se_reduction=16):
        super(ASPPBlock, self).__init__()
        self.dilation_rates = dilation_rates
        mid_channels = (len(dilation_rates) + 2) * out_channels
        
        # 1x1卷积用于捕获通道间的依赖关系
        self.conv_1x1 = ConvGroupNormReLU(in_channels, out_channels, kernel_size=1, groups=groups)
        
        # 使用不同空洞率的卷积
        self.conv_3x3_dilated = nn.ModuleList([
            ConvGroupNormReLU(in_channels, out_channels, kernel_size=3, dilation=d, groups=groups) for d in dilation_rates
        ])
        
        # 使用池化层获取全局上下文
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = ConvGroupNormReLU(in_channels, out_channels, kernel_size=1, groups=groups)
        
        # 最终卷积层用于融合所有的ASPP特征
        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # SE注意力模块
        self.attention = SEAttention(mid_channels, reduction=se_reduction)

    def forward(self, x):
        # 1x1卷积分支
        x1 = self.conv_1x1(x)

        # 空洞卷积分支
        dilated_features = [conv(x) for conv, x in zip(self.conv_3x3_dilated, [x] * len(self.dilation_rates))]

        # 全局平均池化分支
        x_global = self.global_avg_pool(x)
        x_global = self.conv_global(x_global)
        x_global = F.interpolate(x_global, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 将各分支的输出拼接
        output = torch.cat([x1, x_global] + dilated_features, dim=1)

        # 应用 SE 注意力
        output = output + self.attention(output)

        # 最终卷积融合特征
        output = self.final_conv(output)

        return output

class MultiScaleASPP(nn.Module):
    def __init__(self, in_channels, channels, dilation_rates=[1, 3, 7, 11], 
                 num_blocks=3, groups=8, se_reduction=16):
        super(MultiScaleASPP, self).__init__()
        self.in_conv = ConvGroupNormReLU(in_channels, channels, kernel_size=1, groups=groups)
        self.blocks = nn.ModuleList([ASPPBlock(channels, channels, dilation_rates, groups, se_reduction) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.in_conv(x)
        # 依次通过多个ASPP Block
        for block in self.blocks:
            x = x + block(x)
        return x


if __name__ == '__main__':
    # 示例使用
    B, C = 4, 64

    size_list = [(7, 7), (14, 7), (14, 28), (56, 7), (28, 56), (56, 56)]

    model = MultiScaleASPP(in_channels=C, out_channels=128, dilation_rates=[1, 3, 7, 11], num_blocks=3, groups=8, se_reduction=16)

    for H, W in size_list:
        x = torch.randn(B, C, H, W)
        output = model(x)
        print(f"Input size: {H}x{W}, Output size: {output.shape}")  # 输出形状
