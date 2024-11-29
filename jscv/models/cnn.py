import torch.nn as nn

def getPadding(kernel_size, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

class ConvBNReLU(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6,
                 bias=False):
        # if norm_layer == nn.BatchNorm2d:
        #     norm = nn.BatchNorm2d(out_channels)
        # elif norm_layer == nn.LayerNorm:
        #     norm = nn.LayerNorm(out_channels)
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      bias=bias,
                      dilation=dilation,
                      stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2
                      ), norm_layer(out_channels), act_layer())


class ConvBN(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      bias=bias,
                      dilation=dilation,
                      stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) //
                      2), norm_layer(out_channels))


class Conv(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      bias=bias,
                      dilation=dilation,
                      stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) //
                      2))


class SeparableConvBNReLU(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels), act_layer())


class SeparableConvBN(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) //
                      2,
                      groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))




class PWConvBNReLU(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=1,
                      groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels), act_layer())
