import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torch.utils
from time import time
import datetime
import cv2
import logging

from torchvision.models.resnet import ResNet, Bottleneck,BasicBlock
from jscv.models.cnn import *
from jscv.utils.overall import global_dict
from jscv.utils.utils import TimeCounter, warmup
from jscv.losses.utils import loss_map

do_debug_time = False

coun_1 = TimeCounter(do_debug_time)


fpn_decoder_args_1 = fpn_decoder_args_256 = dict(dec_channels=[32, 64, 128, 256], blocks=[1, 1, 1, 1], num_classes=2)
fpn_decoder_args_2 = fpn_decoder_args_512 = dict(dec_channels=[48, 96, 256, 512], blocks=[1, 1, 1, 1], num_classes=2)
fpn_decoder_args_512_1232 = dict(dec_channels=[48, 96, 256, 512], blocks=[1, 2, 3, 2], num_classes=2)
fpn_decoder_args_512_1234 = dict(dec_channels=[48, 96, 256, 512], blocks=[1, 2, 3, 4], num_classes=2)
fpn_decoder_args_512_1131 = dict(dec_channels=[48, 96, 256, 512], blocks=[1, 1, 3, 1], num_classes=2)
fpn_decoder_args_512_1133 = dict(dec_channels=[48, 96, 256, 512], blocks=[1, 2, 3, 4], num_classes=2)

# 默认采用
fpn_decoder_args_M = fpn_decoder_args_M_1123 = dict(dec_channels=[64, 128, 256, 512], blocks=[1, 1, 2, 3], num_classes=2)

fpn_decoder_args_M_1111 = dict(dec_channels=[64, 128, 256, 512], blocks=[1, 1, 1, 1], num_classes=2)
fpn_decoder_args_M_1232 = dict(dec_channels=[64, 128, 256, 512], blocks=[1, 2, 3, 2], num_classes=2)


fpn_decoder_args_M_1131 = dict(dec_channels=[64, 128, 256, 512], blocks=[1, 1, 3, 1], num_classes=2)
fpn_decoder_args_M_1133 = dict(dec_channels=[64, 128, 256, 512], blocks=[1, 2, 3, 4], num_classes=2)



class ResNetEncoder(ResNet):
    """
    ResNetEncoder inherits from torchvision's official ResNet. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    global average pooling layer and the fully connected layer that was originally
    used for classification. The forward method  additionally returns the feature
    maps at all resolutions for decoder's use.
    """
    layers_dict = {
        'resnet18':  [2, 2, 2, 2],
        'resnet50':  [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3],
    }
    channels_dict = {
        'resnet18':  [64, 64, 128, 256, 512],
        'resnet50':  [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
        'resnet152': [64, 256, 512, 1024, 2048],
    }

    def __init__(self, 
                 variant='resnet50',
                 return_x0=True,


                #igore
                 norm_layer=None,
                 features_only=False,
                 do_maxpool=True,
                 avg_pool_to=1,
                 maxpool_kernel=3,
                 maxpool_stride=2,
                 num_classes=1000,
                 layer4_downsample=True,
                 ):
        if variant in ["resnet18"]:
            block=BasicBlock
        else:
            block=Bottleneck
        super().__init__(
            block=block,
            layers=self.layers_dict[variant],
            num_classes=num_classes,
            # replace_stride_with_dilation=[False, False, False],
            norm_layer=norm_layer)
        self.features_only = features_only
        self.avg_pool_to = avg_pool_to
        self.layers = self.layers_dict[variant]
        self.channels = self.channels_dict[variant]
        self.return_x0 = return_x0
        self.layer4_downsample = layer4_downsample

        if features_only:
            del self.avgpool
            del self.fc
        elif avg_pool_to != 1:
            self.avgpool = nn.AdaptiveAvgPool2d(avg_pool_to)

        self.do_maxpool = do_maxpool
        if do_maxpool:
            if maxpool_kernel != 3 or maxpool_stride != 2:
                self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel,
                                            stride=maxpool_stride,
                                            padding=getPadding(maxpool_kernel,maxpool_stride))
        else:
            del self.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.return_x0:
            x0 = x  # 1/2
        
        if self.do_maxpool:
            x = self.maxpool(x) # 1/4

        x = self.layer1(x) # layer1不下采样
        x1 = x  # 1/4
        x = self.layer2(x)
        x2 = x  # 1/8
        x = self.layer3(x)
        x3 = x  # 1/16
        x = self.layer4(x)
        x4 = x  # 1/32

        if self.features_only:
            if self.return_x0:
                return x0, x1, x2, x3, x4
            else:
                return x1, x2, x3, x4
        else:
            if self.avg_pool_to != 1:
                x = self.avgpool(x).permute(0, 2, 3, 1)
                x = self.fc(x)
            else:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
            if self.return_x0:
                return x, [x0, x1, x2, x3, x4]
            else:
                return x, [x1, x2, x3, x4]


    def forward_test(self, x):
        coun_1.begin()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # 1/2
        coun_1.record_time("conv1")
        
        if self.do_maxpool:
            x = self.maxpool(x) # 1/4

        x = self.layer1(x) # layer1不下采样
        x1 = x  # 1/4
        coun_1.record_time("layer1")
        x = self.layer2(x)
        x2 = x  # 1/8
        coun_1.record_time("layer2")
        x = self.layer3(x)
        x3 = x  # 1/16
        coun_1.record_time("layer3")
        x = self.layer4(x)
        x4 = x  # 1/32
        coun_1.last("layer4")
        
        if self.features_only:
            return x0, x1, x2, x3, x4
        else:
            if self.avg_pool_to != 1:
                x = self.avgpool(x).permute(0, 2, 3, 1)
                x = self.fc(x)
            else:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
            return x, [x0, x1, x2, x3, x4]

    def pretrain(self, d:dict, from_where='resnet'):
        if from_where == 'resnet':
            del d['fc.weight']
            del d['fc.bias']
            self.load_state_dict(d)
        elif from_where == 'deeplabv3':
            assert False


class StrongerEncoder(nn.Module):
    def __init__(self,
                 backbone,
                 e4_class,
                 e4_channels_rate=1/4,
                 e4_layers=4,
                 e4_args={},
                 ):
        #assert features_only=False,
        super().__init__()
        self.backbone = backbone
        l_chs = backbone.channels
        psp_inc = l_chs[-1]
        psp_outc = int(psp_inc * e4_channels_rate)
        l_chs[-1] = psp_outc
        self.channels = l_chs
        L = []
        for i in range(e4_layers):
            inc = psp_inc if i == 0 else psp_outc
            L.append(e4_class(inc, psp_outc, **e4_args))
        self.e4 = nn.Sequential(*L)

    def forward(self, x):
        fs = list(self.backbone(x))
        fs[-1] = self.e4(fs[-1])
        return fs

    def pretrain_a(self, ckpt, e4_prefix:str):
        from jscv.utils.load_checkpoint import load_checkpoint
        load_checkpoint(self.backbone, ckpt, 'backbone')
        load_checkpoint(self.e4, ckpt, e4_prefix)


class ResBlocks(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.blocks = nn.ModuleList(models)
    
    def forward(self, x):
        for m in self.blocks:
            x1 = m(x)
            if x1.shape == x.shape:
                x = x + x1
            else:
                x = x1
        return x


def create_res_blocks(*models):
    if len(models) == 1:
        return models[0]
    else:
        return ResBlocks(*models)



# TODO 重构 因为实验的依赖，代码暂时不改
# 用于Global Branch的解码头
#TODO 弃用
class FPNDecoder_Old(nn.Module):

    def __init__(self, 
                 enc_channels,
                 dec_channels=1/2,
                 blocks=[2, 2, 2, 2],

                 use_decoder_0 = True,  # 是否使用Stem后的F_GB0
                 return_d2_feature=False,  # 是否返回context特征 #TODO 改名
                 num_classes=2,

                 classify_head=True,    # 默认
                 save_features=False,   # 保存特征，忽略
                 d2_feature_only=False,   # 忽略
                 last_layer_upsample=True,   # 忽略
                 return_d234_concat=False,   # True: context为LD4+LD3+LD2     #TODO 优化, 可扩展 返回context特征的方式
                #  context_type="gd2",
                 ):
        super().__init__()
        L = len(enc_channels)

        LB_no_f0 = L == 4

        if LB_no_f0:
            C1, C2, C3, C4 = enc_channels
        else:
            C0, C1, C2, C3, C4 = enc_channels

        self.use_decoder_0 = use_decoder_0 and (not LB_no_f0)

        if isinstance(dec_channels, list) or isinstance(dec_channels, tuple):
            DC1, DC2, DC3, DC4 = dec_channels
        else:
            DC = dec_channels
            DC1, DC2, DC3, DC4 = int(C1*DC), int(C2*DC), int(C3*DC), int(C4*DC)
        self.channels = (DC1, DC2, DC3, DC4)
        self.channel_d2 = DC3
        B1, B2, B3, B4 = blocks
        self.return_d2_feature = return_d2_feature
        self.d2_feature_only = d2_feature_only
        self.classify_head = classify_head
        self.save_features = save_features
        self.last_layer_upsample = last_layer_upsample

        self.return_d234_concat = return_d234_concat
        if return_d234_concat:  # return_d234_concat仅控制返回中间特征的方式
            assert return_d2_feature
            self.channel_d2 = DC3 + DC4 + C4
        
        

        self.layer4 = []
        for k in range(B4):
            inc = C3+C4 if k == 0 else DC4
            self.layer4.append(ConvBNReLU(inc, DC4, 3))
        self.layer4 = ResBlocks(*self.layer4)

        self.layer3 = []
        for k in range(B3):
            inc = DC4+C2 if k == 0 else DC3
            self.layer3.append(ConvBNReLU(inc, DC3, 3))
        self.layer3 = ResBlocks(*self.layer3)
        
        
        if not d2_feature_only:
            self.layer2 = []
            for k in range(B2):
                inc = DC3+C1 if k == 0 else DC2
                self.layer2.append(ConvBNReLU(inc, DC2, 3))
            self.layer2 = ResBlocks(*self.layer2)

            if self.use_decoder_0:
                self.layer1 = []
                for k in range(B1):
                    inc = DC2+C0 if k == 0 else DC1
                    self.layer1.append(ConvBNReLU(inc, DC1, 3))
                self.layer1 = ResBlocks(*self.layer1)
            else:
                DC1 = DC2

            if classify_head:
                self.segment_head = nn.Conv2d(DC1, num_classes, 3, 1, 1)

    def clear_features(self):
        self.features.clear()


    def forward(self, *xs):
        
        lenxs = len(xs)
        if lenxs == 4:
            x1, x2, x3, x4 = xs
        else:
            x0, x1, x2, x3, x4 = xs
        # for xxx in (x1, x2, x3, x4):
        #     print(xxx.shape)

        self.features = []
        # print("jscv/hr_models/base_models.py  x4.shape, x3.shape, x2.shape, x1.shape):  ", 
        #       x4.shape, x3.shape, x2.shape, x1.shape)

        # for x in xs:
        #     print(x.shape)

        x = x4
        if self.last_layer_upsample:
            f4 = x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, x3], dim=1)
        f3 = x = self.layer4(x)

        if self.save_features:
            self.features.append(x)

        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.layer3(x)

        if self.save_features:
            self.features.append(x)

        if self.return_d2_feature:
            f2 = x
            if self.return_d234_concat:
                # print(f4.shape, f3.shape, f2.shape)
                f4 = F.interpolate(f4, size=f2.shape[2:], mode='bilinear', align_corners=False)
                f3 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
                f2 = torch.concat([f2, f3, f4], dim=1)

            if self.d2_feature_only:
                return f2


        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.layer2(x)

        if self.save_features:
            self.features.append(x)

        if self.use_decoder_0:
            x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, x0], dim=1)
            x = self.layer1(x)

        if self.save_features:
            self.features.append(x)

        if self.classify_head:
            x = self.segment_head(x)
            # print("jscv/hr_models/base_models.py  x.shape:  ", x.shape)

        if self.return_d2_feature:
            return x, f2
        # print("2", x.shape)
        return x






class FPNDecoder(nn.Module):
    
    #TODO 名字错了
    CXT_D2 = 1          #? 上下文融合的特征有： 解码-2
    CXT_D2_D3_B4 = 2    #? 上下文融合的特征有： 解码-2 解码-3 主干-4
    CXT_B2_B3_B4 = 3    #? 上下文融合的特征有： 主干-2 主干-3 主干-4
    
    CTX_D2 = 1          #? 上下文融合的特征有： 解码-2
    CTX_D2_D3_B4 = 2    #? 上下文融合的特征有： 解码-2 解码-3 主干-4
    CTX_B2_B3_B4 = 3    #? 上下文融合的特征有： 主干-2 主干-3 主干-4


    def __init__(self, 
                 enc_channels,
                 dec_channels=1/2,
                 blocks=[2, 2, 2, 2],

                 use_decoder_0 = True,  # 是否使用Stem后的F_GB0
                 return_context=True,  # 是否返回context特征 #TODO 改名
                 context_type="d2",

                 num_classes=2,
                 context_only=False,   # 忽略
                 classify_head=True,    # 默认
                 last_layer_upsample=True,   # 忽略
                 ):
        super().__init__()

        LB_no_f0 = (len(enc_channels) == 4)
        # print(enc_channels, LB_no_f0)

        if LB_no_f0:
            C1, C2, C3, C4 = enc_channels
        else:
            C0, C1, C2, C3, C4 = enc_channels

        self.use_decoder_0 = use_decoder_0 and (not LB_no_f0)

        if isinstance(dec_channels, list) or isinstance(dec_channels, tuple):
            DC1, DC2, DC3, DC4 = dec_channels
        else:
            DC = dec_channels
            DC1, DC2, DC3, DC4 = int(C1*DC), int(C2*DC), int(C3*DC), int(C4*DC)
        self.channels = (DC1, DC2, DC3, DC4)
        B1, B2, B3, B4 = blocks
        self.return_context = return_context
        self.context_only = context_only
        self.classify_head = classify_head
        # self.save_features = save_features
        self.last_layer_upsample = last_layer_upsample

        self.context_type = context_type
        if context_type == 'd2' or context_type == FPNDecoder.CXT_D2:
            self.context_channel = DC3
            self.context_type = FPNDecoder.CXT_D2
        elif context_type == 'd2+d3+b4' or context_type == FPNDecoder.CXT_D2_D3_B4:
            self.context_channel = DC3 + DC4 + C4
            self.context_type = FPNDecoder.CXT_D2_D3_B4
        elif context_type == 'b2+b3+b4' or context_type == FPNDecoder.CXT_B2_B3_B4:
            self.context_channel = C2 + C3 + C4
            self.context_type = FPNDecoder.CXT_B2_B3_B4
        else:
            assert False, "Unknown context type"


        self.layer4 = []
        for k in range(B4):
            inc = C3+C4 if k == 0 else DC4
            self.layer4.append(ConvBNReLU(inc, DC4, 3))
        self.layer4 = ResBlocks(*self.layer4)

        self.layer3 = []
        for k in range(B3):
            inc = DC4+C2 if k == 0 else DC3
            self.layer3.append(ConvBNReLU(inc, DC3, 3))
        self.layer3 = ResBlocks(*self.layer3)
        
        if not context_only:
            self.layer2 = []
            for k in range(B2):
                inc = DC3+C1 if k == 0 else DC2
                self.layer2.append(ConvBNReLU(inc, DC2, 3))
            self.layer2 = ResBlocks(*self.layer2)

            if self.use_decoder_0:
                self.layer1 = []
                for k in range(B1):
                    inc = DC2+C0 if k == 0 else DC1
                    self.layer1.append(ConvBNReLU(inc, DC1, 3))
                self.layer1 = ResBlocks(*self.layer1)
            else:
                DC1 = DC2

            if classify_head:
                self.segment_head = nn.Conv2d(DC1, num_classes, 3, 1, 1)


    def forward(self, *xs):
        
        lenxs = len(xs)
        if lenxs == 4:
            x1, x2, x3, x4 = xs
        else:
            x0, x1, x2, x3, x4 = xs

        # for x in xs:
        #     print("FPNDecoder:", x.shape)

        x = x4
        if self.last_layer_upsample:
            f4 = x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, x3], dim=1)
        f3 = x = self.layer4(x)


        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.layer3(x)


        if self.return_context:
            # print("@@@@",  self.context_type)
            if self.context_type == FPNDecoder.CXT_D2:
                ctx = x
            elif self.context_type == FPNDecoder.CXT_D2_D3_B4:
                # print(f4.shape, f3.shape, f2.shape)
                f2 = x
                f4 = F.interpolate(f4, size=f2.shape[2:], mode='bilinear', align_corners=False)
                f3 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
                ctx = torch.concat([f2, f3, f4], dim=1)
            elif self.context_type == FPNDecoder.CXT_B2_B3_B4:
                x4 = F.interpolate(x4, size=x2.shape[2:], mode='bilinear', align_corners=False)
                x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
                ctx = torch.concat([x2, x3, x4], dim=1)
            del f3, f4, x2, x3, x4

            if self.context_only:
                return ctx

        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.layer2(x)

        if self.use_decoder_0:
            x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, x0], dim=1)
            x = self.layer1(x)


        if self.classify_head:
            x = self.segment_head(x)

        if self.return_context:
            return x, ctx
        return x

from torch.nn import GroupNorm

def getPadding(kernel_size, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

class ConvGNReLU(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 act_layer=nn.ReLU6,
                 num_groups=8,
                 bias=False):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      bias=bias,
                      dilation=dilation,
                      stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2
                      ), GroupNorm(num_groups, out_channels), act_layer())


'''
    后继局部解码器，只有最后两层
'''
class PostDecodeHead(nn.Module):
    
    def __init__(self, 
                 enc_channels,              # [ F_LB1, F_LB2, F_LD3 ]  OR [ F_LB0, F_LB1, F_LB2, F_LD3 ]
                 channels=[64,128],         # LD1 LD2   OR [LD0 LD1 LD2]
                 blocks=[1, 2],             # LD1 LD2   OR [LD0 LD1 LD2]
                 use_decoder_0 = False,     # True: 主干Stem层后的特征也加入融合,没必要，其语义太弱了
                 use_classify_head=True,
                 num_classes=2,
                 ):
        super().__init__()


        LB_no_f0 = len(channels) == 2

        if LB_no_f0:
            C1, C2, C3 = enc_channels[-3:]
            DC1, DC2 = channels
            B1, B2 = blocks
        else:
            C0, C1, C2, C3 = enc_channels
            DC0, DC1, DC2 = channels
            B0, B1, B2 = blocks

        self.use_decoder_0 = use_decoder_0 and (not LB_no_f0)
        self.channels = channels
        self.use_classify_head = use_classify_head
        self.num_classes = num_classes

        layer2 = []
        for k in range(B2):
            inc = C2+C3 if k == 0 else DC2
            layer2.append(ConvGNReLU(inc, DC2, 3))
        self.layer2 = ResBlocks(*layer2)

        layer1 = []
        for k in range(B1):
            inc = DC2+C1 if k == 0 else DC1
            layer1.append(ConvGNReLU(inc, DC1, 3))
        self.layer1 = ResBlocks(*layer1)

        if self.use_decoder_0:
            layer0 = []
            for k in range(B0):
                inc = DC1+C0 if k == 0 else DC0
                layer0.append(ConvGNReLU(inc, DC0, 3))
            self.layer0 = ResBlocks(*layer0)
        else:
            DC0 = DC1
        if use_classify_head:
            self.segment_head = nn.Conv2d(DC0, num_classes, 3, 1, 1)


    def forward(self, *xs):
        
        lenxs = len(xs)
        if lenxs == 3:
            LB1, LB2, LD3 = xs
        else:
            LB0, LB1, LB2, LD3 = xs
        
        # for xxx in xs:
        #     print(xxx.shape)

        if LD3.shape[2:] != LB2.shape[2:]:
            LD3 = F.interpolate(LD3, size=LB2.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer2(torch.cat([LD3, LB2], dim=1))

        x = F.interpolate(x, size=LB1.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer1(torch.cat([x, LB1], dim=1))

        if self.use_decoder_0:
            x = F.interpolate(x, size=LB0.shape[2:], mode='bilinear', align_corners=False)
            x = self.layer0(torch.cat([x, LB0], dim=1))

        if self.use_classify_head:
            return self.segment_head(x)
        return x



class PostDecodeHead_V1(nn.Module):
    
    def __init__(self, 
                 enc_channels,              # [ F_LB1, F_LB2, F_LD3 ]  OR [ F_LB0, F_LB1, F_LB2, F_LD3 ]
                 channels=[64,128],         # LD1 LD2   OR [LD0 LD1 LD2]
                 blocks=[1, 2],             # LD1 LD2   OR [LD0 LD1 LD2]
                 use_decoder_0 = False,     # True: 主干Stem层后的特征也加入融合,没必要，其语义太弱了
                 use_classify_head=True,
                 num_classes=2,
                 ):
        super().__init__()


        LB_no_f0 = len(channels) == 2

        if LB_no_f0:
            C1, C2, C3 = enc_channels[-3:]
            DC1, DC2 = channels
            B1, B2 = blocks
        else:
            C0, C1, C2, C3 = enc_channels
            DC0, DC1, DC2 = channels
            B0, B1, B2 = blocks

        self.use_decoder_0 = use_decoder_0 and (not LB_no_f0)
        self.channels = channels
        self.use_classify_head = use_classify_head
        self.num_classes = num_classes

        layer2 = []
        for k in range(B2):
            inc = C2+C3 if k == 0 else DC2
            layer2.append(ConvBNReLU(inc, DC2, 3))
        self.layer2 = nn.Sequential(*layer2)

        layer1 = []
        for k in range(B1):
            inc = DC2+C1 if k == 0 else DC1
            layer1.append(ConvBNReLU(inc, DC1, 3))
        self.layer1 = nn.Sequential(*layer1)

        if self.use_decoder_0:
            layer0 = []
            for k in range(B0):
                inc = DC1+C0 if k == 0 else DC0
                layer0.append(ConvBNReLU(inc, DC0, 3))
            self.layer0 = nn.Sequential(*layer0)
        else:
            DC0 = DC1
        if use_classify_head:
            self.segment_head = nn.Conv2d(DC0, num_classes, 3, 1, 1)


    def forward(self, *xs):
        
        lenxs = len(xs)
        if lenxs == 3:
            LB1, LB2, LD3 = xs
        else:
            LB0, LB1, LB2, LD3 = xs
        
        # for xxx in xs:
        #     print(xxx.shape)

        if LD3.shape[2:] != LB2.shape[2:]:
            LD3 = F.interpolate(LD3, size=LB2.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer2(torch.cat([LD3, LB2], dim=1))

        x = F.interpolate(x, size=LB1.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer1(torch.cat([x, LB1], dim=1))

        if self.use_decoder_0:
            x = F.interpolate(x, size=LB0.shape[2:], mode='bilinear', align_corners=False)
            x = self.layer0(torch.cat([x, LB0], dim=1))

        if self.use_classify_head:
            return self.segment_head(x)
        return x



class ResBlocks2(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.blocks = nn.ModuleList(models)
    
    def forward(self, x):
        for m in self.blocks:
            x = m(x) + x
        return x

class FPNDecoder_2(nn.Module):

    def __init__(self, enc_channels,
                 dec_channels=1/2,
                 blocks=[2, 2, 2, 2],
                 num_classes=2,
                 ):
        super().__init__()
        C0, C1, C2, C3, C4 = enc_channels

        if isinstance(dec_channels, list) or isinstance(dec_channels, tuple):
            DC1, DC2, DC3, DC4 = dec_channels
        else:
            DC = dec_channels
            DC1, DC2, DC3, DC4 = int(C1*DC), int(C2*DC), int(C3*DC), int(C4*DC)
        
        B1, B2, B3, B4 = blocks

        self.f4 = ConvBNReLU(C4, DC4, 3)
        self.f3 = ConvBNReLU(DC4, DC3, 3)
        self.f2 = ConvBNReLU(DC3, DC2, 3)
        self.f1 = ConvBNReLU(DC2, DC1, 3)
        self.n3 = ConvBNReLU(C3, DC4, 1)
        self.n2 = ConvBNReLU(C2, DC3, 1)
        self.n1 = ConvBNReLU(C1, DC2, 1)
        self.n0 = ConvBNReLU(C0, DC1, 1)

        self.layer4 = []
        for k in range(B4):
            self.layer4.append(ConvBNReLU(DC4, DC4, 3))
        self.layer4 = ResBlocks2(*self.layer4)

        self.layer3 = []
        for k in range(B3):
            self.layer3.append(ConvBNReLU(DC3, DC3, 3))
        self.layer3 = ResBlocks2(*self.layer3)

        self.layer2 = []
        for k in range(B2):
            self.layer2.append(ConvBNReLU(DC2, DC2, 3))
        self.layer2 = ResBlocks2(*self.layer2)

        self.layer1 = []
        for k in range(B1):
            self.layer1.append(ConvBNReLU(DC1, DC1, 3))
        self.layer1 = ResBlocks2(*self.layer1)
        # self.segment_head = ConvBNReLU(DC1, num_classes, 3)
        self.segment_head = nn.Conv2d(DC1, num_classes, 3, 1, 1)

    def forward(self, x0, x1, x2, x3, x4):
        x = F.interpolate(self.f4(x4), size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer4(x+self.n3(x3))

        x = F.interpolate(self.f3(x), size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer3(x+self.n2(x2))

        x = F.interpolate(self.f2(x), size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer2(x+self.n1(x1))
        
        x = F.interpolate(self.f1(x), size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer1(x+self.n0(x0))
        return self.segment_head(x)


class EncoderDecoder(nn.Module):
    def __init__(self, 
                 backbone, 
                 decoder,
                 DecoderTuple=False):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.DecoderTuple = DecoderTuple

    def forward(self, x, reture_features=False):
        fs = self.backbone(x)
        if self.DecoderTuple:
            x = self.decoder(fs)
        else:
            x = self.decoder(*fs)
        if reture_features:
            return x, fs
        return x






class ResSegNet(EncoderDecoder):
    def __init__(self,
                 backbone_name='resnet50',
                 decoder_class=FPNDecoder,
                 backbone_args={},
                 decoder_args=fpn_decoder_args_1,
                 features_only=True,
    ):
        backbone = ResNetEncoder(backbone_name,features_only=features_only, **backbone_args)
        l_chs = backbone.channels
        decoder = decoder_class(l_chs, return_context=False,**decoder_args)
        super().__init__(backbone, decoder)

    def pretrain_backbone(self, d:dict, from_where='resnet'):
        self.backbone.pretrain(d, from_where)





import os
from jscv.datasets.gid_water import train_mean, train_std
import albumentations as albu
from PIL import Image
import numpy as np

class water_seg_dataset():
    def __init__(self, datadir='data/gid_water/val') -> None:
        self.datadir = datadir

        self.aug = albu.Normalize(train_mean, train_std, max_pixel_value=255)    
        self.listdir = os.listdir(self.datadir+"/image")
        self.len = len(self.listdir)

    def batch(self):
        self.time_count = 0

        for fi, imgf in enumerate(self.listdir):
            t0 = time()
            id = os.path.splitext(imgf)[0]
            labelf = id + "_label.tif"
            f_img = self.datadir+"/image/"+imgf
            f_label = self.datadir+"/label/"+labelf
            img = np.array(Image.open(f_img).convert("RGB"))
            label = np.array(Image.open(f_label).convert("L"))
            aug_res = self.aug(image=img, mask=label)
            img = torch.from_numpy(aug_res['image']).permute(2, 0, 1).float().unsqueeze(0)
            label = torch.from_numpy(aug_res['mask']).unsqueeze(0)
            t0 = self.time_count_once = time()-t0
            self.time_count += t0
            yield img, label, id


rgb_green     =   [0, 255, 0]
rgb_yellow    =   [255, 255, 0]
rgb_blue      =   [0, 0, 255]
rgb_red       =   [255, 0, 0]
rgb_purple    =   [255, 0, 255]
rgb_gray      =   [128,128,128]
def rgb_to_bgr(c):
    r,g,b = c
    return [b,g,r]
bgr_green = rgb_to_bgr(rgb_green)
bgr_yellow = rgb_to_bgr(rgb_yellow)
bgr_blue = rgb_to_bgr(rgb_blue)
bgr_red = rgb_to_bgr(rgb_red)
bgr_purple = rgb_to_bgr(rgb_purple)

def color_tensor(color, t1):
    return torch.tensor(color, dtype=t1.dtype, device=t1.device)

def gray_map_strengthen(gray: np.ndarray):
    gray = gray.astype(np.float64)
    gray = gray - np.min(gray)
    rate = 255.0 / np.max(gray)
    gray = gray * rate
    # print(np.max(gray), np.average(gray))
    gray = gray.astype(np.uint8)
    return gray

'''
    只保存(0,1)图
    TODO 保存原图
    #TODO 代码很乱，需要重构
'''
class SaveImagesManager():
    def __init__(self,
                 action_type='default',
                 per_n_step=15, #? 每 n个batches，保存一次图像。 如果 per_n_step=-1，不保存
                 use_sub_dir=True,
                 save_img=False,  #? 原图比较占存储,要存图置: save_img=True
                 save_wrong=True,
                 save_pred=True,
                 save_label=True,
                 save_coarse_pred=False,
                 save_loss_map=False,
                 add_boxes=None,
                 all_do_hardpathes=False,
                 crop_sub_images=True,
                 min_thick_rate=0.05, 
                 max_thick_rate=0.15,
                 line_width=10,
                 jpeg_ratio=4,) -> None:
        self.per_n_step = per_n_step
        self.count_batch = 0
        self.img_idx = 0
        self.save_images_dir = None
        self.not_create_save_dir = True
        self.jpeg_ratio = jpeg_ratio
        self.save_img = save_img
        self.save_wrong = save_wrong
        self.save_pred = save_pred
        self.save_label = save_label
        self.add_boxes = add_boxes
        self.use_sub_dir = use_sub_dir
        self.save_loss_map = save_loss_map
        self.save_coarse_pred = save_coarse_pred
        self.all_do_hardpathes = all_do_hardpathes
        self.line_width = line_width
        self.action_type = action_type
        self.min_thick_rate = min_thick_rate
        self.max_thick_rate = max_thick_rate
        self.crop_sub_images = crop_sub_images
        if add_boxes is not None:
            self.hard_tensor = torch.zeros(add_boxes)
            self.label_hard_tensor = torch.zeros(add_boxes)
    
    def step(self):
        '''
            training
        '''
        self.count_batch += 1
        do_save_img = False
        if self.per_n_step > 0:
            if self.not_create_save_dir:
                if 'cfg' in global_dict:
                    self.cfg = global_dict['cfg']
                    self.save_dir = os.path.join(global_dict['cfg'].workdir, \
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                    os.makedirs(self.save_dir)
                    self.not_create_save_dir = False
                    print("created images save dir:", self.save_dir)
            if self.count_batch % self.per_n_step == 0:
                self.img_idx += 1
                do_save_img = True
        return do_save_img


    def unnormalize(self, img, mean, std):
        mean = torch.tensor(mean, device=img.device).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor(std, device=img.device).unsqueeze(-1).unsqueeze(-1)
        return ((img * std + mean)*255).permute(1,2,0)
    
    def convert_image(self, image):
        #TODO 弃用
        # image = F.interpolate(image, scale_factor=1/2, mode="bilinear")
        ds = global_dict['cfg'].train_dataset
        image = cv2.cvtColor(self.unnormalize(image, ds.mean, ds.std).cpu().numpy(), cv2.COLOR_RGB2BGR)
        return image, [int(cv2.IMWRITE_JPEG_QUALITY),self.jpeg_ratio]

    def convert_image_new(self, image):
        ds = global_dict['cfg'].train_dataset
        img_tensor = self.unnormalize(image, ds.mean, ds.std)
        #TODO 改
        img_tensor = torch.tensor(cv2.cvtColor(img_tensor.cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR))
        return img_tensor

    def save_org_image(self, fname, img_tensor):
        # print(img_tensor.shape)
        cv2.imwrite(fname, img_tensor.cpu().numpy(), [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_ratio])

    def create_prefix(self, img_id):
        if img_id is None:
            img_id = ''
        else:
            img_id = '_'+img_id
        stage = global_dict['trainer'].stage[0]
        self.pfx = f'{self.img_idx}{stage}_'
        self.pfx_full = self.save_dir + '/' + self.pfx
        self.img_id = img_id
    def jpg(self, name):
        return self.filepath(name, '.jpg')

    def png(self, name):
        return self.filepath(name, '.png')

    def filepath(self, name, subfix=''):
        if self.use_sub_dir:
            subdir = self.save_dir + '/'  + self.pfx + self.img_id
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            return subdir + '/'  + name + subfix
        return self.pfx_full + name + self.img_id + subfix

    def save_image(self, image: torch.Tensor,
                   pred: torch.Tensor,
                   mask: torch.Tensor,
                   img_id=None,
                   coarse_pred=None,
                   pred_logits=None,
                   ):
        torch.cuda.empty_cache()
        if self.add_boxes is not None:
            self.save_image_hard(
                image, pred, mask, self.hard_tensor, self.label_hard_tensor, img_id, all_do=True)
            return

        ignore_idx = global_dict['cfg'].ignore_index
        self.create_prefix(img_id)

 
        if pred_logits is not None and self.save_loss_map:
            lossmap = loss_map(pred_logits, mask, ignore_index=self.cfg.ignore_index,
                               pred_logits=True, dim=0).cpu().numpy()
            lossmap = gray_map_strengthen(lossmap)
            lossmap = cv2.applyColorMap(lossmap, cv2.COLORMAP_JET)
            cv2.imwrite(self.png('lossmap'), lossmap)
            del lossmap, pred_logits

        if self.save_img and image is not None:
            cv2.imwrite(self.jpg('img'), *self.convert_image(image))

        ignore_range = mask == ignore_idx
        mask = mask.float()
        if self.save_label:
            # print((mask==ignore_idx).sum())
            mask[ignore_range] = 0.5
            cv2.imwrite(self.png('label'), mask.cpu().numpy()*255)

        if coarse_pred is not None and self.save_coarse_pred:
            cv2.imwrite(self.png('coarse_pred'), coarse_pred.cpu().numpy()*255)
            del coarse_pred

        if pred is not None:
            # pred = pred.softmax(dim=0).argmax(dim=0)
            if self.save_pred:
                cv2.imwrite(self.png('pred'), pred.cpu().numpy()*255)
                pass
            if self.save_wrong:
                H,W = mask.shape
                mask = mask.to(pred.device)
                wrong_mask = ~ignore_range & (pred!=mask)
                wrong = torch.zeros(H,W,3, device=pred.device)
                wrong[wrong_mask] = color_tensor(bgr_red,wrong)
                wrong[ignore_range] = color_tensor(rgb_gray,wrong)
                # cv2.imwrite(self.png('wrong_alone'), wrong.cpu().numpy())
                
                wrong_rate = round(float(torch.sum(wrong_mask)) / (H*W) * 100, 2)
                NN = self.pfx + f'wrong_{wrong_rate}%_' + self.img_id + '.png'

                mask = (mask*255).unsqueeze(-1).repeat(1,1,3)                
                # print('mask.shape', mask.shape)
                mask[wrong_mask] = color_tensor(bgr_red,wrong)
                cv2.imwrite(NN, mask.cpu().numpy())


    def save_image_hard(self, image: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor,
                         hard: torch.Tensor, label_hard: torch.Tensor, img_id,
                         coarse_pred=None, Pd=10, **kargs):
        '''
            #TODO 太乱了，懒得改
            同时在mask和wrong叠加, 不再保存原mask和wrong
            mask:H,W, hard:(8,8)
        '''

        if self.action_type == 'save_pdt':
            return self.save_pdt_action(image=image, mask=mask, pred=pred,
                                        hard=hard, img_id=img_id, 
                                        coarse_pred=coarse_pred, **kargs)

        # torch.cuda.empty_cache()f
        H,W = mask.shape
        PH, PW = hard.shape
        h1, w1 = H//PH, W//PW
        self.create_prefix(img_id)

        do_paint_1 = True
        # do_paint_2 = True
        img_do = False
        all_do = True #self.all_do_hardpathes


        if self.save_img and image is not None:
            cv2.imwrite(self.jpg('1_image'), *self.convert_image(image))
            if all_do:
                ds = global_dict['cfg'].train_dataset
                image = self.unnormalize(image, ds.mean, ds.std)
                img_do = True
            # else:

        ignore_idx = global_dict['cfg'].ignore_index
        ignore_range = mask == ignore_idx
        mask = mask.float()
        mask[ignore_range] = 0.5    # [0, 1, 0.5]
        # pred[ignore_range] = 0.5    # [0, 1]
        # print(mask.shape, mask.dtype)

        # pred = pred.softmax(dim=0).argmax(dim=0)
        pred_do = False
        if self.save_pred:
            pred_do = all_do
            if not all_do:
                cv2.imwrite(self.png('4_refined'), pred.cpu().numpy()*255)

        mask_rgb = mask.unsqueeze(-1).repeat(1,1,3).reshape(PH,h1,PW,w1,3).transpose(1, 2)*255

        H,W = mask.shape
        mask = mask.to(pred.device)
        wrong = torch.zeros(H,W,3, dtype=mask_rgb.dtype ,device=pred.device)
        wrong_2 = torch.zeros(H,W,3, dtype=mask_rgb.dtype ,device=pred.device)


        #if coarse_pred is not None:
        wrong[coarse_pred!=mask] = color_tensor(bgr_red,wrong)
        #else:
        wrong_2[pred!=mask] = color_tensor(bgr_red,wrong_2)
        
        wrong[ignore_range] = color_tensor(rgb_gray,wrong)
        wrong_2[ignore_range] = color_tensor(rgb_gray,wrong_2)
        wrong = wrong.reshape(PH,h1,PW,w1,3).transpose(1, 2)
        wrong_2 = wrong_2.reshape(PH,h1,PW,w1,3).transpose(1, 2)

        
        # hard, label_hard, mask_rgb = hard.cuda(), label_hard.cuda(), mask_rgb.cuda()


        #! simple demo
        label_hard = hard
        easy = ~hard
        easy_right = easy & ~label_hard     #绿
        easy_wrong = easy & label_hard      #紫
        hard_right = hard & label_hard      #蓝
        hard_wrong = hard & ~label_hard     #黄
        range_pad = torch.zeros(1,1,h1,w1).to(mask_rgb.device)
        range_pad[:,:,:Pd,] = 1
        range_pad[:,:,-Pd:,] = 1
        range_pad[:,:,:,:Pd] = 1
        range_pad[:,:,:,-Pd:] = 1
        easy_right = easy_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        easy_wrong = easy_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        hard_right = hard_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        hard_wrong = hard_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()

        green = color_tensor(bgr_green,     mask_rgb)
        purple = color_tensor(bgr_purple,   mask_rgb)
        blue = color_tensor(bgr_blue,       mask_rgb)
        yellow = color_tensor(bgr_yellow,   mask_rgb)
        def paint(x, r=True):
            if r:
                x[easy_right] = green
                x[easy_wrong] = purple
                x[hard_right] = blue
                x[hard_wrong] = yellow
            return x.transpose(1, 2).reshape(H,W,3).cpu().numpy()


        cv2.imwrite(self.png('2_label'), (mask*255).int().cpu().numpy())
        # mask_rgb = paint(mask_rgb)
        # cv2.imwrite(self.png('mask_easyhard'), mask_rgb)

        wrong = paint(wrong, do_paint_1)
        wrong_2 = paint(wrong_2, do_paint_1)
        

        cv2.imwrite(self.png('5_wrong_coarse_patches'), wrong)
        cv2.imwrite(self.png('6_wrong_refine_patches'), wrong_2)
        
        del mask_rgb, wrong, wrong_2
        
        if img_do:
            image = image.reshape(PH,h1,PW,w1,3).transpose(1, 2)
            image[easy_right] = color_tensor(rgb_green,   image)
            image[easy_wrong] = color_tensor(rgb_purple,  image)
            image[hard_right] = color_tensor(rgb_blue,    image)
            image[hard_wrong] = color_tensor(rgb_yellow,  image)
            image = image.transpose(1, 2).reshape(H,W,3).cpu().numpy()
            # print(image.shape,end='  ')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # print(image.shape)
            cv2.imwrite(self.jpg('_8_img__patches'), image, [int(cv2.IMWRITE_JPEG_QUALITY),self.jpeg_ratio])
        if pred_do:
            pred = pred.unsqueeze(-1).repeat(1,1,3).reshape(
                PH,h1,PW,w1,3).transpose(1, 2)*255
            pred = paint(pred, False)
            cv2.imwrite(self.png('4_refined'), pred)

        if coarse_pred is not None and self.save_coarse_pred:
            
            if all_do:
                coarse_pred = coarse_pred.unsqueeze(-1).repeat(1,1,3).reshape(
                    PH,h1,PW,w1,3).transpose(1, 2)*255
                coarse_pred = paint(coarse_pred.float(), True)
                cv2.imwrite(self.png('3_coarse_patches'), coarse_pred)
            else:
                cv2.imwrite(self.png('3_coarse'), coarse_pred.cpu().numpy()*255)

    def pdt_boxes_init(self, mask, hard, label_hard=None):
        if label_hard is None:
            label_hard = hard
        easy = ~hard
        #TODO 
        #? 现在只保存 蓝色框
        # easy_right = easy & ~label_hard     #绿
        # easy_wrong = easy & label_hard      #紫
        hard_right = hard & label_hard      #蓝
        # hard_wrong = hard & ~label_hard     #黄
        
        Pd = self.line_width
        H,W = mask.shape
        PH, PW = hard.shape
        h1, w1 = H//PH, W//PW

        range_pad = torch.zeros(1,1,h1,w1).to(mask.device)
        range_pad[:,:,:Pd,] = 1
        range_pad[:,:,-Pd:,] = 1
        range_pad[:,:,:,:Pd] = 1
        range_pad[:,:,:,-Pd:] = 1
        # easy_right = easy_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        # easy_wrong = easy_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        hard_right = hard_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        # hard_wrong = hard_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()

        # green = color_tensor(bgr_green,     mask)
        # purple = color_tensor(bgr_purple,   mask)
        blue = color_tensor(bgr_blue,       mask)
        # yellow = color_tensor(bgr_yellow,   mask)
        
        # self.pdt_boxes_init_data = (easy_right, easy_wrong, hard_right, hard_wrong,
        #                             green, purple, blue, yellow, PH, PW)
        self.pdt_boxes_init_data = (hard_right, blue, PH, PW)
    
    
    def pdt_boxes_paint(self, x: torch.Tensor, do_paint=True):
        if not do_paint: return x
        hard_right, blue, PH, PW = self.pdt_boxes_init_data

        if x.dim() == 2:
            x = x.unsqueeze(-1).repeat(1,1,3)

        H, W, _ = x.shape
        x = x.reshape(PH,H//PH,PW,W//PW,3).transpose(1, 2)
        # x[easy_right] = green
        # x[easy_wrong] = purple
        x[hard_right] = blue
        # x[hard_wrong] = yellow
        return x.transpose(1, 2).reshape(H,W,3)


    def save_pdt_action(self, image, mask: torch.Tensor, hard: torch.Tensor, 
                        pred, coarse_pred, img_id, PDT: torch.Tensor, **kargs):
        '''
            保存 pdt图、wrong_map 的 方法
        '''
        # torch.cuda.empty_cache()
        H,W = mask.shape
        PH, PW = PDT.shape
        h1, w1 = H//PH, W//PW
        self.create_prefix(img_id)
        self.pdt_boxes_init(pred, hard)
        if self.save_img and image is not None:
            image = self.convert_image_new(image)
            self.save_org_image(self.jpg('0_1_image'), image)
            self.save_org_image(self.jpg('0_2_image_div_(simple_div)'), 
                                self.pdt_boxes_paint(image.to(pred.device).long()))

        ''' label '''
        ignore_idx = global_dict['cfg'].ignore_index
        ignore_range = mask == ignore_idx
        mask = mask.float()
        mask[ignore_range] = 0.5
        mask = mask.to(coarse_pred.device)
        mask255 = mask*255
        cv2.imwrite(self.png('1_1_label'), mask255.cpu().numpy())
        cv2.imwrite(self.png('1_2_label_div_(simple_div)'), 
                    self.pdt_boxes_paint(mask255.long()).cpu().numpy())

        ''' wrong '''
        def get_wrong(pred, base_map: torch.Tensor=None):
            if base_map is None:
                base_map = torch.zeros(H,W,3, dtype=pred.dtype, device=pred.device)
            else:
                if base_map.dim() == 2:
                    base_map = base_map.unsqueeze(-1).repeat(1,1,3)*255
                else:
                    base_map = base_map.clone()

            base_map[pred!=mask] = color_tensor(bgr_red,base_map)
            base_map[ignore_range] = color_tensor(rgb_gray,base_map)
            return base_map

        if self.save_pred:
            refined = pred*255
            cv2.imwrite(self.png('2_1_refined_(simple_div)'), (refined).cpu().numpy())
            refined_w = get_wrong(pred, pred)
            cv2.imwrite(self.png('2_2_refined_w_(simple_div)'), (refined_w).cpu().numpy())
            cv2.imwrite(self.png('2_4_r_div_(simple_div)'), self.pdt_boxes_paint(refined).cpu().numpy())
            cv2.imwrite(self.png('2_5_rw_div_(simple_div)'), self.pdt_boxes_paint(refined_w).cpu().numpy())
        refined_wrong = get_wrong(pred)
        cv2.imwrite(self.png('2_3_refined_wrong(simple_div)'), (refined_wrong).cpu().numpy())
        cv2.imwrite(self.png('2_6_r_wrong_div_(simple_div)'), self.pdt_boxes_paint(refined_wrong).cpu().numpy())

        if self.save_coarse_pred:
            coarse = coarse_pred*255
            cv2.imwrite(self.png('3_1_coarse'), coarse.cpu().numpy())
            coarse_w = get_wrong(coarse_pred, coarse_pred)
            cv2.imwrite(self.png('3_2_coarse_w'), (coarse_w).cpu().numpy())
            cv2.imwrite(self.png('3_4_c_div_(simple_div)'), self.pdt_boxes_paint(coarse).cpu().numpy())
            cv2.imwrite(self.png('3_5_cw_div_(simple_div)'), self.pdt_boxes_paint(coarse_w).cpu().numpy())
        coarse_wrong = get_wrong(coarse_pred)
        cv2.imwrite(self.png('3_3_coarse_wrong'), coarse_wrong.cpu().numpy())
        cv2.imwrite(self.png('3_6_c_wrong_div_(simple_div)'), self.pdt_boxes_paint(coarse_wrong).cpu().numpy())

        PDT_map = PDT.reshape(PH, 1, PW, 1).repeat(1, h1, 1, w1).reshape(PH*h1, PW*w1)*255

        cv2.imwrite(self.png('4_1_PDT'), PDT_map.cpu().numpy())

        cv2.imwrite(self.png('4_2_PDT_div_(simple_div)'), self.pdt_boxes_paint(PDT_map.long()).cpu().numpy())

        torch.save(PDT, self.filepath(f'PDT({PH}x{PW})', '.tensor'))

    def paint_dynamic_grouping_list_on_image(self, image: torch.Tensor, batch_grouping_list, 
                                             manager, global_patch_size, RGB2BGR=False, fill_mode=False):
        from jscv.hr_models.DPRNet.dynamic_patches import paint_on_image, DynamicPatchesGrouping
        assert isinstance(manager, DynamicPatchesGrouping)
        PH, PW = global_patch_size
        if image.dim() == 3:
            H, W, C = image.shape
            assert C == 3
            H2, W2 = H//PH, W//PW
            image = image.reshape(PH, H2, PW, W2, C).transpose(1, 2).reshape(-1, H2, W2, C)
        else:
            H, W = image.shape
            H2, W2 = H//PH, W//PW
            image = image.reshape(PH, H2, PW, W2).transpose(1, 2).reshape(-1, H2, W2)
        images = torch.unbind(image)
        L = []
        croped_list = []
        for im, batch in zip(images, batch_grouping_list):
            im = paint_on_image(im, batch, manager, self.min_thick_rate, 
                                self.max_thick_rate, RGB2BGR=RGB2BGR, 
                                do_crop=self.crop_sub_images, fill_mode=fill_mode)
            if self.crop_sub_images:
                im, cropL = im
                croped_list = croped_list + cropL
            L.append(im)
        image = torch.stack(L).reshape(PH, PW, H2, W2, 3).transpose(1, 2).reshape(H, W, 3)
        if self.crop_sub_images:
            self.croped_list = croped_list
        return image

    def dynamic_patch_image_saving(self, image, mask, coarse_pred, pred, PDT, imgid, manager,
                                   hard_pred, global_patch_size, batch_grouping_list, ):
        '''
            大部分 不保存1x1方式的划分了
        '''
        from jscv.hr_models.DPRNet.dynamic_patches import paint_on_image, DynamicPatchesGrouping
        
        H,W = mask.shape
        PH, PW = PDT.shape
        h1, w1 = H//PH, W//PW
        self.create_prefix(imgid)
        self.pdt_boxes_init(pred, hard_pred)

        def get_dynamic(image, RGB2BGR=True, fill_mode=False):
            return self.paint_dynamic_grouping_list_on_image(
                image, batch_grouping_list, manager, global_patch_size, RGB2BGR=RGB2BGR, fill_mode=fill_mode)
        
        def save_croped_list_png(dir_name):
            dir0 = self.filepath(dir_name + '_croped')
            os.makedirs(dir0)
            for j, crop in enumerate(self.croped_list):
                self.save_org_image(os.path.join(dir0, f'{j}.png'), crop)

        if self.save_img and image is not None:
            image = self.convert_image_new(image)
            self.save_org_image(self.jpg('0_image'), image)
            self.save_org_image(self.jpg('0_image(dynamic)'), get_dynamic(image))
            dir0 = self.filepath('0_image_croped')
            os.makedirs(dir0)
            for j, crop in enumerate(self.croped_list):
                self.save_org_image(os.path.join(dir0, f'{j}.jpg'), crop)

        ''' label '''
        ignore_idx = global_dict['cfg'].ignore_index
        ignore_range = mask == ignore_idx
        mask = mask.float()
        mask[ignore_range] = 0.5
        mask = mask.to(coarse_pred.device)
        mask255 = mask*255
        cv2.imwrite(self.png('1_label'), mask255.cpu().numpy())
        cv2.imwrite(self.png('1_label(dynamic)'), get_dynamic(mask255).cpu().numpy())
        save_croped_list_png('1_label')

        ''' wrong '''
        def get_wrong(pred, base_map: torch.Tensor=None):
            if base_map is None:
                base_map = torch.zeros(H,W,3, dtype=pred.dtype, device=pred.device)
            else:
                if base_map.dim() == 2:
                    base_map = base_map.unsqueeze(-1).repeat(1,1,3)*255
                else:
                    base_map = base_map.clone()

            base_map[pred!=mask] = color_tensor(bgr_red,base_map)
            base_map[ignore_range] = color_tensor(rgb_gray,base_map)
            return base_map

        if self.save_pred:
            refined = pred*255
            cv2.imwrite(self.png('2_1_refined'), (refined).cpu().numpy())
            cv2.imwrite(self.png('2_1_refined(dynamic)'), get_dynamic(refined).cpu().numpy())
            save_croped_list_png('2_1_refined')
            refined_w = get_wrong(pred, pred)
            cv2.imwrite(self.png('2_2_refined_w'), (refined_w).cpu().numpy())
            cv2.imwrite(self.png('2_2_refined_w(dynamic)'), get_dynamic(refined_w).cpu().numpy())
            save_croped_list_png('2_2_refined_w')
        refined_wrong = get_wrong(pred)
        cv2.imwrite(self.png('2_3_refined_wrong'), (refined_wrong).cpu().numpy())
        cv2.imwrite(self.png('2_3_refined_wrong(dynamic)'), get_dynamic(refined_wrong).cpu().numpy())
        save_croped_list_png('2_3_refined_wrong')
        # cv2.imwrite(self.png('2_3_refined_wrong(simple)'), self.pdt_boxes_paint(refined_wrong).cpu().numpy())

        if self.save_coarse_pred:
            coarse = coarse_pred*255
            cv2.imwrite(self.png('3_1_coarse'), coarse.cpu().numpy())
            cv2.imwrite(self.png('3_1_coarse(dynamic)'), get_dynamic(coarse).cpu().numpy())
            save_croped_list_png('3_1_coarse')
            coarse_w = get_wrong(coarse_pred, coarse_pred)
            cv2.imwrite(self.png('3_2_coarse_w'), (coarse_w).cpu().numpy())
            cv2.imwrite(self.png('3_2_coarse_w(dynamic)'), get_dynamic(coarse_w).cpu().numpy())
            save_croped_list_png('3_2_coarse_w')
        coarse_wrong = get_wrong(coarse_pred)
        cv2.imwrite(self.png('3_3_coarse_wrong'), coarse_wrong.cpu().numpy())
        cv2.imwrite(self.png('3_3_coarse_wrong(dynamic)'), get_dynamic(coarse_wrong.clone()).cpu().numpy())
        cv2.imwrite(self.png('3_3_coarse_wrong(simple_div)'), self.pdt_boxes_paint(coarse_wrong).cpu().numpy())
        save_croped_list_png('3_3_coarse_wrong')

        PDT_map = PDT.detach().reshape(PH, 1, PW, 1).repeat(1, h1, 1, w1).reshape(PH*h1, PW*w1)*255

        cv2.imwrite(self.png('4_1_PDT'), PDT_map.cpu().numpy())
        cv2.imwrite(self.png('4_1_PDT(dynamic)'), get_dynamic(PDT_map.clone()).cpu().numpy())
        cv2.imwrite(self.png('4_1_PDT(simple_div)'), self.pdt_boxes_paint(PDT_map.long()).cpu().numpy())
        save_croped_list_png('4_1_PDT')

        gray_map = torch.ones_like(PDT_map, device=PDT_map.device) * 100
        gray_map = self.paint_dynamic_grouping_list_on_image(
            gray_map, batch_grouping_list, manager, global_patch_size, RGB2BGR=True, fill_mode=True)
        cv2.imwrite(self.png('5_feature(dynamic)'), gray_map.cpu().numpy())
        save_croped_list_png('5_feature')

        torch.save(PDT, self.filepath(f'PDT({PH}x{PW})', '.tensor'))
        self.croped_list = None


    #TODO!!!
    def save_image_hard_2(self, image: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor,
                         hard: torch.Tensor, label_hard: torch.Tensor, cls_pred_wrong,
                         img_id, Pd=40, save_mask=True):

        t0 = time()
        H,W = mask.shape
        PH, PW = hard.shape
        h1, w1 = H//PH, W//PW
        stage = global_dict['trainer'].stage[0]
        save_prefix = self.save_dir+f'/{self.img_idx}{stage}_'
        
        if image is not None:
            cv2.imwrite(save_prefix+f"img_{img_id}.jpg", *self.convert_image(image))
            del image
        
        if pred is not None:
            pred = pred.softmax(dim=0).argmax(dim=0)
            cv2.imwrite(save_prefix+f"pred_{img_id}.png", pred.cpu().numpy()*255)
            del pred
        if save_mask:
            cv2.imwrite(save_prefix+f"mask_{img_id}.png", mask.cpu().numpy()*255)
        
        mask_rgb = mask.unsqueeze(-1).repeat(1,1,3).reshape(
            PH,h1,PW,w1,3).transpose(1, 2)*255
        
        
        hard, label_hard, mask_rgb = hard.cuda(), label_hard.cuda(), mask_rgb.cuda()
        cls_pred_wrong = cls_pred_wrong.cuda()

        easy = ~hard
        easy_right = easy & ~cls_pred_wrong & ~label_hard    #绿
        easy_wrong = easy & ~cls_pred_wrong & label_hard     #红
        easy_cls_wrong = easy & cls_pred_wrong  #紫 -- 严重
        hard_right = hard & label_hard      #蓝
        hard_wrong = hard & ~label_hard     #黄
        range_pad = torch.zeros(1,1,h1,w1).to(mask_rgb.device)
        range_pad[:,:,:Pd,] = 1
        range_pad[:,:,-Pd:,] = 1
        range_pad[:,:,:,:Pd] = 1
        range_pad[:,:,:,-Pd:] = 1
        easy_right = easy_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        easy_wrong = easy_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        easy_cls_wrong = easy_cls_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        hard_right = hard_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
        hard_wrong = hard_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()

        mask_rgb[easy_right] = torch.tensor(bgr_green).to(mask_rgb.device).long()
        mask_rgb[easy_wrong] = torch.tensor(bgr_red).to(mask_rgb.device).long()
        mask_rgb[easy_cls_wrong] = torch.tensor(bgr_purple).to(mask_rgb.device).long()
        mask_rgb[hard_right] = torch.tensor(bgr_blue).to(mask_rgb.device).long()
        mask_rgb[hard_wrong] = torch.tensor(bgr_yellow).to(mask_rgb.device).long()
        mask_rgb = mask_rgb.transpose(1, 2).reshape(H,W,3).cpu().numpy()

        # wrong = (pred != mask).int()
        
        
        cv2.imwrite(save_prefix+f"mask_hard_map_{img_id}.png", mask_rgb)
        
        if do_debug_time:
            print("save image, spend:", time()-t0)

    # def save_image_hard_2(self, pred: torch.Tensor, mask: torch.Tensor,
    #                      hard: torch.Tensor, label_hard: torch.Tensor):
        
    #     t0 = time()
    #     _, H,W = pred.shape
    #     PH, PW = hard.shape
    #     h1, w1 = H//PH, W//PW
        
    #     Pd = 40
        
    #     save_prefix = self.save_dir+f'/{self.img_idx}_'
        
    #     pred = pred.softmax(dim=0).argmax(dim=0)
        
    #     mask_1 = mask.unsqueeze(-1).repeat(1,1,3).reshape(PH,h1,PW,w1,3).transpose(1, 2)*255
    #     hard, label_hard = hard.cuda(), label_hard.cuda()

    #     easy = ~hard
    #     easy_right = easy & ~label_hard    #绿
    #     easy_wrong = easy & label_hard     #红
        
    #     hard_right = hard & label_hard      #蓝
    #     hard_wrong = hard & ~label_hard     #黄
    #     range_pad = torch.zeros(1,1,h1,w1).cuda()
    #     range_pad[:,:,:Pd,] = 1
    #     range_pad[:,:,-Pd:,] = 1
    #     range_pad[:,:,:,:Pd] = 1
    #     range_pad[:,:,:,-Pd:] = 1
    #     easy_right = easy_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
    #     easy_wrong = easy_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
    #     hard_right = hard_right.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()
    #     hard_wrong = hard_wrong.unsqueeze(-1).unsqueeze(-1) & range_pad.bool()

    #     mask_rgb[easy_right] = torch.tensor(bgr_green).cuda().long()
    #     mask_rgb[easy_wrong] = torch.tensor(bgr_red).cuda().long()
    #     mask_rgb[hard_right] = torch.tensor(bgr_blue).cuda().long()
    #     mask_rgb[hard_wrong] = torch.tensor(bgr_yellow).cuda().long()
    #     mask_rgb = mask_rgb.transpose(1, 2).reshape(H,W,3).cpu().numpy()

    #     # wrong = (pred != mask).int()
    #     cv2.imwrite(save_prefix+"pred.png", pred.cpu().numpy()*255)
    #     cv2.imwrite(save_prefix+"mask.png", mask.cpu().numpy()*255)
    #     cv2.imwrite(save_prefix+"mask_hard_map.png", mask_rgb)
        
    #     if do_debug_time:
    #         print("save image, spend:", time()-t0)


def resize_to_y(x, y):
    if x.shape[-2:] != y.shape[-2:]:
        x = F.interpolate(x, y.shape[-2:], mode="bilinear", align_corners=False)
    return x

def resize_to(x, r):
    if r != 1:
        x = F.interpolate(x, scale_factor=r, mode="bilinear", align_corners=False)
    return x


if __name__ == "__main__":
    coun_1.DO_DEBUG = True
    B = ResNetEncoder('resnet18', ).cuda()
    x = torch.rand(1, 3, 1024*7, 1024*7).cuda()
    N = 20
    
    warmup()
    
    for i in range(N):
        _,x = B.forward_test()
    print(coun_1.str_total_porp())
    


