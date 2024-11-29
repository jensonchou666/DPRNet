# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from jscv.models.cnn import *
from jscv.models.utils import UpSample

import jscv.utils.analyser as ana

from jscv.models.refine_vit.refine_vit_v2 import *



class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 refine_layer_args,
                 in_index=[0, 1, 2, 3],
                 feature_strides=[4, 8, 16, 32],
                 num_classes=19,
                 dropout_ratio=0.1,
                 conv_cfg=dict(norm_layer=nn.BatchNorm2d,
                               act_layer=nn.ReLU6,
                               bias=False),
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 **kwargs):
        super().__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_channels = in_channels
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.upsample_cfg = upsample_cfg

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(1, int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                inc = self.in_channels[i] if k == 0 else self.channels
                scale_head.append(ConvBNReLU(inc, self.channels, 3, **conv_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(UpSample(scale_factor=2, **upsample_cfg))

            self.scale_heads.append(nn.Sequential(*scale_head))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            
        ''' refine '''
        self.conv_seg_coarse = nn.Conv2d(channels, num_classes, kernel_size=1)
        # self.refine_layer = RefineVitLayer(in_channels=channels, **refine_layer_args)  #?old
        self.refine_layer = RefineVitLayer(in_channels=channels+num_classes, **refine_layer_args)
        channels = self.refine_layer.channels
        
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)


    def forward(self, inputs):
        res = [inputs[i] for i in self.in_index]
        # print("@@@@@@", [xx.shape for xx in x])
        x = self.scale_heads[0](res[0])
        for i in range(1, len(self.feature_strides)):
            size = x.shape[2:]
            # non inplace
            x = x + F.interpolate(self.scale_heads[i](res[i]),
                                  size=size, **self.upsample_cfg)

        coarse_pred = self.conv_seg_coarse(x)
        uncertain_map = self.refine_layer.create_uncertain_map(x, coarse_pred)
        # x = self.refine_layer(x, uncertain_map)
        x = self.refine_layer(torch.concat([x, coarse_pred], dim=1), uncertain_map)

        x = self.cls_seg(x)
        return dict(pred=x, coarse_pred_list=[coarse_pred])

    def cls_seg(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)


def FPNHead_128(num_classes, in_channels, **kargs):
    return FPNHead(
        in_channels=in_channels,
        channels=128,
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        num_classes=num_classes,
        dropout_ratio=0.1,
        **kargs
    )


def FPNHead_256(num_classes, in_channels, **kargs):
    return FPNHead(
        in_channels=in_channels,
        channels=256,
        refine_layer_args=refine_vit_layer_args_normal(112, 2, 0.36),
        # refine_layer_args=refine_vit_layer_args_normal(104, 2, 0.34),
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        num_classes=num_classes,
        dropout_ratio=0.1,
        **kargs
    )

def FPNHead_X(num_classes, in_channels, channels=256, **kargs):
    return FPNHead(
        in_channels=in_channels,
        channels=channels,
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        num_classes=num_classes,
        dropout_ratio=0.1,
        **kargs
    )


if __name__ == '__main__':
    in_channels = [256, 256, 256, 256]
    channels = 256
    num_outs = 4
    decode_head = FPNHead(in_channels, channels)
    import torch
    x1 = torch.randn(2, 256, 128, 128)
    x2 = torch.randn(2, 256, 64, 64)
    x3 = torch.randn(2, 256, 32, 32)
    x4 = torch.randn(2, 256, 16, 16)
    x = decode_head((x1, x2, x3, x4))
    print(x.shape)