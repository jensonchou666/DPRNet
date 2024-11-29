import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from jscv.models.cnn import *
from jscv.utils.utils import to_2tuple
from jscv.utils.statistics import StatisticModel, StatisticScale

def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act_layer=nn.ReLU, norm_layer=None, gate_layer=nn.Sigmoid):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

def create_attn(attn_type, chans):
    if attn_type == 'se':
        return SEModule(chans)
    else:
        assert attn_type is None or str.lower(attn_type) == "none"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, **kargs):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)
        #TODO
        #assert attn_layer is None, "暂不支持 attn_layer"
        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        if x.shape[1] == shortcut.shape[1]:
            x += shortcut
        x = self.act2(x)

        return x





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, **kargs):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        assert attn_layer is None, "暂不支持 attn_layer"
        self.se = None
        # self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        if x.shape[1] == shortcut.shape[1]:
            x += shortcut
        x = self.act3(x)

        return x

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding



def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])



def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7, gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-6)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x

class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False,
            batchwise: bool = False,
            fast: bool = True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)

def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'



class FusionModelV1(nn.Module):
    def __init__(self, encode_chan, decode_in_chan, decode_out_chan, eps=1e-8):
        super(FusionModelV1, self).__init__()
        self.pre_conv = ConvBN(encode_chan, decode_in_chan, kernel_size=3)
        self.pre_conv2 = ConvBNReLU(decode_in_chan, decode_in_chan, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32),
                                    requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_in_chan,
                                    decode_out_chan,
                                    kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * self.pre_conv2(x)
        x = self.post_conv(x)
        return x

class FusionModelV2(nn.Module):
    def __init__(self, encode_chan, decode_in_chan, decode_out_chan, channel_dim=1, eps=1e-8):
        super(FusionModelV2, self).__init__()
        self.conv = ConvBNReLU(encode_chan + decode_in_chan, decode_out_chan, kernel_size=3)
        self.dim = channel_dim

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        x = torch.concat([x, res], dim=self.dim)
        return self.conv(x)

class FusionModelV3(nn.Module):
    def __init__(self, encode_chan, decode_in_chan, decode_out_chan, eps=1e-8):
        super(FusionModelV3, self).__init__()
        self.pre_conv = ConvBNReLU(encode_chan, decode_in_chan, kernel_size=3)
        self.pre_conv2 = ConvBNReLU(decode_in_chan, decode_in_chan, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32),
                                    requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_in_chan,
                                    decode_out_chan,
                                    kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        attn = self.pre_conv(res) * self.pre_conv2(x)
        x = fuse_weights[0] * attn + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class FusionModelV4_FCN(nn.Module):
    def __init__(self, encode_chan, decode_in_chan, decode_out_chan):
        super(FusionModelV4_FCN, self).__init__()
        self.pre_conv = ConvBNReLU(decode_in_chan, decode_in_chan, kernel_size=3)
        self.pre_conv2 = ConvBNReLU(decode_in_chan, decode_out_chan, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        return self.pre_conv2(self.pre_conv(x))


class ResSequential(nn.Module):
    def __init__(self, blocks, input_features=False):
        super(ResSequential, self).__init__()
        self.down_block = blocks[0]
        self.blocks = nn.Sequential(*blocks[1:])
        self.input_features = input_features
        if input_features:
            self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32),
                                        requires_grad=True)
            self.eps = 1e-8

    def forward(self, x, res=None):
        x = self.down_block(x)
        if self.input_features and res is not None:
            weights = F.relu(self.weights)
            fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
            x = fuse_weights[0] * x + fuse_weights[1] * res
        return self.blocks(x)



def make_encode_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, input_features=False,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    num_stages = len(block_repeats)
    net_block_idx = 0
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'encode_layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        # if stage_idx + 1 == num_stages:
        #     dilation *= stride
        #     stride = 1

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append([stage_name, ResSequential(blocks, input_features)])
        feature_info.append(dict(num_chs=inplanes, module=stage_name))

    return stages, feature_info


def make_decode_blocks(
        block_fn, encode_chans, decode_blocks, decode_chans, reduce_first=1,
        fuse_model=FusionModelV2, drop_block_rate=0., **kwargs):
    # print("fuse_model:", fuse_model)
    stages = []
    fusions = []
    num_stages = len(encode_chans)
    in_chan = encode_chans[-1]

    encode_chans = reversed(encode_chans)
    decode_blocks = reversed(decode_blocks)
    decode_chans = reversed(decode_chans)

    for stage_idx, (encode_chan, num_blocks, decode_chan, db) in enumerate(zip(
        encode_chans, decode_blocks, decode_chans, reversed(drop_blocks(drop_block_rate)))):
        i = stage_idx
        stage_idx = num_stages - stage_idx - 1
        stage_name = f'decode_layer{stage_idx + 1}'
        block_kwargs = dict(reduce_first=reduce_first, dilation=1, drop_block=db, **kwargs)
        blocks = []

        if stage_idx < num_stages - 1:
            fusions.append([f'fusion_{stage_idx + 1}', fuse_model(encode_chan, prev_chan, decode_chan)])
            in_chan = decode_chan

        for block_idx in range(num_blocks):
            blocks.append(block_fn(
                in_chan, decode_chan, 1, None, first_dilation=1,
                drop_path=None, **block_kwargs))
            in_chan = decode_chan * block_fn.expansion

        prev_chan = in_chan

        
        stages.append([stage_name, nn.Sequential(*blocks)])

    return stages, fusions


class ResUNet(nn.Module):
    def __init__(
            self, block, encode_blocks, encode_chans, decode_blocks, decode_chans, in_chans,
            out_final_feature=True, input_features=False, cardinality=1, base_width=64, block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_path_rate=0., drop_block_rate=0., zero_init_last=True, **block_args):
        super(ResUNet, self).__init__()
        block_args = block_args or dict()
        assert len(encode_blocks) == len(encode_chans) == len(decode_blocks) == len(decode_chans)
        self.num_stages = len(encode_blocks)
        self.encode_blocks = encode_blocks
        self.encode_chans = encode_chans
        self.decode_blocks = decode_blocks
        self.decode_chans = decode_chans
        self.out_chans = decode_chans[0]
        self.input_features = input_features
        self.out_all_features = not out_final_feature

        self.encode_layers_name = []
        self.decode_layers_name = []
        self.fusion_layers_name = []

        stage_modules, stage_feature_info = make_encode_blocks(
            block, encode_chans, encode_blocks, in_chans, cardinality=cardinality, base_width=base_width,
            reduce_first=block_reduce_first, avg_down=avg_down,  input_features=input_features,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)
            self.encode_layers_name.append(stage[0])
        
        stage_modules, fusions = make_decode_blocks(
            block, encode_chans, decode_blocks, decode_chans, cardinality=cardinality, base_width=base_width,
            reduce_first=block_reduce_first, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)
            self.decode_layers_name.append(stage[0])
        for fusion in fusions:
            self.add_module(*fusion)
            self.fusion_layers_name.append(fusion[0])

        #self.init_weights(zero_init_last=zero_init_last)


    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    def forward(self, x):
        features = []
        out = []

        if self.input_features:
            fs = x
            x = fs[0]
            for i, name in enumerate(self.encode_layers_name):
                layer = self.__getattr__(name)
                if i > 0:
                    x = layer(x, fs[i])
                else:
                    x = layer(x)
                features.append(x)
        else:
            for name in self.encode_layers_name:
                layer = self.__getattr__(name)
                x = layer(x)
                features.append(x)

        for idx, (name, f) in enumerate(zip(self.decode_layers_name, reversed(features))):
            if idx > 0:
                fusion_name = self.fusion_layers_name[idx - 1]
                fusion = self.__getattr__(fusion_name)
                x = fusion(x, f)
            layer = self.__getattr__(name)
            x = layer(x)
            out.append(x)
        
        if self.out_all_features:
            out.reverse()
            return out
        else:
            return x

    def traverse(self, stat: StatisticModel, x):
        stat.statistic_self_alone()
        features = []
        out = []

        if self.input_features:
            fs = x
            x = fs[0]
            for i, name in enumerate(self.encode_layers_name):
                layer = self.__getattr__(name)
                if i > 0:
                    x = stat.step(layer, (x, fs[i]), name=name)
                else:
                    x = stat.step(layer, (x), name=name)
                features.append(x)
        else:
            for name in self.encode_layers_name:
                layer = self.__getattr__(name)
                x = stat.step(layer, (x), name=name)
                features.append(x)
        
        for idx, (name, f) in enumerate(zip(self.decode_layers_name, reversed(features))):
            if idx > 0:
                fusion_name = self.fusion_layers_name[idx - 1]
                fusion = self.__getattr__(fusion_name)
                x = stat.step(fusion, (x, f), name=fusion_name)
            layer = self.__getattr__(name)
            x = stat.step(layer, (x), name=name)
            out.append(x)
        
        if self.out_all_features:
            out.reverse()
            return out
        else:
            return x

class MUNet(nn.Module):
    def __init__(self, unet_list,
                 input_features=False,
                 out_final_feature=False,
                 out_dict=False):
        super(MUNet, self).__init__()
        self.num_unet = len(unet_list)
        for i, n in enumerate(unet_list):
            if i == 0:
                n.input_features = input_features
            else:
                n.input_features = True
            n.out_all_features = True

            self.add_module(f"unet_{i + 1}", n)
        self.out_all_features = not out_final_feature
        self.out_dict = out_dict
        self.out_chans = unet_list[-1].out_chans

    def forward(self, x):
        out = []
        for i in range(self.num_unet):
            unet = self.__getattr__(f"unet_{i + 1}")
            x = unet(x)
            pred = x[0]
            out.append(pred)
        if self.out_all_features:
            if self.out_dict:
                return dict(pred=pred, stage_preds=out)
            return out
        else:
            if self.out_dict:
                return dict(pred=pred)
            return pred

    def traverse(self, stat: StatisticModel, x):
        stat.statistic_self_alone()
        for i in range(self.num_unet):
            unet = self.__getattr__(f"unet_{i + 1}")
            x = stat.step(unet, (x,), name=f"unet_{i + 1}")

# drop_path_rate=0., drop_block_rate=0.,
def ResUNet_MiniMini(in_chans=32, block=BasicBlock, **args):
    return ResUNet(
        block, [2, 2, 3, 3], [32, 60, 80, 100], [2, 2, 3, 1], [32, 60, 80, 100], in_chans, **args
    )
def ResUNet_Mini(in_chans=48, block=BasicBlock, **args):
    return ResUNet(
        block, [2, 4, 6, 6], [48, 80, 112, 168], [2, 3, 4, 2], [48, 80, 112, 168], in_chans, **args
    )
def ResUNet_Tiny(in_chans=64, block=BasicBlock, **args):
    return ResUNet(
        block, [3, 5, 10, 10], [64, 112, 168, 256], [2, 4, 6, 4], [64, 112, 168, 256], in_chans, **args
    )

def ResUNet_Tiny_SE(in_chans=64, block=BasicBlock, **args):
    return ResUNet(
        block, [2, 4, 8, 8], [64, 96, 144, 244], [2, 3, 6, 2], [64, 96, 144, 244], in_chans, attn_layer="se", **args
    )

def ResUNet_Small(in_chans=64, block=BasicBlock, **args):
    return ResUNet(
        block, [5, 7, 16, 12], [64, 128, 192, 310], [4, 6, 12, 8], [64, 128, 192, 310], in_chans, **args
    )
def ResUNet_Normal(in_chans=92, block=BasicBlock, **args):
    return ResUNet(
        block, [5, 8, 16, 12], [92, 144, 248, 360], [4, 6, 12, 8], [92, 144, 248, 360], in_chans, **args
    )




if __name__ == "__main__":
    
    # net = ResUNet_Normal(64)

    # StatisticScale.stat(
    #     net, [2, 64, 128, 128]
    # )
    # input = torch.randn([2, 64, 128, 128])
    # ret = net(input)
    # for f in [ret]:
    #     print(f.shape)

    net = MUNet([ResUNet_Mini() for i in range(5)], out_final_feature=True)
    StatisticScale.stat(net, [2, 48, 128, 128])
    input = torch.randn([2, 48, 128, 128])
    ret = net(input)
    for f in [ret]:
        print(f.shape)


# class UNet(nn.Module):

#     """ 模版 """
#     empty_block_args = dict(
#         block_args=dict(cls=None),
#         encode=dict(
#             channels=[64, 128, 256, 512],
#             blocks=[3, 5, 10, 5],
#             block_args=dict(cls=None),
#             stage_block_args=[
#                 dict(cls=None), None, None, None
#             ],
#         ),
#         decode=dict(
#             channels=[64, 128, 256, 512],
#             blocks=[2, 4, 8, 4],
#             block_args=dict(cls=None),
#         )
#     )

#     def __init__(self, block_args: dict, index_prompt=True):
#         """
#             <no stem_stage>
#             保证init单线程(不被其它Unet的init打断)
#             index_prompt: args contain: stage_idx、 block_idx、 direct(encode/decode)
#         """
#         super(UNet, self).__init__()

#         global_args = block_args.get("block_args", {})
#         encode_dict = block_args["encode"]
#         decode_dict = block_args["decode"]
        
#         self.encode_channels = encode_channels = encode_dict["channels"]
#         self.encode_blocks = encode_blocks = encode_dict["blocks"]
#         encode_args = global_args.copy()
#         encode_args.update(encode_dict.get("block_args", {}))
#         stage_block_args = encode_dict.get("stage_block_args", [None] * 4)
#         for stage_idx, (channels, blocks, st_args) in enumerate(zip(
#             encode_channels, encode_blocks, stage_block_args)):
#             blk_args = encode_args.copy()
#             if st_args is not None:
#                 blk_args.update(st_args)
#             stage_name = f'encode_layer{stage_idx + 1}'
