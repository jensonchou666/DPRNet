import math
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import Mlp, DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Attention
 


Size_ = Tuple[int, int]


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinLocallyGroupedAttn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1, shift_size=0):
        assert ws != 1
        super(SwinLocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.shift_size = shift_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, mask_matrix=None):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size

        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.ws)  # nW*B, window_size, window_size, C
        #x_windows = x_windows.view(-1, self.ws * self.ws, C)  # nW*B, window_size*window_size, C

        _h, _w = Hp // self.ws, Wp // self.ws

        wsws = self.ws * self.ws
        qkv = self.qkv(x_windows).reshape(-1, wsws, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B*_h*_w,  num_heads, wsws, C//self.num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # B*_h*_w,  num_heads, wsws, wsws
        attn_shape = attn.shape
        # Swin
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(-1, self.num_heads, wsws, wsws)
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads, wsws, wsws) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(*attn_shape)

            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        
        attn = self.attn_drop(attn)
        # attn = (attn @ v).transpose(1, 2).reshape(B, _h, _w, self.ws, self.ws, C)
        attn = (attn @ v).transpose(1, 2).reshape(-1, wsws, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        attn = attn.view(-1, self.ws, self.ws, C)
        shifted_x = window_reverse(attn, self.ws, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)

        return x



# reason: FX can't symbolically trace control flow in forward method
class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(
            B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # def forward_mask(self, x, size: Size_):
    #     B, N, C = x.shape
    #     H, W = size
    #     x = x.view(B, H, W, C)
    #     pad_l = pad_t = 0
    #     pad_r = (self.ws - W % self.ws) % self.ws
    #     pad_b = (self.ws - H % self.ws) % self.ws
    #     x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    #     _, Hp, Wp, _ = x.shape
    #     _h, _w = Hp // self.ws, Wp // self.ws
    #     mask = torch.zeros((1, Hp, Wp), device=x.device)
    #     mask[:, -pad_b:, :].fill_(1)
    #     mask[:, :, -pad_r:].fill_(1)
    #
    #     x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)  # B, _h, _w, ws, ws, C
    #     mask = mask.reshape(1, _h, self.ws, _w, self.ws).transpose(2, 3).reshape(1,  _h * _w, self.ws * self.ws)
    #     attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)  # 1, _h*_w, ws*ws, ws*ws
    #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
    #     qkv = self.qkv(x).reshape(
    #         B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
    #     # n_h, B, _w*_h, nhead, ws*ws, dim
    #     q, k, v = qkv[0], qkv[1], qkv[2]  # B, _h*_w, n_head, ws*ws, dim_head
    #     attn = (q @ k.transpose(-2, -1)) * self.scale  # B, _h*_w, n_head, ws*ws, ws*ws
    #     attn = attn + attn_mask.unsqueeze(2)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)  # attn @v ->  B, _h*_w, n_head, ws*ws, dim_head
    #     attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
    #     x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
    #     if pad_r > 0 or pad_b > 0:
    #         x = x[:, :H, :W, :].contiguous()
    #     x = x.reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x


class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print("size ", size)
        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None, use_swin=False, shift_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_swin = False
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        elif use_swin:
            self.attn = SwinLocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws, shift_size=shift_size)
            self.use_swin = True
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Size_, **kwargs):
        if self.use_swin:
            x = x + self.drop_path(self.attn(self.norm1(x), size, kwargs['mask_matrix']))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim), )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        out_size = (H // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class Twins(nn.Module):
    """ Twins Vision Transfomer (Revisiting Spatial Attention)

    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """
    def __init__(
            self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1), wss=None, use_swin=False, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), block_cls=Block, features_only=False):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.depths = depths
        self.embed_dims = embed_dims
        if features_only:
            self.num_features = embed_dims
        else:
            self.num_features = embed_dims[-1]
        self.grad_checkpointing = False

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # 全shift_sizes
        if wss is None:
            shift_sizes = [0] * len(depths)
        else:
            shift_sizes = [window_size // 2 for window_size in wss]

        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate, use_swin=use_swin,
                attn_drop=attn_drop_rate, shift_size=shift_sizes[k], drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.wss = wss
        self.shift_sizes = shift_sizes

        self.features_only = features_only
        if not features_only:
            self.norm = norm_layer(self.num_features)
            # classification head
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embeds.0',  # stem and embed
            blocks=[
                (r'^(?:blocks|patch_embeds|pos_block)\.(\d+)', None),
                ('^norm', (99999,))
            ] if coarse else [
                (r'^blocks\.(\d+)\.(\d+)', None),
                (r'^(?:patch_embeds|pos_block)\.(\d+)', (0,)),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def make_mask_matrix(self, x, H, W, k):
        # calculate attention mask for SW-MSA
        ws = self.wss[k]
        sz = self.shift_sizes[k]

        Hp = int(np.ceil(H / ws)) * ws
        Wp = int(np.ceil(W / ws)) * ws
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -ws),
                    slice(-ws, -sz), slice(-sz, None))
        w_slices = (slice(0, -ws),
                    slice(-ws, -sz), slice(-sz, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                                              attn_mask == 0, float(0.0))
        return attn_mask


    def forward_features(self, x):
        B = x.shape[0]
        ret = []
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            H, W = size
            x = drop(x)

            if self.wss is not None:
                # attn_mask
                mask_matrix = self.make_mask_matrix(x, H, W, i)
            else:
                mask_matrix = None

            for j, blk in enumerate(blocks):
                x = blk(x, size, mask_matrix=mask_matrix)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            # if i < len(self.depths) - 1:
            x = x.reshape(B, *size, -1).permute(0, 3, 1, 2)
            ret.append(x.contiguous())
        return ret

    def forward_head(self, x, pre_logits: bool = False):
        x = self.norm(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        if self.features_only:
            return self.forward_features(x)
        else:
            x = self.forward_features(x)
            x = self.forward_head(x[-1])
        return x



def _create_twins(variant, pretrained=False, **kwargs):
    # if kwargs.get('features_only', None):
    #     raise RuntimeError('features_only not implemented for Vision Transformer models.')
    pass

  

   

def twins_pcpvt_small(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_pcpvt_small', pretrained=pretrained, **model_kwargs)



def twins_pcpvt_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_pcpvt_base', pretrained=pretrained, **model_kwargs)



def twins_pcpvt_large(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_pcpvt_large', pretrained=pretrained, **model_kwargs)



def twins_svt_small(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_svt_small', pretrained=pretrained, **model_kwargs)



def twins_svt_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_svt_base', pretrained=pretrained, **model_kwargs)



def twins_svt_large(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_svt_large', pretrained=pretrained, **model_kwargs)




'''
#TODO Shift window -> cross swin

def window_partition(x, ws_h, ws_w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // ws_h, ws_h, W // ws_w, ws_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws_h, ws_w, C)
    return windows

def window_reverse(windows, ws_h, ws_w, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / ws_h / ws_w))
    x = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



def make_mask_matrix(H, W, ws_h, ws_w, shift_size, device):
    # calculate attention mask for SW-MSA
    sz = shift_size

    Hp = int(np.ceil(H / ws_h)) * ws_h
    Wp = int(np.ceil(W / ws_w)) * ws_w
    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
    h_slices = (slice(0, -ws_h),
                slice(-ws_h, -sz), slice(-sz, None))
    w_slices = (slice(0, -ws_w),
                slice(-ws_w, -sz), slice(-sz, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, ws_h, ws_w)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, ws_h * ws_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                                            attn_mask == 0, float(0.0))
    return attn_mask



class ShiftWindowTool:
    def __init__(self, num_heads, shift_size, mask_matrix):
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.mask_matrix = mask_matrix


    def shift(self, x):
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            self.attn_mask = self.mask_matrix
        else:
            shifted_x = x
            self.attn_mask = None

        # partition windows
        return window_partition(shifted_x, self.ws)

    def swin(self, attn):
        # Swin
        attn_shape = attn.shape
        attn_mask = self.attn_mask
        toksq, tokskv = attn[-2], attn[-1]
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(-1, self.num_heads, toksq, tokskv)
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads, toksq, tokskv) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(*attn_shape)

            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        return attn

    def resume(self, x):
        pass


class ShiftedWindowSelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1, shift_size=0):
        assert ws != 1
        super(ShiftedWindowSelfAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.shift_size = shift_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Tuple[int, int], mask_matrix=None):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        swt = ShiftWindowTool(self.num_heads, self.shift_size, mask_matrix)
        B, N, C = x.shape
        H, W = size

        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = swt.shift(x)

        _h, _w = Hp // self.ws, Wp // self.ws

        tokens = self.ws * self.ws
        qkv = self.qkv(x_windows).reshape(-1, tokens, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B*_h*_w,  num_heads, wsws, C//self.num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = swt.swin(attn)
        
        attn = self.attn_drop(attn)
        # attn = (attn @ v).transpose(1, 2).reshape(B, _h, _w, self.ws, self.ws, C)
        attn = (attn @ v).transpose(1, 2).reshape(-1, tokens, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        attn = attn.view(-1, self.ws, self.ws, C)
        shifted_x = window_reverse(attn, self.ws, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)

        return x



'''