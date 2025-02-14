import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
import math

from jscv.utils.utils import TimeCounter

DO_Analyse = False
DO_DEBUG = False
DO_DEBUG_L = False
DO_DEBUG_C = False
counter = TimeCounter(DO_DEBUG)
count_c = TimeCounter(DO_DEBUG_C)
count_layer = TimeCounter(DO_DEBUG_L)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



analyse_dict = {}
import numpy as np
from jscv.utils.analyser import AnalyseItem, add_analyse_item

class AnalyseAttn(AnalyseItem):
    name_dict = dict()

    def call(self, datas: dict, dist_datas: dict):
        global analyse_dict
        self.blkids = []
        if DO_Analyse:
            for bid, v in analyse_dict["attn_map"].items():
                self.blkids.append(bid)
                dist_datas[bid] = v # B, 3, nH, H, W


    def show(self, datas, disted_datas: dict):
        global analyse_dict
        if not DO_Analyse:
            return

        do_compose = False
        if "org_img" in disted_datas:
            org_img = disted_datas["org_img"]
            oH, oW, _ = org_img.shape
            do_compose = True

        for bid in self.blkids:
            v = disted_datas[bid]
            nH, nW, H, W = v.shape
            
            map = []
            for vi in v:
                vjL = []
                for vj in vi:
                    x = vj.cpu().numpy()    #H,W
                    x = self.to_jet(x)
                    vjL.append(x)
                map.append(np.stack(vjL, 0))
            map = np.stack(map, 0)
            map = map.transpose(0, 2, 1, 3, 4).reshape(nH * H, nW * W, 3)

            # if nH*nW <= 20 and do_compose:
            #     org_img1 = torch.from_numpy(org_img).view(1, 1, oH, oW, 3).expand(
            #         nH, nW, oH, oW, 3
            #     ).transpose(1, 2).reshape(nH*oH, nW*oW, 3)
            #     map1 = torch.from_numpy(map).permute(2, 0, 1).unsqueeze(0)
            #     map1 = F.interpolate(map1, [nH*oH, nW*oW], mode="nearest")[0].permute(1, 2, 0)
            #     map1 = org_img1 * 0.6 + map1 * 0.4
            #     map1 = map1.numpy()
            #     for i in range(1, nH):
            #         map1[i*oH-1] = 255
            #     for j in range(1, nW):
            #         map1[:, j*oW-1] = 255
            #     self.save_next_image(f"attn_map-b_{bid}.png", map1)


            for i in range(1, nH):
                map[i*H-1] = 255
            for j in range(1, nW):
                map[:, j*W-1] = 255
            self.save_next_image(f"attn_map_alone-b_{bid}.png", map)



class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim,
                                    dim,
                                    kernel_size=sr_ratio,
                                    stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        
        
        self.apply(self._init_weights)


    def forward(self, x, H, W):
        B, N, C = x.shape
        sr_ratio = self.sr_ratio

        #counter.record_time(first=True)

        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        #counter.record_time("q(x)")


        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                         C // self.num_heads).permute(
                                             2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                                            2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4)

        #counter.record_time("sr + kv(x)")

        k, v = kv[0], kv[1]
        # print(q.shape, k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if DO_DEBUG:
            print(f"q: {q.shape}, kv: {k.shape}")
        x = self.proj(x)
        x = self.proj_drop(x)

        if DO_Analyse:
            Gp = 16
            # 1
            pvt_attn = attn[:, 0] #B, hw, hw//64
            pvt_attn = pvt_attn.reshape(B, Gp, H//Gp, Gp, W//Gp, H//sr_ratio, W//sr_ratio)[:, :, 0, :, 0]

            # 2
            # pvt_attn = attn.mean(2)
            # pvt_attn = pvt_attn.reshape(B, self.num_heads,  H//sr_ratio, W//sr_ratio)
            # pvt_attn = F.interpolate(pvt_attn, (512, 512)).unsqueeze(1)

            
            global analyse_dict
            i, j = analyse_dict["stage"], analyse_dict["block"]
            blockid = f"{i}_{j}"
            assert blockid not in analyse_dict["attn_map"]
            analyse_dict["attn_map"][blockid] = pvt_attn


        #counter.record_time("q@k@v", last=True)
        if DO_DEBUG:
            print(counter)

        return x

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




class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio,
                              linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       linear=linear)

        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        #count_c .begin()
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        #count_c.record_time("attn")
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        #count_c.record_time("conv", last=True)

        # print(count_c)
        if DO_Analyse:
            global analyse_dict
            analyse_dict["block"] += 1
        return x

    def forward2(self, x, H, W):
        c = TimeCounter(True)
        c.record_time(first=True)
        x = self.norm1(x)
        c.record_time("norm1")
        x = x + self.drop_path(self.attn(x, H, W))
        c.record_time("attn")
        x = self.norm2(x)
        c.record_time("norm2")
        x = x + self.drop_path(self.mlp(x, H, W))
        c.record_time("mlp", last=True)
        print(c)
        
        if DO_Analyse:
            global analyse_dict
            analyse_dict["block"] += 1
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                #  num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 block_cls=Block,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 return_bchw=True, # True 特征 BCHW; False: 特征 BHWC 
                 linear=False):
        super().__init__()

        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.embed_dims = embed_dims
        self.return_bchw = return_bchw

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2**(i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])

            block = nn.ModuleList([
                block_cls(dim=embed_dims[i],
                          num_heads=num_heads[i],
                          mlp_ratio=mlp_ratios[i],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[cur + j],
                          norm_layer=norm_layer,
                          sr_ratio=sr_ratios[i],
                          linear=linear) for j in range(depths[i])
            ])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        if DO_Analyse:
            al = AnalyseAttn()
            al.model = self
            add_analyse_item(al)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)


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

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         # print("load pretrained backbone:", pretrained)
    #         #logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # print('1',x.shape)
        B = x.shape[0]
        outs = []
        #count_layer.begin()
        if "attn_map" in analyse_dict:
            analyse_dict["attn_map"].clear()
        analyse_dict["attn_map"] = {}

        for i in range(self.num_stages):
            # print("################", i, x.shape)
            if DO_Analyse:
                analyse_dict["stage"] = i
                analyse_dict["block"] = -1

            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1)
            if not self.return_bchw:
                xout = x
            x = x.permute(0, 3, 1, 2).contiguous()
            if self.return_bchw:
                xout = x
            outs.append(xout)

        return outs



    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

def pvtv2_small():
    # [2, 3, 512, 512]      52.2653271484375 ms
    # [2, 3, 1024, 1024]    329.58203125 ms
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 256, 384],
                                      num_heads=[2, 4, 8, 8],
                                      mlp_ratios=[4, 4, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 4, 6, 4],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)


def pvtv2_b0():
    # [2, 3, 512, 512]      21.71091552734375 ms
    # [2, 3, 1024, 1024]    120.43447265625 ms
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[32, 64, 160, 256],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[2, 2, 2, 2],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def pvtv2_b1():
    # [2, 3, 512, 512]      33.1154248046875 ms
    # [2, 3, 1024, 1024]    160.485771484375 ms
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[2, 2, 2, 2],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def pvtv2_b2():
    # [2, 3, 512, 512]      57.2468359375 ms
    # [2, 3, 1024, 1024]    281.36466796875 ms
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 4, 6, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def pvtv2_b3():
    # [2, 3, 512, 512]      81.414033203125 ms
    # [2, 3, 1024, 1024]    399.9346484375 ms
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 4, 18, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def pvtv2_small_2(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    # B=2
    # [2, 3, 512, 512]        51.367890625 ms
    # [2, 3, 1024, 1024]      326.51240234375 ms
    
    # B=1
    # [1, 3, 512, 512]        31.42905517578125 ms
    # [1, 3, 1024, 1024]      194.2116015625 ms

    backbone_features = [64, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[3, 4, 6, 4],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone


def pvtv2_small_2_for_refine(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [64, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[2, 4, 7, 4],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone

if __name__ == '__main__':
    from jscv.utils.utils import test_model_latency
    from jscv.utils.utils import TimeCounter, warmup
    # backbone = pvtv2_small_2()
    # test_model_latency(backbone, 2)
    torch.cuda.set_device(2)
    
    net = pvtv2_small_2_for_refine().cuda().eval()
    warmup(20)
    
    DO_DEBUG_L = True
    count_layer.DO_DEBUG = True


    x = torch.randn([2, 3, 512, 512]).cuda()
    with torch.no_grad():
        result = net(x)
    print(count_layer.str_total_porp())


    # model = PyramidVisionTransformerV2(patch_size=4,
    #                                  embed_dims=[64, 128, 320, 512],
    #                                 #  num_heads=[1, 2, 5, 8],
    #                                  num_heads=[1, 1, 1, 1],
    #                                  mlp_ratios=[8, 8, 4, 4],
    #                                  qkv_bias=True,
    #                                  norm_layer=partial(nn.LayerNorm,
    #                                                     eps=1e-6),
    #                                  depths=[3, 4, 18, 3],
    #                                  sr_ratios=[8, 4, 2, 1],
    #                                  drop_rate=0.0,
    #                                  drop_path_rate=0.1).cuda()

    # x1 = torch.randn(2, 3, 512, 512).cuda()
    # x2 = torch.randn(2, 3, 1024, 1024).cuda()

    # print('warm up ...\n')
    # with torch.no_grad():
    #     for _ in range(10):
    #         model(x1)
    # torch.cuda.synchronize()

    # DO_DEBUG = False
    # counter.DO_DEBUG = False
    

    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # starter.record()

    # with torch.no_grad():
    #     for i in range(1):
    #         model(x1)

    # ender.record()
    # torch.cuda.synchronize()
    # if counter.DO_DEBUG:
    #     print(counter.str_total())
    # print("spend", starter.elapsed_time(ender))