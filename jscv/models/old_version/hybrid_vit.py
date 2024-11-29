import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttention(nn.Module):

    def __init__(self,
                 dim,
                 out_dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hybrid_levels=hybrid_levels_1,
                 subsample_style=None, #TODO maxpool+conv
                 ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.hybrid_levels = hybrid_levels
        self.hybrid_layers = len(hybrid_levels)

        for i, (wsz, sr_ratio) in enumerate(self.hybrid_levels):
            #if sr_ratio > 1:
            #q = nn.Linear(dim, dim, bias=qkv_bias)
            conv = nn.Conv2d(dim,
                             dim * 2,
                             kernel_size=sr_ratio,
                             stride=sr_ratio)
            norm = nn.LayerNorm(dim * 2)
            attn_drop_l = nn.Dropout(attn_drop)
            #self.__setattr__(f'q_{i}', q)
            self.__setattr__(f'conv_{i}', conv)
            self.__setattr__(f'norm_{i}', norm)
            self.__setattr__(f'attn_drop_{i}', attn_drop_l)

        self.proj = nn.Linear(dim * self.hybrid_layers, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.apply(self._init_weights)


    def forward(self, x):
        B, H, W, C = x.shape
        nH = self.num_heads
        q = self.q(x)

        y = []

        for i, (wsz, sr_ratio) in enumerate(self.hybrid_levels):
            conv = self.__getattr__(f'conv_{i}')
            norm = self.__getattr__(f'norm_{i}')
            attn_drop = self.__getattr__(f'attn_drop_{i}')
            
            #TODO wsz必须能整除H, W
            wH, wW = H // wsz, W // wsz
            
            # --> [B, nH, wH, wW, wsz*wsz, C//nH]
            q = q.reshape(B, wH, wsz, wW, wsz, nH, C // nH
                           ).permute(0, 5, 1, 3, 2, 4, 6).reshape(
                               B, nH, wH, wW, wsz * wsz, C // nH)

            # [B*wH*wW, C, wsz, wsz]
            wX = x.reshape(B, wH, wsz, wW, wsz, C).permute(
                0, 1, 3, 5, 2, 4).reshape(B * wH * wW, C, wsz, wsz)

            wX = conv(wX).permute(0, 2, 3, 1)   # B*wH*wW, sh, sw, 2C
            _, sh, sw, _ = wX.shape
            wX = norm(wX)
            kv = wX.reshape(B, wH, wW, sh, sw, 2, nH, C // nH
                            ).permute(5, 0, 6, 1, 2, 3, 4, 7
                                      ).reshape(2, B, nH, wH, wW, sh * sw, C // nH)
            k, v = kv[0], kv[1]
            # print(i, q.shape, k.shape)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            x = (attn @ v) # B, nH, wH, wW, wsz * wsz, C // nH
            # print(x.shape)
            x = x.reshape(B, nH, wH, wW, wsz, wsz, C // nH
                          ).permute(
                              0, 2, 4, 3, 5, 1, 6
                              ).reshape(
                                  B, H, W, C
                              )
            y.append(x)

        L = len(y)
        y = torch.concat(y, dim=-1)

        y = self.proj(y)
        y = self.proj_drop(y)

        return y