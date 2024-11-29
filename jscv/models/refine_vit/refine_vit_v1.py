import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from jscv.utils.statistics import StatisticModel
from jscv.utils.utils import TimeCounter, warmup, do_once

'''
    无滑动窗口, 固定filter_rate, 全局分支直接使用筛选窗口
    多尺度窗口, test是否需要多头
'''

DO_DEBUG = False
DO_DEBUG_DETAIL = False

count_blocks = TimeCounter(DO_DEBUG)
count_refine = TimeCounter(DO_DEBUG)



default_cfg = dict(
    num_heads=2,
    window_size=8,
    sr_ratio=8,
    filter_rate=0.3,
    qkv_bias=False,
    attn_drop=0,
    proj_drop=0.)

def _cfg(cfg_dict: dict={}):
    cfgs = default_cfg.copy()
    for k, v in cfg_dict.items():
        cfgs[k] = v
    return cfgs


def create_batch_base_idx(B, N, device="cuda"):
    return torch.arange(0, B*N, N, dtype=torch.int, device=device).unsqueeze(-1)

def drop_layer(drop_rate=0.):
    return nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

def init_weights(m):
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


class RefineVitBlock(nn.Module):

    def __init__(self,
                 channels,
                 num_heads=2,
                 window_size=8,
                 sr_ratio=8,
                 filter_rate=0.3,
                 qkv_bias=False,
                 attn_drop=0,
                 proj_drop=0.,
                 ):
        super().__init__()
        C = self.channels = channels
        self.window_size = window_size
        self.sr_ratio = sr_ratio
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.filter_rate = filter_rate
        self.scale = (C//num_heads)**-0.5
        assert sr_ratio > 1

        self.sr_conv = nn.Conv2d(C, C, sr_ratio, sr_ratio, groups=C)
        self.sr_norm = nn.LayerNorm(C)
        self.act = nn.GELU()

        # self.norm1 = nn.LayerNorm(C)   #TODO 费时
        self.attn_drop = drop_layer(attn_drop)

        self.q_l = nn.Linear(C, C, bias=qkv_bias)
        self.kv_l = nn.Linear(C, C * 2, bias=qkv_bias)
        self.q_g = nn.Linear(C, C, bias=qkv_bias)
        self.kv_g = nn.Linear(C, C * 2, bias=qkv_bias)
        self.proj = nn.Linear(C, C)
        self.proj_drop = drop_layer(proj_drop)

        self.apply(init_weights)

    def forward(self, feature_map: torch.Tensor, uncertain_map: torch.Tensor):
        """
            feature_map:      B, C, H, W
            uncertain_map:    B, H, W
        """
        wsz = self.window_size
        sr_ratio = self.sr_ratio
        filter_rate = self.filter_rate
        num_heads = self.num_heads
        
        x = feature_map
        B, C, H, W = x.shape
        Ch = C//num_heads
        H1, W1 = H//sr_ratio, W//sr_ratio
        hw, hw1 = H*W, H1*W1
        nH, nW = H//wsz, W//wsz
        nWin = nH*nW
        winsz = wsz*wsz

        
        count_refine.record_time(first=True)
        
        ''' local branch'''
        win_x = x.permute(0, 2, 3, 1).reshape(B, nH, wsz, nW, wsz, C).transpose(2, 3).reshape(
            B*nWin, winsz, C)

        win_unc_map = uncertain_map.reshape(B, nH, wsz, nW, wsz).transpose(
            2, 3).reshape(B, nWin, winsz)

        count_refine.record_time("pre")
        # B, nWin
        win_score = win_unc_map.mean(-1)    # mean? max?
        nWF = nW_filter = int(nWin * filter_rate)
        # B, nWF
        win_score_sorted, win_score_idx = torch.topk(win_score, k=nWF, dim=1, sorted=False)

        win_score_idx += create_batch_base_idx(B, nWin, win_score_idx.device)
        # count_refine.record_time("topk")

        win_score_idx = win_score_idx.reshape(B*nWF)
        win_x_filter = torch.index_select(win_x, dim=0, index=win_score_idx,)
        count_refine.record_time("index_select")
        
        # win_x_filter = self.norm1(win_x_filter) #B*nWF, winsz, C
        
        # .reshape(B, nWF, winsz, C)

        q = self.q_l(win_x_filter).reshape(B*nWF, winsz, num_heads, Ch
                                           ).transpose(1, 2)
        kv = self.kv_l(win_x_filter).reshape(B*nWF, winsz, 2, num_heads, Ch
                                             ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if DO_DEBUG and do_once(None, "_local_"):
            print("local q", q.shape, ", k", k.shape)

        count_refine.record_time("local q_l-kv_l")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        win_x_filter = win_x_filter + (attn @ v).transpose(1, 2).reshape(B*nWF, winsz, C)
        
        count_refine.record_time("local qkv")

        ''' global branch'''
        global_x = self.sr_conv(x).reshape(B, C, hw1).transpose(1, 2)
        global_x = self.act(self.sr_norm(global_x))
        
        count_refine.record_time("global sr_conv")

        q = self.q_g(win_x_filter).reshape(B, nWF*winsz, num_heads, Ch
                                           ).transpose(1, 2)
        kv = self.kv_g(global_x).reshape(B, hw1, 2, num_heads, Ch
                                             ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        win_x_filter = win_x_filter + (attn @ v).transpose(1, 2).reshape(B*nWF, winsz, C)
        count_refine.record_time("global qkv")

        if DO_DEBUG and do_once(None, "_g_"):
            print("global q", q.shape, ", k", k.shape)

        win_x_filter = self.proj(win_x_filter)
        win_x_filter = self.proj_drop(win_x_filter)
        
        count_refine.record_time("proj")

        ''' compose '''
        win_x.index_add_(0, win_score_idx, win_x_filter)
        #win_x.index_copy_(0, win_score_idx, win_x_filter)  #TODO

        count_refine.record_time("index_add", last=True)
        
        x = win_x.reshape(B, nH, nW, wsz, wsz, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        
        return x


class NormalVitBlock(nn.Module):

    def __init__(self,
                 channels,
                 num_heads=2,
                 window_size=8,
                 sr_ratio=8,
                 filter_rate=0.3,
                 qkv_bias=False,
                 attn_drop=0,
                 proj_drop=0.,
                 ):
        super().__init__()
        C = self.channels = channels
        self.window_size = window_size
        self.sr_ratio = sr_ratio
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.filter_rate = filter_rate
        self.scale = (C//num_heads)**-0.5
        assert sr_ratio > 1

        self.sr_conv = nn.Conv2d(C, C, sr_ratio, sr_ratio, groups=C)
        self.sr_norm = nn.LayerNorm(C)
        self.act = nn.GELU()

        #self.norm1 = nn.LayerNorm(C)
        self.attn_drop = drop_layer(attn_drop)

        self.q_l = nn.Linear(C, C, bias=qkv_bias)
        self.kv_l = nn.Linear(C, C * 2, bias=qkv_bias)
        self.q_g = nn.Linear(C, C, bias=qkv_bias)
        self.kv_g = nn.Linear(C, C * 2, bias=qkv_bias)
        self.proj = nn.Linear(C, C)
        self.proj_drop = drop_layer(proj_drop)

        self.apply(init_weights)

    def forward(self, feature_map: torch.Tensor, uncertain_map: torch.Tensor):
        """
            feature_map:      B, C, H, W
            uncertain_map:    B, H, W
        """
        wsz = self.window_size
        sr_ratio = self.sr_ratio
        filter_rate = self.filter_rate
        num_heads = self.num_heads
        
        x = feature_map
        B, C, H, W = x.shape
        Ch = C//num_heads
        H1, W1 = H//sr_ratio, W//sr_ratio
        hw, hw1 = H*W, H1*W1
        nH, nW = H//wsz, W//wsz
        nWin = nH*nW
        winsz = wsz*wsz

        x1 = x.permute(0, 2, 3, 1)
        
        count_refine.record_time(first=True)
        
        ''' local branch'''
        win_x = x1.reshape(B, nH, wsz, nW, wsz, C).transpose(2, 3).reshape(
            B*nWin, winsz, C)

        count_refine.record_time("pre")

        q = self.q_l(win_x).reshape(B*nWin, winsz, num_heads, Ch
                                           ).transpose(1, 2)
        kv = self.kv_l(win_x).reshape(B*nWin, winsz, 2, num_heads, Ch
                                             ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        count_refine.record_time("local q_l-kv_l")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        win_x = win_x + (attn @ v).transpose(1, 2).reshape(B*nWin, winsz, C)

        if DO_DEBUG and do_once(None, "_local_"):
            print("local q", q.shape, ", k", k.shape)

        count_refine.record_time("local qkv")

        ''' global branch'''
        global_x = self.sr_conv(x).reshape(B, C, hw1).transpose(1, 2)
        global_x = self.act(self.sr_norm(global_x))
        
        count_refine.record_time("global sr_conv")
        
        x = win_x.reshape(B, nH, nW, wsz, wsz, C).transpose(2, 3).reshape(B, H*W, C)

        q = self.q_g(win_x).reshape(B, H*W, num_heads, Ch
                                    ).transpose(1, 2)
        kv = self.kv_g(global_x).reshape(B, hw1, 2, num_heads, Ch
                                         ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = x + (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        count_refine.record_time("global qkv")

        if DO_DEBUG and do_once(None, "_g_"):
            print("global q", q.shape, ", k", k.shape)

        x = self.proj(x)
        x = self.proj_drop(x)
        
        count_refine.record_time("proj")

        ''' compose '''
        
        count_refine.record_time("index_add", last=True)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x



def get_padding(kernel_size, stride):
    """
        kernel_size 奇数
    """
    return (kernel_size - stride + 1) // 2

class ConvBNReLU(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      bias=bias,
                      stride=stride,
                      padding=get_padding(kernel_size, stride)),
            norm_layer(out_channels), act_layer())

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      padding=get_padding(kernel_size, stride),
                      groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels), act_layer())



class RefineVitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_blocks,
                 block_args: dict,
                 uncertain_type="top1", # top1 or top2
                 embed_conv=True,
                 conv_cls=ConvBNReLU,
                 use_conv_in=True,
                 refine_block_cls=RefineVitBlock,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.block_args = block_args
        self.embed_conv = embed_conv
        self.uncertain_type = uncertain_type
        self.use_conv_in = use_conv_in
        if use_conv_in:
            self.conv_in = conv_cls(in_channels, channels)
        else:
            assert in_channels == channels
        for i in range(num_blocks):
            if i != 0 and embed_conv:
                self.__setattr__(f"conv_{i}", conv_cls(channels, channels))
            self.__setattr__(f"refine_{i}", refine_block_cls(channels, **block_args))

        self.apply(init_weights)

    def create_uncertain_map(self, feature_map: torch.Tensor, coarse_pred: torch.Tensor):
        B, C, H, W = feature_map.shape
        B, nC, Hc, Wc = coarse_pred.shape

        ''' 调整到一致大小 '''
        if H != Hc or W != Wc:
            coarse_pred = F.interpolate(coarse_pred, (H, W), mode="bilinear")

        ''' uncertain_map '''
        if not self.after_softmax:
            coarse_pred = torch.softmax(coarse_pred, dim=1)
        if self.uncertain_type == "top1":
            #coarse_pred.argmax(dim=dim)
            uncertain_map = 1 - torch.max(coarse_pred, dim=1)[0]  # B, H, W
        elif self.uncertain_type == "top2":
            top2_scores = torch.topk(coarse_pred, k=2, dim=1)[0]
            uncertain_map = top2_scores[:, 0] - top2_scores[:, 1]
            #TODO
            uncertain_map = torch.max(uncertain_map) - uncertain_map
        else:
            assert "wrong uncertain_type"
        
        return uncertain_map # B,H,W

    def forward(self, feature_map: torch.Tensor, uncertain_map: torch.Tensor):
        """
            feature_map:      B, C, H, W
            uncertain_map:    B, H, W
        """
        count_blocks.record_time(first=True)
        if self.use_conv_in:
            x = self.conv_in(feature_map)
        else:
            x = feature_map

        count_blocks.record_time("conv_in")

        for i in range(self.num_blocks):
            if i != 0 and self.embed_conv:
                conv = self.__getattr__(f"conv_{i}")
                x = x + conv(x)
                count_blocks.record_time(f"conv_{i}")
                
            refine_block = self.__getattr__(f"refine_{i}")
            x = refine_block(x, uncertain_map)
            
            count_blocks.record_time(f"refine_{i}")

        count_blocks.record_time(last=True)
        if DO_DEBUG_DETAIL:
            print(count_blocks)
        
        return x

if __name__ == "__main__":
    debug_block = 1

    torch.cuda.set_device(3)
    warmup(20)
    
    if debug_block == 0:
        #(160)
        # 1.0       4254.31494140625
        # 0.3       1730.33984375       2.4586<3.33             下降 1/2.2211(比normal)
        #(128)
        # 1.0       2161.6513671875
        # 0.6       1430.2847900390625  1/1.511
        # 0.5       1254.1595458984375  1/1.723                 下降 1/1.58712(比normal) 0.79356
        # 0.3       892.0120849609375   1/2.423 <3.33           下降 1/2.2314(比normal)  0.6694
        # 0.25      807.8231811523438   1/2.675                 下降 1/2.4640(比normal)  0.616
        # 0.15      617.8622436523438   1/3.498
        #(64)
        # 1.0       499.5614318847656
        # 0.3       398.6073303222656

        # NormalVitBlock    3843.426513671875(160(x1.5625))     1.9308倍
        # NormalVitBlock    1990.505615234375(128)              4.2532倍
        # NormalVitBlock    467.99481201171875(64)
        cfgs = dict(
            num_heads=1,
            window_size=8,
            sr_ratio=8,
            filter_rate=0.5)
        H = 256
        C = 64
        m = RefineVitLayer(C, C, 4, _cfg(cfgs), refine_block_cls=NormalVitBlock ,
                        use_conv_in=False, embed_conv=False).cuda().eval()
        x = torch.randn(2, C, H, H).cuda()
        mask = torch.randn(2, H, H).cuda()
        ct = TimeCounter(True)
        ct.record_time(first=True)

        with torch.no_grad():
            for i in range(100):
                m(x, mask)
        ct.record_time("result", last=True)
        print(ct)

    elif debug_block == 1:
        DO_DEBUG = True
        DO_DEBUG_DETAIL = False
        count_blocks.DO_DEBUG = True
        count_refine.DO_DEBUG = True

        cfgs = dict(
            num_heads=1,
            window_size=8,
            sr_ratio=8,
            filter_rate=0.25)
        H = 128
        C = 64

        m = RefineVitLayer(C, C, 4, _cfg(cfgs), refine_block_cls=RefineVitBlock,
                        use_conv_in=False, embed_conv=False).cuda().eval()
        x = torch.randn(2, C, H, H).cuda()
        mask = torch.randn(2, H, H).cuda()

        with torch.no_grad():
            for i in range(100):
                m(x, mask)
        # print(count_blocks.str_total())
        print(count_refine.str_total())

'''
@ 0.3, 256, SeparableConvBNReLU
conv_in: 809.0287981033325, refine_0: 1872.8063297271729, conv_1: 528.459997177124,
refine_1: 1866.9838047027588, conv_2: 528.3396143913269, refine_2: 1866.813819885254,
conv_3: 531.2255663871765, refine_3: 1866.5091819763184

@ 0.3, 256, ConvBNReLU:
conv_in: 504.7813115119934, refine_0: 1871.290880203247, conv_1: 361.64678406715393, 
refine_1: 1850.1326065063477, conv_2: 361.4538242816925, refine_2: 1866.677095413208, 
conv_3: 361.05420780181885, refine_3: 1849.7959098815918

    @ 0.3, 128, ConvBNReLU:
    conv_in: 136.12988817691803, refine_0: 295.93270349502563, conv_1: 99.47635191679001,
    refine_1: 290.22112107276917, conv_2: 99.19190448522568, refine_2: 294.20719957351685,
    conv_3: 99.17142421007156, refine_3: 290.3256323337555
    @ 0.6, 128, ConvBNReLU:
    conv_in: 136.1936321258545, refine_0: 469.30422496795654, conv_1: 98.99612820148468,
    refine_1: 457.7097911834717, conv_2: 98.89708852767944, refine_2: 467.2232332229614,
    conv_3: 105.39036822319031, refine_3: 457.1436777114868
    @ 1.0, 128, ConvBNReLU:
    conv_in: 135.8121280670166, refine_0: 699.1440615653992, conv_1: 99.31955218315125,
    refine_1: 678.4814705848694, conv_2: 99.03718423843384, refine_2: 695.2453141212463,
    conv_3: 98.94192004203796, refine_3: 679.939742565155

@ 0.15, 256, ConvBNReLU:
conv_in: 503.4232006072998, refine_0: 1243.250015258789, conv_1: 359.89888048171997,
refine_1: 1234.6368894577026, conv_2: 359.78463888168335, refine_2: 1246.1372184753418,
conv_3: 359.62422466278076, refine_3: 1230.4522876739502, None: 2.130783997476101

@ 0.6, 256, ConvBNReLU:
conv_in: 506.5385274887085, refine_0: 3123.0229110717773, conv_1: 363.3917119503021,
refine_1: 3081.0002307891846, conv_2: 362.8057613372803, refine_2: 3114.7546253204346,
conv_3: 363.5554881095886, refine_3: 3078.772228240967, None: 2.1734719835221767
'''