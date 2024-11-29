import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from jscv.utils.statistics import StatisticModel
from jscv.utils.utils import TimeCounter, warmup, do_once

'''
    TODO:
        1、增大QKV通道， 仅窗口、 不确定截断的(精炼)全局下采样自注意力, 仅全局SA

        4、stage3使用二倍下采样PVT、stage4使用SA（精炼的）

        5、把不确定图也加入特征（可以乘以特征再加入）
        6、主干的前面必须有足够多的conv编码局部

        2、设置torch.var变量， 记录全局不确定值阈值、 替代topk筛选, 暂时不做（必然更好）
        3、插入conv、 插入FPN网络（在两个滑动块中间）

    下采样到窗口大小，滑动窗口



    固定filter_rate, 全局分支直接使用筛选窗口
    多尺度窗口, test是否需要多头
'''

DO_DEBUG = False
DO_DEBUG_DETAIL = False

count_blocks = TimeCounter(DO_DEBUG)
count_refine = TimeCounter(DO_DEBUG)


'''

'''
default_cfg = dict(
    num_heads=1,
    window_size=8,
    local_repeat=1,
    channel_ratio=2,
    filter_rate=0.3,
    qkv_bias=False,)

def _cfg(cfg_dict: dict={}):
    cfgs = default_cfg.copy()
    for k, v in cfg_dict.items():
        cfgs[k] = v
    return cfgs
def cfg_block_args(cfg_dict: dict={}):
    return _cfg(cfg_dict)






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


No_Shift = 0
Shift_H = 1
Shift_V = 2
Shift_HV = 3


class RefineVitBlock(nn.Module):

    def __init__(self,
                 channels,
                 shift_direct=No_Shift,
                 num_heads=1,
                 window_size=8,
                 local_repeat=1,
                 channel_ratio=2,
                 filter_rate=0.3,
                 qkv_bias=False,
                 ):
        super().__init__()
        C = self.channels = channels
        self.window_size = window_size
        self.local_repeat = local_repeat
        cr = self.channel_ratio = channel_ratio
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.filter_rate = filter_rate
        self.shift_direct = shift_direct
        self.scale = (C//num_heads)**-0.5

        self.norm = nn.LayerNorm(C)   #TODO 费时

        for i in range(local_repeat):
            C2 = C
            if i != 0: C2 = int(C*cr)
            self.__setattr__(f"linner_{i}", nn.Sequential(
                nn.Linear(C2, int(C*cr)), nn.GELU(),   #todo layernorm
            ))
            self.__setattr__(f"qkv_l_{i}", nn.Linear(int(C*cr), int(C*cr)*3, bias=qkv_bias))

        self.proj = nn.Sequential(
            nn.Linear(int(C*cr), C), nn.GELU(),
            )

        self.apply(init_weights)


    def window_shift(self, x, uncertain_map):
        shift_direct = self.shift_direct
        wsz = self.window_size
        wszS2 = wsz//2

        if shift_direct == Shift_H:
            ''' roll '''
            # torch.roll(x, shifts=(0, -wsz//2), dims=(1, 2))
            ''' pad '''
            x = F.pad(x, (wszS2, wszS2, 0, 0))
            uncertain_map = F.pad(uncertain_map, (wszS2, wszS2, 0, 0))
        elif shift_direct == Shift_V:
            x = F.pad(x, (0, 0, wszS2, wszS2))
            uncertain_map = F.pad(uncertain_map, (0, 0, wszS2, wszS2))
        elif shift_direct == Shift_HV:
            x = F.pad(x, (wszS2, wszS2, wszS2, wszS2))
            uncertain_map = F.pad(uncertain_map, (wszS2, wszS2, wszS2, wszS2))

        count_refine.record_time("pad")
        return x, uncertain_map


    def shift_recover(self, x):
        shift_direct = self.shift_direct
        wsz = self.window_size
        wszS2 = wsz//2
        
        # print("@1", x.shape)
        if shift_direct == Shift_H:
            x = x[:, :, :, wszS2:-wszS2]
        elif shift_direct == Shift_V:
            x = x[:, :, wszS2:-wszS2]
        elif shift_direct == Shift_HV:
            x = x[:, :, wszS2:-wszS2, wszS2:-wszS2]
        # print("@2", x.shape)
        return x


    def filter_window(self, x, uncertain_map):
        B, C, H, W = x.shape
        wsz = self.window_size
        nH, nW = H//wsz, W//wsz
        nWin = nH*nW
        winsz = wsz*wsz

        ''' 窗口筛选 '''
        win_unc_map = uncertain_map.reshape(B, nH, wsz, nW, wsz).transpose(
            2, 3).reshape(B, nWin, winsz)

        #1
        win_x = x.permute(0, 2, 3, 1).reshape(
            B, nH, wsz, nW, wsz, C).transpose(2, 3).contiguous().view(
                B*nWin, winsz, C)

        #todo 2 error
        # win_x = x.reshape(B, C, nH, wsz, nW, wsz).transpose(-2, -3).reshape(
        #     B, C, nWin, winsz).permute(0, 2, 3, 1)
        count_refine.record_time("make window")


        # B, nWin
        win_score = win_unc_map.mean(-1)    # mean? max?
        nWF = nW_filter = int(nWin * self.filter_rate)
        # B, nWF
        _, win_score_idx = torch.topk(win_score, k=nWF, dim=1, sorted=False)

        win_score_idx += create_batch_base_idx(B, nWin, win_score_idx.device)
        # count_refine.record_time("topk")

        win_score_idx = win_score_idx.view(B*nWF)
        win_x_filter = torch.index_select(win_x, dim=0, index=win_score_idx,)

        count_refine.record_time("filter_window")

        return win_x, win_x_filter, win_score_idx, nWF


    def forward(self, feature_map: torch.Tensor, uncertain_map: torch.Tensor):
        """
            单头
            feature_map:      B, C, H, W
            uncertain_map:    B, H, W
        """
        wsz = self.window_size
        local_repeat = self.local_repeat
        num_heads = self.num_heads
        winsz = wsz*wsz

        count_refine.record_time(first=True)

        x, uncertain_map = self.window_shift(feature_map, uncertain_map)


        win_x, win_x_filter, win_score_idx, nWF = self.filter_window(x, uncertain_map)


        xf = win_x_filter = self.norm(win_x_filter) #B*nWF, winsz, C

        count_refine.record_time("norm")

        B, C, H, W = x.shape
        nH, nW = H//wsz, W//wsz


        ''' local branch '''
        for i in range(local_repeat):
            linner = self.__getattr__(f"linner_{i}")
            qkv_l = self.__getattr__(f"qkv_l_{i}")
            if i == 0:
                xf = linner(xf) #todo no shortcut
            else:
                xf = xf + linner(xf)
            C2 = int(C*self.channel_ratio)
            xf = xf

            count_refine.record_time(f"linner ({i})")

            qkv = qkv_l(xf).view(B*nWF, winsz, 3, num_heads, C2//num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]    # B*nWF, num_heads, winsz, C2//num_heads

            #todo
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale  #!!!!!!!!!!!!!!!!!!!!
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            xf = xf + (attn @ v).transpose(1, 2).reshape(B*nWF, winsz, C2)

            if DO_DEBUG and do_once(None, "_local_"):
                print("local q", q.shape, ", k", k.shape)
            count_refine.record_time(f"local qkv ({i})")


        win_x_filter = win_x_filter + self.proj(xf)
        # win_x_filter = self.proj_drop(win_x_filter)
        
        count_refine.record_time("proj")

        ''' compose '''
        win_x.index_add_(0, win_score_idx, win_x_filter)
        # win_x.index_copy_(0, win_score_idx, win_x_filter)  #TODO compare


        x = win_x.reshape(B, nH, nW, wsz, wsz, C).transpose(2, 3).reshape(
            B, H, W, C).permute(0, 3, 1, 2)


        count_refine.record_time("compose + reshape")

        x = self.shift_recover(x)

        count_refine.record_time("shift_recover", last=True)
        
        if DO_DEBUG:
            if do_once(self, "q-shape"):
                print("q", q.shape)
            if DO_DEBUG_DETAIL:
                print(f"\nrefine_block({self.shift_direct})", count_refine)

        return x.contiguous()



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


class DWConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6):
        super(DWConvBNReLU, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      padding=get_padding(kernel_size, stride),
                      groups=in_channels,
                      bias=False),
            norm_layer(out_channels), act_layer())


class FPNNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,):
        pass


class RefineVitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_blocks,
                 block_args: dict,
                 uncertain_type="top1", # top1 or top2
                 coarse_pred_after_softmax=False,
                 embed_conv=True,
                 conv_cls=SeparableConvBNReLU,
                 conv_kernel_size=7,
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
        self.after_softmax = coarse_pred_after_softmax
        if use_conv_in:
            self.conv_in = conv_cls(in_channels, channels, conv_kernel_size)
        else:
            assert in_channels == channels
        for i in range(num_blocks):
            if i != 0 and embed_conv:
                self.__setattr__(f"conv_{i}", conv_cls(channels, channels, conv_kernel_size))
            
            j = i % 4
            # if j == 0:
            #     shift_direct = No_Shift
            # elif j == 1:
            #     shift_direct = Shift_H
            # elif j == 2:
            #     shift_direct = Shift_V
            # elif j == 3:
            #     shift_direct = Shift_HV

            self.__setattr__(f"refine_{i}", refine_block_cls(channels, shift_direct=j, **block_args))

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







def refine_vit_layer_args_normal(channels=128,
                                 channel_ratio=2,
                                 local_repeat=2,
                                 filter_rate=0.4):
    return dict(channels=channels,
                num_blocks=4,
                block_args=cfg_block_args(dict(
                    channel_ratio=channel_ratio,
                    local_repeat=local_repeat,
                    filter_rate=filter_rate,
                    window_size=8)),
                use_conv_in=True,
                uncertain_type="top1",
                embed_conv=True,
                conv_cls=SeparableConvBNReLU,
                conv_kernel_size=3,
                )







if __name__ == "__main__":
    debug_block = 0

    torch.cuda.set_device(3)
    warmup(20)
    
    if debug_block == 0:
        '''
            H=512, C=128, {'num_heads': 4, 'channel_ratio': 2, 'window_size': 8, 'local_repeat': 2, 'filter_rate': 0.1}:    17565.70
            H=512, C=128, {'num_heads': 4, 'channel_ratio': 2, 'window_size': 8, 'local_repeat': 2, 'filter_rate': 1}:      85297.40
        '''
        cfgs = dict(
            num_heads=4,
            channel_ratio=2,
            window_size=8,
            local_repeat=2,
            filter_rate=0.4)
        H = 128
        C = 128
        m = RefineVitLayer(C, C, 4, _cfg(cfgs), refine_block_cls=RefineVitBlock,
                        use_conv_in=False, embed_conv=False).cuda().eval()
        x = torch.randn(2, C, H, H).cuda()
        mask = torch.randn(2, H, H).cuda()
        ct = TimeCounter(True)
        ct.record_time(first=True)

        with torch.no_grad():
            for i in range(100):
                m(x, mask)
        ct.record_time(f"H={H}, C={C}, {cfgs}", last=True)
        print(ct)

    elif debug_block == 1:
        '''
        512, 0.08, C=128
            pad: 685.7559368032962, make window: 1104.8465288877487, filter_window: 123.72047933936119,
            norm: 562.1419498920441, global qkv: 1697.941825389862, linner (0): 347.5468803048134,
            local qkv (0): 646.4980791807175, linner (1): 344.80160015821457, local qkv (1): 646.662432551384,
            proj: 344.2577275633812, compose + reshape: 817.9468673467636, shift_recover: 12.772128006443381

        512, 0.08, C=64
            pad: 351.2609611079097, make window: 548.005538880825, filter_window: 82.86662422120571,
            norm: 532.0736955404282, global qkv: 1052.1280002593994, linner (0): 207.6184322834015,
            local qkv (0): 289.5943676829338, linner (1): 204.85212776064873, local qkv (1): 289.5105280280113,
            proj: 204.2684475183487, compose + reshape: 411.4712040424347, shift_recover: 12.892831971868873
        '''
        DO_DEBUG = True
        DO_DEBUG_DETAIL = False
        count_blocks.DO_DEBUG = True
        count_refine.DO_DEBUG = True

        cfgs = dict(
            num_heads=4,
            window_size=8,
            local_repeat=2,
            channel_ratio=2,
            filter_rate=0.5)
        H = 256
        C = 128

        m = RefineVitLayer(C, C, 4, _cfg(cfgs), refine_block_cls=RefineVitBlock, conv_cls=ConvBNReLU,
                           use_conv_in=True, embed_conv=True).cuda().eval()
        x = torch.randn(2, C, H, H).cuda()
        mask = torch.randn(2, H, H).cuda()

        with torch.no_grad():
            if DO_DEBUG_DETAIL:
                m(x, mask)
            else:
                for i in range(20):
                    m(x, mask)

        
        print(count_blocks.str_total(), "\n")
        print(count_refine.str_total())
