import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from jscv.utils.statistics import StatisticModel
from jscv.utils.utils import TimeCounter, warmup, do_once


DO_DEBUG = False
DO_DEBUG_DETAIL = False

count_blocks = TimeCounter(DO_DEBUG)
count_refine = TimeCounter(DO_DEBUG)


default_cfg = dict(
    num_heads=1,
    window_size=8,
    local_repeat=1,
    pool_size=8,
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
Shift_HV = 1
Shift_H = 2
Shift_V = 3


class RefineVitBlock(nn.Module):

    def __init__(self,
                 channels,
                 shift_direct=No_Shift,
                 num_heads=1,
                 use_global=True,
                 window_size=8,
                 local_repeat=1,
                 pool_size=8, #全局下采样后的尺度
                 filter_rate=0.3,
                 qkv_bias=False,
                 ):
        super().__init__()
        C = self.channels = channels
        self.use_global = use_global
        self.window_size = window_size
        self.local_repeat = local_repeat
        self.pool_size = pool_size
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.filter_rate = filter_rate
        self.shift_direct = shift_direct
        self.scale = (C//num_heads)**-0.5

        self.norm = nn.LayerNorm(C)   #TODO 费时

        if use_global:
            #self.max_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
            self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
            self.q_g = nn.Linear(C, C, bias=qkv_bias)
            self.kv_g = nn.Linear(C, C*2, bias=qkv_bias)

        for i in range(local_repeat):
            self.__setattr__(f"linner_{i}", nn.Sequential(
                nn.Linear(C, C), nn.GELU(),   #todo layernorm
            ))
            self.__setattr__(f"qkv_l_{i}", nn.Linear(C, C*3, bias=qkv_bias))

        self.proj = nn.Sequential(
            nn.Linear(C, C), nn.GELU(),
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

        # count_refine.record_time("pad")
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
        win_x = x.permute(0, 2, 3, 1).contiguous().reshape(
            B, nH, wsz, nW, wsz, C).transpose(2, 3).contiguous().reshape(
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

        win_score_idx = win_score_idx.reshape(B*nWF)
        win_x_filter = torch.index_select(win_x, dim=0, index=win_score_idx,)

        count_refine.record_time("filter_window")

        return win_x.contiguous(), win_x_filter.contiguous(), win_score_idx, nWF


    def forward(self, feature_map: torch.Tensor, uncertain_map: torch.Tensor):
        """
            单头
            feature_map:      B, C, H, W
            uncertain_map:    B, H, W
        """
        wsz = self.window_size
        local_repeat = self.local_repeat
        pz = self.pool_size
        winsz = wsz*wsz

        count_refine.record_time(first=True)

        x, uncertain_map = self.window_shift(feature_map, uncertain_map)


        win_x, win_x_filter, win_score_idx, nWF = self.filter_window(x, uncertain_map)


        # win_x_filter = self.norm(win_x_filter) #B*nWF, winsz, C
        # count_refine.record_time("norm")

        B, C, H, W = x.shape
        nH, nW = H//wsz, W//wsz
        
        if self.use_global:
            ''' global branch '''
            global_x = self.pool(feature_map) # B, C, pz, pz
            q = self.q_g(win_x_filter).reshape(B, nWF*winsz, C)
            kv = self.kv_g(global_x.reshape(B, C, pz*pz).transpose(1, 2)).reshape(B, pz*pz, 2, C)
            k, v = kv[:, :, 0], kv[:, :, 1]
            del kv

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            win_x_filter = win_x_filter + (attn @ v).reshape(B*nWF, winsz, C)

            count_refine.record_time("global qkv")
            if DO_DEBUG and do_once(None, "_g_"):
                print("global q", q.shape, ", k", k.shape)


        ''' local branch '''
        for i in range(local_repeat):
            linner = self.__getattr__(f"linner_{i}")
            qkv_l = self.__getattr__(f"qkv_l_{i}")

            win_x_filter = win_x_filter + linner(win_x_filter) #todo no shortcut
            
            count_refine.record_time(f"linner ({i})")
            
            qkv = qkv_l(win_x_filter).reshape(B*nWF, winsz, 3, C)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

            # q, k, v = torch.split(qkv_l(win_x_filter), (C, C, C), dim=-1) #todo compare
            

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            win_x_filter = win_x_filter + (attn @ v).transpose(1, 2).reshape(B*nWF, winsz, C)


            if DO_DEBUG and do_once(None, "_local_"):
                print("local q", q.shape, ", k", k.shape)
            count_refine.record_time(f"local qkv ({i})")



        win_x_filter = win_x_filter + self.proj(win_x_filter)
        # win_x_filter = self.proj_drop(win_x_filter)
        
        count_refine.record_time("proj")

        ''' compose '''
        # win_x.index_add_(0, win_score_idx, win_x_filter)
        win_x.index_copy_(0, win_score_idx, win_x_filter)  #TODO compare
        
        # bad
        # x = win_x.reshape(B, nH, nW, wsz, wsz, C).permute(
        #     0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

        x = win_x.reshape(B, nH, nW, wsz, wsz, C).transpose(2, 3).reshape(
            B, H, W, C).permute(0, 3, 1, 2)
        


        count_refine.record_time("compose + reshape")

        x = self.shift_recover(x)
        
        count_refine.record_time("shift_recover", last=True)
        
        if DO_DEBUG_DETAIL:
            print(f"\nrefine_block({self.shift_direct})", count_refine)

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



class RefineVitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 out_channels,
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

        self.conv_out = conv_cls(channels, out_channels, 1)

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

        x = self.conv_out(x.contiguous())

        count_blocks.record_time("conv_out", last=True)
        if DO_DEBUG_DETAIL:
            print(count_blocks)
        
        return x



from jscv.models.matte.matte import *



def refine_vit_layer_args_normal(channels=20, local_repeat=1, filter_rate=0.15):
    return dict(channels=channels,
                out_channels=1,
                num_blocks=2,
                block_args=cfg_block_args(dict(
                    local_repeat=local_repeat,
                    filter_rate=filter_rate,
                    window_size=16,
                    pool_size=8)),
                use_conv_in=False,
                embed_conv=False,
                # conv_cls=SeparableConvBNReLU,
                # conv_kernel_size=3,
                # refine_block_cls=RefineVitBlock,
                )


class Matting_RefineVit(MattingBase):

    
    def __init__(self,
                 backbone: str,
                 in_chan=3,
                 backbone_scale: float = 1/4,
                 refine_scale: float = 1/2,
                 refine_cfgs=refine_vit_layer_args_normal(),
                 ):
        refine_channels = refine_cfgs['channels']
        super().__init__(backbone, out_channels = 1 + refine_channels - 3)
        self.backbone_scale = backbone_scale
        self.refine_scale = refine_scale
        self.in_chan = in_chan
        
        self.refiner = RefineVitLayer(
            refine_channels, **refine_cfgs
        )
        
        self.do_refine = True

    def forward(self, x, **kargs):
        
        if "time_counter" in kargs:
            test_speed = True
            counter = kargs["time_counter"]
            counter.begin()
        else:
            test_speed = False

        x_org = x
        org_shape = x.shape
        h, w = org_shape[2:]
        x = F.interpolate(x,
                          scale_factor=self.backbone_scale,
                          mode='bilinear',
                          align_corners=False,
                          recompute_scale_factor=True)
        
        # Base
        x, *shortcuts = self.backbone(x)

        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)

        pha_sm = x[:, 0:1].clamp_(0., 1.)
        err_sm = x[:, 1:2].clamp_(0., 1.)
        hid_sm = x[:, 2:].relu_()
        
        if test_speed:
            counter.record_time("coarse")

        if self.do_refine:
            x = torch.cat([pha_sm, hid_sm], 1)
            x = F.interpolate(x, [h//2, w//2], mode='bilinear')
            x_org = F.interpolate(x_org, [h//2, w//2], mode='bilinear')
            x = torch.cat([x, x_org], 1)
            err_sm = F.interpolate(err_sm, [h//2, w//2], mode='bilinear')

            # print("@@@@@@@@", x.shape, err_sm.shape)
            if do_once(self, "_22"):
                print(x.shape)
            pha = self.refiner(x, torch.squeeze(err_sm, 1))
            pha = pha.clamp_(0., 1.)

            if test_speed:
                counter.record_time("refine", last=True)

            # print(pha.shape, pha_sm.shape, err_sm.shape)
            return {'pred': pha, "coarse_pred": pha_sm, "pred_err": err_sm}
        else:
            # print(pha_sm.shape, err_sm.shape)
            pha_sm = F.interpolate(pha_sm, org_shape[2:], mode='bilinear')
            return {'pred': pha_sm, "pred_err": err_sm}



def refine_vit_layer_args(channels=40, local_repeat=2, filter_rate=0.2):
    return dict(channels=channels,
                out_channels=1,
                num_blocks=2,

                block_args=cfg_block_args(dict(
                    local_repeat=local_repeat,
                    filter_rate=filter_rate,
                    use_global=False,
                    window_size=8,
                    pool_size=16)),
                use_conv_in=False,
                embed_conv=False,
                
                )

if __name__ == "__main__":

    torch.cuda.set_device(3)

    epoach = 100
    total = False

    model = Matting_RefineVit('resnet50',
                              backbone_scale=1/4,
                              refine_scale=1/2,
                              refine_cfgs=refine_vit_layer_args(),).cuda()
    model.do_refine = True
    
    
    from jscv.utils.utils import warmup
    from jscv.utils.utils import TimeCounter

    
    warmup(20)

    x = torch.rand([2, 3, 1024, 1024]).cuda()



    if total:
        ct = TimeCounter(True)

        for i in range(50):
            model(x, time_counter=ct)

        print(ct.str_total(50))
    else:
        DO_DEBUG = True

        count_refine.DO_DEBUG = True
        count_blocks.DO_DEBUG = True

        for i in range(50):
            model(x)

        print(count_refine.str_total())
        print("\ncount_blocks:")
        print(count_blocks.str_total())
