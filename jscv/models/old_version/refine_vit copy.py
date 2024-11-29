import torch.nn.functional as F
import torch.nn as nn
import torch

from jscv.utils.statistics import StatisticModel

class RefineVitBlock(nn.Module):

    def __init__(self,
                 channels,
                 window_size=8,
                 groups=8,
                 sr_ratio=8,
                 qkv_bias=False,
                 ):
        C = self.channels = channels
        self.window_size = window_size
        self.groups = groups
        self.sr_ratio = sr_ratio
        self.qkv_bias = qkv_bias
        assert sr_ratio > 1

        self.sr_conv = nn.Conv2d(C, C, sr_ratio, sr_ratio)
        self.sr_norm = nn.LayerNorm(C)
        self.act = nn.GELU()

        self.q_g = nn.Linear(C, C, bias=qkv_bias)
        self.kv_g = nn.Linear(C, C * 2, bias=qkv_bias)
        self.q_l = nn.Linear(C, C, bias=qkv_bias)
        self.kv_l = nn.Linear(C, C * 2, bias=qkv_bias)

    def forward(self, feature_map: torch.Tensor, uncertain_map: torch.Tensor):
        """
            feature_map:      B, C, H, W
            uncertain_map:    B, H, W
        """
        wsz = self.window_size
        groups = self.groups
        sr_ratio = self.sr_ratio
        
        x = feature_map
        B, C, H, W = x.shape
        H1, W1 = H//sr_ratio, W//sr_ratio
        hw, hw1 = H*W, H1*W1
        nH, nW = H//wsz, W//wsz
        nWin = nH*nW

        x1 = x.permute(0, 2, 3, 1)
        
        ''' local branch'''
        
        win_x = x1.reshape(B, nH, wsz, nW, wsz, C).transpose(2, 3).reshape(B, nWin, wsz*wsz, C)

        win_unc_map = uncertain_map.reshape(B, nH, wsz, nW, wsz).transpose(2, 3).reshape(B, nWin, wsz*wsz)
        win_score = win_unc_map.mean(-1)    # mean? max?

        win_score_sorted, win_score_idx = torch.sort(win_score, dim=1) # B, nWin
        extra_g = nWin % groups
        if extra_g > 0:
            win_score_idx = win_score_idx[:, extra_g:]
        nWin2 = win_score_idx.shape[-1]

        win_x_gatherd = torch.gather(

        win_unc_map_sorted = torch.gather(
            win_unc_map, dim=1, index=win_score_idx.unsqueeze(-1).expand_as(win_unc_map))
        win_unc_map_sorted = win_unc_map_sorted.reshape(B, groups, nWin2//groups, wsz*wsz)


        local_q = self.q_l(win_x)
        local_kv = self.kv_l(win_x)


        ''' global branch'''
        global_x = self.sr_conv(x).reshape(B, C, hw1).permute(0, 2, 1)
        global_x = self.act(self.sr_norm(global_x))



class RefineVitLayer(nn.Module):
    def __init__(self,
                 channels,
                 uncertain_type="top1", # top1 or top2
                 ):
        self.channels = channels
        self.uncertain_type = uncertain_type

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
        pass