import torch.nn.functional as F
import torch.nn as nn
import torch

from .cnn import SeparableConvBNReLU
from jscv.utils.statistics import StatisticModel


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=False,
                 bn_kargs={},
                 act_cls=nn.ReLU6
                 ):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(in_channels, out_channels, bias=~use_bn)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels, **bn_kargs)
        self.act = act_cls()

    def forward(self, x):
        x = self.mlp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.act(x)


class UncertainReFineBlock(nn.Module):
    def __init__(self,
                 channels,
                 out_channels,
                 uctr_channels,
                 uctr_mlp_blocks=5,
                 fullmap_conv_blocks=2,
                 return_BCHW=False,
                 mlp_cfg={},
                 conv_cfg={}
                 ):
        """
            ucr_channels: uncertain-range mlp-block channels
            uctr_mlp_blocks: uncertain-range mlp-block numbers
        """
        super(UncertainReFineBlock, self).__init__()

        assert uctr_mlp_blocks > 2, "Too little uncertain-range mlp blocks"
        assert fullmap_conv_blocks >= 1, "Miss conv blocks"
        self.channels = channels
        self.out_channels = out_channels
        self.uctr_channels = uctr_channels
        self.uctr_mlp_blocks = uctr_mlp_blocks
        self.fullmap_conv_blocks = fullmap_conv_blocks
        self.return_BCHW = return_BCHW

        """ uncertain-range mlp """
        self.u_mlp_in = MLP(channels, uctr_channels, **mlp_cfg)
        self.u_mlps = nn.ModuleList([
            MLP(uctr_channels, uctr_channels, **mlp_cfg) for i
            in range(uctr_mlp_blocks - 2)
        ])
        self.u_mlp_out = MLP(uctr_channels, channels, **mlp_cfg)

        """ full-map conv """
        if fullmap_conv_blocks > 1:
            self.conv_blocks = nn.ModuleList([
                SeparableConvBNReLU(
                    channels, channels, 3, 1, 1, **conv_cfg
                ) for i in range(fullmap_conv_blocks - 1)
            ])

        self.out_conv = SeparableConvBNReLU(channels, out_channels, 3, 1, 1, **conv_cfg)

    #TODO 增加 shortcut
    def forward(self,
                feature_map: torch.Tensor,
                uncertain_mask: torch.Tensor,
                origin_shape: torch.Size):
        '''
            feature_map: [BHW, C]
            uncertain_mask: [BHW]
        '''
        B, C, H, W = origin_shape

        assert self.channels == C == feature_map.shape[1]

        """ uncertain-range mlp """
        uncertain = feature_map[uncertain_mask]  # [X, C]

        uncertain = self.u_mlp_in(uncertain)
        for blk in self.u_mlps:
            uncertain = uncertain + blk(uncertain)
        uncertain = self.u_mlp_out(uncertain)

        feature_map[uncertain_mask] = uncertain

        feature_map = feature_map.reshape(B, H, W, C).permute(0, 3, 1, 2)

        """ full-map conv """
        if self.fullmap_conv_blocks > 1:
            for blk in self.conv_blocks:
                feature_map = feature_map + blk(feature_map)
        feature_map = self.out_conv(feature_map)

        if self.return_BCHW:
            return feature_map
        else:
            return feature_map.permute(0, 2, 3, 1).reshape(-1, self.out_channels)

    def traverse(self, stat: StatisticModel,
                 feature_map: torch.Tensor,
                 uncertain_mask: torch.Tensor,
                 origin_shape: torch.Size):
        stat.statistic_self_alone()
        B, C, H, W = origin_shape
        assert self.channels == C == feature_map.shape[1]
        uncertain = feature_map[uncertain_mask]  # [X, C]
        uncertain = stat.step(self.u_mlp_in, (uncertain))
        for blk in self.u_mlps:
            uncertain = uncertain + stat.step(blk, (uncertain))
        uncertain = stat.step(self.u_mlp_out, (uncertain))
        feature_map[uncertain_mask] = uncertain
        feature_map = feature_map.reshape(B, H, W, C).permute(0, 3, 1, 2)
        if self.fullmap_conv_blocks > 1:
            for blk in self.conv_blocks:
                feature_map = feature_map + stat.step(blk, (feature_map))
        feature_map = stat.step(self.out_conv, (feature_map))



class UncertainReFineModel(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 num_blocks=2,
                 channel_div=2,
                 uctr_channel_multi=2,

                 uctr_mlp_blocks=4,
                 fullmap_conv_blocks=3,
                 mlp_cfg={},
                 conv_cfg={},

                 dilate_kernel_size=3,
                 filter_gate=0.4,
                 uncertain_type="top1",
                 concat_coarse=True,
                 coarse_pred_after_softmax=True):
        super(UncertainReFineModel, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.channel_div = channel_div
        self.uctr_channel_multi = uctr_channel_multi

        self.dilate_kernel_size = dilate_kernel_size
        self.filter_gate = filter_gate
        self.uncertain_type = uncertain_type
        self.after_softmax = coarse_pred_after_softmax
        self.concat_coarse = concat_coarse
        
        dilate_padding = (dilate_kernel_size - 1) // 2

        self.dilate_block = torch.nn.MaxPool2d(kernel_size=dilate_kernel_size,
                                               stride=1, padding=dilate_padding)
        
        C = in_channels
        if concat_coarse:
            C += num_classes
        blocks = []
        self.block_channels = []
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                return_BCHW = True
            else:
                return_BCHW = False
            cout = int(C // channel_div)
            self.block_channels.append(cout)
            blocks.append(UncertainReFineBlock(
                C, cout, int(C * uctr_channel_multi),
                uctr_mlp_blocks, fullmap_conv_blocks, return_BCHW,
                mlp_cfg, conv_cfg
            ))

            C = cout

        self.blocks = nn.ModuleList(blocks)


    def before_forward(self, feature_map: torch.Tensor, coarse_pred: torch.Tensor):
        B, C, H, W = feature_map.shape
        assert coarse_pred.shape[0] == B
        B, nC, Hc, Wc = coarse_pred.shape

        assert C == self.in_channels and nC == self.num_classes

        ''' 调整到一致大小 '''
        if H != Hc or W != Wc:
            coarse_pred = F.interpolate(coarse_pred, (H, W), mode="bilinear")
        
        ''' uncertain_map '''
        if not self.after_softmax:
            coarse_pred = torch.softmax(coarse_pred, dim=1)

        #coarse_pred.argmax(dim=dim)
        if self.uncertain_type == "top1":
            uncertain_map = 1 - torch.max(coarse_pred, dim=1)[0]  # B, H, W
        elif self.uncertain_type == "top2":
            top2_scores = torch.topk(coarse_pred, k=2, dim=1)[0]
            uncertain_map = top2_scores[:, 0] - top2_scores[:, 1]
            #TODO
            uncertain_map = torch.max(uncertain_map) - uncertain_map
        else:
            assert "wrong uncertain_type"

        # [B, H, W] -> [BHW]
        uncertain_map = self.dilate_block(uncertain_map).reshape(-1)

        #TODO 若结果为空向量
        uncertain_mask = uncertain_map > self.filter_gate

        if self.concat_coarse:
            feature_map = torch.concat([feature_map, coarse_pred], dim=1)

        C = feature_map.shape[1]
        feature_shape = [B, C, H, W]

        # [B, C, H, W] -> [B, H, W, C] - > [BHW, C]
        feature_map = feature_map.permute(0, 2, 3, 1).reshape(-1, C)


        return feature_map, uncertain_mask, feature_shape


    def forward(self, feature_map: torch.Tensor, coarse_pred: torch.Tensor):
        """
            feature_map:    经多级特征融合后的高分辨率特征图
            coarse_pred:    粗预测, 需监督训练(分配loss)
        """
        feature_map, uncertain_mask, feature_shape = self.before_forward(feature_map, coarse_pred)

        for blk, blk_c in zip(self.blocks, self.block_channels):
            assert isinstance(blk, UncertainReFineBlock)
            feature_map = blk(feature_map, uncertain_mask, feature_shape)
            feature_shape[1] = blk_c

        return feature_map

    def traverse(self, stat: StatisticModel,
                 feature_map: torch.Tensor,
                 coarse_pred: torch.Tensor):
        stat.statistic_self_alone()
        feature_map, uncertain_mask, feature_shape = self.before_forward(feature_map, coarse_pred)

        for blk, blk_c in zip(self.blocks, self.block_channels):
            assert isinstance(blk, UncertainReFineBlock)
            feature_map = stat.step(blk, (feature_map, uncertain_mask, feature_shape))
            feature_shape[1] = blk_c
