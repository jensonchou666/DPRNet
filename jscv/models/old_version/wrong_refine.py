import torch.nn.functional as F
import torch.nn as nn
import torch

from .cnn import SeparableConvBNReLU
from jscv.utils.statistics import StatisticModel
from jscv.utils.analyser import add_analyse_item, AnalyseItem
from jscv.utils.overall import global_dict
from jscv.utils.utils import do_once

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=True,
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


class WrongReFineBlock(nn.Module):
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
            ucr_channels: wrong-range mlp-block channels
            uctr_mlp_blocks: wrong-range mlp-block numbers
        """
        super(WrongReFineBlock, self).__init__()

        assert uctr_mlp_blocks > 2, "Too little wrong-range mlp blocks"
        assert fullmap_conv_blocks >= 1, "Miss conv blocks"
        self.channels = channels
        self.out_channels = out_channels
        self.uctr_channels = uctr_channels
        self.uctr_mlp_blocks = uctr_mlp_blocks
        self.fullmap_conv_blocks = fullmap_conv_blocks
        self.return_BCHW = return_BCHW

        """ wrong-range mlp """
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
                wrong_mask: torch.Tensor,
                origin_shape: torch.Size):
        '''
            feature_map: [BHW, C]
            wrong_mask: [BHW]
        '''
        B, C, H, W = origin_shape

        assert self.channels == C == feature_map.shape[1]

        """ wrong-range mlp """
        wrong = feature_map[wrong_mask]  # [X, C]

        wrong = self.u_mlp_in(wrong)
        for blk in self.u_mlps:
            wrong = wrong + blk(wrong)
        wrong = self.u_mlp_out(wrong)

        feature_map[wrong_mask] = wrong

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
                 wrong_mask: torch.Tensor,
                 origin_shape: torch.Size):
        stat.statistic_self_alone()
        B, C, H, W = origin_shape
        assert self.channels == C == feature_map.shape[1]
        wrong = feature_map[wrong_mask]  # [X, C]
        wrong = stat.step(self.u_mlp_in, (wrong))
        for blk in self.u_mlps:
            wrong = wrong + stat.step(blk, (wrong))
        wrong = stat.step(self.u_mlp_out, (wrong))
        feature_map[wrong_mask] = wrong
        feature_map = feature_map.reshape(B, H, W, C).permute(0, 3, 1, 2)
        if self.fullmap_conv_blocks > 1:
            for blk in self.conv_blocks:
                feature_map = feature_map + stat.step(blk, (feature_map))
        feature_map = stat.step(self.out_conv, (feature_map))



class AnalyseWrong(AnalyseItem):
    name_dict = dict(pred_wrong='pred_wrong.png',
                     wrong_dilate="wrong_dilate.png",
                     wrong_mask="wrong_mask.png")

    def call(self, datas: dict, dist_datas: dict):
        dist_datas["pred_wrong"] = {"data": self.model.wrong_map, "type": "gray", "tensor": True}
        dist_datas["wrong_dilate"] = {"data": self.model.wrong_dilate, "type": "gray", "tensor": True}
        dist_datas["wrong_mask"] = {"data": self.model.wrong_mask, "type": "gray", "tensor": True}




class WrongReFineModel(nn.Module):
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
                 filter_gate=0.4):
        super(WrongReFineModel, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.channel_div = channel_div
        self.uctr_channel_multi = uctr_channel_multi

        self.dilate_kernel_size = dilate_kernel_size
        self.filter_gate = filter_gate
        
        dilate_padding = (dilate_kernel_size - 1) // 2


        if dilate_kernel_size <= 1:
            self.dilate_block = torch.nn.Identity()
        else:
            self.dilate_block = torch.nn.MaxPool2d(kernel_size=dilate_kernel_size,
                                                   stride=1, padding=dilate_padding)

        C = in_channels

        blocks = []
        self.block_channels = []
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                return_BCHW = True
            else:
                return_BCHW = False
            cout = int(C // channel_div)
            self.block_channels.append(cout)
            blocks.append(WrongReFineBlock(
                C, cout, int(C * uctr_channel_multi),
                uctr_mlp_blocks, fullmap_conv_blocks, return_BCHW,
                mlp_cfg, conv_cfg
            ))

            C = cout

        self.blocks = nn.ModuleList(blocks)

        ''' 分析 '''
        aw = AnalyseWrong()
        aw.model = self
        add_analyse_item(aw)


    def before_forward(self, feature_map: torch.Tensor, pred_wrong: torch.Tensor):

        B, C, H, W = feature_map.shape
        assert pred_wrong.shape[0] == B
        B, _, Hc, Wc = pred_wrong.shape

        assert C == self.in_channels

        ''' 调整到一致大小 '''
        if H != Hc or W != Wc:
            pred_wrong = F.interpolate(pred_wrong, (H, W), mode="bilinear")
        
        pred_wrong = pred_wrong.squeeze(1)

        self.wrong_map = torch.sigmoid(pred_wrong)
        
        # # [B, H, W] -> [BHW]
        self.wrong_dilate = self.dilate_block(self.wrong_map)

        #TODO 若结果为空向量
        self.wrong_mask = self.wrong_dilate > self.filter_gate


        wrong_mask = self.wrong_mask.reshape(-1)


        C = feature_map.shape[1]
        feature_shape = [B, C, H, W]

        # [B, C, H, W] -> [B, H, W, C] - > [BHW, C]
        feature_map = feature_map.permute(0, 2, 3, 1).reshape(-1, C)

        return feature_map, wrong_mask, feature_shape



    def forward(self, feature_map: torch.Tensor, pred_wrong: torch.Tensor):
        """
            feature_map:    经多级特征融合后的高分辨率特征图
            pred_wrong:     错误预测, 需监督训练(分配loss)
        """



        feature_map, wrong_mask, feature_shape = self.before_forward(feature_map, pred_wrong)

        for blk, blk_c in zip(self.blocks, self.block_channels):
            assert isinstance(blk, WrongReFineBlock)
            feature_map = blk(feature_map, wrong_mask, feature_shape)
            feature_shape[1] = blk_c



        return feature_map

    def traverse(self, stat: StatisticModel,
                 feature_map: torch.Tensor,
                 pred_wrong: torch.Tensor):
        stat.statistic_self_alone()
        feature_map, wrong_mask, feature_shape = self.before_forward(feature_map, pred_wrong)

        for blk, blk_c in zip(self.blocks, self.block_channels):
            assert isinstance(blk, WrongReFineBlock)
            feature_map = stat.step(blk, (feature_map, wrong_mask, feature_shape))
            feature_shape[1] = blk_c
