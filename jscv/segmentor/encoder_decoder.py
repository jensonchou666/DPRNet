import sys, os
import torch.nn.functional as F

import torch.nn as nn


from jscv.utils.statistics import StatisticModel

from jscv.models.loss_filter import *
from jscv.models.pred_stat import *

import jscv.utils.analyser as ana
from jscv.utils.overall import global_dict
from jscv.utils.utils import set_default



def encode_channels_pyramid(backbone):
    embed_dim = backbone.embed_dim
    return [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]


def encode_channels_columnar(backbone):
    embed_dim = backbone.embed_dim
    return [embed_dim, embed_dim, embed_dim, embed_dim]


class EncoderDecoder(nn.Module):

    # BACKBONE_OUT = 0
    # DECODER_OUT = 1

    def __init__(
            self,
            backbone: nn.Module,
            decoder: nn.Module,
            loss_layer: nn.Module = None,
            neck: nn.Module = None,

            decoder_input_tuple=False,
            decoder_input_include_h_w=False,
            loss_layer_input_pred=False,  # pred or result

            pred_reshape=True,
            extra_operations=[],  # "pred_statistic" "loss_statistic"
            stages=4):
        super().__init__()

        self.backbone = backbone
        self.decoder = decoder
        self.loss_layer = loss_layer
        self.stages = stages
        self.neck = neck

        self.extra_opts = extra_operations

        self.d_h_w = decoder_input_include_h_w
        self.decoder_input_tuple = decoder_input_tuple
        self.pred_reshape = pred_reshape

        self.loss_layer_input_pred = loss_layer_input_pred  # (loss_layer_input == "pred")

        #self.nonemodels = {}
        self.extera_init()

    def extera_init(self):
        cfg = global_dict["cfg"]
        if "loss_statistic" in self.extra_opts:
            global_dict["loss_statistic"] = LossStatistic.from_cfg(cfg)
        elif "pred_statistic" in self.extra_opts:
            global_dict["pred_statistic"] = PredStatistic.from_cfg(cfg)

        
    def extra_operation(self, result):
        pred = result["pred"]
        if "loss_statistic" in self.extra_opts:
            global_dict["loss_statistic"].step(pred)
        elif "pred_statistic" in self.extra_opts:
            global_dict["pred_statistic"].step(pred)


    def forward(self, img, target=None):
        features = self.backbone(img)
        if self.neck is not None:
            features = self.neck(features)

        dit = self.decoder_input_tuple
        if self.d_h_w:
            h, w = img.size()[-2:]
            arghw = {'h': h, 'w': w}
        else:
            arghw = {}

        if dit:
            result = self.decoder(features, **arghw)
        else:
            result = self.decoder(*features, **arghw)
        
        #? dict or tensor
        if not isinstance(result, dict):
            result = {"pred": result}

        if self.pred_reshape:
            if result["pred"].shape[-2:] != img.shape[-2:]:
                result["pred"] = F.interpolate(result["pred"], img.shape[-2:], mode='bilinear')

        skip_loss = target is None or self.loss_layer is None

        if not skip_loss:
            if self.loss_layer_input_pred:
                loss_in = result["pred"]
            else:
                loss_in = result
            loss = self.loss_layer(loss_in, target)
            if isinstance(loss, dict):
                result["losses"] = loss
            else:
                result["losses"] = {'main_loss': loss}

        self.extra_operation(result)
        return result




    def traverse(self, stat: StatisticModel, img, target=None):
        features = stat.step(self.backbone, (img), name=f'(backbone){self.backbone._get_name()}')

        if self.neck is not None:
            features = stat.step(self.neck, (features,), name=f'(neck){self.neck._get_name()}')

        dint = self.decoder_input_tuple
        if self.d_h_w:
            h, w = img.size()[-2:]
            arghw = {'h': h, 'w': w}
        else:
            arghw = {}
        
        decode_name = f'(decode_head){self.decoder._get_name()}'
        if dint:
            result = stat.step(self.decoder, (features,), in_kargs=arghw, name=decode_name)
        else:
            result = stat.step(self.decoder, features, in_kargs=arghw, name=decode_name)
