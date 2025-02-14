import sys, os
import torch.nn.functional as F

import torch.nn as nn
import thop

from jscv.utils.statistics import StatisticModel

from jscv.models.loss_filter import *
from jscv.models.pred_stat import *
import jscv.models.pred_stat_c as pred_stat_c

import jscv.utils.analyser as ana
from jscv.utils.overall import global_dict
from jscv.utils.utils import set_default

class StemLayerV1(nn.Module):
    def __init__(self, channels, in_chs=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chs, channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.act1(x)
        return self.maxpool(x)

from jscv.utils.utils import TimeCounter

class EncodeDecodeNet(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            decode_head: nn.Module,
            neck: nn.Module = None,
            extra_opt=None,
            decoder_input_tuple=False,
            decoder_input_include_h_w=False,
            time_counter=TimeCounter(False),
    ):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.neck = neck
        self.d_h_w = decoder_input_include_h_w
        self.decoder_input_tuple = decoder_input_tuple
        self.counter = time_counter
        self.extra_opt = extra_opt

    def forward(self, img, target=None):
        h, w = img.size()[-2:]
        features = self.backbone(img)

        self.counter.record_time("backbone")

        if self.neck is not None:
            features = self.neck(features)
            self.counter.record_time("neck")
        if self.d_h_w:
            arghw = {'h': h, 'w': w}
        else:
            arghw = {}
        if self.decoder_input_tuple:
            result = self.decode_head(features, **arghw)
        else:
            result = self.decode_head(*features, **arghw)
        self.counter.record_time("decode_head", last=True)
        
        args = {'h': h, 'w': w}
        
        if self.extra_opt is not None:
            result = self.extra_opt(result, args)
        
        return result

    def traverse(self, stat: StatisticModel, img, target=None):
        features = stat.step(self.backbone, (img), name=f'(backbone){self.backbone._get_name()}')
        if self.neck is not None:
            features = stat.step(self.neck, (features,), name=f'(neck){self.neck._get_name()}')
        if self.d_h_w:
            h, w = img.size()[-2:]
            arghw = {'h': h, 'w': w}
        else:
            arghw = {}
        decode_name = f'(decode_head){self.decode_head._get_name()}'
        if self.decoder_input_tuple:
            result = stat.step(self.decode_head, (features,), in_kargs=arghw, name=decode_name)
        else:
            result = stat.step(self.decode_head, features, in_kargs=arghw, name=decode_name)



class Segmentor(nn.Module):


    def __init__(
            self,
            net: nn.Module,
            loss_layer: nn.Module = None,
            loss_layer_input_dict=True,  # pred or result

            stem_layer=None,

            do_classify=False,
            out_chans=64,
            num_classes=-1,

            pred_reshape=True,
            target_reshape=False,
            extra_operations=[],  # "pred_statistic" "pred_statistic_c", "loss_statistic"
            ):
        super().__init__()

        self.net = net
        self.loss_layer = loss_layer
        self.do_classify = do_classify
        self.num_classes = num_classes
        self.extra_opts = extra_operations
        self.pred_reshape = pred_reshape
        self.target_reshape = target_reshape
        self.loss_layer_input_dict = loss_layer_input_dict  # (loss_layer_input == "pred")

        self.stem_layer = stem_layer
        if do_classify:
            self.classify_layer = nn.Conv2d(out_chans, num_classes, 1)

        #self.nonemodels = {}
        self.extera_init()



    def extera_init(self):
        # pass
        if "cfg" not in global_dict:
            return
        cfg = global_dict["cfg"]
        if "loss_statistic" in self.extra_opts:
            global_dict["loss_statistic"] = LossStatistic.from_cfg(cfg)
        elif "pred_statistic" in self.extra_opts:
            global_dict["pred_statistic"] = PredStatistic.from_cfg(cfg)
        elif "pred_statistic_c" in self.extra_opts:
            global_dict["pred_statistic_c"] = pred_stat_c.PredStatistic.from_cfg(cfg)

        
    def extra_operation(self, result):
        pred = result["pred"]
        if "loss_statistic" in self.extra_opts:
            global_dict["loss_statistic"].step(pred)
        elif "pred_statistic" in self.extra_opts:
            global_dict["pred_statistic"].step(pred)
        elif "pred_statistic_c" in self.extra_opts:
            global_dict["pred_statistic_c"].step(pred)


    def forward(self, img, target=None):
        x = img
        if self.stem_layer is not None:
            x = self.stem_layer(x)
        result = self.net(x)

        #? dict or tensor
        if not isinstance(result, dict):
            result = {"pred": result}

        if self.do_classify:
            result["pred"] = self.classify_layer(result["pred"])

        # print("@1", result["pred"].shape, img.shape, target.shape)
        if target is not None and result["pred"].shape[-2:] != target.shape[-2:]:
            if self.pred_reshape:
                result["pred"] = F.interpolate(result["pred"], target.shape[-2:], mode='bilinear')
            elif self.target_reshape:
                target = F.interpolate(target, result["pred"].shape[-2:], mode='bilinear')
        # print(result["pred"].shape, img.shape, target.shape)
        skip_loss = target is None or self.loss_layer is None

        if not skip_loss:
            if self.loss_layer_input_dict:
                loss_in = result
            else:
                loss_in = result["pred"]
            loss = self.loss_layer(loss_in, target)
            if isinstance(loss, dict):
                result["losses"] = loss
            else:
                result["losses"] = {'main_loss': loss}

        self.extra_operation(result)
        return result




    def traverse(self, stat: StatisticModel, img, target=None):
        x = img
        if self.stem_layer is not None:
            x = stat.step(self.stem_layer, (x))
        result = stat.step(self.net, (x))

        #? dict or tensor
        if not isinstance(result, dict):
            result = {"pred": result}

        if self.do_classify:
            result["pred"] = stat.step(self.classify_layer, (result["pred"]))
