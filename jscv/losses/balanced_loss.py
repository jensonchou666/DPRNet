import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from functools import partial


from .soft_ce import SoftCrossEntropyLoss

from jscv.utils.analyser import add_analyse_item, AnalyseItem
from jscv.models.pred_stat import *
from jscv.utils.overall import global_dict


class AnalyseLossMap(AnalyseItem):
    name_dict = dict(org_lossmap='org_lossmap.png',
                     weighted_lossmap="weighted_lossmap.png")

    def call(self, datas: dict, dist_datas: dict):
        dist_datas["org_lossmap"] = {"data": self.layer.lossmap, "type": "gray_to_jet", "tensor": True}
        dist_datas["weighted_lossmap"] = {"data": self.layer.wlossmap, "type": "gray_to_jet", "tensor": True}




def weighted_label_smoothed_nll_loss(
    input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
    epsilon: float, ignore_index=None, reduction="mean", dim=1,
    input_logits=True,
) -> torch.Tensor:
    """
        相当于 pos_neg_rate=1, focal_gamma=0 的 MCELoss, 更好, 而且速度更快

        input: [B, C, H, W]
        weight: [B, H, W] or None
    """
    if input_logits:
        lprobs = torch.log_softmax(input, dim=dim)
    else:
        lprobs = torch.log(input)

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

    nll_loss = nll_loss.squeeze(dim)
    smooth_loss = smooth_loss.squeeze(dim)

    eps_i = epsilon / lprobs.size(dim)
    lossmap = nll_loss * (1.0 - epsilon)
    lossmap += eps_i * smooth_loss

    if weight is not None:
        wlossmap = lossmap * weight.to(lossmap.device)
    else:
        wlossmap = lossmap

    if reduction == "sum":
        return wlossmap.sum(), lossmap, wlossmap
    if reduction == "mean":
        return wlossmap.mean(), lossmap, wlossmap



def _ce_loss_(pred, target, nC, pos_neg_rate, focal_gamma=None):
    target_one_hot = F.one_hot(target, num_classes=nC).float()
    # if weight is None:
    #     weight = torch.ones(pred.shape[:-1])
    lossmap = []

    for c in range(nC):
        p, t = pred[..., c], target_one_hot[..., c]
        mapc = F.binary_cross_entropy(p, t, reduction="none")
        mapc *= pos_neg_rate * t + (1 - pos_neg_rate) * (1 - t)
        if focal_gamma is not None and focal_gamma != 0:
            pt = p * (1 - t) + (1 - p) * t
            # if reduced_threshold is None:
            mapc *= pt.pow(focal_gamma)
        lossmap.append(mapc)
    return torch.stack(lossmap, -1).sum(-1)


def multiclass_cross_entropy_loss(input: torch.Tensor, target: torch.Tensor, num_classes,
                                  ignore_index=None, pos_neg_rate=0.5,
                                  focal_gamma=None, input_logits=True):
    """
        要求： input: B, C, H, W ; target: B, H, W
    """
    B, C, H, W = input.shape
    pred = input.permute(0, 2, 3, 1).reshape(-1, C)
    if input_logits:
        pred = torch.softmax(pred, dim=1)
    target = target.to(pred.device).clone().reshape(-1)

    if ignore_index is not None:
        ignore_mask = target == ignore_index
        target[ignore_mask] = 0
        lossmap = _ce_loss_(pred, target, num_classes, pos_neg_rate, focal_gamma)
        lossmap[ignore_mask] = 0
    else:
        lossmap = _ce_loss_(pred, target, num_classes, pos_neg_rate, focal_gamma)
    return lossmap.reshape(B, H, W)


class MCELoss(nn.Module):
    def __init__(self, num_classes, ignore_index=None, pos_neg_rate=0.5,
                 focal_gamma=None, input_logits=True):
        super(MCELoss, self).__init__()
        self.loss_func = partial(
            multiclass_cross_entropy_loss,
            num_classes=num_classes,
            ignore_index=ignore_index,
            pos_neg_rate=pos_neg_rate,
            focal_gamma=focal_gamma,
            input_logits=input_logits
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        lossmap = self.loss_func(input, target)
        return lossmap.mean()



#TODO 暂时
class SoftBalancedLoss_V1(nn.Module):
    def __init__(self, smooth_factor=0.05, ignore_index=None):
        super(SoftBalancedLoss_V1, self).__init__()

        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index

        self.loss_func = partial(
            weighted_label_smoothed_nll_loss,
            epsilon=smooth_factor,
            ignore_index=ignore_index,
            dim=1
        )

        alm = AnalyseLossMap()
        alm.layer = self
        add_analyse_item(alm)


    def _forward_(self, input, target, pstat):
        assert isinstance(pstat, PredStatistic)

        weight_map = pstat.pred_stat(input, target)

        assert weight_map is not None, "wrong!!!"

        loss, self.lossmap, self.wlossmap = self.loss_func(input, target, weight_map)
        return loss


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if "pred_statistic" in global_dict:
            pstat = global_dict["pred_statistic"]
            if pstat.result is not None:
                return self._forward_(input, target, pstat)
        loss, self.lossmap, self.wlossmap = self.loss_func(input, target, None)
        return loss



class AnalyseLossMap_Hard(AnalyseItem):
    name_dict = dict(
        # org_lossmap='org_lossmap.png',
        # weighted_lossmap="weighted_lossmap.png",
        trunced_lable_map = "trunced_lable_map.png",
        trunc_mask_alone_map="trunc_mask_alone_map.png")

    def call(self, datas: dict, dist_datas: dict):
        if not self.layer.data_ready:
            return
        if not self.layer.hard_trunc:
            return

        # dist_datas["org_lossmap"] = {"data": self.layer.lossmap, "type": "gray_to_jet", "tensor": True}

        dist_datas["trunced_lable_map"] = {"data": datas["lable"].cpu().numpy(), "type": "lable",
                                          "mask_key": "trunc_mask", "mask_color": [255, 0, 0]}

        # dist_datas["weighted_lossmap"] = {"data": self.layer.wlossmap, "type": "gray_to_jet",
        #                                   "tensor": True, "mask_key": "trunc_mask", "mask_color": [0, 0, 0]}

        dist_datas["trunc_mask_alone_map"] = {"data": self.layer.trunc_mask.float() * 200, "tensor": True,
                                              "type": "simple_save"}

        dist_datas["trunc_mask"] = self.layer.trunc_mask

    def show(self, datas, disted_datas: dict):
        if not self.layer.data_ready:
            return
        if not self.layer.hard_trunc:
            return
        super().show(datas, disted_datas)



class HardBalancedLoss(nn.Module):
    def __init__(self, smooth_factor=0.05, ignore_index=None, hard_trunc=True):
        """
            各项参数在 pred_stat 类中设置, 此类不管具体方法
            确保 pred_stat中: weight_style='hard_trunc'
        """
        super(HardBalancedLoss, self).__init__()

        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.hard_trunc = hard_trunc

        self.loss_func = partial(
            weighted_label_smoothed_nll_loss,
            epsilon=smooth_factor,
            ignore_index=ignore_index,
            dim=1
        )

        alm = AnalyseLossMap_Hard()
        alm.layer = self
        add_analyse_item(alm)

        self.data_ready = False


    def _forward_(self, input, target, pstat):
        assert isinstance(pstat, PredStatistic)
        
        ret = pstat.pred_stat(input, target)
        if isinstance(ret, tuple):
            self.hard_trunc = True
            weight_map, trunc_mask = ret
            self.trunc_mask = trunc_mask
        else:
            self.hard_trunc = False
            weight_map = ret

        assert weight_map is not None, "wrong!!!"

        loss, self.lossmap, self.wlossmap = self.loss_func(input, target, weight_map)
        self.data_ready = True
        return loss


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data_ready = False

        if "pred_statistic" in global_dict:
            pstat = global_dict["pred_statistic"]
            if pstat.result is not None:
                return self._forward_(input, target, pstat)
        loss, self.lossmap, self.wlossmap = self.loss_func(input, target, None)
        return loss




class HardTruncLoss(nn.Module):
    def __init__(self, smooth_factor=0.05, ignore_index=None):
        super(HardTruncLoss, self).__init__()

        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index

        self.loss_func = partial(
            weighted_label_smoothed_nll_loss,
            epsilon=smooth_factor,
            ignore_index=ignore_index,
            dim=1
        )

        alm = AnalyseLossMap()
        alm.layer = self
        add_analyse_item(alm)


    def _forward_(self, input, target, pstat):
        assert isinstance(pstat, PredStatistic)

        weight_map = pstat.pred_stat(input, target)

        assert weight_map is not None, "wrong!!!"

        loss, self.lossmap, self.wlossmap = self.loss_func(input, target, weight_map)
        return loss


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        pred = input.detach()
        B, C, H, W = pred.shape
        predict = F.softmax(pred, dim=1)

        return loss
