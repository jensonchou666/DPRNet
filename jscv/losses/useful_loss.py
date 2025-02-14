import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import cv2

from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss


import jscv.utils.analyser as analyser
from jscv.utils.utils import edge_detect_target
from jscv.utils.overall import global_dict


def islist(x):
    return isinstance(x, list) or isinstance(x, tuple)

def smooth(s, e, t, maxt):
    if t >= maxt:
        return e
    else:
        return s + (t / maxt) * (e - s)

def smooth_in(s, e, t, maxt, mint=0):
    if t <= mint:
        return s
    else:
        return smooth(s, e, t - mint, maxt - mint)


def smooth_weight(w, t):
    if islist(w):
        s, e, maxt = w
        return smooth(s, e, t, maxt)
    else:
        return w


class JointMultiLoss(nn.Module):
    
    def __init__(self, layers: list, weights: list, assert_main_loss=True):
        super(JointMultiLoss, self).__init__()
        self.smooth = False
        for w in weights:
            if islist(w) or isinstance(w, dict):
                self.smooth = True
                break
        self.layers = layers
        self.weights = weights
        self.assert_main_loss = assert_main_loss

    def forward(self, result: dict, target: Tensor):
        weights = self.weights.copy()
        """ smooth """
        if self.smooth:
            trainer = global_dict["trainer"]
            t = trainer.epoch_idx
            for i in range(len(weights)):
                w = smooth_weight(weights[i], t)
                if isinstance(w, dict):
                    if "delay" in w:
                        delay = w["delay"]
                        if t < delay:
                            weights[i] = w["init"]
                        else:
                            weights[i] = smooth_weight(w["then"], t - delay)
                else:
                    weights[i] = w
        loss_dict = {}

        for layer, w in zip(self.layers, weights):
            # print(">>>>>>>", type(layer), w, flush=True)
            if w != 0:
                ld = layer(result, target)
                for k, v in ld.items():
                    loss_dict[k] = v * w
        # if self.assert_main_loss:
        #     if "main_loss" not in loss_dict and "loss_main" not in loss_dict and "loss" not in loss_dict:
        #         loss_dict['main_loss'] = 0.0
        return loss_dict


class LossLayer(nn.Module):
    def __init__(self, layer, loss_name, input_key="pred", resize=True):
        super(LossLayer, self).__init__()
        self.layer = layer
        self.loss_name = loss_name
        self.input_key = input_key
        self.resize = resize

    def forward(self, result: dict, target: Tensor):
        key = self.input_key
        hw = target.shape[-2:]
        if self.resize and result[key].shape[-2:] != hw:
            result[key] = F.interpolate(result[key], hw, mode='bilinear')
        return {self.loss_name: self.layer(result[key], target)}



class MainLoss(nn.Module):
    def __init__(self, layer, input_pred=True):
        super(MainLoss, self).__init__()
        self.layer = layer
        self.input_pred = input_pred
    
    def forward(self, result: dict, target: Tensor):
        hw = target.shape[-2:]

        if result["pred"].shape[-2:] != hw:
            result["pred"] = F.interpolate(result["pred"], hw, mode='bilinear')

        if self.input_pred:
            return {"main_loss": self.layer(result["pred"], target)}
        else:
            res = self.layer(result, target)
            if isinstance(res, dict):
                if "main_loss" not in res:
                    res["main_loss"] = sum(res.values())
                return res
            else:
                return {"main_loss": res}


class CoarsePredLoss(nn.Module):
    def __init__(self, layer, resize_type="pred"):
        super(CoarsePredLoss, self).__init__()
        self.layer = layer
        self.resize_type = resize_type
    
    def forward(self, result: dict, target: Tensor):
        coarse_pred = result["coarse_pred"]
        tg = target
        hw = target.shape[-2:]
        if coarse_pred.shape[-2:] != hw:
            if self.resize_type == "pred":
                coarse_pred = F.interpolate(coarse_pred, hw, mode='bilinear')
            elif self.resize_type == "target":
                tg = F.interpolate(target, coarse_pred.shape[-2:], mode='bilinear')
        result["coarse_pred"] = coarse_pred
        return {"coarse_loss": self.layer(coarse_pred, tg)}


class PredWrongLoss(nn.Module):
    def __init__(self, ignore_index=None, pred_is_coarse=True, dim=1):
        super(PredWrongLoss, self).__init__()
        self.ignore_index = ignore_index
        self.dim = dim
        self.pred_is_coarse = pred_is_coarse
    
    def forward(self, result: dict, target: Tensor):
        dim = self.dim
        if self.pred_is_coarse:
            pred = result["coarse_pred"]
        else:
            pred = result["pred"]
        pred_wrong = result["pred_wrong"]
        pred_wrong = torch.sigmoid(pred_wrong)
        hw = target.shape[-2:]
        if pred_wrong.shape[-2:] != hw:
            pred_wrong = F.interpolate(pred_wrong, hw, mode='bilinear').squeeze(1)

        mask = nn.Softmax(dim=dim)(pred).argmax(dim=dim)
        wrong = mask != target
        if self.ignore_index is not None:
            wrong[target == self.ignore_index] = False
        wrong = wrong.type_as(pred_wrong)
        predwrong_loss = F.binary_cross_entropy(pred_wrong.squeeze(), wrong.squeeze())
        return {"predwrong_loss": predwrong_loss}






class MultiPredLoss(nn.Module):
    def __init__(self,
                 main_loss_layer,
                 coarse_loss_layer,
                 coarse_loss_weights: list,
                 main_loss_weight=1,
                 coarse_resize_type="pred",  # pred or target
                 coarse_loss_names=None,
                 ):
        super(MultiPredLoss, self).__init__()

        smooth = False
        if islist(main_loss_weight):
            smooth = True
        else:
            for w in coarse_loss_weights:
                if islist(w):
                    smooth = True
                    break

        self.b_smooth_weight = smooth

        self.main_loss_layer = main_loss_layer
        self.coarse_loss_weights = coarse_loss_weights
        self.main_loss_weight = main_loss_weight
        self.coarse_resize_type = coarse_resize_type

        len_extra = len(coarse_loss_weights)

        if isinstance(coarse_loss_layer, nn.Module):
            coarse_loss_layer = [coarse_loss_layer] * len_extra
        self.coarse_loss_layer = coarse_loss_layer

        if coarse_loss_names is None:
            if len_extra == 1:
                coarse_loss_names = ["coarse_pred_loss"]
            else:
                coarse_loss_names = [f"coarse_pred_loss_{i}" for i in range(len_extra)]
        self.coarse_loss_names = coarse_loss_names


    def forward(self, result: dict, target: Tensor):

        """ smooth """
        main_loss_weight = self.main_loss_weight
        coarse_loss_weights = self.coarse_loss_weights.copy()
        if self.b_smooth_weight:
            trainer = global_dict["trainer"]
            t = trainer.epoch_idx
            if islist(main_loss_weight):
                s, e, maxt = main_loss_weight
                main_loss_weight = smooth(s, e, t, maxt)
            for i in range(len(coarse_loss_weights)):
                w = coarse_loss_weights[i]
                if islist(w):
                    s, e, maxt = w
                    coarse_loss_weights[i] = smooth(s, e, t, maxt)


        assert isinstance(result, dict)
        pred = result["pred"]
        coarse_pred_list = result["coarse_pred_list"]
        resize_type = self.coarse_resize_type

        hw = target.shape[-2:]
        if pred.shape[-2:] != hw:
            pred = F.interpolate(pred, hw, mode='bilinear')


        main_loss = self.main_loss_layer(pred, target) * main_loss_weight

        extra_loss = {}
        for coarse_pred, coarse_layer, coarse_weight, name in zip(
            coarse_pred_list, self.coarse_loss_layer,
            coarse_loss_weights, self.coarse_loss_names
        ):
            tg = target
            if coarse_pred[-2:] != hw:
                if resize_type == "pred":
                    coarse_pred = F.interpolate(coarse_pred, hw, mode='bilinear')
                elif resize_type == "target":
                    tg = F.interpolate(target, coarse_pred[-2:], mode='bilinear')
            loss2 = coarse_layer(coarse_pred, tg) * coarse_weight
            extra_loss[name] = loss2
        
        losses = {"main_loss": main_loss}
        losses.update(extra_loss)

        return losses


# class MultiPredLoss(nn.Module):
#     def __init__(self,
#                  main_loss_layer,
#                  aux_loss_layer,
#                  aux_loss_weights: list,
#                  main_loss_weight=1,
#                  aux_resize_type=None,  # pred or target or None
#                  aux_loss_names=None,
#                  ):
#         self.main_loss_layer = main_loss_layer
#         self.main_loss_weight = main_loss_weight
#         self.aux_loss_weights = aux_loss_weights

#         if isinstance(aux_loss_layer, nn.Module):
#             aux_loss_layer = [aux_loss_layer] * len(aux_loss_weights)
#         self.aux_loss_layer = aux_loss_layer

    
#     def forward(self, result: dict, target: Tensor):
#         assert isinstance(result, dict)
#         pred = result["pred"]
#         coarse_pred_list = result["coarse_pred_list"]

#         hw = target.shape[-2:]
#         if pred.shape[-2:] != hw:
#             pred = F.interpolate(pred, hw, mode='bilinear')

#         main_loss = self.main_loss_layer(pred, target) * self.loss_weight_main

#         extra_loss = {}
#         for coarse_pred,











class WrongStrengthenLoss(nn.Module):

    def __init__(self,
                 smooth_factor=0.05,
                 ignore_index=None,
                 dim=1, reduction='mean',
                 loss_weight=1,
                 smooth_epoch=4,
                 ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth_factor = smooth_factor
        self.dim = dim
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_epoch = smooth_epoch

        self.main_loss = JointLoss(SoftCrossEntropyLoss(
            smooth_factor=smooth_factor, ignore_index=ignore_index, dim=dim, reduction=reduction),
            DiceLoss(smooth=smooth_factor, ignore_index=ignore_index), 1.0, 1.0)


        self.i1 = 0
        

    def forward(self, input, target: torch.Tensor):

        dim = self.dim
        smooth_epoch = self.smooth_epoch
        ignore_index = self.ignore_index
        loss_weight = self.loss_weight

        wrong_map = None

        if isinstance(input, dict):
            pred = input['prediction']
            wrong_map = input.get('wrong_map', None)
        else:
            pred = input[0]

        if wrong_map is None:
            mask = F.log_softmax(pred, dim=dim).argmax(dim=dim)
            wrong_map = mask != target
            # if ignore_index is not None:
            #     wrong_map[target == ignore_index] = False
        
        wrong_map = wrong_map.bool()

        if ignore_index is None:
            ignore_index = -1
        #TODO target.copy()
        target[wrong_map is False] = ignore_index

        epoch = global_dict['trainer'].epoch_idx
        if epoch < smooth_epoch:
            loss_weight *= float(epoch) / smooth_epoch

        return self.main_loss(pred, target) * loss_weight



class PredEdgeWrongLoss(nn.Module):

    def __init__(self,
                 smooth_factor=0.05,
                 ignore_index=None,
                 dim=1, reduction='mean',
                 extra_loss=None,
                 extra_loss_weight=1,
                 prededgeloss_weight=2,
                 predwrongloss_weight=2,
                 smooth_epoch=4,

                 # set None,  auto_detect
                 edge_index=None,
                 ):
        super().__init__()
        self.prededgeloss_weight = prededgeloss_weight
        self.predwrongloss_weight = predwrongloss_weight

        self.smooth_epoch = smooth_epoch
        self.dim = dim
        self.ignore_index = ignore_index

        self.extra_loss = extra_loss
        self.extra_loss_weight = extra_loss_weight
        
        self.main_loss = JointLoss(SoftCrossEntropyLoss(
            smooth_factor=smooth_factor, ignore_index=ignore_index, dim=dim, reduction=reduction),
            DiceLoss(smooth=smooth_factor, ignore_index=ignore_index), 1.0, 1.0)
        
        self.edge_index = edge_index

    def forward(self, input, target: torch.Tensor):

        dim = self.dim
        smooth_epoch = self.smooth_epoch
        prededgeloss_weight = self.prededgeloss_weight
        predwrongloss_weight = self.predwrongloss_weight
        ignore_index = self.ignore_index
        edge_index = self.edge_index

        if isinstance(input, dict):
            pred = input['prediction']
            pred_edgeorg = input['pred_edge']
            pred_wrongorg = input['pred_wrongrange']
        else:
            pred, pred_edgeorg, pred_wrongorg = input
        
        pred_edge = torch.sigmoid(pred_edgeorg)
        pred_wrong = torch.sigmoid(pred_wrongorg)

        # print('@@', pred.shape, target.shape)
        main_loss = self.main_loss(pred, target)

        # create wrong_map 是否detach？
        mask = nn.Softmax(dim=dim)(pred).argmax(dim=dim)  # .detach()  # .cpu().numpy()
        # target = target.cpu().numpy()

        wrong = mask != target
        if ignore_index is not None:
            wrong[target == ignore_index] = False
        wrong = wrong.type_as(pred_wrong)
        predwrong_loss = F.binary_cross_entropy(pred_wrong, wrong)

        # edge_lable = torch.sob

        
        if edge_index is None:
            edge = edge_detect_target(target.unsqueeze(1).float()).squeeze(1)
        else:
            edge = torch.zeros(pred_edge.shape, dtype=pred_edge.dtype, device=pred_edge.device)
            edge[target == edge_index] = 1

        #TODO focal动态权重
        bce_weight = torch.full_like(edge, 1)
        bce_weight[edge == 1] = 10
        prededge_loss = F.binary_cross_entropy(pred_edge, edge, bce_weight)

        epoch = global_dict['trainer'].epoch_idx
        if epoch < smooth_epoch:
            # prededgeloss_weight *= float(epoch) / smooth_epoch
            predwrongloss_weight *= float(epoch) / smooth_epoch

        if self.extra_loss is not None:
            input_2 = dict(prediction=pred, wrong_map=wrong, pred_edge=pred_edgeorg, pred_wrong=pred_wrongorg)
            extra_loss = self.extra_loss(input_2, target) * self.extra_loss_weight
            main_loss = main_loss + extra_loss

        losses = {'main_loss': main_loss}
        if prededgeloss_weight != 0:
            losses['prededge_loss'] = prededge_loss * prededgeloss_weight
        if predwrongloss_weight != 0:
            losses['predwrong_loss'] = predwrong_loss * predwrongloss_weight
        
        return losses






class PredWrongLoss2(nn.Module):

    def __init__(self,
                 main_loss_layer,
                 #smooth_factor=0.05,
                 ignore_index=None,
                 dim=1,
                 #reduction='mean',

                 predwrongloss_weight=0.5,
                 smooth_epoch=4,
                 loss_layer_input_pred=True
                 ):
        super().__init__()
        self.predwrongloss_weight = predwrongloss_weight
        self.smooth_epoch = smooth_epoch
        self.dim = dim
        self.ignore_index = ignore_index


        self.main_loss = main_loss_layer
        self.input_pred = loss_layer_input_pred
        # self.main_loss = JointLoss(SoftCrossEntropyLoss(
        #     smooth_factor=smooth_factor, ignore_index=ignore_index, dim=dim, reduction=reduction),
        #     DiceLoss(smooth=smooth_factor, ignore_index=ignore_index), 1.0, 1.0)


    def forward(self, input, target: torch.Tensor):

        dim = self.dim
        smooth_epoch = self.smooth_epoch
        predwrongloss_weight = self.predwrongloss_weight
        ignore_index = self.ignore_index

        if isinstance(input, dict):
            if "pred" in input:
                pred = input['pred']
            else:
                pred = input['prediction']
            pred_wrong = input['pred_wrong']
        else:
            pred, pred_wrong = input
        
        pred_wrong = torch.sigmoid(pred_wrong)
        if pred_wrong.shape[-2:] != target.shape[-2:]:
            pred_wrong = F.interpolate(pred_wrong, target.shape[-2:], mode="bilinear")

        if self.input_pred:
            lossL_input = pred
        else:
            lossL_input = input

        main_loss = self.main_loss(lossL_input, target)

        # create wrong_map 是否detach？
        mask = nn.Softmax(dim=dim)(pred).argmax(dim=dim)  # .detach()  # .cpu().numpy()
        # target = target.cpu().numpy()

        wrong = mask != target

        wrong[target == ignore_index] = False

        #wrong_map = wrong.cpu().numpy().astype(np.uint8)

        wrong = wrong.type_as(pred_wrong)

        predwrong_loss = F.binary_cross_entropy(pred_wrong, wrong)

        epoch = global_dict['trainer'].epoch_idx
        if epoch < smooth_epoch:
            predwrongloss_weight *= float(epoch) / smooth_epoch

        if predwrongloss_weight == 0:
            return main_loss
        else:
            return {'main_loss': main_loss, 'predwrong_loss': predwrong_loss * predwrongloss_weight}



class PredLossLoss(nn.Module):

    def __init__(self,
                 smooth_factor=0.05,
                 ignore_index=None,
                 dim=1, reduction='mean',
                 extra_loss=None,
                 sce_loss_weight=1,
                 predlossloss_weight=0.5,
                 extra_loss_weight=1,
                 smooth_epoch=4,
                 ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth_factor = smooth_factor
        self.dim = dim
        self.reduction = reduction
        self.extra_loss = extra_loss
        self.sce_loss_weight = sce_loss_weight
        self.predlossloss_weight = predlossloss_weight
        self.extra_loss_weight = extra_loss_weight
        self.smooth_epoch = smooth_epoch


        self.i1 = 0
        

    def forward(self, input, target):
        ignore_index = self.ignore_index
        epsilon = self.smooth_factor
        dim = self.dim
        reduction = self.reduction
        smooth_epoch = self.smooth_epoch
        predlossloss_weight = self.predlossloss_weight


        if isinstance(input, dict):
            pred = input['prediction']
            pred_loss = input['pred_loss']
        else:
            pred, pred_loss = input
        logits = pred
        pred = F.log_softmax(pred, dim=self.dim)

        '''#? Temp
        mask = pred.argmax(dim=dim)  # .cpu().numpy()
        wrong = (mask != target)
        # ?
        wrong[target == ignore_index] = False
        wrong_maps = wrong.cpu().numpy().astype(np.uint8)
        wrong = wrong.type_as(pred_loss)

        import cv2
        cfg = global_dict['cfg']
        for msk, tg, wrong_map in zip(mask.cpu().numpy(), target.cpu().numpy(), wrong_maps):
            self.i1 += 1
            rgb_mask = analyser.label2rgb(msk, cfg.num_classes)
            rgb_lable = analyser.label2rgb(tg, cfg.num_classes)
            cv2.imwrite(f'wrong_map_no_boundary/{self.i1}_wrong_map.png', wrong_map * 255)
            cv2.imwrite(f'wrong_map_no_boundary/{self.i1}_prediction.png', rgb_mask)
            cv2.imwrite(f'wrong_map_no_boundary/{self.i1}_groundtruth.png', rgb_lable)
        '''

        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(dim)

        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            target = target.masked_fill(pad_mask, 0)
            nll_loss = -pred.gather(dim=dim, index=target)
            smooth_loss = -pred.sum(dim=dim, keepdim=True)
            nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
            smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
        else:
            nll_loss = -pred.gather(dim=dim, index=target)
            smooth_loss = -pred.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

        eps_i = epsilon / pred.size(dim)

        loss_map = nll_loss * (1.0 - epsilon) + smooth_loss * eps_i

        epoch = global_dict['trainer'].epoch_idx
        if epoch < smooth_epoch:
            predlossloss_weight *= float(epoch) / smooth_epoch

        predloss_loss = F.mse_loss(pred_loss, loss_map) * predlossloss_weight

        if reduction == "sum":
            sce_loss = loss_map.sum()
        if reduction == "mean":
            sce_loss = loss_map.mean()
        main_loss = sce_loss = sce_loss * self.sce_loss_weight

        if self.extra_loss is not None:
            extra_loss = self.extra_loss(logits, target) * self.extra_loss_weight
            main_loss = main_loss + extra_loss


        if predlossloss_weight == 0:
            return main_loss
        else:
            return {'main_loss': main_loss, 'predloss_loss': predloss_loss}





class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        # dice_loss = 1 - ((2. * (boundary_pre * boundary_targets).sum(1) + 1.0) /
        #                  (boundary_pre.sum(1) + boundary_targets.sum(1) + 1.0))
        # dice_loss = dice_loss.mean()
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor
        return loss




class SCE_DIce_Loss(nn.Module):
    def __init__(self, ignore_index=255, use_dice=True):
        super().__init__()
        if use_dice:
            self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        else:
            self.main_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)


    def forward(self, logits, labels):
        # print("@@@0", logits.shape, labels.shape)
        return self.main_loss(logits, labels)



class SceDiceLoss(nn.Module):
    
    def __init__(self, ignore_index=255, use_dice=True):
        super().__init__()
        if use_dice:
            self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        else:
            self.main_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        self.label_numel = labels.numel()
        # print("@@@0", logits.shape, labels.shape)
        return self.main_loss(logits, labels)








''' ---------------------------------------------------------------- '''

class EdgeStrengthSceLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, 
                 ignore_index = -100, dim=1, edge_strength_weight: float = 5.,
                 edge_pool_kernel=7, to_cuda=True):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        self.edge_strength_weight = edge_strength_weight
        def get_edge_conv2d(channel=1):
            conv_op = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
            sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
            sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
            sobel_kernel = np.repeat(sobel_kernel, channel, axis=0)
            sobel_kernel = np.repeat(sobel_kernel, channel, axis=1)
            conv_op.weight.data = torch.from_numpy(sobel_kernel)
            return conv_op
        self.edge_conv2d = get_edge_conv2d()
        if to_cuda:
            self.edge_conv2d = self.edge_conv2d.cuda()
        self.edge_pool = nn.MaxPool2d(kernel_size=edge_pool_kernel, stride=1, padding=(edge_pool_kernel)//2)

    def forward(self, input: Tensor, target: Tensor):
        log_prob = F.log_softmax(input, dim=self.dim)
        return self.label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

    def detect_edge(self, mask):
        edge_range = self.edge_conv2d(mask.float()) > 0.1
        edge_range = self.edge_pool(edge_range.float()).bool()
        return edge_range


    def label_smoothed_nll_loss(
        self, lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
    ):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(dim)
        
        if self.edge_strength_weight > 1:
            edge_mask = self.detect_edge(target)
        
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            target = target.masked_fill(pad_mask, 0)

            nll_loss = -lprobs.gather(dim=dim, index=target)
            smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

            nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
            smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
        else:
            nll_loss = -lprobs.gather(dim=dim, index=target)
            smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

            nll_loss = nll_loss.squeeze(dim)
            smooth_loss = smooth_loss.squeeze(dim)

        # print("@", nll_loss.shape, edge_mask.shape, self.edge_strength_weight)
        if self.edge_strength_weight > 1:
            nll_loss = nll_loss * edge_mask * self.edge_strength_weight + nll_loss * ~edge_mask
            smooth_loss = smooth_loss * edge_mask * self.edge_strength_weight + smooth_loss * ~edge_mask

        if reduction == "sum":
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        if reduction == "mean":
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()

        eps_i = epsilon / lprobs.size(dim)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


class SceDiceEdgeStrengthLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_strength_weight=4, edge_pool_kernel=7,
                 to_cuda=True, global_loss_layer=None, weight_sce=2, weight_dice=0.25):
        super().__init__()
        self.sce = EdgeStrengthSceLoss(reduction='sum', smooth_factor=0.05, ignore_index=ignore_index,
                                       edge_strength_weight=edge_strength_weight,
                                       edge_pool_kernel=edge_pool_kernel, to_cuda=to_cuda)
        self.dice = DiceLoss(smooth=0.05, ignore_index=ignore_index)
        self.global_loss_layer = [global_loss_layer]
        self.weight_sce = weight_sce
        self.weight_dice = weight_dice
        self.count_sce_loss = 0
        self.count_dice_loss = 0
        self.counter = 0

    def forward(self, logits, labels):
        global_loss_layer = self.global_loss_layer[0]
        assert isinstance(global_loss_layer, SceDiceLoss)
        reduction = global_loss_layer.label_numel
        sce_loss = self.sce(logits, labels) * self.weight_sce / reduction
        dice_loss = self.dice(logits, labels) * self.weight_dice
        self.count_sce_loss += sce_loss
        self.count_dice_loss += dice_loss
        self.counter += 1
        if self.counter > 400:
            l1 = self.count_sce_loss / self.counter
            l2 = self.count_dice_loss / self.counter
            print(f"Local sce_loss: {l1:.6f}, dice_loss: {l2:.6f}", flush=True)
            self.count_sce_loss, self.count_dice_loss, self.counter = 0, 0, 0

        return sce_loss + dice_loss
