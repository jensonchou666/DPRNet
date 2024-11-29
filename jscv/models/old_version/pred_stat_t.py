from typing import Any
from torch.nn import functional as F
import torch
import os
import math
from matplotlib import pyplot
import numpy as np
from collections import deque


from jscv.utils.utils import *     # do_once, on_train, redirect, color_tulpe_to_string
from jscv.utils.utils import torch


just_demo = False
#use_ema = False


"""
    不区分类别

"""


class StatisticBlock:
    def __init__(self, count_len, class_sum=False, num_classes=1, device="cpu"):
        if device == "none_block":
            return
        self.count_len = count_len
        self.do_class_sum = class_sum
        self.device = device

        self.counts = torch.zeros(count_len).to(device)
        if class_sum:
            self.class_sum = torch.zeros(num_classes).to(device)
        self.sample_nums = 0

    def clone(self):
        blk = StatisticBlock(0, 0, "none_block")
        blk.counts = self.counts.clone()
        blk.sample_nums = self.sample_nums
        blk.count_len = self.count_len
        if self.do_class_sum:
            blk.class_sum = self.class_sum.clone()
        return blk

    def update(self, other):
        self.counts += other.counts
        self.sample_nums += other.sample_nums
        if self.do_class_sum:
            self.class_sum += other.class_sum
        return self

    def delete(self, other):
        self.counts -= other.counts
        self.sample_nums -= other.sample_nums
        if self.do_class_sum:
            self.class_sum -= other.class_sum
        return self






def fig_reset():
    pyplot.clf()
    # pyplot.cla()
    return []



def sample_capacity(cfg):
    return int(len(cfg.train_dataset) // cfg.train_batch_size)


"""
    weight_style:   1.  'smooth'
                    2.  'hard_trunc'
                    3.  '~~'
    class_balance_rate:
                    建议在 [0.5 ~ 1]
"""

result_kargs_hard_trunc = dict(
    weight_style='hard_trunc',
    correct_weight=0.6,
    wrong_weight=4,
    trunc_after=0.8,
    class_balance_rate=0.5,

    muti=1,
    return_trunc_mask=True,  # need True
    wrong_only=False,
    stat_trunc=1,
)

result_kargs_soft = dict(
    weight_style='smooth',
    correct_weight=0.8,
    wrong_weight=2,
    #trunc_after=0.8,
    #class_balance_rate=0.5,

    muti=1,
    return_trunc_mask=False,
    wrong_only=False,
    stat_trunc=1,
)



class PredStatistic:

    def from_cfg(cfg):
        if not hasattr(cfg, "pred_statistic_args"):
            cfg.pred_statistic_args = dict()
        cargs = cfg.pred_statistic_args

        # if "sample_capacity" not in cargs and on_train():
        #     cargs["sample_capacity"] = int(len(cfg.train_dataset) // cfg.train_batch_size)

        return PredStatistic(**cargs)


    def __init__(
        self,
        cfg,
        count_len=100,              # count_len越大， 统计精度越高
        min_value=0.4,
        #? sample_capacity 会被调整到能被 queue_blocks 整除
        sample_capacity=100,       # 最近样本最大总量
        # 每当最后一个队列块儿满了，popleft掉第一个块， push一个新的empty_block
        queue_blocks=4,            # 样本队列分块， 每一块有sample_capacity//cache_blocks个样本

        do_class_sum=True,
        result_kargs=result_kargs_hard_trunc,

        device="cpu",
    ):
        self.cfg = cfg
        self.device = device
        self.count_len = count_len
        self.min_value = min_value
        self.counter_new_block = -1
        self.do_class_sum = do_class_sum

        self.result_kargs = result_kargs.copy()

        queue_blocks = max(queue_blocks, 1)
        self.block_size = block_size = sample_capacity // queue_blocks
        sample_capacity = block_size * queue_blocks
        self.sample_capacity = sample_capacity
        self.block_nums = queue_blocks
        self.blocks_queue = deque()

        self.un_init = True
        self.queue_full = False
        self.show_aviliable = False
        self.result = None


    def init(self):
        nC = self.num_classes = self.cfg.num_classes
        self.classes = self.cfg.get("classes")
        self.blocks_queue.append(self.create_block())
        self.total_block = self.create_block()

        self.batch_stat_finish = False
        self.frozen = False

        clen = self.count_len
        minv = self.min_value
        self.xpoints = np.linspace(minv, 1, clen, dtype=float)
        self.edges = np.linspace(minv, 1, clen + 1, dtype=float)
        #self.edges = torch.arange(clen + 1).float() / (1 / (1 - minv)) * clen) + minv



    def create_block(self):
        return StatisticBlock(self.count_len, self.do_class_sum, self.num_classes, self.device)


    def pred_stat(self, pred_logits, target, return_weight=True, result_kargs=None):

        if result_kargs is None:
            result_kargs = self.result_kargs
        weight_style = result_kargs["weight_style"]
        wrong_only = result_kargs.get("wrong_only", False)
        stat_trunc = result_kargs.get("stat_trunc", 1)

        #!
        ignore_index = self.cfg.ignore_index

        pred = pred_logits.detach()
        # pred = torch.softmax(pred_logits.detach(), dim=1)
        # if len(pred.shape) == 3:
        #     pred = pred.unsqueeze(1)
        B, C, H, W = pred.shape
        # -> B, H, W, C -> BHW, C
        pred = pred.permute(0, 2, 3, 1).reshape(-1, C)
        
        predict = F.softmax(pred, dim=1)
        # predict = F.softmax(predict, dim=1)

        target = target.to(pred.device).clone().reshape(-1)
        ignore_mask = (target == ignore_index)
        valid_mask = ~ignore_mask
        #target = target[valid_mask]
        #target_org = target.clone()
        target[ignore_mask] = 0
        target_one_hot = F.one_hot(target, num_classes=C)
        assert target_one_hot.shape[1] == C

        #predict = predict[valid_mask]
        last_block = self.blocks_queue[-1]
        counts = last_block.counts
        clen = self.count_len
        edges = self.edges


        #def stat_top1(wrong_only=False, cls_ref="pred", stat_trunc=1):
        """
            cls_ref:   "pred"  or  "target"
        """
        valid = valid_mask
        # top1

        top1_index = torch.argmax(predict, dim=1, keepdim=True)
        top1_class = top1_index.squeeze()
        top1_value = torch.gather(predict, 1, top1_index).squeeze()
        wrong_mask = top1_class != target
        if wrong_only:
            valid &= wrong_mask

        stat = top1_value   # [BHW]
        # print(top1_value)
        rw = return_weight and (self.result is not None)
        # print(rw, return_weight, self.result is not None)
        extra_return = []
        if not rw:
            weight_style = "none"

        if weight_style == "smooth":
            weight_map = torch.zeros(B * H * W)
            wights = self.result['wights']
            inds = valid & (stat < self.min_value)
            w = torch.mean(wights[:4])
            #print(B * H * W, (B, H, W), inds.shape, weight_map.shape, )
            weight_map[inds] = w
        elif weight_style == "hard_trunc":
            weight_map = torch.ones(B * H * W)
            trunc = self.result['trunc']
            if trunc is None:
                inds = torch.zeros([B, H, W])
                print("!!! trunc is None")
            else:
                # 截断策略 >>>
                trunc_left_move = result_kargs.get("trunc_left_move", 0.0)
                inds = valid_mask & wrong_mask & (stat > (trunc - trunc_left_move))
                if result_kargs.get("return_trunc_mask", False):
                    extra_return.append(inds.reshape(B, H, W))
                weight_map[inds] = 0.
        else:
            weight_map = torch.ones(B * H * W)


        for i in range(clen):
            inds = (stat > edges[i]) & (stat < edges[i + 1]) & valid
            if stat_trunc < 1:
                inds = inds & (stat < stat_trunc)
            counts[i] += inds.sum().item()
            # print(i, inds.sum().item(), edges[i], edges[i + 1])

            if weight_style == "smooth":
                weight_map[inds] = wights[i]

        self.batch_stat_finish = True

        if rw:
            weight_map = weight_map.reshape(B, H, W)
            if len(extra_return) > 0:
                return (weight_map, *extra_return)
            else:
                return weight_map



    def complete(self):
        rs = self.result = {}
        result_kargs = self.result_kargs
        
        weight_style = result_kargs["weight_style"]


        counts = self.total_block.counts
        maxidx = torch.argmax(counts)
        maxi = counts[maxidx].item()
        sum0 = torch.sum(counts, dim=0, keepdim=False)
        cl = self.count_len

        counts = counts.clone() #* (cl / sum0)
        
        rs["counts"] = counts


        if weight_style == "hard_trunc":
            trunc_rate = result_kargs.get("trunc_after", 1)
            assert trunc_rate <= 1
            if trunc_rate == 1:
                trunc_i = maxidx
            else:
                i = maxidx
                while i > 0:
                    i -= 1
                    if counts[i] <= maxi * trunc_rate:
                        break
                if i <= 0:
                    trunc_i = None
                else:
                    trunc_i = i
            rs["trunc_i"] = trunc_i
            if trunc_i is None:
                rs["trunc"] = None
            else:
                rs["trunc"] = self.xpoints[trunc_i]

        elif weight_style == "smooth":
            muti = result_kargs.get("muti", 1)

            weights = counts.clone()

            #weights[maxidx:] = maxi
            # torch.pow(weights, 0.8, out=weights)

            if muti != 1:
                weights /= muti
            weights += 1
            weights = torch.reciprocal_(weights)

            #New
            trunci = 4
            v_i = weights[maxidx].item()
            k = - v_i / max((cl - trunci - maxidx), 1)
            for i in range(maxidx + 1, cl):
                weights[i] = max((k * (i - maxidx) + v_i), 0)

            # # #? 平滑：
            # b, a = signal.butter(3, 0.05)
            # x = signal.filtfilt(b, a, count_c.cpu().numpy())
            # weights[c] = torch.tensor(x, device=weights.device)

            weights[weights < 0] = 0
            rs["weights"] = weights

        class_balance_rate = result_kargs.get("class_balance_rate", 0)
        # print("self.do_class_sum", self.do_class_sum)
        if self.do_class_sum:
            class_sum = self.total_block.class_sum
            rs["class_sum"] = class_sum
            if class_balance_rate != 0:
                t = class_balance_rate
                x = class_sum.clone()
                x[x == 0] = 1

                x /= torch.sum(x)     # to [0, 1]
                x2 = torch.exp(x / t)
                sum1 = torch.sum(x2)
                y = sum1 / x2
                y *= y.shape[0] / sum(y)
                # y[y > 4] = 4
                rs["class_balance_weights"] = y



    def step(self, pred, target):
        if not on_train():
            return

        if self.un_init:
            self.init()
            self.un_init = False

        blocks = self.blocks_queue


        last_block = blocks[-1]
        assert isinstance(last_block, StatisticBlock)

        #t0 = time.time()
        
        if not self.batch_stat_finish:
            self.pred_stat(pred, target, return_weight=False)

        self.batch_stat_finish = False

        #spend = time.time() - t0


        total_block = self.total_block
        last_block.sample_nums += 1
        # print(last_block.sample_nums, self.block_size)
        if last_block.sample_nums == self.block_size and not self.frozen:
            self.show_aviliable = True
            total_block.update(last_block)
            blocks.append(self.create_block())
            if len(blocks) > self.block_nums:
                if not just_demo:
                    self.complete()
                self.queue_full = True
                total_block.delete(blocks.popleft())
            self.counter_new_block += 1
            #? 放这儿仅做测试
            if just_demo:
                self.complete()
