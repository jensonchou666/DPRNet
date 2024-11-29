from typing import Any
from torch.nn import functional as F
import torch
import time
import os
import math
from matplotlib import pyplot
import numpy as np
from collections import deque
from numbers import Number
from scipy import signal


import jscv.utils.trainer as tn
from jscv.utils.utils import *     # do_once, on_train, redirect, color_tulpe_to_string
from jscv.utils.overall import global_dict
from jscv.utils.utils import torch


just_demo = True
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
        start_epoch=1,
        count_len=40,              # count_len越大， 统计精度越高
        min_value=0.6,
        #? sample_capacity 会被调整到能被 queue_blocks 整除
        sample_capacity=100,       # 最近样本最大总量
        # 每当最后一个队列块儿满了，popleft掉第一个块， push一个新的empty_block
        queue_blocks=4,            # 样本队列分块， 每一块有sample_capacity//cache_blocks个样本
        per_x_batch_afterfull=1,    # 样本队列满后每x_batch才统计一次
        save_per_x_blocks=2,
        fig_max_lines=15,   # 一个图表里最多允许的曲线数

        do_class_sum=True,
        result_kargs=result_kargs_hard_trunc,

        save_items=["counts", "class_sum", "smooth_weights"],
        lable_key="gt_semantic_seg",
    ):

        self.start_epoch = start_epoch
        self.count_len = count_len
        self.min_value = min_value
        self.trainer = None
        self.save_per_x_blocks = max(save_per_x_blocks, 1)
        self.counter_new_block = -1
        self.fig_max_lines = fig_max_lines
        self.per_xb = per_x_batch_afterfull
        self.do_class_sum = do_class_sum
        self.lable_key = lable_key
        self.save_items = set(save_items)
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
        self.cfg = global_dict["cfg"]
        nC = self.num_classes = self.cfg.num_classes
        self.classes = self.cfg.get("classes")
        self.blocks_queue.append(self.create_block())
        self.total_block = self.create_block()
        self.counter_step = 0
        self.batch_stat_finish = False

        clen = self.count_len
        minv = self.min_value
        self.xpoints = np.linspace(minv, 1, clen, dtype=float)
        self.edges = np.linspace(minv, 1, clen + 1, dtype=float)
        #self.edges = torch.arange(clen + 1).float() / (1 / (1 - minv)) * clen) + minv


        ''' color '''
        from jscv.utils.analyser import cfg as rgbcfg
        self.cls_names = []
        self.cls_colors = []
        for c in range(nC):
            self.cls_names.append(self._cls_name_(c))
            color = color_tulpe_to_string(rgbcfg.palette[c])
            #TODO 因背景白色， 故把白色曲线置黑
            if color == "#FFFFFF":
                color = "#000000"
            self.cls_colors.append(color)

        ''' pred_statistic.txt '''
        odir = self.output_dir = os.path.join(self.cfg.workdir, "pred_statistic")
        if not os.path.exists(odir):
            os.makedirs(odir)
        fname = os.path.join(odir, "1.pred_statistic.txt")
        self.file_out = open(fname, "w+")


    def create_block(self):
        return StatisticBlock(self.count_len, self.do_class_sum, self.num_classes, self.trainer.device)


    def pred_stat(self, pred_logits, target, return_weight=True, result_kargs=None):
        if result_kargs is None:
            result_kargs = self.result_kargs
        weight_style = result_kargs["weight_style"]
        wrong_only = result_kargs.get("wrong_only", False)
        stat_trunc = result_kargs.get("stat_trunc", 1)

        cfg = global_dict["cfg"]
        #!
        ignore_index = cfg.ignore_index

        pred = pred_logits.detach()
        # pred = torch.softmax(pred_logits.detach(), dim=1)
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

        rw = return_weight and (self.result is not None)
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
            else:
                # 截断策略 >>>
                trunc_left_move = result_kargs.get("trunc_left_move", 0.0)
                inds = valid_mask & wrong_mask & (stat > (trunc - trunc_left_move))
                if result_kargs.get("return_trunc_mask", False):
                    extra_return.append(inds.reshape(B, H, W))
                weight_map[inds] = 0.
        else:
            weight_map = torch.ones(B * H * W)


        if rw and "class_balance_weights" in self.result:
            class_balance_weights = self.result["class_balance_weights"]
            do_class_balance = True
        else:
            do_class_balance = False

        if self.do_class_sum:
            #TODO 稍微有点多余操作
            class_sum = last_block.class_sum
            for c in range(self.num_classes):
                inds = valid_mask & (target == c)
                # print(c, inds.sum().item())
                class_sum[c] += inds.sum().item()
                if do_class_balance:
                    cw = class_balance_weights[c]
                    weight_map[inds] *= float(cw)

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
            correct_weight = result_kargs.get("correct_weight", 1)
            wrong_weight = result_kargs.get("wrong_weight", 1)
            if correct_weight != 1:
                weight_map[valid_mask & ~wrong_mask] *= correct_weight
            if wrong_weight != 1:
                weight_map[valid_mask & wrong_mask] *= wrong_weight
            # if result_kargs.get("class_balance_rate", 0) != 0:

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
                # print("trunc_rate", trunc_rate, self.xpoints[trunc_i])
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



    def step(self, pred):
        if not on_train():
            return

        if self.un_init:
            self.trainer = global_dict.get("trainer")
            if self.trainer is None:
                return
            self.init()
            self.un_init = False

        trainer = self.trainer
        blocks = self.blocks_queue
        #assert isinstance(trainer, tn.Model_Trainer)


        #? 只统计train—epoch
        if trainer.stage != "train" or trainer.epoch_idx < self.start_epoch:
            return


        self.counter_step += 1
        if self.result is not None and self.counter_step % self.per_xb != 0:
            return


        last_block = blocks[-1]
        assert isinstance(last_block, StatisticBlock)

        #t0 = time.time()
        
        if not self.batch_stat_finish:
            self.pred_stat(pred, global_dict["trainer"].batch[self.lable_key],
                           return_weight=False)

        self.batch_stat_finish = False

        #spend = time.time() - t0


        total_block = self.total_block
        last_block.sample_nums += 1
        # print(last_block.sample_nums, self.block_size)
        if last_block.sample_nums == self.block_size:
            self.show_aviliable = True
            total_block.update(last_block)
            blocks.append(self.create_block())
            if len(blocks) > self.block_nums:
                self.queue_full = True
                total_block.delete(blocks.popleft())
                if not just_demo:
                    self.complete()
            self.counter_new_block += 1
            #? 放这儿仅做测试
            if just_demo:
                self.complete()


        if self.show_aviliable and self.counter_new_block % self.save_per_x_blocks == 0:
            self.show_aviliable = False
            self.str_epoch_batch = f"e{trainer.epoch_idx}-{trainer.batch_idx + 1}"
            with redirect(self.file_out):
                #print(f"@ {self.str_epoch_batch},  spend time:", spend)
                for item in self.save_items:
                    self.display(item)
                print("-" * 50, "\n", flush=True)


    def _c_sub_(self, c, subidx):
        if subidx is not None:
            return c + self.fig_max_lines * subidx
        return c

    def _savefig_svg_(self, title, subidx=None, dpi=1000, format_batch=True):
        odir = self.output_dir
        s1, s2 = "", ""
        if format_batch:
            s1 = f"_{self.str_epoch_batch}"
        if subidx is not None:
            s2 = f"_{subidx}"
        pyplot.savefig(odir + f"/{title}{s1}{s2}.svg", dpi=dpi, format="svg", bbox_inches = 'tight')


    def max_lines_ditribute(self, _List_, func, *args, **kargs):
        fml = self.fig_max_lines
        if len(_List_) > fml:
            for i in range(math.ceil(len(_List_) / fml)):
                func(_List_[fml * i: fml * i + fml], subidx=i, *args, **kargs)
        else:
            func(_List_, *args, **kargs)


    def print_counts_fig(self, counts, x, title="counts", subidx=None):
        lines = fig_reset()
        lines.extend(pyplot.plot(x, counts.tolist()))
        self._savefig_svg_("counts", subidx)



    def display(self, type):
        if self.result is not None:
            if type == 'counts' and "counts" in self.result:
                self.print_counts_fig(self.result["counts"], self.xpoints)
            elif type == "class_sum":
                k = "class_sum"
                if k in self.result:
                    print("class_sum:")
                    sum0 = sum(self.result[k])
                    for c, v in enumerate(self.result[k]):
                        cls = self.cls_names[c]
                        print(f"{cls}: {v} ( {v / sum0:.2%} )")

                k = "class_balance_weights"
                if k in self.result:
                    print("class_balance_weights:")
                    for c, v in enumerate(self.result[k]):
                        cls = self.cls_names[c]
                        print(f"{cls}: {v}")
            elif type == "smooth_weights":
                k = "weights"
                if k in self.result:
                    self.print_counts_fig(self.result[k], self.xpoints, k)
                
    def save(self):
        #TODO
        return dict(
            blocks_queue=self.blocks_queue,
            
            )
    
    def load(self, seed):
        pass


    def _cls_name_(self, c):
        if self.classes is not None:
            return self.classes[c]
        else:
            return str(c)

    # def max_loss(self, targets: torch.Tensor, lossmaps: torch.Tensor):
    #     lossmaps = lossmaps.detach()

    #     bz = targets.shape[0]

    #     for c in range(self.num_classes):
    #         if self.classes is not None:
    #             cls_name = self.classes[c]
    #         else:
    #             cls_name = ""
    #         mask = targets == c
    #         lossmap_c = lossmaps.clone()
    #         lossmap_c[mask] = -1
    #         lm2 = lossmap_c.reshape(bz, -1)
    #         maxls = float(torch.max(lm2, dim=1)[0])
    #         print("max_loss", c, cls_name, maxls)
