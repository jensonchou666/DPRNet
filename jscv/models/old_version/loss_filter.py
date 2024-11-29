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

import jscv.utils.trainer as tn
from jscv.utils.utils import *     # do_once, on_train, redirect, color_tulpe_to_string
from jscv.losses.utils import loss_map as loss_map_f
from jscv.utils.overall import global_dict
from jscv.utils.utils import torch

#from loss_statistic import loss_statistic

'''
def loss_statistic_py(lossmap: torch.Tensor,
                      loss_count_list: list,
                      maxloss: float):
    #? 之前的低效的方法， for循环遍历统计，花销非常大
    count_len = len(loss_count_list)
    Llast = count_len - 1
    for L in lossmap:
        L = float(L)
        if L >= maxloss:
            loss_count_list[Llast] += 1
        elif L > 0:
            idx = int(L / maxloss * count_len)
            loss_count_list[idx] += 1
'''


def loss_statistic(lossmap: torch.Tensor,
                   loss_count: torch.Tensor,
                   maxloss: float):
    count_len = loss_count.shape[0]
    lastL = count_len - 1
    lossmap *= (count_len / maxloss)
    lossmap = lossmap.int()
    lossmap[lossmap > lastL] = lastL
    count = torch.bincount(lossmap, minlength=count_len)
    loss_count += count  # .cpu()




default_curve_table_len = 400

class WeightCurve:
    def __init__(self, curve_table_len=default_curve_table_len) -> None:
        self.curve_table_len = curve_table_len
        self.create_curve_table()

    def forward(self, prop):
        pass

    def create_curve_table(self):
        ct = self.curve_table = []
        list_len = self.curve_table_len
        for i in range(list_len):
            ct.append(self.forward(float(i) / list_len))

    def from_prop_list(self, loss_prop_list: torch.Tensor):
        """
            for one class, achieve loss-lossweight table, {loss_prop shape: [count_len]}
        """
        table = []
        for lp in loss_prop_list:
            v = float(lp)
            table.append(self.forward(v))
        return table

    def extra_opt(self, *args, **kargs):
        pass

    def get_trunc_loss_idx(self, lp_list):
        self.trunc_loss_idx = []
        for prop_c in lp_list:
            idx = 0
            for prop in prop_c:
                if prop > self.trunc_prop:
                    break
                idx += 1
            self.trunc_loss_idx.append(idx)


class LinnerWeightCurve(WeightCurve):
    def __init__(self,
                 zero_weight=0.6,
                 maxw_prop=0.7,
                 max_weight=1.2,
                 gap_prop=0.85,
                 gap_weight=1,
                 trunc_prop=0.98,
                 curve_table_len=default_curve_table_len):
        self.zero_weight = zero_weight
        self.maxw_prop = maxw_prop
        self.max_weight = max_weight
        self.gap_prop = gap_prop
        self.gap_weight = gap_weight
        self.trunc_prop = trunc_prop
        self.L1 = ((max_weight - zero_weight) / maxw_prop, zero_weight)

        w_gm = (max_weight - gap_weight)
        b2 = w_gm * maxw_prop / (gap_prop - maxw_prop) + max_weight
        # == b2 = w_gm * gap_prop / (gap_prop - maxw_prop) + gap_weight
        self.L2 = (- w_gm / (gap_prop - maxw_prop), b2)

        k3 = - gap_weight / (trunc_prop - gap_prop)
        b3 = gap_weight * trunc_prop / (trunc_prop - gap_prop)
        self.L3 = (k3, b3)

        super().__init__(curve_table_len)


    def forward(self, prop):
        if prop < self.maxw_prop:
            k, b = self.L1
        elif prop < self.gap_prop:
            k, b = self.L2
        elif prop < self.trunc_prop:
            k, b = self.L3
        else:
            return 0
        return k * prop + b





class HardTruncWeightCurve(WeightCurve):
    def __init__(self, trunc_prop=0.92, curve_table_len=default_curve_table_len):
        self.trunc_prop = trunc_prop
        super().__init__(curve_table_len)
    
    def forward(self, prop):
        if prop < self.trunc_prop:
            return 1
        else:
            return 0

    def extra_opt(self, lp_list, *args, **kargs):
        self.get_trunc_loss_idx(lp_list)




class SmoothTruncWeightCurve(WeightCurve):
    def __init__(self,
                 trunc_prop=0.88,
                 zero_prop=0.98,
                 warm_epoch=0,
                 start_trunc_prop=0.95,
                 curve_table_len=default_curve_table_len):
        self.trunc_prop = trunc_prop
        self.zero_prop = zero_prop
        self.a = 1.0 / (zero_prop - trunc_prop) ** 2
        self.epoch_counter = 0
        self.last_epoch_idx = -1
        self.start_trunc_prop = start_trunc_prop
        self.end_trunc_prop = trunc_prop
        self.warm_epoch = warm_epoch

        super().__init__(curve_table_len)

        self.trunc_prop = start_trunc_prop

    def update_curve_table(self):
        if "trainer" not in global_dict or self.epoch_counter >= self.warm_epoch:
            return

        epoch_idx = global_dict["trainer"].epoch_idx
        if self.last_epoch_idx != epoch_idx:
            self.last_epoch_idx = epoch_idx
            self.epoch_counter += 1
            self.trunc_prop = self.start_trunc_prop + (self.epoch_counter / self.warm_epoch) * (
                self.end_trunc_prop - self.start_trunc_prop)
            self.create_curve_table()

    def forward(self, prop):
        if prop < self.trunc_prop:
            return 1
        elif prop > self.zero_prop:
            return 0
        else:
            x = prop - self.trunc_prop
            y = self.a * (x ** 2)
            return 1 - y

    def extra_opt(self, lp_list, *args, **kargs):
        self.get_trunc_loss_idx(lp_list)
        self.update_curve_table()




class StatisticBlock:
    def __init__(self, num_classes, count_len, device="cpu"):
        if device == "none_block":
            return
        self.device = device
        self.loss_counts = torch.zeros(num_classes, count_len).to(device)
        self.num_classes = num_classes
        self.count_len = count_len
        self.sample_nums = 0
        # for i in range(num_classes):
        #     lcl.append(torch.zeros(count_len))

    def compute_ererything(self, weight_curve=None):
        #TODO 更好做法： 计算 loss_proportion 不必加和所有像素后再做除法
        '''
            默认不 * 100,  eg. [0.1, 0.3, 0.4, 0.7, 1.0]
        '''
        lc = self.loss_counts
        cumsum = self.cumsum = torch.cumsum(lc, dim=1)
        self.sum_pixels = torch.sum(lc, dim=1, keepdim=True)
        lp_list = self.loss_proportion = cumsum / self.sum_pixels.expand_as(cumsum)
        if weight_curve is not None:
            self.weight_curve = weight_curve
            self.loss_weight = []
            if isinstance(weight_curve, WeightCurve):
                weight_curve.extra_opt(lp_list)
                weight_curve = [weight_curve] * self.num_classes
            else:
                for w in weight_curve:
                    w.extra_opt(lp_list)

            for wc, lp in zip(weight_curve, lp_list.cpu()):
                self.loss_weight.append(wc.from_prop_list(lp))
            self.loss_weight = torch.tensor(self.loss_weight).to(self.device)


    def clone(self):
        blk = StatisticBlock(0, 0, "none_block")
        blk.loss_counts = self.loss_counts.clone()
        blk.sample_nums = self.sample_nums
        blk.num_classes = self.num_classes
        blk.count_len = self.count_len
        return blk

    def update(self, other):
        self.loss_counts += other.loss_counts
        self.sample_nums += other.sample_nums
        return self

    def delete(self, other):
        self.loss_counts -= other.loss_counts
        self.sample_nums -= other.sample_nums
        return self




def fig_reset():
    pyplot.clf()
    pyplot.cla()
    return []


class LossStatistic:

    def from_cfg(cfg):
        '''
        set_default(cfg, dict(
            loss_statistic_start_epoch=4,
            loss_statistic_maxloss=10,
            loss_statistic_count_len=1000,
            loss_statistic_save_per_x_batch=100,
            loss_statistic_fig_max_lines=15,

            loss_statistic_sample_capacity=None,
            loss_statistic_queue_blocks=None,
            loss_statistic_per_x_batch_afterfull=None, 
        ))
        return LossStatistic(
            start_epoch=cfg.loss_statistic_start_epoch,
            maxloss=cfg.loss_statistic_maxloss,
            count_len=cfg.loss_statistic_count_len,
            save_per_x_batch=cfg.loss_statistic_save_per_x_batch,
            fig_max_lines=cfg.loss_statistic_fig_max_lines,
        )
        '''
        if not hasattr(cfg, "loss_statistic_args"):
            cfg.loss_statistic_args = dict()
        cargs = cfg.loss_statistic_args
        
        if "sample_capacity" not in cargs and on_train():
            cargs["sample_capacity"] = int(len(cfg.train_dataset) // cfg.train_batch_size)

        return LossStatistic(**cargs)


    def __init__(
        self,
        start_epoch=2,
        maxloss=8,                  # 可尽量设得高一点
        count_len=1000,              # count_len越大， 统计精度越高
        #? sample_capacity 会被调整到能被 queue_blocks 整除
        sample_capacity=1000,       # 最近样本最大总量
        # 每当最后一个队列块儿满了，popleft掉第一个块， push一个新的empty_block
        queue_blocks=10,            # 样本队列分块， 每一块有sample_capacity//cache_blocks个样本
        per_x_batch_afterfull=2,    # 样本队列满后每x_batch才统计一次

        save_per_x_blocks=2,
        fig_max_lines=15,

        wight_curve=SmoothTruncWeightCurve(),    # 若为List，则为每个类单独设置权重曲线，否则该曲线适用所有类
        
        # ["loss_counts", "loss_percent", "text_loss_count", "text_loss_%", "text_detail"],
        # "weight_curve", "loss_weight" "sum_pixels"
        save_items=["loss_percent", "loss_weight", "weight_curve", "sum_pixels"],
        lable_key="gt_semantic_seg",
    ):

        self.start_epoch = start_epoch
        self.count_len = count_len
        self.maxloss = maxloss
        self.trainer = None
        self.save_per_x_blocks = save_per_x_blocks
        self.counter_new_block = -1
        self.fig_max_lines = fig_max_lines

        self.wight_curve = wight_curve

        queue_blocks = max(queue_blocks, 1)
        self.block_size = block_size = sample_capacity // queue_blocks
        sample_capacity = block_size * queue_blocks
        self.sample_capacity = sample_capacity
        self.block_nums = queue_blocks
        self.per_xb = per_x_batch_afterfull
        self.lable_key = lable_key
        self.save_items = set(save_items)
        self.blocks_queue = deque()
        self.un_init = True
        self.total_aviliable = False
        self.queue_full = False
        self.show_aviliable = False


    def init(self):
        self.cfg = global_dict["cfg"]
        nC = self.num_classes = self.cfg.num_classes
        self.classes = self.cfg.get("classes")
        self.blocks_queue.append(self.create_block())
        self.total_block = self.create_block()


        if not isinstance(self.wight_curve, WeightCurve):
            assert len(self.wight_curve) == nC
            self.curve_is_list = True
        else:
            self.curve_is_list = False

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

        #if do_once(self, "makedir"):
        odir = self.output_dir = os.path.join(self.cfg.workdir, "loss_statistic")
        if not os.path.exists(odir):
            os.makedirs(odir)
        fname = os.path.join(odir, "1.loss_statistic.txt")
        self.file_loss_statistic = open(fname, "w+")

        def show_weight_curve(wight_curve_list, subidx=None):
            lines = fig_reset()
            for c, wight_curve in enumerate(wight_curve_list):
                assert isinstance(wight_curve, WeightCurve)
                c = self._c_sub_(c, subidx)
                #?低效,但通用
                x = np.linspace(0, 100, wight_curve.curve_table_len, dtype=float)
                lines.extend(pyplot.plot(x, wight_curve.curve_table, color=self.cls_colors[c]))
            pyplot.legend(lines, self.cls_names, loc="best")
            self._savefig_svg_("2.weight_curve", subidx, format_batch=False)

        if "weight_curve" in self.save_items:
            self.save_items.remove("weight_curve")
            if self.curve_is_list:
                #x = np.linspace(0, 100, self.wight_curve[0].curve_table_len, dtype=float)
                self.max_lines_ditribute(self.wight_curve, show_weight_curve)
            else:
                fig_reset()
                x = np.linspace(0, 100, self.wight_curve.curve_table_len, dtype=float)
                pyplot.plot(x, self.wight_curve.curve_table, color="black")
                self._savefig_svg_("2.weight_curve", None, format_batch=False)


    def get_wight_curve(self, c=0):
        if self.curve_is_list:
            return self.wight_curve[c]
        else:
            return self.wight_curve

    def create_block(self):
        return StatisticBlock(self.num_classes, self.count_len, self.trainer.device)


    def get_result(self, contain_last=False) -> StatisticBlock:
        if not self.total_aviliable:
            return None
        if contain_last:
            return self.total_block.clone().update(self.blocks_queue[-1])
        else:
            return self.total_block


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
        if trainer.stage != "train" or trainer.epoch_idx < self.start_epoch:
            return

        last_block = blocks[-1]
        assert isinstance(last_block, StatisticBlock)

        pred = pred.detach()
        targets = trainer.batch[self.lable_key].to(pred.device)
        pred_soft = F.log_softmax(pred, dim=1)

        lossmap = loss_map_f(pred_soft, targets,
                             self.cfg.ignore_index).detach().reshape(-1)

        targets = targets.reshape(-1)

        mL = self.maxloss

        t0 = time.time()
        for c, loss_count in enumerate(last_block.loss_counts):
            mask = targets == c
            lossmap_c = lossmap[mask]
            #idx = torch.nonzero(mask).squeeze()
            #lossmap_c = torch.index_select(lossmap, dim=0, index=idx)
            loss_statistic(lossmap_c, loss_count, mL)
        spend = time.time() - t0

        total_block = self.total_block
        last_block.sample_nums += 1
        if last_block.sample_nums == self.block_size:
            self.total_aviliable = True
            self.show_aviliable = True
            total_block.update(last_block)
            blocks.append(self.create_block())
            if len(blocks) > self.block_nums:
                self.queue_full = True
                total_block.delete(blocks.popleft())
            total_block.compute_ererything(self.wight_curve)
            self.counter_new_block += 1

        if self.show_aviliable and self.counter_new_block % self.save_per_x_blocks == 0:
            self.show_aviliable = False

            self.str_epoch_batch = f"e{trainer.epoch_idx}-{trainer.batch_idx + 1}"
            with redirect(self.file_loss_statistic):
                self.xpoints = np.linspace(0, self.maxloss, self.count_len, dtype=float)
                print(f"@ {self.str_epoch_batch},  spend time:", spend)
                for item in self.save_items:
                    self.print_loss_count(item)
                print("-" * 50, "\n")

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
        pyplot.savefig(odir + f"/{title}{s1}{s2}.svg", dpi=dpi, format="svg")

    def print_loss_prop_fig(self, proportions, x, subidx=None):

        lines = fig_reset()
        #pyplot.figure(facecolor="black")
        for c, proportion in enumerate(proportions.cpu()):
            c = self._c_sub_(c, subidx)
            lines.extend(pyplot.plot(x, proportion * 100, color=self.cls_colors[c]))
        pyplot.legend(lines, self.cls_names, loc="best")
        self._savefig_svg_("loss_percent", subidx)

    def print_loss_weight_fig(self, loss_weight, x, subidx=None):
        lines = fig_reset()
        #pyplot.figure(facecolor="black")
        for c, lw in enumerate(loss_weight):
            c = self._c_sub_(c, subidx)
            lines.extend(pyplot.plot(x, lw.tolist(), color=self.cls_colors[c]))
        pyplot.legend(lines, self.cls_names, loc="best")
        self._savefig_svg_("loss_weight", subidx)

    def print_loss_count_fig(self, loss_count, x, subidx=None):
        lines = fig_reset()
        for c, lc in enumerate(loss_count):
            c = self._c_sub_(c, subidx)
            aa = lc / torch.sum(lc)
            lines.extend(pyplot.plot(x, aa.tolist(), color=self.cls_colors[c]))
        pyplot.legend(lines, self.cls_names, loc="best")
        self._savefig_svg_("loss_count", subidx)

    def max_lines_ditribute(self, _List_, func, *args, **kargs):
        fml = self.fig_max_lines
        if len(_List_) > fml:
            for i in range(math.ceil(len(_List_) / fml)):
                func(_List_[fml * i: fml * i + fml], subidx=i, *args, **kargs)
        else:
            func(_List_, *args, **kargs)

    def print_loss_count(self, type):
        loss_counts = self.total_block.loss_counts
        props = self.total_block.loss_proportion

        if type == 'loss_percent':
            self.max_lines_ditribute(props, self.print_loss_prop_fig, x=self.xpoints)

        elif type == 'loss_counts':
            self.max_lines_ditribute(loss_counts, self.print_loss_count_fig, x=self.xpoints)

        elif type == "loss_weight":
            self.max_lines_ditribute(self.total_block.loss_weight, self.print_loss_weight_fig, x=self.xpoints)

        elif type == "sum_pixels":
            sum_pixels = self.total_block.sum_pixels.reshape(-1)
            pixels_percent = sum_pixels / torch.sum(sum_pixels) * 100
            pixels_percent = pixels_percent.tolist()
            for c, p in enumerate(pixels_percent):
                print(f"{c}, {self.cls_names[c]}, 占比: {p} %")
        else:
            for c, (loss_count, prop) in enumerate(zip(loss_counts, props)):
                assert isinstance(loss_count, torch.Tensor)
                assert isinstance(prop, torch.Tensor)
                cls_name = self._cls_name_(c)
                if type == "text_loss_count":
                    print("\nloss_count", c, cls_name)
                    s0 = torch.sum(loss_count)
                    aa = loss_count
                    if s0 != 0:
                        aa = aa / s0
                    print(aa.tolist(), "\n")
                elif type == "text_loss_%":
                    print("\nloss_%", c, cls_name)
                    print(prop.tolist(), "\n")
                elif type == "text_detail":
                    print("\ndetail", c, cls_name)
                    mL = self.maxloss
                    Lsz = self.count_len
                    for i, p in enumerate(prop):
                        ls = float(i) / Lsz * mL
                        print(f"({ls:.4}, {p:.4}%)  ", end="")
                    print()
            print("\n")

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
