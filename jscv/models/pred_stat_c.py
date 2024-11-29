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
import cv2

import jscv.utils.trainer as tn
from jscv.utils.utils import *     # do_once, on_train, redirect, color_tulpe_to_string
from jscv.utils.overall import global_dict
from jscv.utils.utils import torch

'''
    repair_dataset 的 classes_trunc 写死了，
    不是根据统计结果自动生成，后者比较麻烦没实现

    TODO: 曲线生成写死了，只有第4种
'''

just_demo = True

# repair_dataset_args_default = dict(
#     do_save_refined = True,
#     # save_refined_sample = True,
#     save_refined_dir = './data/water_seg_repaired',
#     default_trunc = [0.9, 0.98],
#     save_refined_save_img = False,
#     save_refined_label = True,
# )
repair_dataset_args_default = dict(
    do_save = True,
    save_dir = './data/water_seg_repaired',
    classes_trunc = [0.9, 0.98],
    save_org_img = False,
    save_label = True,
)


"""
    Prediction Statistics (multiple classes)
"""


class StatisticBlock:
    def __init__(self, num_classes, count_len, class_sum=False, device="cpu"):
        if device == "none_block":
            return
        self.num_classes = num_classes
        self.count_len = count_len
        self.do_class_sum = class_sum
        self.device = device

        self.counts = torch.zeros(num_classes, count_len).to(device)
        if class_sum:
            self.class_sum = torch.zeros(num_classes).to(device)
        self.sample_nums = 0

    def clone(self):
        blk = StatisticBlock(0, 0, device="none_block")
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
        num_classes=None,
        repair_dataset_args=repair_dataset_args_default,
        save_items=["counts", "class_sum", "smooth_weights"],
        lable_key="gt_semantic_seg",
    ):

        self.start_epoch = start_epoch
        self.count_len = count_len
        self.min_value = min_value
        self.trainer = None
        self.save_per_x_blocks = max(save_per_x_blocks, 1)
        self.counter_new_block = -1
        
        fig_max_lines = 4
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
        self.repair_dataset_args = repair_dataset_args


    def init(self):
        
        max_v = 0.97
        minv = self.min_value = 0.88
        # max_v = 1
        # minv = self.min_value = 0.6

        self.cfg = global_dict["cfg"]
        nC = self.num_classes = self.cfg.num_classes
        assert nC == 2
        # self.classes = self.cfg.get("classes")
        # self.classes = ["non-water", "non-water (wrong areas)", "water", "water (wrong areas)"]
        # self.cls_colors = ["#000000", "#00FF00", "#0000FF", "#00FFFF"]
        self.classes = ["non-water", "water"]
        self.cls_colors = ["#000000", "#0000FF"]


        self.blocks_queue.append(self.create_block())
        self.total_block = self.create_block()
        self.counter_step = 0
        self.batch_stat_finish = False

        clen = self.count_len
        minv = self.min_value
        self.xpoints = np.linspace(minv, max_v, clen, dtype=float)
        self.edges = np.linspace(minv, max_v, clen + 1, dtype=float)
        #self.edges = torch.arange(clen + 1).float() / (1 / (1 - minv)) * clen) + minv

        ''' color '''
        from jscv.utils.analyser import cfg as rgbcfg

        
        self.cls_names = self.classes

        ''' pred_statistic.txt '''
        odir = self.output_dir = os.path.join(self.cfg.workdir, "pred_statistic")
        if not os.path.exists(odir):
            os.makedirs(odir)
        fname = os.path.join(odir, "1.pred_statistic.txt")
        self.file_out = open(fname, "w+")


    def create_block(self):
        return StatisticBlock(self.num_classes*2, self.count_len, self.do_class_sum, 
                              self.trainer.device)


    def pred_stat(self, pred_logits, target, return_weight=True, result_kargs=None):
        if result_kargs is None:
            result_kargs = self.result_kargs
        weight_style = result_kargs["weight_style"]
        wrong_only = False
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
        # if wrong_only:
        #     valid &= wrong_mask

        stat = top1_value   # [BHW]


        # count
        for c in range(self.num_classes):
            cls_range = (target == c) & valid
            for i in range(clen):
                inds = (stat > edges[i]) & (stat < edges[i + 1]) & cls_range
                if stat_trunc < 1:
                    inds = inds & (stat < stat_trunc)
                counts[c*2][i] += inds.sum().item()

            cls_range = cls_range & wrong_mask
            for i in range(clen):
                inds = (stat > edges[i]) & (stat < edges[i + 1]) & cls_range
                if stat_trunc < 1:
                    inds = inds & (stat < stat_trunc)
                counts[c*2+1][i] += inds.sum().item()

        self.batch_stat_finish = True



    def complete(self):
        rs = self.result = {}
        result_kargs = self.result_kargs
        
        weight_style = result_kargs["weight_style"]
        
        nC = self.num_classes

        counts = self.total_block.counts    # C, N
        maxidx = torch.argmax(counts, dim=1, keepdim=True)
        maxi = counts.gather(1, maxidx)

        sumc = torch.sum(counts, dim=1, keepdim=False)
        sum0 = torch.sum(sumc)
        mc = (sum0 / nC ) / sumc

        cl = self.count_len

        counts = counts.clone() #* (cl / sum0)

        rs["counts"] = counts #* torch.unsqueeze(mc, 1)

        if weight_style == "hard_trunc":
            #trunc_rate = result_kargs.get("trunc_after", 1)
            # assert trunc_rate <= 1
            # if trunc_rate == 1:
            #     trunc_i = maxidx
            # else:
            #     i = maxidx
            #     while i > 0:
            #         i -= 1
            #         if counts[i] <= maxi * trunc_rate:
            #             break
            #     if i <= 0:
            #         trunc_i = None
            #     else:
            #         trunc_i = i
            
            trunc_i = maxidx

            rs["trunc_i"] = trunc_i
            if trunc_i is None:
                rs["trunc"] = None
            else:
                # print("trunc_rate", trunc_rate, self.xpoints[trunc_i])
                rs["trunc"] = []
                for ti in trunc_i:
                    rs["trunc"].append(self.xpoints[ti])

        elif weight_style == "smooth":
            #TODO smooth
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


    def unnormalize(self, img, mean, std):
        mean = torch.tensor(mean, device=img.device).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor(std, device=img.device).unsqueeze(-1).unsqueeze(-1)
        return ((img * std + mean)*255).permute(1,2,0)
    
    def convert_image(self, image):
        # image = F.interpolate(image, scale_factor=1/2, mode="bilinear")
        ds = global_dict['cfg'].train_dataset
        image = self.unnormalize(image, ds.mean, ds.std).cpu().numpy()
        return image

    def save_refined(self, pred_logits: torch.Tensor, batch):
        
        rda = self.repair_dataset_args
        if not isinstance(rda, dict) or not rda.get('do_save', False):
            return
        
        if not hasattr(self, 'count_batch'):
            self.count_batch = 0
        self.count_batch += 1

        if self.count_batch > rda['batchs']:
            print("Dataset Refinement Finished!\nexit()")
            exit()

        '''默认 ignore'''
        import jscv.utils.analyser as ana

        assert pred_logits.shape[0] == 1

        save_refined_dir = rda['save_dir']
        default_trunc = rda['classes_trunc']
        save_refined_save_img = rda['save_org_img']
        save_refined_label = rda['save_label']


        id = batch['id'][0]
        label = batch[self.lable_key]
        trunc = default_trunc
        if self.result is not None:
            #TODO
            # trunc = self.result.get('trunc', default_trunc)
            pass
        
        dddirs = [save_refined_dir, save_refined_dir+'/refined', save_refined_dir+'/debug']
        for ddd in dddirs:
            if not os.path.exists(ddd):
                os.makedirs(ddd)

        pred_logits = pred_logits[0].detach().softmax(0) #2,H,W
        label = label[0].to(pred_logits.device)

        
        pred = torch.argmax(pred_logits, 0, keepdim=True) #H,W
        top1_value = torch.gather(pred_logits, 0, pred).squeeze()
        pred = pred.squeeze()
        mask_wrong = pred != label
        
        H,W = label.shape
        

        mask_trunc = torch.zeros(label.shape, device=pred_logits.device).bool()
        
        assert len(trunc) == 2, '简化'
        mask_trunc_list = []
        color_list = [(0,0,255),(255,0,0)]
        trunc_len = 0
        for i, ti in enumerate(trunc):
            mask_cls_i = mask_wrong & (label == i)
            _tc = mask_cls_i & (top1_value > ti)
            mask_trunc |= _tc
            mask_trunc_list.append(_tc.cpu().numpy())
            trunc_len += torch.sum(_tc)
        del _tc

        rate = (trunc_len/H/W)*1000
        if not save_refined_label:
            id = f"{(int(rate))}_{id}"
        ignore_index = global_dict['cfg'].ignore_index
        mask_ignore = (label == ignore_index).cpu().numpy()
        
        # print(id, pred_logits.shape, label.shape, trunc, top1_value.shape, rate)
        # print(id, trunc, float(rate))
        
        if save_refined_label:
            label = label.cpu().numpy()
            for tc in mask_trunc_list:
                label[tc] = ignore_index
            fn = save_refined_dir+f'/refined/{id}_label.tif'
            cv2.imwrite(fn, label)
            map = ana.label2rgb(label, self.num_classes)
            map[label == ignore_index] = (255,0,0)
            cv2.imwrite(save_refined_dir+f'/debug/{id}_label.png', cv2.cvtColor(map, cv2.COLOR_RGB2BGR))
            print("Repaired:", fn, f"   {self.count_batch}/{rda['batchs']}")


        if save_refined_save_img:
            img = self.convert_image(batch['img'][0])
            cv2.imwrite(save_refined_dir+f'/debug/{id}-[1]-img.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            for tc, color in zip(mask_trunc_list, color_list):
                img[tc] = color
            cv2.imwrite(save_refined_dir+f'/debug/{id}-[2]-img_trunc.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not save_refined_label:
            map = ana.label2rgb(label.cpu().numpy(), self.num_classes)
            map[mask_ignore] = (100,100,100)
            cv2.imwrite(save_refined_dir+f'/debug/{id}-[3]-label.png', cv2.cvtColor(map, cv2.COLOR_RGB2BGR))



        # for tc, color in zip(mask_trunc_list, color_list):
        #     map[tc] = color
        # cv2.imwrite(save_refined_dir+f'/debug/{id}-[4]-trunc.png', cv2.cvtColor(map, cv2.COLOR_RGB2BGR))

        # map = ana.label2rgb(pred.cpu().numpy(), self.num_classes)
        # map[mask_ignore] = (100,100,100)
        # cv2.imwrite(save_refined_dir+f'/debug/{id}-[5]-pred.png', cv2.cvtColor(map, cv2.COLOR_RGB2BGR))


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
        
        batch = global_dict["trainer"].batch
        if not self.batch_stat_finish:
            self.pred_stat(pred, batch[self.lable_key], return_weight=False)

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

        # if save_refined_sample or self.queue_full:
        # if self.queue_full:
        self.save_refined(pred, batch)

        # print(self.show_aviliable, self.counter_new_block % self.save_per_x_blocks == 0)
        if self.show_aviliable and self.counter_new_block % self.save_per_x_blocks == 0:
            self.show_aviliable = False
            self.str_epoch_batch = f"e{trainer.epoch_idx}-{trainer.batch_idx + 1}"
            with redirect(self.file_out):
                save_items = ["counts", "class_sum"]
                # print("@1")
                #print(f"@ {self.str_epoch_batch},  spend time:", spend)
                for item in save_items:
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


    def print_1_counts_fig(self, counts, x, title="counts", subidx=None):
        lines = fig_reset()
        lines.extend(pyplot.plot(x, counts.tolist()))
        self._savefig_svg_("counts", subidx)

    def print_counts_fig(self, counts, x, title="counts", subidx=None):
        lines = fig_reset()
        c0 = self._c_sub_(0, subidx)
        # print("$$", counts.shape)
        C1 = counts[0].float() / counts[1]
        C2 = counts[2].float() / counts[3]
        # for c, count_c in enumerate(counts):

        lines.extend(pyplot.plot(x, C1.tolist(), color=self.cls_colors[0]))
        lines.extend(pyplot.plot(x, C2.tolist(), color=self.cls_colors[1]))
        pyplot.legend(lines, self.cls_names[c0:], loc="best")

        self._savefig_svg_("counts", subidx)




    def display(self, type):
        if self.result is not None:
            if type == 'counts' and "counts" in self.result:
                # print("@, type == 'counts' print_counts_fig")
                self.max_lines_ditribute(
                    self.result["counts"], self.print_counts_fig, x=self.xpoints
                )

            # elif type == "class_sum":
            #     k = "class_sum"
            #     if k in self.result:
            #         print("class_sum:")
            #         sum0 = sum(self.result[k])
            #         for c, v in enumerate(self.result[k]):
            #             cls = self.cls_names[c]
            #             print(f"{cls}: {v} ( {v / sum0:.2%} )")

            #     k = "class_balance_weights"
            #     if k in self.result:
            #         print("class_balance_weights:")
            #         for c, v in enumerate(self.result[k]):
            #             cls = self.cls_names[c]
            #             print(f"{cls}: {v}")
            # elif type == "smooth_weights":
            #     k = "weights"
            #     if k in self.result:
            #         self.print_counts_fig(self.result[k], self.xpoints, k)
                
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
