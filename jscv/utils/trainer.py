#暂时不支持多GPU!!!

import torch
import torch.nn as nn
import os, re, sys
from typing import Iterable, Optional, Union, Any
import numpy as np
import time
import textwrap
import copy
import gc
STAT_CREATED = 0
STAT_TRAIN = 1
STAT_VAL = 2


def to_cuda(m):
    return m.cuda()


def to_cpu(m):
    return m.cpu()


def set_default(cfg, default_dict: dict):
    for k, v in default_dict.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

def eta_time(spend, batchs, total):
    return float(spend) * total / batchs


def format_time(tm):
    if int(tm / 60) > 0:
        return '{}m{:.2f}s'.format(int(tm / 60), tm % 60)
    else:
        return f'{tm:.4f}s'
    
def format_time_2(tm):
    if tm > 6000:
        h = int(tm / 3600)
        t2 = int(tm % 3600)
        return '{}h{}m{}s'.format(h, t2 // 60, t2 % 60)
    if tm > 60:
        return '{}m{}s'.format(int(tm / 60), int(tm % 60))
    else:
        return f'{tm:.2f}s'
    
#TODO 不能换行
def smart_print(*content, file=sys.stdout, max_line=100):
    # for i in range(len(content)):
    #     content[i] = str(content[i])
    # d = ''
    # for c in content:
    #     d += str(c) + ' '
    # d = d[:-1]
    # if file == sys.stdout:
    #     print(d)
    # else:
    #     print(textwrap.wrap(d, max_line), file=file, flush=True)
    print(*content, file=file, flush=True)


class TrainerCallback:

    def __init__(self, trainer=None):
        #assert isinstance(trainer, Trainer)
        assert isinstance(trainer, Trainer) or trainer is None
        self.trainer = trainer
        if isinstance(self.trainer,
                      Trainer) and self not in self.trainer.callbacks:
            self.trainer.callbacks.append(self)

    def run_begin(self, before):
        pass

    def run_end(self, before):
        pass

    def training_step(self, batch, batch_idx, before):
        pass

    def training_epoch_start(self, before):
        pass

    def training_epoch_end(self, before):
        pass

    def validation_step(self, batch, batch_idx, before):
        pass

    def validation_epoch_start(self, before):
        pass

    def validation_epoch_end(self, before):
        pass

    def if_skip_epoch(self, stage):
        pass


class SkipBatch(TrainerCallback):
    def __init__(self, stage, start=None, end=None, trainer=None):
        super().__init__(trainer)
        if start is None:
            end = -1
        if start is None:
            end = 111111111
        self.start = start
        self.end = end
        if stage == 'train':
            self.training_step = self._step_
        elif stage == 'val':
            self.validation_step = self._step_

    def _step_(self, batch, batch_idx, before):
        if before and batch_idx > self.start and batch_idx < self.end:
            return 'skip'



class CkptManager(TrainerCallback):

    ckpt_name_metrix_default = ['epoch={epoch}', 'monitor={monitor:.4f}']
    ckpt_name_metrix_epoch = ['epoch={epoch}']

    def from_config(cfg):

        set_default(
            cfg, {
                'path_resume_from': None,
                'ckptname_metrix': CkptManager.ckpt_name_metrix_epoch,
                'ckpt_split_char': '@',
                'save_topk': 3,
                'save_ckpt_monitor': None,
                'save_ckpt_monitor_mode': 'max',
                'save_ckpt_last': True,
                'save_ckpt_per_k_epoch': 1,
                'save_ckpt_when_stat': None,
                'save_ckpt_first_epoch_save': True,
                "last_epoch_continue_1e": False,
                "save_ckpt_remove_epoch_exlude": [],
                "save_ckpt_epoch_str": "epoch="
            })
        set_default(
            cfg, {
                "ckptname_monitor_prefix": str(cfg.save_ckpt_monitor) + '='
            })
        return CkptManager(workdir=cfg.workdir,
                           path_resume_from=cfg.path_resume_from,
                           ckptname_metrix=cfg.ckptname_metrix,
                           ckpt_name_split_char=cfg.ckpt_split_char,
                           save_topk=cfg.save_topk,
                           monitor=cfg.save_ckpt_monitor,
                           monitor_mode=cfg.save_ckpt_monitor_mode,
                           monitor_prefix=cfg.ckptname_monitor_prefix,
                           save_last=cfg.save_ckpt_last,
                           save_per_k_stat_epoch=cfg.save_ckpt_per_k_epoch,
                           save_when_stat=cfg.save_ckpt_when_stat,
                           last_epoch_continue_1e=cfg.last_epoch_continue_1e,
                           remove_epoch_exlude=cfg.save_ckpt_remove_epoch_exlude,
                           epoch_str=cfg.save_ckpt_epoch_str,
                           first_epoch_save=cfg.save_ckpt_first_epoch_save)

    def __init__(
            self,
            workdir,
            path_resume_from=None,
            #ckpt_name_format='epoch={epoch}@{monitor:.4f}',
            ckptname_metrix=['epoch={epoch}', 'monitor={monitor:.4f}'],
            ckpt_name_split_char='@',
            save_topk=3,
            monitor=None,  # 'val_OA',   # =None -> save_topk=-1
            monitor_mode='max',
            monitor_prefix=None,
            save_last=True,
            save_per_k_stat_epoch=1,
            save_when_stat=STAT_VAL,
            first_epoch_save=True,
            last_epoch_continue_1e=False,
            remove_epoch_exlude=[],
            epoch_str="epoch=",
            #always_topka1_files=True,
            trainer=None):
        super().__init__(trainer)
        self.workdir = workdir

        self.origin_ckpt_name_metrix = ckptname_metrix.copy()
        self.ckpt_name_split_char = ckpt_name_split_char
        self.save_topk = save_topk
        self.monitor_mode = monitor_mode
        self.save_last = save_last
        self.save_per_k_epoch = save_per_k_stat_epoch
        self.last_epoch_continue_1e = last_epoch_continue_1e
        self.remove_epoch_exlude = remove_epoch_exlude
        self.epoch_str = epoch_str

        if monitor is None:
            print(
                'CkptManager: because monitor is None, auto set self.save_topk = -1'
            )
            self.save_topk = -1

        self.flush_ckpt_name_format(monitor, ckptname_metrix)
        self._ckpt_name_metrix = ckptname_metrix
        self._monitor = monitor

        #self.always_topka1_files = always_topka1_files

        if save_when_stat is None:
            if 'val' in monitor.lower():
                save_when_stat = STAT_VAL
            elif 'train' in monitor.lower():
                save_when_stat = STAT_TRAIN
            else:
                save_when_stat = STAT_VAL
        self.save_when_stat = save_when_stat

        self.resume_from = path_resume_from
        self.exist_ckpt_list = []
        if first_epoch_save:
            self.epoch_counter = -1
        else:
            self.epoch_counter = 0

        self.custom_save_ckpt = []
        self.monitor_prefix = monitor_prefix

        self._skip_train = False
        self._skip_val = False
        self._last_ckpt_rm = None
        self.skip_save_model_cls = True

    def flush_ckpt_name_format(self, monitor, ckpt_name_format_metrix):
        for i, m in enumerate(ckpt_name_format_metrix):
            s1, s2 = 'monitor={monitor', '{monitor'
            if m.startswith(s1):
                if monitor is None:
                    ckpt_name_format_metrix[i] = 'monitor_set_none'
                else:
                    a1 = f'{monitor}={{{monitor}'
                    ckpt_name_format_metrix[i] = a1 + m[len(s1):]
            elif m.startswith(s2):
                if monitor is None:
                    ckpt_name_format_metrix[i] = 'monitor_set_none'
                else:
                    a1 = f'{{{monitor}'
                    ckpt_name_format_metrix[i] = a1 + m[len(s2):]
        s1 = f'{monitor}={{{monitor}'
        self.ckpt_name_has_monitor = False
        self.ckpt_name = ''
        for i, m in enumerate(ckpt_name_format_metrix):
            if m.startswith(s1):
                self.ckpt_name_has_monitor = True
            if i == 0:
                self.ckpt_name += m
            else:
                self.ckpt_name += self.ckpt_name_split_char + m

    @property
    def ckpt_name_metrix(self):
        return self._ckpt_name_metrix

    @ckpt_name_metrix.setter
    def ckpt_name_metrix(self, value):
        self.origin_ckpt_name_metrix = value.copy()
        self._ckpt_name_metrix = value
        self.flush_ckpt_name_format(self._monitor, self._ckpt_name_metrix)

    @property
    def monitor(self):
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value
        self._ckpt_name_metrix = self.origin_ckpt_name_metrix.copy()
        self.flush_ckpt_name_format(self._monitor, self._ckpt_name_metrix)

    def run_begin(self, before):
        if not before:
            if isinstance(self.resume_from, str):
                self.resume_ckpt(self.resume_from)
            self.append_old_ckpt_to_compare()

    def if_skip_epoch(self, stage):
        if stage == STAT_TRAIN:
            t = self._skip_train
            self._skip_train = False
            return t
        return False


    def training_epoch_end(self, before):
        if before or self.save_when_stat != STAT_TRAIN:
            return
        self.monitor_save_ckpt()

    # def validation_epoch_begin(self, before):
    #     if self._skip_val and before:
    #             self.trainer.skip_this_epoch = True

    def validation_epoch_end(self, before):
        # if self._skip_val and not before:
        #     self._skip_val = False
        #     self.trainer.skip_this_epoch = False
        if before or self.save_when_stat != STAT_VAL:
            return
        self.monitor_save_ckpt()

    def monitor_save_ckpt(self):
        self.epoch_counter += 1
        if self.epoch_counter % self.save_per_k_epoch != 0:
            return

        trainer = self.trainer
        d1 = trainer.current_epoch_data
        assert isinstance(d1, dict)
        if self.monitor is not None and self.monitor not in d1:
            print(f'on save_ckpt, no {self.monitor} in current_epoch_data')
            return

        # 如果current_epoch_data里没有 ckpt_name需要的key，则报错
        filename = self.ckpt_name.format(**d1) + '.ckpt'
        filepath = os.path.join(self.workdir, filename)

        last_is_topk = False
        if self.monitor is not None:
            value = d1[self.monitor]
            if len(self.exist_ckpt_list) < self.save_topk:
                last_is_topk = True
            else:
                for v, f in self.exist_ckpt_list:
                    if self.monitor_mode == 'max' and value > v:
                        last_is_topk = True
                        break
                    elif self.monitor_mode == 'min' and value < v:
                        last_is_topk = True
                        break
        if last_is_topk:
            self.exist_ckpt_list.append((value, filepath))

        self.rm_old_ckpt(last_is_topk)
        #TODO ctrl-c 时正好 save_ckpt会中断操作
        if last_is_topk:
            self.save_ckpt(filepath)

        if self.save_last:
            if self._last_ckpt_rm is not None and self.save_topk >= 0:
                self.remove_file(self._last_ckpt_rm)
            if not last_is_topk:
                self._last_ckpt_rm = filepath
                self.save_ckpt(filepath)
            else:
                self._last_ckpt_rm = None

            last_ckpt = os.path.join(self.workdir, 'last.ckpt')
            if os.path.exists(last_ckpt):
                self.remove_file(last_ckpt)

            os.symlink(filepath, last_ckpt + '.tmp')
            os.rename(last_ckpt + '.tmp', last_ckpt)

    def remove_file(self, fpath: str):
        if len(self.remove_epoch_exlude) > 0:
            es = self.epoch_str
            i = fpath.find(es)
            if i == -1:
                print(f"ckpt_manager: can't find '{es}' in '{fpath}'")
            else:
                i += len(es)
                j = fpath.find(self.ckpt_name_split_char, i)
                epoch = int(fpath[i:j])
                if epoch in self.remove_epoch_exlude:
                    return
        os.remove(fpath)


    def rm_old_ckpt(self, last_is_topk):
        f_list = self.exist_ckpt_list

        #TODO self.always_topka1_files保持始终有topk+1个文件
        # if self.save_last and self.always_topka1_files and last_is_topk:
        #     topk = self.save_topk + 1
        # else:
        topk = self.save_topk

        if topk < 0 or len(f_list) <= topk:
            return
        for i in range(len(f_list)):
            for j in range(i + 1, len(f_list)):
                if f_list[i][0] < f_list[j][0]:
                    f_list[i], f_list[j] = f_list[j], f_list[i]
        if self.monitor_mode == 'max' or topk == 0:
            for v, f in f_list[topk:]:
                self.remove_file(f)
            self.exist_ckpt_list = f_list[:topk]
        else:
            for v, f in f_list[:-topk]:
                self.remove_file(f)
            self.exist_ckpt_list = f_list[-topk:]

    def append_old_ckpt_to_compare(self):
        if not self.ckpt_name_has_monitor or self.monitor is None:
            return
        s, e = self.monitor_prefix, self.ckpt_name_split_char
        save_max = (self.monitor_mode == 'max')
        float_min = float("-inf") if save_max else float("inf")
        topk = self.save_topk
        re1 = re.compile(f'{s}.*{e}')

        for f in os.listdir(self.workdir):
            ret = re.search(re1, f)
            fpath = os.path.join(self.workdir, f)
            if ret is None:
                continue
            v = ret.group()[len(s):-len(e)]
            try:
                v = float(v)
            except Exception:
                self.exist_ckpt_list.append((float_min, fpath))
                continue
            self.exist_ckpt_list.append((v, fpath))

    def save_ckpt(self, filepath):
        if not self.skip_save_model_cls:
            try:
                self._save_ckpt_(filepath)
            except Exception as e:
                print("[Error] this model can't save model_class, so only save state_dict instead\n")
                self.skip_save_model_cls = True
            self._save_ckpt_(filepath)
        else:
            self._save_ckpt_(filepath)


    def _save_ckpt_(self, filepath):
        #trainer contrain: net optimizers lr_schedulers
        trainer = self.trainer
        d1 = {}
        d1['epoch'] = trainer.epoch_idx
        if self.save_when_stat == STAT_TRAIN:
            d1['stage'] = 'training_epoch_end'
        elif self.save_when_stat == STAT_VAL:
            d1['stage'] = 'validation_epoch_end'
        else:
            raise Exception('?')

        #TODO 同时保存 model和state_dict 不会占用额外存储空间
        if hasattr(trainer, 'model'):
            #? 若直接保存model错误, 跳过
            # if not self.skip_save_model_cls:
            #     d1['model'] = trainer.model
            d1['state_dict'] = trainer.model.state_dict()
        elif hasattr(trainer, 'net'):
            # if not self.skip_save_model_cls:
            #     d1['model'] = trainer.net
            d1['state_dict'] = trainer.net.state_dict()

        if hasattr(trainer, 'optimizers'):
            d1['optimizer_states'] = []
            for optimizer in trainer.optimizers:
                #print(optimizer, type(optimizer))
                #assert isinstance(optimizer, nn.Module)
                d1['optimizer_states'].append(optimizer.state_dict())

        if hasattr(trainer, 'lr_schedulers'):
            d1['lr_schedulers'] = []
            for lr_scheduler in trainer.lr_schedulers:
                d1['lr_schedulers'].append(lr_scheduler.__dict__)
        for k, savef, loadf in self.custom_save_ckpt:
            if savef is not None:
                # print(savef())
                d1[k] = savef()

        torch.save(d1, filepath)


    def resume_ckpt(self, filepath):
        if os.path.islink(filepath):
            filepath = os.readlink(filepath)
        d1 = torch.load(filepath, map_location='cpu')
        trainer = self.trainer

        if 'train' in d1['stage']:
            trainer.start_epoch = d1['epoch']
            self._skip_train = True
        elif 'val' in d1['stage']:
            trainer.start_epoch = d1['epoch'] + 1
            if self.last_epoch_continue_1e and trainer.start_epoch == trainer.max_epoch:
                trainer.max_epoch += 1
                print("trainer.max_epoch += 1 !")
        else:
            raise Exception("unknown stage")

        if hasattr(trainer, 'model'):
            model = trainer.model
        elif hasattr(trainer, 'net'):
            model = trainer.net

        assert isinstance(model, nn.Module)
        #TODO 这里不直接加载model而是state_dict
        if 'state_dict' in d1:
            model.load_state_dict(d1['state_dict'], strict=True)
        else:
            model.load_state_dict(d1['model'].state_dict(), strict=True)

        if hasattr(trainer, 'optimizers'):
            opts = trainer.optimizers
            for i, opt_state in enumerate(d1['optimizer_states']):
                opt = opts[i]
                #assert isinstance(opt, nn.Module)
                opt.load_state_dict(opt_state)

        if hasattr(trainer, 'lr_schedulers'):
            sches = trainer.lr_schedulers
            for i, state in enumerate(d1['lr_schedulers']):
                # pass
                #TODO 不使用load_state_dict方式
                try:
                    sches[i].load_state_dict(state)
                except Exception as e:
                    print("Error!!! sches[i].load_state_dict(state) :")
                    print(e)
                    print("无法加载 lr_scheduler的state_dict")
                    print("本次忽略，但请不要继续恢复训练, lr_schedulers的定义有问题")

        for k, savef, loadf in self.custom_save_ckpt:
            if loadf is not None:
                if k not in d1:
                    print(f"{k} not in ckpt")
                    loadf(None)
                else:
                    loadf(d1[k])

        print("√ resumed from:", filepath)

    def register_save_load(self, key, save_func=None, load_func=None):
        self.custom_save_ckpt.append((key, save_func, load_func))


class Trainer:

    def __init__(self,
                 max_epoch,
                 val_per_k_epoch=1,
                 accelerator='gpu',
                 gpus=None,
                 devices=None,
                 strategy=None,
                 skip_train=False,
                 std_log_dir='work_dir/STD_LOG',#用于安全中断
                 callbacks=[]):

        assert strategy is None, '暂时不支持多GPU'

        if accelerator == 'gpu':
            self.to_device = to_cuda
            if gpus is not None:
                assert isinstance(gpus, int)
                assert gpus == 1, '暂时不支持多GPU'
                devices = [0]
                torch.cuda.set_device(0)
                self.gpu_id=0
            elif devices is not None:
                assert isinstance(devices, list)
                assert len(devices) == 1, '暂时不支持多GPU'
                gpus = 1
                torch.cuda.set_device(devices[0])
                self.gpu_id=devices[0]
            self.device = "cuda"
        elif accelerator == 'cpu':
            self.to_device = to_cpu
            self.device = "cpu"
        else:
            raise TypeError(' only support cpu 、 gpu')

        self.accelerator = accelerator
        self.gpus = gpus
        self.devices = devices
        self.strategy = strategy
        self.skip_train = skip_train
        self.std_log_dir = std_log_dir

        # for c in callbacks:
        #     assert isinstance(c, TrainerCallback), 'must be TrainerCallback

        self.callbacks = callbacks
        for c in callbacks:
            c.trainer = self

        self.start_epoch = 0
        self.max_epoch = max_epoch

        self.val_per_k_epoch = val_per_k_epoch

        self._val_adder = 1

        self.epoch_idx = -1
        self.batch_idx = -1
        self.stat = STAT_CREATED
        self.this_epoch_skipped = False

        self.epochs_datalist = []
        self.current_epoch_data = {}
        self.batchs_datalist = []
        self.current_batch_data = {}

    @property
    def stage(self):
        if self.stat == STAT_CREATED:
            return 'created'
        elif self.stat == STAT_TRAIN:
            return 'train'
        elif self.stat == STAT_VAL:
            return 'val'
        # else:
        #     return None

    def save_datas(self):
        # 可以在子类进行重载， 如果某些数据不需要保存的话
        return copy.deepcopy(self.epochs_datalist)

    def load_datas(self, datas):
        self.epochs_datalist = datas
        self.current_epoch_data = datas[-1]
        #self.batchs_datalist = self.current_epoch_data['batchs']
        #self.current_batch_data = self.batchs_datalist[-1]

    #TODO 实现auto_batch_size
    def choose_auto_batch_size(self):
        max_epoch = self.max_epoch

    def run(self):
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True
        
        # it_f = self.std_log_dir + f'/interrupt_{os.environ["CUDA_VISIBLE_DEVICES"]}'
        # print('interrupt if', it_f, 'exist.')

        for c in self.callbacks:
            c.run_begin(before=True)
        self.run_begin()
        for c in self.callbacks:
            c.run_begin(before=False)

        for epoch_idx in range(self.start_epoch, self.max_epoch):

            self.epoch_idx = epoch_idx

            skip_this_epoch = False
            for c in self.callbacks:
                skip_this_epoch = True if c.if_skip_epoch(STAT_TRAIN) else skip_this_epoch

            if not self.skip_train and not skip_this_epoch:

                self.stat = STAT_TRAIN
                self.data_loader = self.get_train_dataloader()
                self.epochs_datalist.append({'epoch': epoch_idx, 'stage': 'train'})
                self.current_epoch_data = self.epochs_datalist[-1]

                for c in self.callbacks:
                    c.training_epoch_start(before=True)
                self.training_epoch_start()
                for c in self.callbacks:
                    c.training_epoch_start(before=False)

                self.current_epoch_data['batchs'] = []
                self.batchs_datalist = self.current_epoch_data['batchs']

                for batch_idx, batch in enumerate(self.data_loader):
                    # if os.path.exists(it_f):
                    #     os.remove(it_f)
                    #     print("interrupt!!!")
                    #     exit()
                    
                    self.batch_idx = batch_idx
                    self.batch = batch
                    self.batchs_datalist.append({'batch_idx': batch_idx})
                    self.current_batch_data = self.batchs_datalist[-1]
                    
                    ret = None
                    for c in self.callbacks:
                        ret = c.training_step(batch, batch_idx, before=True)
                        if ret == 'skip':
                            break
                    if ret == 'skip':
                        continue
                    self.training_step(batch, batch_idx)
                    #TODO 是否耗时?
                    for c in self.callbacks:
                        c.training_step(batch, batch_idx, before=False)
                    


                for c in self.callbacks:
                    c.training_epoch_end(before=True)
                self.training_epoch_end()
                for c in self.callbacks:
                    c.training_epoch_end(before=False)

            if (epoch_idx + self._val_adder) % self.val_per_k_epoch != 0:
                continue

            skip_this_epoch = False
            for c in self.callbacks:
                skip_this_epoch = True if c.if_skip_epoch(STAT_VAL) else skip_this_epoch

            if not skip_this_epoch:
                self.stat = STAT_VAL
                self.data_loader = self.get_val_dataloader()
                self.epochs_datalist.append({'epoch': epoch_idx, 'stage': 'val'})
                self.current_epoch_data = self.epochs_datalist[-1]

                for c in self.callbacks:
                    c.validation_epoch_start(before=True)
                self.validation_epoch_start()
                for c in self.callbacks:
                    c.validation_epoch_start(before=False)

                self.current_epoch_data['batchs'] = []
                self.batchs_datalist = self.current_epoch_data['batchs']
                for batch_idx, batch in enumerate(self.data_loader):
                    # if os.path.exists(it_f):
                    #     os.remove(it_f)
                    #     print("interrupt!!!")
                    #     exit()

                    self.batch_idx = batch_idx
                    self.batch = batch
                    self.batchs_datalist.append({'batch_idx': batch_idx})
                    self.current_batch_data = self.batchs_datalist[-1]
                    ret = None
                    for c in self.callbacks:
                        ret = c.validation_step(batch, batch_idx, before=True)
                        if ret == 'skip':
                            break
                    if ret == 'skip':
                        continue
                    self.validation_step(batch, batch_idx)
                    for c in self.callbacks:
                        c.validation_step(batch, batch_idx, before=False)

                for c in self.callbacks:
                    c.validation_epoch_end(before=True)
                self.validation_epoch_end()
                for c in self.callbacks:
                    c.validation_epoch_end(before=False)

        for c in self.callbacks:
            c.run_end(before=True)
        self.run_end()
        for c in self.callbacks:
            c.run_end(before=False)

    #region Define follow functoin:

    def get_train_dataloader(self):
        pass

    def get_val_dataloader(self):
        pass

    def run_begin(self):
        pass

    def run_end(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_start(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_start(self):
        pass

    def validation_epoch_end(self):
        pass

    #endregion --------------------------------

    def last_epoch_data(self, stage='train'):
        epochs_datalist_reversed = reversed(self.epochs_datalist)
        next(epochs_datalist_reversed)
        ret = None
        for e in epochs_datalist_reversed:
            if e["stage"] == stage:
                ret = e
                break
        return ret


    def is_first_epoch(self):
        return self.epoch_idx == self.start_epoch

    def is_last_epoch(self):
        return self.epoch_idx == self.max_epoch - 1

    def is_first_batch(self):
        return self.batch_idx == 0

    def is_last_batch(self):
        return self.batch_idx == len(self.data_loader) - 1

    def fist_epoch_validation(self, _do=True):
        if _do:
            self._val_adder = 0
        else:
            self._val_adder = 1

    def loader_len(self):
        return len(self.data_loader)

    def train_loader_len(self):
        return len(self.get_train_dataloader())

    def val_loader_len(self):
        return len(self.get_val_dataloader())

from .metric import SegEvaluator


class Evaluator:
    def __init__(self) -> None:
        self.stat = 'train'

    def append(self, result):
        pass

    def evaluate(self):
        pass

class Joint_Evaluator(Evaluator):

    def __init__(self, *evals):
        super().__init__()
        self.evals = evals
    
    def append(self, result):
        for e in self.evals:
            e.append(result)
    
    def evaluate(self):
        r = {}
        for e in self.evals:
            e.stat = self.stat
            r.update(e.evaluate())
        return r
    
    def __str__(self) -> str:
        s = f'{self.stat} evaluate:\n'
        for e in self.evals:
            s += str(e.__class__)+":\n" + e.__str__()

        return s



class SegmentEvaluator(Evaluator, SegEvaluator):
    Default_CutOffDatasets = [
        'potsdam', 'vaihingen', 'whubuilding', 'massbuilding', 'inriabuilding'
    ]

    def from_config(cfg, CLS=None):

        if not hasattr(cfg, 'CutOffDatasets'):
            cfg.CutOffDatasets = SegmentEvaluator.Default_CutOffDatasets

        set_default(
            cfg, {'classes': None,
                  'trainer_do_cutoff':
                  SegmentEvaluator.if_do_cutoff(cfg.dataset_name, cfg.CutOffDatasets),
                  'trainer_metrix_percent_style': True,
                  })
        if CLS is None:
            CLS = SegmentEvaluator
        return CLS(
            num_class=cfg.num_classes,
            classes_name=cfg.classes,
            percent_style=cfg.trainer_metrix_percent_style,
        )

    def __init__(self, 
                 num_class,
                 classes_name,
                 pred_key='pred',
                 from_logits=True,
                 percent_style=True):
        Evaluator.__init__(self)
        SegEvaluator.__init__(self, num_class)
        if classes_name is None:
            classes_name = []
            for i in range(num_class):
                classes_name.append(f'cls_{i}')
        self.classes_name = classes_name
        if percent_style:
            self._multi_ = 100
        else:
            self._multi_ = 1
        self.result = None
        self.pred_key = pred_key
        self.from_logits = from_logits


    def if_do_cutoff(dataset_name, _CutOffDatasets=None):
        if _CutOffDatasets is None:
            _CutOffDatasets = SegmentEvaluator.Default_CutOffDatasets
        return dataset_name in _CutOffDatasets

    def append(self, result):
        pred = result[self.pred_key]
        target_cpu = result["target_cpu"]
        if self.from_logits:
            pred = pred.argmax(dim=1).cpu()
        else:
            pred = pred.cpu()
        for i in range(target_cpu.shape[0]):
            self.add_batch(target_cpu[i].numpy(), pred[i].numpy())

    def evaluate(self):
        #TODO 探讨CutOff的简便做法

        iou_per_class = self.Intersection_over_Union()

        # if self.do_cutoff:
        #     mIoU = np.nanmean(iou_per_class[:-1])
        #     F1 = np.nanmean(self.F1()[:-1])
        # else:
        mIoU = np.nanmean(iou_per_class)
        F1 = np.nanmean(self.F1())
        OA = np.nanmean(self.OA())
        mIoU *= self._multi_
        F1 *= self._multi_
        OA *= self._multi_
        
        class_iou = {}
        for class_name, iou in zip(self.classes_name, iou_per_class):
            class_iou[class_name] = iou * self._multi_

        self.reset()
        self.result = {
            f'{self.stat}_mIoU': mIoU,
            f'{self.stat}_F1': F1,
            f'{self.stat}_OA': OA,
            'class_IoU': class_iou
            # 'mIoU': mIoU,
            # 'F1': F1,
            # 'OA': OA,
        }
        self.miou = mIoU
        self.f1 = F1
        self.oa = OA
        self.class_iou = class_iou

        return self.result

    def __str__(self) -> str:
        if self.result is None:
            return 'None result'
        else:
            s = f'Segment Result, key={self.pred_key}\n'+\
                f'mIoU: {self.miou}, F1: {self.f1}, OA: {self.oa}\nclass_IoU:\n'
            for k, v in self.class_iou.items():
                s += f'{k:<20}:{v}\n'
            return s


class EdgeAccEvaluator(Evaluator):

    def __init__(self,
                 edge_pool_kernel=13,
                 ignore_index=None,
                 pred_key='pred',
                 from_logits=False):
        super().__init__()
        def get_edge_conv2d(channel=1):
            conv_op = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
            sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
            sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
            sobel_kernel = np.repeat(sobel_kernel, channel, axis=0)
            sobel_kernel = np.repeat(sobel_kernel, channel, axis=1)
            conv_op.weight.data = torch.from_numpy(sobel_kernel)
            return conv_op
        self.edge_conv2d = get_edge_conv2d().cuda()
        self.edge_pool = nn.MaxPool2d(kernel_size=edge_pool_kernel, stride=1, padding=edge_pool_kernel//2)
        self.ignore_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=5//2)
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.pred_key = pred_key
        
        self.count_acc = 0
        self.count_edge_range = 0


    def append(self, result):
        pred = result[self.pred_key]
        target = result['target'].unsqueeze(1)
        if self.from_logits:
            pred = torch.argmax(pred, 1)
        shape = pred.shape
        if len(shape) == 4:
            # B, C, H, W = pred.shape
            pred = torch.argmax(pred, 1)

        edge_range = self.edge_conv2d(target.float()) > 0.1

        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index).float()
            ignore_mask = self.ignore_pool(ignore_mask).bool()
            edge_range = edge_range & ~ignore_mask
        edge_range = self.edge_pool(edge_range.float()).squeeze(1).bool()
        acc = edge_range & (pred == target)
        self.count_acc += acc.sum()
        self.count_edge_range += edge_range.sum()


    def evaluate(self):
        self.acc = round(float(self.count_acc/self.count_edge_range*100), 2)
        r = {
            f'{self.stat}_EACC': self.acc
        }
        self.result = r
        self.count_acc = 0
        self.count_edge_range = 0
        
        return r
    
    def __str__(self) -> str:
        return f'edge_acc({self.pred_key}): {self.acc}\n'




class Model_Trainer(Trainer):

    def from_config(cfg):
        set_default(
            cfg, {
                'val_per_k_epoch': 1,

                #? 只要模型里有 step_train 就优先调用
                "trainer_custom_train_func": None,
                "trainer_split_input": True,
                
                'img_target_key': ('img', 'gt_semantic_seg'),
                
                'trainer_callbacks': [],
                'trainer_statistic_train_step_data': True,
                
                'trainer_do_log': True,
                'trainer_log_per_k': 50,
                'trainer_log_per_type': 'batch',
                'trainer_log_files': [sys.stdout],
                'trainer_accumulate_n': 1,
                'trainer_no_backward': False,
                'skip_train': False,
            })
        
        if not hasattr(cfg, 'evaluator'):
            cfg.evaluator = SegmentEvaluator.from_config(cfg)
        
        if hasattr(cfg, "optimizer"):
            set_default(cfg, {"optimizers": [cfg.optimizer]})
        else:
            set_default(cfg, {"optimizers": None})
        if hasattr(cfg, "lr_scheduler"):
            set_default(cfg, {"lr_schedulers": [cfg.lr_scheduler]})
        else:
            set_default(cfg, {"lr_schedulers": None})

        if hasattr(cfg, "evaluator_type"):
            cfg.evaluator = cfg.evaluator_type.from_config(cfg)

        return Model_Trainer(
            max_epoch=cfg.max_epoch,
            model=cfg.model,
            train_dataloader=cfg.train_loader,
            val_dataloader=cfg.val_loader,
            optimizers=cfg.optimizers,
            lr_schedulers=cfg.lr_schedulers,
            
            # custom_train=cfg.trainer_custom_train,
            custom_train_func=cfg.trainer_custom_train_func,
            split_input=cfg.trainer_split_input,

            evaluator=cfg.evaluator,
            callbacks=cfg.trainer_callbacks,
            statistic_train_step=cfg.trainer_statistic_train_step_data,
            
            do_log=cfg.trainer_do_log,
            log_per_k=cfg.trainer_log_per_k,
            log_per_type=cfg.trainer_log_per_type,
            log_files=cfg.trainer_log_files,
            accumulate_n=cfg.trainer_accumulate_n,
            img_target_key=cfg.img_target_key,
            val_per_k_epoch=cfg.val_per_k_epoch,
            no_backward=cfg.trainer_no_backward,
            skip_train=cfg.skip_train,
            **cfg.device_arguments)  # TODO device_arguments

    def __init__(
            self,
            max_epoch,
            model,
            train_dataloader,
            val_dataloader,
            evaluator: Evaluator,

            optimizers: Optional[Union[Iterable, nn.Module]] = None,
            lr_schedulers: Optional[Union[Iterable, Any]] = None,

            # if statistic in train_step?  set False will heighten train_step speed, but miss train_OA...
            statistic_train_step=True,
            
            do_log=True,
            log_per_k=50,  # TODO 按time log
            log_per_type='batch',  # 'batch k(batchs)'  or  'time k(s)'
            log_files=[sys.stdout],
            accumulate_n=1,  # update weights per n batches
            img_target_key=('img', 'gt_semantic_seg'),

            skip_train=False,
            custom_train_func=None,
            split_input=False,
            no_backward=False,

            # logger=None,
            callbacks=[],
            val_per_k_epoch=1,
            accelerator='gpu',
            gpus=None,
            devices=None,
            strategy=None,
            **kwargs):

        super().__init__(max_epoch=max_epoch,
                         val_per_k_epoch=val_per_k_epoch,
                         accelerator=accelerator,
                         gpus=gpus,
                         devices=devices,
                         strategy=strategy,
                         skip_train=skip_train,
                         callbacks=callbacks)
        assert isinstance(model, nn.Module)

        # print("@ evaluator", type(evaluator))

        # ??这里能不能现在加载到gpu上??
        self.model = self.to_device(model)

        if isinstance(optimizers, Iterable):
            self.optimizers = optimizers
        elif optimizers is not None:
            self.optimizers = [optimizers]
        if isinstance(lr_schedulers, Iterable):
            self.lr_schedulers = lr_schedulers
        elif lr_schedulers is not None:
            self.lr_schedulers = [lr_schedulers]

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.img_target_key = img_target_key
        self.accumulate_n = accumulate_n
        self.statistic_train = statistic_train_step

        self.do_log = do_log
        self.log_per_k_bz = log_per_k
        self.log_files = log_files

        self.evaluator_train = evaluator
        self.evaluator_val = copy.deepcopy(evaluator)
        self.evaluator_val.stat = 'val'

        self.split_input = split_input
        self.result = {}
        self.no_backward = no_backward

        # if custom_train:
        #     if custom_train_func is None:
        #         assert hasattr(model, 'step_train')
        #         custom_train_func = model.step_train
        #     self.train_func = custom_train_func
        if custom_train_func is not None:
            self.custom_train = True
            self.train_func = custom_train_func
        elif hasattr(model, "step_train"):
            self.custom_train = True
            self.train_func = model.step_train
        else:
            self.custom_train = False
            self.train_func = model

    def run_begin(self):
        self.start_time = time.time()
        self.train_spend = None
        self.val_spend = None

    def training_epoch_start(self):
        self.model.train()
        self.epoch_t0 = time.time()
        # for file in self.log_files:
        #     print(file=file)
        #     print('-'*80, file=file)

    def training_step(self, batch, batch_idx):
        #super().training_step(batch, batch_idx)
        del self.result

        #TODO
        #if self.split_input:
        ki, km = self.img_target_key
        img, target_cpu = batch[ki], batch[km]
        
        # if self.split_input:
        img, target = self.to_device(img), self.to_device(target_cpu)
        batch[ki], batch[km] = img, target

        model = self.model
        

        if self.split_input:
            result = model(img, target)
        else:
            result = model(batch)
        
        result["target_cpu"] = target_cpu
        result["target"] = target
        self.result = result
        losses = result.get("losses")

        sum_loss = 0
        if losses is not None:
            if isinstance(losses, dict):
                for ls in losses.values():
                    sum_loss = ls + sum_loss
                if 'main_loss' in losses:
                    loss = losses.pop('main_loss')
                elif 'loss_main' in losses:
                    loss = losses.pop('loss_main')
                elif 'loss' in losses:
                    loss = losses.pop('loss')
                else:
                    loss = torch.zeros([1])
            else:
                sum_loss = losses
                loss = losses
                losses = {}
        else:
            loss = torch.zeros([1])
            losses = {}

        if not self.no_backward:
            if sum_loss != 0 and sum_loss.requires_grad:
                sum_loss.backward()

        opt_step = (batch_idx + 1) % self.accumulate_n == 0
        sch_step = self.is_last_batch()
        if opt_step:
            for opt in self.optimizers:
                opt.step()
                opt.zero_grad()
        if sch_step:
            for sch in self.lr_schedulers:
                sch.step()


        #------ log ------
        if self.do_log and (batch_idx + 1) % self.log_per_k_bz == 0:
            tnow = time.time()
            t1 = tnow - self.epoch_t0
            eta_e = eta_time(t1, batch_idx + 1, self.loader_len())
            eta = eta_e - t1
            #prog = batch_idx + 1 / (self.loader_len() + self.val_loader_len())
            if self.val_spend is None:
                eta_e_val = eta_e * self.val_loader_len() / self.loader_len()
            else:
                eta_e_val = self.val_spend
            if self.train_spend is not None:
                eta_e = self.train_spend
            total = (eta_e + eta_e_val) * (self.max_epoch - self.start_epoch)
            fromstart = tnow - self.start_time
            eta_t = format_time_2(total - fromstart)
            fromstart = format_time_2(fromstart)
            eta = format_time_2(eta)
            total = format_time_2(total)
            tstr = f'ETA:{eta}|{eta_t}'
            for file in self.log_files:
                str_extra_losses = ''
                for k, ls in losses.items():
                    ls = float(ls)
                    str_extra_losses += f', {k}: {ls}'
                str_1 = f'[epoch_{self.epoch_idx} {batch_idx+1}/{self.loader_len()}]'
                smart_print(
                    f"{fromstart:<6} (train) {str_1:<17} {tstr:<18} loss:{float(loss)}{str_extra_losses}",
                    file=file)

        #------ statistic ------
        if 'skip' not in result:
            if self.statistic_train:
                self.evaluator_train.append(result)
            for k, ls in losses.items():
                if isinstance(ls, torch.Tensor):
                    losses[k] = ls.detach().cpu().numpy()
                else:
                    losses[k] = torch.tensor(ls).numpy()

            self.current_batch_data['loss'] = loss.detach().cpu().numpy()
            if 'extra_losses' not in self.current_epoch_data:
                self.current_epoch_data['extra_losses'] = {k: [v] for k, v in losses.items()}
            else:
                etl = self.current_epoch_data['extra_losses']
                for k in etl.keys():
                    etl[k].append(losses[k])

    # def run_begin(self):
    #     self.model = self.to_device(self.model)
    #     print("!!!!!", len(self.model.state_dict()))


    def training_epoch_end(self):

        self.train_spend = self.current_epoch_data['spend'] = time.time() - self.epoch_t0

        loss = []
        for batch in self.batchs_datalist:
            if 'loss' in batch:
                loss.append(batch['loss'])
        loss = np.stack(loss).mean()
        self.current_epoch_data['train_loss'] = float(loss)
        # pop掉节省空间
        self.current_epoch_data.pop('batchs')
        #self.current_epoch_data['mloss'] = loss

        etl = self.current_epoch_data.pop('extra_losses')
        for k, losses in etl.items():
            etl[k] = float(np.stack(losses).mean())
        self.current_epoch_data['extra_losses'] = etl

        if self.do_log:
            for file in self.log_files:
                #batchs = self.current_epoch_data.pop('batchs')
                smart_print('END ',
                            self.current_epoch_data,
                            file=file)
        if self.statistic_train:
            self.current_epoch_data.update(self.evaluator_train.evaluate())
        if self.do_log:
            for file in self.log_files:
                smart_print(self.evaluator_train, file=file)

        gc.collect()  # 强制进行垃圾回收
        torch.cuda.empty_cache()  # 清理显存缓存
        torch.cuda.synchronize()  # 确保所有操作完成
        time.sleep(5)

    def validation_epoch_start(self):
        self.model.eval()
        self.epoch_t0 = time.time()
        # for file in self.log_files:
        #     print(file=file)
        #     print('-'*80, file=file)

    def validation_step(self, batch, batch_idx):
        del self.result
        torch.cuda.empty_cache()


        ki, km = self.img_target_key
        img, target_cpu = batch[ki], batch[km]
        # if self.split_input:
        img, target = self.to_device(img), self.to_device(target_cpu)
        batch[ki], batch[km] = img, target

        with torch.no_grad():
            if self.split_input:
                result = self.model(img, target)
            else:
                result = self.model(batch)

        result["target_cpu"] = target_cpu
        result["target"] = target
        self.result = result
        losses = result.get("losses")

        if losses is not None:
            if isinstance(losses, dict):
                if 'main_loss' in losses:
                    loss = losses.pop('main_loss')
                elif 'loss_main' in losses:
                    loss = losses.pop('loss_main')
                elif 'loss' in losses:
                    loss = losses.pop('loss')
                else:
                    loss = torch.zeros([1])
            else:
                loss = losses
                losses = {}
        else:
            loss = torch.zeros([1])
            losses = {}

        #------ log ------
        if self.do_log and (batch_idx + 1) % self.log_per_k_bz == 0:
            for file in self.log_files:
                tnow = time.time()
                t1 = tnow - self.epoch_t0
                eta_e = eta_time(t1, batch_idx + 1, self.loader_len())
                eta = eta_e - t1
                #prog = batch_idx + 1 / (self.loader_len() + self.val_loader_len())
                if self.val_spend is not None:
                    eta_e = self.val_spend
                if self.train_spend is None:
                    eta_train = eta_e * self.train_loader_len() / self.loader_len()
                else:
                    eta_train = self.train_spend
                total = (eta_e + eta_train) * (self.max_epoch - self.start_epoch)
                fromstart = tnow - self.start_time
                eta_t = format_time_2(total - fromstart)
                fromstart = format_time_2(fromstart)
                eta = format_time_2(eta)
                total = format_time_2(total)
                tstr = f'ETA:{eta}|{eta_t}'
                str_extra_losses = ''
                for k, ls in losses.items():
                    ls = float(ls)
                    str_extra_losses += f', {k}: {ls}'
                smart_print(
                    f"{fromstart} (val) [e{self.epoch_idx} {batch_idx+1}/{self.loader_len()}] {tstr}, loss:{float(loss)}{str_extra_losses}",
                    file=file)

        #------ statistic ------
        if 'skip' not in result:
            self.evaluator_val.append(result)
            for k, ls in losses.items():
                if isinstance(ls, torch.Tensor):
                    losses[k] = ls.detach().cpu().numpy()
                else:
                    losses[k] = torch.tensor(ls).numpy()

            self.current_batch_data['loss'] = loss.detach().cpu().numpy()
            if 'extra_losses' not in self.current_epoch_data:
                self.current_epoch_data['extra_losses'] = {k: [v] for k, v in losses.items()}
            else:
                etl = self.current_epoch_data['extra_losses']
                for k in etl.keys():
                    etl[k].append(losses[k])


    def validation_epoch_end(self):

        self.val_spend = self.current_epoch_data['spend'] = time.time() - self.epoch_t0

        loss = np.stack([batch['loss'] for batch in self.batchs_datalist]).mean()
        self.current_epoch_data['val_loss'] = float(loss)
        # pop掉节省空间
        self.current_epoch_data.pop('batchs')
        #self.current_epoch_data['mloss'] = loss

        etl = self.current_epoch_data.pop('extra_losses')
        for k, losses in etl.items():
            etl[k] = float(np.stack(losses).mean())
        self.current_epoch_data['extra_losses'] = etl
        if self.do_log:
            for file in self.log_files:
                #batchs = self.current_epoch_data.pop('batchs')
                smart_print('END ',
                            self.current_epoch_data,
                            file=file)
        if self.statistic_train:
            self.current_epoch_data.update(self.evaluator_val.evaluate())
        if self.do_log:
            for file in self.log_files:
                smart_print(self.evaluator_val, file=file)
        
        gc.collect()  # 强制进行垃圾回收
        torch.cuda.empty_cache()  # 清理显存缓存
        torch.cuda.synchronize()  # 确保所有操作完成
        time.sleep(5)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_val_dataloader(self):
        return self.val_dataloader



