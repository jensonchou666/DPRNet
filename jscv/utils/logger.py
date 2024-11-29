from typing import Any, Dict, Optional, Union
import sys, os
from numbers import Number
from .table import MatrixTable
from .utils import Version, redirect, seek_line
from .trainer import TrainerCallback
import shutil




class Logger(MatrixTable):

    def __init__(
            self,
            save_dir: str,
            log_name: str = 'log',
            version: Optional[Union[int, str]] = None,
            #resume_epoch='last',
            use_stdout=False,
            values=[],
            suffix='.txt',
            front_display_topk=0,  # 0-disable, -1:all
            topk_monitor=None,
            show_none_monitor=False,
            monitor_mode='max',
            topk_avg=True,  # compute topk_avg
            first_column_key='epoch',
            save_old_logfile=True  # if resume, save old logfile
    ):
        super().__init__(values=values)
        self.use_stdout = use_stdout

        if self.use_stdout:
            self.filename = 'sys.stdout'
            self.file = sys.stdout
        else:
            if version is None:
                version = Version.last_version(save_dir) + 1
                self.do_resume = False
            else:
                self.do_resume = True
                #self.resume_epoch = resume_epoch
            if isinstance(version, int):
                version = f'version_{version}'
            self.save_dir = save_dir
            self.log_name = log_name
            self.suffix = suffix
            self.filename = log_name + suffix
            self.version_name = version
            self.logdir = os.path.join(save_dir, version)
            self.save_old_logfile = save_old_logfile
        self.top_values = []

        if topk_monitor is None:
            front_display_topk = 0

        self.show_syle['row_number'] = False
        # self.show_syle['default_interval'] = 4

        self.do_concate = False
        self.do_topk_avg = topk_avg
        self.first_key = first_column_key
        
        if front_display_topk < 0:
            self.str_topk_head = '*' * 40 + f' sort({topk_monitor}) ' + '*' * 40
        elif front_display_topk != 0:
            self.str_topk_head = '*' * 40 + f' top_{front_display_topk}({topk_monitor}) ' + '*' * 40
        if monitor_mode == 'max':
            self.fmode = self.bigger
        else:
            self.fmode = self.smaller
        self.monitor_mode = monitor_mode
        self.monitor = topk_monitor
        self.topk = front_display_topk
        self.show_none_monitor = show_none_monitor

    def bigger(self, a, b):
        if self.monitor not in a:
            return False
        elif self.monitor not in b:
            return True
        return a[self.monitor] > b[self.monitor]

    def smaller(self, a, b):
        if self.monitor not in a:
            return False
        elif self.monitor not in b:
            return True
        return a[self.monitor] < b[self.monitor]

    def init(self):
        self.init_log_file()
        if not self.do_concate and self.topk == 0:
            self.log_title()
            self.file.flush()

    def log_title(self):
        with redirect(self.file):
            #self.print_line()
            self.print_title()
            #self.print_line('inner_line')


    def init_log_file(self):
        if not self.use_stdout:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            self.filepath = os.path.join(self.logdir, self.filename)
            if self.do_resume and self.save_old_logfile:
                if os.path.exists(self.filepath):
                    v = Version.copyed_max_version(self.logdir,
                                                   self.log_name) + 1
                    old = self.log_name + f'({v})' + self.suffix
                    shutil.copyfile(self.filepath,
                                    os.path.join(self.logdir, old))
                    self.do_concate = True

            self.file = open(self.filepath, 'a+')

    def resume_log(self, first_row):
        # 老的恢复log的办法, 已弃用
        assert 'epoch' in self.columns_info, "log file doesn't have epoch column, can't concate"
        assert 'epoch' in first_row
        #title = self.columns_info['epoch']['title']
        epoch = first_row['epoch']

        with open(self.filepath, 'r') as f2:
            lines = f2.readlines()

        #in_log_region = True
        in_log_region = False
        idx = -1
        for idx, line in enumerate(reversed(lines)):
            s = line.strip().split(' ')
            if len(s) == 0:
                if in_log_region:
                    break
            s = s[0]
            ovl = self.show_syle['outer_v_line']
            if ovl != '':
                s2 = s[s.index(ovl) + len(ovl):]
                if s2 != '':
                    s = s2
                else:
                    s = s[1]

            if s.isdigit():
                in_log_region = True
                if int(s) < epoch:
                    break
            else:
                if in_log_region:
                    break
        if idx != 0:
            if idx > 0 and idx < len(lines):
                lines = lines[:-idx]

        self.file = open(self.filepath, 'w')
        self.file.writelines(lines)
        self.file.flush()

    def log(self, data=None, rows_len=None, flush=True, **kwargs):
        if data is None:
            data = self.values[-1]
            rows_len = len(self.values)

        if self.do_concate and not self.use_stdout:
            self.resume_log(data)
            self.do_concate = False

        with redirect(self.file):
            self.print_row(data, rows_len, **kwargs)

        if flush:
            self.file.flush()

    def sort_topk(self):
        dl = self.top_values
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                if self.fmode(dl[j], dl[i]):
                    dl[i], dl[j] = dl[j], dl[i]

    def topk_new(self, d):
        dl = self.top_values
        dl.append(d)
        for i in range(len(dl) - 1, 0, -1):
            if self.fmode(dl[i], dl[i - 1]):
                dl[i - 1], dl[i] = dl[i], dl[i - 1]


    def get_topk_avg(self, top_values):
        def getvalue(v):
            if isinstance(v, dict):
                return v.get('value', None)
            else:
                return v
        if self.do_topk_avg:
            sum_list = {}
            for row in top_values:
                assert isinstance(row, dict)
                for k, v in row.items():
                    v1 = getvalue(v)
                    if isinstance(v1, Number):
                        if k in sum_list:
                            sum_list[k][0] += v1
                            sum_list[k][1] += 1
                        else:
                            sum_list[k] = [float(v1), 1]    # TODO float
            avg_value = {}
            for k, sumv in sum_list.items():
                avg_value[k] = sumv[0] / sumv[1]
            
            # fk = self.first_key
            # if fk in avg_value.keys():
            avg_value[self.first_key] = f"avg_{len(top_values)}"
            return avg_value

    def log_with_topk(self, datas: list = None, flush=True, **kwargs):
        if datas is None:
            datas = self.values

        if self.use_stdout or self.topk == 0:
            return self.log(datas[-1], len(datas), flush, **kwargs)
        ret = seek_line(self.file, self.str_topk_head)
        if not ret:
            with redirect(self.file):
                print(self.str_topk_head)
        else:
            self.file.truncate()
        self.log_title()
        if len(self.top_values) == 0:
            self.top_values = datas.copy()
            self.sort_topk()
        else:
            self.topk_new(datas[-1])

        topvs = self.top_values
        if not self.show_none_monitor:
            for i, d in enumerate(topvs):
                if self.monitor not in d:
                    break
            topvs = topvs[:i]

        # self.topk_avg = self.get_topk_avg(topvs)  #wrong
        # topvs.append(self.topk_avg)

        if self.topk > 0:
            topvs = topvs[:self.topk]
        
        with redirect(self.file):
            i = 0
            for i, d in enumerate(topvs):
                self.print_row(d, i, **kwargs)
            self.topk_avg = self.get_topk_avg(topvs)
            self.print_row(self.topk_avg, i, **kwargs)
            i += 1
            # half
            halfk = int(self.topk // 2)
            if len(topvs) > halfk:
                self.print_row(self.get_topk_avg(topvs[:halfk]), i, **kwargs)

            
            self.print_line()
            print()
            self.log_title()
            for i, d in enumerate(datas):
                self.print_row(d, i, **kwargs)
        if flush:
            self.file.flush()

import time


class LoggerCallback(TrainerCallback):

    def __init__(self, logger: Logger, trainer=None):
        super().__init__(trainer)
        self.logger = logger


    def log(self, flush=True):
        logger = self.logger
        
        # print(self.trainer.epochs_datalist[-1])
        logger.log_with_topk(datas=self.trainer.epochs_datalist)
        #data = self.trainer.current_epoch_data
        # logger.log(data, len(self.trainer.epochs_datalist))


    def run_begin(self, before):
        if before:
            self.logger.init()

    def training_epoch_end(self, before):
        if not before:
            self.log()

    def validation_epoch_end(self, before):
        if not before:
            self.log()

    def savetopk(self):
        logger = self.logger
        if not hasattr(logger, 'topk_avg'):
            topk_avg = 0
        else:
            topk_avg = logger.topk_avg
        return {'topk': logger.topk, 'values': logger.top_values, 'avg': topk_avg}





def create_logger(args, cfg):
    if availabe(cfg, 'log_to_infofile') and cfg.log_to_infofile:
        log_name = cfg.info_filename
    else:
        log_name = cfg.log_filename

    set_default(
        cfg, {
            'logger_display_topk': -1,
            'logger_topk_monitor': 'val_mIoU',
            'logger_show_none_monitor': False,
            'logger_monitor_mode': 'max',
            'logger_save_old_logfile': True
        })

    suffix = '.' + log_name.split('.')[-1]
    log_name = log_name[:-len(suffix)]

    if cfg.do_resume:
        version_aug = {'version': cfg.version_name}
    else:
        version_aug = {}

    if availabe(cfg, "do_analyse_val") and cfg.do_analyse_val:
        cfg.logger_save_old_logfile = False

    logger = mylog.Logger(cfg.workdir_model,
                          log_name,
                          suffix=suffix,
                          front_display_topk=cfg.logger_display_topk,
                          topk_monitor=cfg.logger_topk_monitor,
                          first_column_key='epoch',
                          show_none_monitor=cfg.logger_show_none_monitor,
                          save_old_logfile=cfg.logger_save_old_logfile,
                          **version_aug)

    logger.register_columns({
        'epoch': {
            'title': 'epoch',
            'indent': 7
        },
        'stage': {
            'title': 'stage',
            'indent': 7
        },
        'spend': {
            'title': 'spend_time',
            'indent': 12,
            'format_func': format_time
        },
        'val_mIoU': {
            'title': 'val_mIoU%',
            'indent': 11,
            'format': '{:.4f}'
        },
        'val_F1': {
            'title': 'val_F1%',
            'indent': 11,
            'format': '{:.4f}'
        },
        'val_OA': {
            'title': 'val_OA%',
            'indent': 11,
            'format': '{:.4f}'
        },
        'val_loss': {
            'title': 'val_loss',
            'indent': 11,
            'format': '{:.5f}'
        },
        'train_mIoU': {
            'title': 'train_mIoU%',
            'indent': 11,
            'format': '{:.4f}'
        },
        'train_F1': {
            'title': 'train_F1%',
            'indent': 11,
            'format': '{:.4f}'
        },
        'train_OA': {
            'title': 'train_OA%',
            'indent': 11,
            'format': '{:.4f}'
        },
        'train_loss': {
            'title': 'train_loss',
            'indent': 11,
            'format': '{:.5f}'
        },
    })

    logger.show_syle['row_number'] = False
    logger.show_syle['split_v'] = '|'
    logger.show_syle['line_len_extra'] = 7
    logger.set_align('center')
    # logger.columns_info['epoch']['align'] = 'left'

    cfg.version_name = logger.version_name
    cfg.workdir = logger.logdir

    if not os.path.exists(cfg.workdir):
        os.makedirs(cfg.workdir)


    if args.version is not None:
        cfg.version = args.version
    cfg.version_str = ""
    if not availabe(cfg, "version"):
        cfg.version = None
    if cfg.version is not None:
        cfg.version_str = f"-{cfg.version}"
        with open(os.path.join(cfg.workdir, cfg.version_str), 'w'):
            pass

    return logger
