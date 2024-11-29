#! /opt/conda/bin/python

import os, sys
import re
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
import random
import time
import shutil
#import torchsummary

from jscv.utils import logger as mylog, trainer as tn

from jscv.utils.utils import *
from jscv.utils import overall
from jscv.utils.analyser import AnalyseCallback, analyser_from_cfg
from jscv.utils.cfg import *

from jscv.utils.shell_dict import shell_dict


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    
    #? important ☆☆☆☆☆☆☆☆☆
    parser.add_argument("config_path",
                        type=Path,
                        help="Path to the the root configuration file, required.")

    #? important ☆☆☆☆☆☆☆☆☆
    parser.add_argument("-c", "--overload_cfgs_path",
                        nargs='+',
                        type=Path,
                        help="Extra configuration files, usually a dataset configuration, separated from the root config.")

    parser.add_argument("--seed", default="random", help="'time', or 'random', or a specific integer.")


    parser.add_argument("--cpu",
                        default=False,
                        action="store_true",
                        help='use cpu')

    group_gpus = parser.add_mutually_exclusive_group()

    ''' Multiple GPUs are not supported yet, so you can only use the parameter: --gpus 1 '''
    group_gpus.add_argument('--gpus',
                            type=int,
                            help='the number of gpus to use '
                            '(only applicable to non-distributed training)'
                            '. Multiple GPUs are not supported yet, so you can only use the parameter: --gpus 1'
                            )
    group_gpus.add_argument('--gpu-ids',
                            type=int,
                            nargs='+',
                            help='ids of gpus to use '
                            '(only applicable to non-distributed training)'
                            '. Multiple GPUs are not supported yet'
                            )
    parser.add_argument("--strategy",
                        default='ddp',
                        help='gpu strategy, only useful when gpu-num > 1,'
                        'default-strategy: ddp')


    #? important ☆☆☆☆☆☆☆☆☆
    parser.add_argument(
        "-r",
        "--resume-from",
        help=
        'This parameter specifies where to resume training, suppose you write: -r {resume-from}. \n'
        'The location of resume file is: work_dir/{dataset_name}/{model_name}-e{max_epoch}/{version}/{your ckpt-file for resuming}. \n'
        '   Eg. "work_dir/GID_Water/DPRNet-e80/version_0/epoch=21@val_mIoU=87.4@DPRNet-GID_Water.ckpt" \n'
        'The format of {resume-from} include the following:\n'
        "   1. {version index}/{epoch index},            eg. -r 2/12 (*/version_2/epoch=12...ckpt),  -r 4/last (*/version_2/last.ckpt), "
        "-r last/50 (*/version_{last}/epoch=50...ckpt).   notes: * automatically determined by your config.\n"
        "   2. {version name}/{epoch index},             eg. -r v1.1/33 (*/v1.1/epoch=33...ckpt).\n"
        "   3. {version index or name}/{ckpt file name}  eg. -r 0/aaa.ckpt (*/version_0/aaa.ckpt). \n"
        "   4. '-r .  '   It is equivalent to:  '-r last/last' \n"
        "   5. ☆☆☆   '-r {full path of ckpt file}'       eg. -r ./work_dir/GID_Water/seg_r50_d4-e80/a1/epoch=23@val_mIoU=86.83@seg_r50_d4-GID_Water.ckpt\n"

        )

    #? Deprecated, please write the pre-training location directly in the configuration file.
    parser.add_argument("--pretrain-path", '-p',
                        help="pretained backbone ckpt path,  set '-p none' "
                        "means don't load pretrained backbone weights")
    parser.add_argument("--pretrain-prefix",
                        help="pretained backbone ckpt prefix")

    group_batch = parser.add_mutually_exclusive_group()
    group_batch.add_argument(
        "-b", "--batch-size",
        help="batch size  eg. -b 4   (train_batch_size = 4, val_batch_size = 4),  \n"
             "                -b 4,2 (train_batch_size = 4, val_batch_size = 2).  please write it in config.")
    
    #TODO Not supported: auto-batch
    group_batch.add_argument('-a',
                             "--auto-batch-size",
                             default=False,
                             action="store_true",
                             help="Not supported")

    parser.add_argument("-e", "--max_epoch", type=int, help="max_epoch, please write it in config.")

    parser.add_argument("--workdir", type=Path, default=Path('work_dir'), 
                        help="training files put to ./{work_dir}/dataset_name/model_name/version/, No need to change")
    
    # 使用-q, 禁止所有log输出到屏幕
    parser.add_argument("-q", "--quiet", default=False, action="store_true", help="quiet mode")

    parser.add_argument("--shell-dict",
                        default=False,
                        action="store_true",
                        help="Display the state_dict parameters in pretrained-ckpt, model or optimizer, a little trick.")

    # Deprecated
    parser.add_argument("--do-test", default=False, action="store_true", help="do test on last")
    parser.add_argument("--description", "--desc", nargs='+', help="description of this work")
    parser.add_argument("-v", "--version", help="not important.")
    
    parser.add_argument("--no-backward", default=False, action="store_true")

    # 跳过每轮的训练, 只验证
    parser.add_argument("--skip-train", default=False, action="store_true", 
                        help="skip train every epoach. When model validation only, it is useful.")
    

    
    '''
        下面4个参数控制 图片输出功能， 存放在: {taining dir}/analyse/
        通常用验证集，每轮输出几张图像， 具体在config文件里有介绍使用方法。
        每个epoach按次序输出，不会出现重复的batch（除非一个验证集遍历完了）
    '''
    parser.add_argument("-t", "--times-analyse", type=int, default=1, 
                        help="How many batches are output per epoach?")
    
    parser.add_argument("--astage", "--analyse-stage", default='val', choices=('train', 'val'), 
                        help="Training stage or verification stage?")
    
    parser.add_argument("--analyse", "--analyze", 
                        help="Start after the N-th batch, ")
    
    parser.add_argument("--skip-analyse", default=False, action="store_true", 
                        help="If true, No image output.")


    #? Important ☆☆☆
    parser.add_argument("--mini", type=int, 
                        help="Specify a small number of batches, quickly verify that the model has no bugs")

    

    return parser.parse_args()




def shell_dicts(args, cfg):
    #? 如果指定了shell_dict，训练之前可以查看各种parameters
    # 查看net.state_dict()
    # 查看optimizers.state_dict()
    # 查看schedulers.state_dict()
    # 查看torch_load(resume_ckpt)
    # 查看torch_load(pretrain_ckpt)
    if args.shell_dict:
        root_dict = {}
        root_dict['state_dict(model)'] = cfg.model.state_dict()
        optimizers, lr_schedulers = cfg.trainer.optimizers, cfg.trainer.lr_schedulers
        Lop = [x.state_dict() for x in optimizers]
        Llr = [x.state_dict() for x in lr_schedulers]
        root_dict['[state_dict](optimizers)'] = Lop
        root_dict['[state_dict](schedulers)'] = Llr
        if cfg.do_resume:
            root_dict['resume_ckpt'] = torch_load(cfg.path_resume_from)
            root_dict['path_resume_from'] = cfg.path_resume_from
        for i, (ckpt, Mlist) in enumerate(cfg.pretrain_dict.items()):
            if isinstance(ckpt, str):
                ckpt_i = f'pretrain_{i}'
                root_dict[ckpt_i] = torch_load(ckpt)
                root_dict[f'{ckpt_i}_path'] = ckpt
            elif isinstance(ckpt, dict):
                ckpt_i = f'pretrain_{i}'
                root_dict[ckpt_i] = ckpt
        shell_dict(root_dict)


def create_logger(args, cfg):
    '''
        Used to generate 'log-model.txt' file.
        Print the indicator per epoach on chart.
    '''
    
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
            'logger_save_old_logfile': True,
            'logger_items': ['val_mIoU', 'val_F1', 'val_OA', 'val_loss',
                             'train_mIoU', 'train_F1', 'train_OA', 'train_loss']
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
                          first_column_key='epoch',
                          topk_monitor=cfg.logger_topk_monitor,
                          monitor_mode=cfg.logger_monitor_mode,
                          show_none_monitor=cfg.logger_show_none_monitor,
                          save_old_logfile=cfg.logger_save_old_logfile,
                          **version_aug)

    items = {
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
        }
    }
    
    for it in cfg.logger_items:
        items[it] = {
            'title': it,
            'indent': 11,
            'format': '{:.6f}'
        }

    logger.register_columns(items)

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
  

def open_outputs_log(cfg):
    '''
        Used to generate 'outputs.log' file.
    '''
    file2 = open(os.path.join(cfg.workdir, 'outputs.log'), 'a+')
    #file2.write('-' * 40 + f' {cfg.model_name}-{cfg.dataset_name} ' + '-' * 40 + '\n')
    time0 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    file2.write('-' * 40 + f' {time0} ' + '-' * 40 + '\n')
    file2.flush()
    return file2


def pretreat(args, cfg):
    '''
        对配置文件初始化
    '''
    
    #? Important 钩子队列， 用于在训练流程中， 扩展各种各种操作
    # 如： 打印logger的callback， 保存权重的callback
    if not availabe(cfg, "trainer_callbacks"):
        cfg.trainer_callbacks = []

    cfg.analyse_stage = args.astage


    # analyse 相关
    if args.analyse is not None:
        cfg.do_analyse_val = True
        cfg.analyse_begin = int(args.analyse)
        if args.times_analyse == 1:
            args.times_analyse = 20
        #TODO 使train阶段也可analyse
        cfg.analyse_stage = 'val'
        cfg.skip_train = args.skip_train = True
        cfg.analyse_skip_same = True
        args.batch_size = '1,1'
        cfg.trainer_callbacks.append(tn.SkipBatch('val', -1, args.times_analyse - 1))
        cfg.val_crop_size = None
        cfg["donot_rename_workdir"] = True
        cfg["last_epoch_continue_1e"] = True
    cfg.times_analyse = args.times_analyse
    cfg.skip_analyse = args.skip_analyse
    cfg.trainer_no_backward = args.no_backward

    if not cfg.skip_analyse:
        ac = overall.global_dict['analyser'] = AnalyseCallback.from_cfg(cfg)
        cfg.trainer_callbacks.append(ac)
        analyser_from_cfg(cfg)

    if args.mini is not None:
        if not hasattr(cfg, "dataset_kwargs"):
            cfg.dataset_kwargs = {}
        cfg.dataset_kwargs["mini"] = True
        cfg.dataset_kwargs["mini_len"] = args.mini

    overall.before_cfg_create(args, cfg)

    #? Important ☆☆☆☆ 在这里调用cfg文件里的方法

    if "before_create_dataset" in cfg:
        cfg.before_create_dataset(cfg)
    cfg.create_dataset(cfg)
    if 'before_create_model' in cfg:
        cfg.before_create_model(cfg)
    cfg.create_model(cfg)
    if 'after_create_model' in cfg:
        cfg.after_create_model(cfg)

    description = ''
    if args.description is not None and len(args.description) > 0:
        description = args.description[0]
        for des in args.description[1:]:
            description += ' ' + des
        if availabe(cfg, "description"):
            cfg.description = description + ', ' + cfg.description
        else:
            cfg.description = description




def train_model():
    path0 = sys.path[0]
    #Set env before import cfg
    os.environ['run_mode'] = 'train'
    
    #? overall.global_dict 里存放了整个环境要用的变量
    global_dict = overall.global_dict
    global_dict['training'] = True
    global_dict['run_mode'] = 'train'
    #os.environ['train'] = 'true'

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.enable = True

    #? 覆盖配置文件
    args = get_args()
    cfg = py2cfg(args.config_path)
    if args.overload_cfgs_path is not None:
        for c_path in args.overload_cfgs_path:
            other_cfg = py2cfg(c_path)
            cfg.update(other_cfg)


    global_dict['args'] = args
    global_dict['cfg'] = cfg

    #? 配置文件初始化
    pretreat(args, cfg)
    overall.init_device_arguments(args, cfg)
    overall.complete_cfg(args, cfg)
    overall.resume_and_pretrain(args, cfg)


    #? 扩展钩子： logger, 'log-model.txt'的写入
    loggercallback = mylog.LoggerCallback(create_logger(args, cfg))
    #? 扩展钩子： CkptManager, 存权重文件
    ckptm = tn.CkptManager.from_config(cfg)

    cfg.trainer_callbacks.extend([ckptm, loggercallback])

    file2 = open_outputs_log(cfg)
    cfg.trainer_log_files = [file2]
    if not args.quiet:
        cfg.trainer_log_files.append(sys.stdout)
    cfg.skip_train = args.skip_train

    #? Trainer 管理模型的训练流程
    trainer = cfg.trainer = tn.Model_Trainer.from_config(cfg)
    

    #? 向CkptManager注册 需要保存到ckpt文件的项目，以及每次resume时需要加载它们
    # 训练过程中每轮产生的数据
    ckptm.register_save_load('data_list', trainer.save_datas, trainer.load_datas) 
    ckptm.register_save_load('logger_topk', loggercallback.savetopk)

    def save_description():
        return cfg.description
    if availabe(cfg, "description"):
        ckptm.register_save_load('description', save_func=save_description)

    #? 保存seed
    cfg.seed = overall.seed_everything(args.seed, do_init=False)
    def save_seed():
        return cfg.seed

    def load_seed(seed):
        if seed is None:
            seed = cfg.seed
        cfg.seed = overall.seed_everything(seed, do_init=True)
    if not cfg.do_resume:
        overall.seed_everything(cfg.seed)
    ckptm.register_save_load("seed", save_seed, load_seed)

    
    shell_dicts(args, cfg)
    
    #? 向log-model.txt 写入各种信息
    overall.information(args, cfg)

    #? copy config.py file to training dir
    cfgname = os.path.split(args.config_path)[1]
    cpfile = os.path.join(cfg.workdir, cfgname)
    # cpfile = os.path.join(cfg.workdir_model, cfgname)
    if not os.path.exists(cpfile):
        shutil.copyfile(args.config_path, cpfile)
    print("Work Dir: ", cfg.workdir)

    global_dict['trainer'] = trainer

    trainer.run()  #? Important ☆☆☆ <--- run() here  开始训练!!!


    #? 训练完，改训练目录为 'final'
    if "donot_rename_workdir" not in cfg or not cfg["donot_rename_workdir"]:
        final_dir = f'{cfg.workdir_model}/final{cfg.version_str}'
        # if os.path.islink(final_dir):
        #     os.remove(final_dir)
        # os.symlink(cfg.workdir, final_dir)
        if os.path.exists(final_dir):
            v = Version.copyed_max_version(cfg.workdir_model, 'final')
            os.rename(final_dir, final_dir + f'({v + 1})')

        os.rename(cfg.workdir, final_dir)

    # 弃用
    if args.do_test:
        testpy = os.path.join(path0, 'test_segment.py')
        del cfg.model
        del cfg
        for i in range(5):
            torch.cuda.empty_cache()
        os.system(f'python {testpy} -c {args.config_path} --workdir {args.workdir_root} &')
        print("test process begin !")
        exit()




if __name__ == "__main__":
    train_model()
