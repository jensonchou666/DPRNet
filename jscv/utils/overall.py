import torch
from torch.utils.data import DataLoader
import random

from .utils import *

from .logger import Logger
from .load_checkpoint import load_checkpoint
import os


# class GlobalEnv:
#     training = False


# env = GlobalEnv()

global_dict = dict(training=False)



def seed_everything(seed, do_init=True):
    if seed == "time":
        seed = int(time.time())
    elif seed == "random":
        seed = random.randint(10, 100000)
    elif isinstance(seed, str):
        seed = int(seed)
    else:
        assert isinstance(seed, int)
    if do_init:
        print("seed:", seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    return seed




def standard_workdir_model(args, cfg):
    if availabe(cfg, "workdir_model"):
        return
    cfg.workdir_root = args.workdir

    name_md_e = cfg.model_name + '-e' + str(cfg.max_epoch)
    cfg.workdir_model = os.path.join(cfg.workdir_root, cfg.dataset_name, name_md_e)


def format_ckptn_metrix(cfg):
    # if availabe(cfg, "ckptname"):
    #     return

    # if not availabe(cfg, "ckpt_split_char"):
    #     cfg.ckpt_split_char = '@'
    if not availabe(cfg, "ckptname_metrix"):
        cfg.ckptname_metrix = ['epoch={epoch}']

    cm2 = []
    cfgdict = dict(cfg)
    for m in cfg.ckptname_metrix:
        cm2.append(format_if_in_dict(m, cfgdict))
    cfg.ckptname_metrix = cm2


# def create_train_loader(dataset,
#                         batch_size=1,
#                         num_workers=4,
#                         pin_memory=True,
#                         shuffle=True,
#                         ):
#     return DataLoader(dataset=dataset,
#                       batch_size=batch_size,
#                       num_workers=num_workers,
#                       pin_memory=pin_memory,
#                       shuffle=True,
#                       drop_last=True)


# def create_val_loader(dataset,
#                       batch_size=1,
#                       num_workers=4,
#                       pin_memory=True,):
#     return DataLoader(dataset=dataset,
#                       batch_size=batch_size,
#                       num_workers=num_workers,
#                       pin_memory=pin_memory,
#                       shuffle=False,
#                       drop_last=False)


def init_device_arguments(args, cfg):

    if args.cpu:
        cfg.use_cpu = True

    if hasattr(cfg, 'use_cpu') and cfg.use_cpu is True:
        cfg.device_arguments = {'accelerator': 'cpu'}
    else:
        cfg.use_cpu = False
        cfg.device_arguments = {'accelerator': 'gpu'}
        if args.strategy is not None:
            cfg.strategy = args.strategy
        if not availabe(cfg, 'strategy'):
            cfg.strategy = 'ddp'
        if args.gpus is not None:
            cfg.gpus = args.gpus
        elif args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids

        if not hasattr(cfg, 'gpus') and not hasattr(cfg, 'gpu_ids'):
            cfg.gpus = 1

        if hasattr(cfg, 'gpus'):
            if cfg.gpus == 1:
                cfg.strategy = None
            cfg.device_arguments.update({
                'gpus': cfg.gpus,
                'strategy': cfg.strategy
            })
        elif hasattr(cfg, 'gpu_ids'):
            assert type(cfg.gpu_ids) is list
            if len(cfg.gpu_ids) == 1:
                cfg.strategy = None
            cfg.device_arguments.update({
                'devices': cfg.gpu_ids,
                'strategy': cfg.strategy
            })

from numbers import Number
def size_elems(size):
    if size is None:
        return None
    elif isinstance(size, Number):
        return size * size
    else:
        h, w = size
        return h * w

def get_batchsize_by_crop_size(batchsize_dict: dict, crop_size):

    sz = size_elems(crop_size)
    if sz is None:
        return batchsize_dict.get(None, 1)
    elif sz in batchsize_dict:
        return batchsize_dict[sz]
    else:
        maxk, maxv = None, None
        for k, v in batchsize_dict.items():
            if k is not None and k > sz and (maxk is None or k < maxk):
                maxk, maxv = k, v
        return maxv


from .analyser import set_rgb, set_rgb_dict

def before_cfg_create(args, cfg):
    if args.max_epoch is not None:
        cfg.max_epoch = args.max_epoch

    if args.auto_batch_size:
        #TODO
        cfg.auto_batch_size = True
    elif args.batch_size is not None:
        print("@1")
        cfg.auto_batch_size = False
        if (args.batch_size.isdigit()):
            cfg.train_batch_size = cfg.val_batch_size = int(args.batch_size)
        else:
            t, v = args.batch_size.split(',')
            cfg.train_batch_size, cfg.val_batch_size = int(t), int(v)

        if availabe(cfg, 'train_loader'):
            cfg.train_loader.batch_size = args.batch_size
        if availabe(cfg, 'val_loader'):
            cfg.val_loader.batch_size = args.batch_size
    
    elif not hasattr(cfg, "batch_size") and not hasattr(cfg, "train_batch_size"):

        #TODO val_batchsize 如何定？？？
        assert hasattr(cfg, "train_batchsize_dict")
        d2 = cfg.train_batchsize_dict
        if cfg.dataset_name in d2:
            d2 = d2[cfg.dataset_name]
        tbz = get_batchsize_by_crop_size(d2, cfg.train_crop_size)
        cfg.train_batch_size = tbz
        if hasattr(cfg, "val_batchsize_dict"):
            d2 = cfg.val_batchsize_dict
            if cfg.dataset_name in d2:
                d2 = d2[cfg.dataset_name]
            vbz = get_batchsize_by_crop_size(d2, cfg.val_crop_size)
            cfg.val_batch_size = vbz
        else:
            cfg.val_batch_size = tbz

    if not hasattr(cfg, 'auto_batch_size'):
        cfg.auto_batch_size = False

    #if not cfg.auto_batch_size:
    if not availabe(cfg, 'train_batch_size'):
        if availabe(cfg, 'batch_size'):
            cfg.train_batch_size = cfg.batch_size
        else:
            cfg.train_batch_size = 1


    if "palette" in cfg:
        set_rgb(cfg.palette)
    elif "rgb_dict" in cfg:
        set_rgb_dict(cfg.rgb_dict)


def before_cfg_create_test(args, cfg):
    if "train_batchsize_dict" in cfg:
        d2 = cfg.train_batchsize_dict
        if cfg.dataset_name in d2:
            d2 = d2[cfg.dataset_name]
        tbz = get_batchsize_by_crop_size(d2, cfg.train_crop_size)
        cfg.train_batch_size = tbz
    elif "train_batch_size" not in cfg:
        cfg.train_batch_size = 1


def complete_cfg(args, cfg):

    if not cfg.auto_batch_size:

        if availabe(cfg, 'train_loader'):
            cfg.train_batch_size = cfg.train_loader.batch_size
        else:
            cfg.train_loader = DataLoader(cfg.train_dataset,
                                          cfg.train_batch_size,
                                          shuffle=cfg.get("train_loader_shuffle", True),
                                          pin_memory=cfg.get("train_loader_pin_memory", True),
                                          num_workers=cfg.get("train_loader_num_workers", 4),
                                          drop_last=True
                                          )
            # print("train_loader drop_last=True")

        if not availabe(cfg, 'val_batch_size'):
            cfg.val_batch_size = cfg.train_batch_size

        if availabe(cfg, 'val_loader'):
            cfg.val_batch_size = cfg.val_loader.batch_size
        else:
            cfg.val_loader = DataLoader(cfg.val_dataset,
                                        cfg.val_batch_size,
                                        shuffle=cfg.get("val_loader_shuffle", False),
                                        pin_memory=cfg.get("val_loader_pin_memory", True),
                                        num_workers=cfg.get("val_loader_num_workers", 4),
                                        drop_last=False
                                        )



    #cfg.train_img_size = next(iter(cfg.train_loader))['img'].shape
    cfg.train_img_size = cfg.train_dataset[0]['img'].shape
    cfg.val_img_size = cfg.val_dataset[0]['img'].shape

    standard_workdir_model(args, cfg)
    format_ckptn_metrix(cfg)

    if availabe(cfg, 'log_to_infofile') and cfg.log_to_infofile:
        if not availabe(cfg, 'info_filename'):
            cfg.info_filename = f'log-{cfg.model_name}-{cfg.dataset_name}.txt'
        if not availabe(cfg, 'log_filename'):
            cfg.log_filename = cfg.info_filename
    else:
        if not availabe(cfg, 'info_filename'):
            cfg.info_filename = f'info-{cfg.model_name}-{cfg.dataset_name}.txt'
        if not availabe(cfg, 'log_filename'):
            cfg.log_filename = f'log-{cfg.model_name}-{cfg.dataset_name}.txt'


def resume_and_pretrain(args, cfg):
    # 智能生成cfg.path_resume_from 、 统一的pretrain加载权重方法
    #
    _resume_from = 'last/last'
    version = None
    log_prefix = ""

    if not availabe(cfg, "pretrain_dict"):
        cfg.pretrain_dict = {}

    if args.resume_from is not None:
        cfg.do_resume = True
        if args.resume_from != '.':
            _resume_from = args.resume_from

    if availabe(cfg, 'do_resume') and cfg.do_resume:
        if len(_resume_from.split('/')) > 2:
            cfg.workdir, ckpt_name = os.path.split(_resume_from)
            cfg.version_name = cfg.workdir.split('/')[-1]
            # print("@", cfg.workdir, ckpt_name, cfg.version_name)
        else:
            version, ckpt_name = _resume_from.split('/')
            if version == 'last':
                vs = Version.last_version(cfg.workdir_model)
                assert vs >= 0, 'No lastest version_0 dir'
                version = f'version_{vs}'
                # _logger_temp = CSVLogger(total_save_dir='', name=cfg.workdir_model)
                #version = _logger_temp.version - 1
                #version = f'version_{version}'
            elif version.isdigit():
                version = f'version_{version}'
            cfg.version_name = version
            cfg.workdir = os.path.join(cfg.workdir_model, version)

        if ckpt_name.isdigit():
            cfg.resume_epoch = int(ckpt_name)
            assert '{epoch}' in cfg.ckptname_metrix[0], '??? what are you doing ???'
            s = cfg.ckptname_metrix[0].index('{epoch}')
            s1 = cfg.ckptname_metrix[0][:s]
            s1 = f'{s1}{ckpt_name}'
            ckpt_name = None
            for path in os.listdir(cfg.workdir):
                if path.startswith(s1):
                    ckpt_name = path
            if ckpt_name is None:
                raise FileNotFoundError(f"找不到{cfg.workdir}/{path}的ckpt文件")
        #只支持ckpt文件
        if not ckpt_name.endswith(
                '.ckpt'):  # and not ckpt_name.endswith('.pt'):
            ckpt_name += '.ckpt'

        if ckpt_name == 'last.ckpt':
            cfg.resume_epoch = 'last'
        else:
            csc = cfg.ckpt_split_char
            epoch = ckpt_name.split(csc)[0].split('=')[-1]
            cfg.resume_epoch = int(epoch)

        cfg.resume_ckpt_name = ckpt_name
        cfg.path_resume_from = os.path.join(cfg.workdir, ckpt_name)

        # maxi = -1
        # for path in os.listdir(workdir):
        #     if path.startswith('metrics-old'):
        #         i = path[len('metrics-old'):-len('.csv')]
        #         maxi = max(maxi, int(i))
        # fmetrics = os.path.join(workdir, 'metrics.csv')
        # if maxi == -1 and os.path.exists(fmetrics):
        #     maxi = 0
        # if maxi != -1:
        #     maxi += 1
        #     f2 = os.path.join(workdir, f'metrics-old{maxi}.csv')
        #     os.system(f'cp {fmetrics} {f2}')

    else:
        cfg.do_resume = False
        cfg.resume_ckpt_name = None
        cfg.path_resume_from = None

        # pretrain:
        if args.pretrain_path is not None:
            if args.pretrain_path.lower() == 'none':
                cfg.skip_pretrain = True
            else:
                cfg.skip_pretrain = False
                cfg.backbone_ckpt_path = args.pretrain_path
        if args.pretrain_prefix is not None:
            cfg.backbone_prefix = args.pretrain_prefix

        if availabe(cfg, 'skip_pretrain') and cfg.skip_pretrain:
            return

        # if not availabe(cfg, "pretrain_dict"):
        #     cfg.pretrain_dict = {}
        ptd = cfg.pretrain_dict

        if availabe(cfg, "backbone_ckpt_dict"):
            ptd[cfg.backbone_ckpt_dict] = [(cfg.model.backbone,
                                            cfg.backbone_prefix)]
        elif availabe(cfg, "backbone_ckpt_path"):
            ptd[cfg.backbone_ckpt_path] = [(cfg.model.backbone,
                                            cfg.backbone_prefix)]
        # Load pretrain
        for ckpt, Mlist in cfg.pretrain_dict.items():
            for model, prefix in Mlist:
                opt = None
                if hasattr(model, 'operate_on_pretrain'):
                    opt = model.operate_on_pretrain
                if hasattr(model, 'on_pretrain'):
                    opt = model.on_pretrain
                if hasattr(model, 'on_load_ckpt_dict'):
                    opt = model.on_load_ckpt_dict
                load_checkpoint(model=model,
                                checkpoint=ckpt,
                                prefix=prefix,
                                operate_for_dict=opt,
                                config=cfg)
    if not os.path.exists(cfg.workdir_model):
        os.makedirs(cfg.workdir_model)



import sys, time
import pynvml

from .statistics import StatisticScale


def _information_1(cfg, file1):
    t0 = time.localtime(time.time())

    file1.write('{} - {}-{} - INFO\n'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', t0), cfg.model_name,
        cfg.dataset_name))
    _cmd = "python"
    for s in sys.argv:
        _cmd += ' ' + s
    file1.write(f'command:   {_cmd}\n')
    if cfg.use_cpu:
        file1.write('device:    CPU\n')
    else:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        file1.write(f'device:    {gpu_name}\n')
        _ag = cfg.device_arguments.copy()
        _ag.pop('accelerator')
        file1.write('CUDA_VISIBLE_DEVICES={}, use:{}\n'.format(
            os.environ.get('CUDA_VISIBLE_DEVICES', None), _ag))
    file1.write(
        f'max_epoch: {cfg.max_epoch}, batach_size: {cfg.train_batch_size},' +
        f'{cfg.val_batch_size}\n')

    len_t = len(cfg.train_dataset)
    len_v = len(cfg.val_dataset)
    file1.write(f'Total: {len_t} train images, {len_v} val images\n')


def information(args, cfg):
    #TODO copy config file
    # shutil.copy(args.config_path, cfg.logdir)

    str_info_end = '*' * 40 + ' info end ' + '*' * 40

    info_path = os.path.join(cfg.workdir, cfg.info_filename)


    if cfg.do_resume:
        
        # with open(info_path, 'r') as file1:
        #     while True:
        #         line = file1.readline()
        #         if line.startswith(seed_prestr):
        #             PRE = len(seed_prestr)
        #             AFT = line.find("\n", PRE)
        #             seed = int(line[PRE:AFT])
        #             break
        #         elif line == "":
        #             break
        # assert seed is not None, "info_file does not have seed-info"
        # seed_everything(seed)

        return
    else:

        with open(info_path, 'w') as file_info:
            _information_1(cfg, file_info)

            file_info.write('Pretrain: ')
            if cfg.skip_pretrain and not hasattr(cfg, 'pretrain_type'):
                file_info.write('None\n')
            if not cfg.skip_pretrain:
                d2 = {}
                for ckpt, v in cfg.pretrain_dict.items():
                    if isinstance(ckpt, dict):
                        ckpt = type(ckpt)
                    d2[ckpt] = []
                    for (m, p) in v:
                        d2[ckpt].append((type(m), p))
                file_info.write(f'{d2}\n')
            if hasattr(cfg, 'pretrain_type'):
                file_info.write(f' {cfg.pretrain_type}\n')

            file_info.write(f"seed: {cfg.seed}\n")

            if availabe(cfg, 'description'):
                file_info.write(f'description: {cfg.description}\n')
            if availabe(cfg, 'detail_desc'):
                file_info.write(f'detail_desc:\n{cfg.detail_desc}\n')


            model = cfg.model

            #print("Before stat:  len(model.state_dict())=", len(cfg.model.state_dict()))

            #shape = [2, *list(cfg.val_dataset[0]['img'].shape)]
            shape = [2, *list(cfg.train_dataset[0]['img'].shape)]
            
            # test once
            wrong = None
            
            
            # with redirect(file_info):
            #     #??  这里stat 必须深拷贝一份net， 否则直接在net上stat就无法加载权重
            #     try:
            #         StatisticScale.stat(model, input=shape, deepcopy_model=True)
            #         print(f'\n{str_info_end}\n')
            #     except Exception as e:
            #         wrong = e
            #         print("Cant Statistic:", e)

            if wrong is not None:
                print("Cant Statistic:", wrong)

            #print("After stat:  len(model.state_dict())=", len(cfg.model.state_dict()))

    if availabe(cfg, 'description'):
        description_file = os.path.join(cfg.workdir, 'D: ' + cfg.description)
        with open(description_file, 'w'):
            pass

