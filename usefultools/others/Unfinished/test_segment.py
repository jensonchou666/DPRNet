#! /opt/conda/bin/python

import os, sys
# import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from thop.profile import profile
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from jscv.utils.cfg import py2cfg
from jscv.utils import overall
from jscv.utils.load_checkpoint import load_checkpoint
from jscv.utils.utils import set_default, redirect, TimeCounter, warmup
from jscv.utils.metric import SegEvaluator
from jscv.utils.table import MatrixTable
import jscv.utils.analyser as analyser

#TODO 解决中途出错不保存database的问题
#TODO 选择参数： 保留之前的推理时间



img_writer_to_tif = False
img_writer_num_classes = -1
img_writer_rgb_dict = analyser.cfg.palette
test_dataset = None

def img_writer(input):

    bg_index = 0

    global img_writer_to_tif, img_writer_num_classes, img_writer_rgb_dict
    mask_org, masks_true_org, file_path_prefix, imgid, subdir = input


    if img_writer_to_tif:
        path = file_path_prefix + '.tif'
        mask = mask_org.astype(np.uint8)
        cv2.imwrite(path, mask)
    else:
        path_mask = file_path_prefix + '.png'
        path_mask_true = file_path_prefix + '(true).png'
        path_overlap = file_path_prefix + '(overlap).png'
        path_wrong = file_path_prefix + '(wrong).png'

        mask = analyser.label2rgb(mask_org, img_writer_num_classes, img_writer_rgb_dict)
        masks_true = analyser.label2rgb(masks_true_org, img_writer_num_classes, img_writer_rgb_dict)

        wrong = analyser.mask_lable_wrong_map(
            mask_org, masks_true_org, img_writer_num_classes, masks_true, bg_index, img_writer_rgb_dict)

        cv2.imwrite(path_mask, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(path_mask_true, cv2.cvtColor(masks_true, cv2.COLOR_RGB2BGR))
        cv2.imwrite(path_wrong, cv2.cvtColor(wrong, cv2.COLOR_RGB2BGR))
        
        if test_dataset is not None:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            path = test_dataset.get_image_path(imgid, subdir)
            overlap = cv2.addWeighted(cv2.imread(path), 0.8, mask, 0.2, 0)
            cv2.imwrite(path_overlap, overlap)
    


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=Path, help='configure, file or dir')
    parser.add_argument("-c", "--overload_cfgs_path",
                        nargs='+',
                        type=Path,
                        help="Path to the overload configs, if need.")

    parser.add_argument("-g", "--gpu-id", type=int, default=0, help='gpu id')
    parser.add_argument("-d", "--database",
                        type=Path,
                        help='path of database(optional)')

    parser.add_argument("--bz", type=int, default=1, help='test-batch-size')


    # group_cfg = parser.add_mutually_exclusive_group()
    # group_cfg.add_argument("-d",
    #                        "--config-dir",
    #                        type=Path,
    #                        help='Dir of multiple config files(.py), traverse')
    # group_cfg.add_argument("-c",
    #                        "--config-path",
    #                        type=Path,
    #                        help="Path to one of config file")


    arg = parser.add_argument
    arg("-v", "--version", default='all',
        help="version: final({version})   0: final,  or: all(defualt)")

    arg("--workdir",
        type=Path,
        default=Path('work_dir'),
        help="Path where to save resulting masks.")
    # arg("--out", "--output-path", type=Path, default=Path('work_dir'), help="Path where to save resulting masks.")
    # lr is flip TTA, d4 is multi-scale TTA
    # arg("-t", "--tta", help="Test time augmentation.", default="d4", choices=["d4", "lr"])
    # arg("--nockpt-skip", default=False, action='store_true', help='if no final dir or no suit ckpt, skip')
    arg("-f",
        "--force",
        default=False,
        action='store_true',
        help='force test, no skip')
    #TODO 默认保存预测图还是默认不保存？
    arg("-i",
        "--imgs",
        default=False,
        action='store_true',
        help='output pred-mask-images')
    arg("--ckpt-topk", default=1, type=int, help="topk ckpt to load")
    arg("-p",
        "--repeat",
        default=1,
        type=int,
        help="repeat times to reduce deviation")
    #TODO repeat 第一次测试要加载数据到cuda，所以比后面的测试更耗时， repeat > 1 则不准
    arg("--no-eval", default=False, action='store_true')

    arg("--sort-monitor", default='test_OA',
        help="sort-monitor mode: max")  # 用不上
    arg("--no-std",
        default=False,
        action='store_true',
        help='std flops and params')
    arg("--test-last", default=False, action='store_true')
    arg("--profile-cpu", default=False, action='store_true')
    arg("--tif",
        help="whether output tif masks",
        default=False,
        action='store_true')
    # arg("--val", help="whether eval validation set if test-dataset no mask", default=True, action='store_true')

    arg('-b', "--max_batchs", default=200)
    arg( "--label", default='gt_semantic_seg', help="gt_semantic_seg or true_pha")

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_num(str0: str, prefix: str):
    try:
        i = str0.index(prefix) + len(prefix)
    except ValueError:
        return None
    for j in range(i, len(str0)):
        if str0[j] != '.' and not str0[j].isdigit():
            break
    return float(str0[i:j])


def get_ckpt(workdir, cfg):
    monitor_prefix = cfg.ckptname_monitor_prefix
    files_list = os.listdir(workdir)
    ckpt_list = []
    last = None
    for f in files_list:
        if f == 'last.ckpt':
            last = f
        elif f.endswith('.ckpt'):
            ckpt_list.append(f)
    # ckpt_list = ckpt_list[cfg.ontest_loadckpt_topk]

    def first(elem):
        return elem[0]

    list2 = []

    for n in ckpt_list:
        monitor_prefix = monitor_prefix.replace('monitor',
                                                cfg.save_ckpt_monitor)
        v = extract_num(n, monitor_prefix)
        if v is not None:
            list2.append((v, n))
    reverse = True if cfg.save_ckpt_monitor_mode == 'max' else False
    list2.sort(key=first, reverse=reverse)
    # ckpt_list = []
    # for v, name in list2[:cfg.ontest_loadckpt_topk]:
    #     assert '{epoch}' in cfg.ckptname_metrix[0]
    #     s = cfg.ckptname_metrix[0].index('{epoch}')
    #     epoch_prefix = cfg.ckptname_metrix[0][:s]
    #     e = extract_num(name, epoch_prefix)
    #     # assert e is not None
    #     ckpt_list.append((e, v , name))
    list2 = list2[:cfg.ontest_loadckpt_topk]
    return [j for i, j in list2], last


class time_spend:

    def __init__(self, msg, msg2=''):
        self.msg = msg
        self.msg2 = msg2

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.msg:<20}", time.time() - self.t0, self.msg2)


class time_count:

    def __init__(self, counter):
        self.c = counter

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.c.time += time.time() - self.t0


class Counter:
    time = float(0.0)

    def count(self):
        return time_count(self)

    def __str__(self) -> str:
        return str(self.time)


def warm_up(model, x, epoachs=20):
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(epoachs):
            model(x)
    torch.cuda.synchronize()


def database_table():
    # Log
    tb = MatrixTable()
    tb.register_columns({
        'model': {
            'title': 'Model',
        },
        # 'speed': {
        #     'title': 'Inference-Speed',
        #     'format': '{:.4f}'
        # },
        'latency': {
            'title': 'Latency(ms)',
            'format': '{:.2f}'
        },
        'top1_epoch': {
            'title': 'T1',
            'interval': 1
        },
        'top1_mIoU': {
            'title': 'Top1-mIoU',
            'format': '{:.4f}'
        },
        'top1_F1': {
            'title': 'Top1-F1',
            'format': '{:.4f}'
        },
        'top1_OA': {
            'title': 'Top1-OA',
            'format': '{:.4f}'
        },
        'flops': {
            'title': 'Flops(G)',
            'precision': 2,
            'unit': 'G',
            'unit_show': False,
        },
        'params': {
            'title': 'Params(M)',
            'interval': 0,
            'precision': 2,
            'unit': 'M',
            'unit_show': False,
        },
        'input_shape': {
            'title': 'input_shape',
            'interval': 2
        },
        'description': {
            'title': 'description',
            'interval': 1
        }
    })
    tb.show_syle['line_len_extra'] = 2
    tb.set_align('center')
    tb.columns_info['model']['align'] = 'left'
    return tb


def main():
    global img_writer_to_tif, img_writer_num_classes, img_writer_rgb_dict, test_dataset

    args = get_args()
    torch.cuda.set_device(args.gpu_id)

    # seed_everything(42)

    img_writer_to_tif = args.tif

    cfgs_name = []

    assert isinstance(args.config, Path)

    if args.config == Path('/'):
        #仅展示图表
        print('@')
    elif args.config.is_dir():
        assert args.version == 'all' or args.version == '0', "dir, version>0 is not good"
        for root, dirs, files in os.walk(args.config):
            for f in files:
                if f.endswith('.py'):
                    # path = os.path.join(root, f)
                    cfgs_name.append((os.path.join(root, f), f))
    elif args.config.is_file():
        f = os.path.split(args.config)[-1]
        cfgs_name.append((args.config, f))
    else:
        raise Exception("Wrong type")

    if args.database is None:
        test_database_path = os.path.join(args.workdir, 'test.database')
    else:
        test_database_path = args.database
    if os.path.exists(test_database_path):
        test_history = torch.load(test_database_path)
    else:
        test_history = {}
    '''
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
    '''
    lfinal = len('final')

    cfgs_all = []

    extra_cfgs = []
    if args.overload_cfgs_path is not None:
        for c_path in args.overload_cfgs_path:
            other_cfg = py2cfg(c_path)
            extra_cfgs.append(other_cfg)


    for cfg_path, cfg_name in cfgs_name:
        cfg = py2cfg(cfg_path)

        for ecfg in extra_cfgs:
            cfg.update(ecfg)

        overall.standard_workdir_model(args, cfg)

        if args.no_eval:
            cfgs_all.append((cfg_path, cfg_name, f'final(no_eval_{args.version})'))
            continue
        elif not os.path.exists(cfg.workdir_model):
            cfgs_all.append((cfg_path, cfg_name, 'final'))
            continue

        files = os.listdir(cfg.workdir_model)

        appended = False
        for f in files:
            if f.startswith('final'):
                if args.version == 'all' or \
                    (args.version == '0' and f == 'final') or \
                        (f[lfinal + 1: -1] == args.version):
                    cfgs_all.append((cfg_path, cfg_name, f))
                    appended = True
        if not appended:
            cfgs_all.append((cfg_path, cfg_name, 'final'))

    global_dict = overall.global_dict
    global_dict['training'] = False
    global_dict['run_mode'] = 'test'
    global_dict['args'] = args

    for cfg_path, cfg_name, final in cfgs_all:
        # try:
        cfg = py2cfg(cfg_path)
        for ecfg in extra_cfgs:
            cfg.update(ecfg)

        global_dict['cfg'] = cfg
        # except Exception:
        #     print(f'Error: py2cfg({cfg_path}), skiped')
        #     continue
        overall.standard_workdir_model(args, cfg)

        overall.before_cfg_create_test(args, cfg)
        cfg.create_dataset(cfg)
        cfg.create_model(cfg)

        set_default(
            cfg,
            {
                "ckptname_metrix": ['epoch={epoch}'],
                "ontest_loadckpt_topk": args.ckpt_topk,
                "ontest_repeat": args.repeat,
                'ckptname_monitor_prefix': cfg.save_ckpt_monitor + '=',
                'CutOffDatasets': [],
                'dataset_rgb_dict': analyser.cfg.palette,
                'num_classes': 1
                # "test_dataset_exist_mask": False
            })

        img_writer_num_classes = cfg.num_classes
        img_writer_rgb_dict = cfg.dataset_rgb_dict

        name_model = cfg.model_name + '-e' + str(cfg.max_epoch) + final[lfinal:]
        if cfg.dataset_name not in test_history:
            test_history[cfg.dataset_name] = {}
        dict_dataset = test_history[cfg.dataset_name]
        if name_model not in dict_dataset:
            dict_dataset[name_model] = {}
            dict_dataset[name_model]['test_state'] = 'empty'
        dict_this = dict_dataset[name_model]

        test_list = []
        from_ckpt = False
        workdir = os.path.join(cfg.workdir_model, final)
        if os.path.exists(workdir):
            # test_logfile = os.path.join(workdir, 'test.log')

            ckpt_list, lastckpt = get_ckpt(workdir, cfg)
            # test_list = [(x, args.repeat) for x in ckpt_list]
            test_list = ckpt_list
            if args.test_last and lastckpt is not None:
                test_list.append(lastckpt)
        else:
            workdir = cfg.workdir_model
            #os.makedirs(workdir)

        re_test = os.path.join(workdir, 'retest')
        if not args.force:
            if args.config.is_file():
                pass
            elif not os.path.exists(re_test):
                if dict_this['test_state'] == 'completed':
                    continue
                elif dict_this['test_state'] == 'inftime':
                    if len(test_list) == 0:
                        continue
            else:
                os.remove(re_test)

        dict_dataset[name_model] = {}
        dict_this = dict_dataset[name_model]

        if len(test_list) == 0:
            test_list = [None]
            dict_this['test_state'] = 'inftime'
            do_resume = False
        else:
            dict_this['test_state'] = 'completed'
            do_resume = True

        # net = cfg.net
        # assert isinstance(net, nn.Module)

        # net = tta.SegmentationTTAWrapper(net, transforms)

        dict_this['test_datalist'] = []
        test_datalist = dict_this['test_datalist']

        evaluator = None
        if not do_resume:
            do_eval = False
        elif 'gt_semantic_seg' in cfg.test_dataset[0]:
            evaluator = SegEvaluator(num_class=cfg.num_classes)
            evaluator.reset()
            do_eval = True
        else:
            do_eval = False

        if do_resume:
            outmask_isfirst = True
        else:
            outmask_isfirst = False

        if args.no_eval:
            do_eval = False
            outmask_isfirst = False

        if "test_dataset" not in cfg:
            cfg.test_dataset = cfg.val_dataset

        test_dataset = cfg.test_dataset
        batch_size = args.bz
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        dict_this['  '] = len(test_dataset)
        # dict_this['test_tta'] = args.tta

        outmask_dir = os.path.join(workdir, 'output_masks')

        speeds = []
        evals = []

        profile_input_shape = (args.bz, *test_dataset[0]['img'].shape)

        def do_profile(model):
            try:
                if args.profile_cpu:
                    _profile_input_ = torch.randn(*profile_input_shape).cpu()
                    _net2_ = copy.deepcopy(model).cpu()
                else:
                    _profile_input_ = torch.randn(*profile_input_shape).cuda()
                    _net2_ = copy.deepcopy(model).cuda()
                with redirect(open(os.devnull, 'w')):
                    flops, params = profile(_net2_, (_profile_input_, ))
                del _profile_input_, _net2_
                torch.cuda.empty_cache()
            except Exception as e:
                print("Cant Statistic:", e)
                return 0, 0
            return flops, params

        if hasattr(cfg, 'model'):
            model = cfg.model
        else:
            model = cfg.net
        
        print("workdir:", workdir)

        #print("@", "model.cuda() begin")
        model = model.cuda().eval()
        #print("@", "model.cuda() end")

        desc = None
        none_list = [None] * batch_size

        with torch.no_grad():
            for ckpt in test_list:
                ckptdict = {}
                if ckpt is not None:
                    ckptdict = torch.load(os.path.join(workdir, ckpt), map_location='cpu')

                    if 'description' in ckptdict:
                        desc = ckptdict['description']

                    if 'model' in ckptdict:
                        del model
                        model = ckptdict['model'].cuda().eval()
                        print("Load whole model-class")
                    else:
                        load_checkpoint(model, ckptdict['state_dict'])
                        print("Load state_dict only")

                # warm up...
                if not args.profile_cpu:
                    warmup(20)
                    # warm_up(model, torch.randn(*profile_input_shape).cuda())

                for idx0 in range(args.repeat):
                    print(
                        f'cfg: {cfg_name}, resume: {ckpt}  {idx0 + 1}/{args.repeat}'
                    )

                    outmask_results = []
                    if do_eval:
                        evaluator.reset()
                    
                    counter = TimeCounter(True)
                    
                    num_batchs = min(len(test_loader), int(args.max_batchs))
                    for idx, batch in enumerate(tqdm(test_loader, total=num_batchs)):
                        if idx >= num_batchs:
                            break
                        if idx == 0:
                            t_idx_1 = time.time()
 
                        img = batch['img'].cuda()

                        counter.begin()
                        pred = model(img)["pred"]
                        counter.record_time(last=True)

                        if isinstance(pred, tuple):
                            pred = pred[0]
                        pred = nn.Softmax(dim=1)(pred).argmax(dim=1)

                        predcpu = pred.cpu().numpy()
                        masks_true = batch[args.label].cpu().numpy()
                        if idx == 0:
                            t_idx_1 = time.time() - t_idx_1

                        if do_eval or outmask_isfirst:
                            # with ceval.count():
                            ids = batch["id"]
                            subdir = batch.get("subdir", none_list)
                            # imgs_np = batch['img'].permute(0, 2, 3, 1).numpy()
                            for i in range(predcpu.shape[0]):
                                if outmask_isfirst:
                                    mask_path = os.path.join(outmask_dir, ids[i])
                                    outmask_results.append(
                                        (predcpu[i], masks_true[i], mask_path, ids[i], subdir[i]))
                                if do_eval:
                                    evaluator.add_batch(masks_true[i], predcpu[i])
                    torch.cuda.synchronize()
                    spend = counter.TimeList[0] / 1000
                    speed = num_batchs * batch_size / spend
                    print("speed", speed)
                    speeds.append(speed)
                    test_data = {
                        # 'inference_spend': spend,
                        'inference_speed': speed,
                        'load_from': ckpt,
                        'repeat_index': idx0
                    }
                    if ckpt is not None:
                        load_info = test_data['ckpt_info'] = {}
                        load_info['epoch'] = ckptdict['epoch']
                        #把batchs删了 太大
                        if 'data_list' in ckptdict:
                            for e in ckptdict['data_list']:
                                if 'batchs' in e:
                                    e.pop('batchs')
                            load_info['data_list'] = ckptdict['data_list']
                    if do_eval:
                        eval_result = test_data
                        #eval_result = test_data['test_eval_result'] = {}
                        iou_per_class = evaluator.Intersection_over_Union()
                        f1_per_class = evaluator.F1()
                        oa = evaluator.OA() * 100
                        if cfg.dataset_name in cfg.CutOffDatasets:
                            iou = np.nanmean(iou_per_class[:-1]) * 100
                            f1 = np.nanmean(f1_per_class[:-1]) * 100
                        else:
                            iou = np.nanmean(iou_per_class) * 100
                            f1 = np.nanmean(f1_per_class) * 100
                        eval_result['test_mIoU'] = iou
                        eval_result['test_F1'] = f1
                        eval_result['test_OA'] = oa
                        # ckpt not None
                        class_IoU = eval_result['class_IoU'] = {}
                        class_F1 = eval_result['class_F1'] = {}

                        for class_name, class_iou, class_f1 in zip(
                                cfg.classes, iou_per_class, f1_per_class):
                            class_IoU[class_name] = class_iou * 100
                            class_F1[class_name] = class_f1 * 100
                        evals.append((iou, f1, oa, class_IoU, class_F1, ckptdict['epoch']))
                    test_datalist.append(test_data)

                    if outmask_isfirst:
                        outmask_isfirst = False
                        if not os.path.exists(outmask_dir):
                            os.makedirs(outmask_dir)
                        mpp.Pool(processes=mp.cpu_count()).map(img_writer, outmask_results)
        dict_this['inference_speed'] = sum(speeds) / len(speeds)
        print("mean inference_speed:", dict_this['inference_speed'])
        if len(evals) > 0:
            sum_iou = sum_f1 = sum_oa = 0.0
            test_evals = dict_this['test_evals'] = {}
            for iou, f1, oa, class_IoU, class_F1, epoch in evals:
                sum_iou += iou
                sum_f1 += f1
                sum_oa += oa
                if epoch not in test_evals:
                    # class_IoU, class_F1 就不取平均了
                    test_evals[epoch] = {
                        'test_mIoU': iou,
                        'test_F1': f1,
                        'test_OA': oa,
                        'class_IoU': class_IoU,
                        'class_F1': class_F1,
                        'repeat_times': 1
                    }
                else:
                    test_evals[epoch]['test_mIoU'] += iou
                    test_evals[epoch]['test_F1'] += f1
                    test_evals[epoch]['test_OA'] += oa
                    test_evals[epoch]['repeat_times'] += 1
            top_eval = None
            top_epoch = -1
            for e, te0 in test_evals.items():
                te0['test_mIoU'] /= te0['repeat_times']
                te0['test_F1'] /= te0['repeat_times']
                te0['test_OA'] /= te0['repeat_times']
                if top_epoch < 0:
                    top_epoch = e
                    top_eval = te0
                elif te0[args.sort_monitor] > top_eval[args.sort_monitor]:
                    top_epoch = e
                    top_eval = te0
            dict_this['top_eval'] = {'epoch': top_epoch}
            dict_this['top_eval'].update(top_eval)
            dict_this['top_eval']['monitor'] = args.sort_monitor

            iou = dict_this['avg_test_mIoU'] = sum_iou / len(evals)
            f1 = dict_this['avg_test_F1'] = sum_f1 / len(evals)
            oa = dict_this['avg_test_OA'] = sum_oa / len(evals)
            print(f"mIoU: {iou}, F1: {f1}, OA: {oa}")
        
        flops, params = do_profile(model)
        dict_this.update({
            'flops': flops,
            'params': params,
            'profile_input_shape': profile_input_shape
        })

        #TODO 优先使用文件夹里的description描述，便于更改

        if os.path.exists(workdir):
            fs = os.listdir(workdir)
            for f in fs:
                if f.startswith('D: '):
                    desc = f[len('D: '):]
                    break
        
        if desc is not None:
            dict_this['description'] = desc
        elif hasattr(cfg, 'description'):
            dict_this['description'] = cfg.description

        torch.save(test_history, test_database_path)
        del cfg


    #? Show Table
    for dataset_name, dict_dataset in test_history.items():
        tb = database_table()
        for md_name, dict_md in dict_dataset.items():
            assert isinstance(dict_md, dict)
            tb.new_row()
            tb.add_item('model', md_name)

            # if dict_md['test_state'] == "inftime":
            #     #workdir = os.path.join(args.workdir, dataset_name, md_name)
            #     pass
            # elif dict_md['test_state'] == "completed":

            #
            #     # if not os.path.exists(workdir):
            #     #     os.makedirs(workdir)
            # else:
            #     raise Exception("???")

            workdir = os.path.join(args.workdir, dataset_name, md_name, 'final')


            #f1 = open(testlog_path, 'w')
            # _st = dict_md['test_state']
            #TODO 保存 final/test.log
            # s0 = dataset_name + '/' + md_name + f' ({_st})\n'
            spd = dict_md.get('inference_speed', None)
            # s0 += f'inference_speed: {spd} P/s\n'
            # if 'test_mIoU' in dict_md:
            #     s0 += f'test_mIoU={dict_md["test_mIoU"]}, '
            # if 'test_F1' in dict_md:
            #     s0 += f'test_F1={dict_md["test_F1"]}, '
            # if 'test_OA' in dict_md:
            #     s0 += f'test_OA={dict_md["test_OA"]}'
            #?? flops 标定到 [1, 3, 1024, 1024]
            stdrate = 1.0

            img_shape = None
            if 'profile_input_shape' in dict_md:
                shape = dict_md['profile_input_shape']
                img_shape = shape[-2:]
                if not args.no_std:
                    stdrate *= shape[0] / 1
                    stdrate *= shape[1] / 3
                    stdrate *= shape[2] / 1024
                    stdrate *= shape[3] / 1024

            def do_std(v):
                if v is None:
                    return None
                else:
                    return v / stdrate

            tb.add_float_item('flops', do_std(dict_md.get('flops', None)))
            #tb.add_float_item('params', do_std(dict_md.get('params', None)))
            tb.add_float_item('params', dict_md.get('params', None))
            tb.add_float_item('input_shape', img_shape)

            tb.add_item('latency', 1000.0 / spd)
            # tb.add_item('speed', spd)
            top1 = None
            if 'top_eval' in dict_md:
                top1 = dict_md['top_eval']
                tb.add_item('top1_epoch', top1['epoch'])
                tb.add_item('top1_mIoU', top1['test_mIoU'])
                tb.add_item('top1_F1', top1['test_F1'])
                tb.add_item('top1_OA', top1['test_OA'])
            
            if 'description' in dict_md:
                tb.add_item('description', dict_md['description'])

            f_testlog = None
            testlog_path = os.path.join(workdir, 'test.log')
            if os.path.exists(workdir) and not os.path.exists(testlog_path):
                f_testlog = open(testlog_path, 'w')
            if f_testlog is not None:
                with redirect(f_testlog):
                    tb.show([tb.values[-1]])
                if top1 is not None and 'class_IoU' in top1 and 'class_F1' in top1:
                    tb2 = MatrixTable()
                    tb2.register_columns({
                        'class': {
                            'title': 'Class'
                        },
                        'IoU': {
                            'title': 'IoU',
                            'format': '{:.4f}'
                        },
                        'F1': {
                            'title': 'F1',
                            'format': '{:.4f}'
                        },
                    })
                    tb2.set_align('center')
                    tb2.columns_info['class']['align'] = 'left'
                    tb2.show_syle['title'] = 'Class_IoU & Class_F1 (Top1)'
                    d_ciou = top1['class_IoU']
                    d_cf1 = top1['class_IoU']
                    for c, iou in d_ciou.items():
                        d_ciou[c] = (iou, d_cf1.get(c, None))
                    for c, (iou, f1) in d_ciou.items():
                        tb2.new_row()
                        tb2.add_item('class', c)
                        tb2.add_item('IoU', iou)
                        tb2.add_item('F1', f1)
                    with redirect(f_testlog):
                        print()
                        tb2.show()

        # SORT
        # tb.sort([('speed', 'max'), ('top1_OA', 'max')])
        tb.sort([('latency', 'min'), ('top1_OA', 'max')])
        dir_ds = os.path.join(args.workdir, dataset_name)
        if not os.path.exists(dir_ds):
            os.makedirs(dir_ds)
        if args.database is not None:
            fn = os.path.split(args.database)[1]
            f = open(dir_ds + f'/compare({fn}).txt', 'w')
        else:
            f = open(dir_ds + '/compare.txt', 'w')

        tb.show_syle['title'] = f'Compare on {dataset_name}'
        tb.show()
        with redirect(f):
            tb.show()
        f.close()

    # if args.output_path is None:
    #     args.output_path = Path('fig_results/potsdam/{}'.format(
    #         config.test_weights_name))
    # args.output_path.mkdir(exist_ok=True, parents=True)


if __name__ == '__main__':
    os.environ['run_mode'] = 'test'
    main()