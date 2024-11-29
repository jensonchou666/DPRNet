import numpy as np
import cv2
import shutil
from collections import OrderedDict


红 = [255, 0, 0]
蓝 = [0, 0, 255]
绿 = [0, 255, 0]
黄 = [255, 255, 0]
紫 = [255, 0, 255]
青 = [0, 255, 255]

栗 = [128, 0, 0]
橄榄 = [128, 128, 0]
海军 = [0, 0, 128]
深绿 = [0, 128, 0]

棕 = (165, 42, 42)
亮粉 = [255, 105, 180]
小麦 = [245, 222, 179]
巧克力 = [210, 105, 30]
番茄 = [255, 99, 71]

白 = [255, 255, 255]
黑 = [0, 0, 0]



class Config:
    palette = {
        0: 白, 1: 蓝, 2: 黄, 3: 绿, 4: 紫, 5: 青, 6: 亮粉,
        7: 小麦, 8: 巧克力, 9: 栗, 10: 橄榄, 11: 海军, 12: 深绿, 13: 番茄,
        # 3: [0, 0, 255],
        # 4: [159, 129, 183],
        # 5: [0, 255, 0],
        # 6: [255, 195, 128]
    }
    wrong_rgb = 红

cfg = Config()

def set_rgb(rgb):
    global cfg
    cfg.palette = rgb


def set_rgb_dict(rgb_dict: dict):
    global cfg
    cfg.palette = list(rgb_dict.values())
    # for i, v in enumerate(palette.values()):
    #     cfg.palette[i] = v


    # if isinstance(rgb, dict):
    #     default_palette.update(rgb)
    # else:
    #     for i, v in enumerate(rgb):
    #         default_palette[i] = v

# default_palette = {
#     0: 绿, 1: 绿, 2: 绿, 3: 绿, 4: 绿, 5: 绿, 6: 绿
# }

def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


import matplotlib.pyplot as plt

def save_palette(classes_name, path, rgbs=None, show_wrong=True):
    global cfg
    if rgbs is None:
        rgbs = cfg.palette
    w, h = 80, 40
    nc = num_classes = len(classes_name)

    if show_wrong:
        nc += 1

    nH = int(pow(nc, 0.5) * 1.5)
    nW = nc // nH + 1
    fig = plt.figure()
    for i, name in enumerate(classes_name):
        ax = fig.add_subplot(nH, nW, i + 1)
        ax.axis('off')
        color = '#{:02X}{:02X}{:02X}'.format(*rgbs[i])
        rect = plt.Rectangle((0.3, 0.6), 2, 0.5, color=color)
        ax.add_patch(rect)
        ax.text(0.3, 0.35, f'({i}) {name}')

    if show_wrong:
        ax = fig.add_subplot(nH, nW, nc)
        ax.axis('off')
        color = '#{:02X}{:02X}{:02X}'.format(*cfg.wrong_rgb)
        rect = plt.Rectangle((0.3, 0.6), 2, 0.5, color=color)
        ax.add_patch(rect)
        ax.text(0.44, 0.35, 'wrong', color='#FF0000')

    plt.tight_layout()
    plt.savefig(path)


def label2rgb(mask, num_classes, rgbs=None):
    global cfg
    if rgbs is None:
        rgbs = cfg.palette

    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    if isinstance(rgbs, dict):
        it0 = iter(rgbs.values())
    else:
        it0 = iter(rgbs)
    for i in range(num_classes):
        mask_rgb[np.all(mask_convert == i, axis=0)] = next(it0)
    return mask_rgb


def mask_on_image(image: np.ndarray, mask: np.ndarray, num_classes,
                  transparent=0.4, rgbs=None):
    global cfg
    if rgbs is None:
        rgbs = cfg.palette

    mask_rgb = label2rgb(mask, num_classes, rgbs)
    return cv2.addWeighted(image, 1 - transparent, mask_rgb, transparent, 0)

# ？ 红： 预测成了背景, 绿：背景预测成前景

def mask_lable_wrong_map(mask: np.ndarray, lable: np.ndarray, num_classes, bottom='lable',
                         bg_index=0, rgbs=None):
    global cfg
    if rgbs is None:
        rgbs = cfg.palette


    h, w = mask.shape[0], mask.shape[1]

    if bottom is None:
        bottom = np.full(shape=(h, w, 3), fill_value=黑, dtype=np.uint8)
    elif isinstance(bottom, np.ndarray):
        bottom = bottom.copy()
    elif isinstance(bottom, str):
        if bottom == 'mask':
            bottom = label2rgb(mask, num_classes, rgbs)
        elif bottom == 'lable':
            bottom = label2rgb(lable, num_classes, rgbs)
        else:
            raise Exception('Wrong type')
    else:
        bottom = np.full(shape=(h, w, 3), fill_value=bottom, dtype=np.uint8)

    wrong = mask != lable

    #? 只统计 num_classes以内的类别
    wrong[lable >= num_classes] = False

    wrong = wrong.astype(np.uint8)
    wrong = wrong[np.newaxis, :, :]
    bottom[np.all(wrong == 1, axis=0)] = cfg.wrong_rgb
    return bottom


def mask_color_on_bottom(mask: np.ndarray, color, bottom: np.ndarray=None):
    h, w = mask.shape[0], mask.shape[1]
    if bottom is None:
        bottom = np.full(shape=(h, w, 3), fill_value=黑, dtype=np.uint8)
    bottom[mask.astype(np.bool8)] = color
    return bottom


def write_image(image, path: str):
    if path.endswith('tif'):
        image = image.astype(np.uint8)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

import os

img_suffixs = ['.tif', '.png', '.jpg']
__img_suffix__ = None

def original_image(batch, index='all', dataset=None):
    global img_suffixs, __img_suffix__
    rdall = (index == 'all')

    if dataset is not None and hasattr(dataset, 'get_image_path_batch'):
        if rdall:
            img_path = dataset.get_image_path_batch(batch)
            return [cv2.imread(path) for path in img_path]
        else:
            img_path = dataset.get_image_path_batch(batch, index=index)
            return cv2.imread(img_path)
    elif 'img_org' in batch:
        if rdall:
            return batch['img_org']
        else:
            return batch['img_org'][index]
    elif 'img_path' in batch:
        if rdall:
            return [cv2.imread(path) for path in batch['img_path']]
        else:
            return cv2.imread(batch['img_path'][index])


    #TODO 弃用
    elif 'img_id' in batch:
        if dataset is None or not hasattr(dataset, 'dir_img'):
            return None
        imgdir = dataset.dir_img

        # imgdir = 'images'
        # if hasattr(dataset, 'img_dir'):
        #     imgdir = dataset.img_dir
        # imgdir = os.path.join(dataset.data_root, imgdir)
        suffix = None
        if hasattr(dataset, 'img_suffix'):
            suffix = dataset.img_suffix
        elif __img_suffix__ is not None:
            suffix = __img_suffix__
        else:
            for s in img_suffixs:
                imgpath = os.path.join(imgdir, batch['img_id'][0] + s)
                if os.path.exists(imgpath):
                    __img_suffix__ = suffix = s
                    break
            if suffix is None:
                return None

        if not suffix.startswith('.'):
            suffix = '.' + suffix

        if rdall:
            imgs = []
            for img_id in batch['img_id']:
                path = os.path.join(imgdir, img_id + suffix)
                imgs.append(cv2.imread(path))
            return imgs
        else:
            return cv2.imread(os.path.join(imgdir, batch['img_id'][index] + suffix))


def gray_map_strengthen(gray: np.ndarray):
    gray = gray.astype(np.float64)
    gray = gray - np.min(gray)
    rate = 255.0 / np.max(gray)
    gray = gray * rate
    # print(np.max(gray), np.average(gray))
    gray = gray.astype(np.uint8)
    return gray
def gray_map_strengthen_flat(gray: np.ndarray):
    # gray = gray - np.min(gray)
    rate = 255.0 / np.max(gray)
    gray = gray * rate
    # print(np.max(gray), np.average(gray))
    gray = gray.astype(np.uint8)
    return gray
# old
# def gray_map_strengthen(gray: np.ndarray):
#     rate = 255.0 / max(np.max(gray), 1)
#     gray = gray * rate
#     # print(np.max(gray), np.average(gray))
#     gray = gray.astype(np.uint8)
#     return gray

def gray_to_thermodynamic(gray: np.ndarray):
    pass




""" --------------------------- analyser --------------------------- """


from jscv.utils.overall import global_dict
import jscv.utils.trainer as pkg_trainer
import jscv.utils.utils as utils

import torch.nn.functional as F
from jscv.utils.utils import edge_detect_target
import torch
from jscv.losses.utils import loss_map



def class_name(cls):
    return cls.__class__.__name__



def add_analyse_items(items, names=None):
    if "analyser" in global_dict:
        global_dict["analyser"].add_items(items, names)

def add_analyse_item(item, name=None):
    add_analyse_items([item], [name])




class AnalyseCallback(pkg_trainer.TrainerCallback):
    def from_cfg(cfg):
        utils.set_default(cfg, {
            'skip_analyse': False,
            'analyse_stage': 'val',
            # 'analyse_stage': ['train', 'val'], #? 流程较复杂
            'times_analyse': 20,
            'analyse_skip_same': True
        })
        return AnalyseCallback(cfg, cfg.analyse_stage, cfg.times_analyse, cfg.analyse_skip_same)

    def __init__(self,
                 cfg,
                 when_stage='val',
                 times_per_epoch=1,
                 skip_same=True) -> None:
        self.when_stage = when_stage
        self.times_per_epoch = times_per_epoch
        self.skip_same = skip_same
        self._counter = -1
        self._last_epoch = -1
        self.dist_index = -1
        self.counter_save = -1
        self.this_batch_do = False
        self.ctb = char_table = [''] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.cfg = cfg

        self.dict_call = OrderedDict()
        self.dict_show = OrderedDict()
        self.first_index = 0


    def add_items(self, items, names=None):
        self.add_calls(items, names)
        self.add_shows(items, names)

    def add_calls(self, items, names=None):
        if names is None:
            for item in items:
                self.add_next_call(item)
        else:
            for item, name in zip(items, names):
                self.add_next_call(item, name)

    def add_shows(self, items, names=None):
        if names is None:
            for item in items:
                self.add_next_show(item)
        else:
            for item, name in zip(items, names):
                self.add_next_show(item, name)

    def add_next_item(self, item, name=None):
        self.add_next_call(item, name)
        self.add_next_show(item, name)

    def add_next_call(self, item, name=None):
        if name is None:
            name = class_name(item)
        while name in self.dict_call:
            name = name + "_1"
        self.dict_call[name] = item
        item.analyser = self

    def add_next_show(self, item, name=None):
        if name is None:
            name = class_name(item)
        while name in self.dict_call:
            name = name + "_1"
        self.dict_show[name] = item
        item.analyser = self


    def next_filename(self, name):
        cs = self.counter_save
        # if cs >= 26 * 10
        a = cs // 10
        b = cs % 10
        self.counter_save += 1
        return f"{self.fname_prefix}-{self.dist_index}-[{self.ctb[a]}{b}] {name}"


    def init(self):
        cfg = self.cfg
        do_analyse = True
        if not hasattr(cfg, 'workdir'):
            do_analyse = False

        if cfg.skip_analyse:
            do_analyse = False
        self.do_analyse = do_analyse

        if do_analyse:
            if hasattr(cfg, 'analyse_begin'):
                self._counter = cfg.analyse_begin   # * self.times_per_epoch
            else:
                self._counter = 0
            d = self.analyse_dirname = f'/analyse_{self.when_stage}({self._counter}+{self.times_per_epoch}p)'
            self.analyse_dir = d = cfg.workdir + d
            if os.path.exists(d):
                shutil.rmtree(d)
            os.mkdir(d)


    def run_begin(self, before):
        if before:
            self.init()

    def training_step(self, batch, batch_idx, before):
        if not before and self.when_stage == 'train' and self.do_analyse:
            if self.if_do_this_batch(batch, batch_idx):
                self.analyse()

    def validation_step(self, batch, batch_idx, before):
        if not before and self.when_stage == 'val' and self.do_analyse:
            if self.if_do_this_batch(batch, batch_idx):
                self.analyse()

    def if_do_this_batch(self, batch, batch_idx):
        trainer = self.trainer
        if self.skip_same and batch_idx < self._counter:
            return False
        if trainer.epoch_idx != self._last_epoch:
            self._counter += 1
            if self._counter >= trainer.loader_len():
                self._counter = 0
            if self._counter % self.times_per_epoch == 0:
                self._last_epoch = trainer.epoch_idx
            self.counter_save = self.dist_index = 0
            self.batch = batch
            self.dataset = trainer.data_loader.dataset
            self.fname_prefix = self.analyse_dir + f'/{trainer.epoch_idx}-{trainer.batch_idx}'
            return True
        return False


    def analyse(self):
        datas = self.trainer.result
        dist_datas = dict()

        if datas is None:
            return

        for item in self.dict_call.values():
            item.call(datas, dist_datas)

        x = 0
        if len(dist_datas) > 0:
            a = next(iter(dist_datas.values()))
            for i in a:
                x += 1

        if x == 0: return

        disted_datas_list = [dict() for i in range(x)]

        for k, v in dist_datas.items():
            if isinstance(v, dict):
                data = v.pop("data")
                for i, distdata in enumerate(data):
                    disted_datas_list[i][k] = dict(data=distdata, **v)
            else:
                for i, dist_v in enumerate(v):
                    disted_datas_list[i][k] = dist_v

        self.dist_index = self.first_index

        for disted_datas in disted_datas_list:
            self.counter_save = self.first_index

            for item in self.dict_show.values():
                # show
                item.show(datas, disted_datas)
            self.dist_index += 1





'''↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 分析项定义 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'''




class AnalyseItem:
    name_dict = {}
    
    def __init__(self, include='all', exclude=[], name_dict=None):
        if name_dict is None:
            name_dict = self.name_dict
        self.name_dict = name_dict

        if include == 'all':
            include = list(name_dict.keys())
        self.save = []
        for i in include:
            if i not in exclude:
                self.save.append(i)

    def call(self, datas: dict, dist_datas: dict):
        ''' dist_datas 的值必须是可遍历的， 且所有值的遍历长度一致'''
        pass

    def show(self, datas, disted_datas: dict):
        for key in self.save:
            self.general_show(key, datas, disted_datas)


    def general_show(self, key, datas, disted_datas: dict):
        named = self.name_dict
        analyser = self.analyser

        if key not in disted_datas:
            return

        dd = disted_datas[key]
        data = dd["data"]
        t = dd["type"]

        def mask_color(map, RGB2BGR=False):
            if "mask_color" in dd:
                r, g, b = dd["mask_color"]  # R G B
                if RGB2BGR:
                    r, g, b = b, g, r
                mask_key = disted_datas[dd["mask_key"]]  # H, W
                if isinstance(mask_key, torch.Tensor):
                    mask_key = mask_key.cpu().numpy()
                map[mask_key.astype(np.bool8)] = (r, g, b)
            return map


        if "tensor" in dd and dd["tensor"]:
            data = data.cpu().numpy()

        if t == "simple_save":
            # print("@", data.shape, np.max(data), np.min(data))
            self.save_next_image(named[key], data)
        elif t == "gray":
            map = gray_map_strengthen(data)
            self.save_next_image(named[key], map)
        elif t == "gray_to_jet":
            map = gray_map_strengthen(data)
            map = cv2.applyColorMap(map, cv2.COLORMAP_JET)  # H, W, 3

            self.save_next_image(named[key], mask_color(map))
        elif t == "lable":
            map = label2rgb(data, analyser.cfg.num_classes) # H, W, 3
            self.save_next_RGB2BGR(named[key], mask_color(map))


    def to_jet(self, map):
        '''
            map: H, W
        '''
        return cv2.applyColorMap(gray_map_strengthen(map), cv2.COLORMAP_JET)


    def save_next_image(self, name, img):
        cv2.imwrite(self.analyser.next_filename(name), img)

    def save_next_RGB2BGR(self, name, img):
        cv2.imwrite(self.analyser.next_filename(name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))





class SegmentPrepare(AnalyseItem):
    name_dict = dict(org_img='image.png',
                     lable='lable.png',
                     pred='prediction.png',
                     palette='_palette.png')

    def __init__(self, include='all', exclude=[],
                 key_pred='pred', key_lable='gt_semantic_seg', dim=1):

        super().__init__(include, exclude, SegmentPrepare.name_dict)

        self.key_pred = key_pred
        self.key_lable = key_lable
        self.dim = dim

    def call(self, datas: dict, dist_datas: dict):
        '''
        datas:  pred, lable, pred_softmax, pred_numpy, org_imgs*
        '''
        dim = self.dim
        analyser = self.analyser
        batch = analyser.batch
        pred = datas['pred'] = datas[self.key_pred].detach()
        lable = datas['lable'] = batch[self.key_lable]
        if pred.shape[-1] != lable.shape[-1]:
            pred = F.interpolate(pred, lable.shape[-2:], mode='bilinear')
        
        pred_softmax = datas['pred_softmax'] = F.softmax(pred, dim=dim)
        datas['pred_numpy'] = pred_softmax.argmax(dim=dim).cpu().numpy()

        if 'org_img' in self.save:
            org_imgs = original_image(batch, 'all', analyser.dataset)
            if org_imgs is None or org_imgs[0] is None:
                org_imgs = [None] * pred.shape[0]
            dist_datas['org_img'] = datas['org_imgs'] = org_imgs
        #if 'lable' in self.save:
        dist_datas['lable'] = lable.cpu().numpy()
        #if 'pred' in self.save:
        dist_datas['pred'] = datas['pred_numpy']

        if 'palette' in self.save and not hasattr(self, '_save_palette_'):
            self._save_palette_ = True
            save_palette(analyser.cfg.classes,
                         os.path.join(analyser.analyse_dir, self.name_dict['palette']))


    def show(self, datas, disted_datas: dict):
        d = self.name_dict
        analyser = self.analyser

        disted_datas['lable_map'] = label2rgb(disted_datas["lable"], analyser.cfg.num_classes)
        disted_datas['pred_map'] = label2rgb(disted_datas["pred"], analyser.cfg.num_classes)

        for key in self.save:
            if key == 'org_img' and disted_datas[key] is not None:
                cv2.imwrite(analyser.next_filename(d[key]), disted_datas[key])
            elif key in ('lable', 'pred'):
                map = disted_datas[f'{key}_map']
                cv2.imwrite(analyser.next_filename(d[key]), cv2.cvtColor(map, cv2.COLOR_RGB2BGR))




class AnyItems_Segment(AnalyseItem):
    name_dict = dict(loss_map='loss_map.png',
                     wrong_map='wrong_map.png',
                     wrong_alone_map='wrong_alone_map.png',
                     confuse_map='confuse_map.png',
                     confuse_dilate='confuse_dilate.png',
                     confuse_gate='confuse_gate.png',
                     coarse_pred='coarse_pred.png',
                     coarse_confuse='coarse_confuse.png',
                     coarse_confuse_dilate='coarse_confuse_dilate.png',
                     coarse_confuse_gate='coarse_confuse_gate.png',
                     coarse_refine_success="coarse_refine_success.png",
                     edge_lable='edge_lable.png')

    def __init__(self, include='all', exclude=[], dim=1, confuse_ksize=5, confuse_gate=0.22):
        self.dim = dim
        super().__init__(include, exclude, AnyItems_Segment.name_dict)
        pad = (confuse_ksize - 1) // 2
        self.max_pool = torch.nn.MaxPool2d(kernel_size=confuse_ksize, stride=1, padding=pad)
        self.confuse_map_gate = confuse_gate
    

    def reset_confuse_ksize(self):
        if "confuse_ksize" in global_dict:
            confuse_ksize = global_dict["confuse_ksize"]
            confuse_gate = global_dict["confuse_gate"]
            print(f"reseted confuse_ksize={confuse_ksize}, confuse_map_gate={confuse_gate}")
            global_dict.pop("confuse_ksize")
            pad = (confuse_ksize - 1) // 2
            self.max_pool = torch.nn.MaxPool2d(kernel_size=confuse_ksize, stride=1, padding=pad)
            self.confuse_map_gate = confuse_gate


    def confuse_map(self, pred_soft, key, dist_datas: dict):
        top1_scores = torch.max(pred_soft, dim=self.dim)[0]
        confuse_top1 = (1 - top1_scores)
        if key in self.save:
            dist_datas[key] = {"data": confuse_top1.cpu().numpy(), "type": "gray"}
        return confuse_top1

    def confuse_dilate(self, confuse_top1, key, dist_datas: dict):
        confuse_top1_dilate = self.max_pool(confuse_top1)
        if key in self.save:
            dist_datas[key] = {"data": confuse_top1_dilate.cpu().numpy(), "type": "gray"}
        return confuse_top1_dilate

    def confuse_gate(self, confuse_top1_dilate, key, dist_datas: dict):
        confuse_top1_gate = confuse_top1_dilate > self.confuse_map_gate
        if key in self.save:
            dist_datas[key] = {"data": confuse_top1_gate.cpu().numpy(), "type": "gray"}
        return confuse_top1_gate


    def call(self, datas: dict, dist_datas: dict):
        
        self.reset_confuse_ksize()
        
        dim = self.dim
        targets = datas['lable']
        pred_soft = datas['pred_softmax']
        analyser = self.analyser
        cfg = analyser.cfg

        '''---------- edge_lable ----------'''
        if 'edge_lable' in self.save:
            edge_lable = edge_detect_target(targets.unsqueeze(1).float()).squeeze(1).detach()
            datas['edge_lable'] = edge_lable
            datas['edge_lable_np'] = dist_datas['edge_lable'] = edge_lable.cpu().numpy() * 255

        '''---------- confuse_map ----------'''
        if "confuse_map" in self.save:
            confuse_map = self.confuse_map(pred_soft, "confuse_map", dist_datas)
            confuse_dilate = self.confuse_dilate(confuse_map, "confuse_dilate", dist_datas)
            confuse_gate = self.confuse_gate(confuse_dilate, "confuse_gate", dist_datas)


        if "coarse_pred" in self.save:
            if "coarse_pred" in datas:
                coarse_pred = datas["coarse_pred"]
            elif "coarse_pred_list" in datas:
                coarse_pred = datas["coarse_pred"] = datas["coarse_pred_list"][0]
            else:
                coarse_pred = None
            if coarse_pred is not None:
                coarse_pred = datas["coarse_pred_softmax"] = F.softmax(coarse_pred, dim=dim)
                
                coarse_hw = coarse_pred.shape[-2:]
                coarse_pred = F.interpolate(coarse_pred, pred_soft.shape[-2:], mode='bilinear')
                cp = coarse_pred.argmax(dim=dim).cpu().numpy()
                dist_datas["coarse_pred"] = {"data": cp, "type": "lable"}


                confuse_map = self.confuse_map(coarse_pred, "coarse_confuse", dist_datas)
                confuse_dilate = self.confuse_dilate(confuse_map, "coarse_confuse_dilate", dist_datas)

                #?
                confuse_dilate = F.interpolate(confuse_dilate.unsqueeze(1), coarse_hw, mode='bilinear').squeeze(1)

                confuse_gate = self.confuse_gate(confuse_dilate, "coarse_confuse_gate", dist_datas)
                
                

                # k2 = "coarse_confuse_gate_2"
                # self.save.append(k2)
                # self.name_dict[k2] = "coarse_confuse_gate_2.png"
                # dist_datas[k2] = {"data": confuse_gate.cpu().numpy(), "type": "gray"}



        '''
        if 'confuse_top2' in self.save:
            top2_scores = torch.topk(pred_soft, k=2, dim=dim)[0]
            confuse_top2 = (top2_scores[:, 0] - top2_scores[:, 1])
            confuse_top2 = (torch.max(confuse_top2) - confuse_top2) ** 0.3
            datas['confuse_top2_np'] = dist_datas['confuse_top2'] = confuse_top2.cpu().numpy()

        if 'confuse_top3' in self.save:
            top3_scores = torch.topk(pred_soft, k=3, dim=dim)[0]
            confuse_top3 = (top3_scores[:, 0] - 0.6 * top3_scores[:, 1] - 0.4 * top3_scores[:, 2])
            confuse_top3 = (torch.max(confuse_top3) - confuse_top3) ** 3
            datas['confuse_top3_np'] = dist_datas['confuse_top3'] = confuse_top3.cpu().numpy()
        '''

        '''---------- loss_map ----------'''
        datas['loss_map'] = lossmap = loss_map(datas["pred"], targets.to(pred_soft.device), cfg.ignore_index, pred_logits=True)
        datas['loss_map_np'] = dist_datas['loss_map'] = {"data": lossmap.cpu().numpy(), "type": "gray_to_jet"}


    def show(self, datas, disted_datas: dict):
        pred = disted_datas['pred']
        lable = disted_datas['lable']
        analyser = self.analyser
        cfg = analyser.cfg
        named = self.name_dict

        for key in self.save:
            if key == 'wrong_map':
                wrong_map = mask_lable_wrong_map(pred, lable, cfg.num_classes)
                self.save_next_RGB2BGR(named[key], wrong_map)
            elif key == 'wrong_alone_map':
                if "coarse_pred" in disted_datas:
                    #? coarse_pred 需和pred同shape
                    pred0 = disted_datas["coarse_pred"]
                    ps = "coarse_"
                else:
                    pred0 = disted_datas["pred"]
                    ps = ""
                if isinstance(pred0, dict):
                    pred0 = pred0["data"]
                wrong_alone_map = mask_lable_wrong_map(pred0, lable, cfg.num_classes, None)
                self.save_next_RGB2BGR(ps + named[key], wrong_alone_map)
            
            elif key == 'coarse_refine_success':
                dd = disted_datas
                if 'coarse_confuse_gate' not in dd:
                    continue
                else:
                    coarse_pred = dd["coarse_pred"]["data"]
                    m1 = dd['coarse_confuse_gate']["data"]
                    bottom = mask_color_on_bottom(m1, 白)
                    m1_r = m1 & (pred == lable)
                    m1_f = m1 & (pred != lable)
                    
                    m1_r_r = m1_r & (coarse_pred==lable)
                    m1_r_f = m1_r & (coarse_pred!=lable)
                    m1_f_r = m1_f & (coarse_pred==lable)
                    m1_f_f = m1_f & (coarse_pred!=lable)
                    
                    # bottom = mask_color_on_bottom(m1_r_r, 蓝, bottom)
                    bottom = mask_color_on_bottom(m1_r_f, 绿, bottom)
                    bottom = mask_color_on_bottom(m1_f_r, 红, bottom)
                    bottom = mask_color_on_bottom(m1_f_f, 黄, bottom)
                    
                    self.save_next_RGB2BGR(named[key], bottom)

            elif key == 'edge_lable':
                self.save_next_image(named[key], disted_datas[key])
            else:
                self.general_show(key, datas, disted_datas)




class Item_LossGap(AnalyseItem):
    name_dict = dict(loss_map_levels=[20, 40, 60, 80, 120, 160, 200])

    def __init__(self, include='all', exclude=[]):
        super().__init__(include, exclude, Item_LossGap.name_dict)

    
    def call(self, datas: dict, dist_datas: dict):
        analyser = self.analyser
        cfg = analyser.cfg

        if hasattr(analyser, 'trainer'):
            trainer = analyser.trainer

            assert isinstance(trainer, pkg_trainer.Model_Trainer)
            
            last_train_data = trainer.last_epoch_data("train")

            if last_train_data is None:
                return

            if "loss_map" not in datas:
                return

            class_iou = last_train_data["class_IoU"]
            targets = datas['lable']
            pred_soft = datas['pred_softmax']
            loss_map = datas['loss_map']

            if utils.do_once(self, "a1"):
                for i, (k, iou) in enumerate(class_iou.items()):
                    print(i, f" {k:<15}", iou)
                #print("@", len(class_iou), cfg.num_classes, loss_map.shape, targets.shape, pred_soft.shape)



    def show(self, datas, disted_datas: dict):
        analyser = self.analyser
        for key in self.save:
            if key == 'loss_map_levels':
                loss_map = gray_map_strengthen(disted_datas["loss_map"])
                for loss_map_level in self.name_dict[key]:
                    m = loss_map
                    m[m < loss_map_level] = 0
                    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
                    self.save_next_image(f"loss_map_level_{loss_map_level}.png", m)


"""
    用于抠图
"""
class MattePraper(AnalyseItem):
    name_dict = dict(lable='lable.png',
                     pred='prediction.png',
                     err='err.png',
                     pred_err="pred_err.png",
                     coarse_pred='coarse_pred.png',
                     refine_region='refine_region.png',
                     compose_err_refine='compose_err_refine.png',
                     )

    def __init__(self, include='all', exclude=[],
                 key_pred='pred', key_lable='true_pha', dim=1, ):

        super().__init__(include, exclude, MattePraper.name_dict)

        self.key_pred = key_pred
        self.key_lable = key_lable
        self.dim = dim

    def call(self, datas: dict, dist_datas: dict):
        dim = self.dim
        analyser = self.analyser
        batch = analyser.batch
        pred = datas['pred'] = datas[self.key_pred].detach()
        lable = datas['lable'] = batch[self.key_lable]
        if pred.shape[-1] != lable.shape[-1]:
            pred = F.interpolate(pred, lable.shape[-2:], mode='bilinear')


        pred_c = dist_datas['pred'] = pred.cpu().numpy()
        lable_c = dist_datas['lable'] = lable.cpu().numpy()
        dist_datas['err'] = np.abs(pred_c - lable_c)

        def f1(key):
            if key in datas:
                data_key = datas[key]
                if data_key.shape[2:] != pred_c.shape[2:]:
                    if key == "refine_region":
                        # H, W = pred_c.shape[2:]
                        data_key = F.interpolate(data_key, pred_c.shape[2:], mode='nearest')
                    else:
                        data_key = F.interpolate(data_key, pred_c.shape[2:], mode='bilinear')
                dist_datas[key] = data_key.cpu().numpy()
                return data_key
            return None

        f1("pred_err")
        f1("coarse_pred")
        refine_region = f1("refine_region")
        
        if refine_region is not None:
            err = torch.abs(pred - lable.to(pred.device)).unsqueeze(-1).repeat(1, 1, 1, 1, 3)
            err[..., :-1] = 0    # red
            refine_region = refine_region.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
            compose_err_refine = (err * 0.5 + refine_region * 0.5) * 255
            dist_datas["compose_err_refine"] = compose_err_refine.cpu().numpy()


    def show(self, datas, disted_datas: dict):
        d = self.name_dict
        analyser = self.analyser
        
        for key in self.save:
            # if key in ('lable', 'pred', 'err', 'pred_err'):
            if key not in disted_datas:
                continue
            
            if key == "compose_err_refine":
                map = disted_datas[key].squeeze()
                cv2.imwrite(analyser.next_filename(d[key]), map)
            else:
                map = disted_datas[key].squeeze()
                map = gray_map_strengthen(map)
                cv2.imwrite(analyser.next_filename(d[key]), map)



def analyse_segmenter_simple():
    item = SegmentPrepare()
    add_analyse_item(item)

def analyse_segmenter_anyitems():
    items = [SegmentPrepare(), AnyItems_Segment()]
    add_analyse_items(items)

def analyse_segmenter_lossgap(cfg):
    items = [SegmentPrepare(), AnyItems_Segment(), Item_LossGap()]
    add_analyse_items(items)



IKD = include_key_dict = {}

IKD["loss_wronge"] = ["loss_map", "wrong_map", "wrong_alone_map"]

fine_confuse_d = ["confuse_map", "confuse_dilate", "confuse_gate"]
coarse_confuse_d = ["coarse_pred", "coarse_confuse", "coarse_confuse_dilate", "coarse_confuse_gate",
                    'coarse_refine_success']

IKD["confuse"] = IKD["loss_wronge"] + fine_confuse_d

IKD["coarse err"] = IKD["loss_wronge"] + coarse_confuse_d

IKD["fine+coarse err"] = IKD["confuse"] + fine_confuse_d + coarse_confuse_d


#TODO
#!!!!
def analyser_from_cfg(cfg):
    if "analyse_segment" in cfg:
        items = [SegmentPrepare()]
        if "analyser_include" in cfg:
            if "analyser_exclude_items" in cfg:
                exclude = cfg["analyser_exclude_items"]
            else:
                exclude = []
            inc = IKD[cfg["analyser_include"]]
            items.append(AnyItems_Segment(include=inc, exclude=exclude))
    elif "analyse_matte" in cfg:
        items = [MattePraper()]
    elif "analyse_items" in cfg:
        items = cfg['analyse_items']
    else:
        return
    add_analyse_items(items)



''' Old code
    #TODO 给出图片的原文件名
    def _analyse_1(self, datas):
        trainer = datas['trainer'] = global_dict['trainer']
        cfg = datas['cfg'] = global_dict['cfg']
        batch = datas['batch'] = trainer.batch
        analyse_dir = datas['analyse_dir'] = cfg.workdir + self.analyse_dirname
        prefix = datas['prefix'] = analyse_dir + f'/{trainer.epoch_idx}:{trainer.stage}:{trainer.batch_idx}'

        pred = datas['pred'].detach()

        #TODO 暂时这么写 赶时间
        org_imgs = original_image(batch, 'all', trainer.data_loader.dataset)
        if org_imgs is not None and org_imgs[0] is not None:
            pass
        else:
            org_imgs = [None] * pred.shape[0]



        targets = batch['gt_semantic_seg']
        if pred.shape[-1] != targets.shape[-1]:
            pred = F.interpolate(pred, targets.shape[-2:], mode='bilinear')
        pred_soft = F.log_softmax(pred, dim=1)
        pred_np = pred_soft.argmax(dim=1).cpu().numpy()
        

        targets_edge = edge_detect_target(targets.unsqueeze(1).float()).squeeze(1).cpu().numpy()
        targets_edge *= 255

        top2_scores = torch.topk(pred_soft, k=2, dim=1)[0]
        confuses_top2 = (top2_scores[:, 0] - top2_scores[:, 1])
        confuses_top2 = (torch.max(confuses_top2) - confuses_top2) ** 4
        confuses_top2 = confuses_top2.cpu().numpy()

        top3_scores = torch.topk(pred_soft, k=3, dim=1)[0]
        confuses_top3 = (top3_scores[:, 0] - 0.6 * top3_scores[:, 1] - 0.4 * top3_scores[:, 2])
        confuses_top3 = (torch.max(confuses_top3) - confuses_top3) ** 3
        confuses_top3 = confuses_top3.cpu().numpy()

        top1_scores = torch.max(pred_soft, dim=1)[0]
        confuses_top1 = (torch.max(top1_scores) - top1_scores)
        confuses_top1 = confuses_top1.cpu().numpy()

        if not hasattr(self, '_save_palette_'):
            self._save_palette_ = True
            save_palette(cfg.classes, analyse_dir + '/_palette.png')
        
        lossmap = loss_map(pred_soft, targets.to(pred_soft.device), cfg.ignore_index).cpu().numpy()

        for i, (img, target, te, pd, lmp, t1_map, t2_map, t3_map) in enumerate(
                zip(org_imgs, targets.numpy(), targets_edge, pred_np, lossmap,
                    confuses_top1, confuses_top2, confuses_top3)):
            fprefix = prefix + f':{i}-'
            # print(img.shape, target.shape, pd.shape, predloss.shape)
            #(1024, 1024, 3) (1024, 1024) (1024, 1024) (1024, 1024)
            # print(type(img), type(target), type(pd), type(predloss))
            if img is not None:
                cv2.imwrite(fprefix + '[1] image.png', img)

            target_map = label2rgb(target, cfg.num_classes)
            cv2.imwrite(fprefix + '[2] lable.png', cv2.cvtColor(target_map, cv2.COLOR_RGB2BGR))
            
            pred_map = label2rgb(pd, cfg.num_classes)
            cv2.imwrite(fprefix + '[3] prediction.png', cv2.cvtColor(pred_map, cv2.COLOR_RGB2BGR))


            wrong_map = mask_lable_wrong_map(pd, target, cfg.num_classes)
            wrong_alone_map = mask_lable_wrong_map(pd, target, cfg.num_classes, None)
            cv2.imwrite(fprefix + '[4] wrong_map.png', cv2.cvtColor(wrong_map, cv2.COLOR_RGB2BGR))
            cv2.imwrite(fprefix + '[5] wrong_alone_map.png', cv2.cvtColor(wrong_alone_map, cv2.COLOR_RGB2BGR))
            
            #target_edge_map = gray_map_strengthen(te)
            cv2.imwrite(fprefix + '[6] target_edge_map.png', te)

            lmp = gray_map_strengthen(lmp)
            lmp = cv2.applyColorMap(lmp, cv2.COLORMAP_JET)
            cv2.imwrite(fprefix + '[7] loss_map.png', lmp)

            t1_map = gray_map_strengthen(t1_map)
            t1_map = cv2.applyColorMap(t1_map, cv2.COLORMAP_JET)
            cv2.imwrite(fprefix + '[a1] confuse_map_top1.png', t1_map)

            t2_map = gray_map_strengthen(t2_map)
            t2_map = cv2.applyColorMap(t2_map, cv2.COLORMAP_JET)
            cv2.imwrite(fprefix + '[a2] confuse_map_top2.png', t2_map)

            t3_map = gray_map_strengthen(t3_map)
            t3_map = cv2.applyColorMap(t3_map, cv2.COLORMAP_JET)
            cv2.imwrite(fprefix + '[a3] confuse_map_top3.png', t3_map)
    '''