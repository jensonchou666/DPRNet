import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as albu
from typing import Any, Iterable
from functools import partial

from PIL import Image
import random

import jscv.datasets.transform as my_trans

'''
data_root
    imgs
    lables

or

data_root
    sub_dir_1
        imgs
        lables
    sub_dir_2
        imgs
        lables

or

data_root
    sub_dir_1
        sub_dir_1_1
            imgs
            lables
    sub_dir_2
        imgs
        lables

or

data_root
    imgs/train
        sub_dir_1
        sub_dir_2
    lables/train
        sub_dir_1
        sub_dir_2

'''

class AugCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data):
        img, mask = data['image'], data['mask']
        for t in self.transforms:
            img, mask = t(img, mask)
        data['image'], data['mask'] = img, mask
        return data


class SegmentDataset(Dataset):

    def __init__(
        self,
        data_root='data',
        aug_list=[],
        sub_dirs=None,
        img_dir_name='images',
        lable_dir_name='lables',
        img_suffix='',
        lable_suffix='',
        lable_from_rgb=False,
        sub_dirs_at_before=True,
        traverse_search=False,  # sub_dirs
        mini=False,
        mini_len=10,
        error_hint=True,
        collect_filesinfo_func='default',
        concat_filesinfo_func='default',
        load_data_func=None,
        mean=None,
        std=None,
        keys_name=['img', 'gt_semantic_seg'],
        **kwargs
        #img_suffix='.tif', lable_suffix='.png'):
    ):
        '''
            sub_dirs: select: None / 'auto' / ['sub_1', 'sub_2']
                None:  data_root/images  、  data_root/lables
                'auto': data_root/*/images 、  data_root/*/lables  (auto_find)
        '''
        self.data_root = data_root
        self.idn = img_dir_name
        self.mdn = lable_dir_name
        if img_suffix is None:
            img_suffix = ''
        if lable_suffix is None:
            lable_suffix = ''
        self.img_suffix = img_suffix
        self.lable_suffix = lable_suffix
        if aug_list is None or len(aug_list) == 0:
            aug_list = [my_trans.TransNumpy]
        self.aug_list = aug_list
        self.lable_from_rgb = lable_from_rgb

        self.error_hint = error_hint
        self.kwargs = kwargs
        self.mini = mini
        self.mini_len = mini_len
        self.keys_name = keys_name
        self.mean = mean
        self.std = std

        assert osp.exists(data_root), f"{data_root} is not exitst"

        if collect_filesinfo_func is not None and collect_filesinfo_func != 'default':
            self.collect_filesinfo = partial(collect_filesinfo_func, self)
        if concat_filesinfo_func is not None and concat_filesinfo_func != 'default':
            self.concat_filesinfo = partial(concat_filesinfo_func, self)
        
        if load_data_func is not None and load_data_func != 'default':
            self.load_data_func = partial(load_data_func, self)
        else:
            self.load_data_func = self.load_data

        self.use_subdir = False if sub_dirs is None else True
        if self.use_subdir:
            self.traverse_search = traverse_search
            self.sub_dirs_at_before = sub_dirs_at_before
            if sub_dirs_at_before:
                self.get_image_dir = self._imgdir_1
                self.get_lable_dir = self._labledir_1
            else:
                self.get_image_dir = self._imgdir_2
                self.get_lable_dir = self._labledir_2
            self.sub_dirs = self.list_subdir(sub_dirs)
        else:
            self.get_image_dir = self._imgdir_0
            self.get_lable_dir = self._labledir_0

        self.files_info = self.scan_files()
        if self.mini:
            self.files_info = self.files_info[:self.mini_len]


    def __getitem__(self, index):
        data = self.load_data_func(index)
        # if self.aug_list is not None:
        for aug in self.aug_list:
            data = aug(**data)
        return self.result(index, data)


    def __len__(self):
        return len(self.files_info)

    def _imgdir_0(self, subdir=None):
        return osp.join(self.data_root, self.idn)

    def _imgdir_1(self, subdir=''):
        return osp.join(self.data_root, subdir, self.idn)

    def _imgdir_2(self, subdir=''):
        return osp.join(self.data_root, self.idn, subdir)

    def _labledir_0(self, subdir=None):
        return osp.join(self.data_root, self.mdn)

    def _labledir_1(self, subdir=''):
        return osp.join(self.data_root, subdir, self.mdn)

    def _labledir_2(self, subdir=''):
        return osp.join(self.data_root, self.mdn, subdir)

    def list_subdir(self, sub_dirs):
        data_root = self.data_root
        img_dir_name = self.idn
        lable_dir_name = self.mdn
        sub_dirs_at_before = self.sub_dirs_at_before
        traverse_search = self.traverse_search

        def traverse_dir(root, dir_list: set, middir=''):
            dir0 = osp.join(root, middir)
            files = os.listdir(dir0)
            for f in files:
                if osp.isdir(osp.join(dir0, f)):
                    d2 = osp.join(middir, f)
                    dir_list.add(d2)
                    traverse_dir(root, dir_list, d2)

        def traverse_dir_name(root, dir_list: set, stop_name, middir=''):
            dir0 = osp.join(root, middir)
            files = os.listdir(dir0)
            for f in files:
                if f == stop_name:
                    dir_list.add(middir)
                elif osp.isdir(osp.join(dir0, f)):
                    traverse_dir_name(root, dir_list, stop_name, osp.join(middir, f))

        if sub_dirs == 'auto':
            img_sub_dirs = set()
            lable_sub_dirs = set()
            if sub_dirs_at_before:
                if traverse_search:
                    assert osp.split(img_dir_name)[0] == '', "assert '/' not in img_dir_name"
                    assert osp.split(lable_dir_name)[0] == '', "assert '/' not in lable_dir_name"
                    traverse_dir_name(data_root, img_sub_dirs, img_dir_name)
                    traverse_dir_name(data_root, lable_sub_dirs, lable_dir_name)
                else:
                    for f in os.listdir(data_root):
                        if osp.exists(osp.join(data_root, f, img_dir_name)):
                            img_sub_dirs.add(f)
                        if osp.exists(osp.join(data_root, f, lable_dir_name)):
                            lable_sub_dirs.add(f)
                if self.error_hint:
                    for s in (img_sub_dirs - lable_sub_dirs):
                        f1 = osp.join(data_root, s, img_dir_name)
                        f2 = osp.join(data_root, s, lable_dir_name)
                        print(f"In the dir:{s}, {f1} exist, but {f2} not exist, omit! ")
                    for s in (lable_sub_dirs - img_sub_dirs):
                        f1 = osp.join(data_root, s, img_dir_name)
                        f2 = osp.join(data_root, s, lable_dir_name)
                        print(f"In the dir:{s}, {f2} exist, but {f1} not exist, omit! ")
            else:
                imdir = osp.join(data_root, img_dir_name)
                madir = osp.join(data_root, lable_dir_name)
                if traverse_search:
                    traverse_dir(imdir, img_sub_dirs)
                    traverse_dir(madir, lable_sub_dirs)
                else:
                    for f in os.listdir(imdir):
                        if osp.isdir(osp.join(imdir, f)):
                            img_sub_dirs.add(f)
                    for f in os.listdir(madir):
                        if osp.isdir(osp.join(madir, f)):
                            lable_sub_dirs.add(f)
                if self.error_hint:
                    for s in (img_sub_dirs - lable_sub_dirs):
                        print(f"{s} exist in img_dir, but not in lable_dir, omit! ")
                    for s in (lable_sub_dirs - img_sub_dirs):
                        print(f"{s} exist in lable_dir, but not in img_dir, omit! ")
            sub_dirs = img_sub_dirs & lable_sub_dirs

        elif isinstance(sub_dirs, Iterable):
            for sub_dir in sub_dirs:
                imgdir = self.get_image_dir(sub_dir)
                labledir = self.get_lable_dir(sub_dir)
                assert osp.exists(imgdir), f"can't find image dir: {imgdir}"
                assert osp.exists(labledir), f"can't find lable dir: {labledir}"
        else:
            raise TypeError("wrong type")

        return list(sub_dirs)

    def scan_files(self):
        files_info = []
        c = 0
        if self.use_subdir:
            for subdir in self.sub_dirs:
                imgdir = self.get_image_dir(subdir)
                labledir = self.get_lable_dir(subdir)
                xx = self.collect_filesinfo(imgdir, labledir, subdir)
                files_info.append((xx, subdir))
                c += len(xx)
                if self.mini and c > self.mini_len:
                    break
        else:
            imgdir = self.get_image_dir()
            labledir = self.get_lable_dir()
            files_info = self.collect_filesinfo(imgdir, labledir)

        return self.concat_filesinfo(files_info)


    def get_image_path_batch(self, batch, index=None):
        ids = batch['id']
        if self.use_subdir:
            subdir = batch['subdir']
            if index is None:
                ret = []
                for id, s in zip(ids, subdir):
                    ret.append(osp.join(self.get_image_dir(s), id + self.img_suffix))
                return ret
            else:
                return osp.join(self.get_image_dir(subdir[index]), ids[index] + self.img_suffix)
        else:
            if index is None:
                ret = []
                for id in ids:
                    ret.append(osp.join(self.get_image_dir(), id + self.img_suffix))
                return ret
            else:
                return osp.join(self.get_image_dir(), ids[index] + self.img_suffix)

    def get_image_path(self, id, subdir=None):
        if self.use_subdir:
            return osp.join(self.get_image_dir(subdir), id + self.img_suffix)
        else:
            return osp.join(self.get_image_dir(), id + self.img_suffix)

    '''
        ↓ ↓ ↓ ↓ complet these, override! ↓ ↓ ↓ ↓
    '''

    def load_data(self, index):
        '''default load_data schedule, same name'''
        info = self.files_info[index]
        # mp = self.mask_from_palette

        if self.use_subdir:
            id = info[0]
            subdir = info[1]
        else:
            id = info
            subdir = None
        img_path = osp.join(self.get_image_dir(subdir), id + self.img_suffix)
        lable_path = osp.join(self.get_lable_dir(subdir), id + self.lable_suffix)
    
        # global Counter0
        # t0 = time.time()
        img = Image.open(img_path)
        if self.lable_from_rgb:
            #TODO
            mask = Image.open(lable_path).convert('RGB')
        else:
            mask = Image.open(lable_path).convert('L')
        # Counter0 += time.time() - t0

        return {'image': img, 'mask': mask}
        # return img, mask

    def result(self, index, data):
        results = {}
        results[self.keys_name[0]] = torch.from_numpy(data['image']).permute(2, 0, 1).float()
        results[self.keys_name[1]] = torch.from_numpy(data['mask']).long()
        info = self.files_info[index]
        if self.use_subdir:
            results['id'] = info[0]
            results['subdir'] = info[1]
        else:
            results['id'] = info
        return results



    def collect_filesinfo(self, imgdir, labledir, subdir=None):
        '''default filter_schedule, same name'''
        names = []

        img_suffix, lable_suffix = self.img_suffix, self.lable_suffix
        Li = -len(self.img_suffix)

        for i, f in enumerate(os.listdir(imgdir)):
            if self.mini and i > self.mini_len:
                break

            if Li != 0:
                if not f.endswith(img_suffix):
                    continue
                name = f[:Li]
            else:
                name = f
            maskpath = osp.join(labledir, name + lable_suffix)
            if not osp.exists(maskpath):
                if self.error_hint:
                    print(f"{osp.join(imgdir, f)}(image) exist, but {maskpath}(lable) not exist, omit!")
                continue
            if subdir is None:
                names.append(name)
            else:
                names.append((name, subdir))

        return names



    def concat_filesinfo(self, files_info_list):
        '''default filter_schedule'''
        return self.concat_filesinfo_no_shuffle(files_info_list)

    '''
        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
    '''


    def collect_files_omit_hiatus(self, imgdir, labledir, subdir=None):
        imgs = set(os.listdir(imgdir))
        lables = set(os.listdir(labledir))

        if self.error_hint:
            for f in imgs - lables:
                print(f"file [{f}], exist in image_dir, but not in lable_dir, omit!")
            for f in lables - imgs:
                print(f"file [{f}], exist in lable_dir, but not in image_dir, omit!")

        files = imgs & lables

        if subdir is None:
            return list(files)
        else:
            return [(f, subdir) for f in files]


    def concat_filesinfo_shuffle(self, files_info_list):
        if self.use_subdir:
            ret = []
            for files_info, subdir in files_info_list:
                ret += files_info
        else:
            ret = files_info_list
        random.shuffle(ret)
        return ret

    def concat_filesinfo_no_shuffle(self, files_info_list):
        if self.use_subdir:
            ret = []
            for files_info, subdir in files_info_list:
                ret += files_info
            return ret
        else:
            return files_info_list

    #TODO 待完成
    def concat_filesinfo_step_subdir(self, files_info_list):
        '''
            以一定的图片数(或图片数比例), 阶梯式地安排files_info排序策略
            各种参数设置参考以下代码
        '''
        # print(self.kwargs)
        if self.use_subdir:
            ret = []
            for files_info, subdir in files_info_list:
                ret += files_info
            return ret
        else:
            return files_info_list



class MosaicLoad:

    def __init__(self, img_size, mosaic_ratio=0.25):
        '''
            比较费时
        '''
        self.img_size = img_size
        self.ratio = mosaic_ratio

    def _load_(self, dataset, index):
        ''' load_mosaic_img_and_mask '''
        indexes = [index] + [
            random.randint(0, len(dataset.files_info) - 1) for _ in range(3)
        ]
        datas = [
            dataset.load_data(indexes[0]),
            dataset.load_data(indexes[1]),
            dataset.load_data(indexes[2]),
            dataset.load_data(indexes[3])
        ]
        img_a, mask_a = datas[0]['image'], datas[0]['mask']
        img_b, mask_b = datas[1]['image'], datas[1]['mask']
        img_c, mask_c = datas[2]['image'], datas[2]['mask']
        img_d, mask_d = datas[3]['image'], datas[3]['mask']

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0],
                                        height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0],
                                        height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0],
                                        height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0],
                                        height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return {'image': img, 'mask': mask}

    def __call__(self, dataset, index) -> Any:
        p_ratio = random.random()
        if p_ratio < self.ratio:
            # print("1")
            return self._load_(dataset, index)
        else:
            # print("2")
            return dataset.load_data(index)


#TODO Complet it !
class ImageDataset(Dataset):

    def __init__(
        self,
        data_root,
        tranform=None,
        sub_dirs=None,
        img_dir_name='images',
        sub_dirs_at_before=True,
        traverse_search=False,  # sub_dirs
        error_hint=True,
        #img_suffix='.tif', lable_suffix='.png'):
    ):
        '''
            no lable
        '''
        self.data_root = data_root
        self.idn = img_dir_name
        self.error_hint = error_hint
        self.aug_list = tranform
        self.use_subdir = False if sub_dirs is None else True
        if self.use_subdir:
            self.traverse_search = traverse_search
            self.sub_dirs_at_before = sub_dirs_at_before
            if sub_dirs_at_before:
                self.getimgdir = self._imgdir_1
            else:
                self.getimgdir = self._imgdir_2
            self.sub_dirs = self.list_subdir(sub_dirs)
        else:
            self.getimgdir = self._imgdir_0

        # self.img_ids = self.get_img_ids()

    def __getitem__(self, index):
        #!!!
        pass

    def __len__(self):
        return len(self.img_ids)

    def _imgdir_0(self):
        return osp.join(self.data_root, self.idn)

    def _imgdir_1(self, subdir):
        return osp.join(self.data_root, subdir, self.idn)

    def _imgdir_2(self, subdir):
        return osp.join(self.data_root, self.idn, subdir)

    def list_subdir(self, sub_dirs):
        data_root = self.data_root
        img_dir_name = self.idn
        sub_dirs_at_before = self.sub_dirs_at_before
        traverse_search = self.traverse_search

        def traverse_dir(root, dir_list: set, middir=''):
            dir0 = osp.join(root, middir)
            files = os.listdir(dir0)
            for f in files:
                if osp.isdir(osp.join(dir0, f)):
                    d2 = osp.join(middir, f)
                    dir_list.add(d2)
                    traverse_dir(root, dir_list, d2)

        def traverse_dir_name(root, dir_list: set, stop_name, middir=''):
            dir0 = osp.join(root, middir)
            files = os.listdir(dir0)
            for f in files:
                if f == stop_name:
                    dir_list.add(middir)
                elif osp.isdir(osp.join(dir0, f)):
                    traverse_dir_name(root, dir_list, stop_name, osp.join(middir, f))

        if sub_dirs == 'auto':
            img_sub_dirs = set()
            if sub_dirs_at_before:
                if traverse_search:
                    assert osp.split(img_dir_name)[0] == '', "assert '/' not in img_dir_name"
                    traverse_dir_name(data_root, img_sub_dirs, img_dir_name)
                else:
                    for f in os.listdir(data_root):
                        if osp.exists(osp.join(data_root, f, img_dir_name)):
                            img_sub_dirs.add(f)
            else:
                imdir = osp.join(data_root, img_dir_name)
                if traverse_search:
                    traverse_dir(imdir, img_sub_dirs)
                else:
                    for f in os.listdir(imdir):
                        if osp.isdir(osp.join(imdir, f)):
                            img_sub_dirs.add(f)
            sub_dirs = img_sub_dirs

        elif isinstance(sub_dirs, Iterable):
            for sub_dir in sub_dirs:
                imgdir = self.getimgdir(sub_dir)
                assert osp.exists(imgdir), f"can't find image dir: {imgdir}"
        else:
            raise TypeError("wrong type")

        return list(sub_dirs)





if __name__ == '__main__':
    import time
    t0 = time.time()

    class ACls:
        def collect_files_empty(self, ds, imgdir, labledir, subdir=None):
            print(self, ds)
            # print("!!!collect_files_empty!!!")
            return []

    def collect_files_empty(self, imgdir, labledir, subdir=None):
        # print(self)
        # print("!!!collect_files_empty!!!")
        return []

    cityscapes_dataset = SegmentDataset('data/cityscapes',
                                        None,
                                        'auto',
                                        img_dir_name='leftImg8bit/train',
                                        lable_dir_name='gtFine/train',
                                        sub_dirs_at_before=False,
                                        traverse_search=True,
                                        collect_filesinfo_func=ACls().collect_files_empty)
    print(cityscapes_dataset.sub_dirs)
    print('spend: ', time.time() - t0)

    t0 = time.time()
    cityscapes_dataset = SegmentDataset('data/cityscapes',
                                        None,
                                        'auto',
                                        img_dir_name='leftImg8bit/train',
                                        lable_dir_name='gtFine/train',
                                        sub_dirs_at_before=False,
                                        traverse_search=False,
                                        collect_filesinfo_func=collect_files_empty)
    print('\n', cityscapes_dataset.sub_dirs)
    print('spend: ', time.time() - t0)

    dir1 = cityscapes_dataset.get_image_dir(cityscapes_dataset.sub_dirs[0])
    print(dir1, f'  exist:{osp.exists(dir1)}')

    
    import albumentations as albu
    import jscv.datasets.transform as my_trans

    my_transform = [
        my_trans.RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        my_trans.SmartCropV1(crop_size=512,
                             max_ratio=0.75,
                             ignore_index=255,
                             nopad=False),
        my_trans.ToNumpy()
    ]
    albu_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25,
                                      contrast_limit=0.25,
                                      p=0.25),
        albu.Normalize()
    ]


    LoveDA_dataset = SegmentDataset('data/LoveDA/Train',
                                    [AugCompose(my_transform), albu.Compose(albu_transform)],
                                    'auto',
                                    img_dir_name='images_png',
                                    lable_dir_name='images_png',
                                    sub_dirs_at_before=True,
                                    traverse_search=False,
                                    collect_filesinfo_func=SegmentDataset.collect_filesinfo,
                                    concat_filesinfo_func=SegmentDataset.concat_filesinfo_step_subdir,
                                    load_data_func=MosaicLoad((512, 512), 0.5),
                                    A='aa', B='bb')
    print('\n', LoveDA_dataset.sub_dirs)
    
    
    dir1 = LoveDA_dataset.get_image_dir(LoveDA_dataset.sub_dirs[0])
    print(dir1, f'  exist:{osp.exists(dir1)}')
    print(LoveDA_dataset.files_info[:10])
    from torch.utils.data import DataLoader
    Loader = DataLoader(LoveDA_dataset, 4, shuffle=True)

    Counter0 = 0.
    t0 = time.time()
    for i, data in enumerate(Loader):
        if i > 10:
            break
        print(LoveDA_dataset.get_image_path_batch(data))
    print('spend: ', time.time() - t0, Counter0)


    LoveDA_Test_dataset = ImageDataset('data/LoveDA/Test', None, 'auto',
                                       img_dir_name='images_png',
                                       sub_dirs_at_before=True,
                                       traverse_search=False)

    dir1 = LoveDA_Test_dataset.getimgdir(LoveDA_Test_dataset.sub_dirs[0])
    print('\nLoveDA_Test_dataset', dir1, f'  exist:{osp.exists(dir1)}')