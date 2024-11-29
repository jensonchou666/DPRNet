from jscv.datasets.segment_dataset import SegmentDataset, MosaicLoad
from jscv.datasets.augument import *
from jscv.datasets.augument import test_aug as test_aug_org

import os

import jscv.utils.analyser as Ana

'''
Train
Result:
mean: tensor([0.3331, 0.3597, 0.3364])   std: tensor([0.1295, 0.1251, 0.1276])
Accurate:
mean: [0.3331249952316284, 0.3596719801425934, 0.3364335894584656]
std: [0.12954680621623993, 0.12513846158981323, 0.12760326266288757]
'''

rgb_dict = {
    'ImSurf': Ana.白,
    'Building': Ana.蓝,
    'LowVeg': Ana.黄,
    'Tree': Ana.绿,
    'Car': Ana.紫,
    'Clutter': Ana.青
}
classes = list(rgb_dict.keys())
num_classes = len(classes)
ignore_index = len(classes)

ORIGIN_IMG_SIZE = (1024, 1024)

train_crop_size = 768
val_crop_size = None

train_mean = [0.3331, 0.3597, 0.3364]
train_std = [0.1295, 0.1251, 0.1276]
val_mean = None
val_std = None


def train_aug(crop_size=train_crop_size, mean=train_mean, std=train_std):
    if crop_size is None:
        crop_size = ORIGIN_IMG_SIZE
    return train_aug_simple(ignore_index, crop_size, mean, std, max_ratio=0.75)

def val_aug(crop_size=val_crop_size, mean=train_mean, std=train_std):
    return test_aug_org(crop_size, ignore_index, mean, std)

test_aug = val_aug


def PotsdamDataset(data_root='data/potsdam/Train',
                   aug_list=None,
                   mosaic_ratio=0.25,
                   img_size=ORIGIN_IMG_SIZE,
                   img_dir_name='images',
                   lable_dir_name='masks', **kargs):
    return SegmentDataset(
        data_root,
        aug_list,
        None,
        img_dir_name=img_dir_name,
        lable_dir_name=lable_dir_name,
        img_suffix='.tif',
        lable_suffix='.png',
        sub_dirs_at_before=True,
        traverse_search=False,
        # collect_filesinfo_func=SegmentDataset.collect_filesinfo,
        # concat_filesinfo_func=SegmentDataset.concat_filesinfo,
        load_data_func=MosaicLoad(img_size, mosaic_ratio),
        **kargs)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    loader = DataLoader(PotsdamDataset('data/potsdam/train',
                                       train_aug_normal(ignore_index, train_crop_size)), 4)
    for data in loader:
        print(data.keys())