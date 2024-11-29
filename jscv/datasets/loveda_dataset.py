from jscv.datasets.segment_dataset import SegmentDataset, MosaicLoad
from jscv.datasets.augument import *
from jscv.datasets.augument import test_aug as test_aug_org
import os

import jscv.utils.analyser as Ana


rgb_dict = {
    'background': Ana.巧克力,
    'agricultural': Ana.小麦,
    'building': Ana.蓝,
    'road': Ana.白,
    'water': Ana.青,
    'barren': Ana.亮粉,  # Ana.棕,
    'forest': Ana.绿
}

classes = list(rgb_dict.keys())
num_classes = len(classes)
ignore_index = len(classes)

ORIGIN_IMG_SIZE = (1024, 1024)

train_crop_size = 512
val_crop_size = None

train_mean = None
train_std = None
val_mean = None
val_std = None



def train_aug(crop_size=train_crop_size, mean=train_mean, std=train_std):
    if crop_size is None:
        crop_size = ORIGIN_IMG_SIZE
    return train_aug_simple(ignore_index, crop_size, mean, std, max_ratio=0.75)

def val_aug(crop_size=val_crop_size, mean=train_mean, std=train_std):
    return test_aug_org(crop_size, ignore_index, mean, std)

test_aug = val_aug

def LoveDADataset(data_root='data/LoveDA/Train',
                  aug_list=None,
                  mosaic_ratio=0.25,
                  img_size=ORIGIN_IMG_SIZE,
                  img_dir_name='images_png',
                  lable_dir_name='masks_png',
                  **kwargs):
    return SegmentDataset(
        data_root,
        aug_list,
        'auto',
        img_dir_name=img_dir_name,
        lable_dir_name=lable_dir_name,
        sub_dirs_at_before=True,
        traverse_search=False,
        # collect_filesinfo_func=SegmentDataset.collect_filesinfo,
        # concat_filesinfo_func=SegmentDataset.concat_filesinfo,
        load_data_func=MosaicLoad(img_size, mosaic_ratio), **kwargs)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    loader = DataLoader(LoveDADataset('data/LoveDA/Train', train_aug_normal(ignore_index)), 4)
    for data in loader:
        print(data.keys())
    LoveDADataset('data/LoveDA/Val', test_aug_simple(), 0.0)