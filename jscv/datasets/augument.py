from jscv.datasets.transform import *
import albumentations as albu

def albu_aug_compose(albu_aug_list: list):
    L2 = []
    for x in albu_aug_list:
        if x is not None:
            L2.append(x)
    return albu.Compose(L2)


def extend_albu_aug(aug_list: list, albu_aug_list: list):
    L2 = []
    for x in albu_aug_list:
        if x is not None:
            L2.append(x)
    if len(L2) > 0:
        aug_list.append(albu.Compose(L2))

def albu_aug_norm(mean=None, std=None):
    if mean is not None and std is not None:
        return albu.Normalize(mean, std, max_pixel_value=255)
    return None

def train_aug_normal(ignore_index, crop_size=512,
                     mean=None, std=None,
                     max_ratio=0.75, nopad=False):

    my_transform = SegCompose([
        ImgConvertRGB,
        RandomScaleV2(scale_list=[0.75, 1.0, 1.25, 1.5],
                      min_hw=crop_size, mode='value'),
        SmartCropV1(crop_size=crop_size,
                    max_ratio=max_ratio,
                    ignore_index=ignore_index,
                    nopad=nopad),
        ToNumpy
    ])

    albu_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25,
                                      contrast_limit=0.25,
                                      p=0.25),
        # albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu_aug_norm(mean, std)
    ]

    return [my_transform, albu_aug_compose(albu_transform)]


def train_aug_simple(ignore_index, crop_size=512,
                     mean=None, std=None,
                     max_ratio=0.75, nopad=False):

    my_transform = SegCompose([
        ImgConvertRGB,
        RandomScaleV2(scale_list=[0.75, 1.0, 1.25, 1.5],
                      min_hw=crop_size, mode='value'),
        RandomCrop(crop_size, ignore_index, False),
        ToNumpy
    ])

    albu_transform = [
        albu.HorizontalFlip(p=0.2),
        # albu.VerticalFlip(p=0.2),
        albu_aug_norm(mean, std)
    ]
    return [my_transform, albu_aug_compose(albu_transform)]



def test_aug_simple(mean=None, std=None):
    aug = [SegCompose([ImgConvertRGB, ToNumpy])]

    albu_aug_list = []
    albu_aug_list.append(albu_aug_norm(mean, std))
    extend_albu_aug(aug, albu_aug_list)

    return aug


def test_aug_crop(ignore_index, crop_size=512, mean=None, std=None):
    a = SegCompose([ImgConvertRGB, RandomCrop(crop_size, ignore_index, False), ToNumpy])
    aug = [a]

    albu_aug_list = []
    albu_aug_list.append(albu_aug_norm(mean, std))
    extend_albu_aug(aug, albu_aug_list)

    return aug


def test_aug(crop_size, ignore_index, mean=None, std=None):
    if crop_size is None:
        return test_aug_simple(mean, std)
    else:
        return test_aug_crop(ignore_index, crop_size, mean, std)

import numpy

class SingleClass:
    def __init__(self, class_id, ignore_idx=None):
        '''
            data: numpy 格式
            class_id 置0
            其它置 1
            ignore_idx 置 2
        '''
        self.class_id = class_id
        self.ignore_idx = ignore_idx

    def __call__(self, **data):
        mask = data['mask']
        m2 = (mask != self.class_id).astype(mask.dtype)
        if self.ignore_idx is not None:
            m2[mask == self.ignore_idx] = 2
        data['mask'] = m2
        return data
