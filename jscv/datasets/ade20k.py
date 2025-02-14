from jscv.datasets.segment_dataset import SegmentDataset, MosaicLoad
from jscv.datasets.augument import *
import os

import jscv.utils.analyser as Ana

mean_shape = (416.64482758620693, 516.0714285714287)
min_h, min_w, max_h, max_w = 96, 130, 2100, 2100

'''
Train:
mean: [0.42939531803131104, 0.46548423171043396, 0.4889712631702423]
std: [0.24039067327976227, 0.2294204831123352, 0.2284674048423767]

val
mean: [0.4275532066822052, 0.46394553780555725, 0.488803505897522]
std: [0.24168235063552856, 0.23106889426708221, 0.2298356145620346]
'''
train_mean = [0.4294, 0.4655, 0.4890]
train_std = [0.2404, 0.2294, 0.2285]
val_mean = [0.4276, 0.4639, 0.4888]
val_std = [0.2417, 0.2311, 0.2298]



classes = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
           'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person',
           'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
           'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea',
           'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk',
           'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base',
           'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand',
           'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
           'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
           'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind',
           'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
           'countertop', 'stove', 'palm', 'kitchen island', 'computer',
           'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus',
           'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
           'streetlight', 'booth', 'television receiver', 'airplane',
           'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator',
           'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship',
           'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
           'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
           'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
           'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
           'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
           'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier',
           'crt screen', 'plate', 'monitor', 'bulletin board', 'shower',
           'radiator', 'glass', 'clock', 'flag')
palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
           [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
           [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
           [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
           [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
           [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
           [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
           [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
           [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
           [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
           [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
           [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
           [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
           [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
           [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
           [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
           [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
           [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
           [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
           [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
           [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
           [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
           [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
           [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
           [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
           [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
           [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
           [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
           [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
           [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
           [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
           [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
           [102, 255, 0], [92, 0, 255]]

num_classes = len(classes)
ignore_index = len(classes)

#ORIGIN_IMG_SIZE = (1024, 1024)

train_crop_size = (512, 512)
val_crop_size = (512, 512)


def train_aug(
        crop_size=train_crop_size,
        #ignore_index=ignore_index,
        mean=train_mean, std=train_std):
    my_transform = SegCompose([
        ImgConvertRGB,
        RandomScaleV2(scale_list=[0.75, 1.0, 1.25, 1.5],
                      min_hw=crop_size, mode='value'),
        RandomCrop(crop_size, ignore_index, False),
        ToNumpy
    ])
    albu_transform = [
        albu.HorizontalFlip(p=0.2),
        albu.Normalize(mean, std)
    ]
    return [my_transform, albu_aug_compose(albu_transform)]

def val_aug(crop_size=val_crop_size, mean=train_mean, std=train_std):
    if crop_size is None:
        return test_aug_simple(mean, std)
    else:
        return test_aug_crop(ignore_index, crop_size, mean, std)

test_aug = val_aug



def ADE20KDataset(
        data_root='data/ade/ADEChallengeData2016',
        type='training',
        aug_list=None,
        #   mosaic_ratio=0.0,
        img_dir='images',
        lable_dir='annotations',
        img_suffix='.jpg',
        lable_suffix='.png',
        **kargs):
    return SegmentDataset(
        data_root,
        aug_list,
        None,
        img_dir_name=os.path.join(img_dir, type),
        lable_dir_name=os.path.join(lable_dir, type),
        img_suffix=img_suffix,
        lable_suffix=lable_suffix,
        sub_dirs_at_before=True,
        traverse_search=False,
        **kargs
        # collect_filesinfo_func=SegmentDataset.collect_filesinfo,
        # concat_filesinfo_func=SegmentDataset.concat_filesinfo,
    )


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    loader = DataLoader(ADE20KDataset(aug_list=train_aug()), 4)
    for data in loader:
        print(data['img'].shape)