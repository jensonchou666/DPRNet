from jscv.datasets.segment_dataset import SegmentDataset, MosaicLoad
from jscv.datasets.augument import *
from jscv.datasets.augument import test_aug as test_aug_org
import os

'''
In data/cityscapes/leftImg8bit/train/cologne
154/154 1m8s    mean: tensor([0.2839, 0.3210, 0.2769])   std: tensor([0.1843, 0.1874, 0.1826])
In data/cityscapes/leftImg8bit/train/erfurt
109/109 1m53s    mean: tensor([0.2278, 0.2691, 0.2295])   std: tensor([0.1383, 0.1448, 0.1338])
In data/cityscapes/leftImg8bit/train/hamburg
248/248 2m48s    mean: tensor([0.3180, 0.3614, 0.3320])   std: tensor([0.1859, 0.1876, 0.1848])
In data/cityscapes/leftImg8bit/train/monchengladbach
94/94 3m26s    mean: tensor([0.3137, 0.3558, 0.3256])   std: tensor([0.1994, 0.1988, 0.1972])
In data/cityscapes/leftImg8bit/train/krefeld
99/99 8m18s    mean: tensor([0.3048, 0.3493, 0.3207])   std: tensor([0.1936, 0.1950, 0.1947])
In data/cityscapes/leftImg8bit/train/aachen
174/174 11m9s    mean: tensor([0.2250, 0.2647, 0.2240])   std: tensor([0.1361, 0.1418, 0.1316]))
In data/cityscapes/leftImg8bit/train/jena
119/119 12m22s    mean: tensor([0.2662, 0.3089, 0.2640])   std: tensor([0.1636, 0.1659, 0.1597])
In data/cityscapes/leftImg8bit/train/bochum
96/96 13m58s    mean: tensor([0.2707, 0.3182, 0.2811])   std: tensor([0.1536, 0.1577, 0.1559])
In data/cityscapes/leftImg8bit/train/strasbourg
365/365 22m40s    mean: tensor([0.3130, 0.3580, 0.3328])   std: tensor([0.1820, 0.1839, 0.1818])
In data/cityscapes/leftImg8bit/train/ulm
95/95 22m58s    mean: tensor([0.2891, 0.3286, 0.2841])   std: tensor([0.1964, 0.1968, 0.1934])
In data/cityscapes/leftImg8bit/train/hanover
196/196 23m38s    mean: tensor([0.3152, 0.3526, 0.3161])   std: tensor([0.1853, 0.1869, 0.1865])
In data/cityscapes/leftImg8bit/train/zurich
122/122 24m2s    mean: tensor([0.2938, 0.3300, 0.2836])   std: tensor([0.1952, 0.1993, 0.1947]))
In data/cityscapes/leftImg8bit/train/dusseldorf
221/221 24m45s    mean: tensor([0.2205, 0.2567, 0.2125])   std: tensor([0.1400, 0.1463, 0.1350])
In data/cityscapes/leftImg8bit/train/stuttgart
196/196 25m23s    mean: tensor([0.2996, 0.3387, 0.2978])   std: tensor([0.1889, 0.1938, 0.1889])
In data/cityscapes/leftImg8bit/train/tubingen
144/144 26m46s    mean: tensor([0.3061, 0.3537, 0.3105])   std: tensor([0.1867, 0.1911, 0.1869])
In data/cityscapes/leftImg8bit/train/weimar
142/142 28m51s    mean: tensor([0.2837, 0.3307, 0.2926])   std: tensor([0.2053, 0.2112, 0.2100])
In data/cityscapes/leftImg8bit/train/darmstadt
85/85 30m3s    mean: tensor([0.2916, 0.3053, 0.2517])   std: tensor([0.2243, 0.2283, 0.2223]))
In data/cityscapes/leftImg8bit/train/bremen
316/316 33m41s    mean: tensor([0.2682, 0.3142, 0.2715])   std: tensor([0.1742, 0.1764, 0.1686])
Result:
mean: tensor([0.2839, 0.3251, 0.2869])   std: tensor([0.1777, 0.1810, 0.1761])
Accurate:
mean: [0.28389179706573486, 0.3251330554485321, 0.2868955433368683]
std: [0.17772239446640015, 0.18099170923233032, 0.17613644897937775]


1、 In data/cityscapes/leftImg8bit/val/lindau
59/59 11s    mean: tensor([0.2727, 0.2932, 0.2369])   std: tensor([0.1654, 0.1705, 0.1615])
2、 In data/cityscapes/leftImg8bit/val/munster
174/174 49s    mean: tensor([0.2598, 0.3035, 0.2614])   std: tensor([0.1643, 0.1693, 0.1645])
3、 In data/cityscapes/leftImg8bit/val/frankfurt
267/267 1m49s    mean: tensor([0.3218, 0.3618, 0.3296])   std: tensor([0.1977, 0.2018, 0.2015])
Result:
mean: tensor([0.2945, 0.3334, 0.2949])   std: tensor([0.1822, 0.1868, 0.1839])
Accurate:
mean: [0.2944679260253906, 0.3333921432495117, 0.2949221432209015]
std: [0.18224307894706726, 0.18679755926132202, 0.18389944732189178]
'''
train_mean = [0.2839, 0.3251, 0.2869]
train_std = [0.1777, 0.1810, 0.1761]
val_mean = [0.2945, 0.3334, 0.2949]
val_std = [0.1822, 0.1868, 0.1839]

classes = [
    'road', 'sidewalk', 'building', 'wall',
    'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

palett1 = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
           [0, 0, 230], [119, 11, 32]]

palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
           [0, 0, 230], [119, 11, 32]]

CLASSES = classes
num_classes = len(classes)

ignore_index = 255

# H, W
ORIGIN_IMG_SIZE = (1024, 2048)
# H, W
train_crop_size = (512, 512)
val_crop_size = (512, 512)

'''
#TODO
def rgb2label(mask, ignore_index, rgbs=palette):
    h, w = mask.shape[0], mask.shape[1]

    mask_rgb = np.full((h, w), ignore_index, dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    if isinstance(rgbs, dict):
        it0 = iter(rgbs.values())
    else:
        it0 = iter(rgbs)
    for i in range(num_classes):
        mask_rgb[np.all(mask_convert == i, axis=0)] = next(it0)
    return mask_rgb


class MaskFromRGB(object):

    def __init__(self, palette):
        self.palette = palette

    def __call__(self, img, mask):
        pass


class NPRgbToLable(object):

    def __init__(self, palette):
        self.palette = palette

    def __call__(self, img, mask):
        pass
'''

# def train_aug_V(ignore_index, crop_size=512, mean=None, std=None, max_ratio=0.75):
#     return train_aug_simple(ignore_index, crop_size, mean, std, max_ratio)

def train_aug(crop_size=train_crop_size, mean=train_mean, std=train_std):
    if crop_size is None:
        crop_size = ORIGIN_IMG_SIZE
    return train_aug_simple(ignore_index, crop_size, mean, std, max_ratio=0.75)

def val_aug(crop_size=val_crop_size, mean=train_mean, std=train_std):
    return test_aug_org(crop_size, ignore_index, mean, std)

test_aug = val_aug

def CityCapesDataset(
    data_root='data/cityscapes',
    type='train',
    aug_list=None,
    mosaic_ratio=0.25,
    img_size=ORIGIN_IMG_SIZE,
    img_dir='leftImg8bit',
    lable_dir='gtFine',
    img_suffix='_leftImg8bit.png',
    lable_suffix='_gtFine_labelTrainIds.png',
    **kargs,
):
    return SegmentDataset(
        data_root,
        aug_list,
        'auto',
        img_dir_name=os.path.join(img_dir, type),
        lable_dir_name=os.path.join(lable_dir, type),
        img_suffix=img_suffix,
        lable_suffix=lable_suffix,
        sub_dirs_at_before=False,
        traverse_search=False,
        # lable_from_rgb=False,
        # collect_filesinfo_func=SegmentDataset.collect_filesinfo,
        # concat_filesinfo_func=SegmentDataset.concat_filesinfo,
        load_data_func=MosaicLoad(img_size, mosaic_ratio),
        **kargs)





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    print(classes, '\nnum_classes', len(classes))

    loader = DataLoader(
        CityCapesDataset(
            aug_list=train_aug(ignore_index, crop_size=train_crop_size)), 4)

    clss = set()
    for data in loader:
        # print(data['img'].shape, data['gt_semantic_seg'].shape)
        for m in data['gt_semantic_seg']:
            items = 1
            for shp in m.shape:
                items = items * shp
            a = 1
            for v in m.reshape((items)):
                v = int(v)
                if v not in clss:
                    clss.add(v)
                    print(clss)
                    print(data['gt_semantic_seg'].shape)
