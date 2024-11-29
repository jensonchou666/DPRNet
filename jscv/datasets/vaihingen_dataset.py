from jscv.datasets.segment_dataset import SegmentDataset, MosaicLoad
from jscv.datasets.augument import *
from jscv.datasets.augument import test_aug as test_aug_org


import jscv.utils.analyser as Ana
'''
Train
Result:
mean: tensor([0.3187, 0.3204, 0.4805])   std: tensor([0.1405, 0.1448, 0.2086])
Accurate:
mean: [0.31867149472236633, 0.32039061188697815, 0.48048287630081177]
std: [0.14052337408065796, 0.14478902518749237, 0.20859257876873016]
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
VAL_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
TEST_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE



train_crop_size = 512
val_crop_size = None

train_mean = [0.3187, 0.3204, 0.4805]
train_std = [0.1405, 0.1448, 0.2086]
val_mean = None
val_std = None

def train_aug(crop_size=train_crop_size, mean=train_mean, std=train_std):
    if crop_size is None:
        crop_size = ORIGIN_IMG_SIZE
    return train_aug_simple(ignore_index, crop_size, mean, std, max_ratio=0.75)

def val_aug(crop_size=val_crop_size, mean=train_mean, std=train_std):
    return test_aug_org(crop_size, ignore_index, mean, std)

test_aug = val_aug


def VaihingenDataset(data_root='data/vaihingen/train',
                     aug_list=None,
                     mosaic_ratio=0.25,
                     img_size=ORIGIN_IMG_SIZE,
                     img_dir_name='images',
                     lable_dir_name='masks',
                     **kargs):
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
    loader = DataLoader(VaihingenDataset(aug_list=train_aug_normal(ignore_index, train_crop_size)), 4)
    for data in loader:
        print(data.keys())