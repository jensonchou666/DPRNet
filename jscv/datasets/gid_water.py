from jscv.datasets.segment_dataset import SegmentDataset, MosaicLoad
from jscv.datasets.augument import *
from jscv.datasets.augument import test_aug as test_aug_org


import jscv.utils.analyser as Ana

'''
全150张
mean: tensor([0.3435, 0.3699, 0.3505])   std: tensor([0.1869, 0.1978, 0.2063])
Accurate:
mean: [0.34348589181900024, 0.36992815136909485, 0.3505106270313263]
std: [0.18691539764404297, 0.1977958381175995, 0.20627520978450775]

train 105张
mean: tensor([0.3405, 0.3668, 0.3507])   std: tensor([0.2251, 0.2308, 0.2380])
Accurate:
mean: [0.34053924679756165, 0.366849809885025, 0.35073766112327576]
std: [0.22507409751415253, 0.23081770539283752, 0.2379624843597412]
'''



train_mean = [0.3405, 0.3668, 0.3507]
train_std = [0.2251, 0.2308, 0.2380]

rgb_dict = {
    'others': Ana.黑,
    'water': Ana.白
}
classes = list(rgb_dict.keys())
num_classes = len(classes)
ignore_index = 100
ORIGIN_IMG_SIZE = (1024*7, 1024*7)
VAL_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
TEST_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
train_crop_size = None
val_crop_size = None

def train_aug():
    return train_aug_simple(-100, train_crop_size, train_mean, train_std)

def val_aug():
    return test_aug_simple(train_mean, train_std)



def GID_Water_Dataset(
    data_root='data/water_seg/train',
    aug_list=None,
    img_dir_name='image',
    lable_dir_name='label',
    img_suffix='.tif',
    lable_suffix='_label.tif',
    **kargs,
):
    return  SegmentDataset(
        data_root,
        aug_list,
        None,
        img_dir_name=img_dir_name,
        lable_dir_name=lable_dir_name,
        img_suffix=img_suffix,
        lable_suffix=lable_suffix,
        sub_dirs_at_before=True,
        traverse_search=False,
        # lable_from_rgb=False,
        # collect_filesinfo_func=SegmentDataset.collect_filesinfo,
        # concat_filesinfo_func=SegmentDataset.concat_filesinfo,
        **kargs) 

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = GID_Water_Dataset('data/gid_water/val')
    loader = DataLoader(dataset)
    print(len(dataset))
    import tqdm
    for data in tqdm.tqdm(loader):
        print(data["id"][0])
