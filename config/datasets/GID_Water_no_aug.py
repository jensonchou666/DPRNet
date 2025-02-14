from jscv.datasets.gid_water import *
from jscv.utils.cfg import *
import cv2


dataset_name = 'GID_Water'
train_root = 'data/gid_water/train'  # train_val'
val_root = 'data/gid_water/val'
test_root = 'data/gid_water/val'

max_epoch = 80
val_per_k_epoch = 1


def train_aug():
    aug = [SegCompose([ImgConvertRGB, ToNumpy])]
    extend_albu_aug(aug, [albu_aug_norm(train_mean, train_std)])
    return aug

def val_aug():
    aug = [SegCompose([ImgConvertRGB, ToNumpy])]
    extend_albu_aug(aug, [albu_aug_norm(train_mean, train_std)])
    return aug

def create_dataset(cfg):

    Dataset = GID_Water_Dataset

    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)
    
    if on_train():
        cfg.train_dataset = Dataset(cfg.train_root, train_aug(),
                                    mean=train_mean, std=train_std,
                                    **train_kargs)
        cfg.val_dataset = Dataset(cfg.val_root, val_aug(),
                                    mean=train_mean, std=train_std,
                                    **val_kargs)
    else:
        cfg.test_dataset = Dataset(cfg.test_root, val_aug(),
                                    mean=train_mean, std=train_std,
                                    **test_kargs)
