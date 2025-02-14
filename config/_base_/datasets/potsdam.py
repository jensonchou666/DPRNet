from jscv.datasets.potsdam_dataset import *
from jscv.utils.cfg import *

dataset_name = 'potsdam'
train_root = 'data/potsdam/train'  # train_val'
val_root = 'data/potsdam/test'
test_root = 'data/potsdam/test'
max_epoch = 45

test_dataset_exist_mask = False
train_crop_size = 512
val_crop_size = None

train_mosaic_ratio = 0.0

Dataset_CLS = PotsdamDataset

def create_dataset(cfg):

    Dataset = PotsdamDataset

    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)
    
    if on_train():
        train_aug, val_aug = get_train_dataset_aug(cfg)
        cfg.train_dataset = Dataset(cfg.train_root, train_aug, cfg.train_mosaic_ratio, **train_kargs)
        cfg.val_dataset = Dataset(cfg.val_root, val_aug, 0.0, **val_kargs)
    else:
        aug = get_test_dataset_aug(cfg)
        cfg.test_dataset = Dataset(cfg.test_root, aug, 0.0, **test_kargs)
    cfg.description = f"crop={cfg.train_crop_size}, bz={cfg.train_batch_size}"
    if cfg.train_mosaic_ratio > 0:
        cfg.description += f", mosaic={cfg.train_mosaic_ratio}"