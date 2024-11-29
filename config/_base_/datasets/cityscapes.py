from jscv.datasets.cityscapes_dataset import *
from jscv.utils.cfg import *

dataset_name = 'cityscapes'
data_root = 'data/cityscapes'

dir_train = 'train'
dir_val = 'val'
dir_test = 'val'

max_epoch = 60
train_mosaic_ratio = 0.0
train_crop_size = 512
val_crop_size = None

Dataset_CLS = CityCapesDataset


def create_dataset(cfg):
    Dataset = CityCapesDataset
    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)

    if on_train():
        train_aug, val_aug = get_train_dataset_aug(cfg)
        cfg.train_dataset = Dataset(cfg.data_root, cfg.dir_train, train_aug, cfg.train_mosaic_ratio,
                                    **train_kargs)
        cfg.val_dataset = Dataset(cfg.data_root, cfg.dir_val, val_aug, 0.0, **val_kargs)
    else:
        aug = get_test_dataset_aug(cfg)
        cfg.test_dataset = Dataset(cfg.data_root, cfg.dir_test, aug, 0.0, **test_kargs)
    cfg.description = f"crop={cfg.train_crop_size}, bz={cfg.train_batch_size}"
    if cfg.train_mosaic_ratio > 0:
        cfg.description += f", mosaic={cfg.train_mosaic_ratio}"
