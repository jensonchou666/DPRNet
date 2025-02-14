from jscv.datasets.ade20k import *
from jscv.utils.cfg import *

dataset_name = 'ade20k'
data_root = 'data/ade/ADEChallengeData2016'

dir_train = 'training'
dir_val = 'validation'
dir_test = 'validation'

max_epoch = 25
train_crop_size = 512
val_crop_size = None

Dataset_CLS = ADE20KDataset


def create_dataset(cfg):
    Dataset = ADE20KDataset
    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)
    if on_train():
        train_aug, val_aug = get_train_dataset_aug(cfg)
        cfg.train_dataset = Dataset(cfg.data_root, cfg.dir_train, train_aug, **train_kargs)
        cfg.val_dataset = Dataset(cfg.data_root, cfg.dir_val, val_aug, **val_kargs)
    else:
        aug = get_test_dataset_aug(cfg)
        cfg.test_dataset = Dataset(cfg.data_root, cfg.dir_test, aug, **test_kargs)
    
    cfg.description = f"crop={cfg.train_crop_size}, bz={cfg.train_batch_size}"



# datasets_pack = partial(create_datasets_pack,
#                         create_func=create_dataset,
#                         train_crop_size=train_crop_size,
#                         val_crop_size=val_crop_size)
