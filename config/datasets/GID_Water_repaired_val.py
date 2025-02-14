from config.datasets.GID_Water_3d5K import *

dataset_name = 'GID_Water_3d5K_repaired_val'


def create_dataset(cfg):
    
    Dataset = GID_Water_Dataset

    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)
    
    if on_train():
        cfg.train_dataset = Dataset(cfg.train_root, cfg.train_aug(cfg),
                                    lable_dir_name='label_repaired',
                                    mean=train_mean, std=train_std,
                                    **train_kargs)
        cfg.val_dataset = Dataset(cfg.val_root, cfg.val_aug(cfg),
                                  lable_dir_name='label_repaired',
                                    mean=train_mean, std=train_std,
                                    **val_kargs)
    else:
        cfg.test_dataset = Dataset(cfg.test_root, cfg.val_aug(cfg),
                                    mean=train_mean, std=train_std,
                                    **test_kargs)

