from config.datasets.GID_Water import *

dataset_name = 'GID_Water_repaired'
lable_dir_name = 'label_repaired'



def create_dataset(cfg):
    Dataset = GID_Water_Dataset
    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)
    if on_train():
        cfg.train_dataset = Dataset(cfg.train_root, train_aug(),
                                    lable_dir_name=lable_dir_name,
                                    mean=train_mean, std=train_std,
                                    **train_kargs)
        cfg.val_dataset = Dataset(cfg.val_root, val_aug(),
                                    mean=train_mean, std=train_std,
                                    **val_kargs)
    else:
        cfg.test_dataset = Dataset(cfg.test_root, val_aug(),
                                    mean=train_mean, std=train_std,
                                    **test_kargs)
