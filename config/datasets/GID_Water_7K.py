from jscv.datasets.gid_water import *
from jscv.utils.cfg import *
import cv2


dataset_name = 'GID_Water_7K'
train_root = 'data/gid_water/train'
val_root = 'data/gid_water/val'
test_root = 'data/gid_water/val'

max_epoch = 80
val_per_k_epoch = 1

'''
    全图 7K x 7K,  不要用这个文件了

    并不是因为我们的模型无法处理7K图像，相反我们设计的模型实际上能灵活的分配补丁适应
    任何超级无敌高的分辨率

    高分2号里的7K图像之间差异非常大，如果不分图， 无论如何划分数据集，验证集都达不到良好准确率，
    并且总是非常不稳定， 完全无法用来测试模型的好坏。

    在这一点我们吃了巨大的亏， 之前绝大部分的时间里我们都是直接对7K图像
    训练和验证的，

'''


def train_aug(cfg):
    aug = [SegCompose([ImgConvertRGB, ToNumpy])]
    albu_aug = [
        albu.HorizontalFlip(p=0.4),
        albu.RandomRotate90(p=0.5),
        # albu.Rotate(limit=180, p=0.5, border_mode=cv2.BORDER_CONSTANT, mask_value=ignore_index,),
        albu.ShiftScaleRotate(rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
                              mask_value=ignore_index, p=0.5),
        albu.RandomResizedCrop(*cfg.ORIGIN_IMG_SIZE, scale=(0.75, 1), p=0.5),
        albu_aug_norm(train_mean, train_std), #TODO cfg. 忘了加
    ]
    extend_albu_aug(aug, albu_aug)
    return aug
def val_aug(cfg):
    aug = [SegCompose([ImgConvertRGB, ToNumpy])]
    extend_albu_aug(aug, [albu_aug_norm(train_mean, train_std)])
    return aug

def create_dataset(cfg):

    Dataset = GID_Water_Dataset

    train_kargs, val_kargs, test_kargs = get_dataset_kwargs(cfg)
    
    if on_train():
        cfg.train_dataset = Dataset(cfg.train_root, cfg.train_aug(cfg),
                                    # lable_dir_name='label_refined',
                                    mean=train_mean, std=train_std,
                                    **train_kargs)
        cfg.val_dataset = Dataset(cfg.val_root, cfg.val_aug(cfg),
                                    mean=train_mean, std=train_std,
                                    **val_kargs)
    else:
        cfg.test_dataset = Dataset(cfg.test_root, cfg.val_aug(cfg),
                                    mean=train_mean, std=train_std,
                                    **test_kargs)
