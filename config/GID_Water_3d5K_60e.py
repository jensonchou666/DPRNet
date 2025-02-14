from config.datasets.GID_Water_7K import *

dataset_name = 'GID_Water_3d5K'
train_root = 'data/GID_Water_3d5K/train'
val_root = 'data/GID_Water_3d5K/val'
test_root = 'data/GID_Water_3d5K/val'


max_epoch = 60
val_per_k_epoch = 1


ORIGIN_IMG_SIZE = (1024*7//2, 1024*7//2)
VAL_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
TEST_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
