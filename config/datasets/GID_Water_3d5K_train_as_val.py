from config.datasets.GID_Water_7K import *


'''
    用训练集 替换验证集， 是我用来 skip_train 然后 保存训练集里的图像预测的，不能用来正常训练
'''

dataset_name = 'GID_Water_3d5K'
train_root = 'data/GID_Water_3d5K/val'
val_root = 'data/GID_Water_3d5K/train'
test_root = 'data/GID_Water_3d5K/val'


max_epoch = 40
val_per_k_epoch = 1

# train_mean = [0.3405, 0.3668, 0.3507]
# train_std = [0.2251, 0.2308, 0.2380]

#TODO 忘覆盖了
# train_mean = [0.3532, 0.3710, 0.3457]
# train_std = [0.2279, 0.2215, 0.2142]


ORIGIN_IMG_SIZE = (1024*7//2, 1024*7//2)
VAL_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
TEST_ORIGIN_IMG_SIZE = ORIGIN_IMG_SIZE
