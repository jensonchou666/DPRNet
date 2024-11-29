
#use:
 #'train.py config/DPRNet/DPRNet_GR101_LR18.py -c dataset.py config/DPRNet/ex_32x32.py ...'

local_patches           = (32,32)
global_patches          = {'train':(2,4), 'val':(1,2)}
local_batch_size        = {'train': 6, 'val': 2}


def after_create_model(cfg):
    n = cfg.model_name
    if '8x8' in n: n.replace('8x8', '32x32')
    if '16x16' in n: n.replace('16x16', '32x32')
    cfg.model_name = n
    print("altered to 32 x 32 local_patches")


# def create_model(cfg):
#     n = cfg.model_name
#     if '8x8' in n: n.replace('8x8', '32x32')
#     if '16x16' in n: n.replace('16x16', '32x32')
#     cfg.model_name = n
#     return super.create_model #? 配置文件直接覆盖的局限
