from config.DPRNet.DPRNet_GR101_LR18 import *

model_name = 'DPRNet_GR101_noaspp_LR18'

global_net_ckpt_dict['stage1_GNet_pretrained'] = 'Your ckpt path (r101_d4)'

def get_g_l_nets(cfg):
    GB = ResNetEncoder('resnet101', features_only=True)
    return replace_GB(cfg, GB, GB.channels)




