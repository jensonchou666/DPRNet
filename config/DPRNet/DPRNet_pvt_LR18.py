from config.DPRNet.DPRNet_GR101_LR18 import *
from config._base_.backbone.pvtv2 import *


model_name = 'DPRNet_pvt_LR18'

pvt_ckpt = 'pretrain/pvt_v2_b1.pth'

global_net_ckpt_dict['stage0'] = pvt_ckpt
global_net_ckpt_dict['stage1_GNet_pretrained'] = 'Your ckpt path (pvt_d4)'


def get_g_l_nets(cfg):
    GB = backbone_pvtv2_b1()
    return replace_GB(cfg, GB, GB.embed_dims)


def load_global_net_ckpt(cfg, model):
    fn = global_net_ckpt_dict[cfg.global_net_ckpt_from]
    D = torch.load(fn)
    if cfg.global_net_ckpt_from == 'stage1_GNet_pretrained':
        load_checkpoint(model.global_encoder, D, 'net.backbone')
        load_checkpoint(model.global_decoder, D, 'net.decoder')
    else:
        load_checkpoint(model.global_encoder, D)
    print("load GB from", fn)
