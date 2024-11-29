from config.DPRNet.DPRNet_GR101_LR18 import *
from config._base_.backbone.focalnet import Focal_S as Focal

model_name = 'DPRNet_focal_LR18'

global_net_ckpt_dict['stage1_GNet_pretrained'] = 'Your ckpt path (focalnet_d4)'

def get_g_l_nets(cfg):
    GB, cfg.backbone_ckpt_path, cfg.backbone_prefix, embed_dims = Focal()
    return replace_GB(cfg, GB, embed_dims)


def load_global_net_ckpt(cfg, model):
    if cfg.global_net_ckpt_from == 'stage1_GNet_pretrained':
        fn = global_net_ckpt_dict[cfg.global_net_ckpt_from]
        D = torch.load(fn)
        load_checkpoint(model.global_encoder, D, 'net.backbone')
        load_checkpoint(model.global_decoder, D, 'net.decoder')
        print("load GB from", fn)
    else:
        load_checkpoint(net.backbone, net.backbone_ckpt_path, net.backbone_prefix)
        print("load GB from", net.backbone_ckpt_path, net.backbone_prefix)


