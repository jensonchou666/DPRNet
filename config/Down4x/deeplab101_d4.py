from config.Down4x.r101_d4 import *

model_name = 'deeplab101_d4'


def get_network2(cfg):
    from jscv.hr_models.deeplabv3 import Deeplabv3
    
    net = Deeplabv3(cfg.backbone_name, decoder_args=cfg.decoder_args,
                    aspp_out_channels_rate=1/4, aspp_layers=4)
    net.pretrain_backbone(torch.load(ckpt_path))
    return net