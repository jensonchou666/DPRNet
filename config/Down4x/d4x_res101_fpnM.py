from config._base_.models.downsample_x4 import *


from jscv.hr_models.base_models import *
from config._base_.backbone.pvtv2 import *


model_name = 'd4x_res101_fpnM'
backbone_name = 'resnet101'
decoder_args = fpn_decoder_args_M

model_args.update(
    train_setting_list=[((2,2),1/4, 1.)],
)



def get_network(cfg):
    from jscv.hr_models.pathes_segmentor import ResSegNet
    

    decoder_args = cfg.decoder_args
    decoder_args.update(dict(use_decoder_0=False))

    net = ResSegNet(cfg.backbone_name, 
                    backbone_args=dict(return_x0=False), 
                    decoder_args=decoder_args)
    print('backbone_name:', cfg.backbone_name)
    print('decoder_args:', cfg.decoder_args)
    print('ckpt_path:', cfg.ckpt_path)
    print('Use Stem Features:', False)
    net.pretrain_backbone(torch.load(cfg.ckpt_path))
    return net

# fpn_decoder_args_512    blocks=[1,1,3,1]
# FPS: 27.948935385332174    3.58ms





'''
    3.5K

'''