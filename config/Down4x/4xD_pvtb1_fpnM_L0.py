from config._base_.models.downsample_x4 import *


from jscv.hr_models.base_models import *
from config._base_.backbone.pvtv2 import *


model_name = '4xD_pvtb1_fpnM'
backbone_name = 'pvt_v2_b1'
ckpt_path = 'pretrain/pvt_v2_b1.pth'
decoder_args = fpn_decoder_args_M

model_args.update(
    train_setting_list=[((2,2),1/4, 1.)],
)


class PVTSegNet(EncoderDecoder):
    def __init__(self,
                 backbone_name='pvt_v2_b1',
                 decoder_class=FPNDecoder,
                 decoder_args=fpn_decoder_args_1,
                 features_only=True,
    ):
        if backbone_name == 'pvt_v2_b1':
            backbone = backbone_pvtv2_b1()
        print(backbone_name)
        l_chs = backbone.embed_dims
        decoder = decoder_class(l_chs, **decoder_args)
        super().__init__(backbone, decoder)

def get_network(cfg):
    net = PVTSegNet(backbone_name, decoder_args=decoder_args)
    from jscv.utils.load_checkpoint import load_checkpoint
    load_checkpoint(net.backbone, ckpt_path)
    return net






# fpn_decoder_args_512    blocks=[1,1,3,1]
# FPS: 27.948935385332174    3.58ms





'''
    3.5K

'''