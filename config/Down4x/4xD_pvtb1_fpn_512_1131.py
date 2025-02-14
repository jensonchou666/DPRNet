from config._base_.models.downsample_x4 import *


from jscv.hr_models.base_models import *
from config._base_.backbone.pvtv2 import *


model_name = '4xD_pvtb1_fpn_512_1131'
backbone_name = 'pvt_v2_b1'
ckpt_path = 'pretrain/pvt_v2_b1.pth'
decoder_args = fpn_decoder_args_512_1131

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
7K
    fpn_decoder_args_2
    b1 FPS: 5.3595538015668716    18.66ms
    b2 FPS: 2.8576829529730268    34.99ms

    fpn_decoder_args_1
    b1 FPS: 5.3595538015668716    18.13ms
    b2 FPS: 2.8576829529730268    34.25mms
'''


'''
3.5K
    fpn_decoder_args_256
    FPS: 31.04065083249373    3.22ms

    fpn_decoder_args_512
    FPS: 28.9541182970966    3.45ms
    
    fpn_decoder_args_512    blocks=[1,2,3,2]
    FPS: 26.952570256029574    3.71ms
    
    fpn_decoder_args_512    blocks=[1,1,3,2]
    FPS: 27.502413663499244    3.64ms



    fpn_decoder_args_512    blocks=[1,1,2,2]
    FPS: 28.015105954081       3.57ms
    
    fpn_decoder_args_512    blocks=[1,1,2,1]
    FPS: 28.452066084868385    3.51ms

    fpn_decoder_args_512    blocks=[1,1,1,2]
    FPS: 28.64199527229639    3.49ms
    
    --------------------
    B2

    fpn_decoder_args_512    blocks=[1,1,3,1]
    FPS: 17.086574584010357    5.85ms
'''