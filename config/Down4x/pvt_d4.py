from config.Down4x.r101_d4 import *

model_name = 'pvt_d4'

from jscv.hr_models.base_models import *
from config._base_.backbone.pvtv2 import *

ckpt_path = 'pretrain/pvt_v2_b1.pth'
backbone_name = 'pvt_v2_b1'

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

def get_network2(cfg):
    net = PVTSegNet(backbone_name, decoder_args=decoder_args)
    from jscv.utils.load_checkpoint import load_checkpoint
    load_checkpoint(net.backbone, ckpt_path)
    return net