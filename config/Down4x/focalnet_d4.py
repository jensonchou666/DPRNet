from config.Down4x.r101_d4 import *

model_name = 'focalnet_small_d4'   # 4倍下采样


from jscv.hr_models.base_models import *
from config._base_.backbone.focalnet import Focal_S as Focal


class SegNet(EncoderDecoder):
    def __init__(self,
                 decoder_class=FPNDecoder,
                 decoder_args=fpn_decoder_args_1,
    ):
        backbone, self.backbone_ckpt_path, self.backbone_prefix, backbone_features = Focal()

        decoder = decoder_class(backbone_features, **decoder_args)
        super().__init__(backbone, decoder)


def get_network2(cfg):
    net = SegNet(decoder_args=decoder_args)
    from jscv.utils.load_checkpoint import load_checkpoint
    load_checkpoint(net.backbone, net.backbone_ckpt_path, net.backbone_prefix)

    return net