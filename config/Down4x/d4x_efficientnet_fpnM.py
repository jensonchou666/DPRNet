from config._base_.models.downsample_x4 import *


from jscv.hr_models.base_models import *
from jscv.backbone.efficientnet.efficientnet import *


# model_name = 'd4x-efficientnet_b6-fpnM'
# backbone_name = 'efficientnet-b6'
# channels = [32, 40, 72, 200, 576]
# backbone_ckpt = 'pretrain/efficientnet-b6-c76e70fd.pth'

model_name = 'd4x-efficientnet_b3-fpnM'
backbone_name = 'efficientnet-b3'
channels = [24, 32, 48, 136, 384]
backbone_ckpt = 'pretrain/efficientnet-b3-5fb5a3c3.pth'

decoder_args = fpn_decoder_args_M


model_args.update(
    train_setting_list=[((1,1),1/4, 0.8), ((2,2),1/2, 0.2)],
    # save_images_args=dict(per_n_step=100, save_img=True, save_wrong=True),
)


class SegNet(EncoderDecoder):
    def __init__(self,
                 backbone_name='efficientnet-b6',
                 channels=channels,
                 backbone_ckpt=backbone_ckpt,
                 decoder_class=FPNDecoder,
                 decoder_args=fpn_decoder_args_M,
    ):
        backbone = EfficientNet.from_pretrained(backbone_name, backbone_ckpt)
        print(backbone_name)
        decoder_args.update(dict(
            use_decoder_0=True,
            return_context=False,
        ))
        decoder = decoder_class(channels, **decoder_args)
        super().__init__(backbone, decoder)

def get_network(cfg):
    net = SegNet(backbone_name, channels=channels, backbone_ckpt=backbone_ckpt, decoder_args=decoder_args)
    return net






# fpn_decoder_args_512    blocks=[1,1,3,1]
# FPS: 27.948935385332174    3.58ms





'''
    3.5K

'''