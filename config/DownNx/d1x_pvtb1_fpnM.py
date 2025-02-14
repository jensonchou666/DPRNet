from config._base_.models.downsample_x4 import *


from jscv.hr_models.base_models import *
from config._base_.backbone.pvtv2 import *


model_name = 'd1x_pvtb1_fpnM'
backbone_name = 'pvt_v2_b1'
ckpt_path = 'pretrain/pvt_v2_b1.pth'
decoder_args = fpn_decoder_args_M

# pretrained_d4x_path = 'work_dir/GID_Water_3d5K/d4x_pvtb1_fpnM-e40/ckpt/epoch=6@val_mIoU=89.38@d4x_pvtb1_fpnM-GID_Water_3d5K.ckpt'
# pretrained_d4x_path = 'work_dir/GID_Water_3d5K/d4x_pvtb1_fpnM-e40/final/epoch=29@val_mIoU=91.23@4xD_pvtb1_fpnM-GID_Water_3d5K.ckpt'

model_args.update(
    train_setting_list=[((4,4),1, 1.)],
    val_setting=((4,2), 1),
)


class PVTSegNet(EncoderDecoder):
    def __init__(self,
                 backbone_name='pvt_v2_b1',
                 decoder_class=FPNDecoder,
                 decoder_args=fpn_decoder_args_M,
                 features_only=True,
    ):
        if backbone_name == 'pvt_v2_b1':
            backbone = backbone_pvtv2_b1()
        print(backbone_name)
        l_chs = backbone.embed_dims
        decoder = decoder_class(l_chs, return_context=False, **decoder_args)
        super().__init__(backbone, decoder)

def get_network(cfg):
    net = PVTSegNet(backbone_name, decoder_args=decoder_args)
    from jscv.utils.load_checkpoint import load_checkpoint
    load_checkpoint(net.backbone, ckpt_path) # 训练太慢了
    # load_checkpoint(net, pretrained_d4x_path, 'net.')
    return net






# fpn_decoder_args_512    blocks=[1,1,3,1]
# FPS: 27.948935385332174    3.58ms





'''
    3.5K
'''