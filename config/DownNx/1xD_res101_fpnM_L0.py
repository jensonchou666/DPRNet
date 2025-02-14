from config._base_.models.downsample_x4 import *


from jscv.hr_models.base_models import *
from config._base_.backbone.pvtv2 import *


# Stem后的 GB0 会参与到融合

model_name = '1xD_res101_fpnM_L0'
backbone_name = 'resnet101'
decoder_args = fpn_decoder_args_M

model_args.update(
    train_setting_list=[((4,4),1, 1.)],
    val_setting=((2,2), 1),
)



# fpn_decoder_args_512    blocks=[1,1,3,1]
# FPS: 27.948935385332174    3.58ms





'''
    3.5K

'''