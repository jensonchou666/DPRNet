from config._base_.models.ISDNet import *
from config._base_.backbone.pvtv2 import *



model_name = 'ISDNet_G_pvtb1_L_STDC'


global_net_ckpt_from = 'stage1_GNet_pretrained'
global_pretrained_ckpt  = 'work_dir/GID_Water_3d5K/d4x_pvtb1_fpnM-e40/ckpt/epoch=6@val_mIoU=89.38@d4x_pvtb1_fpnM-GID_Water_3d5K.ckpt'
global_net_ckpt_dict['stage1_GNet_pretrained'] = global_pretrained_ckpt


fpn_decoder_args        = fpn_decoder_args_M
stage1_GNet_lr_reduce   = 3/4


def get_global(cfg):
    GB = backbone_pvtv2_b1()
    GD = FPNDecoder(GB.embed_dims, return_context=True, use_decoder_0=False, **cfg.fpn_decoder_args)
    return GB, GD



def get_model(cfg):
    fpn_decoder_args['num_classes'] = cfg.num_classes

    GB, GD = get_global(cfg)

    isd_args = cfg.ISDHead_Kargs

    # print(GD.context_channel)

    isd_args.update(dict(
        loss_layer = SceDiceLoss(cfg.ignore_index, False),
        num_classes=cfg.num_classes,
        prev_channel=256,
    ))

    # TODO if stage0, warmup 
    
    loss = SceDiceLoss(cfg.ignore_index, cfg.use_dice)

    model = DoubleBranchNet_ISDNet(GB, GD, isd_args,
                                   global_seg_loss_layer=loss,
                                   local_seg_loss_layer=loss,
                                   global_patches={'train':(2,2), 'val':(1,2)},
                                   local_patches={'train':(4,2), 'val':(4,2)},

                                   local_batch_size={'train':2, 'val':1},

                                   global_downsample=1/4,
                                   local_downsample=1,
                                   save_images_args=dict(per_n_step=-1),    #TODO
                                   )
    cfg.load_global_net_ckpt(cfg, model)

    return model