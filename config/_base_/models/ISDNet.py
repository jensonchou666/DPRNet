from config._base_.common.uhr_std_env import *


from jscv.losses.useful_loss import SceDiceLoss
from jscv.hr_models.ISDNet.double_branch_net import *


# 下采样分支   ResNet101主干  （可替换）

model_name = 'ISDNet_G_R101_L_STDC'



ISDHead_Kargs.update(dict(
    pretrain_path="pretrain/STDCNet813M_73.91.tar"  # 局部分支
))


global_net_ckpt_from = 'stage0'
global_net_ckpt_dict = {
    'stage0'                    : 'pretrain/resnet101-63fe2227.pth',

    'stage1_GNet_pretrained'  : \
        'work_dir/GID_Water/seg_r101_d4-e80/final/epoch=15@val_mIoU=88.67@seg_r101_d4-GID_Water.ckpt'

}

fpn_decoder_args        = fpn_decoder_args_1
stage1_GNet_lr_reduce   = 1/2           # 若从stage1训练, G-Branch的lr应减小



def load_global_net_ckpt(cfg, model):
    fn = global_net_ckpt_dict[cfg.global_net_ckpt_from]
    D = torch.load(fn)
    if cfg.global_net_ckpt_from == 'stage1_GNet_pretrained':
        load_checkpoint(model.global_encoder, D, 'net.backbone')
        load_checkpoint(model.global_decoder, D, 'net.decoder')
    else:
        load_checkpoint(model.global_encoder, D)
    print("load GB from", fn)
    


def get_global(cfg):
    GB = ResNetEncoder('resnet101', features_only=True)
    GD = FPNDecoder(GB.channels, return_context=True, use_decoder_0=False, **cfg.fpn_decoder_args)
    return GB, GD




def get_model(cfg):
    fpn_decoder_args['num_classes'] = cfg.num_classes

    GB, GD = get_global(cfg)

    isd_args = cfg.ISDHead_Kargs

    isd_args.update(dict(
        loss_layer = SceDiceLoss(cfg.ignore_index, False),
        num_classes=cfg.num_classes,
        channels=GD.context_channel,
    ))

    # TODO if stage0, warmup 
    
    loss = SceDiceLoss(cfg.ignore_index, cfg.use_dice)

    model = DoubleBranchNet_ISDNet(GB, GD, isd_args,
                                   global_seg_loss_layer=loss,
                                   local_seg_loss_layer=loss,
                                   global_patches={'train':(4,4), 'val':(2,4)},
                                   local_patches={'train':(8,8), 'val':(4,8)},
                                   
                                   local_batch_size={'train':2, 'val':1},

                                   global_downsample=1/4,
                                   local_downsample=1,
                                   save_images_args=dict(per_n_step=-1),    #TODO
                                   )
    cfg.load_global_net_ckpt(cfg, model)

    return model



def get_evaluators(cfg):
    return Joint_Evaluator(
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'coarse_pred', False),
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'pred', False),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='coarse_pred'),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='pred'),
    )
    



def create_model(cfg):
    cfg.trainer_no_backward = True
    cfg.trainer_split_input = False
    model = cfg.get_model(cfg)

    if on_train():
        cfg.evaluator = cfg.get_evaluators(cfg)
        
        lr1 = init_learing_rate
        if global_net_ckpt_from == 'stage1_GNet_pretrained':
            lr1 = init_learing_rate*stage1_GNet_lr_reduce


        optg, schg = cfg.get_optimizer(cfg, [
            {'params': model.global_encoder.parameters(), 'lr': lr1},
            {'params': model.global_decoder.parameters(), 'lr': lr1},
            ], lr=lr1)
        optl, schl = cfg.get_optimizer(cfg, [
            {'params': model.ISDNet.parameters()}
            ], lr=init_learing_rate)
        cfg.optimizers = [optg, optl]
        cfg.lr_schedulers = [schg, schl]
        model.optimizer_global = optg
        model.optimizer_local = optl
    cfg.model = model