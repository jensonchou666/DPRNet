from config._base_.common.uhr_std_env import *
from jscv.hr_models.DPRNet.DPRNet import *

#TODO 把 pdt_head提取出来到 cfg里

# 训练输出目录： /work_dir/{dataset_name}/{model_name}-e{max_epoch}/{version(auto create)}

#? 动态补丁细化网络  Dynamic Patch Refinement Network
# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码 （可替换）
# 局部-补丁    分支     efficientnet-b3主干  （可替换） 
# 上下文融合解码器：    ASACI  Adaptive-Downsample Self-Attention Context-Injector
model_name = 'DPR-pvtb1-efficient_b3-ASACI-v2'
description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(efficientnet-b3), \
GLFuser_(ASACI_Decoder)'




# region ----------------------------- 基础网络定义 -----------------------------

# region ----------------------------- Global Define -----------------------------
def global_encoder(cfg):
    from config._base_.backbone.pvtv2 import backbone_pvtv2_b1 as B
    n = B()
    return n, n.embed_dims

def global_decoder(cfg, global_encoder_channels):
    from jscv.hr_models.base_models import FPNDecoder as D, fpn_decoder_args_M as dargs
    # dargs = dargs.copy()
    dargs['num_classes'] = cfg.num_classes
    GD = D(global_encoder_channels, 
           return_context=True,
           context_type=FPNDecoder.CXT_B2_B3_B4, #上下文特征
           use_decoder_0=False,  #是否对stem层特征解码
           **dargs)
    return GD, GD.context_channel

global_ckpt_from        = '2_down4x_trained'
# 从哪里加载全局网络的预训练权重?  global_ckpt_from 从以下dict的key中选
global_ckpt_dict = {
    # 仅从官网下载的主干预训练，选择该项，DPRNet前几轮需要先训练Global-Branch，只建议给想快速开始训练时使用
    '1_backbone_pretrained' : {'path': 'pretrain/resnet101-63fe2227.pth', 'warmup_epoch': 6, 'prefix': ''},
    # 从对应的已训练几轮的 4x-Download模型 开始，必须和DPRNet的G-Branch一致，本文件对应的4xD配置为 Down4x/d4x_pvtb1_fpnM.py
    # 请替换成你自己训练好的权重
    '2_down4x_trained'  : {
        'path': 'work_dir/GID_Water_3d5K/d4x_pvtb1_fpnM-e40/ckpt/epoch=6@val_mIoU=89.38@d4x_pvtb1_fpnM-GID_Water_3d5K.ckpt',
        'lr_reduce': 4/5, #若从2_down4x_trained加载，调整G-Branch学习率
        'prefix':{'encoder': 'net.backbone', 'decoder': 'net.decoder'}
    }
    #'3_pdt_head_trained': None,  # 还未实现，可以不需要提前训练PDT
}

def load_global_ckpt(cfg, E, D):
    ckpt_from = cfg.global_ckpt_from
    global_ckpt_info = cfg.global_ckpt_dict[ckpt_from]
    print("Load global network from: ", ckpt_from, global_ckpt_info)
    Dic = torch.load(global_ckpt_info['path'])
    if ckpt_from == '1_backbone_pretrained':
        load_checkpoint(E, Dic, global_ckpt_info['prefix'])
    elif ckpt_from == '2_down4x_trained':
        load_checkpoint(E, Dic, global_ckpt_info['prefix']['encoder'])
        load_checkpoint(D, Dic, global_ckpt_info['prefix']['decoder'])
    else:
        assert False, "暂不支持"

def get_global_branch(cfg):
    ''' 必须实现的 '''
    E, embed_dims = cfg.global_encoder(cfg)
    D, context_channel = cfg.global_decoder(cfg, embed_dims)
    cfg.load_global_ckpt(cfg, E, D)
    return GlobalEncoderDecoderV1(E, D, embed_dims[-1], context_channel, cfg.context_channel, False)

context_channel = 512

# endregion

# region ----------------------------- Local Define -----------------------------
def local_encoder(cfg):
    from jscv.backbone.efficientnet.efficientnet import EfficientNet
    backbone_name = 'efficientnet-b3'
    channels = [24, 32, 48, 136, 384]
    backbone_ckpt = 'pretrain/efficientnet-b3-5fb5a3c3.pth'
    # local_backbone_ckpt = None # 将从github自动下载
    n = EfficientNet.from_pretrained(backbone_name, backbone_ckpt)
    return n, channels

#TODO 不好用
LocalDecoderClass = AsppCI_Decoder
local_decoder_args = dict(
    ctx_mapping     = 256,
    out_mapping     = 128,
    aspp_args=[
        dict(channels=512, dilation_rates=[1, 3, 7, 11], num_blocks=4, groups=8, se_reduction=16),
        dict(channels=256, dilation_rates=[1, 6, 12, 18], num_blocks=3, groups=8, se_reduction=16),
    ],
    post_decoder_cls=PostDecodeHead,
    post_decoder_args=dict(
        channels=[24, 48, 96],
        blocks=[1, 2, 3],
        use_decoder_0=True,
    )
)

def local_decoder(cfg, local_encoder_channels, context_channel):
    cfg.local_decoder_args['post_decoder_args']['num_classes'] = cfg.num_classes
    LD = cfg.LocalDecoderClass(
        context_channel=context_channel,
        local_encoder_channels=local_encoder_channels,
        **cfg.local_decoder_args
    )
    return LD

# def boundary_head(cfg, local_encoder_channels):
#     from jscv.hr_models.base_models import FPNDecoder as D
#     head = D(
#         local_encoder_channels,
#         [16, 24, 32, 64],
#         [1,1,1,1],
#         use_decoder_0=True,
#         return_context=False,
#         num_classes=1
#     )
#     ''' 不使用 辅助边界损失 '''
#     return None

def get_local_branch(cfg, context_channel):
    ''' 必须实现的 '''
    E, embed_dims = cfg.local_encoder(cfg)
    D = cfg.local_decoder(cfg, embed_dims, context_channel)
    return LocalEncoderDecoderV1(E, D)


# endregion


def get_loss_layers(cfg):
    from jscv.losses.useful_loss import SceDiceLoss
    # from jscv.losses.boundary_loss import WeightedBoundaryLoss
    global_loss = SceDiceLoss(ignore_index=cfg.ignore_index, use_dice=False)
    local_loss = SceDiceLoss(ignore_index=cfg.ignore_index, use_dice=False)
    return global_loss, local_loss


# endregion



# region ----------------------------- Dynamic Patch Refinement -----------------------------

dynamic_manager_common_args = {
    'PDT_size':                 (16,16),
    'learn_epoch_begin':        1,
    'threshold_rate_1x1':       1,
    'score_compute_classtype':  PNScoreComputer, #? 默认的分数计算方式
    'score_compute_kargs':      {
        'fx_points': [(2, 1), (36, 1)],    #? 请参考PNScoreComputer的说明
        'gx_point': (4, 1)
    },
}

dynamic_manager_args_train.update(dynamic_manager_common_args)
dynamic_manager_args_train.update({
    'max_pathes_nums':          12,
    'refinement_rate':          0.35,
    'threshold_init':           0.02,
    'min_pathes_nums': 2,
    # 'min_pathes_nums_rate': 0.1,
})

dynamic_manager_args_val.update(dynamic_manager_common_args)
dynamic_manager_args_val.update({
    'max_pathes_nums':          36,
    'refinement_rate':          0.26,
    'threshold_init':           0.03,
    'use_threshold_learning':   True,
    # 'min_pathes_nums': 2,
    'min_pathes_nums_rate': 0.25,    #? @@@ Edit
})

def update_dynamic_manager_args(cfg, args_train, args_val):
    recursive_update(args_train, {})
    recursive_update(args_val, {})

# endregion

# region ----------------------------- DPRNet Setting -----------------------------

dprnet_args = dict(
    loss_weights = {
        'global': 1,
        'local': 0.3,
        'local_first': 0.4,
        'pdt': 0.4,
    },
    pdt_head_cls        = PDTHead,
    global_patches      = {'train':(2,2), 'val':(1,1)}, #? @@@ Edit
    global_downsample   = 1/4,
    local_downsample    = 1,
    context_zero_prob   = {'init': 0.2, 'last_epoch': 1},
    train_stage         = 3,
    SaveImagesManagerClass = SaveImagesManager,
    save_images_args = dict(
        per_n_step=-1,     #? @@@ Edit
        action_type='dynamic_patch_v1',
        save_img=True,
        save_coarse_pred=True,
        save_wrong=True
        )
)
def update_dprnet_args(cfg, dprnet_args:dict):
    recursive_update(dprnet_args, {})

init_learing_rate       = 1e-4          # 初始lr
weight_decay            = 1e-4
poly_power              = 2
get_optimizer = get_optimizer

def get_evaluators(cfg):
    return Joint_Evaluator(
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'coarse_pred', False),
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'pred', False),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='coarse_pred'),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='pred'),
    )


# endregion

def get_model(cfg):
    G = cfg.get_global_branch(cfg)
    L = cfg.get_local_branch(cfg, G.context_channel)
    g_loss, l_loss = cfg.get_loss_layers(cfg)

    cfg.update_dynamic_manager_args(cfg, cfg.dynamic_manager_args_train, cfg.dynamic_manager_args_val)
    local_warmup_epoch = 0
    if cfg.global_ckpt_from == '1_backbone_pretrained':
        local_warmup_epoch = global_ckpt_dict['1_backbone_pretrained']['warmup_epoch']

    #? Important ☆☆☆☆ DPRNet模型， 可改参数
    model = DPRNet(G, L, global_loss_layer=g_loss, local_loss_layer=l_loss,
                   dynamic_manager_args_train=cfg.dynamic_manager_args_train,
                   dynamic_manager_args_val=cfg.dynamic_manager_args_val,
                   ignore_index=cfg.ignore_index,
                   local_warmup_epoch=local_warmup_epoch,
                   **cfg.dprnet_args)
    return model




def create_model(cfg):
    #? 关掉 Trainer 自带的反向传播，由DPRNet内部管理反向传播
    cfg.trainer_no_backward = True
    #? 不让 Trainer 把 batch拆分成img和target
    cfg.trainer_split_input = False
    model = cfg.get_model(cfg)
    if on_train():
        #? 5个指标检测器
        cfg.evaluator = cfg.get_evaluators(cfg)
        lr1, lr2 = cfg.init_learing_rate, cfg.init_learing_rate
        if cfg.global_ckpt_from == '2_down4x_trained':
            lr1 *= cfg.global_ckpt_dict['2_down4x_trained']['lr_reduce']
        params = [
            {'params': model.global_branch.parameters(), 'lr': lr1},
        ]
        nets = ['pdt_head', 'local_branch']
        for n in nets:
            if hasattr(model, n) and isinstance(getattr(model, n), torch.nn.Module):
                params.append({'params': getattr(model, n).parameters(), 'lr': lr2})
        # print('optimizer params: ', params)
        opt, sch = cfg.get_optimizer(cfg, params, lr2)
        cfg.optimizers = [opt]
        cfg.lr_schedulers = [sch]
        model.optimizer = opt
    cfg.model = model



# analyser的参数参考这里：
# from jscv.utils.analyser import analyser_from_cfg

# analyse_segment = True  # =True 存分割结果图

# # 存的项目还包括： "loss_map", "wrong_map", "wrong_alone_map" "confuse_map", "confuse_dilate", "confuse_gate" 
# analyser_include = "confuse"

