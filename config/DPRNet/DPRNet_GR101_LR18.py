from config._base_.schedule.normal import *

from jscv.hr_models.deeplabv3 import ASPP
from jscv.hr_models.DPRNet import *
from jscv.utils.trainer import SegmentEvaluator, Joint_Evaluator, EdgeAccEvaluator

from jscv.losses.useful_loss import SCE_DIce_Loss
from jscv.utils.load_checkpoint import load_checkpoint
import torch


# 训练输出目录： /work_dir/{dataset_name}/{model_name}-e{max_epoch}/{version(auto create)}

model_name = 'DPRNet_GR101_LR18'



global_net_ckpt_from = 'stage0'
# 从哪里加载全局网络的预训练权重?  global_net_ckpt_from 从以下dict的key中选
global_net_ckpt_dict = {
    # 不预先训练,  直接用官网下载的预训练参数
    'stage0'                    : 'pretrain/resnet101-63fe2227.pth',

    # 已训练若干轮的下采样模型的权重，该下采样模型的 网络结构 必须和 DPRNet-G-Branch一致
    # 请替换成你自己训练好的权重
    # 'stage1_GNet_pretrained'  : 'work_dir/GID_Water/deeplab101_d4-e80/version_0/epoch=21@val_mIoU=88.93@deeplab101_d4-GID_Water.ckpt',
    'stage1_GNet_pretrained'  : 'Your ckpt path (deeplab101_d4)'
    
    #'stage2_PDT_trained': None,  # 弃用，不需要提前训练PDT，可以直接开始 Global-Local-PDT的联合训练
}

local_ckpt_path_pretrain = 'pretrain/resnet18-5c106cde.pth' # L-Branch的预训练位置

#------------------------ Setting ------------------------

local_patches           = (16,16)       # PDT size

PDT_threashold          = 0.06          # PDT threashold
PDT_threashold_training = 0.04          # reduce simple samples training
# G-Branch的补丁划分，解决显存溢出，可调大 如：{'train':(4,4), 'val':(1,2)}
global_patches          = {'train':(2,4), 'val':(1,2)}
# global_patches          = {'train':(2,2), 'val':(1,1)}
# 补丁分组，每个组的补丁数量，可减小到1
local_batch_size        = {'train': 1, 'val': 1}

stage0_warmup_epoch     = 10            # 若从stage0训练，前N轮不训练L-Branch和PDT
stage1_GNet_lr_reduce   = 1/2           # 若从stage1训练, G-Branch的lr应减小
init_learing_rate       = 1e-4          # 初始lr
weight_decay            = 1e-4
poly_power              = 2

# from jscv.hr_models.base_models import SaveImagesManager
# 运行时调用的 SaveImagesManager.save_image_hard() 来保存图像，可修改，改变存图逻辑
save_images_args = dict(
    per_n_step=40,           #? 每 n个batches，保存一次图像。 如果 per_n_step=-1，不保存
    save_img=False,          #? 原图比较占存储,要存图置: save_img=True
    save_coarse_pred=True,
    all_do_hardpathes=False,
    save_wrong=True)



fpn_decoder_args        = fpn_decoder_args_1 # decoder参数
local_fpn_decoder_args  = fpn_decoder_args
ASPP_channels_rate      = 1/4
ASPP_layers             = 4
GL_FFM_channels_rate    = 1/2
GL_FFM_layers           = 2
global_seg_loss_weight  = 1
use_dice                = False # dice损失作用好像不大
optimizer_step_interval = 2     # L-Branch 优化器更新的间隔
model_extra_args        = {}

train_batchsize_dict    = {None: 1}    # 全局的batchsize固定设置为1，别改
trainer_log_per_k       = 10           # 每k次，向 output.log 打印
save_topk               = 5            # 保存前k好的权重
logger_display_topk     = 20           # log.txt 文件前20排序
skip_analyse            = True         # 不使用analyser存图
#? Important 其它的一些设置，参考每个组件的from_config()方法里的说明


def get_g_l_nets(cfg):
    GB = ResNetEncoder('resnet101', features_only=True)
    GB = StrongerEncoder(GB, ASPP, cfg.ASPP_channels_rate, cfg.ASPP_layers)        # 向G-Branch添加ASPP层
    
    GD = FPNDecoder(GB.channels, return_d2_feature=True, **fpn_decoder_args)
    LB = ResNetEncoder('resnet18', features_only=True)

    #? 代码里直接把 GL_FFM 和 L-Branch-Decoder 合并到一起了
    LD = GL_FFN(GD.channel_d2, LB.channels, cfg.GL_FFM_layers, cfg.GL_FFM_channels_rate, FPNDecoder, 
                cfg.local_fpn_decoder_args)
    return GB, GD, LB, LD, GB.channels

def load_global_net_ckpt(cfg, model):
    fn = global_net_ckpt_dict[cfg.global_net_ckpt_from]
    D = torch.load(fn)
    if cfg.global_net_ckpt_from == 'stage1_GNet_pretrained':
        load_checkpoint(model.global_encoder.backbone, D, 'net.backbone')
        load_checkpoint(model.global_encoder.e4, D, 'net.aspp')
        load_checkpoint(model.global_decoder, D, 'net.decoder')
    else:
        load_checkpoint(model.global_encoder.backbone, D)
    print("load GB from", fn)

def get_model(cfg):
    fpn_decoder_args['num_classes'] = cfg.num_classes
    local_fpn_decoder_args['num_classes'] = cfg.num_classes

    #? Important 不用 'cfg.' 会导致无法被覆盖
    GB, GD, LB, LD, GB_channels= cfg.get_g_l_nets(cfg)

    seg_loss_layer = SCE_DIce_Loss(ignore_index=cfg.ignore_index, use_dice=cfg.use_dice)

    #? Important ☆☆☆☆ DPRNet模型， 可改参数
    model = DPRNet(GB, GD, GB_channels, LB, LD, seg_loss_layer, 
                    seg_loss_layer, cfg.global_seg_loss_weight,
                    global_patches=cfg.global_patches,
                    local_patches=cfg.local_patches,
                    local_batch_size=cfg.local_batch_size, 
                    pred_easy_bondary=cfg.PDT_threashold,
                    pred_easy_bondary_training=PDT_threashold_training,
                    labal_err_bondary=2,  
                    ignore_index=cfg.ignore_index,
                    save_images_args=cfg.save_images_args,
                    optimizer_step_interval=cfg.optimizer_step_interval,
                    stage=3,
                    warmup_epoch=stage0_warmup_epoch if global_net_ckpt_from == 'stage0' else 0,
                    **cfg.model_extra_args
                    )
    
    cfg.load_global_net_ckpt(cfg, model)
    load_checkpoint(model.local_encoder, torch.load(local_ckpt_path_pretrain))

    return model

def get_optimizer(cfg, net, lr=1e-4):
    if isinstance(net, list):
        params = net
    else:
        params = net.parameters()
    optimizer = torch.optim.AdamW(params,
                                  lr=lr,
                                  weight_decay=cfg.weight_decay)
    sch = PolyLrScheduler(optimizer, cfg.max_epoch, cfg.poly_power)
    return optimizer, sch

def get_evaluators(cfg):
    return Joint_Evaluator(
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'coarse_pred', False),
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'pred', False),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='coarse_pred'),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='pred'),
        EasyHardEvaluator(local_patches),
    )


def create_model(cfg):

    #? 关掉 Trainer 自带的反向传播，由DPRNet内部管理反向传播
    cfg.trainer_no_backward = True
    #? 不让 Trainer 把 batch拆分成img和target
    cfg.trainer_split_input = False
    model = cfg.get_model(cfg)

    if on_train():
        #? Important 5个指标检测器
        cfg.evaluator = cfg.get_evaluators(cfg)

        if global_net_ckpt_from == 'stage1_GNet_pretrained':
            lr1, lr2 = init_learing_rate*stage1_GNet_lr_reduce, init_learing_rate #*2
        # elif global_net_ckpt_from == 'stage2_PDT_trained':
        #     lr1, lr2 = init_learing_rate/10, init_learing_rate/3
        else:
            lr1, lr2 = init_learing_rate, init_learing_rate

        optg, schg = cfg.get_optimizer(cfg, [
            {'params': model.global_encoder.parameters(), 'lr': lr1},
            {'params': model.global_decoder.parameters(), 'lr': lr1},
            {'params': model.easyhard_head.parameters(), 'lr': lr2},
            ], lr=lr1)
        optl, schl = cfg.get_optimizer(cfg, [
            {'params': model.local_encoder.parameters()},
            {'params': model.GL_ffn_decoder.parameters()},
            ], lr=init_learing_rate)
        cfg.optimizers = [optg, optl]
        cfg.lr_schedulers = [schg, schl]
        model.optimizer_global = optg
        model.optimizer_local = optl
    cfg.model = model



def define_model_name_simple(cfg, lr_net_name):
    H, W = local_patches
    return 'DPRNet_' + lr_net_name + f'_{H}x{W}'
    
def replace_GB(cfg, GB, embed_dims):
    GD = FPNDecoder(embed_dims, return_d2_feature=True, **fpn_decoder_args)
    LB = ResNetEncoder('resnet18', features_only=True)
    LD = GL_FFN(GD.channel_d2, LB.channels, cfg.GL_FFM_layers, cfg.GL_FFM_channels_rate, FPNDecoder, 
                cfg.local_fpn_decoder_args)
    return GB, GD, LB, LD, embed_dims

# analyser的参数参考这里：
# from jscv.utils.analyser import analyser_from_cfg

# analyse_segment = True  # =True 存分割结果图

# # 存的项目还包括： "loss_map", "wrong_map", "wrong_alone_map" "confuse_map", "confuse_dilate", "confuse_gate" 
# analyser_include = "confuse"

