from config._base_.schedule.normal import *
from jscv.hr_models.pathes_segmentor import PatchesSegmentor, PatchErrEvaluator, \
    train_setting_d4_1, fpn_decoder_args_1
from jscv.utils.trainer import SegmentEvaluator, Joint_Evaluator
from jscv.losses.useful_loss import SCE_DIce_Loss
import torch

# Basic Configures
model_name = 'r101_d4'   # 4倍下采样


ckpt_path = 'pretrain/resnet101-63fe2227.pth'
backbone_name = 'resnet101'
decoder_args = fpn_decoder_args_1


'''
    cuda显存溢出, 改
        train_setting_list=[((2,4),1/4, 1.)],
        val_setting=((1,2), 1/4),
    
    ((1,1), 1/4) 改 ((8,8), 1) 就是原分辨率补丁网络
'''
model_args = dict(
    train_setting_list=[((2,2),1/4, 1.)], #train_setting_d4_1,
    val_setting=((1,1), 1/4),

    batch_size=1,
    optimizer_step_interval=4,

    #? 存图间隔
    save_images_args=dict(per_n_step=20, save_img=False, save_wrong=True),
)


train_batchsize_dict        = {None: 1}
trainer_log_per_k           = 10
save_topk                   = 5
logger_display_topk         = 20
skip_analyse                = True


def get_optimizer(cfg, net):
    poly_power = 2
    lr = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)
    sch = PolyLrScheduler(optimizer, cfg.max_epoch, poly_power)
    return optimizer, sch




def get_network(cfg):
    from jscv.hr_models.pathes_segmentor import ResSegNet
    net = ResSegNet(cfg.backbone_name, decoder_args=decoder_args)
    net.pretrain_backbone(torch.load(ckpt_path))
    return net

def get_evaluators(cfg):
    return Joint_Evaluator(SegmentEvaluator.from_config(cfg),
                           PatchErrEvaluator(ignore_idx=cfg.ignore_index))

def create_model(cfg):
    cfg.trainer_no_backward = True
    cfg.trainer_split_input = False

    if "desc_2" in cfg:
        add_detail_desc(cfg, cfg.desc_2)

    cfg.decoder_args['num_classes'] = cfg.num_classes
    net = cfg.get_network(cfg)
    seg_loss_layer = SCE_DIce_Loss(ignore_index=cfg.ignore_index)
    
    model = PatchesSegmentor(net, seg_loss_layer, **model_args)

    if on_train():
        cfg.evaluator = cfg.get_evaluators(cfg)
        opt, sch = cfg.get_optimizer(cfg, model.net)
        cfg.optimizers = [opt]
        cfg.lr_schedulers = [sch]
        model.optimizer = opt   # 模型内部优化

    cfg.model = model


# times_analyse = 10
# analyse_segment = True
# analyser_include = "confuse"
