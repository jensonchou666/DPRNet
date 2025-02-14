from config._base_.common.uhr_std_env import *
from jscv.losses.useful_loss import SceDiceLoss

from jscv.hr_models.pathes_segmentor import PatchesSegmentor, PatchErrEvaluator, \
    train_setting_d4_1, fpn_decoder_args_512_1131
    



# Basic Configures
model_name = '4xD_Res101_fpn_512_1131'   # 4倍下采样


ckpt_path = 'pretrain/resnet101-63fe2227.pth'
backbone_name = 'resnet101'
decoder_args = fpn_decoder_args_512_1131


'''
    cuda显存溢出, 改
        train_setting_list=[((2,2),1/4, 1.)],
        val_setting=((1,2), 1/4),

    ((1,1), 1/4) 改 ((8,8), 1) 就是原分辨率补丁网络
'''
model_args = dict(
    # 支持多尺度，但好像作用不大？
    train_setting_list=[((1,2),1/4, 1.)],
    val_setting=((1,1), 1/4),

    batch_size=1,
    optimizer_step_interval=1,

    #? 存图间隔
    save_images_args=dict(per_n_step=-1, save_img=False, save_wrong=True), #TODO 存图有BUG
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
    net = ResSegNet(cfg.backbone_name, decoder_args=cfg.decoder_args)
    print('backbone_name:', cfg.backbone_name)
    print('decoder_args:', cfg.decoder_args)
    print('ckpt_path:', cfg.ckpt_path)
    net.pretrain_backbone(torch.load(cfg.ckpt_path))
    return net


# def get_evaluators(cfg):
#     return Joint_Evaluator(SegmentEvaluator.from_config(cfg),
#                            PatchErrEvaluator(ignore_idx=cfg.ignore_index))




def create_model(cfg):
    cfg.trainer_no_backward = True
    cfg.trainer_split_input = False

    if "desc_2" in cfg:
        add_detail_desc(cfg, cfg.desc_2)

    cfg.decoder_args['num_classes'] = cfg.num_classes
    net = cfg.get_network(cfg)
    seg_loss_layer = SceDiceLoss(ignore_index=cfg.ignore_index)
    
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
