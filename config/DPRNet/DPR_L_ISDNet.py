from DPRNet import *

# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁    分支     ISDNet的 ISDHead
model_name = 'DPR-pvtb1-ISDNet'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(ISDNet-ShallowNet+Head)'


# region ----------------------------- Local Define -----------------------------

from jscv.hr_models.ISDNet.isdnet import *

context_channel = 256


isd_args = dict(
    pretrain_path="pretrain/STDCNet813M_73.91.tar",
    in_channels=3,
    channels=128,
    num_classes=2,
    dropout_ratio=0.1,
    norm_cfg={'type': 'groupnorm', 'num_groups': 8},
    align_corners=False,
)


def createISDNet(kargs):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method='env://')
    return ISDHead(**kargs)


class LocalISDHead(LocalBranchBase):
    def __init__(self, isd_args):
        super(LocalISDHead, self).__init__()
        self.isd = createISDNet(isd_args)

    def forward(self, img, ctx, mask=None):
        result = {}
        if self.training:
            pred, losses = self.isd.forward_train(img, ctx, mask)
            losses, losses_aux16, losses_aux8, losses_recon, losses_fa = losses
            loss = losses + losses_aux16 + losses_aux8 + losses_recon['recon_losses'] + losses_fa['fa_loss']
            result[self.key_loss] = loss
        else:
            pred = self.isd.forward(img, ctx, False)
            result[self.key_loss] = torch.tensor(0, device=img.device)
        result[self.key_pred] = pred
        return result


def get_local_branch(cfg, context_channel):
    ''' 必须实现的 '''
    from jscv.losses.useful_loss import SceDiceLoss

    cfg.isd_args.update(dict(
        loss_layer = SceDiceLoss(cfg.ignore_index, False),
        num_classes=cfg.num_classes,
        prev_channel=context_channel,
    ))
    
    return LocalISDHead(cfg.isd_args)




# region ----------------------------- Dynamic Patch Refinement -----------------------------

dynamic_manager_args_train.update({
    'max_pathes_nums':          24,
    'refinement_rate':          0.4,
})

dynamic_manager_args_val.update({
    'max_pathes_nums':          72,
    'refinement_rate':          0.28,
    'min_pathes_nums_rate':     0.35,
})
