from config.DPRNet.DPRNet import *

# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁    分支     pvtv2_b1主干
# 上下文融合解码器：    WDCI  Window-Self-Attn + Downsmple-Self-Attn (Context-Injector)
model_name = 'DPR-pvtb1-WDCI_PVT_L'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(pvt), \
GLFuser_(WDViT_CI_Decoder)'

pretrained_PVT = 'work_dir/GID_Water_3d5K/d4x_pvtb1_fpnM-e40/final/epoch=29@val_mIoU=91.23@4xD_pvtb1_fpnM-GID_Water_3d5K.ckpt'

global_ckpt_dict['2_down4x_trained'] = {
        'path': pretrained_PVT,
        'lr_reduce': 1,
        'prefix':{'encoder': 'net.backbone', 'decoder': 'net.decoder'}
    }

init_learing_rate       = 4e-5

# region ----------------------------- Local Define -----------------------------
def local_encoder(cfg):
    from config._base_.backbone.pvtv2 import backbone_pvtv2_b1 as B
    # n = B()
    n = B(return_bchw=False)
    load_checkpoint(n, pretrained_PVT, 'net.backbone')
    return n, n.embed_dims

LocalDecoderClass = WDViT_CI_Decoder

local_decoder_args = dict(
    backbone_feaures_bchw=False,
    sa_blocks       = [4,4],            # 7x7  14x14
    sa_channels     = [768, 384],       # 7x7  14x14
    ctx_mapping     = 512,  # 960/2
    out_mapping     = 256,
    sa_args = dict(
        window_size=7,
        down_ratio=7,
        down_kernel_size=11,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1
        ),
    sa_args2 = dict(
        window_size=14,
        down_ratio=7,
        down_kernel_size=11,
        ),
    post_decoder_cls=PostDecodeHead,
    post_decoder_args=dict(
        channels=[64, 128],
        blocks=[2, 3],
        use_decoder_0=False,
    )
)

trainer_log_per_k       = 5

# region ----------------------------- Dynamic Patch Refinement -----------------------------

dynamic_manager_args_train.update({
    'max_pathes_nums':          16,
    'refinement_rate':          0.27,
    'score_compute_kargs':      {
        'fx_points': [(2, 1), (36, 1)],    #? 请参考PNScoreComputer的说明
        'gx_point': (4, 1)
    },
    'min_pathes_nums_rate':     0.25,
    'threshold_rate_1x1':       1.5,
})

dynamic_manager_args_val.update({
    'max_pathes_nums':          36,
    'refinement_rate':          0.26,

    'score_compute_kargs':      {
        'fx_points': [(2, 2), (36,  1)],    #? 请参考PNScoreComputer的说明
        'gx_point': (4, 1)
    },
    'min_pathes_nums_rate':     0.25,
    'threshold_rate_1x1':       2,
})

print("dynamic_manager_args_val:", dynamic_manager_args_val)

dprnet_args.update(dict(
    loss_weights = {
        'global': 1,
        'local': 0.3,
        'local_first': 0.4,
        'pdt': 1,
    },
))

dprnet_args.update(dict(
    save_images_args = dict(
        per_n_step=-1,
        action_type='dynamic_patch_v1',
        save_img=True,
        save_coarse_pred=True,
        save_wrong=True,
        jpeg_ratio=50,
        )
))