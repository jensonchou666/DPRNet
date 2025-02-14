from config.DPRNet.DPRNet import *

# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁    分支     deeplabv3+ mobile_net 主干
# 上下文融合解码器：    WDCI  Window-Self-Attn + Downsmple-Self-Attn (Context-Injector)
model_name = 'DPR-pvtb1-WDCI_v1'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(eff+), \
GLFuser_(WDViT_CI_Decoder)'



# region ----------------------------- Global Define -----------------------------
def global_decoder(cfg, global_encoder_channels):
    from jscv.hr_models.base_models import FPNDecoder as D, fpn_decoder_args_M as dargs
    # dargs = dargs.copy()
    dargs['num_classes'] = cfg.num_classes
    GD = D(global_encoder_channels, 
           return_context=True,
           context_type=FPNDecoder.CXT_D2,
           use_decoder_0=False,  #是否对stem层特征解码
           **dargs)
    print('context_type=FPNDecoder.CXT_D2')
    return GD, GD.context_channel


# region ----------------------------- Local Define -----------------------------

LocalDecoderClass = WDViT_CI_Decoder

local_decoder_args = dict(
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
        channels=[32, 64, 128],
        blocks=[2, 2, 3],
        use_decoder_0=True,
    )
)


# region ----------------------------- Dynamic Patch Refinement -----------------------------

dynamic_manager_args_train.update({
    'max_pathes_nums':          16,
    'refinement_rate':          0.3,
})

dynamic_manager_args_val.update({
    'max_pathes_nums':          40,
    'refinement_rate':          0.28,
})



# dprnet_args.update(dict(
#     save_images_args = dict(
#         per_n_step=1,
#         action_type='dynamic_patch_v1',
#         save_img=True,
#         save_coarse_pred=True,
#         save_wrong=True,
#         jpeg_ratio=50,
#         )
# ))