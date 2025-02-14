from DPRNet import *

# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁    分支     deeplabv3+ mobile_net 主干
# 上下文融合解码器：    WDCI  Window-Self-Attn + Downsmple-Self-Attn (Context-Injector)
model_name = 'DPR-pvtb1-WDCI_v1_conv_ci'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(eff+), \
GLFuser_(WDViT_CI_Decoder)'


# region ----------------------------- Local Define -----------------------------

LocalDecoderClass = Conv_CI_Decoder

local_decoder_args = dict(
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