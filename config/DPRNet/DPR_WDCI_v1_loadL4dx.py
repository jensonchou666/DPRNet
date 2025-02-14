from DPRNet import *

'''
        L-Backbone 从 d4x 加载
'''


# 全局-下采样   分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁     分支     deeplabv3+ mobile_net 主干
# 上下文融合解码器：    WDCI  Window-Self-Attn + Downsmple-Self-Attn (Context-Injector)
model_name = 'DPR-pvtb1-WDCI_v1_load_4dx'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(eff+), \
GLFuser_(WDViT_CI_Decoder)'


# region ----------------------------- Local Define -----------------------------
LB_d4x_ckpt = 'work_dir/GID_Water_3d5K/d4x-efficientnet_b3-fpnM-e40/version_1/epoch=17@val_mIoU=90.36@d4x-efficientnet_b3-fpnM-GID_Water_3d5K.ckpt'

def local_encoder(cfg):
    from jscv.backbone.efficientnet.efficientnet import EfficientNet
    backbone_name = 'efficientnet-b3'
    channels = [24, 32, 48, 136, 384]
    n = EfficientNet.from_name(backbone_name)
    print("load Local Backbone from: ", LB_d4x_ckpt)
    load_checkpoint(n, LB_d4x_ckpt, 'net.backbone')
    return n, channels


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
    'refinement_rate':          0.4,
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