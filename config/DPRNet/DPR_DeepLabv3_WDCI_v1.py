from DPRNet import *

# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁    分支     deeplabv3+ mobile_net 主干
# 上下文融合解码器：    WDCI  Window-Self-Attn + Downsmple-Self-Attn (Context-Injector)
model_name = 'DPR-pvtb1-deeplabv3-WDCI'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(deeplabv3+), \
GLFuser_(WDViT_CI_Decoder)'


# region ----------------------------- Local Define -----------------------------

local_backbone_name = 'mobilenetv2 + aspp'
local_backbone_ckpt = 'pretrain/deeplab_mobilenetv2.pth'    #? None，则从网络下载（可能访问很慢）
# local_backbone_ckpt = None # 自动下载
local_backbone_channels = [16, 24, 32, 96, 256]

def local_encoder(cfg):
    from jscv.backbone.deeplabv3p.deeplabv3_plus import DeepLab
    if cfg.local_backbone_ckpt is None:
        LB = DeepLab(backbone="mobilenet", pretrained=True)
    else:
        LB = DeepLab(backbone="mobilenet", pretrained=False)
        print("load_checkpoint:", cfg.local_backbone_ckpt)
        load_checkpoint(LB, cfg.local_backbone_ckpt)

    return LB, cfg.local_backbone_channels


LocalDecoderClass = WDViT_CI_Decoder

local_decoder_args = dict(
    sa_blocks       = [4,4],            # 7x7  14x14
    sa_channels     = [512, 256],       # 7x7  14x14
    ctx_mapping     = 256,  # 960/2
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
    'max_pathes_nums':          20,
    'refinement_rate':          0.4,
})

dynamic_manager_args_val.update({
    'max_pathes_nums':          40,
    'refinement_rate':          0.28,
})

