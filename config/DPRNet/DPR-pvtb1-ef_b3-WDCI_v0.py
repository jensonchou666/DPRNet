from DPRNet import *

# 全局-下采样  分支     pvtv2_b1主干 + FPN-M解码
# 局部-补丁    分支     efficientnet-b3主干 
# 上下文融合解码器：    WDCI  Window-Self-Attn + Downsmple-Conv (Context-Injector)
model_name = 'DPR-pvtb1-efficient_b3-WDCI'

description = \
'Dynamic Patch Refinement: \
Global_(d4x_pvtb1_fpnM), \
Local_(efficientnet-b3), \
GLFuser_(WDCI_Decoder)'

#TODO 改字典递归重载
LocalDecoderClass = WDCI_Decoder

local_decoder_args = dict(
    sa_blocks       = [4,3],            # 7x7  14x14
    sa_channels     = [384, 136],       # 7x7  14x14
    ctx_mapping     = 256,  # 960/2
    mid_mapping     = 136,
    out_mapping     = 128,
    sa_args = dict(
        window_size=7,
        downsample_rate=7,
        conv_layers=2,
        num_heads=8,
        cfg_norm={'type': nn.GroupNorm, 'args': dict(num_groups=8)},
        act_type=nn.ReLU,
        attn_drop=0.1,
        proj_drop=0.1
        ),
    sa_args2 = dict(
        window_size=14,
        downsample_rate=7,
        conv_layers=3,
        ),
    post_decoder_cls=PostDecodeHead,
    post_decoder_args=dict(
        channels=[24, 48, 96],
        blocks=[1, 2, 2],
        use_decoder_0=True,
    )
)


# region ----------------------------- Dynamic Patch Refinement -----------------------------

dynamic_manager_args_train.update({
    'max_pathes_nums':          16,
    'refinement_rate':          0.4,
})

dynamic_manager_args_val.update({
    'max_pathes_nums':          32,
    'refinement_rate':          0.28,
})