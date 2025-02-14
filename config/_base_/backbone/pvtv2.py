from jscv.backbone.PVTv2 import *


def backbone_pvtv2_b0():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[32, 64, 160, 256],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[2, 2, 2, 2],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def backbone_pvtv2_b1(**args):
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[2, 2, 2, 2],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1, **args)

def backbone_pvtv2_b2():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 4, 6, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def backbone_pvtv2_b2_li():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 4, 6, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1,
                                      linear=True)

def backbone_pvtv2_b3():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 4, 18, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)


def backbone_pvtv2_b4():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 8, 27, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def backbone_pvtv2_b5():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[4, 4, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[3, 6, 40, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)

def pvtv2_b1(
        backbone_ckpt_path='pretrain/pvt_v2_b1.pth',
        backbone_prefix=''):
    backbone = backbone_pvtv2_b1()
    backbone_features = [64, 128, 320, 512]
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


# def backbone_pvtv2_small():
#     return PyramidVisionTransformerV2(patch_size=4,
#                                       embed_dims=[64, 128, 320, 512],
#                                       num_heads=[1, 2, 5, 8],
#                                       mlp_ratios=[8, 8, 4, 4],
#                                       qkv_bias=True,
#                                       norm_layer=partial(nn.LayerNorm,
#                                                          eps=1e-6),
#                                       depths=[3, 3, 5, 3],
#                                       sr_ratios=[8, 4, 2, 1],
#                                       drop_rate=0.0,
#                                       drop_path_rate=0.1)
def backbone_pvtv2_small():
    return PyramidVisionTransformerV2(patch_size=4,
                                      embed_dims=[64, 128, 256, 512],
                                      num_heads=[1, 2, 4, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      depths=[4, 4, 4, 4],
                                      sr_ratios=[8, 4, 2, 1],
                                      drop_rate=0.0,
                                      drop_path_rate=0.1)
def pvtv2_small(
        backbone_ckpt_path='pretrain_weights/pvtv2_small.pth',
        backbone_prefix=''):
    backbone = backbone_pvtv2_small()
    backbone_features = [64, 128, 256, 512]
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features




def pvtv2_small_2(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [64, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[3, 4, 6, 4],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features



def pvtv2_small_2_for_refine(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [64, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[2, 4, 7, 4],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features



def pvtv2_small_2_for_refine_f512(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [68, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[1, 4, 7, 4],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

def pvtv2_small_2_for_refine_f512_1(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [64, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[2, 2, 4, 3],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

def pvtv2_small_2_for_refine_f512_2(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [64, 128, 256, 384]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 4, 8, 8],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[1, 4, 5, 3],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def pvtv2_small_3(
        backbone_ckpt_path=None,
        backbone_prefix=''):
    backbone_features = [64, 128, 256, 512]
    backbone = PyramidVisionTransformerV2(patch_size=4,
                                          embed_dims=backbone_features,
                                          num_heads=[2, 1, 1, 1],
                                          mlp_ratios=[4, 4, 4, 4],
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm,
                                                             eps=1e-6),
                                          depths=[3, 5, 6, 5],
                                          sr_ratios=[8, 4, 2, 1],
                                          drop_rate=0.0,
                                          drop_path_rate=0.1)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features









def pvtv2_b2(
        backbone_ckpt_path='pretrain/pvt_v2_b2.pth',
        backbone_prefix=''):
    backbone = backbone_pvtv2_b2()
    backbone_features = [64, 128, 320, 512]
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

def pvtv2_b3(
        backbone_ckpt_path='pretrain/pvt_v2_b3.pth',
        backbone_prefix=''):
    backbone = backbone_pvtv2_b3()
    backbone_features = [64, 128, 320, 512]
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def pvtv2_b4(
        backbone_ckpt_path='pretrain/pvt_v2_b4.pth',
        backbone_prefix=''):
    backbone = backbone_pvtv2_b4()
    backbone_features = [64, 128, 320, 512]
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

def pvtv2_b5(
        backbone_ckpt_path='pretrain/pvt_v2_b5.pth',
        backbone_prefix=''):
    backbone = backbone_pvtv2_b5()
    backbone_features = [64, 128, 320, 512]
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

