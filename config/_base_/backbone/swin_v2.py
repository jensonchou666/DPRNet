from jscv.backbone.swin_v2.model import SwinTransformerV2


def swinv2_small_w16(
        img_size=(512, 512), backbone_ckpt_path=None, backbone_prefix=''):

    backbone = SwinTransformerV2(in_channels=3,
                                 embedding_channels=96,
                                 depths=[2, 2, 18, 2],
                                 number_of_heads=[3, 6, 12, 24],
                                 window_size=16,
                                 dropout_path=0.3,
                                 input_resolution=img_size)

    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def swinv2_small_w8(
        img_size=(512, 512), backbone_ckpt_path=None, backbone_prefix=''):
    backbone = SwinTransformerV2(in_channels=3,
                                 embedding_channels=96,
                                 depths=[2, 2, 18, 2],
                                 number_of_heads=[3, 6, 12, 24],
                                 window_size=8,
                                 dropout_path=0.3,
                                 input_resolution=img_size)

    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def swinv2_base_w8(
        img_size=(512, 512), backbone_ckpt_path=None, backbone_prefix=''):

    backbone = SwinTransformerV2(in_channels=3,
                                 embedding_channels=128,
                                 depths=[2, 2, 18, 2],
                                 number_of_heads=[4, 8, 16, 32],
                                 window_size=8,
                                 dropout_path=0.5,
                                 input_resolution=img_size)

    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def swinv2_base_w16(
        img_size=(512, 512), backbone_ckpt_path=None, backbone_prefix=''):

    backbone = SwinTransformerV2(in_channels=3,
                                 embedding_channels=128,
                                 depths=[2, 2, 18, 2],
                                 number_of_heads=[4, 8, 16, 32],
                                 window_size=16,
                                 dropout_path=0.5,
                                 input_resolution=img_size)

    backbone_features = backbone.num_features
    print(backbone_features)
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

def swinv2_small(
        img_size=(512, 512), backbone_ckpt_path=None, backbone_prefix=''):

    backbone = SwinTransformerV2(in_channels=3,
                                 embedding_channels=64,
                                 depths=[13, 13, 13, 13],
                                 number_of_heads=[1, 2, 4, 8],
                                 window_size=8,
                                 dropout_path=0.1,
                                 input_resolution=img_size)

    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


if __name__ == '__main__':
    backbone, _, _, num_features = swinv2_small_w8((512, 512))
    print("num_features", num_features)
    import torch
    ret = backbone(torch.randn(2, 3, 512, 512))
    for x in ret:
        print(x.shape)