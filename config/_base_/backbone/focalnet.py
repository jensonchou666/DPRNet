from jscv.backbone.focalnet import *

from jscv.backbone.focalnet_dilation import FocalDNet





def FocalD_S_L7_F3():
    backbone = FocalDNet(embed_dim=80,
                         depths=[2, 2, 18, 2],
                         drop_path_rate=0.3,
                         patch_norm=True,
                         focal_factor=3,
                         focal_windows=[3] * 4,
                         focal_levels=[7] * 4,
                         mlp_ratio=4.,
                         drop_rate=0.,
                         use_dilation=True,
                         out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, None, None, backbone_features


def FocalD_S_L5():
    backbone = FocalDNet(embed_dim=90,
                         depths=[2, 2, 18, 2],
                         drop_path_rate=0.3,
                         patch_norm=True,
                         focal_factor=2,
                         focal_windows=[3, 3, 3, 3],
                         focal_levels=[5, 5, 5, 5],
                         mlp_ratio=4.,
                         drop_rate=0.,
                         use_dilation=True,
                         out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, None, None, backbone_features


def Focal_S_L7():
    backbone = FocalDNet(
        embed_dim=92,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        #focal_factor=2,
        focal_windows=[1, 1, 1, 1],
        focal_levels=[7, 7, 7, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, None, None, backbone_features


def Focal_S_3X3():
    backbone = FocalDNet(embed_dim=96,
                         depths=[2, 2, 18, 2],
                         drop_path_rate=0.3,
                         patch_norm=True,
                         focal_factor=0,
                         focal_windows=[3] * 4,
                         focal_levels=[3] * 4,
                         mlp_ratio=4.,
                         drop_rate=0.,
                         out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, None, None, backbone_features


def Focal_S(backbone_ckpt_path='pretrain/focalnet_small_lrf.pth',
            backbone_prefix=''):
    # ImageNet-1K
    # from https://github.com/microsoft/FocalNet
    backbone = FocalDNet(embed_dim=96,
                         depths=[2, 2, 18, 2],
                         drop_path_rate=0.3,
                         patch_norm=True,
                         focal_windows=[3, 3, 3, 3],
                         focal_levels=[3, 3, 3, 3],
                         mlp_ratio=4.,
                         drop_rate=0.,
                         out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features



def Focal_Tiny(backbone_ckpt_path='pretrain/focalnet_tiny_lrf.pth',
            backbone_prefix=''):
    # ImageNet-1K
    # from https://github.com/microsoft/FocalNet
    backbone = FocalDNet(embed_dim=96,
                         depths=[2, 2, 6, 2],
                         drop_path_rate=0.2,
                         patch_norm=True,
                         focal_windows=[3, 3, 3, 3],
                         focal_levels=[3, 3, 3, 3],
                         mlp_ratio=4.,
                         drop_rate=0.,
                         out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def Focal_B(backbone_ckpt_path='pretrain_weights/focalnet_base_lrf.pth',
            backbone_prefix=''):
    # ImageNet-1K
    # from https://github.com/microsoft/FocalNet
    backbone = FocalNet(embed_dim=128,
                        depths=[2, 2, 18, 2],
                        focal_windows=[3, 3, 3, 3],
                        focal_levels=[3, 3, 3, 3],
                        drop_path_rate=0.5,
                        patch_norm=True,
                        mlp_ratio=4.,
                        drop_rate=0.,
                        out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features


def Focal_S_UpperNet(
        backbone_ckpt_path='pretrain_weights/focalnet_small_lrf_upernet_160k.pth',
        backbone_prefix='backbone'):
    # upernet trained on ADE20K
    # from https://github.com/microsoft/FocalNet
    backbone = FocalNet(embed_dim=96,
                        depths=[2, 2, 18, 2],
                        drop_path_rate=0.3,
                        focal_windows=[9, 9, 9, 9],
                        focal_levels=[3, 3, 3, 3],
                        patch_norm=True,
                        use_checkpoint=False,
                        mlp_ratio=4.,
                        drop_rate=0.,
                        out_indices=(0, 1, 2, 3))
    backbone_features = backbone.num_features
    return backbone, backbone_ckpt_path, backbone_prefix, backbone_features

if __name__ == '__main__':
    net = Focal_S()[0].cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    features = net(x)
    for i in features:
        print(i.shape)