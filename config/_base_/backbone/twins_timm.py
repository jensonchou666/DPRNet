from jscv.backbone.twins_timm import Twins


def _create_twins_backbone(variant, pretrained=False, **kwargs):
    #TODO pretrained得到对应的pth文件
    backbone = Twins(features_only=True, **kwargs)
    return backbone, None, None, backbone.num_features


def twins_pcpvt_small(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins_backbone('twins_pcpvt_small', pretrained=pretrained, **model_kwargs)



def twins_pcpvt_base(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins_backbone('twins_pcpvt_base', pretrained=pretrained, **model_kwargs)



def twins_pcpvt_large(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins_backbone('twins_pcpvt_large', pretrained=pretrained, **model_kwargs)



def twins_svt_small(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins_backbone('twins_svt_small', pretrained=pretrained, **model_kwargs)



def twins_svt_base(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins_backbone('twins_svt_base', pretrained=pretrained, **model_kwargs)



def twins_svt_large(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins_backbone('twins_svt_large', pretrained=pretrained, **model_kwargs)


#Mine:
def tswins_small(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], use_swin=True, **kwargs)
    return _create_twins_backbone('tswins_small', pretrained=pretrained, **model_kwargs)


def tswins_base(img_size=512, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=img_size, patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], use_swin=True, **kwargs)
    return _create_twins_backbone('tswins_base', pretrained=pretrained, **model_kwargs)
