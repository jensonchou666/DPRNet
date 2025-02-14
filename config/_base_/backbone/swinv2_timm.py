from jscv.backbone.swinv2_timm import SwinTransformerV2


def _create_swin_transformer_v2_backbone(variant, pretrained=False, **kwargs):
    #TODO pretrained得到对应的pth文件
    backbone = SwinTransformerV2(features_only=True, **kwargs)
    return backbone, None, None, backbone.num_features

#@register_model
def swinv2_tiny_window16_256(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_tiny_window16_256', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_tiny_window8_256(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_tiny_window8_256', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_small_window16_256(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=16, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_small_window16_256', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_small_window8_256(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_small_window8_256', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_base_window16_256(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_base_window16_256', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_base_window8_256(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_base_window8_256', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_base_window12_192_22k(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_base_window12_192_22k', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_base_window12to16_192to256_22kft1k(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2_backbone(
        'swinv2_base_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_base_window12to24_192to384_22kft1k(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2_backbone(
        'swinv2_base_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_large_window12_192_22k(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer_v2_backbone('swinv2_large_window12_192_22k', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_large_window12to16_192to256_22kft1k(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2_backbone(
        'swinv2_large_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)


#@register_model
def swinv2_large_window12to24_192to384_22kft1k(img_size=256, pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=img_size, window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2_backbone(
        'swinv2_large_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)



if __name__ == '__main__':
    backbone, _, _, num_features = swinv2_base_window8_256(512)
    print("num_features", num_features)
    import torch
    ret = backbone(torch.randn(2, 3, 512, 512))
    for x in ret:
        print(x.shape)