from jscv.backbone.cmt import *

def Cmt_S(img_size=512):
    backbone = cmt_s(pretrained=False, extract_features=True, img_size=img_size)
    backbone_features = backbone.embed_dims
    return backbone, None, None, backbone_features


def Cmt_B(img_size=512):
    backbone = cmt_b(pretrained=False, extract_features=True, img_size=img_size)
    backbone_features = backbone.embed_dims
    return backbone, None, None, backbone_features