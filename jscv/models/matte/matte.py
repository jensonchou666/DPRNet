import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP



from .decoder import Decoder
from .mobilenet import MobileNetV2Encoder
from .refiner import Refiner
from .resnet import ResNetEncoder
from .utils import load_matched_state_dict


class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """
    
    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])
        
        self.do_pred_err = False


    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x
    
    def load_pretrained_state_dict(self, state_dict, print_stats=True):
        '''
            load_pretrained_deeplabv3_state_dict
        '''
        
        # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
        # This method converts and loads their pretrained state_dict to match with our model structure.
        # This method is not needed if you are not planning to train from deeplab weights.
        # Use load_state_dict() for normal weight loading.
        
        # Convert state_dict naming for aspp module
        state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}

        if isinstance(self.backbone, ResNetEncoder):
            # ResNet backbone does not need change.
            load_matched_state_dict(self, state_dict, print_stats)
        else:
            # Change MobileNetV2 backbone to state_dict format, then change back after loading.
            backbone_features = self.backbone.features
            self.backbone.low_level_features = backbone_features[:4]
            self.backbone.high_level_features = backbone_features[4:]
            del self.backbone.features
            load_matched_state_dict(self, state_dict, print_stats)
            self.backbone.features = backbone_features
            del self.backbone.low_level_features
            del self.backbone.high_level_features




class MattingBase(Base):
    """
    MattingBase is used to produce coarse global results at a lower resolution.
    MattingBase extends Base.
    
    Args:
        backbone: ["resnet50", "resnet101", "mobilenetv2"]
        
    Input:
        src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        bgr: (B, 3, H, W) the background image . Channels are RGB values normalized to 0 ~ 1.
    
    Output:
        pha: (B, 1, H, W) the alpha prediction. Normalized to 0 ~ 1.
        fgr: (B, 3, H, W) the foreground prediction. Channels are RGB values normalized to 0 ~ 1.
        err: (B, 1, H, W) the error prediction. Normalized to 0 ~ 1.
        hid: (B, 32, H, W) the hidden encoding. Used for connecting refiner module.
        
    Example:
        model = MattingBase(backbone='resnet50')
        
        pha, fgr, err, hid = model(src, bgr)    # for training
        pha, fgr = model(src, bgr)[:2]          # for inference
    """
    
    def __init__(self, backbone: str, backbone_scale=1., out_channels=2):
        super().__init__(backbone, in_channels=3, out_channels=out_channels)
        self.backbone_scale = backbone_scale
        self.do_pred_err = True
        
    def forward(self, x):

        x = F.interpolate(x,
                          scale_factor=self.backbone_scale,
                          mode='bilinear',
                          align_corners=False,
                          recompute_scale_factor=True)

        x_org = x
        _, _, h, w = x.shape
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = x[:, 0:1].clamp_(0., 1.)
        err = x[:, 1:2].clamp_(0., 1.)
        #hid = x[:, 2: ].relu_()
        # print("@MattingBase", x.shape, pha.shape)
        return {'pred': pha, "pred_err": err}


class MattingRefine(MattingBase):

    
    def __init__(self,
                 backbone: str,
                 refine_channels=[32, 24, 16, 12],
                 backbone_scale: float = 1/4,
                 refine_mode: str = 'sampling',
                 refine_sample_pixels: int = 80_000,
                 refine_threshold: float = 0.1,
                 refine_kernel_size: int = 3,
                 refine_prevent_oversampling: bool = True,
                 refine_patch_crop_method: str = 'unfold',
                 refine_patch_replace_method: str = 'scatter_nd'):
        # assert backbone_scale <= 1/2, 'backbone_scale should not be greater than 1/2'
        super().__init__(backbone, out_channels=2 + refine_channels[0])
        self.backbone_scale = backbone_scale
        self.refiner = Refiner(refine_channels,
                               refine_mode,
                               refine_sample_pixels,
                               refine_threshold,
                               refine_kernel_size,
                               refine_prevent_oversampling,
                               refine_patch_crop_method,
                               refine_patch_replace_method)
        
        self.do_refine = True

    def forward(self, x, **kargs):
        # assert src.size() == bgr.size(), 'src and bgr must have the same shape'
        # assert src.size(2) // 4 * 4 == src.size(2) and src.size(3) // 4 * 4 == src.size(3), \
        #     'src and bgr must have width and height that are divisible by 4'
        
        # Downsample src and bgr for backbone
        
        if "time_counter" in kargs:
            test_speed = True
            counter = kargs["time_counter"]
            counter.begin()
        else:
            test_speed = False

        
        
        org_shape = x.shape
        x = F.interpolate(x,
                          scale_factor=self.backbone_scale,
                          mode='bilinear',
                          align_corners=False,
                          recompute_scale_factor=True)
        
        # Base
        x, *shortcuts = self.backbone(x)

        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)

        pha_sm = x[:, 0:1].clamp_(0., 1.)
        coarse_pha = torch.clone(pha_sm)
        err_sm = x[:, 1:2].clamp_(0., 1.)
        hid_sm = x[:, 2:].relu_()
        
        if test_speed:
            counter.record_time("coarse")

        if self.do_refine:
            # print(pha_sm.shape, err_sm.shape)
            pha, ref_sm = self.refiner(pha_sm, err_sm, hid_sm, org_shape)
            pha = pha.clamp_(0., 1.)

            if test_speed:
                counter.record_time("refine", last=True)
            
            # print(pha.shape, pha_sm.shape, err_sm.shape)
            return {'pred': pha, "coarse_pred": coarse_pha, "pred_err": err_sm, "refine_region": ref_sm}
        else:
            # print(pha_sm.shape, err_sm.shape)
            pha_sm = F.interpolate(pha_sm, org_shape[2:], mode='bilinear')
            return {'pred': pha_sm, "pred_err": err_sm}


import kornia

def matting_loss_1(pred_pha_fine, true_pha, pred_pha_corse=None, pred_err=None):
    '''
        pred_pha_corse is not None: add coarse_pred_loss
        pred_err is not None:       add pred_err_loss

        pred_pha:   L1损失
        pred_err:   L1损失
    '''

    pha_fine_loss = F.l1_loss(pred_pha_fine, true_pha)
    
    lossD = {"main_loss": pha_fine_loss}

    if pred_pha_corse is not None:
        pred_pha_corse = F.interpolate(pred_pha_corse, true_pha.shape[2:], mode='bilinear')
        pha_coarse_loss = F.l1_loss(pred_pha_corse, true_pha)

        lossD["coarse_pred_loss"] = pha_coarse_loss

    if pred_err is not None:
        if pred_pha_corse is None:
            pred_pha = pred_pha_fine
        else:
            pred_pha = pred_pha_corse    #优先用coarse_pha

        pred_err = F.interpolate(pred_err, true_pha.shape[2:], mode='bilinear')
        true_err = torch.abs(pred_pha.detach() - true_pha)
        pred_err_loss = F.l1_loss(pred_err, true_err)

        lossD["pred_err_loss"] = pred_err_loss

    return lossD


def matting_loss_2(pred_pha_fine, true_pha, pred_pha_corse=None, pred_err=None):
    '''
        pred_pha:   L1损失 + sobel
        pred_err:   L1损失
    '''
    true_pha_sobel = kornia.sobel(true_pha)
    pha_fine_loss = F.l1_loss(pred_pha_fine, true_pha)
    pha_fine_loss += F.l1_loss(kornia.sobel(pred_pha_fine), true_pha_sobel)

    lossD = {"main_loss": pha_fine_loss}

    if pred_pha_corse is not None:
        pred_pha_corse = F.interpolate(pred_pha_corse, true_pha.shape[2:], mode='bilinear')
        pha_coarse_loss = F.l1_loss(pred_pha_corse, true_pha)
        pha_coarse_loss += F.l1_loss(kornia.sobel(pred_pha_corse), true_pha_sobel)

        lossD["coarse_pred_loss"] = pha_coarse_loss

    if pred_err is not None:
        if pred_pha_corse is None:
            pred_pha = pred_pha_fine
        else:
            pred_pha = pred_pha_corse

        pred_err = F.interpolate(pred_err, true_pha.shape[2:], mode='bilinear')
        true_err = torch.abs(pred_pha.detach() - true_pha)
        pred_err_loss = F.l1_loss(pred_err, true_err)

        lossD["pred_err_loss"] = pred_err_loss

    return lossD


def matting_loss_3(pred_pha_fine, true_pha, pred_pha_corse=None, pred_err=None):
    '''
        pred_pha:   L1损失 + sobel
        pred_err:   L1损失 + sobel
    '''
    true_pha_sobel = kornia.sobel(true_pha)
    pha_fine_loss = F.l1_loss(pred_pha_fine, true_pha)
    pha_fine_loss += F.l1_loss(kornia.sobel(pred_pha_fine), true_pha_sobel)

    lossD = {"main_loss": pha_fine_loss}

    if pred_pha_corse is not None:
        pred_pha_corse = F.interpolate(pred_pha_corse, true_pha.shape[2:], mode='bilinear')
        pha_coarse_loss = F.l1_loss(pred_pha_corse, true_pha)
        pha_coarse_loss += F.l1_loss(kornia.sobel(pred_pha_corse), true_pha_sobel)

        lossD["coarse_pred_loss"] = pha_coarse_loss

    if pred_err is not None:
        if pred_pha_corse is None:
            pred_pha = pred_pha_fine
        else:
            pred_pha = pred_pha_corse

        pred_err = F.interpolate(pred_err, true_pha.shape[2:], mode='bilinear')
        true_err = torch.abs(pred_pha.detach() - true_pha)
        pred_err_loss = F.l1_loss(pred_err, true_err)
        pred_err_loss += F.l1_loss(kornia.sobel(pred_err), kornia.sobel(true_err))

        lossD["pred_err_loss"] = pred_err_loss

    return lossD

matting_loss = matting_loss_2


class MattingRefineLoss(torch.nn.Module):
    def __init__(self, global_dict, MattingRefine_model,
                 pred_err_after_e=4, refine_after_e=15) -> None:
        super().__init__()
        self.global_dict = global_dict
        self.model = MattingRefine_model
        MattingRefine_model.do_refine = False
        self.pred_err_after_e = pred_err_after_e
        self.refine_after_e = refine_after_e

    def forward(self, result, true_pha):
        trainer = self.global_dict["trainer"]
        t = trainer.epoch_idx
        
        if self.model.do_refine:
            return matting_loss(result["pred"], true_pha, result["coarse_pred"],
                                result["pred_err"])
        else:
            if t >= self.refine_after_e:
                self.model.do_refine = True
            if t < self.pred_err_after_e:
                return matting_loss(result["pred"], true_pha)
            else:
                return matting_loss(result["pred"], true_pha, pred_err=result['pred_err'])
