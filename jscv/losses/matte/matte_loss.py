
from torch.nn import functional as F
import kornia
import torch



def matting_loss_L1(pred_pha, true_pha):
    '''
        pred_pha, pred_fgr:  Two predictive-outputs of an image matting model
    '''
    #true_msk = true_pha != 0
    return F.l1_loss(pred_pha, true_pha) + \
           F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha))

class MatteLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred_pha, true_pha):
        return matting_loss_L1(pred_pha, true_pha)




def matting_loss_with_err(pred_pha, pred_err, true_pha):
    '''
        pred_err: As input to refiner

    '''
    true_err = torch.abs(pred_pha.detach() - true_pha)
    # true_msk = true_pha != 0

    a = F.l1_loss(pred_pha, true_pha)
    b = F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha))
    c = F.l1_loss(pred_err, true_err)
    '''
    0.04030410200357437 0.004917321726679802 0.04030410200357437
    0.09784364700317383 0.007034607697278261 0.09785336256027222
    0.012440778315067291 0.003817778779193759 0.012440778315067291
    0.031875379383563995 0.004039788618683815 0.037933241575956345
    0.04880847781896591 0.005280421115458012 0.04880847781896591
    0.02254171296954155 0.0025333291850984097 0.02254171296954155
    0.04107508063316345 0.003948458004742861 0.04107508063316345
    0.02689950540661812 0.0053476374596357346 0.02689950540661812
    0.030212152749300003 0.0037761477287858725 0.030212152749300003
    0.11378023028373718 0.00551589485257864 0.1137869581580162
    0.022483276203274727 0.0028124633245170116 0.022483276203274727
    0.059254128485918045 0.006426983512938023 0.059254128485918045
    '''
    return {"main_loss": a + b, "pred_err_loss": c}

    # return F.l1_loss(pred_pha, true_pha) + \
    #        F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha)) + \
    #         F.l1_loss(pred_err, true_err)

        #    F.mse_loss(pred_err, true_err)
        #    F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk) 
        #    + \

from jscv.utils.overall import global_dict

class MatteLoss_with_err(torch.nn.Module):
    def __init__(self, pred_err_after_e=-1) -> None:
        super().__init__()
        self.pred_err_after_e = pred_err_after_e
    
    def forward(self, result, true_pha):
        pred_pha = result['pred']
        trainer = global_dict["trainer"]
        t = trainer.epoch_idx
        
        if self.pred_err_after_e < 0 or t < self.pred_err_after_e:
            return matting_loss_L1(pred_pha, true_pha)
        else:
            return matting_loss_with_err(pred_pha, result['pred_err'], true_pha)

