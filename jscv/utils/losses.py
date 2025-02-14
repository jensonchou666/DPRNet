from jscv.losses.useful_loss import *
import torch.nn.functional as F






weight_wrong_refine = [
    [0.1, 1, 5], [1.2, 0.3, 5], [1.5, 0.8, 5]
]

def wrong_refine_losslayer(ignore_index,
                           weights=weight_wrong_refine,
                           predwrong_input="coarse"):
    
    sce = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
    fine = MainLoss(sce)
    coarse = CoarsePredLoss(sce)
    pred_is_coarse = predwrong_input == "coarse"
    predwrong = PredWrongLoss(ignore_index, pred_is_coarse=pred_is_coarse)
    return JointMultiLoss([fine, coarse, predwrong], weights)






def create_wrongstrengthen_loss_layer(ignore_index,
                                      wrongstrengthen_weight=4,
                                      prededgeloss_weight=2,
                                      predwrongloss_weight=2,
                                      smooth_epoch=4,
                                      edge_index=None):
    wrongst_loss = WrongStrengthenLoss(ignore_index=ignore_index,
                                       smooth_epoch=smooth_epoch,
                                       loss_weight=wrongstrengthen_weight)

    return PredEdgeWrongLoss(0.05,
                             ignore_index,
                             prededgeloss_weight=prededgeloss_weight,
                             predwrongloss_weight=predwrongloss_weight,
                             smooth_epoch=smooth_epoch,
                             extra_loss=wrongst_loss,
                             edge_index=edge_index)


def create_prededgewrong_loss_layer(ignore_index,
                                    prededgeloss_weight=2,
                                    predwrongloss_weight=2,
                                    smooth_epoch=4,
                                    edge_index=None):
    return PredEdgeWrongLoss(0.05,
                             ignore_index,
                             prededgeloss_weight=prededgeloss_weight,
                             predwrongloss_weight=predwrongloss_weight,
                             smooth_epoch=smooth_epoch,
                             edge_index=edge_index)


def create_predwrong_loss_layer(ignore_index,
                                predwrongloss_weight=2,
                                smooth_epoch=4):
    return PredWrongLoss(0.05,
                         ignore_index,
                         predwrongloss_weight=predwrongloss_weight,
                         smooth_epoch=smooth_epoch)


def create_predloss_loss_layer(ignore_index,
                               predlossloss_weight=2,
                               smooth_epoch=4):
    diceLoss = DiceLoss(smooth=0.05, ignore_index=ignore_index)
    return PredLossLoss(0.05,
                        ignore_index,
                        extra_loss=diceLoss,
                        predlossloss_weight=predlossloss_weight,
                        smooth_epoch=smooth_epoch)


def sce_dice_loss(ignore_index):
    # return loss, _, _

    loss = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
        DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    aux_losses = []
    aux_weights = []

    return loss, aux_losses, aux_weights


def unetformer_loss(ignore_index):
    loss = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
        DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    aux_losses = [
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
    ]
    aux_weights = [0.4]

    return loss, aux_losses, aux_weights


# def unetformer_loss(ignore_index):
#     return UnetFormerLoss(ignore_index=ignore_index)