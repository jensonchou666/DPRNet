# define the optimizer, use it after setting backbone_lr、backbone_weight_decay...
import torch
# from catalyst.contrib.nn import Lookahead
from catalyst import utils


# # 创建学习率更新策略，这里是每个step更新一次学习率，以及使用warmup
# def poly_warmup_lr_scheduler(optimizer,
#                              epochs: int,
#                              power=0.9,
#                              num_step: int = 1,
#                              warmup=True,
#                              warmup_epochs=1,
#                              warmup_factor=1e-3):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs = 0

#     def f(x):
#         if warmup and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             # warmup过程中lr倍率因子从warmup_factor -> 1
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             # 参考deeplab_v2: Learning rate policy
#             return (1 - (x / (epochs * num_step)))**power

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


from torch.optim.lr_scheduler import _LRScheduler

class PolyLrScheduler(_LRScheduler):
    def __init__(self, optimizer, epochs, power=0.9, last_epoch=-1, verbose=False):
        self.epochs = epochs
        self.power = power
        super(PolyLrScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lr_rate = (1 - self.last_epoch / self.epochs) ** self.power
        return [group['initial_lr'] * lr_rate for group in self.optimizer.param_groups]



# def poly_lr_scheduler(optimizer, epochs, power):
#     def f(x):
#         lr = (1 - x / epochs) ** power
#         return lr
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def optimizer_AdamW_Step(net: torch.nn.Module,
                         lr=0.001,
                         weight_decay=0.001,
                         step_size=4,
                         max_epoch=44,
                         last_rate=0.01,
                         parameters='all'):

    if parameters == 'all':
        parameters = net.parameters()

    optimizer = torch.optim.AdamW(parameters,
                                  lr=lr,
                                  weight_decay=weight_decay)
    # optimizer = Lookahead(optimizer)
    gamma = last_rate ** (step_size / max_epoch)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, lr_scheduler


def optimizer_AdamWCosine(net: torch.nn.Module,
                          lr,
                          weight_decay,
                          parameters='all',
                          T_0=15,
                          T_mult=2):

    if parameters == 'all':
        parameters = net.parameters()

    base_optimizer = torch.optim.AdamW(parameters,
                                       lr=lr,
                                       weight_decay=weight_decay)
    # optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult)

    return optimizer, lr_scheduler


def optimizer_AdamWCosine_custom(net: torch.nn.Module,
                                 lr,
                                 weight_decay,
                                 layerwise_params={},
                                 T_0=15,
                                 T_mult=2):
    # eg. layerwise_params={ "backbone.*": dict{lr=0.01, weight_decay=0.1} }
    net_params = utils.process_model_params(net,
                                            layerwise_params=layerwise_params)
    return optimizer_AdamWCosine(net, lr, weight_decay, net_params, T_0,
                                 T_mult)


def optimizer_AdamWCosine_backbone(net: torch.nn.Module,
                                   lr,
                                   weight_decay,
                                   backbone_lr,
                                   backbone_weight_decay,
                                   T_0=15,
                                   T_mult=2):
    layerwise_params = {
        "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
    }
    return optimizer_AdamWCosine_custom(net,
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        layerwise_params=layerwise_params)


if __name__ == "__main__":
    net = torch.nn.Conv2d(3, 100, 3)
    optimizer, lr_scheduler_1 = optimizer_AdamWCosine(net, 2.5e-4, 2.5e-4)
    optimizer, lr_scheduler_2 = optimizer_AdamWCosine(net, 1, 2)
    print('\nlr_scheduler_1: ', lr_scheduler_1.__dict__)
    print('\nlr_scheduler_2: ', lr_scheduler_2.__dict__)
    lr_scheduler_2.step()
    lr_scheduler_2.step()
    lr_scheduler_2.step()
    lr_scheduler_2.step()
    print('\nlr_scheduler_2 after step(): ', lr_scheduler_2.__dict__)
    lr_scheduler_1.load_state_dict(lr_scheduler_2.__dict__)
    print('\n1 Loaded 2: ', lr_scheduler_1.__dict__)
