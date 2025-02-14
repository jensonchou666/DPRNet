from config._base_.schedule.normal import *
from jscv.utils.trainer import SegmentEvaluator, Joint_Evaluator, EdgeAccEvaluator
from jscv.utils.load_checkpoint import load_checkpoint
import torch





init_learing_rate       = 1e-4          # 初始lr
weight_decay            = 1e-4
poly_power              = 2
use_dice                = False








#--------------- 非模型部分 ------------------------
train_batchsize_dict    = {None: 1}    # 高分辨率下， 全局的batchsize固定设置为1，别改
trainer_log_per_k       = 20           # 每k次，向 output.log 打印
save_topk               = 4            # 保存前k好的权重
logger_display_topk     = 20           # log.txt 文件前20排序
skip_analyse            = True         # 不使用analyser存图
#? Important 其它的一些设置，参考每个组件的from_config()方法里的说明



def get_optimizer(cfg, net, lr=1e-4):
    if isinstance(net, list):
        params = net
    else:
        params = net.parameters()
    optimizer = torch.optim.AdamW(params,
                                  lr=lr,
                                  weight_decay=cfg.weight_decay)
    sch = PolyLrScheduler(optimizer, cfg.max_epoch, cfg.poly_power)
    return optimizer, sch


def get_evaluators(cfg):
    return Joint_Evaluator(
        SegmentEvaluator(cfg.num_classes, cfg.classes, 'pred'),
        EdgeAccEvaluator(ignore_index=cfg.ignore_index, pred_key='pred'),
    )


def recursive_update(d1, d2):
    """
    使用d2递归更新d1
    """
    for key, value in d2.items():
        if isinstance(value, dict) and isinstance(d1.get(key), dict):
            # 如果两个值都是字典，递归更新
            recursive_update(d1[key], value)
        else:
            # 否则直接更新/新增
            d1[key] = value