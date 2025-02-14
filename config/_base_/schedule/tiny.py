from jscv.utils.optimizers import *

train_batch_size = 8
val_batch_size = 8
# lr设置
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01




check_val_every_n_epoch = 1
# 是否resume, 默认文件：weight_name.ckpt
# resume_ckpt = False
# 默认为加载预训练权重
skip_pretrain = False



# 每n个batch_size进行一次参数更新
accumulate_n = 1


#每轮数据Log到info文件，或者单独文件
log_to_infofile = True
#每轮Log统计topk项数据
logger_display_topk = 8
#logger_topk_monitor = 'val_OA'




# 保存最佳权重的策略
save_ckpt_monitor = 'val_OA'
save_ckpt_monitor_mode = 'max'
# 保存最好的k次权重, k=0：不保存权重， k=-1：每轮都保存
save_topk = 1
save_ckpt_last = True
#save_ckpt_per_k_epoch = 1
ckpt_split_char = '@'
# '{epoch} not in ckptname_metrix[0] is ok
ckptname_monitor_prefix = 'monitor='   # 方便找topk权重文件
ckptname_metrix = ['epoch={epoch}', 'monitor={monitor:.4}', '{model_name}-{dataset_name}']
#ckptname_metrix = ['e{epoch}', 'oa={val_OA:.4}', '{model_name}-{dataset_name}']
#ckptname_metrix = ['epoch={epoch}', 'val_OA={val_OA:.4}']
#ckptname_metrix = ['e{epoch}', 'oa={val_OA:.4}', '{model_name}-{dataset_name}-e{max_epoch}']





# evaluate时这些数据集要特殊操作
CutOffDatasets = [
        'potsdam', 'vaihingen', 'whubuilding', 'massbuilding', 'inriabuilding'
]



# recommend to set gpu-args while run tran.py manually
#gpus = 0
#strategy = None


# do_resume = False
