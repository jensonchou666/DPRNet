from jscv.hr_models.pathes_segmentor import *
from .dynamic_patches import *
from .context_injector import *

from jscv.utils.trainer import Evaluator
from jscv.utils.utils import do_once, wrap_text, TimeCounter
from jscv.utils.overall import global_dict
from jscv.models.pred_stat_c import PredStatistic
import sys
import random
from abc import ABC, abstractmethod

'''
    #TODO 增加纯推理的测试代码， 我只训练集验证集是方便做实验
    
    #? 不再有网络耦合， 全局网络、局部网络可以是任意网络
    
        全局网络， 输入 img, mask,  输出dict: 'prediction', 'ctx_feat' (上下文特征，可以是队列), 
                            'last_feat' (主干最后层特征, 用于生成pdt),  'loss' 损失，可以不要
        局部网络， 输入 img, ctx_feat, mask,  输出dict: 'prediction', 'loss' 损失，可以不要
        
        #TODO 扩展 losses 为 字典

'''

do_validation  = False #? 用来优化、测验推理速度

do_debug = False
timer0 = TimeCounter(do_debug)

_log_counter_ = 0
_log_step_ = 1

''' 此字典一般无需修改 '''
_threshold_learner_args_ = {
    # 'select_rate_target': 0.25,
    # 'threshold_init': 0.05,
    'base_factor': 0.5,
    'adjustment_decay': 0.95,
    'window_size': 3,
    'max_adjustment': 0.01,
    'positive_relation': False,
    'base_factor_min': 0.001,
    'base_factor_max': 0.1,
}

''' 基于以下字典修改 '''
dynamic_manager_args_train = {
    'PDT_size':                 (16,16),        #? patch_difficulty_table 补丁难度表的大小, 现只支持16x16及以下
    'max_pathes_nums':          16,             #? 以PDT里的最小补丁为单位，显存足够就可以调高

    #? 对于整个训练集的细化比例， 比验证集的高，以扩充数据
    #TODO 研究随机采样，否则大量简单样本总是被过滤，就完全不会经过局部分支
    'refinement_rate':          0.4,            #? 整个训练集用于细化的比例，需要先开启阈值学习
    
    'use_threshold_learning':   True,           #? 开启阈值学习
    'learn_epoch_begin':        6,              #? PDT预测比较较稳定后才开始阈值学习
    'threshold_init':           0.02,          #? 初始阈值
    'threshold_rate_1x1':       0.6,            #? 1x1的分配阈值降低
    'score_compute_classtype':  PNScoreComputer, #? 默认的分数计算方式
    'score_compute_kargs':      {
        'fx_points': [(2, 0.4), (36, 0.1)],    #? 请参考PNScoreComputer的说明
        'gx_point': (4, 0.5)
    },
    'min_pathes_nums': 2, #? 扔掉过少补丁数的批次，可提高一点速度
    # 'min_pathes_nums_rate': 0.1,
    'threshold_learner_args': _threshold_learner_args_,
    'kernel_dict': None, #? 替换 kernel_dict可以扔掉一些不想要的kernel分配
    'h_bigger': False, #? True: 竖向分配优先， False: 横向分配优先
    'toCUDA': True, #? 在cuda运行
}

''' 此验证集的设置 开启了阈值学习。 但是在实际的测试推理时，测试集不应该被统计，阈值也不应该通过测试集学习 '''
dynamic_manager_args_val = {
    'PDT_size':                 (16,16),        #? patch_difficulty_table 补丁难度表的大小, 现只支持16x16及以下
    'max_pathes_nums':          36,             #? 无梯度的推理，故比训练集高，
    'refinement_rate':          0.28,            #? 整个验证集用于细化的比例，需要先开启阈值学习

    'use_threshold_learning':   True,           #? 开启阈值学习
    'learn_epoch_begin':        6,              #? PDT预测比较较稳定后才开始阈值学习
    'threshold_init':           0.04,          #? 初始阈值
    'threshold_rate_1x1':       0.6,            #? 1x1的分配阈值降低
    'score_compute_classtype':  PNScoreComputer, #? 默认的分数计算方式
    'score_compute_kargs':      {
        'fx_points': [(2, 0.4), (36, 0.1)],    #? 请参考PNScoreComputer的说明
        'gx_point': (4, 0.5)
    },
    # 'min_pathes_nums': 2,
    'min_pathes_nums_rate': 0.1,
    'threshold_learner_args': _threshold_learner_args_,
    'kernel_dict': None, #? 替换 kernel_dict可以扔掉一些不想要的kernel分配
    'h_bigger': False, #? True: 竖向分配优先， False: 横向分配优先
    'toCUDA': True, #? 在cuda运行
}

dynamic_manager_args_val_no_learning = dynamic_manager_args_val.copy()
dynamic_manager_args_val_no_learning.update({
    'use_threshold_learning':   False,
    'threshold_set_as_another': 2,  #? 固定设定为训练集阈值的两倍
})


class DynamicPatchesManager(nn.Module):
    '''
        仅用于训练或验证
        不能用于纯推理
    '''
    def __init__(self, manager:DynamicPatchesGrouping,
                 threshold, #another_manager,
                 threshold_learner:ThresholdLearner=None, #? 为None则不动态调整阈值
                 epoch_stage='train', #? 'train' or 'val'
                 threshold_set_as_another=-1, #? 若 k>0, 始终设置阈值为 another_manager阈值的k倍
                 learn_epoch_begin=6,
                 ):
        super(DynamicPatchesManager, self).__init__()
        self.manager = manager
        self.threshold_learner = threshold_learner
        self.epoch_idx = -1
        self.epoch_stage = epoch_stage
        # self.another_manager = another_manager
        self.threshold_set_k = threshold_set_as_another
        self.learn_epoch_begin = learn_epoch_begin
        self.flag_on_working = False
        learner_base_factor = None
        if threshold_learner is not None:
            learner_base_factor = threshold_learner.base_factor
        self.set_params(threshold, learner_base_factor)

    def set_another_manager(self, another_manager):
        assert isinstance(another_manager, DynamicPatchesManager)
        self.another_manager = [another_manager] #? 避免循环加载

    def load_state_dict(self, state_dict, strict = True):
        super(DynamicPatchesManager, self).load_state_dict(state_dict, strict)
        print("threshold:", self.threshold, flush=True)
        if self.threshold_learner is not None:
            self.threshold_learner.threshold = float(self.threshold)
            self.threshold_learner.base_factor = float(self.learner_base_factor)

    def set_params(self, threshold, learner_base_factor=None):
        self.threshold = nn.parameter.Parameter(torch.tensor(threshold, dtype=torch.float), False)
        if learner_base_factor is not None:
            self.learner_base_factor = nn.parameter.Parameter(
                torch.tensor(learner_base_factor, dtype=torch.float), False)

    def scan_patches_difficulty_table(self, PDT: torch.Tensor):
        trainer = global_dict['trainer']
        # from jscv.utils.trainer import Model_Trainer
        # assert isinstance(trainer, Model_Trainer)
        assert self.epoch_stage == trainer.stage
        manager = self.manager
        am = self.another_manager[0]
        PDT = PDT.detach()

        if self.epoch_idx != trainer.epoch_idx:
            if self.threshold <= 0:
                self.set_params(0.04)
            print('self.threshold', self.threshold)
            for am in (self, self.another_manager[0]):
                if am.flag_on_working: #am.epoch_idx >= 0:
                    am.manager.clear()
                    am.flag_on_working = False
                    info = am.manager.analyze_all()
                    info['threshold'] = round(float(am.threshold), 8) #? 上一轮的值
                    learner = am.threshold_learner
                    ''' 调整阈值 '''
                    if trainer.epoch_idx > am.learn_epoch_begin and learner is not None:
                        info['learner_base_factor'] = learner.base_factor   #? 上一轮的值
                        info['learner_target'] = learner.select_rate_target
                        new_threshold, diff = learner.adjust_threshold(
                            info['select_rate'])
                        am.set_params(new_threshold, am.learner_base_factor)
                    f = global_dict['log_file']
                    for _f in (sys.stdout, f):
                        print(f"Dynamic Patches Grouping Informations:", file=_f)
                        text = str(info)
                        idx = text.find("'threshold'")
                        text = text[:idx] + '\n' + text[idx:]
                        print(text, '\n', '-'*80, '\n', file=_f, flush=True)
            manager.init()
            manager.analyze_init()
            self.flag_on_working = True
            if self.threshold_set_k > 0:
                threshold = self.another_manager[0].threshold * self.threshold_set_k
                self.set_params(threshold)
            self.epoch_idx = trainer.epoch_idx
        allocated_groups = manager.scan_patches_difficulty_table(PDT, float(self.threshold))
        manager.analyze_once(allocated_groups, PDT)
        return allocated_groups



def create_train_val_dynamic_managers(
    train_kargs=dynamic_manager_args_train,
    val_kargs=dynamic_manager_args_val):
    
    managers = []
    for kargs in (train_kargs, val_kargs):
        if 'min_pathes_nums_rate' in kargs:
            kargs['min_pathes_nums'] = kargs['min_pathes_nums_rate'] * kargs['max_pathes_nums']
        managers.append(DynamicPatchesGrouping(do_init=False, **kargs))
    train_manager, val_manager = managers
    
    learners = [None, None]
    for j, kargs in ((0, train_kargs), (1, val_kargs)):
        if kargs['use_threshold_learning']:
            learners[j] = ThresholdLearner(
                select_rate_target=kargs['refinement_rate'],
                threshold_init=kargs['threshold_init'],
                **kargs['threshold_learner_args']
                )
    train_learner, val_learner = learners
    DM_train = DynamicPatchesManager(
        train_manager, train_kargs['threshold_init'], train_learner, 'train',
        train_kargs.get('threshold_set_as_another', -1), train_kargs['learn_epoch_begin']
    )
    DM_val = DynamicPatchesManager(
        val_manager, val_kargs['threshold_init'], val_learner, 'val',
        val_kargs.get('threshold_set_as_another', -1), val_kargs['learn_epoch_begin']
    )
    DM_train.set_another_manager(DM_val)
    DM_val.set_another_manager(DM_train)
    return DM_train, DM_val


class PDTHead(nn.Sequential):
    def __init__(self, channel):
        super(PDTHead, self).__init__(
            nn.Linear(channel, channel//2),
            nn.ReLU6(),
            nn.Linear(channel//2, 1),
        )

class GlobalBranchBase(nn.Module):
    key_pred        = 'pred'
    key_ctx_feat    = 'ctx_feat'
    key_last_feat   = 'last_feat'
    key_loss        = 'loss'
    def __init__(self, last_channel, context_channel):
        super(GlobalBranchBase, self).__init__()
        self.last_channel = last_channel
        self.context_channel = context_channel

    @abstractmethod
    def forward(self, img, mask=None):
        pass

#? 局部分支 示例
class LocalBranchBase(nn.Module):
    key_pred        = 'pred'
    key_loss        = 'loss'
    
    @abstractmethod
    def forward(self, img, ctx, mask=None):
        pass

#TODO 实现 Decoder仅返回各层特征， 然后由 GlobalEncoderDecoder 决定 ctx_feat 和 last_feat 的选择
class GlobalEncoderDecoderV1(GlobalBranchBase):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 last_channel,
                 context_channel,
                 context_out_channel=None,
                 decoder_tuple=False):
        if context_out_channel is None: context_out_channel = context_channel
        super(GlobalEncoderDecoderV1, self).__init__(last_channel, context_out_channel)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_tuple = decoder_tuple
        self.conv_ctx = nn.Sequential(nn.Conv2d(context_channel, context_out_channel, 1), 
                                      nn.BatchNorm2d(context_out_channel),
                                      nn.ReLU())

    def forward(self, img, mask=None):
        result = {}
        feats = self.encoder(img)
        result[self.key_last_feat] = feats[-1]
        if self.decoder_tuple:
            pred, ctx = self.decoder(feats)
        else:
            pred, ctx = self.decoder(*feats)
        ctx = self.conv_ctx(ctx)
        result[self.key_pred] = pred
        result[self.key_ctx_feat] = ctx
        # result[self.key_loss] = None #交给DPRNet处理损失
        return result


class LocalEncoderDecoderV1(LocalBranchBase):
    def __init__(self, 
                 encoder, 
                 decoder,
                 ):
        super(LocalEncoderDecoderV1, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, ctx, mask=None):
        result = {}
        feats = self.encoder(img)
        
        # feats = [f.contiguous() for f in feats]

        pred = self.decoder(feats, ctx=ctx)
        result[self.key_pred] = pred
        # result[self.key_loss] = None #交给DPRNet处理损失
        return result


class DPRNet(nn.Module):
    '''
        Dynamic Patches Refinement Network
        #TODO 写文档
    '''
    
    #? 如键值不一致，修改这些变量
    key_pred        = 'pred'
    key_ctx_feat    = 'ctx_feat'
    key_last_feat   = 'last_feat'
    key_loss        = 'loss'

    def __init__(
            self,
            #? Global
            global_branch: GlobalBranchBase,
            local_branch: LocalBranchBase=None,
            pdt_head_cls=PDTHead,

            global_loss_layer = None,
            local_loss_layer = None,

            loss_weights:dict = {
                'global': 1,
                'local': 1,
                'local_first': 1,
                'pdt': 1,
            },

            global_patches:dict ={'train':(2,2), 'val':(1,1)}, #? 防止G-Branch图太大, 导致显存溢出, 不影响 dynamic_grouping
            global_downsample=1/4,
            local_downsample=1,
            
            dynamic_manager_args_train:dict = dynamic_manager_args_train,
            dynamic_manager_args_val:dict = dynamic_manager_args_val,

            ignore_index=None,

            #? G-Branch直接加载官方权重时, 前N不训练局部
            local_warmup_epoch=0,
            context_zero_prob={'init': 0.4, 'last_epoch': 2}, #? 上下文特征置0的概率, 从0 ~ last_epoch, 概率从 init ~ 0

            #TODO stage=1 直接交给 patch_segmentor来完成（单独的配置文件）
            train_stage=3,  # 1:仅分割, 2:分割+分类，3:全局-局部网络-联合训练   

            SaveImagesManagerClass=SaveImagesManager,
            save_images_args=dict(per_n_step=15,),

            PredictionStatisticClass=None, #TODO 与模型耦合！
            do_prediction_statistics=False,
            cfg_file=None,
        ):
        '''
            train_setting_list:  per_item: (patches, downsample, Probability)
            val_setting:(patches, downsample)
        '''
        super().__init__()
        assert global_downsample*4 == local_downsample
        self.global_branch = global_branch
        self.local_branch = local_branch
        self.global_loss_layer = global_loss_layer
        self.local_loss_layer = local_loss_layer
 
        self.context_zero_prob = context_zero_prob

        keys = ['global', 'local', 'local_first', 'pdt']
        for k in keys:
            if not loss_weights.get(k, None):
                loss_weights[k] = 1
        self.loss_weights = loss_weights

        self.global_patches = global_patches
        self.global_downsample = global_downsample
        self.local_downsample = local_downsample
        
        ''' 修改单个global_patch下的 PDT size '''
        pdtH, pdtW = dynamic_manager_args_train['PDT_size']
        gH, gW = global_patches['train']
        dynamic_manager_args_train['PDT_size'] = (pdtH//gH, pdtW//gW)
        pdtH, pdtW = dynamic_manager_args_val['PDT_size']
        gH, gW = global_patches['val']
        dynamic_manager_args_val['PDT_size'] = (pdtH//gH, pdtW//gW)
        self.train_dynamic_manager, self.val_dynamic_manager = \
            create_train_val_dynamic_managers(dynamic_manager_args_train, dynamic_manager_args_val)

        self.stage = train_stage

        self.warmup_epoch = local_warmup_epoch

        if self.stage != 1:
            #? patch_difficulty_table 补丁难度表 分类头
            self.pdt_head = pdt_head_cls(channel=global_branch.last_channel)

        self.ignore_index = ignore_index
        self.sim = SaveImagesManagerClass(**save_images_args)
        
        self.do_pred_stat = do_prediction_statistics
        self.cfg_file = cfg_file
        self.counter = 0
        self.pred_logits = False

        if do_prediction_statistics:
            PredStatisticClass = PredStatistic if PredictionStatisticClass is None else PredictionStatisticClass
            self.pred_statistic_c = PredStatisticClass.from_cfg(cfg_file)
        self.stage_bk = self.stage


    def forward_validation(self, batch):
        img, mask = batch['img'], batch['gt_semantic_seg']
        training = 'train' if self.training else 'val'
        PH, PW = global_patches = self.global_patches[training]

        g_outputs, l_outputs, result = [], [], {}
        seg_losses_g = 0
        seg_losses_l = 0
        pdt_losses = 0
        img_patches, mask_patches = div_patches(img, mask, global_patches, 1)
        
        global timer0
        

        if self.stage != 1:
            dynamic_manager = self.dynamic_manager[0]
            pdtH, pdtW = dynamic_manager.manager.PDT_size

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            timer0.begin()
            
            B,C,H,W = imgij.shape
            assert B == 1, '为简化开发， "B≠1" 暂时是不支持的'

            g_input = resize_to(imgij, self.global_downsample)

            #? 阶段一: Global-Branch 粗预测、上下文特征提取

            result = self.global_branch(g_input, mask=maskij)
            g_seg   = result[self.key_pred]
            context = result[self.key_ctx_feat]
            g_last  = result[self.key_last_feat]
            

            g_seg = resize_to_y(g_seg, maskij)
            
            if self.key_loss in result:
                seg_loss_g = result[self.key_loss]
            else:
                seg_loss_g = self.global_loss_layer(g_seg, maskij)
            del result
            seg_loss_g *= self.loss_weights['global']
            seg_losses_g += seg_loss_g.detach()

            g_seg = g_seg.argmax(1)
            g_outputs.append(g_seg)

            timer0.record_time('1 global_branch')
            #? 阶段二: PDT预测， PDT Patch Difficulty Table
            patch_H, patch_W = H//pdtH, W//pdtW
            pdt_pred = F.adaptive_avg_pool2d(g_last, (pdtH, pdtW)).permute(0, 2, 3, 1)
            pdt_pred = self.pdt_head(pdt_pred).squeeze(-1).sigmoid()
            del g_last
            # print('torch.mean(pdt_pred)', torch.mean(pdt_pred))

            #TODO 改
            pdt_loss, pdt_label, err_rate = self.pdt_loss_fn(
                pdt_pred, g_seg.reshape(B, pdtH, patch_H, pdtW, patch_W).transpose(2,3),
                maskij.reshape(B, pdtH, patch_H, pdtW, patch_W).transpose(2,3))

            pdt_losses += pdt_loss.detach()


            timer0.record_time('2 pdt_head')

            refined_seg = g_seg.clone()
            pdt_pred = pdt_pred.squeeze()
            batches = dynamic_manager.scan_patches_difficulty_table(pdt_pred)
            local_img = resize_to(imgij, self.local_downsample)

            timer0.record_time('3 dynamic_manager')
            for i, batch in enumerate(batches):

                img_batch, mask_batch, context_batch = dynamic_manager.manager\
                    .crop_images_one_batch([local_img, maskij, context], batch)
                # img_batch, mask_batch, context_batch = img_batch.contiguous(), \
                #         mask_batch.contiguous(), context_batch.contiguous() #TODO
                context_batch = context_batch.contiguous()

                result = self.local_branch(img_batch, ctx=context_batch, mask=mask_batch)

                # timer0.record_time('local_branch')

                l_seg = result[self.key_pred]
                l_seg = resize_to_y(l_seg, mask_batch)

                if self.key_loss in result:
                    seg_loss_l = result[self.key_loss]
                else:
                    seg_loss_l = self.local_loss_layer(l_seg, mask_batch)

                seg_losses_l += seg_loss_l.detach()

                l_seg = resize_to_y(l_seg, mask_batch)
                dynamic_manager.manager.replace_cropped([refined_seg], [l_seg], batch)
            l_outputs.append(refined_seg)
            timer0.record_time('loss.backward() ', last=True)

        if do_debug:
            print('timer0.str_total_porp()', timer0.str_total_porp())

        g_seg = recover_mask(g_outputs, global_patches)
        l_seg = recover_mask(l_outputs, global_patches)


        result = {'pred': l_seg.detach(), 'coarse_pred': g_seg.detach(), 
                    'losses': {'main_loss':seg_losses_g+pdt_losses+seg_losses_l,
                                'seg_loss_g': seg_losses_g,
                                'seg_loss_l': seg_losses_l,
                                'pdt_loss': pdt_losses,
                                }
                    }

        return result


    def forward(self, batch:dict):
        if do_validation and not self.training:
            return self.forward_validation(batch)
        epoch_idx = global_dict['trainer'].epoch_idx
        do_warm = epoch_idx < self.warmup_epoch
        
        epoch_sub = self.context_zero_prob['last_epoch'] - epoch_idx
        
        if not self.training or epoch_sub <= 0:
            self.ctx_z_prob = 0
        else:
            self.ctx_z_prob = self.context_zero_prob['init'] * (epoch_sub / self.context_zero_prob['last_epoch'])

        self.doJointTrain = True
        if not self.training or self.stage != 3:
            self.doJointTrain = False

        self.dynamic_manager = [self.train_dynamic_manager] if self.training else [self.val_dynamic_manager]
        DM = self.dynamic_manager[0]
        #TODO 前N轮 不能存图像了
        if do_warm: self.stage = 1

        #? 如果显存溢出，调整 batch_size
        try:
            result = self.forward_train(batch)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                #TODO 现在暂时有问题 会发生过量 递减，所以暂时不永久改变 max_pathes_nums
                Success = False
                max_pathes_nums_bk = DM.manager.max_pathes_nums
                while not Success:
                    ss = 'train' if self.training else 'val'
                    _print_all_(f"CUDA out of memory on 【 {ss} 】")
                    max_pathes_nums = DM.manager.max_pathes_nums
                    max_pathes_nums_new = max_pathes_nums - 1
                    if max_pathes_nums_new <= 0: raise "Error, max_pathes_nums_new <= 0"
                    DM.manager.max_pathes_nums = max_pathes_nums_new
                    DM.manager.init()
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    _print_all_(f"max_pathes_nums changed: {max_pathes_nums} -> {max_pathes_nums_new}")
                    try:
                        result = self.forward_train(batch)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            continue
                        else: raise
                    Success = True
                DM.manager.max_pathes_nums = max_pathes_nums_bk-1 #? 只减 1
                DM.manager.init()
                _print_all_(f"@max_pathes_nums set to: {max_pathes_nums_bk-1}")
            else:
                raise

        if do_warm:
            result['coarse_pred'] = result['pred']

        #TODO 现在只支持 stage=3
        if self.sim.step() and self.stage != 1:
            img, mask = batch['img'], batch['gt_semantic_seg']
            
            imgid = batch.get('id')
            if imgid is not None:
                imgid = imgid[0]

            hard_pred = result['pdt_pred'] > float(DM.threshold)  #? 全1x1方式的难易分配
            # hard_label = result['err_rate'] > self.labal_err_bondary
            pdt = result['pdt_pred']
            hard_pred = hard_pred
            img = img.squeeze(0)
            mask = mask.squeeze(0)
            if self.stage == 3:
                g_seg = result['coarse_pred'].squeeze(0)
                l_seg = result['pred'].squeeze(0)
                self.sim.dynamic_patch_image_saving(img, mask, g_seg, l_seg, pdt, imgid=imgid,
                                                    hard_pred=hard_pred, manager=DM.manager,
                                                    global_patch_size = result['global_patches'],
                                                    batch_grouping_list=result['batch_grouping_list'])

        if self.do_pred_stat:
            self.pred_statistic_c.step(result['pred'])

        if do_warm: self.stage = self.stage_bk


        return result


    def forward_train(self, batch:dict):
        torch.cuda.empty_cache()
        img, mask = batch['img'], batch['gt_semantic_seg']
        training = 'train' if self.training else 'val'
        PH, PW = global_patches = self.global_patches[training]

        #? 默认输出直接就是分割图， 但后续的一些操作比如 (预测统计与标签纠错) 要用到 logits
        #TODO 改为 logits 和 classification 同时输出，会更加通用
        do_pred_logits = hasattr(self, 'pred_logits') and self.pred_logits

        
        g_outputs, l_outputs, result = [], [], {}
        seg_losses_g = 0
        seg_losses_l = 0
        pdt_losses = 0
        img_patches, mask_patches = div_patches(img, mask, global_patches, 1)
        
        global timer0
        

        if self.stage != 1:
            pdt_labels, pdt_preds, err_rates = [], [], []
            batch_grouping_list = []
            dynamic_manager = self.dynamic_manager[0]
            pdtH, pdtW = dynamic_manager.manager.PDT_size

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            timer0.begin()
            
            B,C,H,W = imgij.shape
            assert B == 1, '为简化开发， "B≠1" 暂时是不支持的'

            g_input = resize_to(imgij, self.global_downsample)

            #? 阶段一: Global-Branch 粗预测、上下文特征提取

            result = self.global_branch(g_input, mask=maskij)
            g_seg   = result[self.key_pred]
            context = result[self.key_ctx_feat]
            g_last  = result[self.key_last_feat]
            timer0.record_time('1 global_branch')

            g_seg = resize_to_y(g_seg, maskij)
            
            if self.key_loss in result:
                seg_loss_g = result[self.key_loss]
            else:
                seg_loss_g = self.global_loss_layer(g_seg, maskij)
            del result
            seg_loss_g *= self.loss_weights['global']
            seg_losses_g += seg_loss_g.detach()

            if do_pred_logits:
                g_seg_logits = g_seg.detach()
                g_outputs.append(g_seg_logits)
                g_seg = g_seg.argmax(1)
            else:
                g_seg = g_seg.argmax(1)
                g_outputs.append(g_seg.detach())

            if self.stage == 1 and self.training:
                    seg_loss_g.backward()
                    # optimize_step(self, 'optimizer_global')
                    optimize_step_now(self, 'optimizer')
            else:
                #? 阶段二: PDT预测， PDT Patch Difficulty Table
                patch_H, patch_W = H//pdtH, W//pdtW
                pdt_pred = F.adaptive_avg_pool2d(g_last, (pdtH, pdtW)).permute(0, 2, 3, 1)
                pdt_pred = self.pdt_head(pdt_pred).squeeze(-1).sigmoid()
                del g_last
                # print('torch.mean(pdt_pred)', torch.mean(pdt_pred))

                #TODO 改
                pdt_loss, pdt_label, err_rate = self.pdt_loss_fn(
                    pdt_pred, g_seg.reshape(B, pdtH, patch_H, pdtW, patch_W).transpose(2,3),
                    maskij.reshape(B, pdtH, patch_H, pdtW, patch_W).transpose(2,3))

                pdt_losses += pdt_loss.detach()
                pdt_preds.append(pdt_pred)
                pdt_labels.append(pdt_label)
                err_rates.append(err_rate)

                if self.training:
                    if not self.doJointTrain:
                        (seg_loss_g+pdt_loss).backward()
                        optimize_step_now(self, 'optimizer')

                timer0.record_time('2 pdt_head')
                if self.stage == 3:
                    #? 阶段三: Local-Branch 困难补丁细化
                    if do_pred_logits:
                        refined_seg = g_seg_logits.clone()
                    else:
                        refined_seg = g_seg.clone()
                    pdt_pred = pdt_pred.squeeze()
                    batches = dynamic_manager.scan_patches_difficulty_table(pdt_pred)
                    local_img = resize_to(imgij, self.local_downsample)

                    random.shuffle(batches) #TODO 避免 极少补丁分组 跑到 first
                    timer0.record_time('3 dynamic_manager')
                    # print("###", idx, local_img.shape)
                    for i, batch in enumerate(batches):
                        # timer0.begin()

                        img_batch, mask_batch, context_batch = dynamic_manager.manager\
                            .crop_images_one_batch([local_img, maskij, context], batch)
                        img_batch, mask_batch = img_batch.contiguous().detach(), mask_batch.contiguous()
                        context_batch = context_batch.contiguous()
                        # print("  @@@", i, img_batch.shape)

                        doJointTrain = self.doJointTrain and (i==0)
                        if self.training and not doJointTrain:
                            do_zero = False if (self.ctx_z_prob == 0) else (random.random() < self.ctx_z_prob)
                            if do_zero:
                                context_batch = torch.zeros_like(context_batch, device=context_batch.device)
                            else:
                                #? detach(), 因为 后续分组时， G-Branch已经backward了，再backward会抛异常
                                context_batch = context_batch.detach()

                        timer0.record_time('crop_images_one_batch')

                        result = self.local_branch(img_batch, ctx=context_batch, mask=mask_batch)

                        timer0.record_time('local_branch')

                        l_seg = result[self.key_pred]
                        l_seg = resize_to_y(l_seg, mask_batch)
                        
                        if self.key_loss in result:
                            seg_loss_l = result[self.key_loss]
                        else:
                            seg_loss_l = self.local_loss_layer(l_seg, mask_batch)
                        # timer0.record_time('forward')
                        if self.training:
                            if doJointTrain:
                                seg_loss_l *= self.loss_weights['local_first']
                                (seg_loss_l + seg_loss_g + pdt_loss).backward()
                            else:
                                seg_loss_l *= self.loss_weights['local']
                                seg_loss_l.backward()
                            # timer0.record_time('backward')

                        # timer0.last('loss')

                        seg_losses_l += seg_loss_l.detach()
                        
                        if not do_pred_logits:
                            l_seg = l_seg.argmax(1)
                        
                        # l_seg = resize_to_y(l_seg, mask_batch)
                        dynamic_manager.manager.replace_cropped([refined_seg], [l_seg], batch)
                    timer0.record_time('4 local')
                    optimize_step_now(self, 'optimizer')
                    batch_grouping_list.append(batches)
                    timer0.record_time('6 optimize_step_now', last=True)
                    l_outputs.append(refined_seg)
        
        if do_debug:
            print(timer0.str_total_porp())

        g_seg = recover_mask(g_outputs, global_patches)

        if self.stage == 1:
            result = {'pred': g_seg, 'losses': {'main_loss':seg_losses_g}}
        elif self.stage == 2:
            result = {'pred': g_seg, 'losses': {'main_loss':seg_losses_g+pdt_losses,
                                                'seg_loss': seg_losses_g,
                                                'pdt_loss': pdt_losses}}
        else:
            l_seg = recover_mask(l_outputs, global_patches)
            result = {'pred': l_seg.detach(), 'coarse_pred': g_seg.detach(), 
                      'losses': {'main_loss':seg_losses_g+pdt_losses+seg_losses_l,
                                 'seg_loss_g': seg_losses_g,
                                 'seg_loss_l': seg_losses_l,
                                 'pdt_loss': pdt_losses,
                                 }
                      }

        def recover_map(x):
            # print(torch.concat(x).shape, PH, PW, pdtH, pdtW)
            return torch.concat(x).reshape(PH, PW, pdtH, pdtW).transpose(1,2).reshape(PH*pdtH, PW*pdtW)

        if self.stage != 1:
            result['pdt_pred'] = recover_map(pdt_preds)
            result['pdt_label'] = recover_map(pdt_labels)
            result['err_rate'] = recover_map(err_rates)
            result['pdt_size'] = (pdtH, pdtW)
            result['global_patches'] = (PH, PW)
        if self.stage == 3:
            result['batch_grouping_list'] = batch_grouping_list

        return result


    def inference(self, img):
        #TODO 以后需要实现图像推理的方法，不包含任何损失函数等不必要的计算，并做些优化
        pass


    def pdt_loss_fn_v1(self, x, pred, mask):
        B, SH, SW, H, W = mask.shape
        # print('pdt_loss_fn', x.shape, pred.shape, mask.shape)
        #TODO 增加 曲线损失
        err = (pred != mask)
        total = (H*W)
        if self.ignore_index is not None:
            valid = (mask != self.ignore_index)
            err = err & valid
            # total = valid.sum(-1).sum(-1) #!!! todo
        err_rate = err.sum(-1).sum(-1) / total * 100
        label = torch.ones(B,SH,SW, dtype=torch.float, device=pred.device)
        label[err_rate<8] = 0.4
        label[err_rate<4] = 0.2
        label[err_rate<2] = 0.1
        label[err_rate<1] = 0.05
        label[err_rate<0.5] = 0
        easyhard_loss = F.binary_cross_entropy(x, label) * self.loss_weights['pdt']
        # print('easyhard_loss', easyhard_loss)
        return easyhard_loss, label, err_rate

    def pdt_loss_fn(self, x, pred, mask):
        B, SH, SW, H, W = mask.shape
        err = (pred != mask)
        total = (H*W)
        if self.ignore_index is not None:
            valid = (mask != self.ignore_index)
            err = err & valid
        err_rate = err.sum(-1).sum(-1) / total * 100
        label = torch.ones(B,SH,SW, dtype=torch.float, device=pred.device)
        label[err_rate<8] = 0.4
        label[err_rate<4] = 0.2
        label[err_rate<2] = 0.1
        label[err_rate<1] = 0.05
        label[err_rate<0.5] = 0
        easyhard_loss = F.binary_cross_entropy(x, label) * self.loss_weights['pdt']
        # print('easyhard_loss', easyhard_loss)
        return easyhard_loss, label, err_rate

        

class EasyHardEvaluator(Evaluator):
    c0 = 'ground'
    c1 = 'water'

    def __init__(self, 
                 stat_patch=(8,8),
                 boundary=[0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6]
                 ):
        '''
            stat_patch: 只统计(8,8)情况， None：全统计
        '''
        super().__init__()
        self.boundary = boundary
        self.stat_patch = stat_patch
        self.reset()

    def append(self, result):
        if 'easyhard_pred' not in result:
            return

        easyhard_pred = result['easyhard_pred'].reshape(-1)
        err_rate = result['err_rate'].reshape(-1)
        if self.stat_patch is not None:
            if easyhard_pred.shape[0] != self.stat_patch[0] * self.stat_patch[1]:
                return
        # err_rate = torch.clone(err_rate)
        C = easyhard_pred.shape[0]
        self.count += C
        for i, (bd, N, easy_err, hard_err) in enumerate(self.stat_e):
            pred_easy = easyhard_pred < bd
            N += pred_easy.sum()
            s1 = torch.sum(err_rate[torch.nonzero(pred_easy)])
            s2 = torch.sum(err_rate[torch.nonzero(~pred_easy)])
            # print(pred_easy.sum(), C)
            # print('easy:', s1 / pred_easy.sum())
            # print('hard:', s2 / (C-pred_easy.sum()))
            easy_err += s1
            hard_err += s2
            self.stat_e[i] = (bd, N, easy_err, hard_err)
            # print(self.stat_e[i])

    def reset(self):
        self.count = 0
        self.count_label_easy = 0
        self.stat_e = []
        for bd in self.boundary:
            self.stat_e.append([bd, 0, 0, 0])

    def evaluate(self):
        stat_patch = self.stat_patch
        if stat_patch is None:
            stat_patch = ''
        result_str = f"easy-hard prediction{stat_patch}:\n"
        
        for (bd, N, easy_err, hard_err) in self.stat_e:
            if N == 0:
                continue
            r1 = N / self.count * 100
            r2 = easy_err / N
            r3 = hard_err / max(1, self.count-N)
            r1 = f'{r1:.2f}'
            r2 = f'{r2:.2f}'
            r3 = f'{r3:.2f}'
            result_str += f"boundary: {bd:>5}, ratio: {r1:>5}%, easy_err: {r2:>5}%, " + \
                f"hard_err: {r3:>5}%\n"
        self.reset()
        result_str += '-' * 50 + '\n'
        self.result_str = result_str
        return {}

    def __str__(self) -> str:
        return self.result_str

def _print_all_(msg):
    f = global_dict['log_file']
    for _f in (sys.stdout, f):
        print(msg, file=_f)


def test_model_pure_time_3d5K(model, easy_rate = 0.75, cudaId=1):
    '''
        计算模型的纯粹的推理时间， 3d5K图上
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']=f'{cudaId}'
    epochs = 50
    input = torch.zeros(1, 3, 1024*7//2, 1024*7//2)
    load_gpu = True
    

    torch.set_grad_enabled(False)
    if load_gpu:
        input = input.cuda()
    model = model.cuda().eval()
    warmup()


    count_1.DO_DEBUG = False
    import tqdm
    
    print("begin:")
    t0 = time()

    for i in tqdm.tqdm(range(epochs)):
        model.predict(input, zero_img=True, easy_rate=easy_rate)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')





if __name__ == "__main__":

    test_predict_speed_pure()