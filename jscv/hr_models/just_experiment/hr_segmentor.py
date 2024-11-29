'''

ignore、decoder...

#!!!
TODO:
改成bce分类


local_net 只处理复杂块，数据量减少了
增加简单块的训练？
尝试：


note.
为什么直接分类陆地和水，并填充整个块，而不是接一个decoder，
decoder对错误块没有改善效果(globle_net太小)
反而使easy块中出现噪声像素，降低精度

TODO 研究-全局网络, 3网络
TODO 划分数据集

'''

import random
from jscv.hr_models.base_models import *

from jscv.utils.trainer import Evaluator, SegmentEvaluator
from jscv.utils.overall import global_dict


do_debug_time = False
LOG_CLASSIFY_EVERY = -1 #print classify-acc if >0

class HRSegmentor(nn.Module):
    """
        高分辨率 语义分割器
        mask 必须是0、1 二分类
        
        [easy_hard, ground_water]
    """

    def __init__(
            self,

            local_net_cls=None,
            local_net_args=None,

            global_backbone_name="resnet18",
            global_patches=(8,8),    #[列，行]   <=local_patches
            local_patches=(8,8),     #[列，行]   确定能整除
            global_downsample=1,
            global_only=False,

            seg_loss_layer: nn.Module = None,

            easy_hard_predict_boundary=0.5,
            classify_predict_boundary=0.8,
            classify_weights=[0.2, 2.0],    # ground、 water

            label_threshold=[0.1,0.2],   # ground、 water
            label_threshold_2=[0.5,4],

            do_fuse_global_features=True,

            global_pretrain_path=None,
            optimizer_step_interval=10,
            save_images_per=20, # save images if > 0
            save_org_img=True,
            save_mask=True,
            train_cls_pred_after_n_epoachs=0, # N轮后再训练easy块的分类
            # to_cuda=True,  #默认
            keys_name=['img', 'gt_semantic_seg', 'id'],

            ):
        super().__init__()

        self.local_patches = local_patches
        self.global_backbone_name = global_backbone_name

        self.global_only = global_only
        self.global_patches = global_patches
        self.global_downsample = global_downsample

        self.seg_loss_layer = seg_loss_layer
        self.frozen_classify_net_epoachs = 0
        self.do_fuse_global_features = do_fuse_global_features
        self.keys_name = keys_name
        self.optimizer_step_interval = optimizer_step_interval
        self.classify_weights = classify_weights
        self.label_threshold_2 = label_threshold_2
        self.label_threshold = label_threshold
        self.bd1 = easy_hard_predict_boundary
        self.bd2 = classify_predict_boundary
        self.ne_cls = train_cls_pred_after_n_epoachs
        self.train_cls = False
        self.save_org_img = save_org_img
        self.save_mask = save_mask

        if global_patches == local_patches:
            self.global_output_size = 1
            assert False, 'TODO 暂未实现, global_patches通常比local_patches少'
        else:
            self.global_output_size = (local_patches[0]//global_patches[0],
                                       local_patches[1]//global_patches[1])

        self.global_backbone = ResNetEncoder(global_backbone_name, 
                                             features_only=False, 
                                             avg_pool_to=self.global_output_size,
                                             num_classes=2)

        if not global_only:
            if do_fuse_global_features:
                global_channels = self.global_backbone.channels
            else:
                global_channels = None
            self.local_net = local_net_cls(global_channels=global_channels,**local_net_args)

        if global_pretrain_path is not None:
            self.pretrain_global(global_backbone=torch.load(global_pretrain_path))

        self.frozened = False
        self.epoach_idx = 0
        self.counter_easy = 0
        self.counter_hard = 0
        self.count_local_opt_i = 0
        self.sim = SaveImagesManager(per_n_step=save_images_per)
        
        self.time0 = time()


    def patch_classify_loss(self, x:torch.Tensor, mask:torch.Tensor):
        '''
            mask 取值：[0 或 1]
            第2个ground_water通道只对easy_patch计算
        
        '''
        B, N = mask.shape
        ratios = mask.sum(-1) / N * 100 # [B]
        C1,C2 = self.label_threshold

        easyhard_label = torch.zeros_like(ratios, device=x.device)
        easyhard_label[(ratios>C1)&(ratios<(100-C2))] = 1.

        cls_label = torch.zeros_like(ratios, device=x.device)
        cls_label[ratios>0.5] = 1   # cls_0:<0.5, cls_1:>0.5

        '''TODO weights'''
        easyhard_loss = F.binary_cross_entropy(x[:,0], easyhard_label.float())
        cls_loss = 0
        if self.train_cls:
            w0, w1 = self.classify_weights
            C1, C2 = self.label_threshold_2
            weight = (1-cls_label) * w0 + cls_label * w1
            weight[(ratios>C1)&(ratios<(100-C2))] = 0
            cls_loss = F.binary_cross_entropy(x[:,1], cls_label.float(), weight=weight)

        return easyhard_loss, cls_loss, easyhard_label, cls_label



    def div_patches(self, img, mask, n_patches):
        B, C, H, W = img.shape
        PH, PW = n_patches
        HL, WL = H//PH, W//PW
        img_p = img.reshape(B, C, PH, HL, PW, WL)
        mask_p = mask.reshape(B, PH, HL, PW, WL)
        img_patches, mask_patches = [], []
        for i in range(PH):
            for j in range(PW):
                img_patches.append(img_p[:, :, i, :, j])
                mask_patches.append(mask_p[:, i, :, j])
        return img_patches, mask_patches

    def div_img_patches(self, img, n_patches):
        B, C, H, W = img.shape
        PH, PW = n_patches
        HL, WL = H//PH, W//PW
        img_p = img.reshape(B, C, PH, HL, PW, WL)
        img_patches = []
        for i in range(PH):
            for j in range(PW):
                img_patches.append(img_p[:, :, i, :, j])
        return img_patches




    # def crop_features(self, features, is_hard):
    #     fs = []
    #     for f in features:

    def forward(self, batch):
        t0 = time()
        #if not self.global_only:

        result, (img, pred, mask, hard, label_hard, cls_pred_wrong) = self.forward_1(batch)
        if self.sim.step():
            PH, PW = self.global_patches
            SH, SW = self.global_output_size
            
            def compose_pathes(x):
                if len(x) == 1:
                    return x[0].reshape(SH,SW).bool()
                return torch.tensor(x).reshape(PH, PW, SH, SW).transpose(1,2).reshape(PH*SH,PW*SW).bool()
            
            hard = compose_pathes(hard)
            label_hard = compose_pathes(label_hard)
            cls_pred_wrong = compose_pathes(cls_pred_wrong)
            if pred is not None:
                pred = pred.squeeze(0)
            if self.save_org_img:
                img = img.squeeze(0)
            else:
                img = None

            self.sim.save_image_hard(img, pred, mask.squeeze(0),
                                     hard, label_hard, cls_pred_wrong,
                                     batch.get(self.keys_name[2])[0],
                                     save_mask=self.save_mask)
        if do_debug_time:
            print("spend", time()-t0)
        return result


    def forward_1(self, batch):
        if 'trainer' in global_dict:
            epoch_idx = global_dict['trainer'].epoch_idx
            if self.frozened:
                if self.epoach_idx < epoch_idx:
                    self.epoach_idx = epoch_idx
                    if epoch_idx == self.frozen_classify_net_epoachs:
                        for name, param in self.global_backbone.named_parameters():
                            param.requires_grad = True
                        print("frozen over")
                        self.frozened = False
            if not self.train_cls:
                if epoch_idx >= self.ne_cls:
                    self.train_cls = True


        img, mask = batch[self.keys_name[0]], batch[self.keys_name[1]]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        
        assert img.shape[0] == 1, "暂时，方便代码"
        
        result = {}

        seg_losses, easyhard_losses, cls_losses = 0, 0, 0
        img_patches, mask_patches = self.div_patches(img, mask, self.global_patches)

        outputs, hard_list, label_hard_list = [], [], []
        label_cls_list, cls_pred_wrong, cx_list = [], [], []

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            # i, j = idx//self.global_patches[0], idx%self.global_patches[0]
            imgij, maskij = imgij.cuda(), maskij.cuda()
            if self.global_downsample != 1:
                x, (x4, x3, x2, x1, x0) = self.global_backbone(
                    F.interpolate(imgij,scale_factor=self.global_downsample, mode="bilinear"))
            else:
                x, (x4, x3, x2, x1, x0) = self.global_backbone(imgij)
            x = torch.sigmoid(x)
            
            # B, SH, SW
            SH, SW = self.global_output_size
            B, C, H, W = imgij.shape
            
            maskij = maskij.reshape(B, SH, H//SH, SW, W//SW).transpose(2,3)
            
            x = x.reshape(B*SH*SW, -1)

            easyhard_loss, cls_loss, easyhard_label, cls_label = \
                self.patch_classify_loss(x, maskij.reshape(B*SH*SW,-1))
            

            # classify_loss *= self.patch_classify_loss_weight
            easyhard_losses += easyhard_loss
            cls_losses += cls_loss
            c_loss = easyhard_loss+cls_loss

            is_hard = (x[:,0]>self.bd1).int()
            cls_pred = (x[:,1]>self.bd2).int()

            cx_list.append(x)
            hard_list.append(is_hard)
            label_hard_list.append(easyhard_label)
            label_cls_list.append(cls_label)
            
            cls_pred_wrong.append(cls_pred!=easyhard_label)

            if self.global_only:
                if self.training and not self.frozened:  c_loss.backward()
            else:
                x_list = []
                imgij = imgij.reshape(B, C, SH, H//SH, SW, W//SW)
                seg_loss = 0
                hard_patchs = sum(is_hard)

                if not self.do_fuse_global_features:
                    del x4, x3, x2, x1, x0
                    if self.training and not self.frozened:
                        c_loss.backward()
                    for s, (hard, e_cls) in enumerate(zip(is_hard, cls_pred)):
                        k, p = s//SH, s%SH
                        if hard:
                            imgij_kp = imgij[:,:,k,:,p]
                            maskij_kp = maskij[:,k,p]
                            seg_x = self.local_net(imgij_kp)
                            if seg_x.shape[-2:] != (H,W):
                                seg_x = F.interpolate(seg_x, (H,W), mode="bilinear")
                            hard_seg_loss = self.seg_loss_layer(seg_x, maskij_kp)/hard_patchs
                            seg_loss += hard_seg_loss.detach()
                            if self.training:
                                hard_seg_loss.backward()
                                self.count_local_opt_i += 1
                                if self.count_local_opt_i >= self.optimizer_step_interval:
                                    self.count_local_opt_i = 0
                                    if hasattr(self, 'optimizer_local_net'):
                                        self.optimizer_local_net.step()
                                        self.optimizer_local_net.zero_grad()
                        elif e_cls==0:
                            seg_x = torch.tensor([1, 0]).cuda().unsqueeze(0).unsqueeze(
                                -1).unsqueeze(-1).repeat(1, 1, H, W)
                        else:
                            seg_x = torch.tensor([0, 1]).cuda().unsqueeze(0).unsqueeze(
                                -1).unsqueeze(-1).repeat(1, 1, H, W)
                        x_list.append(seg_x)
                else:
                    #TODO
                    for f in (x4, x3, x2, x1, x0):
                        _b, _c, _h, _w = f.shape
                        of.append(f.reshape(_b, _c, SH, _h//SH, SW, _w//SW))
                    x4, x3, x2, x1, x0 = of
                    del of
                    # fs = self.fuse_features((l_x4, l_x3, l_x2, l_x1, l_x0), (
                    #     x4.detach()[:,:,k,:,p],
                    #     x3.detach()[:,:,k,:,p],
                    #     x2.detach()[:,:,k,:,p],
                    #     x1.detach()[:,:,k,:,p],
                    #     x0.detach()[:,:,k,:,p],
                    # ))                               
                    # rand_n = random.randint(0, len(hardPs_idx)-1)

                x = torch.stack(x_list, 0).reshape(SH, SW, B, 2, H, W).permute(
                    2, 3, 0, 4, 1, 5
                ).reshape(B, 2, SH*H, SW*W)
                outputs.append(x)
                
                if seg_loss != 0:
                    seg_losses += seg_loss.detach()

            if self.training and hasattr(self, 'optimizer_global'):
                self.optimizer_global.step()
                self.optimizer_global.zero_grad()

        result.update({'classify_x': cx_list,
                       'label_hard': label_hard_list,
                       'label_cls': label_cls_list})

        result["losses"] = {'main_loss': easyhard_losses+cls_losses+seg_losses,
                            'easyhard_loss':easyhard_losses, 'easy_cls_loss':cls_losses}
        if not self.global_only:
            result["losses"]['seg_loss'] = seg_losses
            outputs = torch.concat(outputs, dim=0)
            B,C,H,W = outputs.shape
            GPH, GPW = self.global_patches
            outputs = outputs.reshape(1, GPH, GPW, C, H, W).permute(0, 3, 1, 4, 2, 5).reshape(1,C,H*GPH, W*GPW)
            result["pred"] = outputs

        return result, (img, result.get('pred'), mask, hard_list, label_hard_list, cls_pred_wrong)


    def predict_1(self, img: torch.Tensor):
        #TODO!!!!!!
        assert img.shape[0] == 1, "暂时，方便代码"

        img_patches = self.div_img_patches(img, self.global_patches)

        outputs = []

        for idx, imgij in enumerate(img_patches):
            imgij = imgij.cuda()
            if self.global_downsample != 1:
                imgij_d = F.interpolate(imgij, scale_factor=self.global_downsample, mode="bilinear")
            x, (x4, x3, x2, x1, x0) = self.global_backbone(imgij_d)
            del imgij_d
            logits_x = torch.sigmoid(x).squeeze(-1)
            
            def _resize_(xx, target=imgij):
                if xx.shape[-2:] != target.shape[-2:]:
                    xx = F.interpolate(xx, target.shape[-2:], mode="bilinear")
                return xx
            
            if self.global_output_size == 1:
                assert "暂不实现"
            else:
                patch_classify_loss = 0
                # B, SH, SW
                SH, SW = self.global_output_size
                B, C, H, W = imgij.shape
                imgij = imgij.reshape(B, C, SH, H//SH, SW, W//SW)
                assert B == 1
                is_hard = []
                for k in range(SH):
                    for p in range(SW):
                        logits_x_kp = logits_x[:, k, p]
                        if logits_x_kp < self.score_threshold:
                            is_hard.append(0)
                        else:
                            is_hard.append(1)

                if not self.global_only:
                    if sum(is_hard) == 0: # 所有均为easy
                        x = self.global_decoder(x4, x3, x2, x1, x0)
                        x = _resize_(x)
                        self.counter_easy += len(is_hard)
                    else:
                        of = []
                        for f in (x4, x3, x2, x1, x0):
                            _b, _c, _h, _w = f.shape
                            of.append(f.reshape(_b, _c, SH, _h//SH, SW, _w//SW))
                        x4, x3, x2, x1, x0 = of
                        del of
                        # hardPx_4,hardPx_3,hardPx_2,hardPx_1,hardPx_0 = [],[],[],[],[]
                        easyPs_idx, hardPs_idx = [], []
                        for s, _is_hard  in enumerate(is_hard):
                            if _is_hard:
                                hardPs_idx.append(s)
                            else:
                                easyPs_idx.append(s)
                        
                        self.counter_easy += len(easyPs_idx)
                        self.counter_hard += len(hardPs_idx)

                        if len(easyPs_idx) > 0:
                            easyPx_4,easyPx_3,easyPx_2,easyPx_1,easyPx_0 = [],[],[],[],[]
                            for s in easyPs_idx:
                                k, p = s//SH, s%SH
                                easyPx_4.append(x4[:,:,k,:,p])
                                easyPx_3.append(x3[:,:,k,:,p])
                                easyPx_2.append(x2[:,:,k,:,p])
                                easyPx_1.append(x1[:,:,k,:,p])
                                easyPx_0.append(x0[:,:,k,:,p])
                            easyPx_4 = torch.concat(easyPx_4, dim=0)
                            easyPx_3 = torch.concat(easyPx_3, dim=0)
                            easyPx_2 = torch.concat(easyPx_2, dim=0)
                            easyPx_1 = torch.concat(easyPx_1, dim=0)
                            easyPx_0 = torch.concat(easyPx_0, dim=0)
                            easy_x = self.global_decoder(easyPx_4, easyPx_3, easyPx_2, easyPx_1, easyPx_0)
                            easy_x = _resize_(easy_x, imgij[:,:, 0,:,0])


                        hard_x = []
                        

                        for s in hardPs_idx:
                            k, p = s//SH, s%SH
                            imgij_kp = imgij[:,:,k,:,p]

                            l_x4, l_x3, l_x2, l_x1, l_x0 = self.local_backbone(imgij_kp)
                            if self.do_fuse_global_features:
                                fs = self.fuse_features((l_x4, l_x3, l_x2, l_x1, l_x0), (
                                    x4[:,:,k,:,p],
                                    x3[:,:,k,:,p],
                                    x2[:,:,k,:,p],
                                    x1[:,:,k,:,p],
                                    x0[:,:,k,:,p],
                                ))
                            else:
                                fs = (l_x4, l_x3, l_x2, l_x1, l_x0)
                            hard_x_kp = self.local_decoder(*fs)
                            hard_x_kp = _resize_(hard_x_kp, imgij_kp)
                            hard_x.append(hard_x_kp)

                        s1, s2, x = 0, 0, []
                        for s, _is_hard  in enumerate(is_hard):
                            if _is_hard:
                                x.append(hard_x[s1])
                                s1+=1
                            else:
                                x.append(easy_x[s2:s2+1])
                                s2+=1
                        B, _, _h, _w = x[0].shape
                        x = torch.stack(x, 0).reshape(SH, SW, B, 2, _h, _w).permute(
                            2, 3, 0, 4, 1, 5
                        ).reshape(B, 2, SH*_h, SW*_w)

            if not self.global_only:
                outputs.append(x)

        if not self.global_only:
            outputs = torch.concat(outputs, dim=0)
            B,C,H,W = outputs.shape
            GPH, GPW = self.global_patches
            pred = outputs.reshape(1, GPH, GPW, C, H, W).permute(0, 3, 1, 4, 2, 5).reshape(1,C,H*GPH, W*GPW)
            return pred
        return None


        

    def pretrain_global(self, global_backbone=None, global_convert_fc=True):
        if global_backbone is not None:
            G_state_dict = self.global_backbone.state_dict()
            if global_convert_fc:
                global_backbone['fc.weight'] = G_state_dict['fc.weight']
                global_backbone['fc.bias'] = G_state_dict['fc.bias']
            self.global_backbone.load_state_dict(global_backbone)
            print("load global-backbone ok")


    def frozen_classify_net(self, epoach):
        self.frozen_classify_net_epoachs = epoach
        if epoach > 0:
            for name, param in self.global_backbone.named_parameters():
                param.requires_grad = False
            self.frozened = True
            print("frozened global_backbone")


class PatchClassifyEvaluator(Evaluator):
    c0 = 'ground'
    c1 = 'water'

    def __init__(self,
                 hard_bd=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7],
                 cls_bd=[0.3, 0.4, 0.50, 0.60, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        super().__init__()
        self.hard_bd = hard_bd
        self.cls_bd = cls_bd
        self.reset()

    def append(self, result):
        x = result['classify_x']
        label_hard = result['label_hard']
        label_cls = result['label_cls']
        x = torch.concat(x, 0)                      # [B,2]
        label_hard = torch.concat(label_hard, 0)    # [B]
        label_cls = torch.concat(label_cls, 0)
        
        label_easy = label_hard == 0
        label_easy_n = label_easy.sum()
        C = x.shape[0]
        self.count += C
        self.count_label_easy += label_easy_n
        

        for i, (bd, N_pred_easy, N_easy_right, N_hard_right) in enumerate(self.stat_e):
            pred_easy = x[:,0] < bd
            N_pred_easy += pred_easy.sum()
            N_easy_right += (pred_easy & label_easy).sum()
            N_hard_right += (~pred_easy & ~label_easy).sum()
            self.stat_e[i] = [bd, N_pred_easy, N_easy_right, N_hard_right]
        
        for i, (bd, N_pred_0, N_0_right, N_1_right) in enumerate(self.stat_c):
            pred_0 = label_easy & (x[:,1] < bd)
            pred_1 = label_easy & (x[:,1] >= bd)
            N_pred_0 += pred_0.sum()
            N_0_right += (pred_0 & (label_cls==0)).sum()
            N_1_right += (pred_1 & (label_cls==1)).sum()
            self.stat_c[i] = [bd, N_pred_0, N_0_right, N_1_right]


    def reset(self):
        self.count = 0
        self.count_label_easy = 0

        self.stat_e, self.stat_c = [], []
        for bd in self.hard_bd:
            self.stat_e.append([bd, 0, 0, 0])
        for bd in self.cls_bd:
            self.stat_c.append([bd, 0, 0, 0])

    def evaluate(self):
        self.result = {}
        self.result_str = ''

        c = self.count
        c_easy = self.count_label_easy
        
        self.result_str += "easy-hard prediction:\n"
        for bd, N_pred_easy, N_easy_right, N_hard_right in self.stat_e:
            pred_easy_rate = N_pred_easy/c *100
            N_pred_hard = c - N_pred_easy
            pred_easy_ACC = N_easy_right/N_pred_easy *100
            pred_hard_ACC = N_hard_right/N_pred_hard *100
            r = {
                f'RE_{int(bd*100)}': pred_easy_rate,
                f'AE_{int(bd*100)}': pred_easy_ACC,
                f'AH_{int(bd*100)}': pred_hard_ACC,
            }
            self.result.update(r)
            
            pred_easy_rate = f'{pred_easy_rate:.2f}'
            pred_easy_ACC = f'{pred_easy_ACC:.2f}'
            pred_hard_ACC = f'{pred_hard_ACC:.2f}'
            str_1 = f'boundary:{bd:>5}, ' \
                f'easy_ratio:{pred_easy_rate:>7}, ' \
                f'easy_acc:{pred_easy_ACC:>7}, ' \
                f'hard_acc:{pred_hard_ACC:>7}, '
            self.result_str += str_1+"\n"

        self.result_str += f"{self.c0}-{self.c1} prediction on easy-patches:\n"
        for bd, N_pred_0, N_0_right, N_1_right in self.stat_c:
            # pred_0_rate = N_pred_0/c_easy
            N_pred_1 = c_easy - N_pred_0
            pred_0_ACC = N_0_right/N_pred_0 *100
            pred_1_ACC = N_1_right/N_pred_1 *100
            r = {
                f'A0_{int(bd*100)}': pred_0_ACC,
                f'A1_{int(bd*100)}': pred_1_ACC,
            }
            self.result.update(r)
            
            pred_0_ACC = f'{pred_0_ACC:.4f}'
            pred_1_ACC = f'{pred_1_ACC:.4f}'
            str_1 = f'boundary:{bd:>5}, ' \
                f'{self.c0}_acc:{pred_0_ACC:>9}, ' \
                f'{self.c1}_acc:{pred_1_ACC:>9}, '
            self.result_str += str_1+"\n"

        self.reset()

        return self.result

    def __str__(self) -> str:
        if self.result is None:
            return 'None result'
        else:
            return self.result_str



class HRSeg_Evaluator(Evaluator):

    def from_config(cfg):
        ps_eval = PatchClassifyEvaluator()
        seg_eval = SegmentEvaluator.from_config(cfg)
        return HRSeg_Evaluator(ps_eval, seg_eval)

    def __init__(self, ps_eval: PatchClassifyEvaluator, seg_eval: SegmentEvaluator):
        super().__init__()
        self.ps_eval = ps_eval
        self.seg_eval = seg_eval
    
    def append(self, result):
        self.ps_eval.append(result)
        self.seg_eval.append(result)
    
    def evaluate(self):
        self.seg_eval.stat = self.stat
        self.ps_eval.stat = self.stat

        r = self.ps_eval.evaluate()
        r.update(self.seg_eval.evaluate())
        return r
    
    def __str__(self) -> str:
        return self.ps_eval.__str__() + "\n" + self.seg_eval.__str__()
    

def get_optimizer(net):
    from jscv.utils.optimizers import PolyLrScheduler
    poly_power = 2
    lr = 0.0001
    weight_decay = 0.0001
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)
    sch = PolyLrScheduler(optimizer, 40, poly_power)
    return optimizer, sch

#TODO :
def test_classify_net_effiction(use_evaluator=True):
    import os
    from tqdm import tqdm
    from jscv.losses.useful_loss import SCE_DIce_Loss
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    model = HRSegmentor(global_backbone_name='resnet18',
                        local_backbone_name='resnet101',
                        global_patches=(2,2),
                        global_downsample=1/2,
                        global_only=False,
                        seg_loss_layer=SCE_DIce_Loss(use_dice=False))

    c1 = 'work_dir/GID_Water/hr_segmentor_g_r18_d2-e40/final/epoch=39@A0_50=0.3969@hr_segmentor_g_r18_d2-GID_Water.ckpt'
    c2 = 'work_dir/GID_Water/hr_segmentor_ppn-e40/version_1/epoch=32@A0_40=0.7966@hr_segmentor_ppn-GID_Water.ckpt'#ppn
    c3 = 'work_dir/GID_Water/hr_segmentor_r101-e40/version_6/epoch=9@val_mIoU=73.05@hr_segmentor_r101-GID_Water.ckpt'
 
    ckpt = torch.load(c3)
    prefix = 'global_backbone.'
 
    state_dict = ckpt['state_dict']
    state_dict2 = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            state_dict2[k[len(prefix):]] = v

    model.global_backbone.load_state_dict(state_dict2)

    torch.set_grad_enabled(False)
    model = model.cuda().eval()
    model.frozen_classify_net(1)

    ds = water_seg_dataset()
    # model.optimizer_gb, sch1 = get_optimizer(model.global_backbone)
    # model.optimizer_gd, sch2 = get_optimizer(model.global_decoder)
    # model.optimizer_lb, sch3 = get_optimizer(model.local_backbone)
    # model.optimizer_ld, sch4 = get_optimizer(model.local_decoder)
    if use_evaluator:
        evalu = PatchClassifyEvaluator()
        tq = tqdm(total=ds.len)
    else:
        log_threshold = 0.5
        thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        LEN = len(thresholds)
        count_pred0, count_pred0_right, count_pred1_right = [0]*LEN, [0]*LEN, [0]*LEN
        count = 0

    t0 = time()

    for fi, (img, label) in enumerate(ds.batch()):
        result = model({"img":img, "gt_semantic_seg":label.long()})
        
        print('patch_classify_loss', result['losses']['patch_classify_loss'])

        if use_evaluator:
            evalu.append(result)
            tq.update(1)
        else:
            patch_classify_logits = result['patch_classify_logits']
            patch_labels = result['patch_labels']
            patch_classify_logits = torch.concat(patch_classify_logits, 0)
            patch_labels = torch.concat(patch_labels, 0)
            count += patch_classify_logits.shape[0]

            for i, threshold in enumerate(thresholds):
                idx = torch.nonzero(patch_classify_logits < threshold)
                pred0 = idx.numel()
                count_pred0[i] += pred0
                pred0_right = pred0 - torch.sum(patch_labels[idx]) #预测0合集里， label为0的元素数量
                count_pred0_right[i] += pred0_right
            
                idx = torch.nonzero(patch_classify_logits > threshold)
                pred1 = idx.numel()
                pred1_right = torch.sum(patch_labels[idx]) #预测1合集里， label为1的元素数量
                count_pred1_right[i] += pred1_right
                if threshold == log_threshold:
                    print(f"{fi}/{ds.len}  pred0-acc: {pred0_right}/{pred0},  pred1-acc: {pred1_right}/{pred1}")

    if use_evaluator:
        res = evalu.evaluate()
        print(evalu)
    else:
        for p0, p0r, p1r, threshold in zip(count_pred0, count_pred0_right, count_pred1_right, thresholds):
            print(f"\nthreshold = {threshold}:")
            print(f"pred0: {p0}/{count}, pred0-acc: {p0r}/{p0},  pred1-acc: {p1r}/{count-p0}")
            print(f"pred0: {(p0/count):.4f}, pred0-acc: {(p0r/p0):.4f},  pred1-acc: {(p1r/(count-p0)):.4f}")

    print("spend", time()-t0)



#TODO :
def test_predict_speed():

    imgs_save_dir = '0/1'
    do_save_imgs = True

    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    torch.set_grad_enabled(False)
    
    warmup()
    
    model = HRSegmentor(
        local_backbone_name='resnet101',
        global_backbone_name='resnet18',
        global_patches=(2,2),
        global_downsample=1/2,
        # g_decoder_args=dict(dec_channels=[10, 20, 40, 80], blocks=[1, 1, 1, 1], num_classes=2),
        # do_fuse_global_features=False,
        global_only=False)

    ''' load state_dict'''
    lb_path = 'pretrain/resnet101-63fe2227.pth'
    # lb_path = 'pretrain/resnet152-394f9c45.pth'

    c2 = 'work_dir/GID_Water/hr_segmentor_ppn-e40/version_1/epoch=32@A0_40=0.7966@hr_segmentor_ppn-GID_Water.ckpt'
    state_dict = torch.load(c2)['state_dict']
    prefix = 'ppn.'
    state_dict2 = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            state_dict2[k[len(prefix):]] = v
    # print(state_dict2.keys())
    model.global_backbone.load_state_dict(state_dict2)
    model.pretrain(local_backbone=torch.load(lb_path))
    model = model.cuda().eval()


    import tqdm

    ds = water_seg_dataset()

    t0 = time()
    tq = tqdm.tqdm(total=ds.len)

    for fi, (img, label) in enumerate(ds.batch()):
        tq.update(1)

        pred = model.predict(img)
        
        # print(fi, pred.shape, label.shape)

    print("finished, spend:", time()-t0)
    print("counter_hard:", model.counter_hard, ", counter_easy:", model.counter_easy, 
          ", rate:", model.counter_hard / (model.counter_hard+model.counter_easy))

    '''eval'''





if __name__ == "__main__":


    test_classify_net_effiction()




    exit()

    HRSegment = HRSegmentor()
    HRSegment.pretrain(torch.load('pretrain/resnet50-19c8e357.pth'), torch.load('pretrain/resnet18-5c106cde.pth'))
    HRSegment = HRSegment.cuda()

    from PIL import Image
    img_path = 'data/water_seg/val_from_train/image/GF2_PMS1__L1A0001765572-MSS1.tif'
    label_path = 'data/water_seg/val_from_train/label/GF2_PMS1__L1A0001765572-MSS1_label.tif'
    img = torch.from_numpy(np.array(Image.open(img_path))).permute(2, 0, 1).unsqueeze(0).float()
    label = torch.from_numpy(np.array(Image.open(label_path).convert("L"))).unsqueeze(0)

    # print(img.shape, label.shape)
    
    img = img.cuda()
    # print("img cuda!!")
    # while 1:
    #     pass
    
    HRSegment({'img': img, 'gt_semantic_seg': label})