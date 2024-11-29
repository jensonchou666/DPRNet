'''
三分类

#!!!
TODO:
训练res101 baseline

尝试：

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

            classify_weights=[1.,4.,1.],# [full_ground, full_water, complex]

            classify_threshold=[0.1,0.5], # ground、 water

            do_fuse_global_features=True,
            patch_classify_loss_weight=1,

            global_pretrain_path=None,
            optimizer_step_interval=10,
            # to_cuda=True,  #默认
            keys_name=['img', 'gt_semantic_seg'],

            save_images_per=5, # save images if > 0
            ):
        super().__init__()

        self.local_patches = local_patches
        self.global_backbone_name = global_backbone_name
        self.classify_threshold = classify_threshold

        self.global_only = global_only
        self.global_patches = global_patches
        self.global_downsample = global_downsample
        self.patch_classify_loss_weight = patch_classify_loss_weight
        self.seg_loss_layer = seg_loss_layer
        self.frozen_classify_net_epoachs = 0
        self.do_fuse_global_features = do_fuse_global_features
        self.keys_name = keys_name
        self.optimizer_step_interval = optimizer_step_interval
        self.classify_weights = torch.tensor(classify_weights)
        

        if global_patches == local_patches:
            self.global_output_size = 1
        else:
            self.global_output_size = (local_patches[0]//global_patches[0],
                                       local_patches[1]//global_patches[1])

        self.global_backbone = ResNetEncoder(global_backbone_name, 
                                           features_only=False, 
                                           avg_pool_to=self.global_output_size,
                                           num_classes=3)

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


    def patch_classify_loss(self, logits_x:torch.Tensor, mask:torch.Tensor):
        '''[full_ground, full_water, complex'''
        B, N = mask.shape
        ratios = mask.sum(-1) / N * 100 # [B]
        C1,C2 = self.classify_threshold

        label = torch.full_like(ratios, 2, dtype=torch.uint8, device=logits_x.device)
        label[ratios<C1] = 0
        label[ratios>(100-C2)] = 1
        # print(logits_x.shape, label.shape)
        loss = F.cross_entropy(logits_x, label)
        # loss = F.cross_entropy(logits_x, label, self.classify_weights.to(logits_x.device))
        return loss, label

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
        do_save_img = False
        if not self.global_only:
            do_save_img = self.sim.step()

        result, (pred, mask, hard_list, label_hard_list) = self.forward_1(batch)
        if do_save_img:
            PH, PW = self.global_patches
            SH, SW = self.global_output_size
            hard = torch.tensor(hard_list).reshape(PH, PW, SH, SW).transpose(
                1,2).reshape(PH*SH,PW*SW)
            label_hard = torch.tensor(label_hard_list).reshape(
                PH, PW, SH, SW).transpose(1,2).reshape(PH*SH,PW*SW)
            self.sim.save_image_hard(pred.squeeze(0), mask.squeeze(0),
                                     hard.bool(), label_hard.bool())
        if do_debug_time:
            print("spend", time()-t0)
        return result


    def forward_1(self, batch):
        if self.frozened and 'trainer' in global_dict:
            epoch_idx = global_dict['trainer'].epoch_idx
            if self.epoach_idx < epoch_idx:
                self.epoach_idx = epoch_idx
                if epoch_idx == self.frozen_classify_net_epoachs:
                    for name, param in self.global_backbone.named_parameters():
                        param.requires_grad = True
                    print("frozen over")
                    self.frozened = False


        img, mask = batch[self.keys_name[0]], batch[self.keys_name[1]]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        
        assert img.shape[0] == 1, "暂时，方便代码"
        
        result = {}


        patch_classify_logits, patch_labels = [], []
        seg_losses, patch_classify_losses = 0, 0
        img_patches, mask_patches = self.div_patches(img, mask, self.global_patches)

        outputs, hard_list, label_hard_list = [], [], []

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            # i, j = idx//self.global_patches[0], idx%self.global_patches[0]
            imgij, maskij = imgij.cuda(), maskij.cuda()
            if self.global_downsample != 1:
                logits_x, (x4, x3, x2, x1, x0) = self.global_backbone(
                    F.interpolate(imgij,scale_factor=self.global_downsample, mode="bilinear"))
            else:
                logits_x, (x4, x3, x2, x1, x0) = self.global_backbone(imgij)
            # logits_x = torch.softmax(x, dim=-1)
            
            assert self.global_output_size != 1, 'TODO 暂未实现'

            # B, SH, SW
            SH, SW = self.global_output_size
            B, C, H, W = imgij.shape
            assert B == 1
            
            maskij = maskij.reshape(B, SH, H//SH, SW, W//SW).transpose(2,3)
            
            logits_x = logits_x.reshape(B*SH*SW, -1)
            patches_pred = logits_x.argmax(-1)
            patch_classify_loss, patches_label = self.patch_classify_loss(
                logits_x, maskij.reshape(B*SH*SW,-1))
            patch_classify_logits.append(logits_x)
            patch_labels.append(patches_label)

            patch_classify_loss *= self.patch_classify_loss_weight
            # if patch_classify_loss != 0:
            patch_classify_losses += patch_classify_loss

            label_is_hard = (patches_label==2).int()
            is_hard = (patches_pred==2).int()
            hard_list.append(is_hard)
            label_hard_list.append(label_is_hard)

            if self.global_only:
                if self.training:  patch_classify_loss.backward()
            else:
                x_list = []
                imgij = imgij.reshape(B, C, SH, H//SH, SW, W//SW)
                seg_loss = 0
                hard_patchs = sum(is_hard)

                if not self.do_fuse_global_features:
                    del x4, x3, x2, x1, x0
                    if self.training and not self.frozened:
                        patch_classify_loss.backward()
                    for s, cls_pred in enumerate(patches_pred):
                        k, p = s//SH, s%SH
                        if cls_pred == 2:
                            imgij_kp = imgij[:,:,k,:,p]
                            maskij_kp = maskij[:,k,p]
                            hard_x_kp = self.local_net(imgij_kp)
                            if hard_x_kp.shape[-2:] != (H,W):
                                hard_x_kp = F.interpolate(hard_x_kp, (H,W), mode="bilinear")
                            x_list.append(hard_x_kp)
                            hard_seg_loss = self.seg_loss_layer(hard_x_kp, maskij_kp)/hard_patchs
                            seg_loss += hard_seg_loss.detach()
                            if self.training:
                                hard_seg_loss.backward()
                                self.count_local_opt_i += 1
                                if self.count_local_opt_i >= self.optimizer_step_interval:
                                    self.count_local_opt_i = 0
                                    if hasattr(self, 'optimizer_local_net'):
                                        self.optimizer_local_net.step()
                                        self.optimizer_local_net.zero_grad()
                        elif cls_pred == 0:
                            easy_x = torch.tensor([1, 0]).cuda().unsqueeze(0).unsqueeze(
                                -1).unsqueeze(-1).repeat(1, 1, H, W)
                            x_list.append(easy_x)
                        elif cls_pred == 1:
                            easy_x = torch.tensor([0, 1]).cuda().unsqueeze(0).unsqueeze(
                                -1).unsqueeze(-1).repeat(1, 1, H, W)
                            x_list.append(easy_x)
                            
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

        result.update({'patch_classify_logits': patch_classify_logits, 'patch_labels': patch_labels})

        result["losses"] = {'main_loss': patch_classify_losses+seg_losses, 'patch_classify_loss':patch_classify_losses}
        if not self.global_only:
            result["losses"]['seg_loss'] = seg_losses
            outputs = torch.concat(outputs, dim=0)
            B,C,H,W = outputs.shape
            GPH, GPW = self.global_patches
            outputs = outputs.reshape(1, GPH, GPW, C, H, W).permute(0, 3, 1, 4, 2, 5).reshape(1,C,H*GPH, W*GPW)
            result["pred"] = outputs

        return result, (outputs, mask, hard_list, label_hard_list)


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
    # c0='full_ground',
    # c1='full_water',
    c0 = 'ground'
    c1 = 'water'
    c2 = 'hard'

    def __init__(self):
        super().__init__()
        self.reset()

    def append(self, result):
        logits = result['patch_classify_logits']
        labels = result['patch_labels']
        logits = torch.concat(logits, 0)  # [B,3]
        labels = torch.concat(labels, 0)    #[B]
        pred = logits.argmax(1)
        
        C = logits.shape[0]
        
        p0, p1, p2 = (pred == 0), (pred == 1), (pred == 2)
        m0, m1, m2 = (labels == 0), (labels == 1), (labels == 2)
        # print("\n@", torch.sum(m0), torch.sum(m1), torch.sum(m2))
        c_0_pred = torch.sum(p0)
        c_1_pred = torch.sum(p1)
        c_0_label = torch.sum(m0)
        c_1_label = torch.sum(m1)
        c_0_to_1 = torch.sum(m0 & p1)   # 0错分为1
        c_0_to_2 = torch.sum(m0 & p2)   # 0错分为2
        c_1_to_0 = torch.sum(m1 & p0)   # 1错分为0
        c_1_to_2 = torch.sum(m1 & p2)   # 1错分为2
        c_2_to_0 = torch.sum(m2 & p0)   # 2错分为0
        c_2_to_1 = torch.sum(m2 & p1)   # 2错分为1
        self.count += C
        self.c_0_pred += c_0_pred
        self.c_1_pred += c_1_pred
        self.c_0_label += c_0_label
        self.c_1_label += c_1_label
        self.c_0_to_1 += c_0_to_1   # 0错分为1
        self.c_0_to_2 += c_0_to_2   # 0错分为2
        self.c_1_to_0 += c_1_to_0   # 1错分为0
        self.c_1_to_2 += c_1_to_2   # 1错分为2
        self.c_2_to_0 += c_2_to_0   # 2错分为0
        self.c_2_to_1 += c_2_to_1   # 2错分为1


        if LOG_CLASSIFY_EVERY>0 and self.cx % LOG_CLASSIFY_EVERY == 0: #TODO
            hard = C-c_0_pred-c_1_pred
            print(f"{hard}/{C}, easy-acc: {c_2_to_0+c_2_to_1}/{c_0_pred+c_1_pred},\
                hard-acc: {c_0_to_2+c_1_to_2}/{hard}")
        self.cx += 1

    def reset(self):
        self.count = 0
        self.c_0_pred = 0
        self.c_1_pred = 0
        self.c_0_label = 0
        self.c_1_label = 0
        
        self.c_0_to_1 = 0   # 0错分为1
        self.c_0_to_2 = 0   # 0错分为2
        self.c_1_to_0 = 0   # 1错分为0
        self.c_1_to_2 = 0   # 1错分为2
        self.c_2_to_0 = 0   # 2错分为0
        self.c_2_to_1 = 0   # 2错分为1

        self.cx = 0

    def evaluate(self):
        self.result = {}
        # self.result_L = []
        
        c = self.count
        c3_p = c-self.c_0_pred-self.c_1_pred
        c3_l = c-self.c_0_label-self.c_1_label
        
        def aa(x, r=4):
            return round(float(x*100), r)
        
        err_1_to_0 = self.c_1_to_0 / self.c_1_label   #重要
        easy_acc = 1 - (self.c_2_to_0+self.c_2_to_1)/(self.c_0_pred+self.c_1_pred)

        # self.c_1_to_0/self.c_0_pred
        acc_0 = 1 - (self.c_1_to_0+self.c_2_to_0)/self.c_0_pred
        recall_0 = 1 - (self.c_0_to_1+self.c_0_to_2)/self.c_0_label

        acc_1 = 1 - (self.c_0_to_1+self.c_2_to_1)/self.c_1_pred
        recall_1 = 1 - (self.c_1_to_0+self.c_1_to_2)/self.c_1_label   #重要

        acc_2 = 1 - (self.c_0_to_2+self.c_1_to_2)/c3_p
        recall_2 = 1 - (self.c_2_to_0+self.c_2_to_1)/c3_l

        self.result.update({
            'err_1t0': aa(err_1_to_0),
            'acc_easy':aa(easy_acc),
            
        })
        self.total_result = {
            f'label_ratio': {self.c0:aa(self.c_0_label/c), self.c1:aa(self.c_1_label/c),
                             self.c2:aa(c3_l/c)},
            f'predict_ratio': {self.c0:aa(self.c_0_pred/c), self.c1:aa(self.c_1_pred/c),
                               self.c2:aa(c3_p/c)},
            f'err_{self.c1}_to_{self.c0}': aa(err_1_to_0),
            'acc_easy':aa(easy_acc),
            f'acc_{self.c0}':aa(acc_0),
            f'recall_{self.c0}':aa(recall_0),
            f'acc_{self.c1}':aa(acc_1),
            f'recall_{self.c1}':aa(recall_1),
            f'acc_{self.c2}':aa(acc_2),
            f'recall_{self.c2}':aa(recall_2),
        }

        self.reset()

        return self.result

    def __str__(self) -> str:
        if self.result is None:
            return 'None result'
        else:
            s = ''
            for k,v in self.total_result.items():
                s += f'{k:<22} {v}\n'
            return s



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