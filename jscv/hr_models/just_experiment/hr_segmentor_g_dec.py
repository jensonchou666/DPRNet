'''
#!!!
TODO:
尝试：

global-decoder不裁剪特征图,直接

不使用global-decoder、仅分类
不融合global特征

TODO 研究-全局网络, 3网络

TODO 

TODO 划分数据集

'''

import random
from jscv.hr_models.base_models import *

from jscv.utils.trainer import Evaluator, SegmentEvaluator
from jscv.utils.overall import global_dict


do_debug_time = False
LOG_SCORE_EVERY = -1 #print score-acc if >0

class HRSegmentor(nn.Module):
    """
        高分辨率 语义分割器
        mask 必须是0、1 二分类
    """

    def __init__(
            self,

            local_patches=(8,8),     #[列，行]   确定能整除
            global_patches=(8,8),       #[列，行]   <=local_patches
            global_downsample=1,
            local_backbone_name="resnet50",
            global_backbone_name="resnet18",
            global_only=False,

            seg_loss_layer: nn.Module = None,
            g_seg_loss_layer: nn.Module = None,

            score_threshold=0.5,#需要先统计
            empty_map_threshold=0.1, #0~10
            
            decoder_class=FPNDecoder,
            g_decoder_args=dict(dec_channels=[18, 48, 96, 192], blocks=[1, 1, 1, 1], num_classes=2),
            l_decoder_args=dict(dec_channels=[32, 64, 256, 512], blocks=[2, 2, 2, 2], num_classes=2),

            do_fuse_global_features=True,
            patch_score_loss_weight=1,

            global_pretrain_path=None,
            optimizer_step_interval=10,
            # to_cuda=True,  #默认
            keys_name=['img', 'gt_semantic_seg'],
            
            save_images_per=5, # save images if > 0
            ):
        super().__init__()


        self.local_patches = local_patches
        self.local_backbone_name = local_backbone_name
        self.global_backbone_name = global_backbone_name
        self.empty_map_threshold = empty_map_threshold
        self.score_threshold = score_threshold
        self.global_only = global_only
        self.global_patches = global_patches
        self.global_downsample = global_downsample
        self.patch_score_loss_weight = patch_score_loss_weight
        self.seg_loss_layer = seg_loss_layer
        self.frozen_score_net_epoachs = 0
        self.do_fuse_global_features = do_fuse_global_features
        self.keys_name = keys_name
        self.optimizer_step_interval = optimizer_step_interval
        
        if g_seg_loss_layer is None:
            g_seg_loss_layer = seg_loss_layer
        self.g_seg_loss_layer = g_seg_loss_layer

        if global_patches == local_patches:
            self.global_output_size = 1
        else:
            self.global_output_size = (local_patches[0]//global_patches[0],
                                       local_patches[1]//global_patches[1])

        self.global_backbone = ResNetEncoder(global_backbone_name, 
                                           features_only=False, 
                                           avg_pool_to=self.global_output_size,
                                           num_classes=1)

        if not global_only:
            self.local_backbone = ResNetEncoder(local_backbone_name, features_only=True)
            self.global_decoder = decoder_class(self.global_backbone.channels, **g_decoder_args)
            if do_fuse_global_features:
                l_chs = [a+b for a,b in zip(self.global_backbone.channels, self.local_backbone.channels)]
            else:
                l_chs = self.local_backbone.channels
            self.local_decoder = decoder_class(l_chs, **l_decoder_args)

        if global_pretrain_path is not None:
            self.pretrain(global_backbone=torch.load(global_pretrain_path))


        self.frozened = False
        self.epoach_idx = 0
        self.counter_easy = 0
        self.counter_hard = 0
        self.count_local_opt_i = 0
        self.sim = SaveImagesManager(per_n_step=save_images_per)
        

        self.time0 = time()


    def patch_score_loss(self, logits_x:torch.Tensor, mask:torch.Tensor):
        B, H, W = mask.shape
        s1 = mask.sum(-1).sum(-1)
        r1 = s1 / (H*W) * 100
        B1 = self.empty_map_threshold < r1
        B2 = r1 < 100-self.empty_map_threshold
        patch_label = B1 & B2
        loss = F.binary_cross_entropy(logits_x, patch_label.float())
        return loss, patch_label.int()

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


    def fuse_features(self, features_1, features_2):
        fs = []
        for f1, f2 in zip(features_1, features_2):
            if f1.shape[2:] != f2.shape[2:]:
                f2 = F.interpolate(f2, f1.shape[2:], mode="bilinear", align_corners=False)
            fs.append(torch.concat([f1,f2], dim=1))
        return fs

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
                if epoch_idx == self.frozen_score_net_epoachs:
                    for name, param in self.global_backbone.named_parameters():
                        param.requires_grad = True
                    print("frozen over")
                    self.frozened = False


        img, mask = batch[self.keys_name[0]], batch[self.keys_name[1]]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        
        assert img.shape[0] == 1, "暂时，方便代码"
        
        result = {}


        patch_score_logits, patch_labels = [], []
        main_losses, seg_losses, patch_score_losses = 0, 0, 0
        img_patches, mask_patches = self.div_patches(img, mask, self.global_patches)

        outputs, hard_list, label_hard_list = [], [], []

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            # i, j = idx//self.global_patches[0], idx%self.global_patches[0]
            imgij, maskij = imgij.cuda(), maskij.cuda()
            if self.global_downsample != 1:
                imgij_d = F.interpolate(imgij, scale_factor=self.global_downsample, mode="bilinear")
            x, (x4, x3, x2, x1, x0) = self.global_backbone(imgij_d)
            del imgij_d
            logits_x = torch.sigmoid(x).squeeze(-1)
            
            def _resize_(xx, _mask=imgij):
                if xx.shape[-2:] != _mask.shape[-2:]:
                    xx = F.interpolate(xx, _mask.shape[-2:], mode="bilinear")
                return xx
            
            if self.global_output_size == 1:
                assert False, 'TODO 暂未实现'
                patch_score_loss, patch_label = self.patch_score_loss(logits_x, maskij)
                patch_score_loss *= self.patch_score_loss_weight
                
                patch_score_logits.append(logits_x)
                patch_labels.append(patch_label.int())
                if self.global_only:
                    if self.training:  patch_score_loss.backward()
                else:
                    if logits_x < self.score_threshold:
                        # simple
                        x = self.global_decoder(x4, x3, x2, x1, x0)
                    else:
                        l_x4, l_x3, l_x2, l_x1, l_x0 = self.local_backbone(imgij)
                        if self.do_fuse_global_features:
                            x = self.local_decoder(*(self.fuse_features([l_x4, l_x3, l_x2, l_x1, l_x0],
                                                                        [x4, x3, x2, x1, x0])))
                        else:
                            x = self.local_decoder(l_x4, l_x3, l_x2, l_x1, l_x0)
                    x = _resize_(x)
                    seg_loss = self.seg_loss_layer(x, maskij)
                    
                    if self.training:
                        if self.frozened:
                            seg_loss.backward()
                        else:
                            (seg_loss+patch_score_loss).backward()
            else:
                patch_score_loss = 0
                # B, SH, SW
                SH, SW = self.global_output_size
                B, C, H, W = imgij.shape
                assert B == 1
                maskij = maskij.reshape(B, SH, H//SH, SW, W//SW)
                is_hard, label_is_hard = [], []
                for k in range(SH):
                    for p in range(SW):
                        logits_x_kp = logits_x[:, k, p]
                        maskij_kp = maskij[:, k, :, p]
                        patch_score_losskp, patch_labelkp = self.patch_score_loss(logits_x_kp, maskij_kp)
                        patch_score_loss += patch_score_losskp
                        patch_score_logits.append(logits_x_kp)
                        patch_labels.append(patch_labelkp)
                        label_is_hard.append(int(patch_labelkp))
                        if logits_x_kp < self.score_threshold:
                            is_hard.append(0)
                        else:
                            is_hard.append(1)
                hard_list.append(is_hard)
                label_hard_list.append(label_is_hard)
                if self.global_only:
                    if self.training:  patch_score_loss.backward()
                else:
                    if sum(is_hard) == 0: # 所有均为easy
                        x = self.global_decoder(x4, x3, x2, x1, x0)
                        maskij = maskij.reshape(B,H,W)
                        x = _resize_(x,maskij)
                        seg_loss = self.g_seg_loss_layer(x, maskij)
                        if self.training:
                            if self.frozened: seg_loss.backward()
                            else: (seg_loss+patch_score_loss).backward()
                    else:
                        of, seg_loss = [], 0
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
                        easy_seg_loss = 0
                        if len(easyPs_idx) > 0:
                            easyPx_4,easyPx_3,easyPx_2,easyPx_1,easyPx_0 = [],[],[],[],[]
                            masks = []
                            for s in easyPs_idx:
                                k, p = s//SH, s%SH
                                easyPx_4.append(x4[:,:,k,:,p])
                                easyPx_3.append(x3[:,:,k,:,p])
                                easyPx_2.append(x2[:,:,k,:,p])
                                easyPx_1.append(x1[:,:,k,:,p])
                                easyPx_0.append(x0[:,:,k,:,p])
                                masks.append(maskij[:, k,:,p])
                            easyPx_4 = torch.concat(easyPx_4, dim=0)
                            easyPx_3 = torch.concat(easyPx_3, dim=0)
                            easyPx_2 = torch.concat(easyPx_2, dim=0)
                            easyPx_1 = torch.concat(easyPx_1, dim=0)
                            easyPx_0 = torch.concat(easyPx_0, dim=0)
                            masks = torch.concat(masks, dim=0)
                            easy_x = self.global_decoder(easyPx_4, easyPx_3, easyPx_2, easyPx_1, easyPx_0)
                            # print(easy_x.shape, masks.shape)
                            easy_x = _resize_(easy_x, masks)
                            # print(easy_x.shape, masks.shape, easy_x.dtype, masks.dtype)
                            easy_seg_loss = self.g_seg_loss_layer(easy_x, masks)
                        seg_loss += easy_seg_loss
                        hard_x = []
                        imgij = imgij.reshape(B, C, SH, H//SH, SW, W//SW)
                        # rand_n = random.randint(0, len(hardPs_idx)-1)
                        if self.training:
                            if self.frozened and easy_seg_loss!=0: easy_seg_loss.backward()
                            else: (easy_seg_loss+patch_score_loss).backward()
                        for s in hardPs_idx:
                            k, p = s//SH, s%SH
                            imgij_kp = imgij[:,:,k,:,p]
                            maskij_kp = maskij[:, k,:,p]
                            # x4A, x3A, x2A, x1A, x0A = x4, x3, x2, x1, x0
                            # if 1:
                            #     #??? 防止多次对 global_backbone  backward
                            #     x4A, x3A, x2A, x1A, x0A = x4A.detach(), x3A.detach(), x2A.detach(),\
                            #         x1A.detach(), x0A.detach()
                            l_x4, l_x3, l_x2, l_x1, l_x0 = self.local_backbone(imgij_kp)
                            if self.do_fuse_global_features:
                                fs = self.fuse_features((l_x4, l_x3, l_x2, l_x1, l_x0), (
                                    x4.detach()[:,:,k,:,p],
                                    x3.detach()[:,:,k,:,p],
                                    x2.detach()[:,:,k,:,p],
                                    x1.detach()[:,:,k,:,p],
                                    x0.detach()[:,:,k,:,p],
                                ))
                            else:
                                fs = (l_x4, l_x3, l_x2, l_x1, l_x0)
                            hard_x_kp = self.local_decoder(*fs)
                            hard_x_kp = _resize_(hard_x_kp, maskij_kp)
                            hard_x.append(hard_x_kp)
                            hard_seg_loss = self.seg_loss_layer(hard_x_kp, maskij_kp)/len(hardPs_idx)
                            seg_loss += hard_seg_loss.detach()
                            if self.training:
                                # if s == rand_n:
                                #     (hard_seg_loss+).backward()
                                # else:
                                hard_seg_loss.backward()
                                
                                self.count_local_opt_i += 1
                                if self.count_local_opt_i >= self.optimizer_step_interval:
                                    self.count_local_opt_i = 0
                                    if hasattr(self, 'optimizer_lb'):
                                        self.optimizer_lb.step()
                                        self.optimizer_lb.zero_grad()
                                    if hasattr(self, 'optimizer_ld'):
                                        self.optimizer_ld.step()
                                        self.optimizer_ld.zero_grad()

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

            if self.global_only:
                loss = patch_score_loss
            else:
                loss = patch_score_loss + seg_loss
                x = x.detach()
                outputs.append(x)
                seg_losses = seg_losses + seg_loss
            
            if self.training:
                if hasattr(self, 'optimizer_gb'):
                    self.optimizer_gb.step()
                    self.optimizer_gb.zero_grad()
                if hasattr(self, 'optimizer_gd'):
                    self.optimizer_gd.step()
                    self.optimizer_gd.zero_grad()

            main_losses += loss.detach()
            patch_score_losses += patch_score_loss.detach()


        result.update({'patch_score_logits': patch_score_logits, 'patch_labels': patch_labels})

        result["losses"] = {'main_loss': main_losses, 'patch_score_loss':patch_score_losses}
        if not self.global_only:
            result["losses"]['seg_loss'] = seg_losses

            outputs = torch.concat(outputs, dim=0)
            B,C,H,W = outputs.shape
            GPH, GPW = self.global_patches
            outputs = outputs.reshape(1, GPH, GPW, C, H, W).permute(0, 3, 1, 4, 2, 5).reshape(1,C,H*GPH, W*GPW)
            result["pred"] = outputs

        return result, (outputs, mask, hard_list, label_hard_list)


    def predict(self, img: torch.Tensor):
        
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
                patch_score_loss = 0
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


        


    def pretrain(self, local_backbone=None, global_backbone=None, global_convert_fc=True):
        if local_backbone is not None:
            del local_backbone['fc.weight']
            del local_backbone['fc.bias']
            self.local_backbone.load_state_dict(local_backbone)
            print("load local-backbone ok")
        
        if global_backbone is not None:
            G_state_dict = self.global_backbone.state_dict()
            if global_convert_fc:
                global_backbone['fc.weight'] = G_state_dict['fc.weight']
                global_backbone['fc.bias'] = G_state_dict['fc.bias']
            self.global_backbone.load_state_dict(global_backbone)
            print("load global-backbone ok")

    def frozen_score_net(self, epoach):
        self.frozen_score_net_epoachs = epoach
        if epoach > 0:
            for name, param in self.global_backbone.named_parameters():
                param.requires_grad = False
            self.frozened = True
            print("frozened global_backbone")


class PatchScoreEvaluator(Evaluator):
    def __init__(self, stat_items=[0.30 ,0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7]):
        super().__init__()
        self.stat_items = stat_items
        self.statD = []
        for j in stat_items:
            self.statD.append([j, 0, 0, 0])
        self.count = 0
        self.result = None
        self.c1 = 0

    def append(self, result):
        patch_score_logits = result['patch_score_logits']
        patch_labels = result['patch_labels']
        patch_score_logits = torch.concat(patch_score_logits, 0)
        patch_labels = torch.concat(patch_labels, 0)
        
        C = patch_score_logits.shape[0]
        self.count += C
        statD2 = []
        for i, (j, N, N_1, N_2) in enumerate(self.statD):
            idx = torch.nonzero(patch_score_logits < j)
            idx2 = torch.nonzero(patch_score_logits > j)
            
            _N = idx.numel()    # 预测为0的集合数量
            _N_1 = torch.sum(patch_labels[idx])   # 预测为0的集合里有 多少 label为1
            _N_2 = torch.sum(patch_labels[idx2])   # 预测为1的集合里有 多少 label为1
            N += _N
            N_1 += _N_1
            N_2 += _N_2
            statD2.append([j, N, N_1, N_2])
            if LOG_SCORE_EVERY>0 and j == 0.5 and self.c1 % LOG_SCORE_EVERY == 0:#TODO
                print(f"{_N}/{C}, pred0-acc: {_N-_N_1}/{_N}, pred1-acc: {_N_2}/{C-_N}")
        self.c1 += 1

        self.statD = statD2
        
    def reset(self):
        self.count = 0
        self.statD = []
        for j in self.stat_items:
            self.statD.append([j, 0, 0, 0])

    def evaluate(self):
        self.result = {}
        self.result_L = []
        for i, (j, N, N_1, N_2) in enumerate(self.statD):
            pred0_ratio = N/self.count
            pred1 = self.count - N
            
            # if N == 0:
            #     pred0_ACC = -1.
            # else:
            pred0_ACC = (N-N_1)/N

            # if pred1 == 0:
            #     pred1_ACC = -1.
            # else:
            pred1_ACC = N_2/pred1
            r = {
                f'R_{int(j*100)}': pred0_ratio,
                f'A0_{int(j*100)}': pred0_ACC,
                f'A1_{int(j*100)}': pred1_ACC,
            }
            self.result.update(r)
            self.result_L.append(r)

        self.reset()

        return self.result

    def __str__(self) -> str:
        if self.result is None:
            return 'None result'
        else:
            s = ''
            for r in self.result_L:
                s += str(r) + '\n'
            return s



class HRSeg_Evaluator(Evaluator):

    def from_config(cfg):
        ps_eval = PatchScoreEvaluator()
        seg_eval = SegmentEvaluator.from_config(cfg)
        return HRSeg_Evaluator(ps_eval, seg_eval)

    def __init__(self, ps_eval: PatchScoreEvaluator, seg_eval: SegmentEvaluator):
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

def test_score_map_effiction(use_evaluator=True):
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
    model.frozen_score_net(1)

    ds = water_seg_dataset()
    # model.optimizer_gb, sch1 = get_optimizer(model.global_backbone)
    # model.optimizer_gd, sch2 = get_optimizer(model.global_decoder)
    # model.optimizer_lb, sch3 = get_optimizer(model.local_backbone)
    # model.optimizer_ld, sch4 = get_optimizer(model.local_decoder)
    if use_evaluator:
        evalu = PatchScoreEvaluator()
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
        
        print('patch_score_loss', result['losses']['patch_score_loss'])

        if use_evaluator:
            evalu.append(result)
            tq.update(1)
        else:
            patch_score_logits = result['patch_score_logits']
            patch_labels = result['patch_labels']
            patch_score_logits = torch.concat(patch_score_logits, 0)
            patch_labels = torch.concat(patch_labels, 0)
            count += patch_score_logits.shape[0]

            for i, threshold in enumerate(thresholds):
                idx = torch.nonzero(patch_score_logits < threshold)
                pred0 = idx.numel()
                count_pred0[i] += pred0
                pred0_right = pred0 - torch.sum(patch_labels[idx]) #预测0合集里， label为0的元素数量
                count_pred0_right[i] += pred0_right
            
                idx = torch.nonzero(patch_score_logits > threshold)
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




def test_predict_speed():


    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    torch.set_grad_enabled(False)
    
    warmup()
    
    model = HRSegmentor(
        local_backbone_name='resnet101',
        global_backbone_name='resnet18',
        global_patches=(2,2),
        global_downsample=1/2,
        l_decoder_args=dict(dec_channels=[32, 64, 256, 512], blocks=[2, 2, 2, 2], num_classes=2),
        score_threshold=0.4,
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
        pred = model.predict(img)
        
        tq.update(1)
        # print(fi, pred.shape, label.shape)

    print("finished, spend:", time()-t0)
    print("counter_hard:", model.counter_hard, ", counter_easy:", model.counter_easy, 
          ", rate:", model.counter_hard / (model.counter_hard+model.counter_easy))

    '''eval'''





if __name__ == "__main__":


    test_predict_speed()




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