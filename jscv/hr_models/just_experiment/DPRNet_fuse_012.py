import random
from jscv.hr_models.pathes_segmentor import *

from jscv.utils.trainer import Evaluator
from jscv.utils.utils import TimeCounter
import cv2
import time as time_model

do_debug = False
count_1 = TimeCounter(do_debug)



class GL_FFN(nn.Module):
    def __init__(self, 
                 global_channels, 
                 local_channels,
                 blocks=2,
                 channel_ratio=1/2,
                 pose_decoder_cls=FPNDecoder,
                 decoder_args={},
                 ):
        '''
            global: [s16, s32]
            local: [s2, s4, s8, s16, s32]
        '''
        super().__init__()
        GB, GD = global_channels
        C0_g = GB[0]
        C1_g = GB[1]
        C2_g = GD[2]

        C0_l, C1_l, C2_l, C3_l, C4_l = local_channels
        self.blocks = blocks

        C2 = C2_l+C0_g
        outc_2 = int(C2*(1/3))
        C3 = C3_l+C1_g
        outc_3 = int(C3*(1/3))
        C4 = C4_l+C2_g
        outc_4 = int(C4*(1/2))


        if blocks > 0:
            self.layer0 = []
            for k in range(blocks):
                inc = C2 if k==0 else outc_2
                self.layer0.append(ConvBNReLU(inc, outc_2, 3))
            self.layer0 = ResBlocks(*self.layer0)


        if blocks > 0:
            self.layer1 = []
            for k in range(blocks):
                inc = C3 if k==0 else outc_3
                self.layer1.append(ConvBNReLU(inc, outc_3, 3))
            self.layer1 = ResBlocks(*self.layer1)

        if blocks > 0:
            self.layer2 = []
            for k in range(blocks):
                inc = C4 if k==0 else outc_4
                self.layer2.append(ConvBNReLU(inc, outc_4, 3))
            self.layer2 = ResBlocks(*self.layer2)


        self.pose_decoder=pose_decoder_cls(
            [C0_l, C1_l, outc_2, outc_3, outc_4],
            **decoder_args
        )

    def forward(self, g_fs, l_fs):
        f0, f1, f2, f3, f4 = l_fs
        g0, g1, g2 = g_fs
        # print("^^^1")
        # print(f2.shape, g0.shape)
        # print(f3.shape, g1.shape)
        # print(f4.shape, g2.shape, '\n')
        # print("^^^2")
        f2 = torch.concat([f2, g0], dim=1)
        f3 = torch.concat([f3, g1], dim=1)
        f4 = torch.concat([f4, g2], dim=1)

        if self.blocks > 0:
            f2 = self.layer0(f2)
            f3 = self.layer1(f3)
            f4 = self.layer2(f4)

        return self.pose_decoder(f0, f1, f2, f3, f4)




class DPRNet(nn.Module):
    def __init__(
            self,

            global_encoder,
            global_decoder,
            global_encoder_channels,
            local_encoder=None, 
            GL_ffn_decoder=None,
            global_seg_loss_layer: nn.Module = None,
            local_seg_loss_layer: nn.Module = None,

            global_seg_loss_weight=1,
            global_patches={'train':(2,2), 'val':(1,1)},
            local_patches=(8,8),
            local_batch_size={'train': 4, 'val': 8},
            global_downsample=1/4,
            local_downsample=1,

            ignore_index=None,
            pred_easy_bondary=0.1,
            pred_easy_bondary_training=-1,
            labal_err_bondary=2,    # <%
            # local_train_err_bondary=1,
            

            stage=1,  # 1:仅分割, 2:分割+分类，3:训练局部网络

            optimizer_step_interval=2,
            save_images_args=dict(per_n_step=15,)
        ):
        '''
            train_setting_list:  per_item: (patches, downsample, Probability)
            val_setting:(patches, downsample)
        '''
        super().__init__()
        assert global_downsample*4 == local_downsample
        self.global_encoder = global_encoder
        self.global_decoder = global_decoder
        self.local_encoder = local_encoder
        self.GL_ffn_decoder = GL_ffn_decoder
        self.global_seg_loss_layer = global_seg_loss_layer
        self.local_seg_loss_layer = local_seg_loss_layer
        self.global_patches = global_patches
        self.local_patches = local_patches
        self.global_downsample = global_downsample
        self.local_downsample = local_downsample
        self.global_seg_loss_weight = global_seg_loss_weight
        self.local_batch_size = local_batch_size

        self.global_encoder_channels = global_encoder_channels
        self.stage = stage
        self.pred_easy_bondary = pred_easy_bondary
        self.labal_err_bondary = labal_err_bondary
        if pred_easy_bondary_training < 0:
            pred_easy_bondary_training = pred_easy_bondary
        print('pred_easy_bondary', pred_easy_bondary, pred_easy_bondary_training)
        self.pred_easy_bondary_training = pred_easy_bondary_training
        
        
        self.global_decoder.return_d2_feature = False
        self.global_decoder.save_features = True

        if self.stage != 1:
            # self.easyhard_head = nn.Linear(global_encoder_channels[-1], 1)
            self.easyhard_head = nn.Sequential(
                nn.Linear(global_encoder_channels[-1], global_encoder_channels[-1]//2),
                nn.ReLU6(),
                nn.Linear(global_encoder_channels[-1]//2, 1),
            )
        self.ignore_index = ignore_index
        self.optimizer_step_interval = optimizer_step_interval
        self.sim = SaveImagesManager(**save_images_args)


    def forward(self, batch:dict):
        
        torch.cuda.empty_cache()

        result = self.forward_train(batch)

 
        if self.sim.step() and self.stage != 1:
            
            img, mask = batch['img'], batch['gt_semantic_seg']
            
            id = batch.get('id')
            if id is not None:
                id = id[0]

            hard_pred = result['easyhard_pred'] > self.pred_easy_bondary
            hard_label = result['err_rate'] > self.labal_err_bondary
            hard_pred = hard_pred.cpu()
            hard_label = hard_pred.cpu()
            img = img.squeeze(0)
            mask = mask.squeeze(0).cpu()
            if self.stage == 2:
                g_seg = result['pred'].cpu()
                self.sim.save_image_hard(img, g_seg.squeeze(0), mask, hard_pred, hard_label, id)
            else:
                g_seg = result['coarse_pred'].cpu()
                l_seg = result['pred'].cpu()
                self.sim.save_image_hard(img, l_seg.squeeze(0), mask, hard_pred, hard_label, id, g_seg.squeeze(0))
        return result


    def forward_train(self, batch:dict):
        img, mask = batch['img'], batch['gt_semantic_seg']
        s1 = 'train' if self.training else 'val'
        PH, PW = global_patches = self.global_patches[s1]

        
        g_outputs, l_outputs, result = [], [], {}
        seg_losses_g = 0
        seg_losses_l = 0
        easyhard_losses = 0
        img_patches, mask_patches = div_patches(img, mask, global_patches, 1)

        if self.stage != 1:
            easyhard_labels, easyhard_preds, err_rates = [], [], []
            PH2, PW2 = self.local_patches
            SH, SW = (PH2//PH), (PW2//PW)

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            B,C,H,W = imgij.shape
            g_input = resize_to(imgij, self.global_downsample)
            
            g_fs = self.global_encoder(g_input)
            g_seg = self.global_decoder(*g_fs)
            # print("^^^global_encoder")
            # for f in g_fs:
            #     print(g_fs.shape)

            g_seg = resize_to_y(g_seg, maskij)
            g_s128 = g_fs[-1]
            
            gd2 = self.global_decoder.features[1]
            self.global_decoder.features.clear()
            gb0, gb1 = g_fs[0], g_fs[1]
            # print("gd2", gd2.shape)

            del g_fs, self.global_decoder.features

            
            seg_loss_g = self.global_seg_loss_layer(g_seg, maskij) * self.global_seg_loss_weight
            # except RuntimeError as e:
            #     print('WARNING: out of memory')
            #     torch.cuda.empty_cache()
            #     time_model.sleep(5)
            #     while True:
            #         try:
            #             seg_loss_g = self.global_seg_loss_layer(g_seg, maskij) * self.global_seg_loss_weight
            #             break
            #         except RuntimeError as e:
            #             print('WARNING: out of memory')
            #             torch.cuda.empty_cache()
            #             time_model.sleep(5)

            seg_losses_g += seg_loss_g.detach()
            g_seg = g_seg.argmax(1)
            g_outputs.append(g_seg.detach())

            if self.stage == 1:
                if self.training:
                    seg_loss_g.backward()
                    optimize_step(self, 'optimizer_global', self.optimizer_step_interval)
            else:
                
                ''' easyhard '''
                H2, W2 = H//SH, W//SW
                easyhard_pred = F.adaptive_avg_pool2d(g_s128, (SH, SW)).permute(0, 2, 3, 1)
                easyhard_pred = self.easyhard_head(easyhard_pred).squeeze(-1).sigmoid()
                del g_s128
                
                imgij = imgij.reshape(B, C, SH, H2, SW, W2)
                maskij_reshape = maskij.reshape(B, SH, H2, SW, W2).transpose(2,3)
                g_seg = g_seg.reshape(B, SH, H2, SW, W2).transpose(2,3)
                easyhard_loss, easyhard_label, err_rate = self.easy_hard_loss(
                    easyhard_pred, g_seg, maskij_reshape)
                easyhard_losses += easyhard_loss.detach()
                easyhard_preds.append(easyhard_pred)
                easyhard_labels.append(easyhard_label)
                err_rates.append(err_rate)

                torch.cuda.empty_cache()
                #!!!
                if self.training:
                    (seg_loss_g+easyhard_loss).backward()
                    optimize_step(self, 'optimizer_global', 1)

                if self.stage == 3:
                    refined_seg_L = [None] * SH * SW
                    
                    B, C32, H32, W32 = gd2.shape
                    gd2 = gd2.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)
                    B, C32, H32, W32 = gb0.shape
                    gb0 = gb0.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)
                    B, C32, H32, W32 = gb1.shape
                    gb1 = gb1.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)
                    # B, SH, SW
                    # is_hard = easyhard_pred > self.pred_easy_bondary_training
                    is_hard = easyhard_pred.reshape(-1) > self.pred_easy_bondary
                    assert B == 1, '不支持'
                    hard_idx = []
                    hard_idx_1 = []
                    bz = self.local_batch_size[s1]
                    for j in range(SH*SW):
                        if is_hard[j]:
                            hard_idx_1.append(j)
                            if len(hard_idx_1) == bz:
                                hard_idx.append(hard_idx_1)
                                hard_idx_1 = []
                        else:
                            p, q = j//SW, j%SW
                            # print(j, SW, p, q, g_seg.shape)
                            refined_seg_L[j] = g_seg[:,p,q]
                    if len(hard_idx_1) > 0:
                        hard_idx.append(hard_idx_1)
    
                    for hard_idx_1 in hard_idx:
                        img_L, mask_L, g_s32_L = [], [], []
                        gb0_L, gb1_L = [], []
                        for j in hard_idx_1:
                            p, q = j//SW, j%SW
                            img_L.append(imgij[:,:,p,:,q])
                            mask_L.append(maskij_reshape[:,p,q])
                            g_s32_L.append(gd2[:,:,p,:,q])
                            gb0_L.append(gb0[:,:,p,:,q])
                            gb1_L.append(gb1[:,:,p,:,q])

                        img_L = torch.concat(img_L, dim=0)
                        mask_L = torch.concat(mask_L, dim=0)
                        g_s32_L = torch.concat(g_s32_L, dim=0)
                        gb0_L = torch.concat(gb0_L, dim=0)
                        gb1_L = torch.concat(gb1_L, dim=0)

                        l_fs = self.local_encoder(resize_to(img_L, self.local_downsample))
                        l_seg = self.GL_ffn_decoder((gb0_L, gb1_L, g_s32_L), l_fs)
                        del l_fs
                        l_seg = resize_to_y(l_seg, mask_L)
                        seg_loss_l = self.local_seg_loss_layer(l_seg, mask_L)
                        if self.training:
                            seg_loss_l.backward()
                            optimize_step(self, 'optimizer_local', self.optimizer_step_interval)
                        seg_losses_l += seg_loss_l.detach()
                        l_seg = torch.split(l_seg.argmax(1), 1, 0)
                        for j, l_seg_1 in zip(hard_idx_1, l_seg):
                            refined_seg_L[j] = l_seg_1
                    l_outputs.append(recover_mask(refined_seg_L, (SH,SW)))
                    
        g_seg = recover_mask(g_outputs, global_patches)
        if self.stage == 1:
            result = {'pred': g_seg, 'losses': {'main_loss':seg_losses_g}}
        elif self.stage == 2:
            result = {'pred': g_seg, 'losses': {'main_loss':seg_losses_g+easyhard_losses,
                                                'seg_loss': seg_losses_g,
                                                'easyhard_loss': easyhard_losses}}
        else:
            l_seg = recover_mask(l_outputs, global_patches)
            result = {'pred': l_seg, 'coarse_pred': g_seg, 
                      'losses': {'main_loss':seg_losses_g+easyhard_losses+seg_losses_l,
                                 'seg_loss_g': seg_losses_g,
                                 'seg_loss_l': seg_losses_l,
                                 'easyhard_loss': easyhard_losses,
                                 }
            }

        def recover_map(x):
            return torch.concat(x).reshape(PH, PW, SH, SW).transpose(1,2).reshape(PH2, PW2)

        if self.stage != 1:
            result['easyhard_pred'] = recover_map(easyhard_preds)
            result['easyhard_label'] = recover_map(easyhard_labels)
            result['err_rate'] = recover_map(err_rates)


        return result


    def predict(self, img, zero_img=False, easy_rate = 0.7):
        assert self.stage == 3
        assert zero_img

        l_outputs = []
        PH, PW = global_patches = self.global_patches['val']
        img_patches = div_img_patches(img, global_patches, 1)
        PH2, PW2 = self.local_patches
        SH, SW = (PH2//PH), (PW2//PW)
        
        for idx, imgij in enumerate(img_patches):
            B,C,H,W = imgij.shape
            g_input = resize_to(imgij, self.global_downsample)
            
            g_fs = self.global_encoder(g_input)
            g_seg = self.global_decoder(*g_fs)
            # g_s128 = g_fs[-1]

            # for f in self.global_decoder.features:
            #     print(f.shape)
            gd2 = self.global_decoder.features[1]
            self.global_decoder.features.clear()
            gb0, gb1 = g_fs[0], g_fs[1]

            del g_fs
            g_seg = resize_to_y(g_seg, imgij)

            H2, W2 = H//SH, W//SW
            imgij = imgij.reshape(B, C, SH, H2, SW, W2)
            g_seg = g_seg.argmax(1).reshape(B, SH, H2, SW, W2).transpose(2,3)


            B, C32, H32, W32 = gd2.shape
            gd2 = gd2.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)
            B, C32, H32, W32 = gb0.shape
            gb0 = gb0.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)
            B, C32, H32, W32 = gb1.shape
            gb1 = gb1.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)


            refined_seg_L = [None] * SH * SW
            # 生成 is_hard
            is_hard = torch.zeros(SH*SW)
            is_hard[int(easy_rate*(SH*SW)):] = 1
            hard_idx = []
            hard_idx_1 = []
            bz = self.local_batch_size['val']
            for j in range(SH*SW):
                if is_hard[j]:
                    hard_idx_1.append(j)
                    if len(hard_idx_1) == bz:
                        hard_idx.append(hard_idx_1)
                        hard_idx_1 = []
                else:
                    p, q = j//SW, j%SW
                    refined_seg_L[j] = g_seg[:,p,q]
            if len(hard_idx_1) > 0:
                hard_idx.append(hard_idx_1)
            for hard_idx_1 in hard_idx:
                img_L, g_s32_L = [], []
                gb0_L, gb1_L = [], []
                for j in hard_idx_1:
                    p, q = j//SH, j%SH
                    img_L.append(imgij[:,:,p,:,q])
                    g_s32_L.append(gd2[:,:,p,:,q])
                    gb0_L.append(gb0[:,:,p,:,q])
                    gb1_L.append(gb1[:,:,p,:,q])


                img_L = torch.concat(img_L, dim=0)
                g_s32_L = torch.concat(g_s32_L, dim=0)
                gb0_L = torch.concat(gb0_L, dim=0)
                gb1_L = torch.concat(gb1_L, dim=0)


                l_fs = self.local_encoder(resize_to(img_L, self.local_downsample))
                l_seg = self.GL_ffn_decoder((gb0_L, gb1_L, g_s32_L), l_fs)
                l_seg = resize_to_y(l_seg, img_L)
                del l_fs
                l_seg = torch.split(l_seg.argmax(1), 1, 0)
                for j, l_seg_1 in zip(hard_idx_1, l_seg):
                    refined_seg_L[j] = l_seg_1
            l_outputs.append(recover_mask(refined_seg_L, (SH,SW)))
        l_seg = recover_mask(l_outputs, global_patches)
        return l_seg


    def easy_hard_loss(self, x, pred, mask):
        B, SH, SW, H, W = mask.shape
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
        easyhard_loss = F.binary_cross_entropy(x, label)
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
            r3 = hard_err / (self.count-N)
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



def test_predict_speed_pure():
    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    from jscv.hr_models.deeplabv3 import ASPP

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 20
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True

    fpn_decoder_args = fpn_decoder_args_1
    local_fpn_decoder_args = fpn_decoder_args_1
    GB = ResNetEncoder('resnet101', features_only=True)

    # GB = StrongerEncoder(GB, ASPP)
    GD = FPNDecoder(GB.channels, return_d2_feature=True, **fpn_decoder_args)
    LB = ResNetEncoder('resnet18', features_only=True)
    print(GD.channels)
    LD = GL_FFN((GB.channels, GD.channels), LB.channels, 2, 1/2, FPNDecoder, local_fpn_decoder_args)

    model = DPRNet(GB, GD, GB.channels, LB, LD, None, None, 1,
                      global_patches={'val':(1,1)},
                      local_patches=(16,16),
                      local_batch_size={'val': 88},
                      ignore_index=100,
                      stage=3,
                      )
    easy_rate = 0.75

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