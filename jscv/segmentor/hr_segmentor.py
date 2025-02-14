
from jscv.utils.trainer import Evaluator
from torch import nn
do_debug = False
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torch.utils
from time import time

from torchvision.models.resnet import ResNet, Bottleneck,BasicBlock
from jscv.models.cnn import *


class ResNetEncoder(ResNet):
    """
    ResNetEncoder inherits from torchvision's official ResNet. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    global average pooling layer and the fully connected layer that was originally
    used for classification. The forward method  additionally returns the feature
    maps at all resolutions for decoder's use.
    """
    layers_dict = {
        'resnet18':  [2, 2, 2, 2],
        'resnet50':  [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3],
    }
    channels_dict = {
        'resnet18':  [64, 64, 128, 256, 512],
        'resnet50':  [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
        'resnet152': [64, 256, 512, 1024, 2048],
    }

    def __init__(self, variant='resnet50',
                 norm_layer=None,
                 features_only=False,
                 avg_pool_to=1,
                 do_maxpool=True,
                 num_classes=1000):
        if variant in ["resnet18"]:
            block=BasicBlock
        else:
            block=Bottleneck
        super().__init__(
            block=block,
            layers=self.layers_dict[variant],
            num_classes=num_classes,
            # replace_stride_with_dilation=[False, False, False],
            norm_layer=norm_layer)
        self.features_only = features_only
        self.avg_pool_to = avg_pool_to
        self.layers = self.layers_dict[variant]
        self.channels = self.channels_dict[variant]
        if features_only:
            del self.avgpool
            del self.fc
        elif avg_pool_to != 1:
            self.avgpool = nn.AdaptiveAvgPool2d(avg_pool_to)

        self.do_maxpool = do_maxpool
        if not do_maxpool:
            del self.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # 1/2

        if self.do_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x1 = x  # 1/4
        x = self.layer2(x)
        x2 = x  # 1/8
        x = self.layer3(x)
        x3 = x  # 1/16
        x = self.layer4(x)
        x4 = x  # 1/32
        
        if self.features_only:
            return x4, x3, x2, x1, x0
        else:
            if self.avg_pool_to != 1:
                x = self.avgpool(x).permute(0, 2, 3, 1)
                x = self.fc(x)
            else:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
            return x, [x4, x3, x2, x1, x0]


class ResBlocks(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.blocks = nn.ModuleList(models)
    
    def forward(self, x):
        for m in self.blocks:
            x1 = m(x)
            if x1.shape == x.shape:
                x = x + x1
            else:
                x = x1
        return x


class FPNDecoder(nn.Module):

    def __init__(self, enc_channels,
                 dec_channels=1/2,
                 blocks=[2, 2, 2, 2],
                 num_classes=2,
                 ):
        super().__init__()
        C0, C1, C2, C3, C4 = enc_channels
        if isinstance(dec_channels, list) or isinstance(dec_channels, tuple):
            DC1, DC2, DC3, DC4 = dec_channels
        else:
            DC = dec_channels
            DC1, DC2, DC3, DC4 = int(C1*DC), int(C2*DC), int(C3*DC), int(C4*DC)
        B1, B2, B3, B4 = blocks

        self.layer4 = []
        for k in range(B4):
            inc = C3+C4 if k == 0 else DC4
            self.layer4.append(ConvBNReLU(inc, DC4, 3))
        self.layer4 = ResBlocks(*self.layer4)

        self.layer3 = []
        for k in range(B3):
            inc = DC4+C2 if k == 0 else DC3
            self.layer3.append(ConvBNReLU(inc, DC3, 3))
        self.layer3 = ResBlocks(*self.layer3)

        self.layer2 = []
        for k in range(B2):
            inc = DC3+C1 if k == 0 else DC2
            self.layer2.append(ConvBNReLU(inc, DC2, 3))
        self.layer2 = ResBlocks(*self.layer2)

        self.layer1 = []
        for k in range(B1):
            inc = DC2+C0 if k == 0 else DC1
            self.layer1.append(ConvBNReLU(inc, DC1, 3))
        self.layer1 = ResBlocks(*self.layer1)

        self.segment_head = ConvBNReLU(DC1, num_classes, 3)



    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.layer4(x)

        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.layer3(x)

        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.layer2(x)
        
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.layer1(x)
        return self.segment_head(x)


class ResBlocks2(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.blocks = nn.ModuleList(models)
    
    def forward(self, x):
        for m in self.blocks:
            x = m(x) + x
        return x

class FPNDecoder_2(nn.Module):

    def __init__(self, enc_channels,
                 dec_channels=1/2,
                 blocks=[2, 2, 2, 2],
                 num_classes=2,
                 ):
        super().__init__()
        C0, C1, C2, C3, C4 = enc_channels

        if isinstance(dec_channels, list) or isinstance(dec_channels, tuple):
            DC1, DC2, DC3, DC4 = dec_channels
        else:
            DC = dec_channels
            DC1, DC2, DC3, DC4 = int(C1*DC), int(C2*DC), int(C3*DC), int(C4*DC)
        
        B1, B2, B3, B4 = blocks

        self.f4 = ConvBNReLU(C4, DC4, 3)
        self.f3 = ConvBNReLU(DC4, DC3, 3)
        self.f2 = ConvBNReLU(DC3, DC2, 3)
        self.f1 = ConvBNReLU(DC2, DC1, 3)
        self.n3 = ConvBNReLU(C3, DC4, 1)
        self.n2 = ConvBNReLU(C2, DC3, 1)
        self.n1 = ConvBNReLU(C1, DC2, 1)
        self.n0 = ConvBNReLU(C0, DC1, 1)

        self.layer4 = []
        for k in range(B4):
            self.layer4.append(ConvBNReLU(DC4, DC4, 3))
        self.layer4 = ResBlocks2(*self.layer4)

        self.layer3 = []
        for k in range(B3):
            self.layer3.append(ConvBNReLU(DC3, DC3, 3))
        self.layer3 = ResBlocks2(*self.layer3)

        self.layer2 = []
        for k in range(B2):
            self.layer2.append(ConvBNReLU(DC2, DC2, 3))
        self.layer2 = ResBlocks2(*self.layer2)

        self.layer1 = []
        for k in range(B1):
            self.layer1.append(ConvBNReLU(DC1, DC1, 3))
        self.layer1 = ResBlocks2(*self.layer1)
        self.segment_head = ConvBNReLU(DC1, num_classes, 3)

    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(self.f4(x4), size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer4(x+self.n3(x3))

        x = F.interpolate(self.f3(x), size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer3(x+self.n2(x2))

        x = F.interpolate(self.f2(x), size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer2(x+self.n1(x1))
        
        x = F.interpolate(self.f1(x), size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = self.layer1(x+self.n0(x0))
        return self.segment_head(x)




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

            empty_map_threshold=0.5, #0~10
            score_threshold=0.5,#需要先统计
            
            backbone_pretrain_path=None,
            global_pretrain_path=None,
            # to_cuda=True,  #默认
            keys_name=['img', 'gt_semantic_seg'],
            ):
        super().__init__()

        self.keys_name = keys_name

        self.local_patches = local_patches
        self.local_backbone_name = local_backbone_name
        self.global_backbone_name = global_backbone_name
        self.empty_map_threshold = empty_map_threshold
        self.score_threshold = score_threshold
        self.global_only = global_only
        self.global_patches = global_patches
        self.global_downsample = global_downsample


        if global_patches == local_patches:
            self.global_output_size = 1
        else:
            self.global_output_size = (local_patches[0]//global_patches[0],
                                       local_patches[1]//global_patches[1])

        self.global_branch = ResNetEncoder(global_backbone_name, 
                                           features_only=False, 
                                           avg_pool_to=self.global_output_size,
                                           num_classes=1)

        if not global_only:
            self.local_branch = ResNetEncoder(local_backbone_name, features_only=True)
            self.global_decoder = FPNDecoder(self.global_branch.channels, )

        if global_pretrain_path is not None:
            self.pretrain(global_backbone=torch.load(global_pretrain_path))

        self.time0 = time()


    def patch_score_loss(self, logits_x:torch.Tensor, mask:torch.Tensor):
        B, H, W = mask.shape
        s1 = mask.sum(-1).sum(-1)
        r1 = s1 / (H*W) * 100
        B1 = self.empty_map_threshold < r1
        B2 = r1 < 100-self.empty_map_threshold
        patch_label = B1 & B2
        loss = F.binary_cross_entropy(logits_x, patch_label.float())
        return loss, patch_label

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

    def forward(self, batch):

        t0 = time() - self.time0
        self.time0 = time()
        

        img, mask = batch[self.keys_name[0]], batch[self.keys_name[1]]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        
        assert img.shape[0] == 1, "暂时，方便代码"
        
        result = {}
        

        ''' Global '''
        patch_score_logits, patch_labels, patch_score_losses = [], [], 0
        img_patches, mask_patches = self.div_patches(img, mask, self.gloabl_patches)

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            i, j = idx//self.gloabl_patches[0], idx%self.gloabl_patches[0]
            imgij, maskij = imgij.cuda(), maskij.cuda()
            if self.gloabl_downsample != 1:
                imgij = F.interpolate(imgij, scale_factor=self.gloabl_downsample, mode="bilinear")
            x, (x4, x3, x2, x1, x0) = self.global_branch(imgij)
            logits_x = torch.sigmoid(x).squeeze(-1)
            if self.gloabl_output_size == 1:
                patch_score_loss, patch_label = self.patch_score_loss(logits_x, maskij)
                patch_score_logits.append(logits_x)
                patch_labels.append(patch_label.int())
                if not self.global_only:
                    if logits_x < self.score_threshold:
                        # simple
                        pass
                        
            else:
                patch_score_loss = 0
                # B, SH, SW
                SH, SW = self.gloabl_output_size
                B, H, W = maskij.shape
                maskij = maskij.reshape(B, SH, H//SH, SW, W//SW)
                for k in range(SH):
                    for p in range(SW):
                        logits_x_kp = logits_x[:, k, p]
                        maskij_kp = maskij[:, k, :, p]
                        patch_score_losskp, patch_labelkp = self.patch_score_loss(logits_x_kp, maskij_kp)
                        patch_score_loss += patch_score_losskp
                        patch_score_logits.append(logits_x_kp)
                        patch_labels.append(patch_labelkp.int())

            if self.training:
                # print("@ backward", )
                patch_score_loss.backward()
            patch_score_losses += patch_score_loss.detach()


                
        result.update({'patch_score_logits': patch_score_logits, 'patch_labels': patch_labels})
        result["losses"] = {'main_loss': patch_score_losses, 'patch_score_loss':patch_score_losses}


        t1 = time() - self.time0
        self.time0 = time()
        
        if do_debug:
            print("time: ", t0, t1)

        return result


    def pretrain(self, local_backbone=None, global_backbone=None, global_convert_fc=True):
        if local_backbone is not None:
            sd1 = self.local_branch.state_dict()
            del local_backbone['fc.weight']
            del local_backbone['fc.bias']
            self.local_branch.load_state_dict(local_backbone)
            print("load local-backbone ok")
        
        if global_backbone is not None:
            G_state_dict = self.global_branch.state_dict()
            if global_convert_fc:
                global_backbone['fc.weight'] = G_state_dict['fc.weight']
                global_backbone['fc.bias'] = G_state_dict['fc.bias']
            self.global_branch.load_state_dict(global_backbone)
            print("load global-backbone ok")


class PatchScoreEvaluator(Evaluator):
    def __init__(self, stat_items=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ,0.35, 0.40, 0.45, 0.50, 0.55, 0.60]):
        super().__init__()
        self.stat_items = stat_items
        self.statD = []
        for j in stat_items:
            self.statD.append([j, 0, 0, 0])
        self.count = 0
        self.result = None

    def append(self, result):
        patch_score_logits = result['patch_score_logits']
        patch_labels = result['patch_labels']
        patch_score_logits = torch.concat(patch_score_logits, 0)
        patch_labels = torch.concat(patch_labels, 0)
        
        
        self.count += patch_score_logits.shape[0]
        statD2 = []
        for i, (j, N, N_1, N_2) in enumerate(self.statD):
            idx = torch.nonzero(patch_score_logits < j)
            N += idx.numel()
            N_1 += torch.sum(patch_labels[idx])   # 预测为0的集合里有 多少 label为1
            
            idx2 = torch.nonzero(patch_score_logits > j)
            N_2 += torch.sum(patch_labels[idx2])   # 预测为1的集合里有 多少 label为1
            # print(j, N, N_1, N_2)
            statD2.append([j, N, N_1, N_2])
        self.statD = statD2
        


    def evaluate(self):
        self.result = {}
        self.result_L = []
        for i, (j, N, N_1, N_2) in enumerate(self.statD):
            pred0_ratio = N/self.count
            pred1 = self.count - N
            
            pred0_ACC = 1. if N == 0 else (N-N_1)/N
            pred1_ACC = 1. if pred1 == 0 else N_2/pred1
            r = {
                f'R_{int(j*100)}': pred0_ratio,
                f'A0_{int(j*100)}': pred0_ACC,
                f'A1_{int(j*100)}': pred1_ACC,
            }
            self.result.update(r)
            self.result_L.append(r)

        self.count = 0
        self.statD = []
        for j in self.stat_items:
            self.statD.append([j, 0, 0, 0])

        return self.result

    def __str__(self) -> str:
        if self.result is None:
            return 'None result'
        else:
            s = ''
            for r in self.result_L:
                s += str(r) + '\n'
            return s



def test_global_effiction(model, ckpt):

    model.load_state_dict(ckpt['state_dict'])
    
    model = model.cuda().eval()
    torch.set_grad_enabled(False)
    
    datadir = 'data/water_seg/train'
    import os, tqdm
    from PIL import Image
    import numpy as np
    import albumentations as albu
    train_mean = [0.3435, 0.3699, 0.3505]
    train_std = [0.1869, 0.1978, 0.2063]
    aug = albu.Normalize(train_mean, train_std, max_pixel_value=255)
    
    listdir = os.listdir(datadir+"/image")
    
    log_threshold = 0.5
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    LEN = len(thresholds)
    count_pred0, count_pred0_right, count_pred1_right = [0]*LEN, [0]*LEN, [0]*LEN
    count = 0

    t0 = time()
        
    for fi, imgf in enumerate(listdir):
        labelf = os.path.splitext(imgf)[0] + "_label.tif"
        f_img = datadir+"/image/"+imgf
        f_label = datadir+"/label/"+labelf
        img = np.array(Image.open(f_img).convert("RGB"))
        label = np.array(Image.open(f_label).convert("L"))
        aug_res = aug(image=img, mask=label)
        img = torch.from_numpy(aug_res['image']).permute(2, 0, 1).float().unsqueeze(0)
        label = torch.from_numpy(aug_res['mask']).unsqueeze(0)
        
        result = model({"img":img, "gt_semantic_seg":label})
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
                print(f"{fi}/{len(listdir)}  pred0-acc: {pred0_right}/{pred0},  pred1-acc: {pred1_right}/{pred1}")
    
    for p0, p0r, p1r, threshold in zip(count_pred0, count_pred0_right, count_pred1_right, thresholds):
        print(f"\nthreshold = {threshold}:")
        print(f"pred0: {p0}/{count}, pred0-acc: {p0r}/{p0},  pred1-acc: {p1r}/{count-p0}")
        print(f"pred0: {(p0/count):.4f}, pred0-acc: {(p0r/p0):.4f},  pred1-acc: {(p1r/(count-p0)):.4f}")

    print("spend", time()-t0)



if __name__ == "__main__":
    # B = ResNetEncoder('resnet50')
    # D = FPNDecoder_2(B.channels)
    # _,x = B(torch.rand(1, 3, 896,896))
    # D(*x)
    exit()


    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'

    model = HRSegmentor(global_backbone_name='resnet18', global_patches=(2,2), global_downsample=1, global_only=True)
    ckpt = torch.load('work_dir/GID_Water/hr_segmentor_ppn-e40/version_0/epoch=15@A0_40=0.7838@hr_segmentor_ppn-GID_Water.ckpt')
    test_global_effiction(model, ckpt)

    # model = HRSegmentor(global_backbone_name='resnet18', global_patches=(1,1), global_downsample=1/2, global_only=True)
    # ckpt = torch.load('work_dir/GID_Water/hr_segmentor_ppn-e40/version_1/epoch=32@A0_40=0.7966@hr_segmentor_ppn-GID_Water.ckpt')
    # test_global_effiction(model, ckpt)

    # model = HRSegmentor(global_backbone_name='resnet50', global_patches=(2,2), global_downsample=1/2, global_only=True)
    # ckpt = torch.load('work_dir/GID_Water/hr_segmentor_ppn-e40/final/epoch=36@A0_40=0.8871@hr_segmentor_ppn-GID_Water.ckpt')
    # test_global_effiction(model, ckpt)



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