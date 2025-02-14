'''
#!!!
全局网络消融实验： 1/8、和 1/4,     res50/res18(?)
#!!!
计算每种模型推理，显存消耗和速度
#
比较各种downsample训练对 各种downsample验证的影响
改损失函数

比较

1、 (2,2) downsample_1      resnet50
2、 (1,1) downsample_4      resnet50
3、 (1,1) downsample_8      resnet50
4、 g_d4_r50 + l_d1_r101


#!!!下采样的方式， 这里是普通的插值下采样


#! 全局网络下采样的设计，
'''

from collections import OrderedDict
import random

from torch import Tensor
from jscv.hr_models.base_models import *

from jscv.utils.trainer import Evaluator
from jscv.utils.utils import TimeCounter
import cv2


do_debug = False
count_1 = TimeCounter(do_debug)

do_train_try = False

train_setting_d4 = [((8,8), 1, 0.1), ((4,4), 1/2, 0.1), ((2,2),1/4, 0.6), ((1,1),1/8,0.2)]
train_setting_d4_1 = [((4,4), 1/2, 0.0), ((2,2),1/4, 1.)]

train_setting_d4_2 = [((4,4), 1/2, 0.1), ((2,2),1/4, 0.75), ((1,1),1/8,0.15)]
train_setting_d4_3 = [((4,4), 1/2, 0.1), ((2,2),1/4, 0.9)]
train_setting_d1 = [((8,8), 1, 0.7), ((4,4), 1/2, 0.2), ((2,2),1/4, 0.1)]
train_setting_d2 = [((4,4), 1/2, 0.8), ((2,2),1/4, 0.2)]



def div_patches(img, mask, n_patches, batch_size=1):
    '''
        TODO 需要优化
    '''
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

    if batch_size != 1:
        L_imgs, L_masks = [], []
        L1, L2 = [], []
        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            L1.append(imgij)
            L2.append(maskij)
            if idx != len(img_patches)-1 and len(L1) < batch_size:
                continue
            if len(L1) == 1:
                imgij, maskij = L1[0], L2[0]
            else:
                imgij = torch.concat(L1, dim=0)
                maskij = torch.concat(L2, dim=0)
            L1, L2 = [], []
            L_imgs.append(imgij)
            L_masks.append(maskij)
        return L_imgs, L_masks

    return img_patches, mask_patches

def div_img_patches(img, n_patches, batch_size=1):
    B, C, H, W = img.shape
    PH, PW = n_patches
    HL, WL = H//PH, W//PW
    img_p = img.reshape(B, C, PH, HL, PW, WL)
    img_patches = []
    for i in range(PH):
        for j in range(PW):
            img_patches.append(img_p[:, :, i, :, j])
    if batch_size != 1:
        L_imgs = []
        L1 = []
        for idx, imgij in enumerate(img_patches):
            L1.append(imgij)
            if idx != len(img_patches)-1 and len(L1) < batch_size:
                continue
            if len(L1) == 1:
                imgij = L1[0]
            else:
                imgij = torch.concat(L1, dim=0)
            L1 = []
            L_imgs.append(imgij)
        return L_imgs
    return img_patches

def recover_img(x, n_patches):
    '''
        交叠
    '''
    if len(x) == 0:
        return
    if len(x[0].shape) == 3:
        return recover_mask(x, n_patches)

    PH, PW = n_patches
    x = torch.concat(x, dim=0)
    B,C,H,W = x.shape
    # print(x.shape)
    assert B == PH*PW
    x = x.reshape(1, PH, PW, C, H, W).permute(
        0, 3, 1, 4, 2, 5).reshape(1,C,H*PH, W*PW)
    # print("@@@@", x.shape)
    return x

def recover_mask(x, n_patches):
    '''
        交叠
    '''
    if len(x) == 0:
        return
    if len(x[0].shape) == 4:
        # print("@@@@", x[0].shape)
        return recover_img(x, n_patches)
    PH, PW = n_patches
    x = torch.concat(x, dim=0)
    B,H,W = x.shape
    assert B == PH*PW
    x = x.reshape(1, PH, PW, H, W).transpose(2,3).reshape(1,H*PH, W*PW)
    return x

def optimize_step(self, optimizer_name, step_interval=1):
    if hasattr(self, optimizer_name):
        cn = optimizer_name+"_counter"
        if not hasattr(self, cn):
            setattr(self, cn, 0)
        ct = getattr(self, cn) + 1
        setattr(self, cn, ct)
        if ct % step_interval == 0:
            opt = getattr(self, optimizer_name)
            if isinstance(opt, list):
                for o in opt:
                    o.step()
                    o.zero_grad()
            else:
                opt.step()
                opt.zero_grad()

def optimize_step_now(self, optimizer_name):
    if not self.training: return
    if hasattr(self, optimizer_name):
        opt = getattr(self, optimizer_name)
        if isinstance(opt, list):
            for o in opt:
                o.step()
                o.zero_grad()
        else:
            # print("@ optimize_step_now")
            opt.step()
            opt.zero_grad()

class PatchesSegmentor(nn.Module):
    def __init__(
            self,
            net,
            seg_loss_layer: nn.Module = None,
            train_setting_list=train_setting_d1,
            # train_setting_list=[((8,8), 1, 0.8), ((4,4), 1/2, 0.15), ((2,2),1/4, 0.05)],
            val_setting=((4,4), 1),
            keys_name=['img', 'gt_semantic_seg', 'id'],
            optimizer_step_interval=10,
            batch_size=1,   #TODO 需要区分 train 和 val
            save_images_args=dict(per_n_step=15,),
            # loss_layer_resize_mask=True,
        ):
        '''
            train_setting_list:  per_item: (patches, downsample, Probability)
            val_setting:(patches, downsample)
        '''
        super().__init__()
        self.seg_loss_layer = seg_loss_layer
        self.keys_name = keys_name

        self.optimizer_step_interval = optimizer_step_interval
        self.batch_size = batch_size
        self.train_setting_list = train_setting_list
        self.val_setting = val_setting
        self.Probabilitys = [p for _,_,p in self.train_setting_list]
        # assert sum(self.Probabilitys) == 1
        # self.loss_layer_resize_mask = loss_layer_resize_mask

        self.net = net
        self.c1 = 0
        self.sim = SaveImagesManager(**save_images_args)




    def forward(self, batch:dict):

        # print("@@@2", self.optimizer.state_dict()['param_groups'][0]['lr'])

        img, mask = batch[self.keys_name[0]], batch[self.keys_name[1]]
        assert img.shape[0] == 1, '为方便代码'
        if self.training:
            patches, downsample, _ = random.choices(self.train_setting_list, self.Probabilitys)[0]
        else:
            patches, downsample = self.val_setting
        
        self.n_patches = patches
        self.n_downsample = downsample
        img_patches, mask_patches = div_patches(img, mask, patches, self.batch_size)
        # print(patches, downsample)
        if do_train_try:
            try:
                result = self.train_forward(img_patches, mask_patches)
            except Exception as e:
                print("Exception:", type(e), e)
                print(f'patches:{patches}, downsample:{downsample}')
                return {'skip': True}
        else:
            result = self.train_forward(img_patches, mask_patches)

        if self.sim.step():
            pred = result.get('pred')
            if pred is not None:
                pred = pred.squeeze(0).argmax(dim=0)
            id = batch.get(self.keys_name[2])
            if id is not None:
                id = id[0]
            self.save_image(img.squeeze(0), pred, mask.squeeze(0), id, result)


        return result


    def save_image(self, img, pred, mask, id, result: dict):
        self.sim.save_image(img, pred, mask, id)



    def train_forward(self, img_patches, mask_patches):
        '''
            TODO 不升采样img、而降采样mask
        '''
        
        outputs, result = [], {}
        PH, PW = self.n_patches
        losses = 0

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            self.idx = idx
            imgij = resize_to(imgij.cuda(), self.n_downsample)

            x = self.net(imgij)
            x = resize_to_y(x, maskij)
            loss = self.seg_loss_layer(x, maskij.cuda())


            if self.training:
                loss.backward()
                optimize_step(self, 'optimizer', self.optimizer_step_interval)

            losses += loss.detach()
            if self.batch_size > 1:
                for y in torch.split(x, 1):
                    outputs.append(y)
            else:
                outputs.append(x)

        result = {'pred':recover_img(outputs, self.n_patches).softmax(1),'losses': {'main_loss':losses}}
        return result


    def predict(self, img, zero_img=False):
        '''
            zero_img: 表示输入图片为全zero
        '''
        B, C, H, W = img.shape
        patches, self.n_downsample = self.val_setting
        self.n_patches = patches
        img_patches = div_img_patches(img, patches, self.batch_size)
        
        outputs = []
        PH, PW = patches
        for idx, (imgij) in enumerate(img_patches):
            imgij = resize_to(imgij.cuda(), self.n_downsample)
            x = self.net(imgij)
            outputs.append(x)

        x = recover_img(outputs, self.n_patches)
        x = resize_to_y(x, img).softmax(1)
        return x

    




class PatchErrEvaluator(Evaluator):
    def __init__(self, pathes=(8,8), ignore_idx=-100,
                 err_gates=[1, 2, 4, 8, 15, 20, 30],
                 easyp_gates=[0.1, 0.5, 1, 2, 5, 10]
                 ):
        super().__init__()
        self.pathes = pathes
        self.err_gates = err_gates
        self.ignore_idx = ignore_idx
        self.easyp_gates = easyp_gates
        self.reset()

    def reset(self):
        self.count = 0
        self.stat_err = []
        for g1 in self.err_gates:
            self.stat_err.append(0)
        self.stat_easyp = []
        for g1 in self.err_gates:
            self.stat_easyp.append([0,0])

    def append(self, result):
        '''
            target:  [0, 1, ignore_idx]
        '''
        pred = result["pred"].squeeze(0)
        target = result["target_cpu"].squeeze(0).to(pred.device)
        H,W = target.shape
        PH, PW = self.pathes
        h1, w1 = H//PH, W//PW
        pred = pred.argmax(dim=0)
        valid = (target!=self.ignore_idx)
        wrong_map = (pred!=target) & valid

        valid = valid.reshape(PH, h1, PW, w1)
        target = target.reshape(PH, h1, PW, w1)
        wrong_map = wrong_map.reshape(PH, h1, PW, w1)
        
        target[~valid] = 0

        for i in range(PH):
            for j in range(PW):
                self.count += 1
                v, t, w = valid[i,:,j], target[i,:,j], wrong_map[i,:,j]
                sum_w, sum_v, sum_t = w.sum(), v.sum(),t.sum()
                rw = (sum_w/sum_v)*100
                rt = (sum_t/sum_v)*100
                
                for p, g1 in enumerate(self.err_gates):
                    # print("@", rw, g1)
                    if rw < g1:
                        self.stat_err[p] += 1

                for p, g2 in enumerate(self.easyp_gates):
                    # print(rt, g2)
                    if rt<g2 or rt>(1-g2):
                        a, b = self.stat_easyp[p]
                        self.stat_easyp[p] = [a+int(sum_w), b+int(sum_v)]
                    
    
    def evaluate(self):
        s = 'Patches-Count of every Err-Rate-Threshold:\n'
        Ct = self.count
        for g, c in zip(self.err_gates, self.stat_err):
            x = (c/Ct)*100
            s += f' [<{g}%] {x:.2f},'
        s +='\nPatches-Err-Rate of every Easy-Threshold\n'
        for g, (w,v) in zip(self.easyp_gates, self.stat_easyp):
            r = (w/v)*100
            s += f' [{g}%] {r:.2f},'
        s += '\n'
        self.str1 = s
        self.reset()
        return {}
    
    def __str__(self) -> str:

        return self.str1


def test_predict_speed():
    '''
        从文件中读取图片消耗了太多时间, 需要减去
        但不减去图片加载到cuda的时间
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    count_1.DO_DEBUG = False
    
    
    
    torch.set_grad_enabled(False)
    warmup()
    
    net = ResSegNet('resnet50')
    model = PatchesSegmentor(net, val_setting=((4,4), 1))

    ''' load state_dict'''
    model = model.cuda().eval()

    import tqdm
    
    ds = water_seg_dataset('data/gid_water/val')

    print("begin:")
    t0 = time()

    tq = tqdm.tqdm(total=ds.len)

    for fi, (img, label) in enumerate(ds.batch()):
        model.predict(img.cuda())

        # print(fi, pred.shape, label.shape)
        tq.update(1)

    total_time = time() - t0
    load_img_time = ds.time_count
    
    print("load spend:", load_img_time)
    print("predict spend:", total_time-load_img_time)

    # print(count_1.str_total())


def test_predict_speed_pure():
    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 20
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    net = ResSegNet('resnet101', decoder_args=fpn_decoder_args_1)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))


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
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')










def test_predict_speed_pure_pvt():

    import config._base_.backbone.pvtv2 as pvtv2

    class PVTSegNet(EncoderDecoder):
        def __init__(self,
                    backbone_name='pvt_v2_b3',
                    decoder_class=FPNDecoder,
                    decoder_args=fpn_decoder_args_1,
                    features_only=True,
        ):
            if backbone_name == 'pvt_v2_b3':
                backbone = pvtv2.backbone_pvtv2_b3()
            elif backbone_name == 'pvt_v2_b1':
                backbone = pvtv2.backbone_pvtv2_b1()
            elif backbone_name == 'pvt_v2_b2':
                backbone = pvtv2.backbone_pvtv2_b2()
            l_chs = backbone.embed_dims
            decoder = decoder_class(l_chs, **decoder_args)
            super().__init__(backbone, decoder)


    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 50
    # input = torch.zeros(1, 3, 1024*7, 1024*7)
    input = torch.zeros(1, 3, 1024*7//2, 1024*7//2)
    load_gpu = True

    '''
    7K
        fpn_decoder_args_2
        b1 FPS: 5.3595538015668716    18.66ms
        b2 FPS: 2.8576829529730268    34.99ms

        fpn_decoder_args_1
        b1 FPS: 5.3595538015668716    18.13ms
        b2 FPS: 2.8576829529730268    34.25mms
    '''


    '''
    3.5K
        fpn_decoder_args_256
        FPS: 31.04065083249373    3.22ms

        fpn_decoder_args_512
        FPS: 28.9541182970966    3.45ms
        
        fpn_decoder_args_512    blocks=[1,2,3,2]
        FPS: 26.952570256029574    3.71ms
        
        fpn_decoder_args_512    blocks=[1,1,3,2]
        FPS: 27.502413663499244    3.64ms

        fpn_decoder_args_512    blocks=[1,1,3,1]
        FPS: 27.948935385332174    3.58ms

        fpn_decoder_args_512    blocks=[1,1,2,2]
        FPS: 28.015105954081       3.57ms
        
        fpn_decoder_args_512    blocks=[1,1,2,1]
        FPS: 28.452066084868385    3.51ms

        fpn_decoder_args_512    blocks=[1,1,1,2]
        FPS: 28.64199527229639    3.49ms
        
        --------------------
        B2

        fpn_decoder_args_512    blocks=[1,1,3,1]
        FPS: 17.086574584010357    5.85ms
    '''
    # fpn_decoder_args = fpn_decoder_args_256
    # fpn_decoder_args = fpn_decoder_args_512
    fpn_decoder_args = fpn_decoder_args_512
    fpn_decoder_args.update(dict(blocks=[1,1,3,1]))
    # [1, 2, 3, 2]


    # net = PVTSegNet('pvt_v2_b1', decoder_args=fpn_decoder_args)
    net = PVTSegNet('pvt_v2_b1', decoder_args=fpn_decoder_args)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))


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
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')



def test_predict_speed_pure_swin():
    import config._base_.backbone.swin as swin
    class SwinSegNet(EncoderDecoder):
        def __init__(self,
                    backbone_type=swin.swin_base_ft,
                    decoder_class=FPNDecoder,
                    decoder_args=fpn_decoder_args_1,
                    features_only=True,
        ):
            backbone, self.backbone_ckpt_path, self.backbone_prefix, backbone_features = backbone_type()
            decoder = decoder_class(backbone_features, **decoder_args)
            super().__init__(backbone, decoder)


    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 20
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    '''
        swin_base_ft    21.22ms
        swin_small_ft   14.29ms
    '''
    
    
    net = SwinSegNet(swin.swin_small_ft, decoder_args=fpn_decoder_args_1)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))


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
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')

def test_predict_speed_pure_focal():
    from config._base_.backbone.focalnet import Focal_S
    class FocalSegNet(EncoderDecoder):
        def __init__(self,
                    backbone_type=Focal_S,
                    decoder_class=FPNDecoder,
                    decoder_args=fpn_decoder_args_1,
                    features_only=True,
        ):
            backbone, self.backbone_ckpt_path, self.backbone_prefix, backbone_features = backbone_type()
            decoder = decoder_class(backbone_features, **decoder_args)
            super().__init__(backbone, decoder)

    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 20
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    '''
        18.04ms ms
    '''
    
    
    net = FocalSegNet(Focal_S, decoder_args=fpn_decoder_args_1)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))


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
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')


def test_predict_speed_pure_RMT():
    
    from jscv.backbone.RMT import RMT_S, RMT_T3

    class RMT_SegNet(EncoderDecoder):
        def __init__(self,
                    backbone = RMT_S,
                    decoder_class=FPNDecoder,
                    decoder_args=fpn_decoder_args_1,
        ):
            if backbone == RMT_S:
                backbone = RMT_S()
                self.ckpt_path = "pretrain/RMT-S-label.pth"
                # ckpt_path = "pretrain/RMT-S.pth"
                print("backbone: RMT_S")
            elif backbone == RMT_T3:
                backbone = RMT_T3()
                self.ckpt_path = "pretrain/RMT-T.pth"
                print("backbone: RMT_T3")
            else:
                raise ValueError("Invalid backbone")

            l_chs = [64, 128, 256, 512]
            print("backbone.embed_dim:", l_chs)

            decoder = decoder_class(l_chs, **decoder_args)
            super().__init__(backbone, decoder)


    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 50
    # input = torch.zeros(1, 3, 1024*7, 1024*7)
    input = torch.zeros(1, 3, 1024*7//2, 1024*7//2)
    load_gpu = True



    '''
    3.5K
    
        RMT_T3  fpn_decoder_args_512_1131   FPS: 14.103961325261995    7.09ms
        RMT_T3  fpn_decoder_args_512_1234   FPS: 13.593301561320922    7.36ms

        RMT_S   fpn_decoder_args_512_1131   FPS: 12.630897537599193    7.92ms
        RMT_S   fpn_decoder_args_512_1234   FPS: 12.065081593621354    8.29ms
        RMT_S   fpn_decoder_args_512_1133   FPS: 12.147896553300116    8.23ms
        
    
    
        
        RMT_T3  fpn_decoder_args_M_1234     FPS: 13.572291027592597    7.37ms

        RMT_S   fpn_decoder_args_M_1234   FPS: 12.499032244510316    8.00ms


    '''
    # fpn_decoder_args = fpn_decoder_args_256
    # fpn_decoder_args = fpn_decoder_args_512
    fpn_decoder_args = fpn_decoder_args_M_1234


    # net = RMT_SegNet(RMT_S, decoder_args=fpn_decoder_args)
    net = RMT_SegNet(RMT_S, decoder_args=fpn_decoder_args)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))


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
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')









def test_model_pure_time_3d5K(model, val_setting, cudaId=1):
    '''
        计算模型的纯粹的推理时间， 3d5K图上
    '''
    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']=f'{cudaId}'
    epochs = 50
    # input = torch.zeros(1, 3, 1024*7, 1024*7)
    input = torch.zeros(1, 3, 1024*7//2, 1024*7//2)
    load_gpu = True

    model = PatchesSegmentor(model, val_setting=val_setting)

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
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')




if __name__ == "__main__":

    test_predict_speed_pure_RMT()
