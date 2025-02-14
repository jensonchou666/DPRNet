import random
from jscv.hr_models.pathes_segmentor import *

from jscv.utils.trainer import Evaluator
from jscv.utils.utils import TimeCounter
import cv2

from jscv.hr_models.ISDNet.isdnet import *
# from .isdnet import *

do_debug = False
count_1 = TimeCounter(do_debug)

print_losses_counter = 0
print_losses_count_once = 200



'''
    #TODO 做成通用 补丁分块模型
    
    本文件 是 对ISDNet的复现, 结构区别不大

    本目录其它文件 是 ISDHead源码 移植过来， 未作改动

    ISDNet的中间层,  就相当于 下面的全局网络
    
    但是 没加最下面的那个辅助头


    原 ISDNet 不分块无法 在 10G 显存显卡上跑，所以这里还是分块了的
    
    拖到最下面的推理时间证明了， 在有批处理的加持下，  适当的补丁划分对速度的影响并不大, 还节省了显存


    当然，  更大显存的显卡，  比如 32G、100+G 的 
    
    
    

'''


ISDHead_Kargs = dict(
    in_channels=3,
    channels=128,
    num_classes=2,
    dropout_ratio=0.1,
    norm_cfg={},
    align_corners=False,
)

# def ISDNet_forward(img, fetures_G, label):
#     '''
#         (2, 3, 512, 512)
#         (2, 512,512)
#         (2, C, 128//N,128//N)
#     '''

# 我的初始化：
def createISDNet(kargs):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method='env://')
    return ISDHead(**kargs)

# 程序之后报错：
#   File "/workspace/JsSeg/0DPRNet/jscv/hr_models/ISDNet/double_branch_net.py", line 205, in forward
#     seg_loss_l.backward()
#   File "/opt/conda/lib/python3.7/site-packages/torch/_tensor.py", line 307, in backward
#     torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
#   File "/opt/conda/lib/python3.7/site-packages/torch/autograd/__init__.py", line 156, in backward
#     allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
# RuntimeError: no valid convolution algorithms available in CuDNN


class DoubleBranchNet_ISDNet(nn.Module):
    def __init__( 
            self,
            global_encoder,
            global_decoder,
            
            isdnet_kargs,
            
            global_seg_loss_layer: nn.Module = None,
            local_seg_loss_layer: nn.Module = None,
            global_seg_loss_weight=1,
            global_patches={'train':(2,2), 'val':(1,1)},
            local_patches={'train':(4,4), 'val':(2,2)},

            global_downsample=1/4,
            local_downsample=1,

            optimizer_step_interval=10,
            local_batch_size={'train':1, 'val': 1},
            save_images_args=dict(per_n_step=15,),
        ):
        super().__init__()
        assert global_downsample*4 == local_downsample
        
        self.global_encoder = global_encoder
        self.global_decoder = global_decoder

        self.ISDNet = createISDNet(isdnet_kargs)



        self.global_seg_loss_layer = global_seg_loss_layer
        self.local_seg_loss_layer = local_seg_loss_layer
        self.global_patches = global_patches
        self.local_patches = local_patches
        self.global_downsample = global_downsample
        self.local_downsample = local_downsample
        self.optimizer_step_interval = optimizer_step_interval
        self.global_seg_loss_weight = global_seg_loss_weight
        self.local_batch_size = local_batch_size

        self.sim = SaveImagesManager(**save_images_args)


    def forward(self, batch:dict):
        
        
        img, mask = batch['img'], batch['gt_semantic_seg']
        s1 = 'train' if self.training else 'val'
        PH, PW = global_patches = self.global_patches[s1]
        PH2, PW2 = self.local_patches[s1]
        SH, SW = (PH2//PH), (PW2//PW)

        g_outputs, outputs, result = [], [], {}
        
        seg_losses_g = 0
        seg_losses_l = 0
        img_patches, mask_patches = div_patches(img, mask, global_patches, 1)
        

        for idx, (imgij, maskij) in enumerate(zip(img_patches, mask_patches)):
            B,C,H,W = imgij.shape
            g_input = resize_to(imgij, self.global_downsample)  #1/4
            H2, W2 = H//SH, W//SW
            imgij = imgij.reshape(B, C, SH, H2, SW, W2)
            maskij_reshape = maskij.reshape(B, SH, H2, SW, W2)
            segs = []
            
            def extract_g():
                self.global_decoder.d2_feature_only = False
                g_fs = self.global_encoder(g_input)
                g_seg, g_s32 = self.global_decoder(*g_fs)
                # g_s16 = g_fs[1]
                g_seg = resize_to_y(g_seg, maskij)
                return g_seg, g_s32

            # 先训练 globalnet
            g_seg, g_s32 = extract_g()
            seg_loss_g = self.global_seg_loss_layer(g_seg, maskij) * self.global_seg_loss_weight
            # if self.training:
            #     seg_loss_g.backward()
            #     optimize_step(self, 'optimizer_global', 1)
            seg_losses_g += seg_loss_g.detach()
            g_outputs.append(g_seg.detach())

            g_s32 = g_s32.detach()
            
            B, C32, H32, W32 = g_s32.shape
            g_s32 = g_s32.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)


            BZ = self.local_batch_size['train']

            if BZ > 1:
                img_L = []
                s32_pq_L = []
                maskijpq_L = []
                
            counter = 0
            for p in range(SH):
                for q in range(SW):

                    img = resize_to(imgij[:,:,p,:,q], self.local_downsample)
                    s32_pq = g_s32[:,:,p,:,q]
                    maskijpq = maskij_reshape[:,p,:,q]
                    
                    if BZ > 1: 
                        img_L.append(img)
                        s32_pq_L.append(s32_pq)
                        maskijpq_L.append(maskijpq)
                        if len(img_L) >= BZ or (p == (SH-1) and q == (SW-1)):
                            img = torch.concat(img_L, 0)
                            s32_pq = torch.concat(s32_pq_L, 0)
                            maskijpq = torch.concat(maskijpq_L, 0)
                            img_L, s32_pq_L, maskijpq_L = [], [], []
                        else:
                            continue

                    if self.training:
                        pred, losses = self.ISDNet.forward_train(img, s32_pq, maskijpq)
                        losses, losses_aux16, losses_aux8, losses_recon, losses_fa = losses
                        seg_loss_l =  losses + losses_aux16 + losses_aux8 + losses_recon['recon_losses'] \
                            + losses_fa['fa_loss']
                    else:
                        pred = self.ISDNet.forward(img, s32_pq, False)
                        seg_loss_l = torch.tensor(0, device=img.device)
                    pred = resize_to_y(pred, maskijpq)

                    # global print_losses_counter
                    # print_losses_counter += 1
                    # if print_losses_counter % print_losses_count_once == 0:
                    #     print("####  seg_loss_g: ", seg_loss_g)
                    #     print("####  isd losses: ", losses)
                    # seg_x = resize_to_y(seg_x, maskijpq)
                    # seg_loss_l = self.local_seg_loss_layer(seg_x, maskijpq)



                    if self.training:
                        if counter == 0:
                            (seg_loss_l+seg_loss_g).backward()
                            optimize_step(self, 'optimizer_global', 1)
                        else:
                            seg_loss_l.backward()
                        optimize_step(self, 'optimizer_local', self.optimizer_step_interval)
                        counter += 1
                    seg_losses_l += seg_loss_l.detach()
                    
                    if BZ > 1:
                        predL = torch.split(pred, 1)
                        for ppp in predL:
                            segs.append(ppp.detach())
                    else:
                        segs.append(pred.detach())

            outputs.append(recover_img(segs, (SH,SW)))

        # pred_logits = recover_img(outputs, global_patches)
        pred = recover_img(outputs, global_patches).argmax(dim=1)
        g_seg = recover_img(g_outputs, global_patches).argmax(dim=1)
        outputs.clear()
        g_outputs.clear()
        del g_s32, g_input


        if self.sim.step():
            id = batch.get('id')
            if id is not None:
                id = id[0]
            self.sim.save_image(img.squeeze(0),
                                pred.squeeze(0),
                                mask.squeeze(0),
                                id,
                                g_seg.squeeze(0))
        result = {'pred':pred,
                  'coarse_pred':g_seg,
                  'losses': {
                      'main_loss':seg_losses_g+seg_losses_l,
                      'seg_loss_g':seg_losses_g,
                      'seg_loss_l':seg_losses_l}}
        return result



    def predict(self, img):
        PH, PW = global_patches = self.global_patches['val']
        PH2, PW2 = self.local_patches['val']
        SH, SW = (PH2//PH), (PW2//PW)

        outputs= []
        img_patches = div_img_patches(img, global_patches, 1)
        
        for idx, imgij in enumerate(img_patches):
            B,C,H,W = imgij.shape
            H2, W2 = H//SH, W//SW
            
            segs = []

            self.global_decoder.d2_feature_only = True
            g_fs = self.global_encoder(resize_to(imgij, self.global_downsample))
            g_s32 = self.global_decoder(*g_fs)
            del g_fs
            
            imgij = imgij.reshape(B, C, SH, H2, SW, W2)
            B, C32, H32, W32 = g_s32.shape
            g_s32 = g_s32.detach().reshape(B, C32, SH, H32//SH, SW, W32//SW)

            BZ = self.local_batch_size['val']
            if BZ > 1:
                img_L = []
                s32_pq_L = []

            for p in range(SH):
                for q in range(SW):
                    
                    img = resize_to(imgij[:,:,p,:,q], self.local_downsample)
                    s32_pq = g_s32[:,:,p,:,q]

                    if BZ > 1: 
                        img_L.append(img)
                        s32_pq_L.append(s32_pq)
                        # print('#', len(img_L) , BZ, p, q, SH, SW)
                        
                        if len(img_L) >= BZ or (p == (SH-1) and q == (SW-1)):
                            # print('##########', len(img_L) , BZ)
                            img = torch.concat(img_L, 0)
                            s32_pq = torch.concat(s32_pq_L, 0)
                            img_L, s32_pq_L = [], []
                        else:
                            continue

                    pred = self.ISDNet.forward_test(img, s32_pq)
                    
                    
                    if BZ > 1:
                        predL = torch.split(pred, 1)
                        for ppp in predL:
                            segs.append(ppp.detach())
                    else:
                        segs.append(pred)

            outputs.append(recover_img(segs, (SH,SW)))
        pred = recover_img(outputs, global_patches)
        pred = resize_to_y(pred, img)
        return pred



def test_predict_speed_pure():
    '''
        
        
    计算纯粹的推理时间，


    结论：
    
    
    

    -------------------- 1 ----------------------------
    batch_size = 1    不使用批处理，  补丁越小速度当然变慢很多
    -------------------- 1 ----------------------------

    1.  local_patches['val'] = (2,2)        local_batch_size['val']= 1      33.86ms
    3.  local_patches['val'] = (4,4)        local_batch_size['val']= 1      35.91ms
    4.  local_patches['val'] = (8,8)        local_batch_size['val']= 1      51.44ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 1      180.50ms

    
    
    -------------------- 2 ----------------------------
    batch_size = {设置}   使用批处理，进行补丁划分对速度的影响很小
    -------------------- 2 ----------------------------


    1.  local_patches['val'] = (2,2)        local_batch_size['val']= 1      33.86ms
    2.  local_patches['val'] = (4,4)        local_batch_size['val']= 4      32.89ms
    3.  local_patches['val'] = (4,4)        local_batch_size['val']= 6      32.49ms
    4.  local_patches['val'] = (8,8)        local_batch_size['val']= 24     33.15ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 100    34.56ms
    6.  local_patches['val'] = (32,32)      local_batch_size['val']= 400    40.23ms # 从 32 开始， 速度才有一点下降
    7.   (64,64)  无法继续划分，


    NEW:
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 100    35.38ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 50     36.40ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 30     36.84ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 10     38.44ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 5      42.39ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 4      51.06ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 3      65.97ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 2      93.75ms
    5.  local_patches['val'] = (16,16)      local_batch_size['val']= 1      175.22ms

    '''






    import os
    from jscv.utils.utils import warmup
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 40
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    GB = ResNetEncoder('resnet101', features_only=True)
    GD = FPNDecoder(GB.channels, 
                    return_context=True,
                    context_only=True,
                    **fpn_decoder_args_1)


    isd_args = ISDHead_Kargs
    
    isd_args.update(dict(
        pretrain_path=None,
    ))

    model = DoubleBranchNet_ISDNet(GB, GD, isd_args,
                                   global_seg_loss_layer=None,
                                   local_seg_loss_layer=None,
                                   global_patches={'train':(2,2), 'val':(1,1)},
                                   local_patches={'train':(4,4), 'val':(2,2)},
                                   
                                   local_batch_size={'train':1, 'val':1},

                                   global_downsample=1/4,
                                   local_downsample=1,
                                #    save_images_args=dict(per_n_step=10000),
                                   )
    print("@@@")
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
        model.predict(input)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')


if __name__ == "__main__":

    test_predict_speed_pure()