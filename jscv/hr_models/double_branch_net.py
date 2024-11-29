import random
from jscv.hr_models.pathes_segmentor import *

from jscv.utils.trainer import Evaluator
from jscv.utils.utils import TimeCounter
import cv2


do_debug = False
count_1 = TimeCounter(do_debug)


# Global Local Feature Fusion Network
class GL_FFN(nn.Module):
    def __init__(self, 
                 global_channels, 
                 local_channels,
                 blocks=2,
                 pose_decoder_cls=FPNDecoder,
                 decoder_args={},
                 ):
        '''
            global: [s16, s32]
            local: [s2, s4, s8, s16, s32]
        '''
        super().__init__()
        C1_g, C2_g = global_channels[-2:]
        C0_l, C1_l, C2_l, C3_l, C4_l = local_channels
        self.blocks = blocks
        # print(C4_l, C2_g)
        if blocks > 0:
            self.layer1 = []
            for k in range(blocks):
                C = C4_l+C2_g
                self.layer1.append(ConvBNReLU(C, C, 3))
            self.layer1 = ResBlocks2(*self.layer1)

        self.pose_decoder=pose_decoder_cls(
            [C0_l, C1_l, C2_l, C3_l+C1_g, C4_l+C2_g],
            **decoder_args
        )

    def forward(self, g_fs, l_fs, coordinate, patches):
        PH, PW = patches
        i, j = coordinate
        for k, g in enumerate(list(g_fs)):
            B,C,H,W = g.shape
            g_fs[k] = g.reshape(B,C, PH, H//PH, PW, W//PW)[:,:,i,:,j]
        g16, g32 = g_fs
        f0, f1, f2, f3, f4 = l_fs
        
        f4 = torch.concat([f4, g32], dim=1)
        f3 = torch.concat([f3, g16], dim=1)
        if self.blocks > 0:
            f4 = self.layer1(f4)
        return self.pose_decoder(f0, f1, f2, f3, f4)


class DoubleBranchNet(nn.Module):
    def __init__( 
            self,
            global_encoder,
            global_decoder,
            local_encoder, 
            GL_ffn_decoder,

            global_seg_loss_layer: nn.Module = None,
            local_seg_loss_layer: nn.Module = None,
            global_seg_loss_weight=1,
            global_patches={'train':(2,2), 'val':(1,1)},
            local_patches={'train':(4,4), 'val':(2,2)},
            global_downsample=1/4,
            local_downsample=1,

            optimizer_step_interval=10,
            # local_batch_size=1,   #直接修改pathes, (8,8) -> (8,4)
            save_images_args=dict(per_n_step=15,),
        ):
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
        self.optimizer_step_interval = optimizer_step_interval
        self.global_seg_loss_weight = global_seg_loss_weight
        # self.local_batch_size = local_batch_size

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
                g_s16 = g_fs[1]
                g_seg = resize_to_y(g_seg, maskij)
                return g_seg, g_s16, g_s32

            # 先训练 globalnet
            g_seg, g_s16, g_s32 = extract_g()
            seg_loss_g = self.global_seg_loss_layer(g_seg, maskij) * self.global_seg_loss_weight
            if self.training:
                seg_loss_g.backward()
                optimize_step(self, 'optimizer_global', 1)
            seg_losses_g += seg_loss_g.detach()
            g_outputs.append(g_seg.detach())

            g_s16, g_s32 = g_s16.detach(), g_s32.detach()

            for p in range(SH):
                for q in range(SW):
                    l_fs = self.local_encoder(resize_to(imgij[:,:,p,:,q], self.local_downsample))
                    seg_x = self.GL_ffn_decoder([g_s16, g_s32], l_fs, coordinate=(p,q), patches=(SH,SW))
                    del l_fs
                    maskijpq = maskij_reshape[:,p,:,q]
                    seg_x = resize_to_y(seg_x, maskijpq)
                    seg_loss_l = self.local_seg_loss_layer(seg_x, maskijpq)
                    if self.training:
                        seg_loss_l.backward()
                        optimize_step(self, 'optimizer_local', self.optimizer_step_interval)
                    seg_losses_l += seg_loss_l.detach()
                    segs.append(seg_x.detach())
            outputs.append(recover_img(segs, (SH,SW)))
            
        # pred_logits = recover_img(outputs, global_patches)
        pred = recover_img(outputs, global_patches).argmax(dim=1)
        g_seg = recover_img(g_outputs, global_patches).argmax(dim=1)
        outputs.clear()
        g_outputs.clear()
        del g_s16, g_s32, g_input


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

    def predict(self, img, zero_img=False):
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
            g_s16 = g_fs[1]
            del g_fs
            
            imgij = imgij.reshape(B, C, SH, H2, SW, W2)

            for p in range(SH):
                for q in range(SW):
                    l_fs = self.local_encoder(resize_to(imgij[:,:,p,:,q], self.local_downsample))
                    seg_x = self.GL_ffn_decoder([g_s16,g_s32], l_fs, coordinate=(p,q), patches=(SH,SW))
                    del l_fs
                    segs.append(seg_x)
            outputs.append(recover_img(segs, (SH,SW)))
        pred = recover_img(outputs, global_patches)
        pred = resize_to_y(pred, img)
        return pred



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
    
    GB = ResNetEncoder('resnet101', features_only=True)
    GD = FPNDecoder(GB.channels, 
                    return_d2_feature=True,
                    d2_feature_only=True,
                    **fpn_decoder_args_1)
    LB = ResNetEncoder('resnet18', features_only=True)
    LD = GL_FFN([GB.channels[1], GD.channel_d2], LB.channels,
                blocks=2,
                pose_decoder_cls=FPNDecoder,
                decoder_args=fpn_decoder_args_1)

    model = DoubleBranchNet(GB, GD, LB, LD, local_patches={'val':(2,2)})

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

    test_predict_speed_pure()