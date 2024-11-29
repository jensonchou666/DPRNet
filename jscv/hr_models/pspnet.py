from jscv.hr_models.base_models import *



class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=x.size()[-2:], mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts

    
class PSPHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 4, 8]):
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class PSPNet(EncoderDecoder):
    def __init__(self,
                 backbone_name='resnet50',
                 decoder_class=FPNDecoder,
                 backbone_args={},
                 decoder_args=fpn_decoder_args_1,
                 features_only=True,
                 psp_out_channels_rate=1/2,
                 psp_pool_sizes=[1, 2, 4, 8],
                #  psp_layers=2,
    ):
        backbone = ResNetEncoder(backbone_name, features_only=features_only, **backbone_args)
        l_chs = backbone.channels
        psp_inc = l_chs[-1]
        psp_outc = int(psp_inc * psp_out_channels_rate)
        
        l_chs[-1] = psp_outc
        decoder = decoder_class(l_chs, **decoder_args)
        super().__init__(backbone, decoder)

        # L = []
        # for i in range(psp_layers):
        #     inc = psp_inc if i == 0 else psp_outc
        #     L.append()
        self.psp = PSPHEAD(psp_inc, psp_outc, psp_pool_sizes)

    def forward(self, x, reture_features=False):
        fs = list(self.backbone(x))
        fs[-1] = self.psp(fs[-1])
        x = self.decoder(*fs)
        if reture_features:
            return x, fs
        return x

    def pretrain_backbone(self, d:dict, from_where='resnet'):
        self.backbone.pretrain(d, from_where)

def test_predict_speed_pure():
    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    from jscv.hr_models.pathes_segmentor import PatchesSegmentor

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 20
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    net = PSPNet('resnet101', decoder_args=fpn_decoder_args_1,
                 psp_out_channels_rate=0.52,
                 psp_pool_sizes=[1,2,4,8])
    # net = ResSegNet('resnet101', decoder_args=fpn_decoder_args_2)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))


    torch.set_grad_enabled(False)
    if load_gpu:
        input = input.cuda()
    model = model.cuda().eval()
    warmup()

    import tqdm
    
    print("begin:")
    t0 = time()

    for i in tqdm.tqdm(range(epochs)):
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')


if __name__ == "__main__":
    test_predict_speed_pure() 