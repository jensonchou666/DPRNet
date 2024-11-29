from jscv.hr_models.base_models import *


class ResNetDEncoder(ResNetEncoder):
    def __init__(self, **args):
        super().__init__(**args)
        output_stride = 8
        if output_stride == 16: s3, s4, d3, d4 = (1, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (d3,d3), getPadding(3,1,d3)
            # elif 'downsample.0' in n:
            #     m.stride = (s3, s3)
 
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (d4,d4), getPadding(3,1,d4)
            # elif 'downsample.0' in n:
            #     m.stride = (s4, s4)



def aspp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP, self).__init__()
 
        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = aspp_branch(in_channels, out_channels, 1, dilation=dilations[0])
        self.aspp2 = aspp_branch(in_channels, out_channels, 3, dilation=dilations[1])
        self.aspp3 = aspp_branch(in_channels, out_channels, 3, dilation=dilations[2])
        self.aspp4 = aspp_branch(in_channels, out_channels, 3, dilation=dilations[3])
 
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
 
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
 
        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
 
        return x




class Deeplabv3(EncoderDecoder):
    def __init__(self,
                 backbone_name='resnet50',
                 decoder_class=FPNDecoder,
                 backbone_args={},
                 decoder_args=fpn_decoder_args_1,
                 features_only=True,
                 aspp_out_channels_rate=1/2,
                 aspp_layers=2,
    ):
        backbone = ResNetEncoder(backbone_name, features_only=features_only, **backbone_args)
        l_chs = backbone.channels
        aspp_inc = l_chs[-1]
        aspp_outc = int(aspp_inc * aspp_out_channels_rate)
        
        l_chs[-1] = aspp_outc
        decoder = decoder_class(l_chs, **decoder_args)
        super().__init__(backbone, decoder)

        L = []
        for i in range(aspp_layers):
            inc = aspp_inc if i == 0 else aspp_outc
            L.append(ASPP(inc, aspp_outc))
        self.aspp = nn.Sequential(*L)

    def forward(self, x, reture_features=False):
        fs = list(self.backbone(x))
        fs[-1] = self.aspp(fs[-1])
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

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    epochs = 20
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    net = Deeplabv3('resnet101', decoder_args=fpn_decoder_args_1,
                    aspp_out_channels_rate=1/4, aspp_layers=4)
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