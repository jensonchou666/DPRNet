from jscv.hr_models.base_models import *


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        #创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        b,c,h,w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h*w).transpose(1, 2), C.view(b, c, h*w)))
        E = torch.matmul(D.view(b, c, h*w), S.transpose(1, 2)).view(b,c,h,w)
        #gamma is a parameter which can be training and iter
        E = self.gamma * E + x
        
        return E
    
class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        b,c,h,w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h*w)).view(b, c, h, w)
        X = self.beta * X + x
        return X
    
class DAHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels//4, in_channels//8, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channels//8),
        #     nn.ReLU(),
        # )
 
        self.PositionAttention = PositionAttention(out_channels)
        self.ChannelAttention = ChannelAttention()
        
    def forward(self, x):
        x_PA = self.conv1(x)
        x_CA = self.conv2(x)
        PosionAttentionMap = self.PositionAttention(x_PA)
        ChannelAttentionMap = self.ChannelAttention(x_CA)
        #这里可以额外分别做PAM和CAM的卷积输出,分别对两个分支做一个上采样和预测, 
        #可以生成一个cam loss和pam loss以及最终融合后的结果的loss.以及做一些可视化工作
        #这里只输出了最终的融合结果.与原文有一些出入.
        output = self.conv3(PosionAttentionMap + ChannelAttentionMap)
        return output


class DANet(EncoderDecoder):
    def __init__(self,
                 backbone_name='resnet50',
                 decoder_class=FPNDecoder,
                 backbone_args={},
                 decoder_args=fpn_decoder_args_1,
                 features_only=True,
                 da_out_channels_rate=1/2,
                 da_layers=2,
    ):
        backbone = ResNetEncoder(backbone_name, features_only=features_only, **backbone_args)
        l_chs = backbone.channels
        f4_inc = l_chs[-1]
        f4_outc = int(f4_inc * da_out_channels_rate)
        l_chs[-1] = f4_outc
        
        decoder = decoder_class(l_chs, **decoder_args)
        super().__init__(backbone, decoder)

        L = []
        for i in range(da_layers):
            inc = f4_inc if i == 0 else f4_outc
            L.append(DAHead(inc, f4_outc))
            # print("@@@ ResBlocks", da_layers)
        self.da_head = nn.Sequential(*L)


    def forward(self, x, reture_features=False):
        fs = list(self.backbone(x))
        fs[-1] = self.da_head(fs[-1])
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
    
    net = DANet('resnet101', decoder_args=fpn_decoder_args_1,
                da_out_channels_rate=1/4, da_layers=4)
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
    # net = DANet('resnet101', decoder_args=fpn_decoder_args_2, num_classes=2)
    # x = net(torch.rand(1, 3, 1024, 1024))
    # print(x.shape)