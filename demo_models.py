import torch

'''
    用来简单测试模型, debug, 看中间特征的shape
'''

def demo_ISDNet():
    from jscv.hr_models.ISDNet.isdnet import ISDHead

    import os



    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method='env://')

    from jscv.losses.useful_loss import SCE_DIce_Loss

    kwargs_ISDHead = dict(
        in_channels=3,
        channels=128,
        num_classes=2,
        dropout_ratio=0.1,
        pretrain_path=None,
        norm_cfg={},
        align_corners=False,
        loss_layer=SCE_DIce_Loss()
    )
    # print(kwargs_ISDHead)


    img = torch.randn(2, 3, 512, 512).float().cuda()
    label = torch.randint(0, 2, (2, 512,512)).cuda()
    f = torch.randn(2, 128, 128//8,128//8).float().cuda()
    # print(label)
    # label = F.one_hot(label, num_classes=2).permute(0, 3, 1, 2)

    ISDH = ISDHead(**kwargs_ISDHead).cuda()
    ret = ISDH.forward_train(img, f, label)
    print(ret[1])

    # ret = ISDH.forward_test(img, f)
    # print(ret)


def demo_ISDNet2():
    from jscv.hr_models.ISDNet.double_branch_net import test_predict_speed_pure
    test_predict_speed_pure()
# demo_ISDNet2()
# exit()


def demo_RMT():
    from jscv.backbone.RMT import RMT_S
    
    H = 1024*7//4   
    
    # torch.Size([2, 3, 1792, 1792])
    img = torch.randn(2, 3, H, H).float().cuda()
    
    
    model = RMT_S().cuda()
    # print(model)
    ret = model(img)
    for f in ret:
        print(f.shape)

# demo_RMT()
# exit()

def demo_PVT():
    from config._base_.backbone.pvtv2 import backbone_pvtv2_b0 as pvtv2
    from jscv.hr_models.base_models import FPNDecoder

    # from config._base_.backbone.pvtv2 import backbone_pvtv2_b1 as pvtv2

    encoder = pvtv2()
    decoder = FPNDecoder(encoder.embed_dims)

    H = 1024
    img = torch.randn(2, 3, H, H).float()

    features = encoder(img)
    x = decoder(*features)

    for r in features:
        print(r.shape)
    print(x.shape)



def demo_FocalNet():
    from config._base_.backbone.focalnet import Focal_Tiny
    from jscv.hr_models.base_models import FPNDecoder
    
    encoder = Focal_Tiny()[0]
    decoder = FPNDecoder(encoder.num_features)

    print("num_features:", encoder.num_features)

    H = 1024
    img = torch.randn(2, 3, H, H).float()

    features = encoder(img)
    x = decoder(*features)

    for r in features:
        print(r.shape)
    print(x.shape)



def demo_DPRNet():
    from jscv.hr_models.DPRNet import test_predict_speed_pure_PVTPVT_SAFFM, test_predict_speed_pure_PVTFocalT_SAFFM
    # test_predict_speed_pure_PVTPVT_SAFFM()
    test_predict_speed_pure_PVTFocalT_SAFFM()



from jscv.hr_models.pathes_segmentor import *

def demo_pathes_segmentor():
    # test_predict_speed_pure_pvt()
    test_predict_speed_pure_RMT()



def demo_efficientnet(name='efficientnet-b3', pretrain='pretrain/efficientnet-b3-5fb5a3c3.pth'):
    x = torch.randn(1, 3, 224, 224)
    from jscv.backbone.efficientnet.efficientnet import EfficientNet
    model = EfficientNet.from_pretrained(name, pretrain)
    model.extract_endpoints_print(x)
    L = model(x)
    for v in L:
        print(f'{v.shape}')


# demo_efficientnet('efficientnet-b3', 'pretrain/efficientnet-b3-5fb5a3c3.pth')
# demo_efficientnet('efficientnet-b6', 'pretrain/efficientnet-b6-c76e70fd.pth')


# demo_pathes_segmentor()

# demo_FocalNet()

# demo_DPRNet()

# demo_PVT()
    
# demo_ISDNet()

# demo_RMT()


# from jscv.hr_models.DPRNet.DPRNet import *

# DPRNet(None, None, None)