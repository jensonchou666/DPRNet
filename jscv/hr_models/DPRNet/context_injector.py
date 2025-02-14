from jscv.hr_models.base_models import *



from jscv.models.vit import Attention, AdaptivePoolSelfAttention, WindowDownConvAttention


class AttnResBlocks(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.blocks = nn.ModuleList(models)
    
    def forward(self, x, H, W):
        for m in self.blocks:
            x = x + m(x, H, W)
        return x

class AttnBlocks(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.blocks = nn.ModuleList(models)
    
    def forward(self, x, H, W):
        for m in self.blocks:
            x = m(x, H, W)
        return x


def shortcut(x, short):
    if x.shape == short.shape:
        return x + short
    return x
# # 使用：
# x = shortcut(self.relu(self.norm_sa1(self.fc_sa1(x))), x)


from jscv.models.vit import Win_Down_ViT


class WDViT_CI_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks = [4,4],               # 7x7  14x14
                 sa_channels = [512, 256],      # 7x7  14x14

                 ctx_mapping=512,
                 out_mapping=256,

                 sa_args:dict=dict(
                     window_size=7,
                     down_ratio=7,
                     down_kernel_size=11,
                     num_heads=8,
                     qkv_bias=True,
                     attn_drop=0.1,
                     proj_drop=0.1),
                 
                 sa_args2:dict=dict(
                     window_size=14,
                     down_ratio=7,
                     down_kernel_size=11,
                 ),
                 
                 backbone_feaures_bchw=True,

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            Self-Attention Context Injector + Post Local-Decoder

        '''
        super(WDViT_CI_Decoder, self).__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.ctx_mapping = ctx_mapping
        self.out_mapping = out_mapping
        self.post_decoder_args = post_decoder_args
        self.backbone_feaures_bchw = backbone_feaures_bchw

        self.linear_ctx = nn.Sequential(
            nn.Linear(context_channel, ctx_mapping),
            nn.LayerNorm(ctx_mapping),
            nn.ReLU()
        )

        self.linear_sa1 = nn.Sequential(
            nn.Linear(ctx_mapping+LB4, sa_channels[0]),
            nn.LayerNorm(sa_channels[0]),
            nn.ReLU()
        )
        # self.linear_sa1 = nn.Sequential(
        #     nn.Linear(LB4, sa_channels[0]),
        #     nn.LayerNorm(sa_channels[0]),
        #     nn.ReLU()
        # )

        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(Win_Down_ViT(sa_channels[0], **sa_args))
        self.sa_layer1 = nn.Sequential(*SA_List)

        self.linear_sa2 = nn.Sequential(
            nn.Linear(sa_channels[0]+LB3, sa_channels[1]),
            nn.LayerNorm(sa_channels[1]),
            nn.ReLU()
        )

        SA_List = []
        sa_args = sa_args.copy()
        sa_args.update(sa_args2)
        for k in range(sa_blocks[1]):
            SA_List.append(Win_Down_ViT(sa_channels[1], **sa_args))
        self.sa_layer2 = nn.Sequential(*SA_List)

        self.linear_out = nn.Sequential(
            nn.Linear(sa_channels[1], out_mapping),
            nn.LayerNorm(out_mapping),
            nn.ReLU()
        )

        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )
        
        # self.linear_out2 = nn.Sequential(
        #     nn.Linear(LB3, out_mapping),
        #     nn.LayerNorm(out_mapping),
        #     nn.ReLU()
        # )

    def forward2(self, local_fs, ctx: torch.Tensor):
        '''
            local_fs  BHWC or BCHW
            ctx BCHW
        '''

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        # if self.backbone_feaures_bchw:
        #     LB3 = LB3.permute(0, 2, 3, 1)#.contiguous()
        #     LB4 = LB4.permute(0, 2, 3, 1)#.contiguous()
        # else:
        #     features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        
        # x = self.linear_out2(LB3)

        # print("#################################################")
        return self.post_decoder(*local_fs[:-1])



    def forward(self, local_fs, ctx: torch.Tensor):
        '''
            local_fs  BHWC or BCHW
            ctx BCHW
        '''

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]


        if self.backbone_feaures_bchw:
            LB3 = LB3.permute(0, 2, 3, 1)#.contiguous()
            LB4 = LB4.permute(0, 2, 3, 1)#.contiguous()
        else:
            features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
            # print(LB3.shape, LB4.shape)

        B, H3, W3, C3 = LB3.shape
        _, H4, W4, C4 = LB4.shape
        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)
        ctx = ctx.permute(0, 2, 3, 1)

        ctx = shortcut(self.linear_ctx(ctx), ctx)

        x = torch.concat([ctx, LB4], 3)

        x = shortcut(self.linear_sa1(x), x)

        x = self.sa_layer1(x)

        if H4 != H3 or W4 != W3:
            x = F.interpolate(x.permute(0, 3, 1, 2), (H3, W3), mode='bilinear', align_corners=False)
            x = x.permute(0, 2, 3, 1)
        # print(x.shape, LB3.shape, )
        x = torch.concat([x, LB3], 3)

        x = shortcut(self.linear_sa2(x), x)

        x = self.sa_layer2(x)

        x = shortcut(self.linear_out(x), x)

        features.append(x.permute(0, 3, 1, 2).contiguous())

        return self.post_decoder(*features)


class WDViT_CI_NoCtx_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks = [4,4],             # 7x7  14x14
                 sa_channels = [512, 256],      # 7x7  14x14

                 ctx_mapping=512,
                 out_mapping=256,

                 sa_args:dict=dict(
                     window_size=7,
                     down_ratio=7,
                     down_kernel_size=11,
                     num_heads=8,
                     qkv_bias=True,
                     attn_drop=0.1,
                     proj_drop=0.1),
                 
                 sa_args2:dict=dict(
                     window_size=14,
                     down_ratio=7,
                     down_kernel_size=11,
                 ),
                 
                 backbone_feaures_bchw=True,

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            Self-Attention Context Injector + Post Local-Decoder

        '''
        super(WDViT_CI_NoCtx_Decoder, self).__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.out_mapping = out_mapping
        self.post_decoder_args = post_decoder_args
        self.backbone_feaures_bchw = backbone_feaures_bchw

        self.linear_sa1 = nn.Sequential(
            nn.Linear(LB4, sa_channels[0]),
            nn.LayerNorm(sa_channels[0]),
            nn.ReLU()
        )

        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(Win_Down_ViT(sa_channels[0], **sa_args))
        self.sa_layer1 = nn.Sequential(*SA_List)

        self.linear_sa2 = nn.Sequential(
            nn.Linear(sa_channels[0]+LB3, sa_channels[1]),
            nn.LayerNorm(sa_channels[1]),
            nn.ReLU()
        )

        SA_List = []
        sa_args = sa_args.copy()
        sa_args.update(sa_args2)
        for k in range(sa_blocks[1]):
            SA_List.append(Win_Down_ViT(sa_channels[1], **sa_args))
        self.sa_layer2 = nn.Sequential(*SA_List)

        self.linear_out = nn.Sequential(
            nn.Linear(sa_channels[1], out_mapping),
            nn.LayerNorm(out_mapping),
            nn.ReLU()
        )

        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )
        

    def forward(self, local_fs, ctx: torch.Tensor):
        '''
            local_fs  BHWC or BCHW
            ctx BCHW
        '''

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]


        if self.backbone_feaures_bchw:
            LB3 = LB3.permute(0, 2, 3, 1)#.contiguous()
            LB4 = LB4.permute(0, 2, 3, 1)#.contiguous()
        else:
            features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
            # print(LB3.shape, LB4.shape)

        B, H3, W3, C3 = LB3.shape
        _, H4, W4, C4 = LB4.shape

        x = LB4

        x = shortcut(self.linear_sa1(x), x)

        x = self.sa_layer1(x)

        if H4 != H3 or W4 != W3:
            x = F.interpolate(x.permute(0, 3, 1, 2), (H3, W3), mode='bilinear', align_corners=False)
            x = x.permute(0, 2, 3, 1)
        # print(x.shape, LB3.shape, )
        x = torch.concat([x, LB3], 3)

        x = shortcut(self.linear_sa2(x), x)

        x = self.sa_layer2(x)

        x = shortcut(self.linear_out(x), x)

        features.append(x.permute(0, 3, 1, 2).contiguous())

        return self.post_decoder(*features)



class Conv_CI_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 ctx_mapping=512,
                 mid_channels = [512, 256],      # 7x7  14x14
                 out_mapping=256,

                 backbone_feaures_bchw=True,

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            Self-Attention Context Injector + Post Local-Decoder

        '''
        super(Conv_CI_Decoder, self).__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.ctx_mapping = ctx_mapping
        self.out_mapping = out_mapping
        self.post_decoder_args = post_decoder_args
        assert backbone_feaures_bchw
        self.backbone_feaures_bchw = backbone_feaures_bchw

        self.linear_ctx = ConvBNReLU(context_channel, ctx_mapping, 3)

        self.linear_sa1 = ConvBNReLU(ctx_mapping+LB4, mid_channels[0], 3)
        
        L = []
        for i in range(6):
            L.append(ConvBNReLU(mid_channels[0], mid_channels[0], 3))
        self.mid1 = ResBlocks(*L)


        self.linear_sa2 = ConvBNReLU(mid_channels[0]+LB3, mid_channels[1], 3)

        L = []
        for i in range(6):
            L.append(ConvBNReLU(mid_channels[1], mid_channels[1], 3))
        self.mid2 = ResBlocks(*L)

        self.linear_out = ConvBNReLU(mid_channels[1], out_mapping, 3)

        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )


    def forward(self, local_fs, ctx: torch.Tensor):
        '''
            local_fs  BHWC or BCHW
            ctx BCHW
        '''

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape
        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)

        ctx = shortcut(self.linear_ctx(ctx), ctx)

        x = torch.concat([ctx, LB4], 1)

        x = shortcut(self.linear_sa1(x), x)

        x = self.mid1(x)

        if H4 != H3 or W4 != W3:
            x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)
        x = torch.concat([x, LB3], 1)

        x = shortcut(self.linear_sa2(x), x)

        x = self.mid2(x)

        x = shortcut(self.linear_out(x), x)

        features.append(x)

        return self.post_decoder(*features)



'''
    问题：  PVT作局部主干时,  用 V1 导致训练问题
    故换成这个
'''
class WDViT_CI_Decoder_V2(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks = [4,4],               # 7x7  14x14
                 sa_channels = [512, 256],      # 7x7  14x14

                 ctx_mapping=512,
                 out_mapping=256,

                 sa_args:dict=dict(
                     window_size=7,
                     down_ratio=7,
                     down_kernel_size=11,
                     num_heads=8,
                     qkv_bias=True,
                     attn_drop=0.1,
                     proj_drop=0.1),
                 
                 sa_args2:dict=dict(
                     window_size=14,
                     down_ratio=7,
                     down_kernel_size=11,
                 ),

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            Self-Attention Context Injector + Post Local-Decoder

        '''
        super(WDViT_CI_Decoder, self).__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.ctx_mapping = ctx_mapping
        self.out_mapping = out_mapping
        self.post_decoder_args = post_decoder_args
        
        self.proj_ctx = nn.Sequential(
            nn.Conv2d(context_channel, ctx_mapping, 3, 1, 1),
            nn.BatchNorm2d(ctx_mapping),
            nn.ReLU()
        )

        self.proj_sa1 = nn.Sequential(
            nn.Conv2d(ctx_mapping+LB4, sa_channels[0], 3, 1, 1),
            nn.BatchNorm2d(sa_channels[0]),
            nn.ReLU()
        )

        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(Win_Down_ViT(sa_channels[0], **sa_args))
        self.sa_layer1 = nn.Sequential(*SA_List)

        self.proj_sa2 = nn.Sequential(
            nn.Conv2d(sa_channels[0]+LB3, sa_channels[1], 3, 1, 1),
            nn.BatchNorm2d(sa_channels[1]),
            nn.ReLU()
        )

        SA_List = []
        sa_args = sa_args.copy()
        sa_args.update(sa_args2)
        for k in range(sa_blocks[1]):
            SA_List.append(Win_Down_ViT(sa_channels[1], **sa_args))
        self.sa_layer2 = nn.Sequential(*SA_List)

        self.proj_out = nn.Sequential(
            nn.Linear(sa_channels[1], out_mapping),
            nn.LayerNorm(out_mapping),
            nn.ReLU()
        )

        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )


    def forward(self, local_fs, ctx: torch.Tensor):

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape


        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)

        ctx = shortcut(self.proj_ctx(ctx), ctx)
        x = torch.concat([ctx, LB4], 1)
        x = shortcut(self.proj_sa1(x), x)

        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.sa_layer1(x)

        if H4 != H3 or W4 != W3:
            x = F.interpolate(x.permute(0, 3, 1, 2).contiguous(), (H3, W3), mode='bilinear', align_corners=False)
        x = torch.concat([x, LB3], 1)

        x = shortcut(self.proj_sa2(x), x)
        
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.sa_layer2(x)

        x = shortcut(self.proj_out(x), x)

        features.append(x.permute(0, 3, 1, 2).contiguous())

        return self.post_decoder(*features)





from .aspp import MultiScaleASPP

'''
设计一个卷积Block， 输入时多种尺度的特征，从7x7到14x28到56x56都可能（7x7相当于一个最小patch， 56x56相当于8x8个patch），
有两个互相并行的路线， 一路用1、2、4的ASPP建模局部， 另一路7倍下采样，然后三层3x3卷积
'''

''' Window + Downsample-CNN '''
class WDCI_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks=[3,3],               # 7x7  14x14
                 sa_channels = [1024, 512],      # 7x7  14x14

                 ctx_mapping=512,
                 mid_mapping=512,
                 out_mapping=256,

                 sa_args:dict=dict(
                     window_size=7,
                     downsample_rate=7,
                     conv_layers=2,
                     num_heads=8,
                     cfg_norm={'type': nn.GroupNorm, 'args': dict(num_groups=8)},
                     act_type=nn.ReLU,
                     qkv_bias=True,
                     attn_drop=0.1,
                     proj_drop=0.1),
                 
                 sa_args2:dict=dict(
                     conv_layers=3,
                 ),

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            Self-Attention Context Injector + Post Local-Decoder

        '''
        super(WDCI_Decoder, self).__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.ctx_mapping = ctx_mapping
        self.mid_mapping = mid_mapping
        self.out_mapping = out_mapping
        self.post_decoder_args = post_decoder_args
        
        # self.fc_ctx = nn.Linear(context_channel, ctx_mapping)
        # self.norm_ctx = nn.LayerNorm(ctx_mapping)
        self.layer_ctx = nn.Sequential(
            nn.Conv2d(context_channel, ctx_mapping, 1),
            nn.ReLU()
        )
        self.layer_sa1 = nn.Sequential(
            nn.Conv2d(ctx_mapping+LB4, sa_channels[0], 1),
            nn.ReLU()
        )

        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(WindowDownConvAttention(dim=sa_channels[0], **sa_args))
        self.sa_layer1 = nn.Sequential(*SA_List)

        self.layer_mid = nn.Sequential(
            nn.Conv2d(sa_channels[0], mid_mapping, 1),
            nn.BatchNorm2d(mid_mapping),
            nn.ReLU()
        )
        self.layer_sa2 = nn.Sequential(
            nn.Conv2d(mid_mapping+LB3, sa_channels[1], 1),
            nn.BatchNorm2d(sa_channels[1]),
            nn.ReLU()
        )
        SA_List = []
        sa_args = sa_args.copy()
        sa_args.update(sa_args2)
        for k in range(sa_blocks[1]):
            SA_List.append(WindowDownConvAttention(dim=sa_channels[1], **sa_args))
        self.sa_layer2 = nn.Sequential(*SA_List)

        self.layer_out = nn.Sequential(
            nn.Conv2d(sa_channels[1], out_mapping, 1),
            nn.BatchNorm2d(out_mapping),
            nn.ReLU()
        )

        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )


    def forward(self, local_fs, ctx: torch.Tensor):

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape

        ctx = shortcut(self.layer_ctx(ctx), ctx)

        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)

        x = torch.concat([ctx, LB4], 1)

        x = shortcut(self.layer_sa1(x), x)

        x = self.sa_layer1(x)

        x = shortcut(self.layer_mid(x), x)

        if x.shape[-2:] != (H3, W3):
            x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)

        x = torch.concat([x, LB3], 1)

        x = shortcut(self.layer_sa2(x), x)

        x = self.sa_layer2(x)

        x = shortcut(self.layer_out(x), x)

        features.append(x)

        # for f in features:
        #     print(f.shape)

        return self.post_decoder(*features)








class AsppCI_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)
                 aspp_args=[
                     dict(channels=512, dilation_rates=[1, 3, 7, 11], num_blocks=4, groups=8, se_reduction=16),
                     dict(channels=256, dilation_rates=[1, 6, 12, 18], num_blocks=3, groups=8, se_reduction=16),
                 ],
                 ctx_mapping=512,
                 out_mapping=256,
                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        super(AsppCI_Decoder, self).__init__()
        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.layer_ctx = nn.Sequential(
            nn.Conv2d(context_channel, ctx_mapping, 1),
            nn.GroupNorm(8, ctx_mapping),
            nn.ReLU()
        )
        self.aspp_layer1 = MultiScaleASPP(in_channels=(LB4+ctx_mapping), **aspp_args[0])
        self.aspp_layer2 = MultiScaleASPP(in_channels=(LB3+aspp_args[0]['channels']), **aspp_args[1])

        self.layer_out = nn.Sequential(
            nn.Conv2d(aspp_args[1]['channels'], out_mapping, 1),
            nn.GroupNorm(8, out_mapping),
            nn.ReLU()
        )
        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )

    def forward(self, local_fs, ctx: torch.Tensor):

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape

        ctx = shortcut(self.layer_ctx(ctx), ctx)
        
        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)
        x = torch.concat([ctx, LB4], 1)
        
        x = self.aspp_layer1(x)
        
        if x.shape[-2:] != (H3, W3):
            x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)
        x = torch.concat([x, LB3], 1)

        x = self.aspp_layer2(x)
        x = shortcut(self.layer_out(x), x)

        features.append(x)

        return self.post_decoder(*features)


''' V2 '''
class ASACI_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks=[3,3],               # 7x7  14x14
                 sa_channels = [1024, 512],      # 7x7  14x14
                 pool_size = [(7,7), (14,14)],  # SA的KV,  L1和L2分别池化到 (7,7) (14,14)

                 ctx_mapping=512,
                 mid_mapping=512,
                 out_mapping=256,

                 sa_args=dict(num_heads=8,
                              qkv_bias=True,
                              attn_drop=0.1,
                              proj_drop=0.1),

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            AdaptivePool-Self-Attention Context Injector + Post Local-Decoder
        '''
        super().__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.ctx_mapping = ctx_mapping
        self.mid_mapping = mid_mapping
        self.out_mapping = out_mapping
        self.sa_args = sa_args
        self.post_decoder_args = post_decoder_args
        
        # self.fc_ctx = nn.Linear(context_channel, ctx_mapping)
        # self.norm_ctx = nn.LayerNorm(ctx_mapping)
        self.layer_ctx = nn.Sequential(
            nn.Conv2d(context_channel, ctx_mapping, 1),
            nn.BatchNorm2d(ctx_mapping),
            nn.ReLU()
        )
        self.layer_sa1 = nn.Sequential(
            nn.Conv2d(ctx_mapping+LB4, sa_channels[0], 1),
            nn.BatchNorm2d(sa_channels[0]),
            nn.ReLU()
        )
        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(AdaptivePoolSelfAttention(dim=sa_channels[0], pool_size=pool_size[0], **sa_args))
        self.sa_layer1 = nn.Sequential(*SA_List)

        self.layer_mid = nn.Sequential(
            nn.Conv2d(sa_channels[0], mid_mapping, 1),
            nn.BatchNorm2d(mid_mapping),
            nn.ReLU()
        )
        self.layer_sa2 = nn.Sequential(
            nn.Conv2d(mid_mapping+LB3, sa_channels[1], 1),
            nn.BatchNorm2d(sa_channels[1]),
            nn.ReLU()
        )
        SA_List = []
        for k in range(sa_blocks[1]):
            SA_List.append(AdaptivePoolSelfAttention(dim=sa_channels[1], pool_size=pool_size[1], **sa_args))
        self.sa_layer2 = nn.Sequential(*SA_List)

        self.layer_out = nn.Sequential(
            nn.Conv2d(sa_channels[1], out_mapping, 1),
            nn.BatchNorm2d(out_mapping),
            nn.ReLU()
        )

        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )


    def forward(self, local_fs, ctx: torch.Tensor):
        
        # for fff in local_fs:
        #     print(fff.shape)
        # print(ctx.shape)
        
        # local_fs = list(local_fs)

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape

        ctx = shortcut(self.layer_ctx(ctx), ctx)

        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)

        x = torch.concat([ctx, LB4], 1)

        x = shortcut(self.layer_sa1(x), x)

        x = self.sa_layer1(x)
        
        x = shortcut(self.layer_mid(x), x)

        if x.shape[-2:] != (H3, W3):
            x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)

        x = torch.concat([x, LB3], 1)

        x = shortcut(self.layer_sa2(x), x)

        x = self.sa_layer2(x)

        x = shortcut(self.layer_out(x), x)

        features.append(x)

        # for f in features:
        #     print(f.shape)

        return self.post_decoder(*features)




################################################################
#? SACI + LocalDecodeHead
class SACI_Decoder(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks=[3,3],               # 7x7  14x14
                 sa_channels = [1024, 512],      # 7x7  14x14
                 

                 ctx_mapping=512,
                 mid_mapping=512,
                 out_mapping=256,

                 sa_args=dict(num_heads=8,
                              qkv_bias=True,
                              attn_drop=0.1,
                              proj_drop=0.1),

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            Self-Attention Context Injector + Post Local-Decoder

        '''
        super().__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.ctx_mapping = ctx_mapping
        self.mid_mapping = mid_mapping
        self.out_mapping = out_mapping
        self.sa_args = sa_args
        self.post_decoder_args = post_decoder_args
        

        # self.fc_ctx = nn.Linear(context_channel, ctx_mapping)
        # self.norm_ctx = nn.LayerNorm(ctx_mapping)
        self.conv_ctx = nn.Conv2d(context_channel, ctx_mapping, 1)
        self.norm_ctx = nn.BatchNorm2d(ctx_mapping)
        
        self.fc_sa1 = nn.Linear(ctx_mapping+LB4, sa_channels[0])
        self.norm_sa1 = nn.LayerNorm(sa_channels[0])

        self.fc_mid = nn.Linear(sa_channels[0], mid_mapping)
        self.norm_mid = nn.LayerNorm(mid_mapping)

        self.fc_sa2 = nn.Linear(mid_mapping+LB3, sa_channels[1])
        self.norm_sa2 = nn.LayerNorm(sa_channels[1])
        
        self.fc_out = nn.Linear(sa_channels[1], out_mapping)
        self.norm_out = nn.LayerNorm(out_mapping)

        self.relu = nn.ReLU6()
        self.relu = nn.ReLU()


        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(Attention(sa_channels[0], **sa_args))
        self.sa_layer1 = nn.Sequential(*SA_List)

        SA_List = []
        for k in range(sa_blocks[1]):
            SA_List.append(Attention(sa_channels[1], **sa_args))
        self.sa_layer2 = nn.Sequential(*SA_List)


        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )


    def forward(self, local_fs, ctx: torch.Tensor):
        
        # for fff in local_fs:
        #     print(fff.shape)
        # print(ctx.shape)
        
        # local_fs = list(local_fs)
        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape

        ctx = shortcut(self.relu(self.norm_ctx(self.conv_ctx(ctx))), ctx)

        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)

        x = torch.concat([ctx, LB4], 1).reshape(B, self.ctx_mapping+C4, -1).transpose(1,2).contiguous()

        x = shortcut(self.relu(self.norm_sa1(self.fc_sa1(x))), x)

        x = self.sa_layer1(x)
        
        x = shortcut(self.relu(self.norm_mid(self.fc_mid(x))), x)
        
        x = x.transpose(1,2).reshape(B, self.mid_mapping, H4, W4)

        x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)
        
        x = torch.concat([x, LB3], 1).reshape(B, self.mid_mapping+C3, -1).transpose(1,2).contiguous()

        x = shortcut(self.relu(self.norm_sa2(self.fc_sa2(x))), x)

        x = self.sa_layer2(x)
        
        x = shortcut(self.relu(self.norm_out(self.fc_out(x))), x)
        
        x = x.transpose(1,2).reshape(B, self.out_mapping, H3, W3).contiguous()

        features.append(x)
        
        # for f in features:
        #     print(f.shape)

        return self.post_decoder(*features)


class ASACI_Decoder_V1(nn.Module):
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks=[3,3],               # 7x7  14x14
                 sa_channels = [1024, 512],      # 7x7  14x14
                 kv_pool_size = [(7,7), (14,14)],  # SA的KV,  L1和L2分别池化到 (7,7) (14,14)

                 ctx_mapping=512,
                 mid_mapping=512,
                 out_mapping=256,

                 sa_args=dict(num_heads=8,
                              qkv_bias=True,
                              attn_drop=0.1,
                              proj_drop=0.1),

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={},
                 ):
        '''
            AdaptivePool-Self-Attention Context Injector + Post Local-Decoder
        '''
        super().__init__()

        LB3, LB4 = local_encoder_channels[-2], local_encoder_channels[-1]
        
        self.context_channel = context_channel
        self.local_encoder_channels = local_encoder_channels
        self.sa_blocks = sa_blocks
        self.sa_channels = sa_channels
        self.ctx_mapping = ctx_mapping
        self.mid_mapping = mid_mapping
        self.out_mapping = out_mapping
        self.sa_args = sa_args
        self.post_decoder_args = post_decoder_args
        

        # self.fc_ctx = nn.Linear(context_channel, ctx_mapping)
        # self.norm_ctx = nn.LayerNorm(ctx_mapping)
        self.conv_ctx = nn.Conv2d(context_channel, ctx_mapping, 1)
        self.norm_ctx = nn.BatchNorm2d(ctx_mapping)
        
        self.fc_sa1 = nn.Linear(ctx_mapping+LB4, sa_channels[0])
        self.norm_sa1 = nn.LayerNorm(sa_channels[0])

        self.fc_mid = nn.Linear(sa_channels[0], mid_mapping)
        self.norm_mid = nn.LayerNorm(mid_mapping)

        self.fc_sa2 = nn.Linear(mid_mapping+LB3, sa_channels[1])
        self.norm_sa2 = nn.LayerNorm(sa_channels[1])
        
        self.fc_out = nn.Linear(sa_channels[1], out_mapping)
        self.norm_out = nn.LayerNorm(out_mapping)

        self.relu = nn.ReLU6()
        self.relu = nn.ReLU()


        SA_List = []
        for k in range(sa_blocks[0]):
            SA_List.append(AdaptivePoolSelfAttention(dim=sa_channels[0], kv_pool_size=kv_pool_size[0], **sa_args))
        self.sa_layer1 = AttnBlocks(*SA_List)

        SA_List = []
        for k in range(sa_blocks[1]):
            SA_List.append(AdaptivePoolSelfAttention(dim=sa_channels[1], kv_pool_size=kv_pool_size[1], **sa_args))
        self.sa_layer2 = AttnBlocks(*SA_List)


        channels = local_encoder_channels[:-2]
        channels.append(out_mapping)

        self.post_decoder=post_decoder_cls(
            channels,
            **post_decoder_args
        )


    def forward(self, local_fs, ctx: torch.Tensor):
        
        # for fff in local_fs:
        #     print(fff.shape)
        # print(ctx.shape)
        
        # local_fs = list(local_fs)

        features = list(local_fs[:-2])
        LB3, LB4 = local_fs[-2:]

        B, C3, H3, W3 = LB3.shape
        _, C4, H4, W4 = LB4.shape

        ctx = shortcut(self.relu(self.norm_ctx(self.conv_ctx(ctx))), ctx)

        if ctx.shape[-2:] != (H4, W4):
            ctx = F.interpolate(ctx, (H4, W4), mode='bilinear', align_corners=False)

        x = torch.concat([ctx, LB4], 1).reshape(B, self.ctx_mapping+C4, -1).transpose(1,2).contiguous()

        x = shortcut(self.relu(self.norm_sa1(self.fc_sa1(x))), x)

        x = self.sa_layer1(x, H4, W4)
        
        x = shortcut(self.relu(self.norm_mid(self.fc_mid(x))), x)
        
        x = x.transpose(1,2).reshape(B, self.mid_mapping, H4, W4).contiguous()

        if x.shape[-2:] != (H3, W3):
            x = F.interpolate(x, (H3, W3), mode='bilinear', align_corners=False)

        x = torch.concat([x, LB3], 1).reshape(B, self.mid_mapping+C3, -1).transpose(1,2).contiguous()

        x = shortcut(self.relu(self.norm_sa2(self.fc_sa2(x))), x)

        x = self.sa_layer2(x, H3, W3)
        
        x = shortcut(self.relu(self.norm_out(self.fc_out(x))), x)
        
        x = x.transpose(1,2).reshape(B, self.out_mapping, H3, W3).contiguous()

        features.append(x)
        
        # for f in features:
        #     print(f.shape)

        return self.post_decoder(*features)


'''
    SCAI None-Global Decoder
'''
class SACI_NG_Decoder(SACI_Decoder):
    
    def __init__(self, 
                 context_channel,               # F_GD2
                 local_encoder_channels,        # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)

                 sa_blocks=[3,3],               # 7x7  14x14
                 sa_channels = [1024, 512],      # 7x7  14x14

                 ctx_mapping=512,
                 mid_mapping=512,
                 out_mapping=256,

                 sa_args=dict(num_heads=8, 
                              qkv_bias=True,
                              attn_drop=0.1,
                              proj_drop=0.1),

                 post_decoder_cls=PostDecodeHead,
                 post_decoder_args={}):
        super().__init__(context_channel, local_encoder_channels, sa_blocks,
            sa_channels, ctx_mapping, mid_mapping, out_mapping,
            sa_args, post_decoder_cls, post_decoder_args
        )

    def forward(self, local_fs):
        ''' 直接生成一个 zero ctx'''
        LB4 = local_fs[-1]
        # assert isinstance(LB4, torch.Tensor)
        B, C, H, W = LB4.shape
        ctx = torch.zeros(size=(B, self.context_channel, H, W), device=LB4.device)
        return super().forward(local_fs, ctx)




#? 弃用 简单的Global-Local Feature Fusion Module (GLFFM) + FPNDecoder,  GLFFM 只使用了2个卷积层
class GL_FFN(nn.Module):
    def __init__(self, 
                 global_channels,   # G-Branch-Decoder (2)
                 local_channels,    # G-Branch-Encoder (0,1,2,3,4)
                 blocks=2,
                 channel_ratio=1/2,
                 post_decoder_cls=FPNDecoder,
                 decoder_args={},
                 ):
        '''global: [stride=32]
            local: [stride=2, stride=4, stride=8, stride=16, stride=32]
        '''
        super().__init__()
        C2_g = global_channels
        C4_l = local_channels[-1]
        self.blocks = blocks
        
        C4 = C4_l+C2_g
        outc = int(C4*channel_ratio)

        if blocks > 0:
            self.layer1 = []
            for k in range(blocks):
                inc = C4 if k==0 else outc
                self.layer1.append(ConvBNReLU(inc, outc, 3))
            self.layer1 = ResBlocks(*self.layer1)

        local_channels[-1] = outc

        self.pose_decoder=post_decoder_cls(
            local_channels,
            **decoder_args
        )

    def forward(self, g_fs, l_fs):
        l_fs = list(l_fs)

        f4 = l_fs[-1]
        # print(f4.shape, g_fs.shape)
        f4 = torch.concat([f4, g_fs], dim=1)
        if self.blocks > 0:
            f4 = self.layer1(f4)
        l_fs[-1] = f4
        return self.pose_decoder(*l_fs)



#? 弃用
class SA_GL_FFN(nn.Module):
    def __init__(self, 
                 global_channels,   # G-Branch-Decoder (2)
                 local_channels,    # G-Branch-Encoder (0,1,2,3,4) or (1,2,3,4)
                 blocks=4,
                 
                 global_channel_to=512,
                 local_channel_to=512,

                 self_attn_args=dict(num_heads=8),

                 post_decoder_cls=FPNDecoder,
                 decoder_args={},
                 ):
        '''
            自注意力

            global: [stride=32]
            local: [stride=2, stride=4, stride=8, stride=16, stride=32]
        '''
        super().__init__()
        C2_g = global_channels

        C4_l = local_channels[-1]

        self.block_nums = blocks
        
        
        self.GConv = ConvBNReLU(C2_g, global_channel_to, 3)
        self.LConv = ConvBNReLU(C4_l, local_channel_to, 3)

        channel = global_channel_to + local_channel_to


        # print("#########", channel)


        if blocks > 0:
            
            self.self_attn_layers = []

            for k in range(blocks):

                #num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., sr_ratio=1)
                
                self.self_attn_layers.append(Attention(channel, **self_attn_args))


            self.self_attn_layers = AttnResBlocks(*self.self_attn_layers)

        local_channels[-1] = channel

        self.pose_decoder=post_decoder_cls(
            local_channels,
            **decoder_args
        )

    def forward(self, gd2, l_fs):
        
        l_fs = list(l_fs)
        f4 = l_fs[-1]
        
        # print("@@@@@@@@", f4.shape, gd2.shape)
        
        gd2 = self.GConv(gd2)
        f4 = self.LConv(f4)


        if f4.shape[2:] != gd2.shape[2]:
            gd2 = F.interpolate(gd2, f4.shape[2:], mode='bilinear', align_corners=False)

        f4 = torch.concat([f4, gd2], dim=1)
        # print("@@@@@@@@", f4.shape)


        B,C,H,W = f4.shape
        N = H*W

        if self.block_nums > 0:
            f4 = f4.reshape(B, C, N).transpose(1, 2).contiguous()   # [B,C,H,W] -> [B,N,C]
            f4 = self.self_attn_layers(f4, H, W)
            f4 = f4.transpose(1, 2).reshape(B, C, H, W).contiguous()   #  [B,N,C] -> [B,C,H,W]


        l_fs[-1] = f4

        return self.pose_decoder(*l_fs)

