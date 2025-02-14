import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from timm.models.layers import trunc_normal_

def drop_layer(drop_rate=0.):
    return nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()


def getPadding(kernel_size, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def get_padding(kernel_size, down_ratio):
    # 计算padding，要求 kernel_size > down_ratio
    if kernel_size <= down_ratio:
        raise ValueError("kernel_size must be greater than down_ratio")
    # 计算 padding
    padding = (kernel_size - down_ratio) // 2
    return padding


class Win_Down_ViT(nn.Module):
    ''' Win + Down ViT '''
    def __init__(self,
                 channels,
                 window_size=7,
                 down_ratio=7,
                 down_kernel_size=11,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0,
                 proj_drop=0.,
                 ):
        super().__init__()
        C = self.channels = channels
        self.window_size = window_size
        self.down_ratio = down_ratio
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.scale = (C//num_heads)**-0.5
        assert down_ratio > 1
        # print(down_kernel_size, down_ratio, get_padding(down_kernel_size, down_ratio))
        self.down_conv = nn.Conv2d(C, C, down_kernel_size, down_ratio, get_padding(down_kernel_size, down_ratio), groups=C)
        self.down_norm = nn.LayerNorm(C)
        self.act = nn.GELU()

        self.qkv_l = nn.Linear(C, C * 3, bias=qkv_bias)
        self.win_norm = nn.LayerNorm(C)
        self.attn_drop = drop_layer(attn_drop)

        self.q_g = nn.Linear(C, C, bias=qkv_bias)
        self.kv_g = nn.Linear(C, C * 2, bias=qkv_bias)
        self.proj = nn.Linear(C, C)
        self.proj_norm = nn.LayerNorm(C)
        self.proj_drop = drop_layer(proj_drop)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        """
            feature_map:      B, H, W, C
        """
        # count1.begin()
        wsz = self.window_size
        down_ratio = self.down_ratio
        num_heads = self.num_heads
        
        B, H, W, C = x.shape
        Ch = C//num_heads
        H1, W1 = H//down_ratio, W//down_ratio
        hw, hw1 = H*W, H1*W1
        nH, nW = H//wsz, W//wsz
        nWin = nH*nW
        winsz = wsz*wsz
        
        ''' local branch'''
        # print(x.shape, B, nH, wsz, nW, wsz, C)
        win_x = x.reshape(B, nH, wsz, nW, wsz, C).transpose(2, 3).reshape(B*nWin, winsz, C)

        qkv = self.qkv_l(win_x).reshape(B*nWin, winsz, 3, num_heads, Ch).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # print('window', q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B*nWin, winsz, C)
        win_x = win_x + self.act(self.win_norm(attn))
        #TODO 加线性层

        x = win_x.reshape(B, nH, nW, wsz, wsz, C).transpose(2, 3).reshape(B, H, W, C)
        # count1.record_time('local')

        ''' global branch'''
        # print(x.shape, x.permute(0,3,1,2).shape, self.down_conv(x.permute(0,3,1,2)).shape)
        global_x = self.down_conv(x.permute(0,3,1,2).contiguous()).reshape(B, C, hw1).transpose(1, 2)
        # print(global_x.shape)
        global_x = self.act(self.down_norm(global_x))
        # count1.record_time('down_conv')

        q = self.q_g(x).reshape(B, hw, num_heads, Ch).transpose(1, 2)
        kv = self.kv_g(global_x).reshape(B, hw1, 2, num_heads, Ch).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print('down', q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = x + (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        # count1.record_time('global')

        x = self.act(self.proj_norm(self.proj(x)))
        x = x + self.proj_drop(x)
        # count1.last('proj')

        return x



class WindowDownConvAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size=7,
                 num_heads=8,
                 conv_layers=2,
                 downsample_rate=7,
                 cfg_norm={'type': nn.GroupNorm, 'args': dict(num_groups=8)},
                 act_type=nn.ReLU,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 ):
        """
        窗口自注意力模块，附加均值下采样和卷积处理。
        """
        super().__init__()
        
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.window_size = window_size  # 支持正方形窗口
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        norm_type = cfg_norm['type']
        norm_args = cfg_norm['args']

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relu = act_type()
        self.norm_sa = nn.LayerNorm(dim)
        self.norm_fc = nn.LayerNorm(dim)

        self.downsample_rate = downsample_rate
        self.conv_layers = conv_layers

        self.pool = nn.AvgPool2d(self.downsample_rate, self.downsample_rate)

        # 卷积、归一化和激活组合
        self.conv_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                norm_type(num_channels=dim, **norm_args),
                act_type()
            ) for _ in range(conv_layers)
        ])
        self.conv_out = nn.Sequential(
                nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1),
                norm_type(num_channels=dim, **norm_args),
                act_type()
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m.bias, nn.Parameter):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)
        Returns:
            x: 输出张量，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        wsz = self.window_size
        nH, nW = H//wsz, W//wsz

        # 划分窗口
        x_windows = x.unfold(2, wsz, wsz).unfold(3, wsz, wsz)
        x_windows = x_windows.reshape(B, C, nH*nW, wsz*wsz).permute(0, 2, 3, 1).flatten(0,1).contiguous()
        B1, N, C = x_windows.shape
        # print(x_windows.shape)
        # 转置并计算QKV
        qkv = self.qkv(x_windows).reshape(B1, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*num_windows, num_heads, window_size*window_size, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权
        attn = (attn @ v).transpose(1, 2).reshape(B1, N, C)

        # 残差连接和投影
        x_windows = x_windows + self.relu(self.norm_sa(attn))
        x_windows = x_windows + self.proj_drop(self.relu(self.norm_fc(self.proj(x_windows))))

        
        # #TODO 恢复窗口形状
        # 把 x_windows 恢复成 x : [B*num_windows, window_size*window_size, C]    --->  [B,C,H,W]

        x = x_windows.reshape(B, nH, nW, wsz, wsz, C).transpose(2,3).reshape(B,H,W,C).permute(0,3,1,2).contiguous()

        # 均值下采样
        x1 = self.pool(x)

        # 卷积块处理
        for blk in self.conv_blocks:
            x1 = blk(x1) + x1

        # 上采样回原始尺寸
        x1_upsampled = nn.functional.interpolate(x1, size=(H, W), mode='bilinear', align_corners=False)

        x = torch.concat([x,x1_upsampled], dim=1)
        x = self.conv_out(x)

        return x

# attn = WindowDownConvAttention(256)
# # x1 = torch.randn(1,256,7,7)
# x2 = torch.randn(1,256,28,56)

# # attn(x1)
# attn(x2)
# exit()

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 act_type=nn.ReLU6,
                 norm_type=nn.LayerNorm,
                 attn_drop=0.,
                 proj_drop=0.):
        '''
            已自带残差结构, 直接nn.Seq即可
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        self.relu = act_type()
        self.norm_sa = norm_type(dim)
        self.norm_fc = norm_type(dim)

        
        self.apply(self._init_weights)


    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        # print(q.shape, k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = x + self.relu(self.norm_sa(attn))

        x = x + self.proj_drop(self.relu(self.norm_fc(self.proj(x))))

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class DWConv(nn.Module):
    
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptivePoolSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 pool_size=(7,7),
                #  use_post_conv=False,
                #  kv_conv_pose='before', # 'post'
                 qkv_bias=True,
                 qk_scale=None,
                #  act_type=nn.ReLU6,
                 act_type=nn.GELU,  #?改用 GELU
                 norm_type=nn.LayerNorm,
                 attn_drop=0.,
                 proj_drop=0.):
        '''
            已自带残差结构, 直接nn.Seq即可
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.pool_size = pool_size
        self.num_tokens = pool_size[0]*pool_size[1]

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        self.relu = act_type()
        self.norm_sa = norm_type(dim)
        self.norm_fc = norm_type(dim)
        
        #TODO 修改网络结构，测试

        self.pre_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1),
            nn.GroupNorm(16, dim),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.apply(self._init_weights)



    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        H2, W2 = self.pool_size
        N = H2*W2

        x0 = x + self.pre_conv(x)

        x = self.pool(x0)
        # print('@1', x.shape)
        x = x.reshape(B, C, -1).transpose(1,2).contiguous()
        # print('@2', x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # print(q.shape, k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.relu(self.norm_sa(attn))
        x = x + self.proj_drop(self.relu(self.norm_fc(self.proj(x))))
        
        x = x.transpose(1,2).reshape(B, C, H2, W2).contiguous()

        x = x0 + F.interpolate(x, (H,W), mode='bilinear', align_corners=False)

        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



class AdaptivePoolSelfAttention_V1(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 kv_pool_size=(7,7),
                #  use_post_conv=False,
                #  kv_conv_pose='before', # 'post'
                 qkv_bias=True,
                 qk_scale=None,
                #  act_type=nn.ReLU6,
                 act_type=nn.GELU,  #?改用 GELU
                 norm_type=nn.LayerNorm,
                 attn_drop=0.,
                 proj_drop=0.):
        '''
            已自带残差结构, 直接nn.Seq即可
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv_pool_size = kv_pool_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        self.relu = act_type()
        self.norm_sa = norm_type(dim)
        self.norm_fc = norm_type(dim)
        
        #TODO 修改网络结构，测试

        self.kv_conv_pre = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1, groups=dim), #TODO 此处用的 DWConv 减少计算
            # nn.BatchNorm2d(dim), #? conv后跟BN 而非 LN
            # nn.LayerNorm(dim),
            nn.GroupNorm(16, dim),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(kv_pool_size)

        #? kv_conv_post 在最低分辨率下 融合空间和通道， 能有效提升感受野
        self.kv_conv_post = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1, groups=dim),   #TODO pool后， 分辨率很小了，可以不分组
            # nn.BatchNorm2d(dim),
            # nn.LayerNorm(dim),
            nn.GroupNorm(16, dim),
            nn.ReLU()
        )

        # self.post_conv = DWConv(dim)


        self.apply(self._init_weights)



    def forward(self, x: torch.Tensor, H, W):

        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        #? 卷积池化操作
        x_ = x.transpose(1,2).reshape(B, C, H, W).contiguous()
        #? 卷积 池化 再卷积
        x_ = self.pool(x_ + self.kv_conv_pre(x_))
        x_ = x_ + self.kv_conv_post(x_)
        x_ = x_.reshape(B, C, -1).transpose(1,2).contiguous()

        # x_ = x.transpose(1,2).reshape(B, C, H, W)
        # x_ = self.pool(x_)
        # # x_ = x_ + self.kv_conv_post(x_)
        # x_ = x_.reshape(B, C, -1).transpose(1,2).contiguous()


        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # print(q.shape, k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = x + self.relu(self.norm_sa(attn))

        x = x + self.proj_drop(self.relu(self.norm_fc(self.proj(x))))

        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward2(self, x: torch.Tensor, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        #? 卷积池化操作
        print("\n@1", x.shape)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        print("@2", x_.shape)
        #? 卷积 池化 再卷积
        x_ = self.pool(x_ + self.kv_conv_pre(x_))
        print("@3", x_.shape)
        x_ = x_ + self.kv_conv_post(x_)
        x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        
        print("@4", x_.shape)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # print(q.shape, k.shape)
        print("@q,k,v", q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        print("@attn", attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print("@qkv", attn.shape)
        x = x + self.relu(self.norm_sa(attn))

        x = x + self.proj_drop(self.relu(self.norm_fc(self.proj(x))))

        return x
