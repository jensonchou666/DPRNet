import torch.nn.functional as F
import torch.nn as nn
import torch

import math

def init_conv2d(m):
    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    fan_out //= m.groups
    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    if m.bias is not None:
        m.bias.data.zero_()


class RestoreShape(nn.Module):
    def __init__(self, model, keys=['pred']):
        super().__init__()
        self.model = model
        self.keys = keys

    def forward(self, x, *args, **kargs):
        B, C, H, W = x.shape
        res = self.model(x, *args, **kargs)
        if isinstance(res, dict):
            for k in self.keys:
                res[k] = F.interpolate(res[k], size=(H, W), mode='bilinear')
        else:
            res = F.interpolate(res, size=(H, W), mode='bilinear')
        return res


class UpSample(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        kargs = kargs.copy()
        self.scale = False
        if 'scale_factor' in kargs:
            self.scale = True
            self.scale_factor = kargs.pop('scale_factor')
            assert 'size' not in kargs
        else:
            assert 'size' in kargs
        self.args = kargs
        
    def forward(self, x):
        if self.scale:
            h, w = x.shape[-2:]
            sz = (h * self.scale_factor, w * self.scale_factor)
            return F.interpolate(x, size=sz, **self.args)
        else:
            return F.interpolate(x, **self.args)
            

def upsample(x, h, w):
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

if __name__ == '__main__':
    x = torch.randn(1, 2, 4, 4)
    x = UpSample(scale_factor=4)(x)
    print(x.shape)