import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from typing import Tuple

do_debug = False
do_debug_detail = False

def print2(*args):
    if do_debug_detail:
        print(*args)


from jscv.utils.utils import TimeCounter, warmup, do_once
count_refine = TimeCounter(False)


class Refiner(nn.Module):

    
    # For TorchScript export optimization.
    __constants__ = ['kernel_size', 'patch_crop_method', 'patch_replace_method']
    
    def __init__(self,
                 in_channel=30,
                 mid_channel=80,
                 out_channel=1,
                 
                 patch_size=16,
                 patch_padding=4,
                 post_conv_layers=8,

                 mode: str='sampling',
                 sample_ratio: int=1/8,
                 threshold: float=0.1,
                 prevent_oversampling: bool = True,
                 patch_crop_method: str = 'unfold',
                 patch_replace_method: str = 'scatter_nd'):
        super().__init__()
        assert mode in ['full', 'sampling', 'thresholding']
        # assert kernel_size in [1, 3]
        assert patch_crop_method in ['unfold', 'roi_align', 'gather']
        assert patch_replace_method in ['scatter_nd', 'scatter_element']
        # assert post_conv_layers > 0 # and patch_padding > 0 
        
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.post_conv_layers = post_conv_layers
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.threshold = threshold

        self.prevent_oversampling = prevent_oversampling
        self.patch_crop_method = patch_crop_method
        self.patch_replace_method = patch_replace_method

        if patch_padding == 0:
            pad = 1
            self.pre_conv_layers = 1
        else:
            pad = 0
            self.pre_conv_layers = patch_padding
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel + 1, mid_channel, 3, padding=pad, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )
        for i in range(1, patch_padding):
            if i == patch_padding - 1 and post_conv_layers == 0:
                conv = nn.Conv2d(mid_channel, out_channel, 3, bias=False)
            else:
                conv = nn.Sequential(
                    nn.Conv2d(mid_channel, mid_channel, 3, bias=False),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU()
                )
            self.__setattr__(f"conv{i}", conv)

        for i in range(post_conv_layers):
            if i == post_conv_layers - 1:
                conv = nn.Conv2d(mid_channel, out_channel, 3, padding=1, bias=False)
                self.__setattr__(f"post_conv{i}", conv)
            else:
                conv = nn.Sequential(
                    nn.Conv2d(mid_channel, mid_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU()
                )
                self.__setattr__(f"post_conv{i}", conv)


        self.relu = nn.ReLU(False)

        self.do_debug = False
    
    def forward(self,
                pha: torch.Tensor,
                err: torch.Tensor,
                hid: torch.Tensor,):

        stride = self.patch_size

        B, C_pha, H, W = pha.shape
        H_err, W_err = H//stride, W//stride

        from jscv.utils.utils import TimeCounter
        self.count = count_refine
        self.count.begin()

        if self.mode != 'full':
            err = F.interpolate(err, (H_err, W_err))
            refine_regions = self.select_refinement_regions(err)
 
            self.count.record_time("select_refinement_regions")

            idx = torch.nonzero(refine_regions.squeeze(1))
            
            self.count.record_time("torch.nonzero")
            
            idx = idx[:, 0], idx[:, 1], idx[:, 2]
            
            if idx[0].size(0) > 0:
                x = torch.cat([hid, pha], dim=1).detach()

                x = self.crop_patch(x, idx)
                self.count.record_time("crop_patch")


                for i in range(self.pre_conv_layers):
                    conv = self.__getattr__(f"conv{i}")
                    x = conv(x)

                self.count.record_time("pre_conv")

                for i in range(self.post_conv_layers):
                    if i == self.post_conv_layers - 1:
                        conv = self.__getattr__(f"post_conv{i}")
                        x = conv(x)
                    else:
                        conv = self.__getattr__(f"post_conv{i}")
                        x = x + conv(x)

                self.count.record_time("post_conv")

                pha = self.replace_patch(pha, x, idx)
                self.count.record_time("replace_patch", last=True)

            else:
                print("???")
                return pha

        return pha, refine_regions


    def select_refinement_regions(self, err: torch.Tensor):
        """
        Select refinement regions.
        Input:
            err: error map (B, 1, H, W)
        Output:
            ref: refinement regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
        """
        if self.mode == 'sampling':
            # Sampling mode.
            b, _, h, w = err.shape

            err = err.view(b, -1)
            # print(self.sample_ratio, err.shape)
            idx = err.topk(int(err.shape[-1] * self.sample_ratio), dim=1, sorted=False).indices
            # print(self.sample_ratio, idx.shape)
            ref = torch.zeros_like(err)
            ref.scatter_(1, idx, 1.)
            if self.prevent_oversampling:
                ref.mul_(err.gt(0).float())
            ref = ref.view(b, 1, h, w)
        else:
            # Thresholding mode.
            ref = err.gt(self.threshold).float()
        return ref
    
    def crop_patch(self,
                   x: torch.Tensor,
                   idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        '''
        #1 x = torch.Size([2, 33, 512, 512])
        #2 torch.Size([2, 518, 518, 33])
        #3 torch.Size([2, 256, 518, 33, 8])
                                                                                                                                                     
        #4 torch.Size([2, 256, 256, 33, 8, 8])

        idx torch.Size([10000]) torch.Size([10000]) torch.Size([10000])
        #5 torch.Size([10000, 33, 8, 8])

        '''

        patch_size = self.patch_size
        extra = self.patch_padding*2

        # print2("#1", x.shape)
        
        if patch_size != 0:
            x = F.pad(x, (self.patch_padding,) * 4)

        y = x.permute(0, 2, 3, 1)
        # print2("#2", y.shape)
        
        # self.patch_crop_method = 'unfold'
        # self.patch_crop_method = 'roi_align'
        

        if self.patch_crop_method == 'unfold':
            # Use unfold. Best performance for PyTorch and TorchScript.
            x = x.permute(0, 2, 3, 1) \
                    .unfold(1, extra + patch_size, patch_size) \
                    .unfold(2, extra + patch_size, patch_size)[idx[0], idx[1], idx[2]]

            if do_debug and do_once(self, "_patch_crop"):
                print("unfold", x.shape)
            return x

        elif self.patch_crop_method == 'roi_align':
            # Use roi_align. Best compatibility for ONNX.
            idx = idx[0].type_as(x), idx[1].type_as(x), idx[2].type_as(x)
            b = idx[0]
            x1 = idx[2] * patch_size - 0.5
            y1 = idx[1] * patch_size - 0.5
            x2 = idx[2] * patch_size + patch_size + extra - 0.5
            y2 = idx[1] * patch_size + patch_size + extra - 0.5
            boxes = torch.stack([b, x1, y1, x2, y2], dim=1)
            x = torchvision.ops.roi_align(x, boxes, extra + patch_size, sampling_ratio=1)
            if do_debug and do_once(self, "_patch_crop"):
                print("roi_align", x.shape)
            return x
        else:
            # Use gather. Crops out patches pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, patch_size, self.patch_padding)
            pat = torch.gather(x.view(-1), 0, idx_pix.view(-1))
            pat = pat.view(-1, x.size(1), extra + patch_size, extra + patch_size)

            if do_debug and do_once(self, "_patch_crop"):
                print("gather", pat.shape)

            return pat

    def replace_patch(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Replaces patches back into image given index.
          c
        Inputs:
            x: image (B, C, H, W)
            y: patches (P, C, h, w)
            idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.
        
        Output:
            image: (B, C, H, W), where patches at idx locations are replaced with y.
        """
        xB, xC, xH, xW = x.shape
        yB, yC, yH, yW = y.shape
        # print(x.shape,  y.shape)
        if self.patch_replace_method == 'scatter_nd':
            # print("scatter_nd")
            # Use scatter_nd. Best performance for PyTorch and TorchScript. Replacing patch by patch.
            x = x.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
            # print("@@", x.shape)

            x[idx[0], idx[1], idx[2]] = y
            x = x.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
            # print(x.shape)
            return x
        else:
            # Use scatter_element. Best compatibility for ONNX. Replacing pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size=4, padding=0)
            return x.view(-1).scatter_(0, idx_pix.view(-1), y.view(-1)).view(x.shape)

    def compute_pixel_indices(self,
                              x: torch.Tensor,
                              idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                              size: int,
                              padding: int):
        """
        Compute selected pixel indices in the tensor.
        Used for crop_method == 'gather' and replace_method == 'scatter_element', which crop and replace pixel by pixel.
        Input:
            x: image: (B, C, H, W)
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            idx: (P, C, O, O) long tensor where O is the output size: size + 2 * padding, P is number of patches.
                 the element are indices pointing to the input x.view(-1).
        """
        B, C, H, W = x.shape
        S, P = size, padding
        O = S + 2 * P
        b, y, x = idx
        n = b.size(0)
        c = torch.arange(C)
        o = torch.arange(O)
        idx_pat = (c * H * W).view(C, 1, 1).expand([C, O, O]) + (o * W).view(1, O, 1).expand([C, O, O]) + o.view(1, 1, O).expand([C, O, O])
        idx_loc = b * W * H + y * W * S + x * S
        idx_pix = idx_loc.view(-1, 1, 1, 1).expand([n, C, O, O]) + idx_pat.view(
            1, C, O, O).expand([n, C, O, O]).to(idx_loc.device)
        return idx_pix


'''
f1:
select_refinement_regions: 0.2479366409778595, torch.nonzero: 0.13529279977083206, 
crop_patch: 2.9570118236541747, conv 1~4: 1.2471488046646118, replace_patch: 0.14419391974806786


f2:
select_refinement_regions: 0.2316857597231865, torch.nonzero: 0.11448383957147598, 
crop_patch: 3.8090067315101623, conv 1~4: 4.493140449523926, replace_patch: 0.12943104043602943

f3:
select_refinement_regions: 0.24073151886463165, torch.nonzero: 0.12432639941573143,
crop_patch: 2.2692928099632264, conv 1~4: 4.516302075386047, replace_patch: 0.1312921606004238
'''

if __name__ == "__main__":
     
    torch.cuda.set_device(2)
    
    refiner_args=dict(
        in_channel          =   30,
        mid_channel         =   80,
        patch_size          =   16,
        patch_padding       =   4,
        post_conv_layers    =   8,
        sample_ratio        =   1/6,
        patch_crop_method   =   "unfold"
    )

    r = Refiner(**refiner_args).cuda()
    
    pha = torch.rand([1, 1, 512, 512]).cuda()
    err = torch.rand([1, 1, 512, 512]).cuda()
    hid = torch.rand([1, refiner_args["in_channel"], 512, 512]).cuda()
    
    count_refine.DO_DEBUG = True
    epoachs = 50
    do_debug = True
    warmup(20)
    
    for i in range(epoachs):
        r(pha, err, hid)
    
    print(count_refine.str_total(epoachs))
