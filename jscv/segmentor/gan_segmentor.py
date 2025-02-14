import sys, os
import torch
import torch.nn.functional as F
import torch.nn as nn

import functools

from jscv.utils.statistics import StatisticModel
from jscv.losses.utils import *
from jscv.utils.overall import global_dict
from jscv.utils.utils import do_once


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GANSegmentor(nn.Module):
    def __init__(self,
                 num_classes,
                 netG,  # (img, target) -> prediction
                 netD=None,  # (pred, target) -> real_or_fake
                 optimizer_G=None,
                 optimizer_D=None,
                 lr_schedulers=None,
                 ignore_index=None,
                 Seg_loss_layer=None,
                 concat_input=True,
                 gan_mode='vanilla',
                 warmup_epoch=0,
                 weight_Seg_loss=0.5,  # weight of: main_loss and GAN_loss
                 weight_loss_D_real=0.5,
                 channel_dim=1):      # 1 or 3
        super().__init__()
        assert channel_dim in [1, 3]

        self.concat_input = concat_input
        self.Seg_loss_layer = Seg_loss_layer
        self.weight_Seg_loss = weight_Seg_loss
        self.weight_loss_D_real = weight_loss_D_real

        self.netG = netG
        self.netD = netD
        self.criterionGAN = GANLoss(gan_mode)


        nonnet_models = self.nonnet_models = dict()

        nonnet_models["optimizer_G"] = optimizer_G
        nonnet_models["optimizer_D"] = optimizer_D
        nonnet_models["lr_schedulers"] = lr_schedulers

        self.num_classes = num_classes
        self.channel_dim = channel_dim
        self.ignore_index = ignore_index

        if warmup_epoch <= 0:
            warmup_epoch = -1
        self.warmup_epoch = warmup_epoch




    def farward_D(self, prediction, target):
        """Calculate GAN loss for the discriminator"""

        pred_fake = self.netD(prediction.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * (1 - self.weight_loss_D_real)

        pred_real = self.netD(target)  # .detach())
        loss_D_real = self.criterionGAN(pred_real, True) * self.weight_loss_D_real

        return loss_D_fake + loss_D_real

    def farward_G(self, prediction):
        """Calculate GAN and loss for the generator"""
        pred_fake = self.netD(prediction)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        return loss_G_GAN

    def handel_pred_target(self, pred, target, img=None):
        ign = self.ignore_index
        nc = self.num_classes

        ''' handle: ignore_index'''
        if ign is not None:
            mask = target == ign
            target[mask] = 0
            target = F.one_hot(target, nc).float()
            target[mask] = 0
            if self.channel_dim == 1:
                b, h, w = mask.shape
                mask = mask.reshape(b, 1, h, w).expand(b, nc, h, w)
            pred[mask] = 0
        else:
            target = F.one_hot(target, nc).float()

        if self.channel_dim == 1:
            target = target.permute(0, 3, 1, 2)

        if self.concat_input:
            pred = torch.cat((img, pred), 1)
            target = torch.cat((img, target), 1)
        
        return pred, target

    def step_train_no_gan(self, img, target, opt_step=True, sch_step=False):
        assert self.Seg_loss_layer is not None, "Miss: Seg_loss_layer"

        self.nonnet_models["optimizer_G"].zero_grad()
        pred = self.netG(img)
        losses = self.Seg_loss_layer(pred, target)
        sum_losses(losses).backward()
        self.nonnet_models["optimizer_G"].step()

        return pred, losses

    #TODO data_pool

    def step_train(self, img, target, opt_step=True, sch_step=False):
        if "trainer" in global_dict:
            trainer = global_dict["trainer"]
            if trainer.epoch_idx < self.warmup_epoch:
                if do_once(self, "print_warmup_epoch"):
                    print("@ warmup_epoch:", self.warmup_epoch)
                return self.step_train_no_gan(img, target, opt_step, sch_step)
            elif trainer.epoch_idx == self.warmup_epoch:
                #?? 先进行若干次 netD的训练
                pass

        weight_Seg, weight_GAN = self.weight_Seg_loss, 1 - self.weight_Seg_loss
        loss_dict = dict()

        #self.set_requires_grad(self.netG, True)
        pred = self.netG(img)

        pred_org = pred.clone()
        if self.Seg_loss_layer is not None:
            target_org = target.clone()

        pred, target = self.handel_pred_target(pred, target, img)

        # update D
        self.set_requires_grad(self.netD, True)
        self.nonnet_models["optimizer_D"].zero_grad()
        loss_d = self.farward_D(pred, target) * weight_GAN
        loss_d.backward()
        self.nonnet_models["optimizer_D"].step()

        loss_dict["loss_D"] = loss_d.detach()

        # update G
        self.set_requires_grad(self.netD, False)
        self.nonnet_models["optimizer_G"].zero_grad()

        loss_G_GAN = self.farward_G(pred) * weight_GAN
        loss_dict["loss_G_GAN"] = loss_G_GAN.detach()

        if self.Seg_loss_layer is not None:
            loss_Seg = losses_weighted(self.Seg_loss_layer(pred_org, target_org), weight_Seg)
            loss_g = loss_G_GAN + sum_losses(loss_Seg)
            loss_Seg = losses_add_suffix_detached(loss_Seg, "_Seg")
            loss_dict.update(loss_Seg)
        else:
            loss_g = loss_G_GAN

        loss_g.backward()
        self.nonnet_models["optimizer_G"].step()
 
        loss_dict["loss_main"] = loss_g.detach()

        if sch_step:
            for sch in self.nonnet_models["lr_schedulers"]:
                sch.step()

        return pred_org, loss_dict
    
    def forward(self, img, target=None):
        ''' validation or test step, with torch.no_grad():'''
        #self.set_requires_grad(self.netG, False)
        pred = self.netG(img)

        if target is None or self.netD is None:
            return pred
        else:
            #self.set_requires_grad(self.netD, False)
            weight_Seg, weight_GAN = self.weight_Seg_loss, 1 - self.weight_Seg_loss
            loss_dict = dict()

            pred_org = pred.clone()
            if self.Seg_loss_layer is not None:
                target_org = target.clone()

            pred, target = self.handel_pred_target(pred, target, img)

            loss_d = self.farward_D(pred, target) * weight_GAN
            loss_dict["loss_D"] = loss_d

            loss_G_GAN = self.farward_G(pred) * weight_GAN
            loss_dict["loss_G_GAN"] = loss_G_GAN

            if self.Seg_loss_layer is not None:
                loss_Seg = losses_weighted(self.Seg_loss_layer(pred_org, target_org), weight_Seg)
                loss_g = loss_G_GAN + sum_losses(loss_Seg)
                loss_Seg = losses_add_suffix(loss_Seg, "_Seg")
                loss_dict.update(loss_Seg)
            else:
                loss_g = loss_G_GAN
            
            loss_dict["loss_main"] = loss_g

            return pred_org, loss_dict

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def traverse(self, stat: StatisticModel, img, target=None):
        pred = stat.step(self.netG, (img,))
        if self.concat_input:
            pred = torch.cat((img, pred), 1)
        stat.step(self.netD, (pred,))



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


if __name__ == "__main__":
    from jscv.utils.losses import sce_dice_loss

    test_num_class = 6
    test_netG = UnetGenerator(3, test_num_class, 6)
    test_netD = NLayerDiscriminator((test_num_class + 3), 64, 4)
    opt_G = torch.optim.Adam(test_netG.parameters(), lr=1e-3, weight_decay=1e-3)
    opt_D = torch.optim.Adam(test_netD.parameters(), lr=1e-3, weight_decay=1e-3)
    SCH = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    schs = [SCH(opt_G, T_0=15, T_mult=2), SCH(opt_D, T_0=15, T_mult=2)]


    img = torch.randn(2, 3, 128, 128)
    target = torch.randint(0, test_num_class, [2, 128, 128])

    '''
    pred = test_netG(img)
    target_on_hot = F.one_hot(target, 6).permute(0, 3, 1, 2).float()
    print("@", pred.shape, target_on_hot.shape)
    x = test_netD(pred)
    y = test_netD(target_on_hot)
    print(x.shape, y.shape)
    '''

    seg_loss_layer = sce_dice_loss(test_num_class)[0]
    model = GANSegmentor(test_netG, test_netD, opt_G, opt_D, schs, test_num_class, seg_loss_layer)

    pred, loss_dict = model.step_train(img, target, True, False)
    print(pred.shape, loss_dict)