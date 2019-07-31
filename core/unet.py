import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Convolution
from .residual_block import ResidualOriginal as Residual
from .residual_block import SEResidualComplex as ResSE
from .residualnet import ResidualNet


class ConvBlock(nn.Module):
    '''
    Building block based on original UNet paper::
    https://arxiv.org/pdf/1505.04597.pdf

    chop: squeeze the number of channels by the ratio in the first convolution
    operation
    '''
    def __init__(self, tensor_size, out_channels, pad, chop=2,
                 *args, **kwargs):
        super(ConvBlock, self).__init__()
        norm = "batch"
        pad = True
        self.baseconv = nn.Sequential()
        self.baseconv.add_module("cnv_1", Convolution(tensor_size, 3,
                                                      out_channels//chop,
                                                      pad=pad,
                                                      normalization=norm))
        self.baseconv.add_module("cnv_2",
                                 Convolution(self.baseconv[-1].tensor_size, 3,
                                             out_channels,
                                             pad=pad,
                                             normalization=norm))
        self.tensor_size = self.baseconv[-1].tensor_size

    def forward(self, tensor):
        return self.baseconv(tensor)


class ResSEBlock(nn.Module):
    '''
    Building block based on AnatomyNet paper::
    https://arxiv.org/pdf/1808.05238.pdf
    ResSE - Residual Block with Squeeze and Excitation block
    '''
    def __init__(self, tensor_size, out_channels, pad=True, *args, **kwargs):
        super(ResSEBlock, self).__init__()
        norm, activation, pad = "batch", "lklu", True
        self.resSE = nn.Sequential()
        self.resSE.add_module("resSE_1", ResSE(tensor_size, 3, out_channels,
                                               pad=pad, activation=activation,
                                               normalization=norm, r=4))
        self.resSE.add_module("resSE_2", ResSE(self.resSE[-1].tensor_size, 3,
                                               out_channels, pad=pad,
                                               activation=activation,
                                               normalization=norm, r=4))
        self.tensor_size = self.resSE[-1].tensor_size

    def forward(self, tensor):
        return self.resSE(tensor)


class Down(nn.Module):
    '''
    Downsampling block for U-Net and AnatomyNet
    change: uses convolution with strides instead of pooling to downsample
    '''
    def __init__(self, tensor_size, in_channels, out_channels, strides=(1, 1),
                 pad=False, dropout=0.0, nettype='unet', *args, **kwargs):
        super(Down, self).__init__()

        assert nettype.lower() in ["unet", "anatomynet", "none"], \
            "network sould be unet or anatomynet or none"

        self.down = nn.Sequential()
        self.down.add_module("down1",
                             Convolution(tensor_size, 3, in_channels,
                                         (2, 2), pad=True))
        if nettype is 'unet':
            self.down.add_module("down2",
                                 ConvBlock(self.down[-1].tensor_size,
                                           out_channels, pad=pad))
        elif nettype is 'anatomynet':
            self.down.add_module("down2",
                                 ResSEBlock(self.down[-1].tensor_size,
                                            out_channels, pad=True))
        self.tensor_size = self.down[-1].tensor_size

    def forward(self, tensor):
        return self.down(tensor)


class Up(nn.Module):
    '''
    upsampling block for U-Net and AnatomyNet
    '''
    def __init__(self, tensor_size, out_shape, strides=(2, 2), pad=False,
                 dropout=0.0, nettype='unet', *args, **kwargs):
        super(Up, self).__init__()

        assert nettype.lower() in ["unet", "anatomynet", "none"],\
            "network sould be unet or anatomynet or none"
        Kwargs = {}
        if isinstance(strides, tuple):
            strides = strides[0]
        Kwargs["output_padding"] = int(strides//2)
        self.up = Convolution(tensor_size, 3, tensor_size[1], strides=2,
                              pad=True, transpose=True, maintain_out_size=True,
                              **Kwargs)
        _tensor_size = self.up.tensor_size
        self.up.tensor_size = (_tensor_size[0], _tensor_size[1]+out_shape[1]) \
            + (tensor_size[2]*2, tensor_size[2]*2)

        if nettype is 'unet':
            self.up_base = ConvBlock(self.up.tensor_size, out_shape[1],
                                     pad=pad)
            self.tensor_size = self.up_base.tensor_size
        elif nettype is 'anatomynet':
            self.up_base = ResSEBlock(self.up.tensor_size, out_shape[1],
                                      pad=pad)
            self.tensor_size = self.up_base.tensor_size
        else:
            self.tensor_size = self.up.tensor_size

    def forward(self, tensor1, tensor2, nettype="unet"):
        tensor1 = self.up(tensor1)
        _, _, h, w = list(map(int.__sub__,
                          list(tensor1.shape), list(tensor2.shape)))
        #  (padLeft, padRight, padTop, padBottom)
        pad = (w//2, w-w//2, h//2, h-h//2)
        tensor2 = F.pad(tensor2, pad)
        if nettype is not "none":
            tensor = torch.cat([tensor2, tensor1], dim=1)
            return self.up_base(tensor)
        else:
            return tensor1


class ResUp(nn.Module):
    '''
    upsampling block for U-Net pretrained
    '''
    def __init__(self, tensor_size, out_shape, *args, **kwargs):
        super(ResUp, self).__init__()
        pre_block_params = kwargs["pre_block_params"]
        pst_block_params = kwargs["pst_block_params"]

        self.up = Convolution(tensor_size, 3, tensor_size[1], strides=2,
                              pad=True, transpose=True, maintain_out_size=True)
        _tensor_size = self.up.tensor_size
        self.up.tensor_size = (_tensor_size[0], _tensor_size[1]) \
            + (tensor_size[2]*2, tensor_size[2]*2)

        self.up_pre = nn.Sequential()
        for i, (oc, s) in enumerate(pre_block_params):
            self.up_pre.add_module("upResidual"+str(i),
                                   Residual(_tensor_size, 3, oc, s, True))
            _tensor_size = self.up_pre[-1].tensor_size
            # print("upResidual"+str(i), _tensor_size)

        _tensor_size = (_tensor_size[0], _tensor_size[1]+out_shape[1]) \
            + (_tensor_size[2], _tensor_size[2])

        self.up_pst = nn.Sequential()
        for i, (oc, s) in enumerate(pst_block_params):
            self.up_pst.add_module("upResidual"+str(i),
                                   Residual(_tensor_size, 3, oc, s, True))
            _tensor_size = self.up_pst[-1].tensor_size
            # print("upResidual"+str(i), _tensor_size)

    def forward(self, tensor1, tensor2):
        tensor1 = self.up_pre(self.up(tensor1))
        tensor = torch.cat([tensor2, tensor1], dim=1)
        return self.up_pst(tensor)


class UNetPreTrained(nn.Module):
    def __init__(self, tensor_size, out_channels, n_classes, *args, **kwargs):
        super(UNetPreTrained, self).__init__()
        self.encoder = ResidualNet(tensor_size, type="r34-unet",
                                   pretrained=True)
        kwargs = {}
        kwargs["pre_block_params"] = [(256, 1)]*6
        kwargs["pst_block_params"] = [(256, 1)]*2
        self.up1 = ResUp(self.encoder.Net46[-1].tensor_size,
                         self.encoder.Net46.Residual6.tensor_size, **kwargs)
        kwargs = {}
        kwargs["pre_block_params"] = [(128, 1)]*4
        kwargs["pst_block_params"] = [(128, 1)]*2
        self.up2 = ResUp(self.up1.up_pst[-1].tensor_size,
                         self.encoder.Net46.Residual2.tensor_size, **kwargs)
        kwargs = {}
        kwargs["pre_block_params"] = [(64, 1)]*3
        kwargs["pst_block_params"] = [(64, 1)]*2
        self.up3 = ResUp(self.up2.up_pst[-1].tensor_size,
                         tensor_size, **kwargs)
        self.final_base = ConvBlock(self.up3.up_pst[-1].tensor_size,
                                    self.up3.up_pst[-1].tensor_size[1]//2,
                                    pad=True)
        self.final_layer = Convolution(self.final_base.tensor_size,
                                       1, n_classes)
        self.tensor_size = self.final_layer.tensor_size

    def forward(self, tensor):
        encoder01 = self.encoder.Net46.Convolution(tensor)
        Residual00 = self.encoder.Net46.Residual0(encoder01)
        Residual01 = self.encoder.Net46.Residual1(Residual00)
        Residual02 = self.encoder.Net46.Residual2(Residual01)
        Residual03 = self.encoder.Net46.Residual3(Residual02)
        Residual04 = self.encoder.Net46.Residual4(Residual03)
        Residual05 = self.encoder.Net46.Residual5(Residual04)
        Residual06 = self.encoder.Net46.Residual6(Residual05)
        Residual07 = self.encoder.Net46.Residual7(Residual06)
        Residual08 = self.encoder.Net46.Residual8(Residual07)
        Residual09 = self.encoder.Net46.Residual9(Residual08)
        Residual10 = self.encoder.Net46.Residual10(Residual09)
        Residual11 = self.encoder.Net46.Residual11(Residual10)
        Residual12 = self.encoder.Net46.Residual12(Residual11)
        self.up1.up.tensor_size = Residual06.shape
        up1 = self.up1(Residual12, Residual06)
        self.up2.up.tensor_size = Residual02.shape
        up2 = self.up2(up1, Residual02)
        self.up3.up.tensor_size = tensor.shape
        up3 = self.up3(up2, tensor)
        final = self.final_layer(self.final_base(up3))
        return final


class UNet(nn.Module):
    '''
    UNet: https://arxiv.org/pdf/1505.04597.pdf
    4 down blocks and 4 up blocks
    '''
    def __init__(self, tensor_size, out_channels, n_classes, *args, **kwargs):
        super(UNet, self).__init__()
        out_c = out_channels
        self.d1 = ConvBlock(tensor_size, out_c, pad=True)
        self.d2 = Down(self.d1.tensor_size, out_c*1, out_channels*2, pad=True)
        self.d3 = Down(self.d2.tensor_size, out_c*2, out_channels*4, pad=True)
        self.d4 = Down(self.d3.tensor_size, out_c*4, out_channels*8, pad=True)
        self.d5 = Down(self.d4.tensor_size, out_c*8, out_channels*16, pad=True)
        self.u1 = Up(self.d5.tensor_size, self.d4.tensor_size)
        self.u2 = Up(self.u1.tensor_size, self.d3.tensor_size)
        self.u3 = Up(self.u2.tensor_size, self.d2.tensor_size)
        self.u4 = Up(self.u3.tensor_size, self.d1.tensor_size)
        self.final_layer = Convolution(self.u4.tensor_size, 1, n_classes)
        self.tensor_size = self.final_layer.tensor_size

    def forward(self, tensor):
        d1 = self.d1(tensor)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        self.u1.up.tensor_size = self.d4.tensor_size
        u1 = self.u1(d5, d4)
        self.u2.up.tensor_size = self.d3.tensor_size
        u2 = self.u2(u1, d3)
        self.u3.up.tensor_size = self.d2.tensor_size
        u3 = self.u3(u2, d2)
        self.u4.up.tensor_size = self.d1.tensor_size
        u4 = self.u4(u3, d1)
        return self.final_layer(u4)


class ANet(nn.Module):
    '''
    AnatomyNet architecture -- https://arxiv.org/pdf/1808.05238.pdf
    '''
    def __init__(self, tensor_size, out_channels, n_classes=2,
                 *args, **kwargs):
        super(ANet, self).__init__()
        self.d1 = Down(tensor_size, tensor_size[1], out_channels,
                       pad=True, nettype='anatomynet')
        self.b1 = ResSEBlock(self.d1.tensor_size, int(out_channels*1.25))
        self.b2 = ResSEBlock(self.b1.tensor_size, int(out_channels*1.50))
        self.b3 = ResSEBlock(self.b2.tensor_size, int(out_channels*1.75))
        self.b4 = ResSEBlock(self.b3.tensor_size, int(out_channels*1.75))

        _tensor_size = self.b4.tensor_size
        _tensor_size = (_tensor_size[0], self.b4.tensor_size[1] +
                        self.b2.tensor_size[1],
                        _tensor_size[2], _tensor_size[3])
        self.c1 = ResSEBlock(_tensor_size, int(out_channels*1.50))

        _tensor_size = self.c1.tensor_size
        _tensor_size = (_tensor_size[0], self.c1.tensor_size[1] +
                        self.b1.tensor_size[1],
                        _tensor_size[2], _tensor_size[3])
        self.c2 = ResSEBlock(_tensor_size, int(out_channels*1.25))

        _tensor_size = self.c2.tensor_size
        _tensor_size = (_tensor_size[0], self.c2.tensor_size[1] +
                        self.d1.tensor_size[1],
                        _tensor_size[2], _tensor_size[3])
        self.c3 = ResSEBlock(_tensor_size, int(out_channels))

        self.u1 = Up(self.c3.tensor_size, tensor_size, nettype="none")

        _tensor_size = self.u1.tensor_size
        self.f1 = Convolution(_tensor_size, 3, int(out_channels*0.5), pad=True,
                              normalization="batch")
        self.f2 = Convolution(self.f1.tensor_size, 3, n_classes, pad=True,
                              normalization="batch")
        self.tensor_size = (tensor_size[0], n_classes,
                            tensor_size[2], tensor_size[3])

    def forward(self, tensor):
        assert((tensor.shape[2] % 2 == 0) & (tensor.shape[3] % 2 == 0)), \
                "tensor height and width should be divisible by 2"
        d1 = self.d1(tensor)
        b1 = self.b1(d1)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        c1 = self.c1(torch.cat([b4, b2], dim=1))
        c2 = self.c2(torch.cat([c1, b1], dim=1))
        c3 = self.c3(torch.cat([c2, d1], dim=1))
        up = self.u1(c3, tensor, "none")
        f1 = self.f1(torch.cat([up, tensor], dim=1))
        f2 = self.f2(f1)
        return f2
#
# tsize = (1,1,572,572)
# unet = UNet(tsize, 64)
# anet = ANet(tsize, 32)
# unet(torch.rand(tsize)).shape
# anet(torch.rand(tsize)).shape
