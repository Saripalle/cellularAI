import torch.nn as nn
from .convolution import Convolution
from .activations import Activations


def check_strides(strides):
    return (strides > 1 if isinstance(strides, int) else
            (strides[0] > 1 or strides[1] > 1))


def check_residue(strides, t_size, out_channels):
    return check_strides(strides) or t_size[1] != out_channels


def update_kwargs(kwargs, *args):
    if len(args) > 0 and args[0] is not None:
        kwargs["tensor_size"] = args[0]
    if len(args) > 1 and args[1] is not None:
        kwargs["filter_size"] = args[1]
    if len(args) > 2 and args[2] is not None:
        kwargs["out_channels"] = args[2]
    if len(args) > 3 and args[3] is not None:
        kwargs["strides"] = args[3]
    if len(args) > 4 and args[4] is not None:
        kwargs["pad"] = args[4]
    if len(args) > 5 and args[5] is not None:
        kwargs["activation"] = args[5]
    if len(args) > 6 and args[6] is not None:
        kwargs["dropout"] = args[6]
    if len(args) > 7 and args[7] is not None:
        kwargs["normalization"] = args[7]
    if len(args) > 8 and args[8] is not None:
        kwargs["pre_nm"] = args[8]
    if len(args) > 9 and args[9] is not None:
        kwargs["groups"] = args[9]
    if len(args) > 10 and args[10] is not None:
        kwargs["weight_nm"] = args[10]
    if len(args) > 11 and args[11] is not None:
        kwargs["equalized"] = args[11]
    return kwargs


class ResidualOriginal(nn.Module):
    r""" Residual block with two 3x3 convolutions - used in ResNet18 and
    ResNet34. All args are similar to Convolution.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 *args, **kwargs):

        super(ResidualOriginal, self).__init__()
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized)

        self.Block1 = Convolution(tensor_size, filter_size, out_channels,
                                  strides, **kwargs)
        self.Block2 = Convolution(self.Block1.tensor_size, filter_size,
                                  out_channels, 1, **kwargs)

        if check_residue(strides, tensor_size, out_channels):
            if not pre_nm:
                kwargs["activation"] = ""
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(activation.lower())
        self.tensor_size = self.Block2.tensor_size

    def forward(self, tensor):
        if hasattr(self, "dropout"):  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block2(self.Block1(tensor)) + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor
# =========================================================================== #


class SEResidualComplex(nn.Module):
    r"""Bottleneck Residual block with squeeze and excitation added. All args
    are similar to Convolution.
    Implemented - https://arxiv.org/pdf/1709.01507.pdf

    Args:
        r: channel reduction factor, default = 16
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, r=16, *args, **kwargs):
        super(SEResidualComplex, self).__init__()
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift)

        self.Block1 = Convolution(tensor_size, 1, out_channels//4, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//4,
                        strides, groups=groups, **kwargs)
        if not pre_nm:
            kwargs["activation"] = ""
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)

        se = [Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1,
                          False, "relu"),
              Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1,
                          False, "sigm")]
        self.SE = nn.Sequential(*se)

        if check_residue(strides, tensor_size, out_channels):
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(activation.lower())
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if hasattr(self, "dropout"):  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        tensor = tensor * self.SE(F.avg_pool2d(tensor, tensor.shape[2:]))
        tensor = tensor + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor
