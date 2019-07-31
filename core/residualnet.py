import os
import wget
import torch
import torch.nn as nn
from .convolution import Convolution
from .residual_block import ResidualOriginal
# =========================================================================== #


class ImageNetNorm(nn.Module):
    def forward(self, tensor):
        if tensor.size(1) == 1:  # convert to rgb
            tensor = torch.cat((tensor, tensor, tensor), 1)
        if tensor.min() >= 0:  # do imagenet normalization
            tensor[:, 0].add_(-0.485).div_(0.229)
            tensor[:, 1].add_(-0.456).div_(0.224)
            tensor[:, 2].add_(-0.406).div_(0.225)
        return tensor


def map_pretrained(state_dict, type):
    # no fully connected
    if type == "r18":
        url = r'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif type == "r34" or type == "r34-unet":
        url = r'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    else:
        print(" ... pretrained weights are not avaiable for {}".format(type))
        return state_dict

    # download is not in models
    filename = os.path.join(".../models" if os.path.isdir(".../models") else
                            "./models", url.split("/")[-1])
    if not os.path.isfile(filename):
        print(" ... downloading pretrained")
        wget.download(url, filename)
        print(" ... downloaded")

    # relate keys from TensorMONK's model to pretrained
    labels = state_dict.keys()
    prestate_dict = torch.load(filename)
    prelabels = prestate_dict.keys()

    pairs = []
    _labels = [x for x in labels if "num_batches_tracked" not in x and
               "edit_residue" not in x and "embedding" not in x]
    _prelabels = [x for x in prelabels if "fc" not in x and
                  "downsample" not in x]
    for x, y in zip(_labels, _prelabels):
        pairs.append((x, y.replace(y.split(".")[-1], x.split(".")[-1])))

    _labels = [x for x in labels if "num_batches_tracked" not in x and
               "edit_residue" in x]
    _prelabels = [x for x in prelabels if "fc" not in x and
                  "downsample" in x]
    for x, y in zip(_labels, _prelabels):
        pairs.append((x, y.replace(y.split(".")[-1], x.split(".")[-1])))

    # update the state_dict
    for x, y in pairs:
        if state_dict[x].size() == prestate_dict[y].size():
            state_dict[x] = prestate_dict[y]

    del prestate_dict
    return state_dict


class ResidualNet(nn.Sequential):
    r"""Versions of residual networks.
        Implemented
        ResNet*   from https://arxiv.org/pdf/1512.03385.pdf
        SEResNet* from https://arxiv.org/pdf/1709.01507.pdf
    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        type (string): model type
            Available models        type
            ================================
            ResNet18                r18
            ResNet34                r34
        pretrained: downloads and updates the weights with pretrained weights
    """

    def __init__(self,
                 tensor_size=(6, 3, 128, 128),
                 type: str = "r18",
                 activation: str = "relu",
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 pretrained: bool = False,
                 n_layers: int = None,
                 *args, **kwargs):
        super(ResidualNet, self).__init__()

        type = type.lower()
        assert type in ("r18", "r34", "r34-unet"),\
            """ResidualNet -- type must be r18/r34/"""

        self.pretrained = pretrained
        if self.pretrained:
            assert tensor_size[1] == 1 or tensor_size[1] == 3, """ResidualNet ::
                rgb(preferred)/grey image is required for pretrained"""
            activation, normalization, pre_nm = "relu", "batch", True
            groups, weight_nm, equalized = 1, False, False
        self.model_type = type
        self.in_tensor_size = tensor_size

        if type in ("r18", "r34"):
            BaseBlock = ResidualOriginal
            if type == "r18":
                # 2x 64; 2x 128; 2x 256; 2x 512
                block_params = [(64, 1), (64, 1), (128, 2), (128, 1),
                                (256, 2), (256, 1), (512, 2), (512, 1)]
            else:
                # 3x 64; 4x 128; 6x 256; 3x 512
                block_params = [(64, 1)]*3 + \
                               [(128, 2)] + [(128, 1)]*3 + \
                               [(256, 2)] + [(256, 1)]*5 + \
                               [(512, 2)] + [(512, 1)]*2
        else:  # This is used for segmentaion task
            BaseBlock = ResidualOriginal
            # 2x 64; 2x 128; 2x 256; 2x 512
            block_params = [(64, 1)]*3 + \
                           [(128, 2)] + [(128, 1)]*3 + \
                           [(256, 2)] + [(256, 1)]*5

        if pretrained:
            print("ImageNetNorm = ON")
            self.add_module("ImageNetNorm", ImageNetNorm())
        else:
            n_layers = None

        self.Net46 = nn.Sequential()
        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64:
            # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("Initial convolution strides changed from 2 to 1, \
                    as min(tensor_size[2], tensor_size[3]) <  64")

        self.Net46.add_module("Convolution",
                              Convolution(tensor_size, 7, 64, s, True,
                                          activation, 0., normalization,
                                          False, 1, weight_nm, equalized,
                                          **kwargs))

        print("Convolution", self.Net46[-1].tensor_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.Net46.add_module("MaxPool", nn.MaxPool2d((3, 3),
                                                          stride=(2, 2),
                                                          padding=1))
            _tensor_size = (self.Net46[-2].tensor_size[0],
                            self.Net46[-2].tensor_size[1],
                            self.Net46[-2].tensor_size[2]//2,
                            self.Net46[-2].tensor_size[3]//2)
            print("MaxPool", _tensor_size)
        else:  # Addon -- To make it flexible for other tensor_size's
            print("MaxPool is ignored if min(tensor_size[2], tensor_size[3]) \
                  <= 128")
            _tensor_size = self.Net46[-1].tensor_size

        for i, (oc, s) in enumerate(block_params):
            self.Net46.add_module("Residual"+str(i),
                                  BaseBlock(_tensor_size, 3, oc, s, True,
                                            activation, 0., normalization,
                                            pre_nm, groups, weight_nm,
                                            equalized, **kwargs))
            _tensor_size = self.Net46[-1].tensor_size
            print("Residual"+str(i), _tensor_size)

        if type != "r34-unet":
            self.Net46.add_module("AveragePool",
                                  nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
            print("AveragePool", (1, oc, 1, 1))
            self.tensor_size = (6, oc)
        else:
            self.tensor_size = self.Net46[-1].tensor_size

        if n_layers is None:
            self.add_module("AveragePool",
                            nn.AvgPool2d(self.tensor_size[2:]))
            print("AveragePool", (1, oc, 1, 1))
            self.tensor_size = (1, oc)

            if n_embedding is not None and n_embedding > 0:
                self.add_module("embedding", nn.Linear(oc, n_embedding,
                                                       bias=False))
                self.tensor_size = (1, n_embedding)
                print("Linear", (1, n_embedding))

        if self.pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        if self.in_tensor_size[1] == 1 or self.in_tensor_size[1] == 3:
            self.load_state_dict(map_pretrained(self.state_dict(),
                                                self.model_type))
        else:
            print(" ... pretrained not available")
            self.pretrained = False
