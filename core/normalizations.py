import torch


def Normalizations(tensor_size=None, normalization=None, available=False,
                   **kwargs):
    r"""Does normalization on 4D tensor.

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        normalization: None/batch/group/instance
    """
    list_available = ["batch", "group", "instance"]

    if available:
        return list_available

    normalization = normalization.lower()
    assert normalization in list_available, \
        "Normalization must be None/" + "/".join(list_available)

    if normalization == "batch":
        return torch.nn.BatchNorm2d(tensor_size[1])

    elif normalization == "group":
        affine = kwargs["affine"] if "affine" in \
            kwargs.keys() else False

        if "groups" in kwargs.keys():
            return torch.nn.GroupNorm(kwargs["groups"], tensor_size[1],
                                      affine=affine)
        else:
            possible = [tensor_size[1]//i for i in range(tensor_size[1], 0, -1)
                        if tensor_size[1] % i == 0]
            groups = possible[len(possible)//2]
            return torch.nn.GroupNorm(groups, tensor_size[1], affine=affine)

    elif normalization == "instance":
        affine = kwargs["affine"] if "affine" in \
            kwargs.keys() else False
        return torch.nn.InstanceNorm2d(tensor_size[1], affine=affine)
