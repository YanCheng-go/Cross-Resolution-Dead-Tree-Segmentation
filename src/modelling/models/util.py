from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import spectral_norm, weight_norm


class NoOp(nn.Sequential):
    def __init__(self, **kwargs): super().__init__()


def noop(x): return x


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def forward(self, x):
        return mish(x)


def get_act(name, inplace=True):
    name = "none" if name is None else name.lower()
    if name == "elu":
        return nn.ELU(inplace=inplace, alpha=0.54)
    elif name == "mish":
        return Mish()
    elif name == "gelu":
        return nn.GELU()
    elif name in ["swish", "silu"]:
        return nn.SiLU(inplace=inplace)
    elif name in ["leakyrelu", "leaky_relu"]:
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name in ["no", "none"]:
        return nn.Sequential()
    else:
        return nn.ReLU(inplace=inplace)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dims, eps: float = 1e-6, elementwise_affine: bool = True, device=None, dtype=None):
        super(LayerNorm2d, self).__init__(dims, eps, elementwise_affine, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm3d(LayerNorm2d):

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


def get_norm_layer(name):
    name = "none" if name is None else name.lower()
    if name in ["bn", "batch", "batchnorm"]:
        return nn.BatchNorm2d
    elif name in ["bn_mnasnet", "batch_mnasnet", "batchnorm_mnasnet"]:
        return BNMNasNet
    elif name in ["bn_efficientnet", "batch_efficientnet", "batchnorm_efficientnet"]:
        return BNEfficientNet
    elif name in ["bn_efficientnetv2", "batch_efficientnetv2", "batchnorm_efficientnetv2"]:
        return BNEfficientNetV2
    elif name in ["gn", "group", "groupnorm"]:
        return AutoGroupNorm
    elif name in ["ln", "layer", "layernorm"]:
        return LayerNorm2d
    elif name in ["in", "instance", "instancenorm"]:
        return nn.InstanceNorm2d
    elif name in ["spectral"]:
        return spectral_norm
    elif name in ["weight"]:
        return weight_norm
    elif name in ["no", "none"]:
        return None
    else:
        raise NotImplementedError(f"{name} is not implemented")


class AutoGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None):
        d = 32
        while d > 0:
            if num_channels % d == 0:
                break
            d -= 1

        super().__init__(d, num_channels, eps, affine, device, dtype)


class BatchNorm2dNonAffine(nn.BatchNorm2d):
    '''BNorm according to MNasNet (changed defaults)'''

    def __init__(self, num_features, eps=1e-5, momentum=1 - 0.9997,
                 affine=False, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class BNMNasNet(nn.BatchNorm2d):
    '''BNorm according to MNasNet (changed defaults)'''

    def __init__(self, num_features, eps=1e-5, momentum=1 - 0.9997,
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class BNEfficientNet(nn.BatchNorm2d):
    '''BNorm according to EfficientNet b5-7 (changed defaults)'''

    def __init__(self, num_features, eps=0.001, momentum=0.01,
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class BNEfficientNetV2(nn.BatchNorm2d):
    '''BNorm according to EfficientNet v2 (changed defaults)'''

    def __init__(self, num_features, eps=1e-03, momentum=0.1,
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


def icnr_init(x, scale=2):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = nn.init.xavier_normal_(torch.zeros([ni2, nf, h, w]), gain=1.55).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    return k


class LinearNormActivation(nn.Sequential):
    """
    Configurable block used for Linear-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Linear-Normalization-Activation block
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ) -> None:
        layers = [
            nn.Linear(
                in_channels,
                out_channels,
                bias=norm_layer is None,
            )
        ]
        # if norm_layer is not None:
        #     # if class, invoke it, else call as a function on layer
        #     if isinstance(norm_layer, type):
        #         layers.append(self.check_norm_layer(norm_layer)(out_channels))
        #     else:
        #         norm_layer(layers[0])
        if activation_layer is not None:
            layers.append(activation_layer)
        super().__init__(*layers)
        self.out_channels = out_channels

    @staticmethod
    def check_norm_layer(norm_layer):
        if norm_layer in [nn.BatchNorm2d, BNMNasNet, BNEfficientNet, nn.InstanceNorm2d]:
            mapping = {
                nn.BatchNorm2d: nn.BatchNorm1d,
                BNMNasNet: partial(nn.BatchNorm1d, eps=1e-5, momentum=1 - 0.9997),
                BNEfficientNet: partial(nn.BatchNorm1d, eps=0.001, momentum=0.01),
                nn.InstanceNorm2d: nn.InstanceNorm1d,
            }
            norm_layer = mapping[norm_layer]
        return norm_layer


class ConvNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = None,
            dilation: int = 1,
            self_attention: bool = False,
            bias: bool = None
    ) -> None:
        if bias is None:  # if it is set, use the given values
            # no bias if batch norm but use it in all other cases
            if norm_layer is not None and issubclass(norm_layer, _BatchNorm):
                bias = False
            else:
                bias = True

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            # if class, invoke it, else call as a function on conv layer
            if isinstance(norm_layer, type):
                layers.append(norm_layer(out_channels))
            else:
                norm_layer(layers[0])
        if activation_layer is not None:
            layers.append(activation_layer)
        if self_attention:
            layers.append(SelfAttention(out_channels))
        super().__init__(*layers)
        self.out_channels = out_channels


class CustomPixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self, ni: int, act_fn: nn.Module, norm_layer: nn.Module, nf: int = None, scale: int = 2):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [
            ConvNormActivation(
                ni, nf * (scale ** 2), kernel_size=1, activation_layer=act_fn, norm_layer=norm_layer
            ),
            nn.PixelShuffle(scale)
        ]
        super().__init__(*layers)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


class SelfAttention(nn.Module):
    "Self attention layer for nd."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
