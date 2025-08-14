# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/efficientnet.py
import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, List, Sequence, Tuple, Union

import torch.nn as nn

__all__ = [
    "NoPadEfficientNet",
    "nopad_efficientnet_b0",
    "nopad_efficientnet_b1",
    "nopad_efficientnet_b2",
    "nopad_efficientnet_b3",
    "nopad_efficientnet_b4",
    "nopad_efficientnet_b5",
    "nopad_efficientnet_b6",
    "nopad_efficientnet_b7",
    "nopad_efficientnet_v2_s",
    "nopad_efficientnet_v2_m",
    "nopad_efficientnet_v2_l",
]

from torch import Tensor
from torch.nn.functional import interpolate
from torchvision.ops import StochasticDepth

from src.modelling.models.modules.base import BaseModule
from src.modelling.models.util import ConvNormActivation

ConvNormAct = partial(ConvNormActivation, padding=0)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            patch_size: tuple,
            activation: Callable[..., nn.Module] = nn.ReLU,
            scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.manual_broadcast = patch_size is not None
        self.avgpool = nn.AvgPool2d(patch_size, stride=1) if self.manual_broadcast else nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        if self.manual_broadcast:
            scale = interpolate(scale, input.shape[2:])  # , mode='nearest-exact') for newer pytorch versions >0.11
        return scale * input


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    dilation: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            dilation: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, dilation, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            dilation: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, dilation, input_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            patch_size: tuple,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormAct(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormAct(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                dilation=cnf.dilation,
            )
        )
        if patch_size is not None:
            patch_size = (patch_size[0] - (cnf.kernel // 2 * cnf.dilation * 2),
                          patch_size[1] - (cnf.kernel // 2 * cnf.dilation * 2))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, patch_size, activation=activation_layer))

        # project
        layers.append(
            ConvNormAct(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels
        self.squeeze = (cnf.kernel // 2 * cnf.dilation, -(cnf.kernel // 2 * cnf.dilation))

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input[:, :, self.squeeze[0]:self.squeeze[1], self.squeeze[0]:self.squeeze[1]]
        return result



class FusedMBConv(nn.Module):
    def __init__(
            self,
            cnf: FusedMBConvConfig,
            patch_size: tuple,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                ConvNormAct(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    dilation=cnf.dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                ConvNormAct(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                ConvNormAct(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
        self.squeeze = (cnf.kernel // 2 * cnf.dilation, -(cnf.kernel // 2 * cnf.dilation))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input[:, :, self.squeeze[0]:self.squeeze[1], self.squeeze[0]:self.squeeze[1]]
        return result


class NoPadEfficientNet(BaseModule):
    def __init__(
            self,
            inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
            in_channels: int,
            patch_size: tuple,
            stochastic_depth_prob: float = 0.2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            initial_stride: int = 1,
            **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class
        Args:
            inverted_residual_setting (Sequence[Union[[MBConvConfig]]): Network structure
            patch_size (int or tuple): expected training size during training
            stochastic_depth_prob (float): The stochastic depth probability
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            activation_layer (Optional[Callable[..., nn.Module]]): Module specifying the activation layer to use
        """
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormAct(
                in_channels, firstconv_output_channels, kernel_size=3, stride=initial_stride,  # original stride is 2
                norm_layer=norm_layer, activation_layer=activation_layer
            )
        )
        if patch_size is not None:
            patch_size = (patch_size[0] - 2, patch_size[1] - 2)

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                    block_cnf.dilation = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, patch_size, sd_prob, norm_layer, activation_layer))

                if patch_size is not None:
                    patch_size = (patch_size[0] - (block_cnf.kernel // 2 * block_cnf.dilation * 2),
                                  patch_size[1] - (block_cnf.kernel // 2 * block_cnf.dilation * 2))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        self.blocks = nn.ModuleList(layers)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        super(NoPadEfficientNet, NoPadEfficientNet).init_weights(m)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _efficientnet(
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        **kwargs: Any,
) -> NoPadEfficientNet:
    model = NoPadEfficientNet(inverted_residual_setting, **kwargs)
    return model


def _efficientnet_conf(
        arch: str,
        **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 1, 32, 16, 1),
            bneck_conf(6, 3, 1, 2, 16, 24, 2),
            bneck_conf(6, 3, 1, 3, 24, 40, 2),
            bneck_conf(6, 3, 1, 2, 40, 80, 3),
            bneck_conf(6, 3, 1, 2, 80, 112, 3),
            bneck_conf(6, 3, 1, 3, 112, 192, 4),
            bneck_conf(6, 3, 1, 1, 192, 320, 1),
        ]
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 1, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 1, 2, 48, 64, 4),
            MBConvConfig(4, 3, 1, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 1, 128, 160, 9),
            MBConvConfig(6, 3, 1, 2, 160, 256, 15),
        ]
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 1, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 1, 2, 48, 80, 5),
            MBConvConfig(4, 3, 1, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 1, 160, 176, 14),
            MBConvConfig(6, 3, 1, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 1, 304, 512, 5),
        ]
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 1, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 1, 2, 64, 96, 7),
            MBConvConfig(4, 3, 1, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 1, 192, 224, 19),
            MBConvConfig(6, 3, 1, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 1, 384, 640, 7),
        ]
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting


def nopad_efficientnet_b0(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B0 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b1(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B1 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b2(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B2 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b3(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B3 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b4(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B4 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b5(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B5 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b6(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B6 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_b7(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B7 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_v2_s(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B7 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_v2_m(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B7 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(inverted_residual_setting, **kwargs)


def nopad_efficientnet_v2_l(**kwargs: Any) -> NoPadEfficientNet:
    """
    Constructs a NoPadEfficientNet B7 architecture from
    `"NoPadEfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    """
    inverted_residual_setting = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(inverted_residual_setting, **kwargs)
