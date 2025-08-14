# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/convnext.py
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from src.modelling.models.modules.base import BaseModule
from src.modelling.models.util import ConvNormActivation, LayerNorm2d

__all__ = [
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class CNBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            activation_layer: Callable[..., nn.Module],
            norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if norm_layer is not None and issubclass(norm_layer, LayerNorm2d):
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
                Permute([0, 2, 3, 1]),
                nn.LayerNorm(dim, eps=1e-6),
                nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                activation_layer,
                nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                Permute([0, 3, 1, 2]),
            )
        else:
            bias = not issubclass(norm_layer, nn.BatchNorm2d)
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=bias),
                norm_layer(dim),
                nn.Conv2d(dim, 4 * dim, 1, bias=bias),
                activation_layer,
                nn.Conv2d(4 * dim, dim, 1, bias=bias),
            )

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
            self,
            input_channels: int,
            out_channels: Optional[int],
            num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(BaseModule):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            in_channels: int,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            initial_stride: int = 4,
            **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        self.use_bias = issubclass(norm_layer, LayerNorm2d)

        block = CNBlock

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels

        stage: List[nn.Module] = []
        stage.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=7 if initial_stride == 1 else initial_stride,
                stride=initial_stride,
                padding=None,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=self.use_bias,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(
                    cnf.input_channels, layer_scale, sd_prob, norm_layer=norm_layer, activation_layer=activation_layer
                ))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                stage: List[nn.Module] = []
                # Downsampling
                stage.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2, bias=self.use_bias),
                    )
                )

        self.blocks = nn.ModuleList(layers)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        super(ConvNeXt, ConvNeXt).init_weights(m)
        # special init here
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def _convnext(
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float,
        **kwargs: Any,
) -> ConvNeXt:
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    return model


def convnext_tiny(**kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.convnext.ConvNeXt_Tiny_Weights
        :members:
    """

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)


def convnext_small(**kwargs: Any) -> ConvNeXt:
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.convnext.ConvNeXt_Small_Weights
        :members:
    """

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)


def convnext_base(**kwargs: Any) -> ConvNeXt:
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.convnext.ConvNeXt_Base_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)


def convnext_large(**kwargs: Any) -> ConvNeXt:
    """ConvNeXt Large model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.convnext.ConvNeXt_Large_Weights
        :members:
    """

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, **kwargs)
