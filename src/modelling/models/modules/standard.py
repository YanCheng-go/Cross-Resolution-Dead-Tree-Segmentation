import torch.nn as nn

from src.modelling.models.modules.base import BaseModule
from src.modelling.models.util import ConvNormActivation

__all__ = ['StandardNet', 'standard']


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_layer, norm_layer, pool: bool = False,
                 self_attention: bool = False):
        super(DoubleConv, self).__init__()
        self.pool = nn.MaxPool2d(2, ceil_mode=True) if pool else nn.Sequential()
        self.conv1 = ConvNormActivation(in_channels, out_channels, activation_layer=activation_layer,
                                        norm_layer=norm_layer)
        self.conv2 = ConvNormActivation(
            out_channels, out_channels, activation_layer=activation_layer,
            norm_layer=norm_layer, self_attention=self_attention
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DoubleConvSkip(DoubleConv):
    def __init__(self, in_channels, out_channels, activation_layer, norm_layer, pool: bool = False,
                 self_attention: bool = False):
        super(DoubleConvSkip, self).__init__(
            in_channels, out_channels, activation_layer, norm_layer, pool, self_attention
        )
        self.skip = ConvNormActivation(in_channels, out_channels, 1) if in_channels != out_channels else nn.Sequential()

    def forward(self, x):
        x = self.pool(x)
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + skip


class StandardNet(BaseModule):
    def __init__(self, in_channels: int, activation_layer: nn.Module, norm_layer: nn.Module,
                 skip_connection: bool = True, initial_stride: int = 1, **kwargs):
        super(StandardNet, self).__init__()
        assert initial_stride in [1, 2], "standard model only supports initial_stride of 1 or 2"
        block = DoubleConvSkip if skip_connection else DoubleConv

        depths = [in_channels, 64, 128, 256, 512, 1024]
        layers = [
            block(depths[0], depths[1], activation_layer, norm_layer, pool=initial_stride == 2),
            block(depths[1], depths[2], activation_layer, norm_layer, pool=True),
            block(depths[2], depths[3], activation_layer, norm_layer, pool=True),
            block(depths[3], depths[4], activation_layer, norm_layer, pool=True),
            block(depths[4], depths[5], activation_layer, norm_layer, pool=True),
        ]

        self.blocks = nn.ModuleList(layers)


def standard(**kwargs):
    return StandardNet(**kwargs)
