from src.modelling.models.modules.base import BaseModule
from torch import nn, Tensor


class TwoConvNet(BaseModule):
    def __init__(self, in_channels: int, **kwargs):
        super(TwoConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, (3, 3), 1, 1)
        self.conv2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        layers = [self.conv1,
                  self.conv2]
        self.blocks = nn.ModuleList(layers)

def twoconvnet(**kwargs):
    model = TwoConvNet(**kwargs)
    return model