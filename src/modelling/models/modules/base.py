import math

from torch.nn import Module, init
from torch import Tensor, nn


class BaseModule(Module):

    def forward(self, x: Tensor):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs[-1], outs[:-1]

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                nn.InstanceNorm2d, nn.InstanceNorm3d
        )):
            if (hasattr(m, "affine") and m.affine) or (hasattr(m, "elementwise_affine") and m.elementwise_affine):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # also resetting running stats
                m.reset_running_stats()

        elif isinstance(m, nn.Linear):
            init_range = 1.0 / math.sqrt(m.out_features)
            nn.init.uniform_(m.weight, -init_range, init_range)
            nn.init.zeros_(m.bias)
