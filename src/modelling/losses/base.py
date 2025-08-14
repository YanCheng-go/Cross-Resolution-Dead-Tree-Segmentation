from functools import partial

import torch
import torch.nn as nn
from kornia.losses import BinaryFocalLossWithLogits, FocalLoss, TverskyLoss
from src.modelling.losses.focal import BinaryFocalLossWithLogits
from torch import Tensor

from src.modelling.losses.tversky import BinaryTverskyLoss, FocalTverskyLoss


def get_regression_loss(name: str, **kwargs):
    try:
        loss = regression_losses[name]
    except KeyError:
        raise KeyError(f"Loss '{name}' not implemented")
    return loss(**kwargs)


class DistillLoss(nn.KLDivLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # assumes that input is logits and target is softmax probs
        return super().forward(input.log_softmax(0), target)


class DistillLossBinary(nn.KLDivLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # assumes that input is logits and target is sigmoid probs
        input = torch.stack([nn.functional.logsigmoid(input), nn.functional.logsigmoid(-input)], 1)
        target = torch.stack([target, 1 - target], 1)

        return super().forward(nn.functional.logsigmoid(input), target)


regression_losses = {
    "smooth_l1": nn.SmoothL1Loss,
    "smoothl1": nn.SmoothL1Loss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "kld": nn.KLDivLoss,
    "kd_binary": partial(DistillLossBinary, size_average=False),
    "kd": partial(DistillLoss, size_average=False)
}


def get_classification_loss(name: str, **kwargs):
    try:
        loss = classification_losses[name]
    except KeyError:
        raise KeyError(f"Loss '{name}' not implemented")
    return loss(**kwargs)


class CE_Plus_BCE(nn.Module):
    def __init__(self, bce_wight=0.33333):
        super().__init__()
        self.bce_weight = bce_wight
        self.bce = nn.BCELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, inp: Tensor, target: Tensor) -> Tensor:
        c = self.ce(inp, target)
        inp_p = torch.softmax(inp, dim=1)
        inp_b = 0.5 * ((1 - inp_p[:, 0]) + torch.sum(inp_p[:, 1:], dim=1,
                                                     keepdim=False))  # N, d1, d2; probability of positive class = 1 - probability negative class
        target_b = target.clone().float()
        target_b[target_b > 0] = 1.  # Replace class labels with 1
        b = self.bce(inp_b, target_b)
        return b * self.bce_weight + c * (1 - self.bce_weight)


class CE_Plus_BCE2(nn.Module):
    def __init__(self, bce_wight=0.33333):
        super().__init__()
        self.bce_weight = bce_wight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, inp: Tensor, target: Tensor) -> Tensor:
        c = self.ce(inp, target)

        target_b = target.clone().float()
        target_b[target_b > 0] = 1.  # Replace class labels with 1

        b1 = self.bce(-inp[:, 0], target_b)
        b2 = self.bce(torch.sum(inp[:, 1:], dim=1, keepdim=False), target_b)

        return 0.5 * (b1 + b2) * self.bce_weight + c * (1 - self.bce_weight)


class KorniaLossWrapper(nn.Module):
    def __init__(self, loss_cls):
        super().__init__()
        self.loss = loss_cls()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.loss(input.unsqueeze(-1), target.unsqueeze(-1), weights.unsqueeze(-1))


classification_losses = {
    "ce": nn.CrossEntropyLoss,
    "focal": partial(FocalLoss, alpha=0.25, reduction="mean"),
    "focal5": partial(FocalLoss, alpha=0.125, gamma=5., reduction="mean"),
    "ce+bce": CE_Plus_BCE,
    "ce+bce2": CE_Plus_BCE2,
    "tversky": partial(KorniaLossWrapper, partial(TverskyLoss, alpha=0.3, beta=0.7)),
    "tversky_g2": partial(KorniaLossWrapper, partial(FocalTverskyLoss, alpha=0.3, beta=0.7, gamma=2)),
    "tversky46": partial(KorniaLossWrapper, partial(TverskyLoss, alpha=0.4, beta=0.6)),
    "tversky46_g2": partial(KorniaLossWrapper, partial(FocalTverskyLoss, alpha=0.4, beta=0.6, gamma=2)),
    "tversky64": partial(KorniaLossWrapper, partial(TverskyLoss, alpha=0.6, beta=0.4)),
    "tversky64_g2": partial(KorniaLossWrapper, partial(FocalTverskyLoss, alpha=0.6, beta=0.4, gamma=2)),
}


def get_binary_classification_loss(name: str, **kwargs):
    try:
        loss = binary_classification_losses[name]
    except KeyError:
        raise KeyError(f"Loss '{name}' not implemented")
    return loss(**kwargs)


binary_classification_losses = {
    "bce": nn.BCEWithLogitsLoss,
    "focal": partial(BinaryFocalLossWithLogits, alpha=0.25, reduction="mean"),
    "focal5": partial(BinaryFocalLossWithLogits, alpha=0.125, gamma=5., reduction="mean"),
    "tversky": partial(BinaryTverskyLoss, alpha=0.3, beta=0.7, gamma=1),
    "tversky_g2": partial(BinaryTverskyLoss, alpha=0.3, beta=0.7, gamma=2),
    "tversky64": partial(BinaryTverskyLoss, alpha=0.6, beta=0.4, gamma=1),
    "tversky64_g2": partial(BinaryTverskyLoss, alpha=0.6, beta=0.4, gamma=2),
    "tversky46": partial(BinaryTverskyLoss, alpha=0.4, beta=0.6, gamma=1),
    "tversky46_g2": partial(BinaryTverskyLoss, alpha=0.4, beta=0.6, gamma=2),
    "tversky28_g2": partial(BinaryTverskyLoss, alpha=0.2, beta=0.8, gamma=2),
}
