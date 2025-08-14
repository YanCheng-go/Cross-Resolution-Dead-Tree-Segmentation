import torch
from torch import nn
from torch.nn.functional import one_hot


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float = 1, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights=None) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft: torch.Tensor = preds.softmax(dim=1)

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = one_hot(targets, num_classes=preds.shape[1]).swapaxes(1, 2)

        if weights is None:
            weights = torch.ones_like(input_soft).to(input_soft.device)

        # compute the actual dice score
        dims = (1, 2) if len(preds.shape) == 3 else (1, 2, 3)
        intersection = torch.sum(weights * input_soft * target_one_hot, dims)
        fps = torch.sum(weights * input_soft * (-target_one_hot + 1.0), dims)
        fns = torch.sum(weights * (-input_soft + 1.0) * target_one_hot, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)

        return (1. - tversky_loss).pow(1 / self.gamma).mean()


class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6, gamma: float = 1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.gamma = gamma

    def forward(self, preds, targets, weights=None):
        preds = torch.sigmoid(preds)
        if weights is not None:
            preds = preds * weights

        # True Positives, False Positives & False Negatives
        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()

        tversky = TP / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1. - tversky).pow(1 / self.gamma)
