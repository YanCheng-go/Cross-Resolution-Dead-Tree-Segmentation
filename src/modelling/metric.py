from functools import partial

import numpy as np
import torch
import torchmetrics
from sklearn import metrics

# uses various torchemetric classes
from src.visualization.visualize import image_from_confusion_matrix

_metrics_dict = {
    "ap": metrics.average_precision_score,
    "roc_auc": metrics.roc_auc_score,
    "rmse": partial(metrics.mean_squared_error, squared=False),
    "r2": metrics.r2_score,
}
_binary_metrics_dict = {
    "accuracy": lambda y, pred: metrics.accuracy_score(y, (pred > 0).astype(int)),
    "precision": lambda y, pred: metrics.precision_score(y, (pred > 0).astype(int)),
    "recall": lambda y, pred: metrics.recall_score(y, (pred > 0).astype(int)),
    "f1": lambda y, pred: metrics.f1_score(y, (pred > 0).astype(int)),
    "ap": metrics.average_precision_score,
    **_metrics_dict
}
_multi_metrics_dict = {
    "accuracy": lambda y, pred: metrics.accuracy_score(y, np.argmax(pred, 1)),
    "precision": lambda y, pred: metrics.precision_score(y, np.argmax(pred, 1), average="macro"),
    "recall": lambda y, pred: metrics.recall_score(y, np.argmax(pred, 1), average="macro"),
    "f1": lambda y, pred: metrics.f1_score(y, np.argmax(pred, 1), average="macro"),
    **_metrics_dict
}
_batch_metrics_dict = {
    # classification/segmentation
    "accuracy": partial(torchmetrics.Accuracy, compute_on_step=False),
    "ap": partial(torchmetrics.AveragePrecision, compute_on_step=False),
    "rocauc": partial(torchmetrics.AUROC, compute_on_step=False),
    "f1": partial(torchmetrics.F1Score, compute_on_step=False),
    "precision": partial(torchmetrics.Precision, compute_on_step=False),
    "recall": partial(torchmetrics.Recall, compute_on_step=False),
    "kappa": partial(torchmetrics.CohenKappa, compute_on_step=False),
    # regression
    "r2": partial(torchmetrics.R2Score, compute_on_step=False),
    "mae": partial(torchmetrics.MeanAbsoluteError, compute_on_step=False),
    "mse": partial(torchmetrics.MeanSquaredError, compute_on_step=False, squared=False),
    "rmse": partial(torchmetrics.MeanSquaredError, compute_on_step=False, squared=True),
}


def get_binary_metric(metric):
    """
    returns a metric if possible
    @param metric: name of metric (e.g., 'accuracy', 'precision', 'recall', 'ap', 'roc_auc', 'f1', "rmse", "r2")
    """
    return _binary_metrics_dict[metric]


def get_multi_metric(metric):
    """
    returns a metric if possible
    @param metric: name of metric (e.g., 'accuracy', 'precision', 'recall', 'ap', 'roc_auc', 'f1', "rmse", "r2")
    """
    return _multi_metrics_dict[metric]


def get_batch_metric(metric):
    """
    returns a metric if possible
    @param metric: name of metric (e.g., "rmse", "r2", "mae", "rmse)
    """
    return _batch_metrics_dict[metric]


def get_confusion_matrix(preds, target, mask=None, confusion_matrix=None):
    # Borrowed from torchmetrics/functional/classification/confusion_matrix.py; def _confusion_matrix_update
    if mask is None:
        mask = torch.ones_like(target).bool()
    nb_classes = preds.shape[1]
    preds = torch.argmax(preds, 1, keepdim=False)
    preds = preds[mask]  # 1D tensor
    target = target[mask]  # 1D tensor

    unique_mapping = (target * nb_classes + preds).to(torch.long)
    minlength = nb_classes ** 2

    bins = torch.bincount(unique_mapping, minlength=minlength)
    confmat = bins.reshape(nb_classes, nb_classes)

    if confusion_matrix is None:
        confusion_matrix = confmat
    else:
        confusion_matrix += confmat
    return confusion_matrix
