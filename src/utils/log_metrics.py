import logging
from functools import reduce

import numpy as np
import torch
from numpy import nan_to_num
from tensorboardX import SummaryWriter

from src.utils.jaccard import jaccard_from_confmat
from src.visualization.visualize import plot_confusion_matrix


def log_scalar(value, log_name, writer, global_step):
    if isinstance(value, torch.Tensor):
        value = value.cpu().item()
    logging.info(f"{log_name}: {value}")
    writer.add_scalar(log_name, value, global_step)
    return {log_name: value}


def log_cmat_stats(cmat, split_name: str, writer: SummaryWriter, global_step: int, class_names: list):
    stats = {}
    numel = cmat.sum(1)
    mask = numel > 0
    if mask.sum() == 0:  # nothing to log
        return stats
    tp = torch.diag(cmat)[mask]
    stats[f"tp/{split_name}"] = tp.sum().item()
    fp = (cmat.sum(0)[mask] - tp)
    stats[f"fp/{split_name}"] = fp.sum().item()
    fn = (cmat.sum(1)[mask] - tp)
    stats[f"accuracy/{split_name}"] = (tp.sum() / numel.sum())

    # macro statistics
    miou = jaccard_from_confmat(cmat, num_classes=cmat.shape[0], average="none")[mask]
    stats[f"miou/{split_name}"] = miou.mean().item()

    acc = (tp / numel[mask])
    stats[f"macc/{split_name}"] = acc.mean().item()

    precision = tp / (tp + fp + torch.finfo(torch.float32).eps)
    stats[f"precision/{split_name}"] = precision.mean().item()

    recall = tp / (tp + fn + torch.finfo(torch.float32).eps)
    stats[f"recall/{split_name}"] = recall.mean().item()

    f1 = 2 * ((precision * recall) / (precision + recall + torch.finfo(torch.float32).eps))
    stats[f"f1/{split_name}"] = f1.mean().item()

    # also log metrics per class stats
    if len(class_names) == 1:
        class_names = ["0"] + class_names
    for i, class_name in enumerate(np.array(class_names)[mask.cpu().numpy()]):
        stats[f"iou/{split_name}/{class_name}"] = miou[i]
        stats[f"accuracy/{split_name}/{class_name}"] = acc[i]
        stats[f"tp/{split_name}/{class_name}"] = tp[i]
        stats[f"recall/{split_name}/{class_name}"] = recall[i]
        stats[f"precision/{split_name}/{class_name}"] = precision[i]
        stats[f"f1/{split_name}/{class_name}"] = f1[i]

    # logging and resaving dict to ensure that it is not a torch Tensor anymore
    stats = reduce(lambda a, b: {**a, **b}, [log_scalar(stats[key], key, writer, global_step) for key in stats])

    logging.info(f"confusion_matrix/{split_name}: {cmat}")
    # normalize conf matrix
    cmatn = cmat / cmat.sum(axis=1, keepdim=True)
    cmatn = nan_to_num(cmatn.cpu().numpy())
    writer.add_figure(f"{split_name}/confusion_matrix", plot_confusion_matrix(cmatn, class_names), global_step)
    return stats
