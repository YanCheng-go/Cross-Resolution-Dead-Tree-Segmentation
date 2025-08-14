import numpy as np
import torch

from src.modelling.metric import get_confusion_matrix


def test_confusion_metric():
    # Edge case; rectangle < patch
    cl = 5
    gt = np.array([[[1, 2, 3], [4, 0, 1], [2, 3, 4]]])
    assert gt.shape == (1, 3, 3)
    pred = np.zeros(gt.shape)
    pred = np.tile(pred, (1, cl, 1, 1))

    # Wrong ones
    pred[:, 2, 0, 0] = 1
    pred[:, 3, 0, 1] = 1
    pred[:, 4, 0, 2] = 1
    pred[:, 0, 1, 0] = 1
    # Correct ones
    pred[:, 0, 1, 1] = 1
    pred[:, 1, 1, 2] = 1
    pred[:, 2, 2, 0] = 1
    pred[:, 3, 2, 1] = 1
    pred[:, 4, 2, 2] = 1

    pred = torch.tensor(pred, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)
    cf = np.zeros((cl, cl), dtype=np.long)
    np.fill_diagonal(cf, 1)
    cf[1, 2] = 1
    cf[2, 3] = 1
    cf[3, 4] = 1
    cf[4, 0] = 1

    cf = torch.tensor(cf, dtype=torch.long)
    confusion_matrix = get_confusion_matrix(pred, gt)
    # logging.info("First\n")
    # logging.info(confusion_matrix)
    # logging.info(cf)
    assert torch.equal(cf, confusion_matrix)

    mask = torch.ones_like(gt).bool()

    # Ignore two wrong ones
    mask[:, 0, 0] = False
    mask[:, 0, 1] = False
    # Ignore one correct one
    mask[:, 2, 2] = False

    cf[1, 2] = 0
    cf[2, 3] = 0
    cf[4, 4] = 0

    confusion_matrix = get_confusion_matrix(pred, gt, mask)

    # logging.info("Second\n")
    # logging.info(confusion_matrix)
    # logging.info(cf)
    assert torch.equal(cf, confusion_matrix)


if __name__ == "__main__":
    test_confusion_metric()
