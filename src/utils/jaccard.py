# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
from torch import Tensor

def jaccard_from_confmat(
        confmat: Tensor,
        num_classes: int,
        average: Optional[str] = "macro",
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
) -> Tensor:
    """Computes the intersection over union from confusion matrix.
    Args:
        confmat: Confusion matrix without normalization
        num_classes: Number of classes for a given prediction and target tensor
        average:
            Defines the reduction that is applied. Should be one of the following:
            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class. Note that if a given class doesn't occur in the
              `preds` or `target`, the value for the class will be ``nan``.
        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.
        absent_score: score to use for an individual class, if no instances of the class index were present in `pred`
            AND no instances of the class index were present in `target`.
    """
    allowed_average = ["micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    # Remove the ignored class index from the scores.
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        confmat[ignore_index] = 0.0

    if average == "none" or average is None:
        intersection = torch.diag(confmat)
        union = confmat.sum(0) + confmat.sum(1) - intersection

        # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
        scores = intersection.float() / union.float()
        scores[union == 0] = absent_score

        if ignore_index is not None and 0 <= ignore_index < num_classes:
            scores = torch.cat(
                [
                    scores[:ignore_index],
                    scores[ignore_index + 1 :],
                ]
            )
        return scores

    if average == "macro":
        scores = jaccard_from_confmat(
            confmat, num_classes, average="none", ignore_index=ignore_index, absent_score=absent_score
        )
        return torch.mean(scores)

    if average == "micro":
        intersection = torch.sum(torch.diag(confmat))
        union = torch.sum(torch.sum(confmat, dim=1) + torch.sum(confmat, dim=0) - torch.diag(confmat))
        return intersection.float() / union.float()

    weights = torch.sum(confmat, dim=1).float() / torch.sum(confmat).float()
    scores = jaccard_from_confmat(
        confmat, num_classes, average="none", ignore_index=ignore_index, absent_score=absent_score
    )
    return torch.sum(weights * scores)