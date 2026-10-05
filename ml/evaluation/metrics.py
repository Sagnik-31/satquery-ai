"""Binary change-segmentation metrics with explicit empty-mask semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryConfusion:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def as_dict(self) -> dict[str, int]:
        return {
            "tp": self.true_positive,
            "tn": self.true_negative,
            "fp": self.false_positive,
            "fn": self.false_negative,
        }


def binary_confusion(prediction: np.ndarray, target: np.ndarray) -> BinaryConfusion:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target masks must have identical shapes.")
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    return BinaryConfusion(
        true_positive=int(np.count_nonzero(pred & truth)),
        true_negative=int(np.count_nonzero(~pred & ~truth)),
        false_positive=int(np.count_nonzero(pred & ~truth)),
        false_negative=int(np.count_nonzero(~pred & truth)),
    )


def merge_confusions(confusions: list[BinaryConfusion]) -> BinaryConfusion:
    return BinaryConfusion(
        **{
            field: sum(getattr(confusion, field) for confusion in confusions)
            for field in BinaryConfusion.__dataclass_fields__
        }
    )


def metrics_from_confusion(confusion: BinaryConfusion) -> dict[str, float]:
    """Return precision/recall/F1/IoU.

    When both prediction and ground truth are empty, all overlap metrics are
    defined as 1.0 because the unchanged-pair prediction is exact. For any
    other zero denominator, the corresponding metric is 0.0.
    """
    tp, fp, fn = confusion.true_positive, confusion.false_positive, confusion.false_negative
    empty_both = tp == fp == fn == 0
    precision = 1.0 if empty_both else (tp / (tp + fp) if tp + fp else 0.0)
    recall = 1.0 if empty_both else (tp / (tp + fn) if tp + fn else 0.0)
    f1 = 1.0 if empty_both else (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    iou = 1.0 if empty_both else (tp / (tp + fp + fn) if tp + fp + fn else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
