from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.evaluation.evaluate_model import evaluate_validation_model, sliding_window_logits
from ml.training.config import ThresholdConfig


class DifferenceLogitModel(nn.Module):
    downsampling_factor = 1

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return (t2[:, :1] - t1[:, :1]) * 20.0 - 10.0


class TinyValidationDataset:
    split = "val"

    def __init__(self):
        self.t1 = torch.zeros(3, 32, 32)
        self.t2 = self.t1.clone()
        self.t2[:, 8:24, 8:24] = 1.0
        self.target = torch.zeros(1, 32, 32)
        self.target[:, 8:24, 8:24] = 1.0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return {"id": "val_sample", "t1": self.t1, "t2": self.t2, "mask": self.target}


def test_sliding_window_averages_overlapping_logits_at_full_image_shape():
    t1 = torch.zeros(3, 32, 32)
    t2 = t1.clone()
    t2[:, 8:24, 8:24] = 1.0
    logits = sliding_window_logits(
        DifferenceLogitModel(), t1, t2, patch_size=16, stride=8, device=torch.device("cpu")
    )
    assert logits.shape == (32, 32)
    assert logits[16, 16].item() == pytest.approx(10.0)
    assert logits[0, 0].item() == pytest.approx(-10.0)


def test_validation_evaluator_reports_global_metrics_and_validation_threshold_only():
    result = evaluate_validation_model(
        DifferenceLogitModel(),
        TinyValidationDataset(),
        device=torch.device("cpu"),
        patch_size=16,
        stride=8,
        threshold=ThresholdConfig(policy="validation_iou_search", candidates=(0.4, 0.5, 0.6)),
    )
    assert result["split"] == "val"
    assert result["selected_threshold"] == 0.4
    assert result["tp"] == 256
    assert result["fp"] == result["fn"] == 0
    assert result["iou"] == result["f1"] == 1.0
    assert result["per_image"][0]["id"] == "val_sample"


def test_validation_evaluator_rejects_non_validation_datasets():
    dataset = TinyValidationDataset()
    dataset.split = "test"
    with pytest.raises(ValueError, match="validation-only"):
        evaluate_validation_model(
            DifferenceLogitModel(), dataset, device=torch.device("cpu"), patch_size=16, stride=8,
            threshold=ThresholdConfig(),
        )
