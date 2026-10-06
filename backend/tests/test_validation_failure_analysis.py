from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.datasets.levir_cd import RGBNormalization
from ml.evaluation.analyze_validation_failures import analyze_validation_failures, summarize_validation_scenes


class DifferenceLogitModel(nn.Module):
    downsampling_factor = 1

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return (t2[:, :1] - t1[:, :1]) * 20.0 - 10.0


class TinyValidationDataset:
    split = "val"

    def __init__(self) -> None:
        self.t1 = torch.zeros(3, 32, 32)
        self.t2 = self.t1.clone()
        self.t2[:, 8:24, 8:24] = 1.0
        self.mask = torch.zeros(1, 32, 32)
        self.mask[:, 8:24, 8:24] = 1.0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return {"id": "worst_val", "t1": self.t1, "t2": self.t2, "mask": self.mask}


def test_failure_analysis_ranks_scenes_and_writes_validation_artifacts(tmp_path):
    result = analyze_validation_failures(
        DifferenceLogitModel(), TinyValidationDataset(), device=torch.device("cpu"),
        patch_size=16, stride=8, threshold=0.5, tile_batch_size=2,
        normalization=RGBNormalization(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        artifact_dir=tmp_path, max_artifacts=1,
    )
    assert result["split"] == "val"
    assert result["mean_iou"] == result["median_iou"] == 1.0
    assert result["mean_f1"] == result["median_f1"] == 1.0
    assert result["ranked_by_iou_ascending"][0]["id"] == "worst_val"
    assert result["ranked_by_iou_ascending"][0]["tp"] == 256
    assert result["artifact_scene_ids"] == ["worst_val"]
    for suffix in ("t1", "t2", "ground_truth", "prediction", "error"):
        assert (tmp_path / f"worst_val_{suffix}.png").is_file()


def test_failure_summary_identifies_false_positive_and_false_negative_dominance():
    summary = summarize_validation_scenes([
        {"id": "fp", "iou": 0.1, "f1": 0.2, "tp": 1, "tn": 1, "fp": 10, "fn": 1, "error_imbalance": "false_positive_dominant"},
        {"id": "fn", "iou": 0.2, "f1": 0.3, "tp": 1, "tn": 1, "fp": 1, "fn": 10, "error_imbalance": "false_negative_dominant"},
    ])
    assert [record["id"] for record in summary["ranked_by_iou_ascending"]] == ["fp", "fn"]
    assert [record["id"] for record in summary["severe_false_positive_scenes"]] == ["fp"]
    assert [record["id"] for record in summary["severe_false_negative_scenes"]] == ["fn"]


def test_failure_analysis_rejects_non_validation_split():
    dataset = TinyValidationDataset()
    dataset.split = "test"
    with pytest.raises(ValueError, match="validation-only"):
        analyze_validation_failures(DifferenceLogitModel(), dataset, device=torch.device("cpu"), patch_size=16, stride=8, threshold=0.5, tile_batch_size=2)
