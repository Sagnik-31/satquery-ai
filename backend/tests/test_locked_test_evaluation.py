from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from ml.evaluation.evaluate_locked_test import (
    FrozenCheckpoint,
    evaluate_locked_dataset,
    main,
    require_locked_confirmation,
    require_test_split,
    verify_frozen_checkpoint,
)


class DifferenceLogitModel(nn.Module):
    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return (t2[:, :1] - t1[:, :1]) * 20.0 - 10.0


class SyntheticLockedTestDataset:
    split = "test"

    def __init__(self):
        self.t1 = torch.zeros(3, 1024, 1024)
        self.t2 = self.t1.clone()
        self.t2[:, 256:768, 256:768] = 1.0
        self.mask = torch.zeros(1, 1024, 1024)
        self.mask[:, 256:768, 256:768] = 1.0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return {"id": "synthetic_test", "t1": self.t1, "t2": self.t2, "mask": self.mask}


def _frozen(path: Path, payload: dict, *, checkpoint_hash: str | None = None) -> FrozenCheckpoint:
    provenance = payload["provenance"]
    return FrozenCheckpoint(
        checkpoint=path.resolve(),
        sha256=checkpoint_hash or hashlib.sha256(path.read_bytes()).hexdigest(),
        epoch=17,
        threshold=0.5,
        config_sha256="config-hash",
        git_revision="revision",
        dataset_fingerprint="dataset-hash",
        split_manifest_sha256="manifest-hash",
    )


def _payload() -> dict:
    return {
        "epoch": 17,
        "architecture": "SiameseUNet",
        "channel_schema": {"t1": 3, "t2": 3, "output_logits": 1},
        "config_sha256": "config-hash",
        "selected_validation_threshold": 0.5,
        "provenance": {
            "git_revision": "revision",
            "dataset_fingerprint": {"development_splits_sha256": "dataset-hash"},
            "split_manifest_sha256": "manifest-hash",
        },
    }


def test_locked_evaluator_refuses_without_confirmation_before_dataset_access():
    with pytest.raises(PermissionError, match="confirm-locked-test"):
        require_locked_confirmation(False)


def test_cli_confirmation_gate_runs_before_any_checkpoint_or_dataset_access(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_locked_test", "--dataset-root", "/not/read", "--checkpoint", "/not/read/checkpoint.pt"],
    )
    with pytest.raises(PermissionError, match="confirm-locked-test"):
        main()


@pytest.mark.parametrize("split", ["train", "val", "development"])
def test_locked_evaluator_rejects_any_non_test_split(split):
    with pytest.raises(ValueError, match="only the official 'test'"):
        require_test_split(split)


def test_frozen_checkpoint_hash_and_metadata_verification(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen-weights")
    payload = _payload()
    verify_frozen_checkpoint(checkpoint, payload, _frozen(checkpoint, payload))


def test_frozen_checkpoint_rejects_hash_config_and_channel_mismatches(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen-weights")
    payload = _payload()
    with pytest.raises(ValueError, match="SHA-256"):
        verify_frozen_checkpoint(checkpoint, payload, _frozen(checkpoint, payload, checkpoint_hash="wrong"))
    payload["config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="config hash"):
        verify_frozen_checkpoint(checkpoint, payload, _frozen(checkpoint, _payload()))
    payload = _payload()
    payload["channel_schema"] = {"t1": 4, "t2": 4, "output_logits": 1}
    with pytest.raises(ValueError, match="channel schema"):
        verify_frozen_checkpoint(checkpoint, payload, _frozen(checkpoint, _payload()))


def test_locked_evaluator_uses_frozen_threshold_without_selection_and_reports_global_metrics():
    result = evaluate_locked_dataset(
        DifferenceLogitModel(),
        SyntheticLockedTestDataset(),
        device=torch.device("cpu"),
        threshold=0.5,
        tile_batch_size=8,
    )
    assert result["split"] == "test"
    assert result["threshold"] == 0.5
    assert result["threshold_policy"] == "frozen_checkpoint_no_selection"
    assert result["tile_size"] == 256 and result["tile_stride"] == 128
    assert result["tp"] == 512 * 512
    assert result["tn"] == 1024 * 1024 - 512 * 512
    assert result["fp"] == result["fn"] == 0
    assert result["precision"] == result["recall"] == result["f1"] == result["iou"] == 1.0
    assert result["per_image"][0]["id"] == "synthetic_test"


def test_locked_evaluator_cannot_accept_train_or_validation_dataset():
    dataset = SyntheticLockedTestDataset()
    dataset.split = "val"
    with pytest.raises(ValueError, match="other than split='test'"):
        evaluate_locked_dataset(DifferenceLogitModel(), dataset, device=torch.device("cpu"), threshold=0.5, tile_batch_size=8)
