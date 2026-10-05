from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.evaluation.evaluate_model import (
    evaluate_validation_model,
    frozen_threshold_from_checkpoint,
    sliding_window_logits,
    verify_checkpoint_configuration,
)
from ml.training.config import ModelConfig, ThresholdConfig, TrainingConfig
from ml.training.checkpointing import load_checkpoint, save_checkpoint


class DifferenceLogitModel(nn.Module):
    downsampling_factor = 1

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return (t2[:, :1] - t1[:, :1]) * 20.0 - 10.0


class ConstantLogitModel(nn.Module):
    downsampling_factor = 1

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return torch.full((t1.shape[0], 1, *t1.shape[-2:]), self.value, device=t1.device)


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


def test_batched_tiles_match_single_tile_aggregation():
    t1 = torch.zeros(3, 32, 32)
    t2 = t1.clone()
    t2[:, 8:24, 8:24] = 1.0
    single = sliding_window_logits(
        DifferenceLogitModel(), t1, t2, patch_size=16, stride=8, device=torch.device("cpu"), tile_batch_size=1
    )
    batched = sliding_window_logits(
        DifferenceLogitModel(), t1, t2, patch_size=16, stride=8, device=torch.device("cpu"), tile_batch_size=4
    )
    assert batched.shape == single.shape == (32, 32)
    assert torch.equal(batched, single)


def test_sliding_window_reconstructs_1024_with_256_tiles_and_128_stride():
    image = torch.zeros(3, 1024, 1024)
    logits = sliding_window_logits(
        ConstantLogitModel(2.5), image, image, patch_size=256, stride=128, device=torch.device("cpu")
    )
    assert logits.shape == (1024, 1024)
    # A constant prediction remains constant regardless of overlap count;
    # coverage gaps would instead raise inside sliding_window_logits.
    assert torch.all(logits == 2.5)


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


def _checkpoint_metadata(config: TrainingConfig, *, threshold: float = 0.6) -> dict:
    return {
        "architecture": "SiameseUNet",
        "config_sha256": config.fingerprint(),
        "channel_schema": {"t1": config.model.input_channels, "t2": config.model.input_channels, "output_logits": 1},
        "selected_validation_threshold": threshold,
    }


def test_matching_checkpoint_configuration_and_frozen_threshold_succeed():
    config = TrainingConfig()
    payload = _checkpoint_metadata(config, threshold=0.6)
    verify_checkpoint_configuration(payload, config)
    threshold = frozen_threshold_from_checkpoint(payload)
    assert threshold.policy == "fixed"
    assert threshold.value == 0.6


def test_checkpoint_configuration_hash_mismatch_fails():
    config = TrainingConfig()
    payload = _checkpoint_metadata(config)
    payload["config_sha256"] = "not-the-config-hash"
    with pytest.raises(ValueError, match="configuration hash"):
        verify_checkpoint_configuration(payload, config)


def test_checkpoint_channel_schema_mismatch_fails():
    config = TrainingConfig()
    payload = _checkpoint_metadata(config)
    payload["channel_schema"] = {"t1": 4, "t2": 4, "output_logits": 1}
    with pytest.raises(ValueError, match="channel schema"):
        verify_checkpoint_configuration(payload, config)


def test_saved_checkpoint_contains_frozen_metadata_and_sha256_sidecar(tmp_path):
    model = nn.Conv2d(1, 1, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    config = TrainingConfig()
    destination = tmp_path / "checkpoint.pt"
    checkpoint_hash = save_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        config=config.as_dict(),
        normalization={"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]},
        validation={"selected_threshold": 0.6, "iou": 0.2},
        provenance={
            "architecture": "SiameseUNet",
            "channel_schema": {"t1": 3, "t2": 3, "output_logits": 1},
            "config_sha256": config.fingerprint(),
        },
        dataloader_generator_state=torch.Generator().manual_seed(123).get_state(),
        history=[{"epoch": 1, "validation": {"iou": 0.2}}],
        best_validation_iou=0.2,
        best_checkpoint_epoch=1,
    )
    payload = load_checkpoint(destination, device=torch.device("cpu"))
    assert payload["selected_validation_threshold"] == 0.6
    assert payload["config_sha256"] == config.fingerprint()
    assert destination.with_suffix(".pt.sha256").read_text().strip() == checkpoint_hash
