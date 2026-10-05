from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from ml.training.checkpointing import (
    ensure_new_epoch_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from ml.training.config import TrainingConfig
from ml.training.train_change_model import validate_resume_identity


def _components():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    generator = torch.Generator().manual_seed(77)
    return model, optimizer, scheduler, generator


def _provenance(config: TrainingConfig, *, dataset_fingerprint=None, manifest="manifest", git="revision"):
    return {
        "architecture": "SiameseUNet",
        "channel_schema": {"t1": 3, "t2": 3, "output_logits": 1},
        "config_sha256": config.fingerprint(),
        "dataset_fingerprint": dataset_fingerprint or {"development_splits_sha256": "dataset"},
        "split_manifest_sha256": manifest,
        "git_revision": git,
    }


def _save(tmp_path, *, config=None):
    config = config or TrainingConfig()
    model, optimizer, scheduler, generator = _components()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    destination = tmp_path / "epoch_001.pt"
    save_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        config=config.as_dict(),
        normalization={"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]},
        validation={"selected_threshold": 0.5, "iou": 0.4},
        provenance=_provenance(config),
        dataloader_generator_state=generator.get_state(),
        history=[{"epoch": 1, "validation": {"iou": 0.4}}],
        best_validation_iou=0.4,
        best_checkpoint_epoch=1,
    )
    return destination, model, optimizer, scheduler, generator, config


def test_checkpoint_round_trip_restores_optimizer_scheduler_rng_generator_and_history(tmp_path):
    path, _, _, _, generator, _ = _save(tmp_path)
    expected_generator = torch.rand(4, generator=generator)
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)
    restored_model, restored_optimizer, restored_scheduler, restored_generator = _components()
    random.random(); np.random.rand(); torch.rand(1); torch.rand(4, generator=restored_generator)
    state = restore_training_state(
        load_checkpoint(path, device=torch.device("cpu")),
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        dataloader_generator=restored_generator,
    )
    assert state.start_epoch == 2
    assert state.resume_exact is True
    assert state.history == [{"epoch": 1, "validation": {"iou": 0.4}}]
    assert state.best_validation_iou == 0.4
    assert state.best_checkpoint_epoch == 1
    assert torch.equal(torch.rand(4, generator=restored_generator), expected_generator)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    assert restored_optimizer.state_dict()["param_groups"] == load_checkpoint(path, device=torch.device("cpu"))["optimizer_state_dict"]["param_groups"]
    assert restored_scheduler.state_dict()["last_epoch"] == 0


def test_legacy_checkpoint_requires_explicit_nonexact_acknowledgement(tmp_path):
    path, *_ = _save(tmp_path)
    payload = load_checkpoint(path, device=torch.device("cpu"))
    for key in ("dataloader_generator_state", "history", "best_validation_iou", "best_checkpoint_epoch"):
        payload.pop(key)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    model, optimizer, scheduler, generator = _components()
    with pytest.raises(ValueError, match="allow-nonexact-resume"):
        restore_training_state(payload, model=model, optimizer=optimizer, scheduler=scheduler, dataloader_generator=generator)
    state = restore_training_state(
        payload, model=model, optimizer=optimizer, scheduler=scheduler, dataloader_generator=generator, allow_nonexact_resume=True
    )
    assert state.resume_exact is False
    assert state.start_epoch == 2


@pytest.mark.parametrize("field, value, message", [
    ("config", None, "config hash"),
    ("dataset", {"development_splits_sha256": "other"}, "dataset fingerprint"),
    ("schema", {"t1": 4, "t2": 4, "output_logits": 1}, "channel schema"),
])
def test_resume_identity_rejects_mismatches(field, value, message):
    config = TrainingConfig()
    payload = {"architecture": "SiameseUNet", "config_sha256": config.fingerprint(), "channel_schema": {"t1": 3, "t2": 3, "output_logits": 1}, "normalization": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}, "provenance": _provenance(config)}
    if field == "config":
        payload["config_sha256"] = "wrong"
    elif field == "dataset":
        payload["provenance"]["dataset_fingerprint"] = value
    else:
        payload["channel_schema"] = value
    with pytest.raises(ValueError, match=message):
        validate_resume_identity(payload, config=config, normalization={"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}, dataset_fingerprint={"development_splits_sha256": "dataset"}, split_manifest_sha256="manifest", git_revision="revision")


def test_resume_identity_rejects_normalization_and_manifest_mismatches():
    config = TrainingConfig()
    payload = {"architecture": "SiameseUNet", "config_sha256": config.fingerprint(), "channel_schema": {"t1": 3, "t2": 3, "output_logits": 1}, "normalization": {"mean": [9, 9, 9], "std": [1, 1, 1]}, "provenance": _provenance(config)}
    with pytest.raises(ValueError, match="normalization"):
        validate_resume_identity(payload, config=config, normalization={"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}, dataset_fingerprint={"development_splits_sha256": "dataset"}, split_manifest_sha256="manifest", git_revision="revision")
    payload["normalization"] = {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
    with pytest.raises(ValueError, match="split manifest"):
        validate_resume_identity(payload, config=config, normalization=payload["normalization"], dataset_fingerprint={"development_splits_sha256": "dataset"}, split_manifest_sha256="different", git_revision="revision")


def test_resumed_run_checkpoint_guard_does_not_overwrite_existing_epoch(tmp_path):
    path = tmp_path / "epoch_002.pt"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="overwrite"):
        ensure_new_epoch_checkpoint(path)
    assert path.read_bytes() == b"original"
