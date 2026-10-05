"""Checkpoint payloads with enough metadata for a reproducible reload."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RestoredTrainingState:
    start_epoch: int
    history: list[dict[str, Any]]
    best_validation_iou: float
    best_checkpoint_epoch: int | None
    resume_exact: bool


def checkpoint_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_new_epoch_checkpoint(path: str | Path) -> Path:
    """Refuse accidental overwrite when a resumed run reaches an old epoch."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing epoch checkpoint: {destination}")
    return destination


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: dict[str, Any],
    normalization: dict[str, list[float]],
    validation: dict[str, Any],
    provenance: dict[str, Any],
    dataloader_generator_state: torch.Tensor,
    history: list[dict[str, Any]],
    best_validation_iou: float,
    best_checkpoint_epoch: int | None,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "normalization": normalization,
            "validation": validation,
            "provenance": provenance,
            "architecture": provenance["architecture"],
            "channel_schema": provenance["channel_schema"],
            "config_sha256": provenance["config_sha256"],
            "selected_validation_threshold": validation["selected_threshold"],
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "dataloader_generator_state": dataloader_generator_state.cpu(),
            "history": history,
            "best_validation_iou": best_validation_iou,
            "best_checkpoint_epoch": best_checkpoint_epoch,
        },
        destination,
    )
    checkpoint_hash = checkpoint_sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(checkpoint_hash + "\n")
    return checkpoint_hash


def load_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config",
        "normalization", "validation", "provenance", "random_state", "epoch",
        "architecture", "channel_schema", "config_sha256", "selected_validation_threshold",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required metadata: {sorted(missing)}")
    return payload


def restore_training_state(
    payload: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader_generator: torch.Generator,
    allow_nonexact_resume: bool = False,
) -> RestoredTrainingState:
    """Restore all state needed to continue at the epoch after a checkpoint.

    Pre-resume-support checkpoints lack the DataLoader generator/history fields;
    they require explicit acknowledgement because their next shuffle cannot be
    reproduced bit-for-bit.
    """
    has_exact_state = all(
        key in payload
        for key in ("dataloader_generator_state", "history", "best_validation_iou", "best_checkpoint_epoch")
    )
    if not has_exact_state and not allow_nonexact_resume:
        raise ValueError(
            "Checkpoint predates exact resume support; rerun with --allow-nonexact-resume to acknowledge "
            "that the DataLoader shuffle state cannot be reproduced."
        )
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    state = payload["random_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if has_exact_state:
        dataloader_generator.set_state(payload["dataloader_generator_state"])
        history = list(payload["history"])
        best_iou = float(payload["best_validation_iou"])
        best_epoch = payload["best_checkpoint_epoch"]
    else:
        history = []
        best_iou = float(payload["validation"]["iou"])
        best_epoch = int(payload["epoch"])
    return RestoredTrainingState(
        start_epoch=int(payload["epoch"]) + 1,
        history=history,
        best_validation_iou=best_iou,
        best_checkpoint_epoch=best_epoch,
        resume_exact=has_exact_state,
    )
