"""Checkpoint payloads with enough metadata for a reproducible reload."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from torch import nn


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
) -> None:
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
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        destination,
    )


def load_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config",
        "normalization", "validation", "provenance", "random_state", "epoch",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required metadata: {sorted(missing)}")
    return payload
