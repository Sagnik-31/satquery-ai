"""Typed configuration loading for the LEVIR-CD v1 learned baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ThresholdConfig:
    policy: str = "fixed"
    value: float = 0.5
    candidates: tuple[float, ...] = (0.5,)

    def __post_init__(self) -> None:
        if self.policy not in {"fixed", "validation_iou_search"}:
            raise ValueError("threshold.policy must be 'fixed' or 'validation_iou_search'.")
        if not 0.0 < self.value < 1.0:
            raise ValueError("threshold.value must be between 0 and 1.")
        if not self.candidates or any(not 0.0 < value < 1.0 for value in self.candidates):
            raise ValueError("threshold.candidates must contain values strictly between 0 and 1.")


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 3
    base_channels: int = 24


@dataclass(frozen=True)
class DatasetConfig:
    identifier: str = "LEVIR-CD"
    version: str = "local-official-split-unversioned"


@dataclass(frozen=True)
class CropSamplingConfig:
    positive_crop_fraction: float = 0.5
    negative_crop_attempts: int = 32

    def __post_init__(self) -> None:
        if not 0.0 <= self.positive_crop_fraction <= 1.0:
            raise ValueError("sampling.positive_crop_fraction must be between 0 and 1.")
        if self.negative_crop_attempts <= 0:
            raise ValueError("sampling.negative_crop_attempts must be positive.")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20261005
    device: str = "auto"
    patch_size: int = 256
    batch_size: int = 4
    epochs: int = 20
    num_workers: int = 0
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_t_max: int = 20
    validation_tile_batch_size: int = 8
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    selection_metric: str = "iou"
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: CropSamplingConfig = field(default_factory=CropSamplingConfig)

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda.")
        if self.patch_size <= 0 or self.patch_size % 16:
            raise ValueError("patch_size must be positive and divisible by 16.")
        if self.batch_size <= 0 or self.epochs <= 0 or self.num_workers < 0:
            raise ValueError("batch_size/epochs must be positive and num_workers non-negative.")
        if self.validation_tile_batch_size <= 0:
            raise ValueError("validation_tile_batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.optimizer != "adamw" or self.scheduler != "cosine":
            raise ValueError("v1 supports optimizer=adamw and scheduler=cosine only.")
        if self.selection_metric != "iou":
            raise ValueError("v1 selects checkpoints by global validation IoU only.")
        if self.bce_weight != 0.5 or self.dice_weight != 0.5:
            raise ValueError("v1 loss weights are fixed at 0.5 BCE and 0.5 Soft Dice.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        identity = self.as_dict()
        # Tile batch size changes throughput only. It does not change model
        # weights, tiles, overlap aggregation, thresholding, or metrics, so it
        # is deliberately excluded from scientific experiment identity.
        identity.pop("validation_tile_batch_size", None)
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be a mapping.")
    return value


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load the committed YAML config; dataset paths are CLI-only by design."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    threshold = ThresholdConfig(**_section(raw, "threshold"))
    model = ModelConfig(**_section(raw, "model"))
    dataset = DatasetConfig(**_section(raw, "dataset"))
    sampling = CropSamplingConfig(**_section(raw, "sampling"))
    training_values = _section(raw, "training")
    return TrainingConfig(**training_values, threshold=threshold, model=model, dataset=dataset, sampling=sampling)
