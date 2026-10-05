"""Validation-only full-scene evaluator for frozen learned change models.

This module has no split argument. It permits only ``val`` datasets so the
official LEVIR-CD test split cannot be selected through this workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from ml.datasets.levir_cd import LevirPairedDataset, RGBNormalization
from ml.evaluation.metrics import BinaryConfusion, binary_confusion, merge_confusions, metrics_from_confusion
from ml.models.siamese_unet import SiameseUNet
from ml.training.checkpointing import load_checkpoint
from ml.training.config import ThresholdConfig, TrainingConfig, load_training_config
from ml.training.reproducibility import select_device


def _window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        raise ValueError(f"Full-scene dimension {length} is smaller than patch size {patch_size}.")
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def sliding_window_logits(
    model: nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Average overlapping patch logits into one full-resolution logit map."""
    if t1.ndim != 3 or t2.ndim != 3 or t1.shape != t2.shape:
        raise ValueError("T1 and T2 must be matching unbatched [C, H, W] tensors.")
    if stride <= 0 or stride > patch_size:
        raise ValueError("stride must be positive and no larger than patch_size.")
    _, height, width = t1.shape
    summed = torch.zeros((height, width), dtype=torch.float32)
    counts = torch.zeros((height, width), dtype=torch.float32)
    for top in _window_starts(height, patch_size, stride):
        for left in _window_starts(width, patch_size, stride):
            patch_t1 = t1[:, top : top + patch_size, left : left + patch_size].unsqueeze(0).to(device)
            patch_t2 = t2[:, top : top + patch_size, left : left + patch_size].unsqueeze(0).to(device)
            logits = model(patch_t1, patch_t2).squeeze(0).squeeze(0).detach().cpu()
            summed[top : top + patch_size, left : left + patch_size] += logits
            counts[top : top + patch_size, left : left + patch_size] += 1
    if torch.any(counts == 0):
        raise RuntimeError("Sliding-window coverage failed for at least one validation pixel.")
    return summed / counts


def _candidate_thresholds(config: ThresholdConfig) -> tuple[float, ...]:
    return (config.value,) if config.policy == "fixed" else tuple(sorted(set(config.candidates)))


def verify_checkpoint_configuration(payload: dict[str, Any], config: TrainingConfig) -> None:
    """Reject config drift before evaluating a frozen checkpoint."""
    expected_schema = {
        "t1": config.model.input_channels,
        "t2": config.model.input_channels,
        "output_logits": 1,
    }
    if payload.get("architecture") != "SiameseUNet":
        raise ValueError(f"Checkpoint architecture {payload.get('architecture')!r} is not supported.")
    if payload.get("config_sha256") != config.fingerprint():
        raise ValueError(
            "Supplied configuration does not match the frozen checkpoint configuration hash."
        )
    if payload.get("channel_schema") != expected_schema:
        raise ValueError(
            f"Checkpoint channel schema {payload.get('channel_schema')!r} does not match supplied configuration {expected_schema!r}."
        )
    if not 0.0 < float(payload.get("selected_validation_threshold", -1)) < 1.0:
        raise ValueError("Checkpoint has no valid selected validation threshold.")


def frozen_threshold_from_checkpoint(payload: dict[str, Any]) -> ThresholdConfig:
    """Return the recorded validation threshold without a new threshold search."""
    return ThresholdConfig(policy="fixed", value=float(payload["selected_validation_threshold"]))


def evaluate_validation_model(
    model: nn.Module,
    dataset: LevirPairedDataset,
    *,
    device: torch.device,
    patch_size: int,
    stride: int,
    threshold: ThresholdConfig,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a model over full validation scenes and select only on validation."""
    if dataset.split != "val":
        raise ValueError("Learned-model evaluation is validation-only; dataset.split must be 'val'.")
    if patch_size % getattr(model, "downsampling_factor", 1):
        raise ValueError("patch_size must be divisible by the model downsampling factor.")
    model.eval()
    candidates = _candidate_thresholds(threshold)
    all_confusions: dict[float, list[BinaryConfusion]] = {value: [] for value in candidates}
    per_image: list[dict[str, Any]] = []
    cached_predictions: list[tuple[str, np.ndarray, np.ndarray]] = []

    for index in range(len(dataset)):
        example = dataset[index]
        t1, t2, target = example["t1"], example["t2"], example["mask"]
        assert isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor) and isinstance(target, torch.Tensor)
        logits = sliding_window_logits(model, t1, t2, patch_size=patch_size, stride=stride, device=device)
        probabilities = torch.sigmoid(logits).numpy()
        truth = target.squeeze(0).numpy().astype(bool)
        by_threshold: dict[float, BinaryConfusion] = {}
        for value in candidates:
            confusion = binary_confusion(probabilities >= value, truth)
            all_confusions[value].append(confusion)
            by_threshold[value] = confusion
        cached_predictions.append((str(example["id"]), probabilities, truth))
        per_image.append({"id": str(example["id"]), "by_threshold": {str(value): metrics_from_confusion(confusion) for value, confusion in by_threshold.items()}})

    aggregates = {value: merge_confusions(confusions) for value, confusions in all_confusions.items()}
    # Ties intentionally prefer the lower threshold, which is deterministic and
    # stored with the selected checkpoint/run metadata.
    selected = max(candidates, key=lambda value: (metrics_from_confusion(aggregates[value])["iou"], -value))
    aggregate = aggregates[selected]
    selected_per_image = []
    for item in per_image:
        selected_per_image.append({"id": item["id"], **item["by_threshold"][str(selected)]})

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, probabilities, truth in cached_predictions:
            prediction = probabilities >= selected
            error = np.zeros((*truth.shape, 3), dtype=np.uint8)
            error[(prediction & ~truth)] = (0, 0, 255)
            error[(~prediction & truth)] = (255, 0, 0)
            cv2.imwrite(str(artifact_dir / f"{sample_id}_prediction.png"), prediction.astype(np.uint8) * 255)
            cv2.imwrite(str(artifact_dir / f"{sample_id}_error.png"), error)

    return {
        "split": "val",
        "threshold_policy": threshold.policy,
        "selected_threshold": selected,
        "threshold_candidates": list(candidates),
        **aggregate.as_dict(),
        **metrics_from_confusion(aggregate),
        "per_image": selected_per_image,
    }


def _normalization_from_payload(payload: dict[str, Any]) -> RGBNormalization:
    raw = payload["normalization"]
    return RGBNormalization(mean=tuple(raw["mean"]), std=tuple(raw["std"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Local LEVIR-CD root. Validation only; no test option exists.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Frozen learned-model checkpoint.")
    parser.add_argument("--config", required=True, type=Path, help="Versioned YAML experiment config.")
    parser.add_argument("--output", required=True, type=Path, help="Path for validation metrics JSON.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Optional validation-only error artifacts.")
    args = parser.parse_args()

    config = load_training_config(args.config)
    device = select_device(config.device)
    payload = load_checkpoint(args.checkpoint, device=device)
    verify_checkpoint_configuration(payload, config)
    model = SiameseUNet(input_channels=config.model.input_channels, base_channels=config.model.base_channels).to(device)
    model.load_state_dict(payload["model_state_dict"])
    dataset = LevirPairedDataset(args.dataset_root, split="val", normalization=_normalization_from_payload(payload))
    result = evaluate_validation_model(
        model,
        dataset,
        device=device,
        patch_size=config.patch_size,
        stride=config.patch_size // 2,
        threshold=frozen_threshold_from_checkpoint(payload),
        artifact_dir=args.artifact_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
