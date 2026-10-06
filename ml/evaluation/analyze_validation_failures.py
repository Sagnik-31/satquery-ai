"""Validation-only, post-hoc failure analysis for LEVIR-CD learned models.

This module has no test-split option. It is deliberately separate from model
selection: it uses the checkpoint-recorded validation threshold and only
describes per-scene validation errors.
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
from torch.utils.data import Dataset

from ml.datasets.levir_cd import LevirPairedDataset, RGBNormalization
from ml.evaluation.evaluate_model import _normalization_from_payload, sliding_window_logits, verify_checkpoint_configuration
from ml.evaluation.metrics import binary_confusion, metrics_from_confusion
from ml.models.siamese_unet import SiameseUNet
from ml.training.checkpointing import load_checkpoint
from ml.training.config import load_training_config
from ml.training.reproducibility import select_device


def _scene_record(identifier: str, prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    confusion = binary_confusion(prediction, truth)
    metrics = metrics_from_confusion(confusion)
    fp, fn = confusion.false_positive, confusion.false_negative
    record: dict[str, Any] = {
        "id": identifier, **confusion.as_dict(), **metrics,
        "predicted_change_pixels": int(np.count_nonzero(prediction)),
        "ground_truth_change_pixels": int(np.count_nonzero(truth)),
    }
    if fp >= 2 * max(fn, 1) and fp > 0:
        record["error_imbalance"] = "false_positive_dominant"
    elif fn >= 2 * max(fp, 1) and fn > 0:
        record["error_imbalance"] = "false_negative_dominant"
    else:
        record["error_imbalance"] = "balanced_or_low_error"
    return record


def summarize_validation_scenes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank scenes by IoU and summarize validation-only error patterns."""
    if not records:
        raise ValueError("Failure analysis requires at least one validation scene.")
    ranked = sorted(records, key=lambda record: (float(record["iou"]), str(record["id"])))
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    f1s = np.asarray([record["f1"] for record in records], dtype=np.float64)
    return {
        "split": "val", "num_scenes": len(records),
        "mean_iou": float(ious.mean()), "median_iou": float(np.median(ious)),
        "mean_f1": float(f1s.mean()), "median_f1": float(np.median(f1s)),
        "ranked_by_iou_ascending": ranked, "worst_by_iou": ranked,
        "severe_false_positive_scenes": [r for r in ranked if r["error_imbalance"] == "false_positive_dominant"],
        "severe_false_negative_scenes": [r for r in ranked if r["error_imbalance"] == "false_negative_dominant"],
    }


def _to_uint8_rgb(image: torch.Tensor, normalization: RGBNormalization) -> np.ndarray:
    mean = torch.tensor(normalization.mean, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(normalization.std, dtype=image.dtype).view(3, 1, 1)
    return np.ascontiguousarray(((image * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy())


def _write_artifacts(artifact_dir: Path, identifier: str, t1: torch.Tensor, t2: torch.Tensor, truth: np.ndarray, prediction: np.ndarray, normalization: RGBNormalization) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    error = np.zeros((*truth.shape, 3), dtype=np.uint8)
    error[prediction & ~truth] = (255, 0, 0)  # red: false positive
    error[~prediction & truth] = (0, 0, 255)  # blue: false negative
    for suffix, rgb in (("t1", _to_uint8_rgb(t1, normalization)), ("t2", _to_uint8_rgb(t2, normalization)), ("error", error)):
        cv2.imwrite(str(artifact_dir / f"{identifier}_{suffix}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(artifact_dir / f"{identifier}_ground_truth.png"), truth.astype(np.uint8) * 255)
    cv2.imwrite(str(artifact_dir / f"{identifier}_prediction.png"), prediction.astype(np.uint8) * 255)


def analyze_validation_failures(
    model: nn.Module, dataset: Dataset[dict[str, torch.Tensor | str]], *, device: torch.device,
    patch_size: int, stride: int, threshold: float, tile_batch_size: int,
    normalization: RGBNormalization | None = None, artifact_dir: Path | None = None, max_artifacts: int = 0,
) -> dict[str, Any]:
    """Run validation inference and return per-scene failure analysis."""
    if getattr(dataset, "split", None) != "val":
        raise ValueError("Failure analysis is validation-only; dataset.split must be 'val'.")
    if max_artifacts < 0:
        raise ValueError("max_artifacts must be non-negative.")
    model.eval()
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            t1, t2, mask = item["t1"], item["t2"], item["mask"]
            assert isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor) and isinstance(mask, torch.Tensor)
            probabilities = torch.sigmoid(sliding_window_logits(model, t1, t2, patch_size=patch_size, stride=stride, device=device, tile_batch_size=tile_batch_size)).cpu().numpy()
            truth = mask.squeeze(0).cpu().numpy().astype(bool)
            prediction, identifier = probabilities >= threshold, str(item["id"])
            records.append(_scene_record(identifier, prediction, truth))
            cached[identifier] = (t1.cpu(), t2.cpu(), truth, prediction)
    summary = summarize_validation_scenes(records)
    summary.update({"threshold": threshold, "threshold_policy": "checkpoint_recorded_validation_threshold"})
    if artifact_dir is not None and max_artifacts:
        if normalization is None:
            raise ValueError("normalization is required when writing RGB artifacts.")
        selected = summary["ranked_by_iou_ascending"][:max_artifacts]
        for record in selected:
            _write_artifacts(artifact_dir, record["id"], *cached[record["id"]], normalization)
        summary["artifact_scene_ids"] = [record["id"] for record in selected]
    else:
        summary["artifact_scene_ids"] = []
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Local LEVIR-CD root; this command reads val/ only.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--max-artifacts", type=int, default=0)
    args = parser.parse_args()
    config = load_training_config(args.config)
    device = select_device(config.device)
    payload = load_checkpoint(args.checkpoint, device=device)
    verify_checkpoint_configuration(payload, config)
    normalization = _normalization_from_payload(payload)
    model = SiameseUNet(input_channels=config.model.input_channels, base_channels=config.model.base_channels).to(device)
    model.load_state_dict(payload["model_state_dict"])
    dataset = LevirPairedDataset(args.dataset_root, split="val", normalization=normalization)
    result = analyze_validation_failures(model, dataset, device=device, patch_size=config.patch_size, stride=config.patch_size // 2, threshold=float(payload["selected_validation_threshold"]), tile_batch_size=config.validation_tile_batch_size, normalization=normalization, artifact_dir=args.artifact_dir, max_artifacts=args.max_artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
