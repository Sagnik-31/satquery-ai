"""One-shot final evaluator for the single frozen LEVIR-CD test checkpoint.

This module intentionally has no train/validation split, threshold-search, or
checkpoint-selection option. It may be executed only with explicit human
confirmation after the model was frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ml.datasets.levir_cd import LevirCDDataset, RGBNormalization, _read_mask_tensor, _read_rgb_tensor
from ml.evaluation.evaluate_model import sliding_window_logits
from ml.evaluation.metrics import BinaryConfusion, binary_confusion, merge_confusions, metrics_from_confusion
from ml.models.siamese_unet import SiameseUNet
from ml.training.checkpointing import checkpoint_sha256, load_checkpoint
from ml.training.config import load_training_config
from ml.training.reproducibility import select_device


ROOT = Path(__file__).resolve().parents[2]
FREEZE_RECORD = ROOT / "experiments/siamese_unet_levir_v1/FROZEN_CHECKPOINT.json"
CONFIG_PATH = ROOT / "ml/configs/levir_siamese_unet_v1.yaml"


@dataclass(frozen=True)
class FrozenCheckpoint:
    checkpoint: Path
    sha256: str
    epoch: int
    threshold: float
    config_sha256: str
    git_revision: str
    dataset_fingerprint: str
    split_manifest_sha256: str


def load_freeze_record(path: Path = FREEZE_RECORD) -> FrozenCheckpoint:
    raw = json.loads(path.read_text())
    return FrozenCheckpoint(
        checkpoint=(ROOT / raw["checkpoint"]).resolve(),
        sha256=raw["checkpoint_sha256"],
        epoch=int(raw["selected_epoch"]),
        threshold=float(raw["threshold"]),
        config_sha256=raw["config_sha256"],
        git_revision=raw["git_revision"],
        dataset_fingerprint=raw["dataset_fingerprint"],
        split_manifest_sha256=raw["split_manifest_sha256"],
    )


def require_locked_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise PermissionError("Locked test evaluation requires --confirm-locked-test before any test dataset is constructed.")


def require_test_split(split: str) -> None:
    if split != "test":
        raise ValueError("Locked evaluator accepts only the official 'test' split.")


def verify_frozen_checkpoint(path: Path, payload: dict[str, Any], frozen: FrozenCheckpoint) -> None:
    if path.resolve() != frozen.checkpoint:
        raise ValueError("Locked evaluator accepts only the frozen checkpoint path in FROZEN_CHECKPOINT.json.")
    actual_hash = checkpoint_sha256(path)
    if actual_hash != frozen.sha256:
        raise ValueError("Frozen checkpoint SHA-256 mismatch; refusing locked test evaluation.")
    expected_schema = {"t1": 3, "t2": 3, "output_logits": 1}
    provenance = payload.get("provenance", {})
    checks = {
        "epoch": (payload.get("epoch"), frozen.epoch),
        "architecture": (payload.get("architecture"), "SiameseUNet"),
        "channel schema": (payload.get("channel_schema"), expected_schema),
        "config hash": (payload.get("config_sha256"), frozen.config_sha256),
        "threshold": (float(payload.get("selected_validation_threshold", -1)), frozen.threshold),
        "Git revision": (provenance.get("git_revision"), frozen.git_revision),
        "dataset fingerprint": (provenance.get("dataset_fingerprint", {}).get("development_splits_sha256"), frozen.dataset_fingerprint),
        "split manifest hash": (provenance.get("split_manifest_sha256"), frozen.split_manifest_sha256),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Frozen checkpoint {label} mismatch; refusing locked test evaluation.")


class LockedLevirTestDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Test-only loader, constructed solely after confirmation and verification."""

    split = "test"

    def __init__(self, root: str | Path, normalization: RGBNormalization):
        require_test_split(self.split)
        self.adapter = LevirCDDataset(root, split=self.split)
        self.samples = self.adapter.samples
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        t1, t2, mask = _read_rgb_tensor(sample.before_path), _read_rgb_tensor(sample.after_path), _read_mask_tensor(sample.mask_path)
        if t1.shape != t2.shape or mask.shape[1:] != t1.shape[1:]:
            raise ValueError(f"Malformed locked test pair: {sample.identifier}")
        mean = torch.tensor(self.normalization.mean, dtype=t1.dtype).view(3, 1, 1)
        std = torch.tensor(self.normalization.std, dtype=t1.dtype).view(3, 1, 1)
        return {"id": sample.identifier, "t1": (t1 - mean) / std, "t2": (t2 - mean) / std, "mask": mask}


def evaluate_locked_dataset(
    model: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor | str]],
    *,
    device: torch.device,
    threshold: float,
    tile_batch_size: int,
) -> dict[str, Any]:
    """Compute final metrics at the checkpoint-recorded threshold only."""
    if getattr(dataset, "split", None) != "test":
        raise ValueError("Locked evaluator refuses any dataset other than split='test'.")
    model.eval()
    started = time.perf_counter()
    confusions: list[BinaryConfusion] = []
    per_image: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        t1, t2, mask = item["t1"], item["t2"], item["mask"]
        assert isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor) and isinstance(mask, torch.Tensor)
        logits = sliding_window_logits(model, t1, t2, patch_size=256, stride=128, device=device, tile_batch_size=tile_batch_size)
        confusion = binary_confusion(torch.sigmoid(logits).numpy() >= threshold, mask.squeeze(0).numpy().astype(bool))
        confusions.append(confusion)
        per_image.append({"id": str(item["id"]), **confusion.as_dict(), **metrics_from_confusion(confusion)})
    aggregate = merge_confusions(confusions)
    return {
        "split": "test",
        "threshold": threshold,
        "threshold_policy": "frozen_checkpoint_no_selection",
        "tile_size": 256,
        "tile_stride": 128,
        "num_samples": len(dataset),
        "runtime_seconds": time.perf_counter() - started,
        **aggregate.as_dict(),
        **metrics_from_confusion(aggregate),
        "per_image": per_image,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Official LEVIR-CD root; the evaluator reads test/ only after confirmation.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Must equal the frozen checkpoint in FROZEN_CHECKPOINT.json.")
    parser.add_argument("--confirm-locked-test", action="store_true", help="Explicit authorization for the one final 128-scene test evaluation.")
    args = parser.parse_args()

    require_locked_confirmation(args.confirm_locked_test)
    frozen = load_freeze_record()
    config = load_training_config(CONFIG_PATH)
    if config.fingerprint() != frozen.config_sha256:
        raise ValueError("Current frozen configuration hash does not match the freeze record.")
    device = select_device(config.device)
    payload = load_checkpoint(args.checkpoint, device=device)
    verify_frozen_checkpoint(args.checkpoint, payload, frozen)
    raw_normalization = payload["normalization"]
    normalization = RGBNormalization(mean=tuple(raw_normalization["mean"]), std=tuple(raw_normalization["std"]))
    model = SiameseUNet(input_channels=3, base_channels=config.model.base_channels).to(device)
    model.load_state_dict(payload["model_state_dict"])
    dataset = LockedLevirTestDataset(args.dataset_root, normalization)
    result = evaluate_locked_dataset(model, dataset, device=device, threshold=frozen.threshold, tile_batch_size=config.validation_tile_batch_size)
    result.update({"checkpoint": str(frozen.checkpoint.relative_to(ROOT)), "checkpoint_sha256": frozen.sha256, "checkpoint_epoch": frozen.epoch})
    output = frozen.checkpoint.parent.parent / "final_locked_test_result.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing final locked test result: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
