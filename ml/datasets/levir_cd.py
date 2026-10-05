"""Filesystem adapter for the official LEVIR-CD split layout.

Expected dataset root (kept outside Git)::

    LEVIR-CD/
      train/A, train/B, train/label
      val/A,   val/B,   val/label
      test/A,  test/B,  test/label
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "val", "test"}
TRAINING_SPLITS = {"train", "val"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class LevirSample:
    identifier: str
    before_path: Path
    after_path: Path
    mask_path: Path


class LevirCDDataset:
    """Validate and enumerate a LEVIR-CD split without changing image data."""

    def __init__(self, root: str | Path, split: str = "test"):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split {split!r}; choose one of {sorted(VALID_SPLITS)}.")
        self.samples = self._discover()

    def _discover(self) -> list[LevirSample]:
        split_root = self.root / self.split
        directories = {name: split_root / name for name in ("A", "B", "label")}
        missing = [str(path) for path in directories.values() if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "LEVIR-CD split is incomplete. Expected directories: " + ", ".join(missing)
            )

        before_files = sorted(
            path for path in directories["A"].iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not before_files:
            raise FileNotFoundError(f"No supported image files found in {directories['A']}")

        samples: list[LevirSample] = []
        for before_path in before_files:
            candidates_b = [directories["B"] / before_path.name]
            candidates_label = [directories["label"] / before_path.name]
            after_path = next((path for path in candidates_b if path.is_file()), None)
            mask_path = next((path for path in candidates_label if path.is_file()), None)
            if after_path is None or mask_path is None:
                raise FileNotFoundError(
                    f"Sample {before_path.name} must exist in A, B, and label for split {self.split}."
                )
            samples.append(LevirSample(before_path.stem, before_path, after_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


@dataclass(frozen=True)
class RGBNormalization:
    """Per-channel RGB normalization derived from the training split only."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


def _read_rgb_tensor(path: Path) -> torch.Tensor:
    """Read a LEVIR image as a float RGB tensor in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode LEVIR RGB image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a three-channel RGB image at {path}, got shape {image.shape}.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


def _read_mask_tensor(path: Path) -> torch.Tensor:
    """Read a LEVIR binary label as [1, H, W] float values in {0.0, 1.0}."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode LEVIR label image: {path}")
    return torch.from_numpy(np.ascontiguousarray(mask[None] > 0)).float()


def compute_train_rgb_normalization(root: str | Path) -> RGBNormalization:
    """Compute RGB mean/std from official LEVIR-CD training pairs only.

    Both temporal images are observations from the training split and therefore
    contribute equally. Validation and the locked test split are never opened.
    """
    dataset = LevirCDDataset(root, split="train")
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for sample in dataset:
        for path in (sample.before_path, sample.after_path):
            image = _read_rgb_tensor(path).to(dtype=torch.float64)
            channel_sum += image.sum(dim=(1, 2))
            channel_sq_sum += image.square().sum(dim=(1, 2))
            pixel_count += image.shape[1] * image.shape[2]
    if pixel_count == 0:
        raise ValueError("Cannot compute normalization from an empty LEVIR-CD training split.")
    mean = channel_sum / pixel_count
    variance = (channel_sq_sum / pixel_count) - mean.square()
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return RGBNormalization(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
    )


class LevirPairedDataset(Dataset[dict[str, torch.Tensor | str]]):
    """PyTorch LEVIR-CD dataset restricted to train/validation development work.

    The legacy :class:`LevirCDDataset` remains available for the separately
    frozen classical test evaluation. This class intentionally rejects ``test``
    so training or validation code cannot silently read the locked split.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        transform: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        normalization: RGBNormalization | None = None,
        seed: int | None = None,
    ):
        if split not in TRAINING_SPLITS:
            raise ValueError(
                f"LevirPairedDataset only permits development splits {sorted(TRAINING_SPLITS)}; got {split!r}. "
                "The official test split is locked."
            )
        self.split = split
        self.adapter = LevirCDDataset(root, split=split)
        self.samples = self.adapter.samples
        self.transform = transform
        self.normalization = normalization
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch for a training loader."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_normalization(self, image: torch.Tensor) -> torch.Tensor:
        if self.normalization is None:
            return image
        mean = torch.tensor(self.normalization.mean, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(self.normalization.std, dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        t1 = _read_rgb_tensor(sample.before_path)
        t2 = _read_rgb_tensor(sample.after_path)
        mask = _read_mask_tensor(sample.mask_path)
        if t1.shape != t2.shape:
            raise ValueError(
                f"Malformed LEVIR pair {sample.identifier}: T1 shape {tuple(t1.shape)} does not match T2 {tuple(t2.shape)}."
            )
        if mask.shape[1:] != t1.shape[1:]:
            raise ValueError(
                f"Malformed LEVIR pair {sample.identifier}: label shape {tuple(mask.shape)} does not match images {tuple(t1.shape)}."
            )
        if self.transform is not None:
            # Transform implementations may use this reproducible state; no test
            # split identifier is ever incorporated or resolved here.
            if hasattr(self.transform, "with_seed") and self.seed is not None:
                transform = self.transform.with_seed(self.seed + self._epoch * len(self) + index)
                t1, t2, mask = transform(t1, t2, mask)
            else:
                t1, t2, mask = self.transform(t1, t2, mask)
        return {
            "t1": self._apply_normalization(t1),
            "t2": self._apply_normalization(t2),
            "mask": mask,
            "id": sample.identifier,
        }
