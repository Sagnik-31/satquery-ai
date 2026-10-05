"""Paired spatial transforms for bi-temporal change segmentation.

Every geometric operation is sampled once and applied identically to T1, T2,
and the change mask. Photometric augmentation is deliberately absent in v1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class PairedSpatialTransform:
    patch_size: int = 256
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_90: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        for name, value in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("vertical_flip_probability", self.vertical_flip_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def with_seed(self, seed: int) -> "PairedSpatialTransform":
        return replace(self, seed=seed)

    def __call__(
        self, t1: torch.Tensor, t2: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if t1.ndim != 3 or t2.ndim != 3 or mask.ndim != 3:
            raise ValueError("Paired transforms expect [C, H, W], [C, H, W], and [1, H, W] tensors.")
        if t1.shape != t2.shape or t1.shape[1:] != mask.shape[1:]:
            raise ValueError("T1, T2, and mask must have matching spatial dimensions.")
        height, width = t1.shape[1:]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Input {height}x{width} is smaller than configured patch size {self.patch_size}."
            )
        rng = random.Random(self.seed)
        if rng.random() < self.horizontal_flip_probability:
            t1, t2, mask = (torch.flip(value, dims=(2,)) for value in (t1, t2, mask))
        if rng.random() < self.vertical_flip_probability:
            t1, t2, mask = (torch.flip(value, dims=(1,)) for value in (t1, t2, mask))
        if self.rotate_90:
            turns = rng.randrange(4)
            if turns:
                t1, t2, mask = (torch.rot90(value, turns, dims=(1, 2)) for value in (t1, t2, mask))
        height, width = t1.shape[1:]
        top = rng.randint(0, height - self.patch_size)
        left = rng.randint(0, width - self.patch_size)
        crop = lambda value: value[:, top : top + self.patch_size, left : left + self.patch_size].contiguous()
        return crop(t1), crop(t2), crop(mask)
