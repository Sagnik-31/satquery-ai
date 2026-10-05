from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ml.datasets.levir_cd import LevirPairedDataset, compute_train_rgb_normalization
from ml.datasets.paired_transforms import PairedSpatialTransform


def _write(path: Path, array: np.ndarray) -> None:
    assert cv2.imwrite(str(path), array)


def _make_split(root: Path, split: str, *, name: str = "sample.png", shape=(32, 32)) -> None:
    for folder in ("A", "B", "label"):
        (root / split / folder).mkdir(parents=True, exist_ok=True)
    before = np.zeros((*shape, 3), dtype=np.uint8)
    before[..., 2] = 255  # BGR red, proving conversion is defined by OpenCV path.
    after = before.copy()
    after[8:16, 8:16] = (0, 255, 0)
    label = np.zeros(shape, dtype=np.uint8)
    label[8:16, 8:16] = 255
    _write(root / split / "A" / name, before)
    _write(root / split / "B" / name, after)
    _write(root / split / "label" / name, label)


def test_training_dataset_loads_paired_float_tensors_and_binary_mask(tmp_path):
    root = tmp_path / "LEVIR-CD"
    _make_split(root, "train")
    dataset = LevirPairedDataset(root, split="train")
    item = dataset[0]
    assert item["t1"].shape == (3, 32, 32)
    assert item["t1"].dtype == torch.float32
    assert item["mask"].shape == (1, 32, 32)
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}
    # OpenCV BGR input is converted to RGB: red belongs in channel zero.
    assert item["t1"][0, 0, 0].item() == 1.0


def test_training_dataset_order_is_deterministic_and_train_val_only(tmp_path):
    root = tmp_path / "LEVIR-CD"
    _make_split(root, "train", name="z.png")
    _make_split(root, "train", name="a.png")
    _make_split(root, "val")
    _make_split(root, "test")
    assert [sample.identifier for sample in LevirPairedDataset(root, split="train").samples] == ["a", "z"]
    assert len(LevirPairedDataset(root, split="val")) == 1
    with pytest.raises(ValueError, match="locked"):
        LevirPairedDataset(root, split="test")
    with pytest.raises(TypeError):
        LevirPairedDataset(root)  # type: ignore[call-arg]


def test_training_dataset_fails_loudly_on_malformed_pair_shapes(tmp_path):
    root = tmp_path / "LEVIR-CD"
    _make_split(root, "train")
    _write(root / "train" / "B" / "sample.png", np.zeros((16, 32, 3), dtype=np.uint8))
    dataset = LevirPairedDataset(root, split="train")
    with pytest.raises(ValueError, match="does not match T2"):
        dataset[0]


def test_train_normalization_never_requires_validation_or_test_directories(tmp_path):
    root = tmp_path / "LEVIR-CD"
    _make_split(root, "train")
    normalization = compute_train_rgb_normalization(root)
    assert len(normalization.mean) == len(normalization.std) == 3
    assert all(value > 0 for value in normalization.std)


def test_paired_transform_is_repeatable_with_seed_and_keeps_dates_mask_aligned():
    base = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32)
    t1 = base.repeat(3, 1, 1)
    t2 = t1 + 10000
    mask = base.clone()
    transform = PairedSpatialTransform(patch_size=16).with_seed(11)
    first = transform(t1, t2, mask)
    second = transform(t1, t2, mask)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert torch.equal(first[1] - first[0], torch.full_like(first[0], 10000))
    assert torch.equal(first[2], first[0][0:1])
