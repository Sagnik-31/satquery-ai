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


VALID_SPLITS = {"train", "val", "test"}
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
