"""Evaluate the production classical detector on a labeled LEVIR-CD split.

This module calls ``backend.main.run_change_analysis`` and decodes its returned
mask. It intentionally does not reimplement the detector.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.main import run_change_analysis
from ml.datasets.levir_cd import LevirCDDataset
from ml.evaluation.metrics import BinaryConfusion, binary_confusion, merge_confusions, metrics_from_confusion


METHOD = "classical_orb_lab_edge_otsu_morphology_connected_components"


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode label mask: {path}")
    return mask > 0


def _decode_mask(result: dict) -> np.ndarray:
    raw = base64.b64decode(result["visuals"]["mask_png"])
    mask = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Production detector returned an unreadable mask PNG.")
    return mask > 0


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_artifact(directory: Path, rank: int, sample_id: str, before, after, truth, prediction) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    error = np.zeros((*truth.shape, 3), dtype=np.uint8)
    error[(prediction & ~truth)] = (0, 0, 255)  # false positive: red, BGR
    error[(~prediction & truth)] = (255, 0, 0)  # false negative: blue, BGR
    cv2.imwrite(str(directory / f"{rank:02d}_{sample_id}_t1.png"), before)
    cv2.imwrite(str(directory / f"{rank:02d}_{sample_id}_t2.png"), after)
    cv2.imwrite(str(directory / f"{rank:02d}_{sample_id}_ground_truth.png"), truth.astype(np.uint8) * 255)
    cv2.imwrite(str(directory / f"{rank:02d}_{sample_id}_prediction.png"), prediction.astype(np.uint8) * 255)
    cv2.imwrite(str(directory / f"{rank:02d}_{sample_id}_error.png"), error)


def evaluate(
    dataset: LevirCDDataset,
    *,
    min_area: int,
    max_artifacts: int,
    artifact_dir: Path | None,
) -> dict:
    started = time.perf_counter()
    confusions: list[BinaryConfusion] = []
    examples: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for sample in dataset:
        before = _read_color(sample.before_path)
        after = _read_color(sample.after_path)
        truth = _read_mask(sample.mask_path)
        result = run_change_analysis(before, after, "What changed?", min_area=min_area)
        prediction = _decode_mask(result)
        if truth.shape != prediction.shape:
            truth = cv2.resize(
                truth.astype(np.uint8),
                (prediction.shape[1], prediction.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        confusion = binary_confusion(prediction, truth)
        confusions.append(confusion)
        examples.append((confusion.false_positive + confusion.false_negative, sample.identifier, before, after, truth, prediction))

    aggregate = merge_confusions(confusions)
    if artifact_dir is not None:
        for rank, (_, sample_id, before, after, truth, prediction) in enumerate(
            sorted(examples, key=lambda row: row[0], reverse=True)[:max_artifacts], start=1
        ):
            _write_artifact(artifact_dir, rank, sample_id, before, after, truth, prediction)

    return {
        "dataset": "LEVIR-CD",
        "split": dataset.split,
        "method": METHOD,
        "num_samples": len(dataset),
        **aggregate.as_dict(),
        **metrics_from_confusion(aggregate),
        "runtime_seconds": time.perf_counter() - started,
        "git_commit": git_commit(),
        "mask_convention": "ground_truth_nonzero_is_change; production_mask_nonzero_is_change",
        "min_area_input_px": min_area,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Local LEVIR-CD root; never stored in Git.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--min-area", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/baseline/results/classical_baseline.json")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "experiments/baseline/results/artifacts")
    parser.add_argument("--max-artifacts", type=int, default=8)
    args = parser.parse_args()

    try:
        dataset = LevirCDDataset(args.dataset_root, args.split)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(
            f"Could not prepare the real LEVIR-CD {args.split!r} split: {exc}. "
            "No evaluation metrics or result file were produced."
        )
    result = evaluate(dataset, min_area=args.min_area, max_artifacts=args.max_artifacts, artifact_dir=args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
