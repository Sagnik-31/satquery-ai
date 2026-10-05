from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.datasets.levir_cd import LevirCDDataset
from ml.evaluation import evaluate_baseline
from ml.evaluation.metrics import BinaryConfusion, binary_confusion, metrics_from_confusion


def test_binary_metrics_define_empty_pair_as_exact_unchanged_prediction():
    confusion = binary_confusion(np.zeros((2, 2)), np.zeros((2, 2)))

    assert confusion == BinaryConfusion(0, 4, 0, 0)
    assert metrics_from_confusion(confusion) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "iou": 1.0,
    }


def test_binary_metrics_cover_perfect_incorrect_and_partial_masks():
    perfect = binary_confusion(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    incorrect = binary_confusion(np.array([[1, 1]]), np.array([[0, 0]]))
    partial = binary_confusion(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))

    assert metrics_from_confusion(perfect)["f1"] == 1.0
    assert metrics_from_confusion(incorrect) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    assert metrics_from_confusion(partial) == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "iou": 1 / 3}


def test_binary_confusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        binary_confusion(np.zeros((2, 2)), np.zeros((1, 2)))


def _write_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    assert ok


def test_levir_adapter_requires_official_split_layout(tmp_path):
    root = tmp_path / "LEVIR-CD"
    for directory in (root / "test" / "A", root / "test" / "B", root / "test" / "label"):
        directory.mkdir(parents=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    _write_image(root / "test" / "A" / "sample.png", image)
    _write_image(root / "test" / "B" / "sample.png", image)
    _write_image(root / "test" / "label" / "sample.png", np.zeros((4, 4), dtype=np.uint8))

    dataset = LevirCDDataset(root, "test")

    assert len(dataset) == 1
    assert dataset.samples[0].identifier == "sample"


def test_evaluator_returns_machine_readable_structure_without_real_dataset_download(tmp_path, monkeypatch):
    root = tmp_path / "LEVIR-CD"
    for directory in (root / "test" / "A", root / "test" / "B", root / "test" / "label"):
        directory.mkdir(parents=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    _write_image(root / "test" / "A" / "sample.png", image)
    _write_image(root / "test" / "B" / "sample.png", image)
    _write_image(root / "test" / "label" / "sample.png", mask)
    encoded_ok, encoded = cv2.imencode(".png", mask)
    assert encoded_ok
    monkeypatch.setattr(
        evaluate_baseline,
        "run_change_analysis",
        lambda *_args, **_kwargs: {"visuals": {"mask_png": base64.b64encode(encoded.tobytes()).decode("ascii")}},
    )

    result = evaluate_baseline.evaluate(
        LevirCDDataset(root, "test"),
        min_area=3000,
        max_artifacts=0,
        artifact_dir=None,
    )

    assert result["dataset"] == "LEVIR-CD"
    assert result["split"] == "test"
    assert result["num_samples"] == 1
    assert result["precision"] == result["recall"] == result["f1"] == result["iou"] == 1.0
    assert {"tp", "tn", "fp", "fn", "runtime_seconds", "git_commit"}.issubset(result)
