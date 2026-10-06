from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_final_model_artifact_matches_the_frozen_checkpoint_identity():
    artifact = json.loads((ROOT / "experiments/siamese_unet_levir_v1/FINAL_MODEL_ARTIFACT.json").read_text())
    checkpoint = ROOT / artifact["checkpoint"]
    assert artifact["artifact_type"] == "frozen_levir_rgb_change_segmentation_model"
    assert artifact["architecture"] == "SiameseUNet"
    assert artifact["input_channels_per_date"] == 3
    assert artifact["threshold"] == 0.5
    assert artifact["locked_test_benchmark"]["identity"].startswith("official LEVIR-CD test split")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == artifact["checkpoint_sha256"]


def test_final_benchmark_documents_the_locked_scope_and_sentinel2_limit():
    benchmark = (ROOT / "experiments/siamese_unet_levir_v1/FINAL_BENCHMARK.md").read_text()
    assert "5,023,231" in benchmark
    assert "0.6243082496745308" in benchmark
    assert "**LOCKED**" in benchmark
    assert "must not be presented as Sentinel-2 weights" in benchmark
