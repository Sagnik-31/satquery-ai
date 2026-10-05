# Classical Baseline Evaluation

This experiment evaluates the actual production function `backend.main.run_change_analysis` on a labeled, bi-temporal remote-sensing split. It does not duplicate the detector and does not train a model.

## Dataset

The first supported adapter is **LEVIR-CD**. Obtain the dataset from its official distribution under its applicable terms, store it outside Git, and preserve its official split directories:

```text
LEVIR-CD/
  train/A  train/B  train/label
  val/A    val/B    val/label
  test/A   test/B   test/label
```

The default scientific evaluation split is `test`. Do not randomly mix official train, validation, and test samples.

## Run

From the repository root, with backend dependencies installed:

```bash
python3 -m ml.evaluation.evaluate_baseline \
  --dataset-root /absolute/path/to/LEVIR-CD \
  --split test
```

This writes a machine-readable JSON result to `experiments/baseline/results/classical_baseline.json` and at most eight high pixel-error examples under `experiments/baseline/results/artifacts/`. Both are ignored by Git to prevent measured results and dataset-derived artifacts from being committed accidentally.

## Input and metric conventions

- Images are decoded with OpenCV as BGR arrays; no dataset-specific normalization is added.
- The production detector resizes T1 to T2, attempts ORB/RANSAC registration, then runs its existing LAB/edge, threshold, morphology, and connected-component pipeline.
- LEVIR-CD labels are interpreted as binary: any nonzero label pixel is change.
- Precision, recall, F1, and IoU are aggregated from global TP/TN/FP/FN counts across the selected split.
- If both prediction and target are empty, precision, recall, F1, and IoU are defined as 1.0 because unchanged was predicted exactly. Other zero denominators evaluate to 0.0.

The result measures this classical detector only. It is not model accuracy, scientific validation, or a claim about a future trained model.
