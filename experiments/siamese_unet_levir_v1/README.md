# LEVIR-CD Siamese U-Net v1

## Objective

This experiment foundation defines SatQuery's first learned binary
bi-temporal change-segmentation baseline. It is designed for reproducible
experimentation, not for production inference and not as a validated
Sentinel-2 model.

## Data and split policy

LEVIR-CD contains paired 1024×1024 RGB images and binary labels. The official
split is fixed: 445 train scenes, 64 validation scenes, and 128 test scenes.

- Training reads only `train/`.
- Model selection, threshold selection, and error analysis read only `val/`.
- The official `test/` split is locked. The learned-model train/validation API
  rejects it and this experiment must not read it until a model and protocol
  are frozen.

The existing classical test benchmark is separate. Its measured result is not
input to learned-model tuning.

## Model

`ml.models.siamese_unet.SiameseUNet` uses one shared compact ResNet-style CNN
encoder for T1 and T2. At each scale it fuses
`concat(feature_t1, feature_t2, abs(feature_t1 - feature_t2))`, then uses a
U-Net-style decoder and emits one full-resolution binary logit channel. V1
uses no external pretrained weights.

## Preprocessing and patching

Images load as float RGB tensors in `[0, 1]`; labels map `{0,255}` to
`{0.0,1.0}`. RGB mean/std are calculated from train T1/T2 images only and
saved into the local run directory. Validation reuses that exact metadata.

Training uses 256×256 paired patches. Horizontal/vertical flips, 90-degree
rotations, and random crops are sampled once per pair and applied identically
to T1, T2, and the mask. Photometric augmentation is disabled in v1.
The train-only crop policy requests a change-containing patch 50% of the time
when a scene has changed pixels; otherwise it selects a genuinely unchanged
patch when one exists. This retains negative examples while improving exposure
to sparse changes. It is a fixed conservative development choice, not an
optimized hyperparameter. Train-mask positive-pixel prevalence is recorded
with each run.
Validation runs full 1024×1024 scenes with 256×256 windows, 128-pixel stride,
and averaged overlapping logits before thresholding.

## Objective and selection

The fixed v1 loss is:

`0.5 * BCEWithLogitsLoss + 0.5 * SoftDiceLoss`.

The best checkpoint is selected solely by **global validation IoU**. Validation
records TP, TN, FP, FN, precision, recall, F1, IoU, per-image results, and the
explicit validation-only threshold policy.

## Commands

Do not run these until the experiment is approved. They never provide a test
split option.

```bash
python3 -m ml.training.train_change_model \
  --dataset-root "$HOME/datasets/LEVIR-CD" \
  --config ml/configs/levir_siamese_unet_v1.yaml \
  --run-dir experiments/siamese_unet_levir_v1/runs/local_v1

python3 -m ml.evaluation.evaluate_model \
  --dataset-root "$HOME/datasets/LEVIR-CD" \
  --checkpoint experiments/siamese_unet_levir_v1/runs/local_v1/checkpoints/best_val_iou.pt \
  --config ml/configs/levir_siamese_unet_v1.yaml \
  --output experiments/siamese_unet_levir_v1/runs/local_v1/validation_metrics.json
```

## Checkpoint provenance

Checkpoints contain weights, optimizer and scheduler state, epoch, random
state, model/channel schema, configuration and hash, Git revision, selected
validation metrics/threshold, train-derived normalization, a train/validation
content fingerprint, split-manifest hash, and environment/package snapshot.
Checkpoint SHA-256 sidecars identify finalized files. MPS execution is
best-effort reproducible; exact bitwise replay is not guaranteed. Generated
checkpoints, manifests, logs, metrics, normalization, and validation artifacts
are ignored by Git.

## Resume safety

New checkpoints save model, optimizer, scheduler, Python/NumPy/Torch RNG, the
DataLoader shuffle-generator state, training history, and best-validation
state. A normal resume validates Git revision, configuration hash, channel
schema, train-derived normalization, dataset fingerprint, and split-manifest
hash; it continues at epoch `N + 1` and never overwrites existing epoch files.

For a future exact stateful resume, use the same run directory and a checkpoint
created by this resume-capable implementation:

```bash
python3 -m ml.training.train_change_model \
  --dataset-root "$HOME/datasets/LEVIR-CD" \
  --config ml/configs/levir_siamese_unet_v1.yaml \
  --run-dir experiments/siamese_unet_levir_v1/runs/<same-run> \
  --resume-from experiments/siamese_unet_levir_v1/runs/<same-run>/checkpoints/epoch_012.pt
```

The interrupted `controlled_v1_20261005/epoch_011.pt` predates DataLoader
generator-state support. It cannot be resumed bit-for-bit. If a reviewed
decision is made to continue that pilot, it must explicitly acknowledge this:

```bash
python3 -m ml.training.train_change_model \
  --dataset-root "$HOME/datasets/LEVIR-CD" \
  --config ml/configs/levir_siamese_unet_v1.yaml \
  --run-dir experiments/siamese_unet_levir_v1/runs/controlled_v1_20261005 \
  --resume-from experiments/siamese_unet_levir_v1/runs/controlled_v1_20261005/checkpoints/epoch_011.pt \
  --allow-nonexact-resume
```

That command records `resume_exact=false` and the source checkpoint hash. It
does not make the interrupted pilot equivalent to an uninterrupted 20-epoch
run.

## Final locked test benchmark

`FROZEN_CHECKPOINT.json` freezes the controlled-v2 epoch-17 checkpoint and its
SHA-256. No additional model selection or threshold selection is permitted
before the final benchmark. The following command is intentionally a one-shot,
explicitly confirmed test action; do not run it during development:

```bash
python3 -m ml.evaluation.evaluate_locked_test \
  --dataset-root /Users/kuldeepsinghchauhan/datasets/LEVIR-CD \
  --checkpoint experiments/siamese_unet_levir_v1/runs/controlled_v2_20261005/checkpoints/best_val_iou.pt \
  --confirm-locked-test
```

It evaluates the locked 128-scene official test split exactly once. Before
constructing the test dataset, it verifies the frozen checkpoint path, SHA-256,
configuration hash, architecture, and channel schema, then uses the
checkpoint-recorded normalization and threshold. It performs no tuning, threshold search, or
checkpoint comparison. The final JSON result is written under the ignored run
directory and cannot be overwritten by the command.

## Post-benchmark failure analysis

`ml.evaluation.analyze_validation_failures` is validation-only. It ranks
scenes by IoU, records per-scene TP/TN/FP/FN and precision/recall/F1/IoU,
summarizes mean/median scene scores, flags false-positive/false-negative
imbalance, and can save T1/T2/ground-truth/prediction/error images for the
lowest-IoU validation scenes. It uses the checkpoint-recorded validation
threshold and has no test-split argument.

```bash
python3 -m ml.evaluation.analyze_validation_failures \
  --dataset-root "$HOME/datasets/LEVIR-CD" \
  --checkpoint experiments/siamese_unet_levir_v1/runs/controlled_v2_20261005/checkpoints/best_val_iou.pt \
  --config ml/configs/levir_siamese_unet_v1.yaml \
  --output experiments/siamese_unet_levir_v1/runs/controlled_v2_20261005/validation_failure_analysis.json \
  --artifact-dir experiments/siamese_unet_levir_v1/runs/controlled_v2_20261005/validation_failure_artifacts \
  --max-artifacts 8
```

The final LEVIR-CD test benchmark is recorded in `FINAL_BENCHMARK.md`; it is
frozen and must not be used in this analysis.

Future Sentinel-2 work requires a separate dataset and validation protocol.
Its planned channel order is `[B02, B03, B04, B08, B11, B12]`; this RGB model
does not validate that schema or sensor domain.
