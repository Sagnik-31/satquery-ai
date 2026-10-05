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
validation metrics/threshold, and train-derived normalization. Generated
checkpoints, manifests, logs, metrics, normalization, and validation artifacts
are ignored by Git.

Future Sentinel-2 work requires a separate dataset and validation protocol.
Its planned channel order is `[B02, B03, B04, B08, B11, B12]`; this RGB model
does not validate that schema or sensor domain.
