# Final LEVIR-CD locked-test benchmark

## Frozen evaluation record

| Field | Value |
|---|---:|
| Dataset / split | Official LEVIR-CD `test` (128 RGB/VHR building-change scenes) |
| Frozen checkpoint | `runs/controlled_v2_20261005/checkpoints/best_val_iou.pt` |
| Checkpoint SHA-256 | `7375603ebe3392aeea5ae6c5517fd440267b6d75f7eb693d35cf18709314dbd1` |
| Selected epoch | 17 |
| Frozen threshold | 0.5 |
| Test runtime | 154.1994927499909 seconds |
| Git revision used for training | `66b533ebf10e7656fd25459c5c6d458ee1c0f48e` |

The final result is the one-shot locked-test artifact at `runs/controlled_v2_20261005/final_locked_test_result.json`. It was produced only after validation selection had frozen the checkpoint and threshold.

## Learned-model result

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| 5,023,231 | 126,171,653 | 1,208,671 | 1,814,173 |

| Precision | Recall | F1 | IoU |
|---:|---:|---:|---:|
| 0.8060510258344884 | 0.7346693277156067 | 0.7687066168624409 | 0.6243082496745308 |

## Comparison with the locked classical baseline

Both methods are reported on the official locked LEVIR-CD test split. The classical detector uses ORB registration, LAB/edge differences, Otsu, morphology, and connected components; it is not a learned model.

| Metric | Classical baseline | Frozen Siamese U-Net | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Precision | 0.12386368516074106 | 0.8060510258344884 | +0.6821873406737473 | +550.756535128481% |
| Recall | 0.10673246746864745 | 0.7346693277156067 | +0.6279368602469593 | +588.327877298831% |
| F1 | 0.11466172870813732 | 0.7687066168624409 | +0.6540448881543036 | +570.412547868631% |
| IoU | 0.06081758931757607 | 0.6243082496745308 | +0.5634906603569547 | +926.525807220886% |

Relative change is `(frozen learned metric - classical metric) / classical metric`; it is descriptive, not an uncertainty estimate or a claim of generalization beyond this benchmark.

## Freeze and scope limits

The LEVIR-CD test split is **LOCKED** after this evaluation. It must not be used again for development, architecture selection, threshold tuning, hyperparameter tuning, qualitative selection, or model comparison. Any later benchmark requires a separately versioned protocol and should not overwrite this record.

LEVIR-CD is RGB, very-high-resolution building-change data. These scores do not validate Sentinel-2 performance, geographic generalization, multispectral inputs, cloud/shadow robustness, or operational change-area estimates. The RGB LEVIR weights must not be presented as Sentinel-2 weights.
