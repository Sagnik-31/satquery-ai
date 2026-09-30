# Data and Evaluation

## Current data assets

`public/demo_data/before_demo.png` and `after_demo.png` are controlled visual demo images. `demo_truth.json` lists intended bounding boxes, but its own metadata marks the pair as non-georeferenced and not measured real-world ground truth. It may be used for UI/regression demonstration, not for reporting model performance.

Location-driven analysis obtains Sentinel-2 L2A imagery through Element84 Earth Search. These scenes provide operational provenance (scene IDs, dates, cloud cover, AOI), but the repository does not include authoritative change masks or a curated training/evaluation dataset.

## Data requirements for trained ML

A selected dataset must have clear licensing and document:

- sensor, spatial resolution, spectral bands, preprocessing level, and geography;
- exact definition of “change” and annotation method;
- temporal interval, clouds/shadows, and known label noise;
- spatially disjoint train/validation/test partitions;
- class distribution and ignored/uncertain pixels.

Do not randomly split neighboring chips: spatial leakage can inflate results because nearly identical geography appears in both training and test sets.

## Metrics

For a change mask, with true positives \(TP\), false positives \(FP\), and false negatives \(FN\):

\[
\mathrm{Precision}=TP/(TP+FP),\quad
\mathrm{Recall}=TP/(TP+FN),
\]
\[
F1=2TP/(2TP+FP+FN),\quad
IoU=TP/(TP+FP+FN).
\]

Report more than one aggregate score: include precision/recall, class-wise values for semantic labels, per-region or per-scene distributions, and confidence intervals where feasible. Pixel accuracy alone is inappropriate for heavily imbalanced change masks because unchanged pixels dominate.

## Evaluation protocol

1. Freeze the test split before model selection.
2. Tune thresholds and architectures only using training/validation data.
3. Evaluate the classical baseline and learned candidates on the same held-out data.
4. Report failure slices: small objects, changed fraction, region, season, acquisition interval, cloud/shadow condition, and registration quality.
5. Save qualitative true-positive, false-positive, and false-negative examples with scene metadata.
6. For production-oriented confidence, measure calibration separately from segmentation quality.

## Presenting current output

Call the current percentage **visual difference** or **detected visual-change area**, not accuracy. The backend's own response disclaimer requires ground-truth masks plus IoU, precision, recall, and F1 for defensible accuracy claims.
