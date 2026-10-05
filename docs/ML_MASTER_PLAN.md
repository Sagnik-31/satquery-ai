# ML Master Plan

## North star

Replace prototype semantic enrichment with a trained, evaluated, remote-sensing bi-temporal change model whose claims are tied to held-out data, documented labels, and reproducible experiments. The initial model task should be binary change segmentation; semantic change classification should be added only after the binary task is reliable.

## Baseline versus target system

| Aspect | Current prototype | Target ML system |
|---|---|---|
| Change signal | Hand-designed color/edge difference + thresholding | Learned bi-temporal pixel predictions |
| Semantics | NDVI/NDWI rules, ADE20K model, or visual heuristics | Remote-sensing-trained change/land-cover model |
| Supervision | No change-mask training loop in repository | Train/validation/test splits with versioned labels |
| Evidence | Visual regions and operational warnings | Metrics, calibration, error slices, and model/data versions |
| Output | Pixel bounding boxes; catalog provenance | Georeferenced masks/polygons with uncertainty and provenance |

## Problem definition

Given co-registered images \(x_1, x_2\), learn a function \(f_\theta(x_1,x_2)\) that returns a change probability \(p_{ij}\) for each pixel \((i,j)\). For a binary mask \(y_{ij}\in\{0,1\}\), optimize a segmentation loss such as BCE plus Dice:

\[
\mathcal{L}=\lambda\,\mathrm{BCE}(p,y)+(1-\lambda)(1-\mathrm{Dice}(p,y)).
\]

For semantic change, predict class labels only where change is present, or model transition classes explicitly. Transition labels make the task harder and amplify annotation requirements, so they should follow a validated binary baseline.

## Phased plan

1. **Dataset and protocol.** Select an appropriate licensed remote-sensing change dataset; record sensor, resolution, geography, acquisition interval, labels, and split policy. Split by geography/time to avoid overlapping tiles in train and test.
2. **Reproducible baseline.** Implement a data loader, exact preprocessing, a simple Siamese or early-fusion U-Net baseline, configuration files, fixed seeds, checkpoints, and a test-only evaluator.
3. **Model comparison.** Compare architectures such as Siamese encoder-decoder and transformer-based bitemporal models under the identical split and augmentation policy. Keep the classical baseline as a non-learned comparator.
4. **Robustness.** Measure effects of season, cloud/shadow contamination, registration shift, region, land-cover type, and class imbalance. Add cloud/quality masking where supported by the data.
5. **Uncertainty and calibration.** Assess probability calibration (for example, expected calibration error) and define abstention/review behavior rather than presenting raw scores as certainty.
6. **Product integration.** Version the model and preprocessing, validate API inputs, preserve source imagery identifiers, and expose confidence as model output only after calibration evidence exists.

## Minimum experiment record

Every run should store: Git revision, dataset/label version and license, split manifest, configuration, random seed, environment/dependency versions, checkpoint hash, training logs, metrics, per-scene predictions, and failure analysis. A claim of reproduction requires matching the cited paper's task, data, preprocessing, split, and metrics closely enough to support the claim.

## Classical baseline evaluation

Before training, SatQuery evaluates the existing `run_change_analysis` detector through the LEVIR-CD adapter on the official `test` split. The evaluator invokes the production detector rather than duplicating its ORB registration, LAB/edge difference, thresholding, morphology, and component filtering. It aggregates TP/TN/FP/FN and reports precision, recall, F1, and IoU. Results are written outside Git only after a real labeled dataset is supplied; no placeholder metric values are committed.

## Research questions

- Does a learned bi-temporal model materially improve IoU/F1 over the current classical baseline on the same held-out scenes?
- How much does performance degrade across geographic regions and seasons?
- Does adding multispectral bands improve target categories compared with RGB-only inputs?
- Are predicted probabilities calibrated enough for threshold-based operational decisions?
