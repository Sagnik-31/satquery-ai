# Codex First Tasks

This order reduces scientific and engineering risk before expanding features.

1. **Add a test foundation.** Create focused tests for query classification, request validation, image-size guards, spectral-index calculations, and deterministic change-mask behavior using small synthetic arrays. This establishes regression protection without requiring remote services.
2. **Make retrieval testable.** Isolate STAC and COG network access behind mockable adapters; add fixtures for scene metadata and known AOIs. Test scene-selection fallbacks and warning propagation.
3. **Define the evaluation dataset and split.** Choose a licensed remote-sensing change dataset, document labels and a geographic split manifest, and keep the controlled demo pair out of accuracy evaluation.
4. **Implement a reproducible learned baseline.** Add a separate training/evaluation package rather than embedding training logic in `backend/main.py`. Start with binary bitemporal change segmentation, configs, fixed seeds, checkpoints, and IoU/precision/recall/F1 evaluation.
5. **Benchmark honestly.** Compare the learned baseline and the existing classical pipeline on the same frozen test scenes. Publish failure cases and do not claim reproduction until the protocol matches a cited paper.
6. **Design an inference contract.** Add model/data/preprocessing versions, masks with CRS-aware geospatial outputs, quality flags, and calibrated confidence only after evaluation supports them.
7. **Integrate incrementally.** Put trained inference behind an explicit backend strategy/endpoint, retain the classical method as a labeled baseline, and update the UI to show source, model version, confidence definition, and warnings.

## Non-goals for the first ML iteration

Do not claim SAR fusion, multi-temporal forecasting, elevation-change measurement, natural-language factual reasoning, or industry-grade accuracy before data, models, and evaluations are implemented. The current frontend labels some of these workflows for demonstration; that is not implementation evidence.
