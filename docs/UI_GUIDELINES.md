# UI Guidelines

## Truthful product language

Use “latest available acquisition,” not “real-time.” Label the current result as visual change or a prototype baseline. Surface acquisition dates, cloud cover, AOI, source catalog, processing warnings, and the distinction between general visual change and semantic enrichment.

Never present the current visual-difference percentage as accuracy. Do not present deterministic heuristics, ADE20K predictions, or their scores as validated remote-sensing confidence. The existing UI already renders warnings and a fallback note; preserve that behavior.

## Workflow clarity

- Make the supported live paths clear: **Location + date** and **Manual Before / After**.
- Keep single-image and SAR modes visually marked as illustrative/demo until dedicated backends exist.
- Explain whether an AOI comes from a location-radius default or user-drawn bounding box.
- For catalog results, identify T1 as the selected historical acquisition and T2 as the later latest available acquisition.
- Explain that map overlays for demo features are illustrative and that backend regions are image-coordinate boxes, not guaranteed geospatial polygons.

## Interaction and accessibility

Retain button labels/`aria-label`s, loading/cancellation states, useful error messages, and keyboard-usable controls. Do not hide retrieval failures behind generic success-looking demo output. Keep visual overlays accompanied by text summaries and warnings so a color-only representation is not the sole channel.

## Future ML UI requirements

Before showing a trained-model result, specify the model and preprocessing version, output units/CRS, calibrated confidence meaning, threshold, quality masks, and limitations. Let users inspect provenance and export machine-readable evidence. A confidence number without calibration and a clear definition should be omitted.
