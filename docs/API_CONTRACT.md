# API Contract

Base URL defaults to `http://localhost:8000`; the frontend can override it with `VITE_API_URL`. CORS is currently configured permissively for all origins. Responses documented below are based on `backend/main.py` and should be updated with tests whenever the implementation changes.

## `GET /health`

Returns service status and runtime capability information, including device, whether Transformers imported, whether the optional SegFormer has loaded, STAC endpoint/collection, and Rasterio availability.

## `POST /analyze`

Manual before/after image analysis. Content type: `multipart/form-data`.

| Field | Required | Type | Notes |
|---|---:|---|---|
| `before_image` | yes | image upload | Earlier image; must have image content type |
| `after_image` | yes | image upload | Later image; must have image content type |
| `query` | no | string | Defaults to `What changed between these images?` |
| `min_area` | no | integer | Defaults to 3000; pipeline subsequently bounds/scales it |

Files over 35 MB or 100 megapixels are rejected. Typical client errors are 400/413; unexpected processing failures return 500.

## `POST /api/temporal-analysis`

Retrieves and analyzes a Sentinel-2 pair. Content type: `application/json`.

```json
{
  "location": "Bengaluru, India",
  "before_date": "2024-01-15",
  "cloud_threshold": 20,
  "query": "What are the vegetation changes?",
  "min_area": 3000,
  "aoi_radius_km": 2.5,
  "bbox": [77.5, 12.8, 77.7, 13.0],
  "pair_id": "optional-cached-pair-id"
}
```

`before_date` is required. Supply either `location`, or both `latitude` and `longitude`. A valid four-value `bbox` overrides the radius-derived AOI. Validation constrains cloud threshold to 5–80, minimum area to 200–10,000, and AOI radius to 0.5–15 km. Scene-selection failures return 400; retrieval failures return 503; analysis failures return 500.

## Successful analysis response

Both analysis endpoints return the common analysis fields: `ok`, `query`, `category`, `answer`, `method`, `alignment`, `metrics`, `technical`, `semantic`, `regions`, `semantic_regions`, `warnings`, `visuals`, `timing_ms`, and `disclaimer`.

- `metrics` contains display strings such as change-region count, visual difference, largest region, and processing time.
- `regions` and `semantic_regions` are image-coordinate bounding-box features, with normalized coordinates; they are not geospatial polygons.
- `visuals` contains base64 PNG payloads: `overlay_png`, `heatmap_png`, `mask_png`, and optionally `semantic_overlay_png`.
- `semantic.backend` can identify `sentinel2_ndvi`, `sentinel2_ndwi`, `segformer`, or `deterministic_visual_fallback`; its confidence must not be interpreted as validated remote-sensing uncertainty.

The temporal endpoint additionally returns `pair_id`, `analysis_source: "satellite_catalog"`, `location`, `before`, `after`, `provenance`, `general_change`, and base64 `before_png`/`after_png` visuals.

## Compatibility rules

Treat response fields as additive unless a versioned API change is made. Consumers must display backend warnings and disclaimers rather than suppressing them. Any future trained-model response should add explicit `model_version`, `preprocessing_version`, calibrated confidence semantics, geospatial CRS/geometry, and data provenance fields.
