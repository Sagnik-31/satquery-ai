# Architecture

## System boundary

The repository contains a browser frontend and a Python API. There is no persistent database, model registry, job queue, or production deployment configuration in the current codebase.

```text
Browser (React + Vite)
  ├─ Leaflet map, AOI bounds, uploads, UI state
  ├─ POST /analyze (manual image pair)
  └─ POST /api/temporal-analysis (location/date)
             │
FastAPI
  ├─ satellite_service.py: geocode → STAC search → COG bands → common grid
  ├─ main.py: classical change pipeline → response/PNG overlays
  └─ in-memory temporal-pair cache (maximum 4 pairs)
             │
External services
  ├─ Nominatim geocoding
  └─ Element84 Earth Search STAC / Sentinel-2 L2A COG assets
```

## Frontend

`app/main.jsx` starts React; `app/page.jsx` owns the workspace state and calls the API URL from `VITE_API_URL` or `http://localhost:8000`. The UI exposes four modes, but only catalog and manual before/after modes use the backend change pipeline. Single-image and SAR/fusion paths return simulated/demo GeoJSON results from `lib/satquery.js`; they are not connected to specialist inference services.

`MapViewer.jsx` dynamically imports Leaflet, supplies Esri or OpenStreetMap basemaps, renders illustrative GeoJSON features, overlays georeferenced upload previews when bounds exist, and collects a rectangular AOI using two clicks.

`lib/imagery.js` accepts PNG, JPEG, TIFF, and GeoTIFF previews. GeoTIFF bounds are converted to WGS84 where its CRS is supported; this browser-side preview is not a backend geospatial analysis contract.

## Backend data flow

### Manual pair

`/analyze` receives two multipart image files, decodes them through OpenCV, limits raw-file size to 35 MB and image size to 100 megapixels, resizes the first image to the second image's dimensions, then calls `run_change_analysis`.

### Location/date pair

`/api/temporal-analysis` validates a JSON request, computes a cache key, and retrieves a pair if it is not cached. Retrieval resolves a location or coordinates, derives/uses an AOI bounding box, selects a historical scene near the requested date and a later latest scene, and loads Sentinel-2 bands. B04 defines the target T1 grid; every available T1/T2 band is explicitly checked and resampled to that grid when CRS, affine transform, shape, resolution, or bounds differ. The analysis response includes provenance, scene-quality fields, physical-area fields, and encoded before/after PNGs.

### Catalog preprocessing and quality

The current catalog path requests B02 (blue), B03 (green), B04 (red), B08 (NIR), and—when supplied by the STAC item—B11 and B12 (SWIR). B02/B03/B04/B08 are normally 10 m Sentinel-2 bands; B11/B12 are normally 20 m and are bilinearly resampled to the B04 target grid before analysis. Raster reads retain `float32` analysis arrays and a valid-pixel mask derived from Rasterio masking/NoData handling. Per-band provenance records source CRS, transform, shape, resolution, source dtype, declared NoData, source valid/NoData fractions, and whether resampling occurred. The API reports valid-pixel and NoData fractions, while STAC scene cloud cover and any available cloud-shadow/NoData metadata are kept in provenance/warnings.

The system rejects catalog scenes with fewer than 50% valid RGB pixels in the AOI and warns below 80%. These are operational quality gates, not accuracy measurements.

## Classical baseline pipeline

1. Match image sizes.
2. For uploads, attempt ORB feature matching and RANSAC homography. Catalog pairs instead rely on common-grid alignment.
3. Normalize luminance in LAB space using CLAHE.
4. Compute a weighted color-distance and Canny edge-difference image.
5. Threshold with the maximum of Otsu and the 82nd percentile.
6. Apply opening/closing morphology, filter connected components, and form contours.
7. Return masks, a heatmap, an overlay, bounding-box regions, and descriptive metrics.

This is image processing, not supervised change detection. It is sensitive to residual misregistration, seasonal effects, illumination, clouds, shadows, atmosphere, and scene-selection differences. For catalog results, changed area is calculated from the verified raster grid and the valid AOI mask in square metres/hectares; manual uploads remain image-only and do not receive geographic-area outputs.

## Semantic enrichment path

When the query maps to a category, the backend prefers NDVI/NDWI rules for vegetation/water if spectral bands are available. Otherwise it attempts a general ADE20K SegFormer model and rejects it when indoor-label dominance suggests scene mismatch; a deterministic HSV/LAB/edge heuristic then provides a fallback. No path has been established as a trained and evaluated remote-sensing change model.
