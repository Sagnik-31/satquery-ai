# SatQuery AI: Project Context

## Purpose

SatQuery AI is a Smart India Hackathon prototype for exploring satellite-image change intelligence. The current product lets a user retrieve two Sentinel-2 scenes for one area of interest (AOI), or upload a before/after image pair, then inspect detected visual change and ask a category-oriented question.

The long-term objective is an industry-grade, scientifically defensible satellite intelligence platform. That requires a genuinely trained and evaluated remote-sensing model, explicit data provenance, geospatially meaningful outputs, and reproducible evaluation—not only a polished interface.

## What is implemented now

- React/Vite frontend with Leaflet map interaction, AOI drawing, upload preview, results display, and JSON export.
- FastAPI backend with two analysis paths:
  - `POST /analyze` for a user-provided image pair.
  - `POST /api/temporal-analysis` for location/date-driven Sentinel-2 retrieval and analysis.
- Location/date retrieval from Element84 Earth Search STAC using Sentinel-2 L2A scenes, with Nominatim geocoding when a location name is supplied.
- Same-AOI scene handling, per-band loading, common-grid reprojection, and an in-memory pair cache.
- A classical visual-change baseline: registration where needed, LAB normalization, color/edge differencing, adaptive thresholding, morphology, and connected-component filtering.
- NDVI-based vegetation and NDWI-based water enrichment when NIR is available.
- Controlled local demo assets in `public/demo_data/`; their README states they are not georeferenced ground truth.

## What must not be claimed

- The visual-difference percentage is not model accuracy.
- The included controlled demo pair is not a measured real-world change dataset.
- Sentinel-2 acquisition is not real-time; the product selects the latest available acquisition after the requested historical scene.
- The current system does not contain a trained remote-sensing change-detection model.
- Terrain queries describe visible ground-cover/surface cues, not elevation or digital surface model change.

## Current semantic behavior

For vegetation and water catalog imagery, the backend may use Sentinel-2 spectral-index rules. Other category behavior can use an off-the-shelf ADE20K SegFormer semantic model or deterministic color/texture heuristics. The code explicitly detects likely ADE20K scene mismatch and falls back. These are prototype enrichments, not validated remote-sensing semantic-change predictions.

## Source-of-truth files

- `backend/main.py`: analysis pipeline and HTTP endpoints.
- `backend/satellite_service.py`: geocoding, STAC selection, COG loading, grid alignment, and retrieval cache.
- `app/page.jsx`: frontend workflow and backend integration.
- `components/MapViewer.jsx`: map, imagery overlay, and AOI selection.
- `lib/imagery.js` and `lib/satquery.js`: upload/GeoTIFF handling and demo utilities.
- `README_HACKATHON.md`: run instructions and accuracy language.
