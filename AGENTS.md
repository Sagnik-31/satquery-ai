# SatQuery AI — Codex Master Instructions

## 1. Mission

SatQuery AI is a prototype unified satellite imagery analysis platform being developed for the Smart India Hackathon (SIH).

The long-term goal is to evolve this prototype into an industry-grade, scientifically defensible satellite intelligence platform.

The system should eventually support:

- single-image understanding
- bi-temporal image comparison
- multi-temporal analysis
- change detection
- change segmentation
- semantic change classification
- geospatial area measurement
- image quality assessment
- uncertainty/confidence estimation
- natural-language querying
- evidence-grounded AI explanations
- reproducible ML experiments

The objective is NOT merely to make the interface look impressive.

The objective is to build a technically strong system whose important claims can be measured, reproduced, and explained.

---

## 2. Current Project State

The current repository contains a working integrated prototype with:

- React
- Vite
- FastAPI
- Python
- Sentinel-2 imagery
- STAC-based imagery discovery
- Raster processing
- OpenCV
- NumPy
- Rasterio
- PyProj
- PyTorch
- Transformers

Important current files include:

```text
app/
    main.jsx
    page.jsx
    globals.css

backend/
    main.py
    satellite_service.py
    requirements.txt

components/
    MapViewer.jsx

lib/
    imagery.js
    satquery.js

public/
    demo_data/

package.json
vite.config.js
start_backend.sh
README_HACKATHON.md