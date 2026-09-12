# SatQuery AI — Integrated Change Intelligence Demo

This project integrates the provided React/Vite frontend with a Python FastAPI
change-detection backend.

## Primary workflow: Location + historical date

The default analysis source is **Location + date**:

1. Enter a place (`Bengaluru, India` or `12.9716,77.5946`) and a historical date.
2. SatQuery geocodes the location and builds a small AOI (default 2.5 km radius, or a map-drawn box).
3. It searches **Element84 Earth Search STAC** for Sentinel-2 L2A:
   - **T1** = closest suitable scene near the requested historical date
   - **T2** = latest available acquisition for the **same AOI**
4. Both scenes are cropped, reprojected onto a common grid, then analyzed.
5. Follow-up questions reuse the cached pair (no re-download).

Language used in the product: **latest available satellite imagery**, never “real-time”.

No API key is required for Earth Search. Optional variables are in `.env.example`.

## What is included

- Sentinel-2 L2A retrieval via STAC (same AOI for T1 and T2).
- Cloud-cover filtering with fallback windows and explicit warnings.
- Bi-temporal alignment (geospatial grid for catalog; ORB + RANSAC for uploads).
- Illumination normalization in LAB space.
- Adaptive change detection and connected-component filtering.
- NDVI / NDWI when NIR is present; RGB semantic fallback otherwise.
- Natural-language routing for buildings, vegetation, terrain, water, roads.
- Data provenance panel (sensor, acquisition times, cloud cover, source).
- Manual Before/After upload and controlled demo pair remain available.

## Important accuracy note

The returned "Visual Difference" is NOT accuracy. It is the proportion of
pixels detected as visually different.

For a defensible accuracy number, evaluate against ground-truth change masks
using IoU, precision, recall, and F1.

"Terrain" is interpreted as visible surface/ground-cover classes. It does not
measure elevation change.

## Run the backend

From the project root:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The first semantic query downloads/caches the SegFormer checkpoint.
The first Location + date query downloads Sentinel-2 COG windows (can take 30–90s).

Health check:

```bash
curl http://localhost:8000/health
```

## Run the frontend

Open another terminal in the project root:

```bash
npm install
npm run dev
```

The frontend expects the backend at:

```text
http://localhost:8000
```

To use a different backend URL:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Demo workflow (judges)

1. Open SatQuery AI (workflow defaults to **Location + date**).
2. Location: `Bengaluru, India`
3. Historical date: `2024-01-15`
4. Click **Retrieve & analyze** (or **Live satellite demo**).
5. Wait for archive search; T1 and T2 appear with actual acquisition dates.
6. Ask:
   - "What changed between these dates?"
   - "What are the vegetation changes?"
   - "What are the changes in buildings?"
   - "What changed in the water?"

Fallback if the catalog is slow or unavailable:

1. Select **Manual upload**.
2. Click **Load demo pair**.
3. Ask the same category questions on the controlled RGB pair.

## Hackathon presentation language

Use "semantic change-intelligence prototype" or "bi-temporal change
detection baseline". Do not claim a model accuracy percentage unless you
have evaluated the system against ground-truth masks. Do not call Sentinel-2
imagery real-time.
