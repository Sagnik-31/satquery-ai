# Development Workflow

## Local setup

Frontend requirements are declared in `package.json` (Node `>=22.13.0`). Run `npm install` and `npm run dev` from the repository root. The frontend expects the API at `http://localhost:8000` unless `VITE_API_URL` is set.

For the backend, use the commands in `README_HACKATHON.md`: create `backend/.venv`, install `backend/requirements.txt`, then run Uvicorn from `backend` on port 8000. `start_backend.sh` performs the same setup/run sequence. The first optional SegFormer query can download its checkpoint; location/date retrieval accesses remote STAC/COG services and may be slow.

## Tests

Run the offline backend regression suite from the repository root:

```bash
pytest backend/tests -q
```

The suite uses synthetic imagery and mocked retrieval failures. It must not contact Nominatim, Element84/STAC, remote COG assets, or download/load the optional ADE20K SegFormer model.

## Change discipline

1. Read `AGENTS.md`, the relevant source module, and these docs before changing behavior.
2. Make the smallest cohesive change. Keep UI simulations clearly separate from live analysis.
3. Preserve accuracy language: visual difference is not accuracy, and current semantic fallbacks are not validated remote-sensing ML.
4. For backend changes, inspect API compatibility and add/update focused tests before claiming verification.
5. For frontend changes, build with `npm run build`; manually exercise affected workflows when a browser/runtime is available.
6. For retrieval changes, test failures as well as successful scenes: invalid locations, no scenes, high cloud cover, missing Rasterio, and remote-service errors.
7. Record evidence for any ML claim: dataset version, split, configuration, seed, checkpoint, metrics, and error analysis.

## Existing verification gaps

The repository currently has no committed test suite or CI configuration visible at the top level. A successful build or API health check does not validate satellite-scene correctness, model accuracy, or geospatial measurement. Add tests progressively around pure functions, endpoint validation, retrieval adapters, and evaluation pipelines.

## Git hygiene

Check `git status` before and after work. Do not mix generated `dist/`, virtual environments, local caches, or downloaded model artifacts with intentional source/documentation changes. Do not commit or push unless explicitly asked.
