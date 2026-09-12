"""Satellite imagery retrieval for SatQuery AI temporal analysis.

Uses Element84 Earth Search STAC (Sentinel-2 L2A) with optional
Microsoft Planetary Computer fallback. No API keys required for Earth Search.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# Public Sentinel-2 COGs on AWS do not require credentials.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")

import cv2
import httpx
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    RASTERIO_AVAILABLE = False

EARTH_SEARCH_URL = os.getenv(
    "STAC_API_URL",
    "https://earth-search.aws.element84.com/v1",
)
STAC_COLLECTION = os.getenv("STAC_COLLECTION", "sentinel-2-l2a")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = os.getenv(
    "SATQUERY_USER_AGENT",
    "SatQuery-AI/1.0 (hackathon; contact: demo@satquery.local)",
)
MAX_OUTPUT_PX = int(os.getenv("SATELLITE_MAX_PX", "1400"))

# Filled after RetrievedPair is defined.
_PAIR_CACHE: dict[str, Any] = {}
_CACHE_ORDER: list[str] = []
_CACHE_LIMIT = 4


class SceneRetrievalError(Exception):
    """Raised when satellite scenes cannot be retrieved."""


@dataclass
class SceneMetadata:
    date: str
    datetime_utc: str
    sensor: str
    cloud_cover: float
    source: str
    stac_id: str
    mgrs_tile: str = ""
    platform: str = "Sentinel-2"


@dataclass
class RetrievedPair:
    before_bgr: np.ndarray
    after_bgr: np.ndarray
    before_bands: dict[str, np.ndarray] | None
    after_bands: dict[str, np.ndarray] | None
    location: dict[str, Any]
    before_meta: SceneMetadata
    after_meta: SceneMetadata
    warnings: list[str] = field(default_factory=list)
    aoi_bbox: list[float] = field(default_factory=list)
    has_nir: bool = False

    def provenance(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "aoi_bbox_wgs84": self.aoi_bbox,
            "before": asdict(self.before_meta),
            "after": asdict(self.after_meta),
            "source_catalog": self.before_meta.source,
            "processing": [
                "Common AOI clip",
                "Sentinel-2 L2A surface reflectance",
                "Per-band percentile stretch to RGB",
                "Grid alignment for T1/T2",
            ],
            "has_nir": self.has_nir,
        }


def pair_cache_key(
    *,
    location: str | None,
    latitude: float | None,
    longitude: float | None,
    before_date: str,
    cloud_threshold: float,
    bbox: list[float] | None,
    aoi_radius_km: float,
) -> str:
    payload = {
        "location": (location or "").strip().lower(),
        "latitude": round(latitude, 5) if latitude is not None else None,
        "longitude": round(longitude, 5) if longitude is not None else None,
        "before_date": before_date,
        "cloud_threshold": float(cloud_threshold),
        "bbox": [round(v, 5) for v in bbox] if bbox else None,
        "aoi_radius_km": float(aoi_radius_km),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def get_cached_pair(pair_id: str | None) -> RetrievedPair | None:
    if not pair_id:
        return None
    return _PAIR_CACHE.get(pair_id)


def store_cached_pair(pair_id: str, pair: RetrievedPair) -> None:
    _PAIR_CACHE[pair_id] = pair
    if pair_id in _CACHE_ORDER:
        _CACHE_ORDER.remove(pair_id)
    _CACHE_ORDER.append(pair_id)
    while len(_CACHE_ORDER) > _CACHE_LIMIT:
        oldest = _CACHE_ORDER.pop(0)
        _PAIR_CACHE.pop(oldest, None)


def bbox_from_center(lat: float, lon: float, radius_km: float = 2.5) -> list[float]:
    """Return [west, south, east, north] in WGS84."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def parse_latlon(text: str) -> tuple[float, float] | None:
    parts = [p.strip() for p in text.replace(" ", "").split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    return lat, lon


def geocode_location(location: str) -> dict[str, Any]:
    coords = parse_latlon(location)
    if coords:
        lat, lon = coords
        return {
            "latitude": lat,
            "longitude": lon,
            "name": f"{lat:.4f}, {lon:.4f}",
        }

    with httpx.Client(timeout=30) as client:
        response = client.get(
            NOMINATIM_URL,
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()

    if not data:
        raise SceneRetrievalError(
            f"Could not geocode location: {location!r}. "
            "Try a city name or latitude,longitude."
        )

    return {
        "latitude": float(data[0]["lat"]),
        "longitude": float(data[0]["lon"]),
        "name": data[0].get("display_name", location),
    }


def _stac_search(
    bbox: list[float],
    datetime_range: str,
    cloud_max: float,
    limit: int = 30,
    sort: list[dict] | None = None,
    mgrs_tile: str | None = None,
) -> list[dict]:
    query: dict[str, Any] = {"eo:cloud_cover": {"lt": cloud_max}}
    if mgrs_tile:
        query["s2:mgrs_tile"] = {"eq": mgrs_tile}

    body: dict[str, Any] = {
        "collections": [STAC_COLLECTION],
        "bbox": bbox,
        "datetime": datetime_range,
        "query": query,
        "limit": limit,
    }
    if sort:
        body["sortby"] = sort

    with httpx.Client(timeout=90) as client:
        response = client.post(f"{EARTH_SEARCH_URL}/search", json=body)
        response.raise_for_status()
        return response.json().get("features", [])


def _parse_scene_datetime(item: dict) -> datetime:
    raw = item["properties"]["datetime"]
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _scene_cloud(item: dict) -> float:
    return float(item["properties"].get("eo:cloud_cover", 100))


def _scene_mgrs(item: dict) -> str:
    return str(item["properties"].get("s2:mgrs_tile", "") or "")


def pick_historical_scene(
    features: list[dict],
    target_date: str,
) -> dict | None:
    if not features:
        return None
    target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
    scored: list[tuple[int, float, dict]] = []
    for item in features:
        dt = _parse_scene_datetime(item)
        scored.append((abs((dt - target).days), _scene_cloud(item), item))
    scored.sort(key=lambda row: (row[0], row[1]))
    return scored[0][2]


def pick_latest_scene(features: list[dict]) -> dict | None:
    if not features:
        return None
    return max(features, key=lambda item: _parse_scene_datetime(item))


def search_historical_with_fallback(
    bbox: list[float],
    before_date: str,
    cloud_threshold: float,
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    target = datetime.fromisoformat(before_date).replace(tzinfo=timezone.utc)
    windows = [7, 14, 21, 30]
    thresholds = [
        cloud_threshold,
        min(cloud_threshold + 10, 50),
        min(cloud_threshold + 20, 60),
        50,
    ]

    for days, cloud_max in zip(windows, thresholds):
        start = (target - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
        end = (target + timedelta(days=days)).strftime("%Y-%m-%dT23:59:59Z")
        features = _stac_search(
            bbox,
            f"{start}/{end}",
            cloud_max,
            limit=40,
            sort=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        )
        scene = pick_historical_scene(features, before_date)
        if scene:
            cloud = _scene_cloud(scene)
            if cloud > cloud_threshold:
                warnings.append(
                    f"Best available historical image has {cloud:.0f}% cloud cover "
                    f"(requested threshold was {cloud_threshold:.0f}%)."
                )
            if days > 7:
                warnings.append(
                    f"Historical scene selected from ±{days}-day search window "
                    f"around {before_date}."
                )
            return scene, warnings

    raise SceneRetrievalError(
        f"No suitable satellite scene was found within ±30 days of {before_date} "
        f"under a relaxed 50% cloud threshold."
    )


def search_latest_with_fallback(
    bbox: list[float],
    cloud_threshold: float,
    mgrs_tile: str | None,
    after_not_before: datetime,
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    now = datetime.now(timezone.utc)
    thresholds = [
        cloud_threshold,
        min(cloud_threshold + 10, 50),
        min(cloud_threshold + 20, 60),
        50,
    ]
    lookbacks = [120, 240, 365]

    for days, cloud_max in zip(lookbacks, thresholds):
        start = (now - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
        end = now.strftime("%Y-%m-%dT23:59:59Z")
        features = _stac_search(
            bbox,
            f"{start}/{end}",
            cloud_max,
            limit=60,
            sort=[{"field": "properties.datetime", "direction": "desc"}],
            mgrs_tile=mgrs_tile or None,
        )
        # Keep scenes strictly after the historical acquisition.
        filtered = [
            item for item in features
            if _parse_scene_datetime(item) > after_not_before
        ]
        scene = pick_latest_scene(filtered)
        if scene:
            cloud = _scene_cloud(scene)
            if cloud > cloud_threshold:
                warnings.append(
                    f"Latest available image has {cloud:.0f}% cloud cover "
                    f"(requested threshold was {cloud_threshold:.0f}%)."
                )
            if days > 120:
                warnings.append(
                    f"Latest available acquisition was selected after scanning the past {days} days."
                )
            return scene, warnings

    raise SceneRetrievalError(
        "No latest available Sentinel-2 scene was found for this AOI under "
        "relaxed cloud thresholds."
    )


def _asset_href(item: dict, candidates: list[str]) -> str | None:
    assets = item.get("assets", {})
    for name in candidates:
        if name in assets and assets[name].get("href"):
            return assets[name]["href"]
    return None


def _read_band_window(
    href: str,
    bbox_wgs84: list[float],
) -> tuple[np.ndarray, Any, Any]:
    if not RASTERIO_AVAILABLE:
        raise SceneRetrievalError(
            "rasterio is required for satellite retrieval. Install backend requirements."
        )

    west, south, east, north = bbox_wgs84
    with rasterio.open(href) as src:
        bbox_proj = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
        window = rasterio.windows.from_bounds(*bbox_proj, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
        transform = src.window_transform(window)
        return data, transform, src.crs


def _load_scene_bands(
    item: dict,
    bbox_wgs84: list[float],
) -> tuple[dict[str, np.ndarray], Any, Any]:
    band_names = ["B04", "B03", "B02", "B08"]
    bands: dict[str, np.ndarray] = {}
    ref_transform = None
    ref_crs = None

    aliases = {
        "B04": ["red", "B04", "b04"],
        "B03": ["green", "B03", "b03"],
        "B02": ["blue", "B02", "b02"],
        "B08": ["nir", "B08", "b08", "nir08"],
    }
    for band in band_names:
        href = _asset_href(item, aliases.get(band, [band]))

        if not href:
            continue

        data, transform, crs = _read_band_window(href, bbox_wgs84)
        bands[band] = data
        if ref_transform is None:
            ref_transform = transform
            ref_crs = crs

    if not {"B02", "B03", "B04"}.issubset(bands.keys()):
        visual_href = _asset_href(item, ["visual", "rendered_preview", "thumbnail"])
        if visual_href:
            with rasterio.open(visual_href) as src:
                bbox_proj = transform_bounds(
                    "EPSG:4326", src.crs, *bbox_wgs84
                )
                window = rasterio.windows.from_bounds(*bbox_proj, transform=src.transform)
                if src.count >= 3:
                    rgb = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                    bands["B04"] = rgb[0].astype(np.float32)
                    bands["B03"] = rgb[1].astype(np.float32)
                    bands["B02"] = rgb[2].astype(np.float32)
                    ref_transform = src.window_transform(window)
                    ref_crs = src.crs

    if not {"B02", "B03", "B04"}.issubset(bands.keys()):
        raise SceneRetrievalError(
            f"Scene {item.get('id')} is missing RGB assets for the selected AOI."
        )

    return bands, ref_transform, ref_crs


def _align_band_to_reference(
    band: np.ndarray,
    src_transform,
    src_crs,
    ref_shape: tuple[int, int],
    ref_transform,
    ref_crs,
) -> np.ndarray:
    if not RASTERIO_AVAILABLE:
        return band

    if band.shape == ref_shape:
        return band

    dst = np.zeros(ref_shape, dtype=np.float32)
    reproject(
        source=band,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
    return dst


def _bands_to_bgr(bands: dict[str, np.ndarray]) -> np.ndarray:
    red = bands["B04"]
    green = bands["B03"]
    blue = bands["B02"]
    stack = np.stack([red, green, blue], axis=-1)
    valid = stack[np.isfinite(stack)]
    if valid.size == 0:
        raise SceneRetrievalError("Selected scene contains no valid pixels in the AOI.")
    p2, p98 = np.percentile(valid, (2, 98))
    rgb = np.clip((stack - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_OUTPUT_PX / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        for key in list(bands.keys()):
            bh, bw = bands[key].shape[:2]
            bands[key] = cv2.resize(
                bands[key],
                (int(bw * scale), int(bh * scale)),
                interpolation=cv2.INTER_AREA,
            )
    return bgr


def _scene_metadata(item: dict) -> SceneMetadata:
    dt = _parse_scene_datetime(item)
    return SceneMetadata(
        date=dt.strftime("%Y-%m-%d"),
        datetime_utc=dt.strftime("%Y-%m-%d %H:%M UTC"),
        sensor="Sentinel-2 L2A",
        cloud_cover=round(_scene_cloud(item), 1),
        source=f"Element84 Earth Search ({STAC_COLLECTION})",
        stac_id=item.get("id", ""),
        mgrs_tile=_scene_mgrs(item),
        platform="Sentinel-2",
    )


def retrieve_temporal_pair(
    *,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    before_date: str,
    cloud_threshold: float = 20.0,
    bbox: list[float] | None = None,
    aoi_radius_km: float = 2.5,
) -> RetrievedPair:
    if not RASTERIO_AVAILABLE:
        raise SceneRetrievalError(
            "Satellite retrieval requires rasterio. "
            "Run: pip install -r backend/requirements.txt"
        )

    if latitude is not None and longitude is not None:
        loc = {
            "latitude": latitude,
            "longitude": longitude,
            "name": location or f"{latitude:.4f}, {longitude:.4f}",
        }
    elif location:
        loc = geocode_location(location)
    else:
        raise SceneRetrievalError("Provide a location name or latitude/longitude.")

    if bbox and len(bbox) == 4:
        aoi = bbox
    else:
        aoi = bbox_from_center(loc["latitude"], loc["longitude"], aoi_radius_km)

    historical, hist_warnings = search_historical_with_fallback(
        aoi,
        before_date,
        cloud_threshold,
    )
    hist_dt = _parse_scene_datetime(historical)
    mgrs = _scene_mgrs(historical)

    latest, latest_warnings = search_latest_with_fallback(
        aoi,
        cloud_threshold,
        mgrs,
        hist_dt,
    )

    before_bands, before_transform, before_crs = _load_scene_bands(historical, aoi)
    after_bands_raw, after_transform, after_crs = _load_scene_bands(latest, aoi)

    ref_shape = before_bands["B04"].shape
    aligned_after: dict[str, np.ndarray] = {}
    for band, arr in after_bands_raw.items():
        aligned_after[band] = _align_band_to_reference(
            arr,
            after_transform,
            after_crs,
            ref_shape,
            before_transform,
            before_crs,
        )

    before_bgr = _bands_to_bgr(before_bands)
    after_bgr = _bands_to_bgr(aligned_after)
    has_nir = "B08" in before_bands and "B08" in aligned_after

    warnings = hist_warnings + latest_warnings
    latest_mgrs = _scene_mgrs(latest)
    if mgrs and latest_mgrs and latest_mgrs != mgrs:
        warnings.append(
            "Historical and latest scenes use different Sentinel-2 tiles; "
            "grids were reprojected to a common AOI."
        )

    return RetrievedPair(
        before_bgr=before_bgr,
        after_bgr=after_bgr,
        before_bands=before_bands if has_nir else None,
        after_bands=aligned_after if has_nir else None,
        location={
            **loc,
            "aoi_radius_km": aoi_radius_km,
        },
        before_meta=_scene_metadata(historical),
        after_meta=_scene_metadata(latest),
        warnings=warnings,
        aoi_bbox=aoi,
        has_nir=has_nir,
    )
