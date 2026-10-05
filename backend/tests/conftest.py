from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
import satellite_service  # noqa: E402


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch):
    """Block optional model loading and clear process-local retrieval state."""
    monkeypatch.setattr(main, "load_segmenter", lambda: (None, None))
    satellite_service._PAIR_CACHE.clear()
    satellite_service._CACHE_ORDER.clear()
    yield
    satellite_service._PAIR_CACHE.clear()
    satellite_service._CACHE_ORDER.clear()


@pytest.fixture
def rgb_pair() -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros((64, 64, 3), dtype=np.uint8)
    before[:, :, 1] = 80
    after = before.copy()
    after[20:44, 20:44] = (220, 220, 220)
    return before, after


@pytest.fixture
def encoded_png_pair(rgb_pair) -> tuple[bytes, bytes]:
    encoded = []
    for image in rgb_pair:
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        encoded.append(buffer.tobytes())
    return tuple(encoded)


@pytest.fixture
def spectral_bands() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {
        "B02": np.full((4, 4), 0.10, dtype=np.float32),
        "B03": np.full((4, 4), 0.20, dtype=np.float32),
        "B04": np.full((4, 4), 0.30, dtype=np.float32),
        "B08": np.full((4, 4), 0.60, dtype=np.float32),
    }
    after = {name: values.copy() for name, values in before.items()}
    after["B08"][1:3, 1:3] = 0.90
    return before, after


@pytest.fixture
def raster_grids():
    return {
        "shape": (4, 4),
        "wgs84": CRS.from_epsg(4326),
        "web_mercator": CRS.from_epsg(3857),
        "reference_transform": from_origin(0, 4, 1, 1),
        "shifted_transform": from_origin(1, 4, 1, 1),
    }


@pytest.fixture
def stac_item_factory():
    def make_item(identifier: str, timestamp: str, cloud: float, tile: str = "43PGQ"):
        return {
            "id": identifier,
            "properties": {
                "datetime": timestamp,
                "eo:cloud_cover": cloud,
                "s2:mgrs_tile": tile,
            },
            "assets": {},
        }

    return make_item
