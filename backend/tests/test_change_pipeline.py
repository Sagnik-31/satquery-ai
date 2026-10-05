from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

import main
from satellite_service import RasterGrid


def test_calculate_difference_is_zero_for_identical_images(rgb_pair):
    before, _ = rgb_pair
    difference = main.calculate_difference(before, before.copy())

    assert difference.shape == before.shape[:2]
    assert difference.dtype == np.uint8
    assert int(difference.max()) == 0


def test_change_mask_and_contours_detect_synthetic_region():
    difference = np.zeros((64, 64), dtype=np.uint8)
    difference[20:44, 20:44] = 255

    mask, threshold = main.create_change_mask(difference, min_area=100)
    contours = main.get_contours(mask, min_area=100)

    assert threshold >= 0
    assert int(np.count_nonzero(mask)) > 0
    assert len(contours) == 1


def test_spectral_indices_follow_the_implemented_formula():
    red = np.array([[0.2]], dtype=np.float32)
    green = np.array([[0.3]], dtype=np.float32)
    nir = np.array([[0.6]], dtype=np.float32)

    assert np.allclose(main.compute_ndvi(red, nir), [[(0.6 - 0.2) / (0.6 + 0.2 + 1e-6)]])
    assert np.allclose(main.compute_ndwi(green, nir), [[(0.3 - 0.6) / (0.3 + 0.6 + 1e-6)]])
    swir = np.array([[0.8]], dtype=np.float32)
    assert np.allclose(main.compute_ndbi(swir, nir), [[(0.8 - 0.6) / (0.8 + 0.6 + 1e-6)]])


def test_query_classification_routes_supported_categories():
    assert main.classify_query("What are the changes in buildings?") == "building"
    assert main.classify_query("Show vegetation loss") == "vegetation"
    assert main.classify_query("Did the lake change?") == "water"
    assert main.classify_query("Find road changes") == "road"
    assert main.classify_query("What changed?") == "general"


def test_run_change_analysis_returns_bounded_visual_metrics(rgb_pair):
    before, after = rgb_pair
    result = main.run_change_analysis(before, after, "What changed?", min_area=100)

    visual_difference = next(metric["value"] for metric in result["metrics"] if metric["label"] == "Visual difference")
    percentage = float(visual_difference.removesuffix("%"))

    assert result["ok"] is True
    assert result["category"] == "general"
    assert 0.0 <= percentage <= 100.0
    assert result["alignment"]["success"] is False
    assert result["visuals"]["mask_png"]
    assert "not model accuracy" in result["disclaimer"]


def test_run_change_analysis_reports_physical_area_only_with_a_grid(rgb_pair):
    before, after = rgb_pair
    grid = RasterGrid(
        shape=before.shape[:2],
        transform=from_origin(0, 640, 10, 10),
        crs=CRS.from_epsg(3857),
    )
    result = main.run_change_analysis(
        before,
        after,
        "What changed?",
        min_area=100,
        geospatial_grid=grid,
        valid_mask=np.ones(grid.shape, dtype=bool),
    )

    assert result["geospatial"] is not None
    assert result["geospatial"]["aoi_area_m2"] == 64 * 64 * 100
    assert result["geospatial"]["changed_area_m2"] >= 0


def test_spectral_vegetation_enrichment_uses_change_mask(spectral_bands):
    before_bands, after_bands = spectral_bands
    general_mask = np.full((4, 4), 255, dtype=np.uint8)

    result = main.build_spectral_semantic(before_bands, after_bands, general_mask, "vegetation")

    assert result is not None
    assert result["backend"] == "sentinel2_ndvi"
    assert result["new_pixels"] == 4
    assert result["removed_pixels"] == 0
