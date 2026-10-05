from __future__ import annotations

import numpy as np

import satellite_service


def test_parse_coordinates_and_reject_invalid_values():
    assert satellite_service.parse_latlon("12.9716, 77.5946") == (12.9716, 77.5946)
    assert satellite_service.parse_latlon("91,77") is None
    assert satellite_service.parse_latlon("not coordinates") is None


def test_bbox_from_center_has_expected_wgs84_ordering():
    west, south, east, north = satellite_service.bbox_from_center(0.0, 0.0, radius_km=2.5)

    assert west < 0 < east
    assert south < 0 < north
    assert np.isclose(east, 2.5 / 111.0)


def test_historical_and_latest_selection(stac_item_factory):
    early = stac_item_factory("early", "2024-01-10T10:00:00Z", 2)
    closest_cloudy = stac_item_factory("closest", "2024-01-15T10:00:00Z", 25)
    later = stac_item_factory("later", "2024-02-01T10:00:00Z", 5)

    assert satellite_service.pick_historical_scene([early, closest_cloudy], "2024-01-15")["id"] == "closest"
    assert satellite_service.pick_latest_scene([early, closest_cloudy, later])["id"] == "later"


def test_historical_search_relaxes_cloud_filter_and_reports_warnings(monkeypatch, stac_item_factory):
    calls = []
    item = stac_item_factory("relaxed", "2024-01-15T10:00:00Z", 35)

    def fake_search(_bbox, _date_range, cloud_max, **_kwargs):
        calls.append(cloud_max)
        return [item] if cloud_max >= 40 else []

    monkeypatch.setattr(satellite_service, "_stac_search", fake_search)
    scene, warnings = satellite_service.search_historical_with_fallback([0, 0, 1, 1], "2024-01-15", 20)

    assert scene["id"] == "relaxed"
    assert calls == [20, 30, 40]
    assert any("35% cloud cover" in warning for warning in warnings)
    assert any("±21-day" in warning for warning in warnings)


def test_latest_search_relaxes_cloud_filter_and_reports_warnings(monkeypatch, stac_item_factory):
    calls = []
    item = stac_item_factory("latest", "2025-01-15T10:00:00Z", 35)

    def fake_search(_bbox, _date_range, cloud_max, **_kwargs):
        calls.append(cloud_max)
        return [item] if cloud_max >= 30 else []

    monkeypatch.setattr(satellite_service, "_stac_search", fake_search)
    scene, warnings = satellite_service.search_latest_with_fallback(
        [0, 0, 1, 1],
        20,
        "43PGQ",
        satellite_service.datetime.fromisoformat("2024-01-15T00:00:00+00:00"),
    )

    assert scene["id"] == "latest"
    assert calls == [20, 30]
    assert any("35% cloud cover" in warning for warning in warnings)
    assert any("past 240 days" in warning for warning in warnings)


def test_pair_cache_replaces_oldest_entry():
    pairs = [object() for _ in range(5)]
    for index, pair in enumerate(pairs):
        satellite_service.store_cached_pair(f"key-{index}", pair)

    assert satellite_service.get_cached_pair("key-0") is None
    assert satellite_service.get_cached_pair("key-4") is pairs[4]


def test_equal_shape_with_different_transform_is_reprojected(raster_grids):
    band = np.arange(16, dtype=np.float32).reshape(raster_grids["shape"])

    aligned = satellite_service._align_band_to_reference(
        band,
        raster_grids["shifted_transform"],
        raster_grids["wgs84"],
        raster_grids["shape"],
        raster_grids["reference_transform"],
        raster_grids["wgs84"],
    )

    assert aligned.shape == band.shape
    assert not np.array_equal(aligned, band)


def test_equal_shape_with_different_crs_is_reprojected(raster_grids):
    band = np.arange(16, dtype=np.float32).reshape(raster_grids["shape"])

    aligned = satellite_service._align_band_to_reference(
        band,
        raster_grids["reference_transform"],
        raster_grids["web_mercator"],
        raster_grids["shape"],
        raster_grids["reference_transform"],
        raster_grids["wgs84"],
    )

    assert aligned.shape == band.shape
    assert not np.array_equal(aligned, band)


def test_equal_shape_with_matching_grid_skips_reprojection(raster_grids):
    band = np.arange(16, dtype=np.float32).reshape(raster_grids["shape"])

    aligned = satellite_service._align_band_to_reference(
        band,
        raster_grids["reference_transform"],
        raster_grids["wgs84"],
        raster_grids["shape"],
        raster_grids["reference_transform"],
        raster_grids["wgs84"],
    )

    assert aligned is band
