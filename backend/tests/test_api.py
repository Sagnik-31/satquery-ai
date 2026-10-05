from __future__ import annotations

from fastapi.testclient import TestClient

import main
import satellite_service


def test_health_reports_runtime_capabilities():
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "rasterio_available" in body


def test_analyze_rejects_missing_uploads():
    with TestClient(main.app) as client:
        response = client.post("/analyze")

    assert response.status_code == 422


def test_analyze_rejects_non_image_content():
    with TestClient(main.app) as client:
        response = client.post(
            "/analyze",
            files={
                "before_image": ("before.txt", b"not an image", "text/plain"),
                "after_image": ("after.txt", b"not an image", "text/plain"),
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "The earlier file must be an image."


def test_analyze_accepts_small_synthetic_pair(encoded_png_pair):
    before_png, after_png = encoded_png_pair
    with TestClient(main.app) as client:
        response = client.post(
            "/analyze",
            data={"query": "What changed?", "min_area": "100"},
            files={
                "before_image": ("before.png", before_png, "image/png"),
                "after_image": ("after.png", after_png, "image/png"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["category"] == "general"
    assert body["visuals"]["overlay_png"]


def test_temporal_analysis_requires_before_date():
    with TestClient(main.app) as client:
        response = client.post("/api/temporal-analysis", json={"location": "Bengaluru, India"})

    assert response.status_code == 422


def test_temporal_analysis_returns_mocked_retrieval_error(monkeypatch):
    def fail_retrieval(**_kwargs):
        raise satellite_service.SceneRetrievalError("offline fixture: no scenes")

    monkeypatch.setattr(main, "retrieve_temporal_pair", fail_retrieval)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/temporal-analysis",
            json={"location": "Bengaluru, India", "before_date": "2024-01-15"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "offline fixture: no scenes"
