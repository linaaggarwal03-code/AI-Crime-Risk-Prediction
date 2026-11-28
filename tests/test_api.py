"""
Integration tests for FastAPI application endpoints (Stage 3).
Verifies HTTP status codes, schema validation, error translation,
prediction accuracy, risk tier assignments, and spatial hotspot queries.
"""

import datetime
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from app.main import app, API_VERSION


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provides a TestClient managing the FastAPI lifespan context."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# General & Health Endpoints
# ---------------------------------------------------------------------------

def test_root_endpoint(client: TestClient):
    """Verifies the discovery root endpoint."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["api_version"] == API_VERSION
    assert data["status"] == "operational"
    assert "/docs" in data["docs_url"]
    assert "predict" in data["endpoints"]
    assert "hotspots" in data["endpoints"]
    assert "health" in data["endpoints"]


def test_health_live_endpoint(client: TestClient):
    """Verifies process liveness probe."""
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json() == {"status": "alive"}


def test_health_ready_and_summary_endpoint(client: TestClient):
    """Verifies service readiness probe and verified record count of 166,474."""
    for path in ["/health/ready", "/health"]:
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        checks = data["checks"]
        assert checks["model_loaded"] is True
        assert checks["hotspots_loaded"] is True
        assert checks["kde_loaded"] is True
        assert checks["historical_store_ready"] is True
        assert data["historical_store_records"] == 166474


def test_cors_preflight_and_headers(client: TestClient):
    """Verifies CORS headers for allowed localhost origins."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    }
    res = client.options("/api/predict", headers=headers)
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ---------------------------------------------------------------------------
# Area Registry Endpoint
# ---------------------------------------------------------------------------

def test_community_areas_endpoint(client: TestClient):
    """Verifies that exactly 77 community areas are returned with strictly IDs 1 to 77."""
    res = client.get("/api/areas")
    assert res.status_code == 200
    data = res.json()
    assert data["total_areas"] == 77
    assert len(data["areas"]) == 77

    ids = [item["community_area_id"] for item in data["areas"]]
    assert ids == list(range(1, 78))


# ---------------------------------------------------------------------------
# Prediction Inference Endpoint (POST /api/predict)
# ---------------------------------------------------------------------------

def test_predict_historical_success(client: TestClient):
    """Verifies valid historical prediction with separated actual count and risk tiers."""
    payload = {
        "community_area": 8,
        "date": "2024-06-15",
        "include_hotspots": True,
        "hotspot_limit": 5,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Query echo
    assert data["query"]["community_area"] == 8
    assert data["query"]["date"] == "2024-06-15"
    assert data["query"]["is_historical"] is True

    # Prediction
    pred = data["prediction"]
    assert pred["predicted_crime_count"] >= 0.0
    assert pred["risk_tier"] in ["Low", "Moderate", "High", "Very High"]
    assert pred["risk_tier_code"] in [1, 2, 3, 4]
    assert pred["thresholds_used"] == {"low": 3.0, "moderate": 6.0, "high": 12.0}

    # Context & Error
    ctx = data["context"]
    assert ctx["historical_actual_count"] is not None
    assert ctx["prediction_error"] is not None
    expected_error = round(pred["predicted_crime_count"] - ctx["historical_actual_count"], 4)
    assert abs(ctx["prediction_error"] - expected_error) < 1e-5

    # Hotspots
    hotspots = data["hotspots"]
    assert hotspots["total_matching_clusters"] > 0
    assert len(hotspots["clusters"]) <= 5


def test_predict_without_hotspots(client: TestClient):
    """Verifies that include_hotspots=False returns an empty hotspots payload."""
    payload = {
        "community_area": 8,
        "date": "2024-06-15",
        "include_hotspots": False,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["hotspots"] == {}


def test_predict_warmup_date_rejection(client: TestClient):
    """Verifies rejection of dates prior to 2020-01-31 (warmup period)."""
    payload = {
        "community_area": 8,
        "date": "2020-01-15",
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 400
    assert "Dates prior to 2020-01-31 lack required 30-day historical lag features" in res.json()["detail"]


def test_predict_invalid_community_area(client: TestClient):
    """Verifies rejection of out-of-bounds community areas."""
    # Out of bounds area 0
    res_low = client.post("/api/predict", json={"community_area": 0, "date": "2024-06-15"})
    assert res_low.status_code == 422

    # Out of bounds area 78
    res_high = client.post("/api/predict", json={"community_area": 78, "date": "2024-06-15"})
    assert res_high.status_code == 422


def test_predict_future_insufficient_history_rejection(client: TestClient):
    """Verifies that requesting dates > 2025-12-31 without history raises HTTP 422."""
    payload = {
        "community_area": 8,
        "date": "2026-06-01",
        "recent_history": None,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
    assert "requires recent_history_df" in res.json()["detail"]


def test_predict_future_with_valid_history_success(client: TestClient):
    """Verifies dynamic feature synthesis for future date when valid 30-day history is supplied."""
    target_date = datetime.date(2026, 1, 1)
    dates_30 = [target_date - datetime.timedelta(days=d) for d in range(30, 0, -1)]

    records = []
    for dt in dates_30:
        for ca in range(1, 78):
            records.append({
                "community_area": ca,
                "date": str(dt),
                "crime_count": float((ca % 10) + (dt.day % 5)),
            })

    payload = {
        "community_area": 8,
        "date": "2026-01-01",
        "recent_history": records,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["query"]["is_historical"] is False
    assert data["query"]["date"] == "2026-01-01"
    assert data["context"]["historical_actual_count"] is None
    assert data["context"]["prediction_error"] is None
    assert data["prediction"]["predicted_crime_count"] >= 0.0
    assert data["prediction"]["risk_tier"] in ["Low", "Moderate", "High", "Very High"]


# ---------------------------------------------------------------------------
# Hotspot & Spatial Endpoints
# ---------------------------------------------------------------------------

def test_get_hotspots_area_32_cross_boundary(client: TestClient):
    """Verifies that Area 32 (The Loop) retrieves Hotspot Rank 1 (Cluster ID 9) with 8,125 incidents."""
    res = client.get("/api/hotspots/32?limit=10")
    assert res.status_code == 200
    data = res.json()

    assert data["community_area"] == 32
    assert data["total_matching_clusters"] == 6
    assert data["dominant_clusters_count"] == 5
    assert data["cross_boundary_clusters_count"] == 3

    # Cluster 9 must be present
    cluster_9 = next((c for c in data["clusters"] if c["hotspot_id"] == 9), None)
    assert cluster_9 is not None
    assert cluster_9["hotspot_rank"] == 1
    assert cluster_9["incident_count_total"] == 17199
    assert cluster_9["incident_count_in_area"] == 8125
    assert cluster_9["is_dominant_in_this_area"] is False
    assert abs(cluster_9["radius_meters"] - 2717.8) < 1.0


def test_get_hotspots_invalid_area_id(client: TestClient):
    """Verifies rejection of invalid area IDs in hotspot endpoint."""
    res = client.get("/api/hotspots/100")
    assert res.status_code == 422


def test_get_kde_density_surface(client: TestClient):
    """Verifies continuous 50x50 KDE density surface endpoint."""
    res = client.get("/api/kde")
    assert res.status_code == 200
    data = res.json()

    assert data["grid_size"] == 50
    assert data["bandwidth_meters"] == 500.0
    assert data["target_year"] == 2024

    lat_grid = data["lat_grid"]
    lon_grid = data["lon_grid"]
    density_grid = data["density_grid"]

    assert len(lat_grid) == 50 and len(lat_grid[0]) == 50
    assert len(lon_grid) == 50 and len(lon_grid[0]) == 50
    assert len(density_grid) == 50 and len(density_grid[0]) == 50


# ---------------------------------------------------------------------------
# Model Governance Endpoint
# ---------------------------------------------------------------------------

def test_get_model_info(client: TestClient):
    """Verifies model architecture and risk threshold governance endpoint."""
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()

    assert data["api_version"] == API_VERSION
    assert data["model_name"] == "Random Forest Regressor (Champion)"
    assert data["feature_count"] == 34
    assert len(data["feature_names"]) == 34
    assert data["model_version"] is None  # strictly verified not invented

    hyper = data["hyperparameters"]
    assert hyper["n_estimators"] == 150
    assert hyper["max_depth"] == 12

    thresholds = data["risk_thresholds"]
    assert thresholds["low_threshold_q25"] == 3.0
    assert thresholds["median_threshold_q50"] == 6.0
    assert thresholds["high_threshold_q75"] == 12.0
    assert thresholds["categories"] == ["Low", "Moderate", "High", "Very High"]

    date_range = data["historical_data_range"]
    assert date_range["start_date"] == "2020-01-31"
    assert date_range["end_date"] == "2025-12-31"
