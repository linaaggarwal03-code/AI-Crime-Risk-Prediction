"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "dataset_records" in data
    assert "clusters_identified" in data


def test_summary_endpoint():
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)"
    assert "results_summary" in data


def test_hotspots_endpoint():
    response = client.get("/api/hotspots")
    assert response.status_code == 200
    data = response.json()
    assert data["total_clusters"] > 0
    assert len(data["clusters"]) == data["total_clusters"]
    first = data["clusters"][0]
    assert "cluster_id" in first
    assert "centroid_latitude" in first
    assert "centroid_longitude" in first


def test_cluster_summary_endpoint():
    response = client.get("/api/cluster-summary")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_historical_incidents_endpoint():
    response = client.get("/api/historical-incidents?limit=50")
    assert response.status_code == 200
    data = response.json()
    assert data["total_returned"] <= 50
    assert len(data["incidents"]) == data["total_returned"]


def test_predict_risk_provisional_endpoint():
    # Chicago Loop query point
    payload = {
        "latitude": 41.8781,
        "longitude": -87.6298,
        "date": "2026-10-27",
        "time": "14:30",
        "community_area": 32,
    }
    response = client.post("/api/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SPATIAL_PROXIMITY_ONLY"
    assert data["model_integration_status"] == "SEPARATE_ML_ENDPOINT_ACTIVE"
    assert "nearest_hotspot" in data
    assert "distance_to_hotspot_km" in data
    assert "disclaimer" in data
    assert data["query_id"] is not None
    assert data["database_backend"] in ("SQLite", "PostgreSQL")


def test_spatial_temporal_endpoint():
    response = client.get("/api/spatial-temporal")
    assert response.status_code == 200
    data = response.json()
    assert len(data["hourly_distribution"]) == 24
    assert len(data["time_windows"]) == 4
    assert "cluster_day_night_shifts" in data


def test_historical_incidents_filtered_endpoint():
    response = client.get("/api/historical-incidents?limit=25&time_window=Morning&category=THEFT")
    assert response.status_code == 200
    data = response.json()
    assert data["total_returned"] <= 25
    for pt in data["incidents"]:
        assert pt["primary_type"] == "THEFT"


def test_prediction_history_endpoint():
    response = client.get("/api/prediction-history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["database_backend"] in ("SQLite", "PostgreSQL")
    assert "history" in data
    assert len(data["history"]) >= 1


def test_ml_health_live_and_ready():
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "alive"}

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    data = ready.json()
    assert data["status"] == "ready"
    assert data["historical_store_records"] == 166474
    assert all(data["checks"].values())


def test_ml_areas_and_model_info():
    areas = client.get("/api/areas")
    assert areas.status_code == 200
    assert areas.json()["total_areas"] == 77

    info = client.get("/api/model-info")
    assert info.status_code == 200
    data = info.json()
    assert data["feature_count"] == 34
    assert data["risk_thresholds"]["categories"] == ["Low", "Moderate", "High", "Very High"]


def test_ml_predict_historical_and_future_guard():
    # Historical prediction via CrimeRiskService
    res = client.post("/api/predict", json={"community_area": 8, "date": "2024-06-15"})
    assert res.status_code == 200
    data = res.json()
    assert data["query"]["is_historical"] is True
    assert data["prediction"]["predicted_crime_count"] >= 0.0
    assert data["prediction"]["risk_tier"] in ["Low", "Moderate", "High", "Very High"]
    assert data["context"]["historical_actual_count"] is not None

    # Future prediction without 30-day history must return 422
    future_res = client.post("/api/predict", json={"community_area": 8, "date": "2026-06-01"})
    assert future_res.status_code == 422


def test_ml_area_hotspots_and_kde():
    hs = client.get("/api/hotspots/32?limit=5")
    assert hs.status_code == 200
    hs_data = hs.json()
    assert hs_data["community_area"] == 32
    assert hs_data["total_matching_clusters"] == 6

    kde = client.get("/api/kde")
    assert kde.status_code == 200
    kde_data = kde.json()
    assert kde_data["grid_size"] == 50
    assert len(kde_data["density_grid"]) == 50


def test_ml_predict_persists_and_retrieves_history():
    res = client.post("/api/predict", json={"community_area": 8, "date": "2024-06-15"})
    assert res.status_code == 200

    hist = client.get("/api/prediction-history?prediction_type=ml_forecast&limit=5")
    assert hist.status_code == 200
    data = hist.json()
    assert data["total_returned"] >= 1
    latest = data["history"][0]
    assert latest["prediction_type"] == "ml_forecast"
    assert latest["community_area"] == 8
    assert latest["latitude"] is None
    assert latest["longitude"] is None
    assert latest["predicted_crime_count"] is not None
    assert latest["risk_tier"] in ["Low", "Moderate", "High", "Very High"]


def test_ml_predict_survives_database_failure(monkeypatch):
    from backend.main import get_cached_data

    db = get_cached_data().get("db")
    if db is not None:
        monkeypatch.setattr(
            db,
            "save_ml_prediction",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Simulated DB failure")),
        )

    res = client.post("/api/predict", json={"community_area": 32, "date": "2024-06-15"})
    assert res.status_code == 200
    assert res.json()["prediction"]["predicted_crime_count"] >= 0.0
