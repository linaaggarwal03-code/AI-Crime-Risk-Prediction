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
    assert data["status"] == "PROVISIONAL_SPATIAL_PROXIMITY_ONLY"
    assert data["model_integration_status"] == "PENDING_LINA_SUPERVISED_MODEL_EXPORT"
    assert "nearest_hotspot" in data
    assert "distance_to_hotspot_km" in data
    assert "disclaimer" in data
