"""
FastAPI Backend Application for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Serves precomputed spatial analysis, DBSCAN hotspot clusters, and provisional prediction hooks.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import (
    OUTPUTS_DIR,
    SAMPLE_DATA_PATH,
    FULL_DATA_PATH,
    CLUSTER_SUMMARY_PATH,
    HOTSPOT_RESULTS_PATH,
    ANALYSIS_SUMMARY_PATH,
    PARAMETER_EXPERIMENTS_PATH,
    CORS_ORIGINS,
)
from backend.schemas import (
    HealthResponse,
    HotspotsResponse,
    ClusterItem,
    IncidentsResponse,
    IncidentPoint,
    PredictionRequest,
    PredictionResponse,
)
from src.feature_engineering import SpatialFeatureExtractor, EARTH_RADIUS_KM

app = FastAPI(
    title="Urban Crime Risk & Hotspot Prediction API",
    description="Academic REST backend serving spatial crime intelligence and DBSCAN hotspot clustering.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory as static route
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# In-memory cached artifacts
_CACHE: Dict[str, Any] = {}


def get_cached_data() -> Dict[str, Any]:
    """
    Loads and caches analysis artifacts in memory to prevent expensive file reads on every request.
    """
    if "loaded" in _CACHE:
        return _CACHE

    # 1. Summary JSON
    summary_data = {}
    if ANALYSIS_SUMMARY_PATH.exists():
        with open(ANALYSIS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    # 2. Cluster Summary DataFrame
    clusters_df = pd.DataFrame()
    if CLUSTER_SUMMARY_PATH.exists():
        clusters_df = pd.read_csv(CLUSTER_SUMMARY_PATH)

    # 3. Hotspot Assignments DataFrame (Subsampled if needed for fast queries)
    incidents_df = pd.DataFrame()
    if HOTSPOT_RESULTS_PATH.exists():
        incidents_df = pd.read_csv(HOTSPOT_RESULTS_PATH)
    elif SAMPLE_DATA_PATH.exists():
        incidents_df = pd.read_csv(SAMPLE_DATA_PATH)

    # 4. Feature Extractor
    feature_extractor = None
    if not clusters_df.empty:
        feature_extractor = SpatialFeatureExtractor(clusters_df)

    _CACHE["summary"] = summary_data
    _CACHE["clusters_df"] = clusters_df
    _CACHE["incidents_df"] = incidents_df
    _CACHE["feature_extractor"] = feature_extractor
    _CACHE["loaded"] = True
    return _CACHE


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check returning dataset volume and active spatial clustering state.
    """
    cache = get_cached_data()
    clusters_df = cache["clusters_df"]
    incidents_df = cache["incidents_df"]
    
    data_source = (
        "Crimes_-_2001_to_Present_20260820.csv"
        if FULL_DATA_PATH.exists()
        else "Crimes_Sample_2020_2025.csv"
    )

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="Urban Crime Risk & Hotspot Prediction API",
        dataset_records=len(incidents_df),
        clusters_identified=len(clusters_df),
        data_source=data_source,
    )


@app.get("/api/summary")
def get_analysis_summary():
    """
    Returns full analysis summary metadata, hyperparameters, and overall statistics.
    """
    cache = get_cached_data()
    if not cache["summary"]:
        raise HTTPException(
            status_code=404,
            detail="Analysis summary not found. Please run the spatial pipeline first.",
        )
    return cache["summary"]


@app.get("/api/hotspots", response_model=HotspotsResponse)
def get_hotspots():
    """
    Returns all detected DBSCAN crime hotspot clusters with centroids and extents.
    """
    cache = get_cached_data()
    clusters_df = cache["clusters_df"]
    summary = cache["summary"]

    if clusters_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Hotspots cluster summary not available. Please run DBSCAN clustering first.",
        )

    cluster_items = [
        ClusterItem(
            cluster_id=int(row["cluster_id"]),
            incident_count=int(row["incident_count"]),
            percentage_of_total=float(row["percentage_of_total"]),
            centroid_latitude=float(row["centroid_latitude"]),
            centroid_longitude=float(row["centroid_longitude"]),
            min_latitude=float(row["min_latitude"]),
            max_latitude=float(row["max_latitude"]),
            min_longitude=float(row["min_longitude"]),
            max_longitude=float(row["max_longitude"]),
            primary_crime_type=str(row["primary_crime_type"]),
            dominant_community_area=str(row["dominant_community_area"]),
        )
        for _, row in clusters_df.iterrows()
    ]

    params = summary.get("selected_hyperparameters", {})
    return HotspotsResponse(
        total_clusters=len(cluster_items),
        algorithm=summary.get("algorithm", "DBSCAN"),
        metric=summary.get("distance_metric", "Haversine"),
        eps_meters=float(params.get("eps_meters", 500.0)),
        min_samples=int(params.get("min_samples", 50)),
        clusters=cluster_items,
    )


@app.get("/api/cluster-summary")
def get_cluster_summary_table():
    """
    Returns raw cluster summary rows as JSON.
    """
    cache = get_cached_data()
    clusters_df = cache["clusters_df"]
    if clusters_df.empty:
        raise HTTPException(status_code=404, detail="Cluster summary data is missing.")
    return clusters_df.to_dict(orient="records")


@app.get("/api/experiments")
def get_parameter_experiments():
    """
    Returns grid of DBSCAN parameter sensitivity experiments.
    """
    if not PARAMETER_EXPERIMENTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Parameter experiments table not found.")
    df = pd.read_csv(PARAMETER_EXPERIMENTS_PATH)
    return df.to_dict(orient="records")


@app.get("/api/historical-incidents", response_model=IncidentsResponse)
def get_historical_incidents(
    limit: int = Query(default=1000, le=5000, description="Max incidents to return"),
    category: Optional[str] = Query(None, description="Filter by primary crime type"),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID (-1 for noise)"),
):
    """
    Returns historical incident points for interactive map rendering.
    """
    cache = get_cached_data()
    incidents_df = cache["incidents_df"]

    if incidents_df.empty:
        return IncidentsResponse(total_returned=0, total_available=0, incidents=[])

    filtered = incidents_df
    if category and "Primary Type" in filtered.columns:
        filtered = filtered[filtered["Primary Type"].str.upper() == category.upper()]

    if cluster_id is not None and "cluster_id" in filtered.columns:
        filtered = filtered[filtered["cluster_id"] == cluster_id]

    total_avail = len(filtered)
    sample_df = filtered.sample(n=min(limit, total_avail), random_state=42)

    lat_col = "Latitude" if "Latitude" in sample_df.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in sample_df.columns else "longitude"
    id_col = "ID" if "ID" in sample_df.columns else "id"
    cat_col = "Primary Type" if "Primary Type" in sample_df.columns else "primary_type"
    area_col = "Community Area" if "Community Area" in sample_df.columns else "community_area"
    cid_col = "cluster_id" if "cluster_id" in sample_df.columns else None

    points = []
    for _, row in sample_df.iterrows():
        points.append(
            IncidentPoint(
                id=str(row.get(id_col, "")),
                date=str(row.get("Date", "")),
                primary_type=str(row.get(cat_col, "Unknown")),
                community_area=str(row.get(area_col, "N/A")),
                latitude=float(row[lat_col]),
                longitude=float(row[lon_col]),
                cluster_id=int(row[cid_col]) if cid_col and pd.notnull(row[cid_col]) else None,
            )
        )

    return IncidentsResponse(
        total_returned=len(points),
        total_available=total_avail,
        incidents=points,
    )


@app.post("/api/predict-risk", response_model=PredictionResponse)
def predict_crime_risk(req: PredictionRequest):
    """
    Provisional risk prediction endpoint.
    Performs spatial proximity estimation against historical DBSCAN hotspots,
    while honestly documenting that live ML inference is pending Lina's final model testing.
    """
    cache = get_cached_data()
    clusters_df = cache["clusters_df"]
    extractor: Optional[SpatialFeatureExtractor] = cache["feature_extractor"]

    nearest_cluster = None
    dist_km = None
    risk_tier = "Moderate / Baseline"

    if extractor and not clusters_df.empty:
        dist_arr = extractor.distance_to_nearest_hotspot_km(
            np.array([req.latitude]), np.array([req.longitude])
        )
        dist_km = float(dist_arr[0])

        # Find nearest cluster details
        query_rad = np.radians([[req.latitude, req.longitude]])
        _, idx = extractor.centroid_tree.query(query_rad, k=1)
        nearest_row = clusters_df.iloc[idx[0][0]]
        nearest_cluster = {
            "cluster_id": int(nearest_row["cluster_id"]),
            "incident_count": int(nearest_row["incident_count"]),
            "dominant_community_area": str(nearest_row["dominant_community_area"]),
            "primary_crime_type": str(nearest_row["primary_crime_type"]),
            "centroid_latitude": float(nearest_row["centroid_latitude"]),
            "centroid_longitude": float(nearest_row["centroid_longitude"]),
        }

        # Provisional spatial risk heuristic based purely on distance to historic high-density cluster
        if dist_km <= 0.5:
            risk_tier = "High (Proximity to Historic Hotspot < 500m)"
        elif dist_km <= 1.5:
            risk_tier = "Moderate (Proximity to Historic Hotspot 0.5km - 1.5km)"
        else:
            risk_tier = "Low / Baseline (Dispersed Location > 1.5km from Hotspots)"

    return PredictionResponse(
        status="PROVISIONAL_SPATIAL_PROXIMITY_ONLY",
        query_coordinates={"latitude": req.latitude, "longitude": req.longitude},
        nearest_hotspot=nearest_cluster,
        distance_to_hotspot_km=dist_km,
        provisional_spatial_risk_tier=risk_tier,
        model_integration_status="PENDING_LINA_SUPERVISED_MODEL_EXPORT",
        disclaimer=(
            "Notice: This is a geometric proximity calculation relative to historical DBSCAN clusters. "
            "Final supervised classification and count forecasting are pending completion of Member 1's "
            "(Lina) model testing and artifact serialization. This does NOT represent a confirmed future crime forecast."
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
