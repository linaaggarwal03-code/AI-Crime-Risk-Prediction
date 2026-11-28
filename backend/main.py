"""
FastAPI Backend Application for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Serves precomputed spatial analysis, DBSCAN hotspot clusters, and provisional prediction hooks.
"""

from contextlib import asynccontextmanager
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Path as ApiPath, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    SPATIAL_TEMPORAL_PATH,
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
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
from src.feature_engineering import SpatialFeatureExtractor, EARTH_RADIUS_KM
from src.service import CrimeRiskService, InsufficientHistoryError
from database.db_manager import DatabaseManager
from app.schemas import (
    LivenessResponse,
    HealthResponse as MLHealthResponse,
    CommunityAreaItem,
    CommunityAreasResponse,
    PredictionRequest as MLPredictionRequest,
    PredictionResponse as MLPredictionResponse,
    HotspotsResponse as AreaHotspotsResponse,
    KdeResponse,
    ModelInfoResponse,
)

API_VERSION = "1.0.0"
_CACHE: Dict[str, Any] = {}


def _init_ml_service() -> Optional[CrimeRiskService]:
    """Initializes and caches the CrimeRiskService singleton once."""
    if "ml_service" not in _CACHE and "ml_service_error" not in _CACHE:
        try:
            _CACHE["ml_service"] = CrimeRiskService()
            _CACHE["ml_service_error"] = None
        except Exception as exc:
            _CACHE["ml_service"] = None
            _CACHE["ml_service_error"] = str(exc)
    return _CACHE.get("ml_service")


def get_ml_service() -> CrimeRiskService:
    """FastAPI dependency returning the initialized CrimeRiskService or 503."""
    service = _init_ml_service()
    if service is None:
        err = _CACHE.get("ml_service_error", "Unknown initialization failure")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"CrimeRiskService is not ready: {err}",
        )
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes CrimeRiskService on ASGI startup."""
    _init_ml_service()
    yield


app = FastAPI(
    title="Urban Crime Risk & Hotspot Prediction API",
    description="Unified REST backend serving spatial crime intelligence, DBSCAN hotspot clustering, and supervised ML crime-risk forecasting.",
    version=API_VERSION,
    lifespan=lifespan,
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


@app.exception_handler(InsufficientHistoryError)
async def insufficient_history_handler(request: Request, exc: InsufficientHistoryError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


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

    # 3. Hotspot Assignments DataFrame
    incidents_df = pd.DataFrame()
    if HOTSPOT_RESULTS_PATH.exists():
        incidents_df = pd.read_csv(HOTSPOT_RESULTS_PATH)
    elif SAMPLE_DATA_PATH.exists():
        incidents_df = pd.read_csv(SAMPLE_DATA_PATH)

    if not incidents_df.empty and "Date" in incidents_df.columns:
        parsed_dt = pd.to_datetime(incidents_df["Date"], errors="coerce")
        incidents_df["_hour"] = parsed_dt.dt.hour
        def _win(h):
            if pd.isna(h):
                return "Unknown"
            h = int(h)
            if 6 <= h < 12:
                return "Morning"
            elif 12 <= h < 18:
                return "Afternoon"
            elif 18 <= h <= 23:
                return "Evening"
            return "Night"
        incidents_df["_time_window"] = incidents_df["_hour"].apply(_win)

    # 4. Feature Extractor
    feature_extractor = None
    if not clusters_df.empty:
        feature_extractor = SpatialFeatureExtractor(clusters_df)

    # 5. Spatial-Temporal Summary
    spatial_temporal_data = {}
    if SPATIAL_TEMPORAL_PATH.exists():
        with open(SPATIAL_TEMPORAL_PATH, "r", encoding="utf-8") as f:
            spatial_temporal_data = json.load(f)

    # 6. Database Persistence Manager
    db_manager = DatabaseManager()

    _CACHE["summary"] = summary_data
    _CACHE["clusters_df"] = clusters_df
    _CACHE["incidents_df"] = incidents_df
    _CACHE["feature_extractor"] = feature_extractor
    _CACHE["spatial_temporal"] = spatial_temporal_data
    _CACHE["db"] = db_manager
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


@app.get("/api/spatial-temporal")
def get_spatial_temporal_analytics():
    """
    Returns Milestone 3 spatial-temporal breakdown (hourly, time-of-day windows,
    daytime vs. nighttime hotspot shifts, and category hotspot concentrations).
    """
    cache = get_cached_data()
    if not cache.get("spatial_temporal"):
        raise HTTPException(
            status_code=404,
            detail="Spatial-temporal summary not found. Run src/spatial_temporal_analysis.py first.",
        )
    return cache["spatial_temporal"]


@app.get("/api/historical-incidents", response_model=IncidentsResponse)
def get_historical_incidents(
    limit: int = Query(default=1000, le=5000, description="Max incidents to return"),
    category: Optional[str] = Query(None, description="Filter by primary crime type"),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID (-1 for noise)"),
    time_window: Optional[str] = Query(None, description="Filter by Morning, Afternoon, Evening, or Night"),
    community_area: Optional[int] = Query(None, description="Filter by Chicago Community Area ID"),
):
    """
    Returns historical incident points for interactive map rendering with multi-dimensional filtering.
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

    if time_window and "_time_window" in filtered.columns:
        filtered = filtered[filtered["_time_window"].str.lower() == time_window.lower()]

    if community_area is not None and "Community Area" in filtered.columns:
        filtered = filtered[filtered["Community Area"] == community_area]

    total_avail = len(filtered)
    if total_avail == 0:
        return IncidentsResponse(total_returned=0, total_available=0, incidents=[])

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
    Performs spatial proximity estimation against historical DBSCAN hotspots and persists
    the query to the relational database (PostgreSQL / SQLite fallback),
    while honestly documenting that live ML inference is pending Lina's final model integration.
    """
    cache = get_cached_data()
    clusters_df = cache["clusters_df"]
    extractor: Optional[SpatialFeatureExtractor] = cache["feature_extractor"]
    db: Optional[DatabaseManager] = cache.get("db")

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

    query_id = None
    db_backend = None
    if db is not None:
        target_ts = f"{req.date or '2026-11-25'} {req.time or '12:00'}"
        query_id = db.save_prediction_query(
            latitude=req.latitude,
            longitude=req.longitude,
            target_timestamp=target_ts,
            risk_tier=risk_tier,
            nearest_cluster_id=nearest_cluster["cluster_id"] if nearest_cluster else None,
            distance_km=dist_km,
        )
        db_backend = db.backend_type

    return PredictionResponse(
        status="SPATIAL_PROXIMITY_ONLY",
        query_id=query_id,
        database_backend=db_backend,
        query_coordinates={"latitude": req.latitude, "longitude": req.longitude},
        nearest_hotspot=nearest_cluster,
        distance_to_hotspot_km=dist_km,
        provisional_spatial_risk_tier=risk_tier,
        model_integration_status="SEPARATE_ML_ENDPOINT_ACTIVE",
        disclaimer=(
            "Notice: This endpoint performs geometric Haversine proximity analysis against historical DBSCAN "
            "hotspot clusters. For supervised Random Forest crime-count forecasting and four-tier risk "
            "classification, use POST /api/predict."
        ),
    )


@app.get("/api/prediction-history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    limit: int = Query(default=15, le=100),
    prediction_type: Optional[str] = Query(
        default=None,
        pattern="^(ml_forecast|spatial_proximity)$",
        description="Optional filter: 'ml_forecast' or 'spatial_proximity'",
    ),
):
    """
    Returns recent ML forecasts and spatial risk queries persisted in PostgreSQL / SQLite.
    """
    cache = get_cached_data()
    db: Optional[DatabaseManager] = cache.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Database persistence layer unavailable.")
    rows = db.get_recent_predictions(limit=limit, prediction_type=prediction_type)
    return PredictionHistoryResponse(
        database_backend=db.backend_type,
        total_returned=len(rows),
        history=[PredictionHistoryItem(**r) for r in rows],
    )


# ---------------------------------------------------------------------------
# Integrated ML Prediction & Hotspot Service Endpoints (CrimeRiskService)
# ---------------------------------------------------------------------------

@app.get("/health/live", response_model=LivenessResponse, tags=["Health"])
def health_liveness():
    """Liveness probe: verifies the ASGI process is alive and receiving traffic."""
    return LivenessResponse(status="alive")


@app.get("/health/ready", response_model=MLHealthResponse, tags=["Health"])
def health_readiness():
    """
    Readiness probe: verifies all serialized ML artifacts and the historical feature store
    are loaded in CrimeRiskService.
    """
    service = _init_ml_service()
    if service is None:
        err = _CACHE.get("ml_service_error", "CrimeRiskService failed to initialize")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "checks": {
                    "model_loaded": False,
                    "hotspots_loaded": False,
                    "kde_loaded": False,
                    "historical_store_ready": False,
                },
                "historical_store_records": 0,
                "error": err,
            },
        )

    model_ok = service.model is not None
    hotspots_ok = (service.hotspots_df is not None) and (len(service.hotspots_df) == 617)
    kde_ok = (service.kde_grid is not None) and (service.kde_grid["density_grid"].shape == (50, 50))
    history_records = len(service._historical_index)
    history_ok = history_records == 166474
    all_ready = model_ok and hotspots_ok and kde_ok and history_ok

    payload = {
        "status": "ready" if all_ready else "degraded",
        "checks": {
            "model_loaded": model_ok,
            "hotspots_loaded": hotspots_ok,
            "kde_loaded": kde_ok,
            "historical_store_ready": history_ok,
        },
        "historical_store_records": history_records,
    }
    if not all_ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
    return MLHealthResponse(**payload)


@app.get("/api/areas", response_model=CommunityAreasResponse, tags=["ML Prediction"])
def list_community_areas():
    """Returns valid Chicago Community Area IDs (1 through 77)."""
    areas = [CommunityAreaItem(community_area_id=i) for i in range(1, 78)]
    return CommunityAreasResponse(total_areas=len(areas), areas=areas)


@app.post("/api/predict", response_model=MLPredictionResponse, tags=["ML Prediction"])
def predict_area_crime_risk(
    request: MLPredictionRequest,
    service: CrimeRiskService = Depends(get_ml_service),
):
    """
    Unified supervised Random Forest crime-count forecast, four-tier risk classification,
    and localized cross-boundary DBSCAN hotspot retrieval via CrimeRiskService.
    """
    history_df: Optional[pd.DataFrame] = None
    if request.recent_history is not None:
        records = [r.model_dump() for r in request.recent_history]
        history_df = pd.DataFrame(records)

    result = service.predict_area_risk(
        community_area=request.community_area,
        date=request.date,
        recent_history_df=history_df,
        include_hotspots=request.include_hotspots,
        hotspot_limit=request.hotspot_limit,
    )

    # Persist successful ML forecast to DatabaseManager (non-fatal if DB write fails)
    try:
        db: Optional[DatabaseManager] = get_cached_data().get("db")
        if db is not None:
            db.save_ml_prediction(
                community_area=result["query"]["community_area"],
                prediction_date=result["query"]["date"],
                predicted_crime_count=result["prediction"]["predicted_crime_count"],
                risk_tier=result["prediction"]["risk_tier"],
                model_name=result["model_info"]["model_name"],
                model_version=result["model_info"].get("version"),
                is_historical=result["query"]["is_historical"],
                historical_actual_count=result["context"].get("historical_actual_count"),
            )
    except Exception as exc:
        print(f"[backend.main] Warning: failed to persist ML prediction to database: {exc}")

    return MLPredictionResponse(**result)


@app.get("/api/hotspots/{area_id}", response_model=AreaHotspotsResponse, tags=["ML Prediction"])
def get_area_hotspots_ml(
    area_id: int = ApiPath(..., ge=1, le=77, description="Chicago Community Area ID (1-77)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of clusters to return"),
    service: CrimeRiskService = Depends(get_ml_service),
):
    """Retrieves localized DBSCAN hotspot clusters touching the specified Community Area."""
    result = service.get_area_hotspots(community_area=area_id, limit=limit)
    result["community_area"] = area_id
    return AreaHotspotsResponse(**result)


@app.get("/api/kde", response_model=KdeResponse, tags=["ML Prediction"])
def get_kde_density_surface(service: CrimeRiskService = Depends(get_ml_service)):
    """Returns the precomputed continuous 50x50 macroscopic KDE density surface."""
    surface = service.get_kde_density_surface()
    return KdeResponse(
        grid_size=int(surface["grid_size"]),
        bandwidth_meters=float(surface["bandwidth_meters"]),
        target_year=int(surface["target_year"]),
        lat_grid=surface["lat_grid"].tolist(),
        lon_grid=surface["lon_grid"].tolist(),
        density_grid=surface["density_grid"].tolist(),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["ML Prediction"])
def get_ml_model_info(service: CrimeRiskService = Depends(get_ml_service)):
    """Returns champion Random Forest metadata, hyperparameters, and frozen risk thresholds."""
    return ModelInfoResponse(
        api_version=API_VERSION,
        model_name=service.model_name,
        model_type=service.metadata.get("model_type", "sklearn.ensemble.RandomForestRegressor"),
        model_version=service.model_version,
        feature_count=len(service.feature_names),
        feature_names=service.feature_names,
        hyperparameters=service.metadata.get("hyperparameters", {}),
        risk_thresholds={
            "low_threshold_q25": service.thresholds["low_threshold_q25"],
            "median_threshold_q50": service.thresholds["median_threshold_q50"],
            "high_threshold_q75": service.thresholds["high_threshold_q75"],
            "categories": service.thresholds_payload.get("categories", ["Low", "Moderate", "High", "Very High"]),
            "boundary_rules": service.thresholds_payload.get("boundary_rules", {}),
        },
        historical_data_range={
            "start_date": str(service.min_historical_date),
            "end_date": str(service.max_historical_date),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
