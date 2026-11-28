"""
FastAPI Backend Application for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Connects serialized Random Forest champion model, risk thresholds, historical feature store,
and spatial DBSCAN/KDE hotspot artifacts via CrimeRiskService.
"""

from contextlib import asynccontextmanager
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Query, Path, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.service import CrimeRiskService, InsufficientHistoryError
from app.schemas import (
    RootResponse,
    LivenessResponse,
    HealthChecks,
    HealthResponse,
    CommunityAreaItem,
    CommunityAreasResponse,
    PredictionRequest,
    PredictionResponse,
    HotspotsResponse,
    KdeResponse,
    ModelInfoResponse,
)

API_VERSION = "1.0.0"

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]


def get_cors_origins() -> List[str]:
    """Resolves configured CORS allowed origins from environment or sensible localhost defaults."""
    env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return DEFAULT_CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.
    Initializes CrimeRiskService singleton into application state.
    """
    try:
        app.state.service = CrimeRiskService()
        app.state.service_error = None
    except Exception as exc:
        app.state.service = None
        app.state.service_error = str(exc)
    yield


app = FastAPI(
    title="AI-Powered Urban Crime Risk & Hotspot Prediction API",
    version=API_VERSION,
    description=(
        "Production REST API for daily community-area crime count forecasting, "
        "operational four-tier risk classification, and localized DBSCAN/KDE hotspot retrieval."
    ),
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(InsufficientHistoryError)
async def insufficient_history_handler(request: Request, exc: InsufficientHistoryError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------

def get_service() -> CrimeRiskService:
    """Dependency that provides the initialized CrimeRiskService singleton."""
    service = getattr(app.state, "service", None)
    if service is None:
        err = getattr(app.state, "service_error", "Unknown initialization failure")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"CrimeRiskService is not ready: {err}",
        )
    return service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=RootResponse, tags=["General"])
async def root() -> RootResponse:
    """Root discovery endpoint providing service status and endpoint links."""
    return RootResponse(
        name="AI-Powered Urban Crime Risk & Hotspot Prediction API",
        api_version=API_VERSION,
        status="operational",
        docs_url="/docs",
        endpoints={
            "health": "/health",
            "health_liveness": "/health/live",
            "health_readiness": "/health/ready",
            "areas": "/api/areas",
            "predict": "/api/predict",
            "hotspots": "/api/hotspots/{area_id}",
            "kde": "/api/kde",
            "model_info": "/api/model-info",
        },
    )


@app.get("/health/live", response_model=LivenessResponse, tags=["Health"])
async def health_liveness() -> LivenessResponse:
    """Liveness probe: verifies the ASGI process is alive and receiving traffic."""
    return LivenessResponse(status="alive")


@app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
async def health_readiness() -> Any:
    """
    Readiness probe: verifies all ML artifacts and the historical feature store are loaded.
    Returns HTTP 200 when ready; HTTP 503 if any core artifact is missing or corrupted.
    """
    service = getattr(app.state, "service", None)
    if service is None:
        err = getattr(app.state, "service_error", "Service failed to initialize")
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
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload,
        )

    return HealthResponse(**payload)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_summary() -> Any:
    """Comprehensive health check aggregating liveness and readiness."""
    return await health_readiness()


@app.get("/api/areas", response_model=CommunityAreasResponse, tags=["Metadata"])
async def list_community_areas() -> CommunityAreasResponse:
    """
    Returns valid Chicago Community Areas (IDs 1 through 77).
    Strictly reports verified area IDs without fabricating unsupported area names.
    """
    areas = [CommunityAreaItem(community_area_id=i) for i in range(1, 78)]
    return CommunityAreasResponse(total_areas=len(areas), areas=areas)


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_crime_risk(
    request: PredictionRequest,
    service: CrimeRiskService = Depends(get_service),
) -> PredictionResponse:
    """
    Unified daily crime risk prediction, risk tier classification, and localized hotspot retrieval.
    - Historical dates (2020-01-31 to 2025-12-31) use precomputed feature store.
    - Future dates (> 2025-12-31) strictly require recent_history containing exactly 30 continuous observed days.
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

    return PredictionResponse(**result)


@app.get("/api/hotspots/{area_id}", response_model=HotspotsResponse, tags=["Hotspots"])
async def get_area_hotspots(
    area_id: int = Path(..., ge=1, le=77, description="Chicago Community Area ID (1-77)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of clusters to return"),
    service: CrimeRiskService = Depends(get_service),
) -> HotspotsResponse:
    """
    Retrieves localized DBSCAN hotspot clusters touching the specified Community Area.
    Preserves multi-area breakdown counts for cross-boundary clusters.
    """
    result = service.get_area_hotspots(community_area=area_id, limit=limit)
    result["community_area"] = area_id
    return HotspotsResponse(**result)


@app.get("/api/kde", response_model=KdeResponse, tags=["Hotspots"])
async def get_kde_density_surface(
    service: CrimeRiskService = Depends(get_service),
) -> KdeResponse:
    """
    Returns the precomputed continuous 50x50 macroscopic KDE density grid over Chicago.
    Converts 2D NumPy arrays to nested lists for JSON transport.
    """
    surface = service.get_kde_density_surface()
    return KdeResponse(
        grid_size=int(surface["grid_size"]),
        bandwidth_meters=float(surface["bandwidth_meters"]),
        target_year=int(surface["target_year"]),
        lat_grid=surface["lat_grid"].tolist(),
        lon_grid=surface["lon_grid"].tolist(),
        density_grid=surface["density_grid"].tolist(),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
async def get_model_info(
    service: CrimeRiskService = Depends(get_service),
) -> ModelInfoResponse:
    """
    Returns champion model architecture, training hyperparameters, frozen four-tier risk thresholds,
    and historical feature coverage dates.
    """
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
