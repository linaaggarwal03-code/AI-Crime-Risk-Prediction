"""
Pydantic v2 schemas for the AI-Powered Urban Crime Risk & Hotspot Prediction API.
Strictly mirrors CrimeRiskService input/output contracts without fabricating metadata.
"""

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class RecentHistoryRecord(BaseModel):
    """A single observed daily crime record for a specific community area."""
    community_area: int = Field(..., ge=1, le=77, description="Chicago Community Area ID (1-77)")
    date: datetime.date = Field(..., description="Observation calendar date (YYYY-MM-DD)")
    crime_count: float = Field(..., ge=0.0, description="Observed non-negative crime incident count")


class PredictionRequest(BaseModel):
    """Inference request body for daily crime risk and optional localized hotspots."""
    community_area: int = Field(..., ge=1, le=77, description="Chicago Community Area ID (1-77)")
    date: str = Field(..., description="Target query date in ISO format (YYYY-MM-DD)")
    include_hotspots: bool = Field(True, description="Whether to include localized DBSCAN hotspot clusters")
    hotspot_limit: int = Field(10, ge=1, le=50, description="Maximum number of hotspot clusters to return")
    recent_history: Optional[List[RecentHistoryRecord]] = Field(
        None,
        description=(
            "Required strictly for future unobserved dates (> 2025-12-31). "
            "Must contain exactly 30 continuous calendar dates across all 77 community areas."
        )
    )


# ---------------------------------------------------------------------------
# General & Health Schemas
# ---------------------------------------------------------------------------

class RootResponse(BaseModel):
    """Root discovery payload."""
    name: str
    api_version: str
    status: str
    docs_url: str
    endpoints: Dict[str, str]


class LivenessResponse(BaseModel):
    """Liveness probe payload."""
    status: str = "alive"


class HealthChecks(BaseModel):
    """Component readiness status indicators."""
    model_loaded: bool
    hotspots_loaded: bool
    kde_loaded: bool
    historical_store_ready: bool


class HealthResponse(BaseModel):
    """Readiness and comprehensive health payload."""
    status: str
    checks: HealthChecks
    historical_store_records: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Area Registry Schemas
# ---------------------------------------------------------------------------

class CommunityAreaItem(BaseModel):
    """Community Area ID item (strictly IDs 1-77, without invented names)."""
    community_area_id: int = Field(..., ge=1, le=77)


class CommunityAreasResponse(BaseModel):
    """Registry response of valid Community Areas."""
    total_areas: int = 77
    areas: List[CommunityAreaItem]


# ---------------------------------------------------------------------------
# Hotspot Schemas
# ---------------------------------------------------------------------------

class CentroidCoordinates(BaseModel):
    latitude: float
    longitude: float


class ClusterDetail(BaseModel):
    hotspot_rank: int
    hotspot_id: int
    incident_count_total: int
    incident_count_in_area: int
    is_dominant_in_this_area: bool
    centroid: CentroidCoordinates
    radius_meters: float
    top_crime_types: str


class HotspotsResponse(BaseModel):
    """Response containing localized DBSCAN clusters."""
    community_area: Optional[int] = None
    total_matching_clusters: int
    dominant_clusters_count: int
    cross_boundary_clusters_count: int
    clusters: List[ClusterDetail]


# ---------------------------------------------------------------------------
# Prediction Inference Schemas
# ---------------------------------------------------------------------------

class QueryEcho(BaseModel):
    community_area: int
    date: str
    is_historical: bool


class ThresholdsUsed(BaseModel):
    low: float
    moderate: float
    high: float


class PredictionDetails(BaseModel):
    predicted_crime_count: float
    risk_tier: str
    risk_tier_code: int
    thresholds_used: ThresholdsUsed


class PredictionContext(BaseModel):
    historical_actual_count: Optional[float] = None
    baseline_prev_day: float
    baseline_prev_week: float
    recent_7d_mean: float
    citywide_trend_lag1: float
    prediction_error: Optional[float] = None


class ModelInfoEcho(BaseModel):
    model_name: str
    feature_count: int
    version: Optional[str] = None


class PredictionResponse(BaseModel):
    """Standardized response from predict_area_risk."""
    query: QueryEcho
    prediction: PredictionDetails
    context: PredictionContext
    hotspots: Dict[str, Any]
    model_info: ModelInfoEcho


# ---------------------------------------------------------------------------
# KDE Surface Schemas
# ---------------------------------------------------------------------------

class KdeResponse(BaseModel):
    """Precomputed continuous 50x50 KDE density surface."""
    grid_size: int
    bandwidth_meters: float
    target_year: int
    lat_grid: List[List[float]]
    lon_grid: List[List[float]]
    density_grid: List[List[float]]


# ---------------------------------------------------------------------------
# Model Metadata & Governance Schemas
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    """Model governance and threshold metadata response."""
    api_version: str
    model_name: str
    model_type: str
    model_version: Optional[str] = None
    feature_count: int
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    risk_thresholds: Dict[str, Any]
    historical_data_range: Dict[str, str]
