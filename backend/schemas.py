"""
Pydantic response and request schemas for the Urban Crime Risk API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "Urban Crime Risk & Hotspot Prediction API"
    dataset_records: int
    clusters_identified: int
    data_source: str


class ClusterItem(BaseModel):
    cluster_id: int
    incident_count: int
    percentage_of_total: float
    centroid_latitude: float
    centroid_longitude: float
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float
    primary_crime_type: str
    dominant_community_area: str


class HotspotsResponse(BaseModel):
    total_clusters: int
    algorithm: str
    metric: str
    eps_meters: float
    min_samples: int
    clusters: List[ClusterItem]


class IncidentPoint(BaseModel):
    id: Optional[str] = None
    date: Optional[str] = None
    primary_type: Optional[str] = None
    community_area: Optional[str] = None
    latitude: float
    longitude: float
    cluster_id: Optional[int] = None


class IncidentsResponse(BaseModel):
    total_returned: int
    total_available: int
    incidents: List[IncidentPoint]


class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    date: Optional[str] = Field(None, description="Target date string (YYYY-MM-DD)")
    time: Optional[str] = Field(None, description="Target time string (HH:MM)")
    community_area: Optional[int] = Field(None, ge=1, le=77, description="Chicago Community Area (1-77)")


class PredictionResponse(BaseModel):
    status: str
    query_id: Optional[int] = None
    database_backend: Optional[str] = None
    query_coordinates: Dict[str, float]
    nearest_hotspot: Optional[Dict[str, Any]] = None
    distance_to_hotspot_km: Optional[float] = None
    provisional_spatial_risk_tier: str
    model_integration_status: str
    disclaimer: str


class PredictionHistoryItem(BaseModel):
    id: int
    latitude: float
    longitude: float
    target_timestamp: str
    risk_tier: str
    nearest_cluster_id: Optional[int] = None
    distance_to_hotspot_km: Optional[float] = None
    model_name: str
    model_version: str
    created_at: str


class PredictionHistoryResponse(BaseModel):
    database_backend: str
    total_returned: int
    history: List[PredictionHistoryItem]
