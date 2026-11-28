"""
Unit tests for SpatialFeatureExtractor and Data Leakage Prevention.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import SpatialFeatureExtractor


@pytest.fixture
def mock_clusters_and_temporal_data():
    clusters_df = pd.DataFrame({
        "cluster_id": [0, 1],
        "centroid_latitude": [41.8800, 41.8000],
        "centroid_longitude": [-87.6300, -87.6500],
    })

    # Historical crime sequence:
    # 3 events in Jan 2025
    # 2 events in Feb 2025
    # 2 events in Mar 2025
    crimes_df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5, 6, 7],
        "Date": [
            "2025-01-10 10:00:00",
            "2025-01-15 10:00:00",
            "2025-01-20 10:00:00",
            "2025-02-05 10:00:00",
            "2025-02-15 10:00:00",
            "2025-03-01 10:00:00",
            "2025-03-10 10:00:00",
        ],
        "Latitude": [41.8801, 41.8802, 41.8803, 41.8804, 41.8805, 41.8806, 41.8807],
        "Longitude": [-87.6301, -87.6302, -87.6303, -87.6304, -87.6305, -87.6306, -87.6307],
        "Community Area": [8, 8, 8, 8, 25, 25, 25],
    })
    return clusters_df, crimes_df


def test_distance_to_nearest_hotspot(mock_clusters_and_temporal_data):
    clusters_df, _ = mock_clusters_and_temporal_data
    extractor = SpatialFeatureExtractor(clusters_df)

    # Point right on centroid 0
    lats = np.array([41.8800])
    lons = np.array([-87.6300])
    dist = extractor.distance_to_nearest_hotspot_km(lats, lons)
    assert dist[0] == pytest.approx(0.0, abs=0.01)


def test_point_in_time_data_leakage_safeguard(mock_clusters_and_temporal_data):
    clusters_df, crimes_df = mock_clusters_and_temporal_data
    extractor = SpatialFeatureExtractor(clusters_df)

    # Prediction time T = 2025-02-20
    # Lookback = 30 days -> Window: [2025-01-21, 2025-02-20)
    # Jan 10, Jan 15, Jan 20 are BEFORE window (excluded)
    # Feb 05, Feb 15 are IN window (included -> 2 events)
    # Mar 01, Mar 10 are AFTER T (FUTURE -> MUST BE EXCLUDED!)
    prediction_time = pd.Timestamp("2025-02-20 00:00:00")
    target_lats = np.array([41.8800])
    target_lons = np.array([-87.6300])

    counts = extractor.compute_point_in_time_density(
        historical_df=crimes_df,
        target_latitudes=target_lats,
        target_longitudes=target_lons,
        prediction_timestamp=prediction_time,
        lookback_days=30,
        radius_meters=1000.0,
    )

    assert counts[0] == 2, f"Expected 2 historical events in window; got {counts[0]}. Future events must not leak!"
