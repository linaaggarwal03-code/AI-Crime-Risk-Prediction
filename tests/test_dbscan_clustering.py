"""
Unit tests for DBSCANHotspotDetector and Haversine Distance Clustering.
"""

import pytest
import numpy as np
import pandas as pd
from src.dbscan_clustering import DBSCANHotspotDetector, EARTH_RADIUS_KM


@pytest.fixture
def synthetic_cluster_data():
    """
    Creates synthetic points with two tightly clustered groups and one isolated noise point.
    Group 1: Chicago Loop (~41.88°N, -87.63°W)
    Group 2: Austin Community Area (~41.89°N, -87.76°W)
    Noise: Far south (~41.65°N, -87.55°W)
    """
    # 5 points in Group 1
    g1_lat = [41.8800, 41.8805, 41.8810, 41.8795, 41.8802]
    g1_lon = [-87.6300, -87.6305, -87.6310, -87.6295, -87.6301]

    # 5 points in Group 2
    g2_lat = [41.8900, 41.8905, 41.8910, 41.8895, 41.8902]
    g2_lon = [-87.7600, -87.7605, -87.7610, -87.7595, -87.7601]

    # 1 noise point
    noise_lat = [41.6500]
    noise_lon = [-87.5500]

    df = pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(11)],
        "Latitude": g1_lat + g2_lat + noise_lat,
        "Longitude": g1_lon + g2_lon + noise_lon,
        "Primary Type": ["THEFT"] * 5 + ["BATTERY"] * 5 + ["HOMICIDE"],
        "Community Area": [8] * 5 + [25] * 5 + [54],
    })
    return df


def test_haversine_eps_conversion():
    # 500 meters = 0.5 km
    detector = DBSCANHotspotDetector(eps_meters=500.0, min_samples=3)
    expected_radians = 0.5 / EARTH_RADIUS_KM
    assert pytest.approx(detector.eps_radians, rel=1e-5) == expected_radians


def test_dbscan_clustering_and_noise_separation(synthetic_cluster_data):
    # min_samples=4, eps=300m -> Should identify 2 clusters and 1 noise point
    detector = DBSCANHotspotDetector(eps_meters=300.0, min_samples=4)
    labels = detector.fit_predict(synthetic_cluster_data)

    assert len(labels) == 11
    # Noise point should have label -1
    assert labels[-1] == -1

    # Two distinct clusters formed
    non_noise_labels = set(labels[:-1])
    assert -1 not in non_noise_labels
    assert len(non_noise_labels) == 2


def test_cluster_summary_computation(synthetic_cluster_data):
    detector = DBSCANHotspotDetector(eps_meters=300.0, min_samples=4)
    labels = detector.fit_predict(synthetic_cluster_data)
    summary_df = detector.compute_cluster_summary(synthetic_cluster_data, labels)

    assert len(summary_df) == 2
    # Verify columns exist
    for col in [
        "cluster_id", "incident_count", "percentage_of_total",
        "centroid_latitude", "centroid_longitude", "primary_crime_type"
    ]:
        assert col in summary_df.columns

    # Verify counts
    assert list(summary_df["incident_count"]) == [5, 5]


def test_empty_dataframe_raises_error():
    detector = DBSCANHotspotDetector()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        detector.fit_predict(pd.DataFrame())


def test_insufficient_samples_raises_error():
    detector = DBSCANHotspotDetector(min_samples=20)
    small_df = pd.DataFrame({"Latitude": [41.88], "Longitude": [-87.63]})
    with pytest.raises(ValueError, match="fewer than min_samples"):
        detector.fit_predict(small_df)
