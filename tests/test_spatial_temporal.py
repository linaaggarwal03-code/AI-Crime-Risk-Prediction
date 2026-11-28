"""
Unit tests for SpatialTemporalAnalyzer (Milestone 3).
"""

import pandas as pd
import pytest
from src.spatial_temporal_analysis import (
    classify_time_window,
    classify_day_night,
    SpatialTemporalAnalyzer,
)


def test_time_window_classification():
    assert classify_time_window(8) == "Morning (06:00-11:59)"
    assert classify_time_window(14) == "Afternoon (12:00-17:59)"
    assert classify_time_window(20) == "Evening (18:00-23:59)"
    assert classify_time_window(2) == "Night (00:00-05:59)"


def test_day_night_classification():
    assert classify_day_night(6) == "Daytime"
    assert classify_day_night(17) == "Daytime"
    assert classify_day_night(18) == "Nighttime"
    assert classify_day_night(0) == "Nighttime"


def test_spatial_temporal_analyzer_summary(tmp_path):
    csv_path = tmp_path / "mock_hotspots.csv"
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Date": [
            "2025-05-01 09:00:00",
            "2025-05-01 15:00:00",
            "2025-05-01 21:00:00",
            "2025-05-01 02:00:00",
        ],
        "Primary Type": ["THEFT", "THEFT", "BATTERY", "ASSAULT"],
        "Community Area": [8, 8, 25, 25],
        "Latitude": [41.88, 41.88, 41.89, 41.89],
        "Longitude": [-87.63, -87.63, -87.76, -87.76],
        "cluster_id": [0, 0, 1, -1],
    })
    df.to_csv(csv_path, index=False)

    analyzer = SpatialTemporalAnalyzer(str(csv_path))
    analyzer.load_and_enrich()
    summary = analyzer.compute_summary()

    assert summary["total_records_with_valid_timestamps"] == 4
    assert summary["daytime_total"] == 2
    assert summary["nighttime_total"] == 2
    assert len(summary["hourly_distribution"]) == 24
    assert len(summary["time_windows"]) == 4
    assert len(summary["day_of_week_distribution"]) == 7
    assert len(summary["cluster_day_night_shifts"]) == 2
