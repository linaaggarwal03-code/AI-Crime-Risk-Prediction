"""
Unit tests for SpatialDataLoader and Geographic Validation.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import SpatialDataLoader, CHICAGO_GEO_BOUNDS


@pytest.fixture
def sample_raw_dataframe():
    """
    Creates a synthetic fixture with known valid, missing, and invalid coordinates.
    """
    data = {
        "ID": [101, 102, 103, 104, 105, 106, 107],
        "Case Number": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "Date": [
            "2025-01-01 12:00:00",
            "2025-01-02 13:00:00",
            "2025-01-03 14:00:00",
            "2025-01-04 15:00:00",
            "2025-01-05 16:00:00",
            "2025-01-06 17:00:00",
            "2025-01-07 18:00:00",
        ],
        "Primary Type": ["THEFT", "BATTERY", "ROBBERY", "ASSAULT", "THEFT", "BURGLARY", "THEFT"],
        "Community Area": [8, 25, 28, 32, 43, 8, 25],
        # 101: valid inside Chicago
        # 102: valid inside Chicago
        # 103: missing Latitude (NaN)
        # 104: missing Longitude (None)
        # 105: mathematically invalid lat (> 90)
        # 106: out-of-bounds (New York City coordinates)
        # 107: valid inside Chicago
        "Latitude": [41.8781, 41.8900, np.nan, 41.8500, 95.0000, 40.7128, 41.8781],
        "Longitude": [-87.6298, -87.6300, -87.6500, None, -87.6300, -74.0060, -87.6298],
    }
    return pd.DataFrame(data)


def test_column_auto_discovery(sample_raw_dataframe):
    loader = SpatialDataLoader(filepath="dummy.csv")
    mapping = loader.discover_columns(sample_raw_dataframe)
    assert mapping["latitude"] == "Latitude"
    assert mapping["longitude"] == "Longitude"
    assert mapping["category"] == "Primary Type"
    assert mapping["area"] == "Community Area"
    assert mapping["id"] == "ID"


def test_spatial_validation_audit(sample_raw_dataframe):
    loader = SpatialDataLoader(filepath="dummy.csv")
    loader.column_mapping = loader.discover_columns(sample_raw_dataframe)
    clean_df, audit = loader.validate_spatial_data(sample_raw_dataframe)

    # 7 total records:
    # Valid Chicago records: 101, 102, 107 -> 3 valid
    # Missing coords: 103 (lat NaN), 104 (lon None) -> 2
    # Invalid math coords: 105 (lat 95.0) -> 1
    # Out of bounds: 106 (NYC) -> 1
    assert audit["total_records"] == 7
    assert audit["valid_records"] == 3
    assert audit["excluded_records"] == 4
    assert audit["missing_coordinates"] == 2
    assert audit["invalid_mathematical_coordinates"] == 1
    assert audit["out_of_bounds_coordinates"] == 1

    # Check preserved records
    assert set(clean_df["ID"]) == {101, 102, 107}


def test_missing_coordinate_columns_raises_error():
    loader = SpatialDataLoader(filepath="dummy.csv")
    df_no_coords = pd.DataFrame({"ID": [1, 2], "Crime": ["THEFT", "BATTERY"]})
    loader.column_mapping = {}

    with pytest.raises(KeyError, match="Latitude or Longitude column not found"):
        loader.validate_spatial_data(df_no_coords)
