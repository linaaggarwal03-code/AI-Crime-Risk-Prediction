"""
Data Loading and Geographic Coordinate Validation Module.
Handles automated column discovery, spatial coordinate boundary validation,
and exclusion auditing for Chicago Crime records.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# Default geographic bounding box for the City of Chicago and immediate jurisdiction
CHICAGO_GEO_BOUNDS = {
    "min_latitude": 41.60,
    "max_latitude": 42.10,
    "min_longitude": -88.05,
    "max_longitude": -87.45,
}

# Known candidate file paths in priority order
DEFAULT_DATA_PATHS = [
    os.path.join("data", "Crimes_-_2001_to_Present_20260820.csv"),
    os.path.join("..", "data", "Crimes_-_2001_to_Present_20260820.csv"),
    os.path.join("data", "Crimes_Sample_2020_2025.csv"),
    os.path.join("..", "data", "Crimes_Sample_2020_2025.csv"),
]


class SpatialDataLoader:
    """
    Robust data loader and geographic validator for urban crime records.
    Discovers schema programmatically and guarantees preservation of original records.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or self._resolve_default_filepath()
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.exclusion_audit: Dict[str, Any] = {}

    def _resolve_default_filepath(self) -> str:
        for path in DEFAULT_DATA_PATHS:
            if os.path.exists(path):
                return path
        # Fallback to the sample file path even if not yet created
        return DEFAULT_DATA_PATHS[2]

    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Loads CSV dataset with robust error and encoding handling.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Crime dataset could not be found at '{self.filepath}'. "
                f"Please ensure Lina's full dataset or the sample dataset is placed in 'data/'."
            )

        print(f"[SpatialDataLoader] Ingesting crime data from: {self.filepath}")
        
        # Attempt UTF-8, fallback to latin-1
        try:
            df = pd.read_csv(self.filepath, nrows=nrows, low_memory=False)
        except UnicodeDecodeError:
            print("[SpatialDataLoader] UTF-8 decoding failed; attempting latin-1...")
            df = pd.read_csv(self.filepath, nrows=nrows, low_memory=False, encoding="latin-1")

        self.raw_df = df
        print(f"[SpatialDataLoader] Successfully loaded {len(df):,} records with {len(df.columns)} columns.")
        self.column_mapping = self.discover_columns(df)
        return df

    @staticmethod
    def discover_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Discovers correct column names programmatically without hardcoded assumptions.
        """
        mapping: Dict[str, str] = {}
        cols_lower = {col.strip().lower(): col for col in df.columns}

        # Coordinate discovery
        lat_candidates = ["latitude", "lat", "y_coordinate", "y"]
        for c in lat_candidates:
            if c in cols_lower and "coordinate" not in c:
                mapping["latitude"] = cols_lower[c]
                break

        lon_candidates = ["longitude", "lon", "lng", "long", "x_coordinate", "x"]
        for c in lon_candidates:
            if c in cols_lower and "coordinate" not in c:
                mapping["longitude"] = cols_lower[c]
                break

        # Date & Time discovery
        date_candidates = ["date", "parsed_date", "occurred_at", "incident_date", "datetime"]
        for c in date_candidates:
            if c in cols_lower:
                mapping["date"] = cols_lower[c]
                break

        # Crime Category discovery
        cat_candidates = ["primary type", "primary_type", "crime_type", "offense_type", "category", "iucr"]
        for c in cat_candidates:
            if c in cols_lower:
                mapping["category"] = cols_lower[c]
                break

        # Geographic Area discovery
        area_candidates = ["community area", "community_area", "district", "ward", "beat"]
        for c in area_candidates:
            if c in cols_lower:
                mapping["area"] = cols_lower[c]
                break

        # Unique ID discovery
        id_candidates = ["id", "case number", "case_number", "incident_id"]
        for c in id_candidates:
            if c in cols_lower:
                mapping["id"] = cols_lower[c]
                break

        print(f"[SpatialDataLoader] Discovered column mapping: {mapping}")
        return mapping

    def validate_spatial_data(
        self,
        df: Optional[pd.DataFrame] = None,
        custom_bounds: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates spatial coordinates and documents all exclusion criteria.
        Returns a clean dataframe and an exclusion audit dictionary.
        """
        if df is None:
            if self.raw_df is None:
                self.load_data()
            df = self.raw_df

        bounds = custom_bounds or CHICAGO_GEO_BOUNDS
        lat_col = self.column_mapping.get("latitude")
        lon_col = self.column_mapping.get("longitude")

        if not lat_col or not lon_col:
            raise KeyError(
                f"Latitude or Longitude column not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        total_records = len(df)
        working_df = df.copy()

        # Coerce coordinates to numeric
        working_df[lat_col] = pd.to_numeric(working_df[lat_col], errors="coerce")
        working_df[lon_col] = pd.to_numeric(working_df[lon_col], errors="coerce")

        # Missing or non-numeric coordinates
        missing_lat = working_df[lat_col].isna()
        missing_lon = working_df[lon_col].isna()
        missing_coords_mask = missing_lat | missing_lon
        missing_count = int(missing_coords_mask.sum())

        # Mathematical coordinate validity (-90 to +90, -180 to +180)
        math_valid_mask = (
            (~missing_coords_mask)
            & (working_df[lat_col] >= -90.0)
            & (working_df[lat_col] <= 90.0)
            & (working_df[lon_col] >= -180.0)
            & (working_df[lon_col] <= 180.0)
        )
        invalid_math_count = int(((~missing_coords_mask) & (~math_valid_mask)).sum())

        # Study area bounds validation (Chicago jurisdiction)
        in_bounds_mask = (
            math_valid_mask
            & (working_df[lat_col] >= bounds["min_latitude"])
            & (working_df[lat_col] <= bounds["max_latitude"])
            & (working_df[lon_col] >= bounds["min_longitude"])
            & (working_df[lon_col] <= bounds["max_longitude"])
        )
        out_of_bounds_count = int((math_valid_mask & (~in_bounds_mask)).sum())

        # Duplicate coordinate points tracking (valid, but documented)
        duplicate_coord_count = int(working_df.loc[in_bounds_mask, [lat_col, lon_col]].duplicated().sum())

        # Build clean dataset
        valid_df = working_df[in_bounds_mask].copy()

        # Audit report
        audit = {
            "total_records": total_records,
            "valid_records": len(valid_df),
            "excluded_records": total_records - len(valid_df),
            "missing_coordinates": missing_count,
            "invalid_mathematical_coordinates": invalid_math_count,
            "out_of_bounds_coordinates": out_of_bounds_count,
            "duplicate_coordinate_pairs": duplicate_coord_count,
            "bounds_used": bounds,
            "latitude_range": (
                float(valid_df[lat_col].min()) if not valid_df.empty else None,
                float(valid_df[lat_col].max()) if not valid_df.empty else None,
            ),
            "longitude_range": (
                float(valid_df[lon_col].min()) if not valid_df.empty else None,
                float(valid_df[lon_col].max()) if not valid_df.empty else None,
            ),
        }

        self.clean_df = valid_df
        self.exclusion_audit = audit
        print(f"[SpatialDataLoader] Spatial Validation Audit:")
        print(f"  - Total records: {audit['total_records']:,}")
        print(f"  - Valid spatial records: {audit['valid_records']:,} ({audit['valid_records']/total_records*100:.2f}%)")
        print(f"  - Excluded records: {audit['excluded_records']:,} ({audit['excluded_records']/total_records*100:.2f}%)")
        print(f"    * Missing coords: {audit['missing_coordinates']:,}")
        print(f"    * Invalid coords: {audit['invalid_mathematical_coordinates']:,}")
        print(f"    * Out-of-bounds: {audit['out_of_bounds_coordinates']:,}")

        return valid_df, audit


def load_and_validate_crime_data(filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    """
    Convenience function returning (clean_df, column_mapping, exclusion_audit).
    """
    loader = SpatialDataLoader(filepath)
    loader.load_data()
    clean_df, audit = loader.validate_spatial_data()
    return clean_df, loader.column_mapping, audit


if __name__ == "__main__":
    df_clean, col_map, audit_info = load_and_validate_crime_data()
    print("DataLoader test completed successfully.")
