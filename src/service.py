"""
Unified Prediction-Hotspot Service Module
AI-Powered Urban Crime Risk & Hotspot Prediction System

Implements CrimeRiskService, which integrates:
1. Serialized Champion RandomForestRegressor
2. Frozen four-tier risk classification quartiles
3. Pre-computed historical feature datasets (2020-01-31 through 2025-12-31)
4. Localized DBSCAN hotspot cluster summaries with cross-boundary multi-area breakdown
5. Macroscopic continuous KDE density grid
6. Strict validation and error handling for both historical and future queries
"""

import os
import sys
import json
import datetime
from typing import Dict, Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Guarantee project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import classify_risk_categories, add_calendar_features


class InsufficientHistoryError(ValueError):
    """Raised when an unobserved or future date lacks the required 30 continuous days of genuine observed history."""
    pass


class CrimeRiskService:
    """
    Unified operational service for crime risk forecasting, operational tier classification,
    and localized spatial hotspot retrieval.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        thresholds_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        hotspots_path: Optional[str] = None,
        kde_path: Optional[str] = None,
        feature_data_dir: Optional[str] = None,
    ):
        # 1. Resolve file paths
        self.model_path = model_path or os.path.join(PROJECT_ROOT, "models", "random_forest_champion.joblib")
        self.thresholds_path = thresholds_path or os.path.join(PROJECT_ROOT, "models", "risk_thresholds.json")
        self.metadata_path = metadata_path or os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
        self.hotspots_path = hotspots_path or os.path.join(PROJECT_ROOT, "data", "processed", "hotspot_clusters.parquet")
        self.kde_path = kde_path or os.path.join(PROJECT_ROOT, "data", "processed", "kde_density_grid.npz")
        self.feature_data_dir = feature_data_dir or os.path.join(PROJECT_ROOT, "data", "processed")

        # 2. Load serialized champion model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Serialized model artifact not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)

        # 3. Load frozen risk thresholds
        if not os.path.exists(self.thresholds_path):
            raise FileNotFoundError(f"Risk thresholds artifact not found at: {self.thresholds_path}")
        with open(self.thresholds_path, "r", encoding="utf-8") as f:
            self.thresholds_payload = json.load(f)
        self.thresholds = {
            'low_threshold_q25': float(self.thresholds_payload['low_threshold_q25']),
            'median_threshold_q50': float(self.thresholds_payload['median_threshold_q50']),
            'high_threshold_q75': float(self.thresholds_payload['high_threshold_q75']),
        }

        # 4. Load model metadata & exact ordered features
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Model metadata artifact not found at: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata['feature_names']
        self.model_name = self.metadata.get('model_name', 'Random Forest Regressor (Champion)')
        # Do not invent unsupported version strings; read version from metadata or default to None
        self.model_version = self.metadata.get('model_version', None)

        # 5. Load hotspot clusters artifact
        if not os.path.exists(self.hotspots_path):
            raise FileNotFoundError(f"Hotspot clusters artifact not found at: {self.hotspots_path}")
        self.hotspots_df = pd.read_parquet(self.hotspots_path)

        # 6. Load KDE density grid artifact
        if not os.path.exists(self.kde_path):
            raise FileNotFoundError(f"KDE density grid artifact not found at: {self.kde_path}")
        self.kde_grid = np.load(self.kde_path)

        # 7. Load and index historical feature datasets (2020-01-31 through 2025-12-31)
        self._load_historical_feature_store()

    def _load_historical_feature_store(self) -> None:
        """
        Loads train, val, and test feature Parquet datasets into an indexed in-memory store
        covering 2020-01-31 through 2025-12-31.
        """
        train_path = os.path.join(self.feature_data_dir, "train_features.parquet")
        val_path = os.path.join(self.feature_data_dir, "val_features.parquet")
        test_path = os.path.join(self.feature_data_dir, "test_features.parquet")

        for p in [train_path, val_path, test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required historical feature partition missing: {p}")

        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_test = pd.read_parquet(test_path)

        cols_to_keep = ['community_area', 'date', 'crime_count'] + [
            c for c in self.feature_names if c not in ['community_area', 'date']
        ]

        df_combined = pd.concat([df_train[cols_to_keep], df_val[cols_to_keep], df_test[cols_to_keep]], ignore_index=True)
        # Convert date to standard datetime.date
        df_combined['date'] = pd.to_datetime(df_combined['date']).dt.date
        df_combined['community_area'] = df_combined['community_area'].astype(int)

        # Store min and max historical bounds
        self.min_historical_date = df_combined['date'].min()  # 2020-01-31
        self.max_historical_date = df_combined['date'].max()  # 2025-12-31

        # Index by (community_area, date) for O(1) indexed access (preserve columns for feature extraction)
        self._historical_index = df_combined.set_index(['community_area', 'date'], drop=False)

    def _validate_inputs(self, community_area: Any, date: Any) -> tuple[int, datetime.date]:
        """
        Strictly validates community_area and date.
        """
        # Validate community_area
        try:
            ca_int = int(community_area)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid community_area '{community_area}'. Must be an integer between 1 and 77.")

        if ca_int < 1 or ca_int > 77:
            raise ValueError(f"Invalid community_area '{community_area}'. Must be an integer between 1 and 77.")

        # Validate date
        if isinstance(date, datetime.datetime):
            query_date = date.date()
        elif isinstance(date, datetime.date):
            query_date = date
        elif isinstance(date, str):
            try:
                query_date = datetime.date.fromisoformat(date.strip())
            except ValueError:
                raise ValueError(f"Invalid date format '{date}'. Expected ISO format 'YYYY-MM-DD'.")
        else:
            raise TypeError(f"Unsupported date type: {type(date)}. Expected 'YYYY-MM-DD' string or datetime.date.")

        # Reject warmup dates strictly before 2020-01-31
        earliest_valid = datetime.date(2020, 1, 31)
        if query_date < earliest_valid:
            raise ValueError(
                f"Requested date '{query_date}' is before {earliest_valid}. "
                f"Dates prior to 2020-01-31 lack required 30-day historical lag features."
            )

        return ca_int, query_date

    def _validate_recent_history_df(self, recent_history_df: pd.DataFrame, target_date: datetime.date) -> pd.DataFrame:
        """
        Strictly validates the 30-day recent history table:
        - Exactly 30 continuous calendar dates immediately preceding target_date (T-30 through T-1).
        - All 77 Community Areas present for every date (exactly 30 * 77 = 2,310 rows).
        - No duplicate (community_area, date) pairs.
        - Non-negative, numeric, non-null observed crime counts.
        """
        if not isinstance(recent_history_df, pd.DataFrame):
            raise TypeError(
                f"recent_history_df must be a pandas DataFrame, got {type(recent_history_df)}."
            )

        required_cols = {'community_area', 'date', 'crime_count'}
        if not required_cols.issubset(recent_history_df.columns):
            raise ValueError(
                f"recent_history_df is missing required columns. Expected at least {required_cols}, "
                f"got {set(recent_history_df.columns)}."
            )

        df = recent_history_df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['community_area'] = pd.to_numeric(df['community_area'], errors='coerce')
        df['crime_count'] = pd.to_numeric(df['crime_count'], errors='coerce')

        # Check for NaNs
        if df['community_area'].isna().any() or df['date'].isna().any() or df['crime_count'].isna().any():
            raise ValueError("recent_history_df contains NaN or non-numeric values.")

        df['community_area'] = df['community_area'].astype(int)

        # Check for non-negative counts
        if (df['crime_count'] < 0.0).any():
            raise ValueError("recent_history_df contains negative crime counts. Observed counts must be >= 0.")

        # Check for duplicate (community_area, date)
        if df.duplicated(subset=['community_area', 'date']).any():
            raise ValueError("recent_history_df contains duplicate (community_area, date) records.")

        # Check expected continuous date sequence: T-30 through T-1
        expected_dates = [target_date - datetime.timedelta(days=d) for d in range(30, 0, -1)]
        unique_dates = sorted(df['date'].unique())

        if unique_dates != expected_dates:
            raise InsufficientHistoryError(
                f"recent_history_df does not contain the exact required 30 continuous dates "
                f"from {expected_dates[0]} through {expected_dates[-1]} immediately preceding {target_date}."
            )

        # Check that all 77 community areas are present for every date
        expected_areas = set(range(1, 78))
        counts_per_date = df.groupby('date')['community_area'].apply(set)
        for dt, areas in counts_per_date.items():
            if areas != expected_areas:
                missing_areas = expected_areas - areas
                raise InsufficientHistoryError(
                    f"recent_history_df is missing community areas on date {dt}: {sorted(list(missing_areas))}."
                )

        if len(df) != 30 * 77:
            raise InsufficientHistoryError(
                f"recent_history_df must contain exactly 2,310 rows (30 dates x 77 areas), got {len(df)}."
            )

        return df

    def _synthesize_future_features(
        self,
        community_area: int,
        target_date: datetime.date,
        history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Dynamically synthesizes the exact 34-feature vector for a future date T
        using the validated 30-day genuine history buffer (T-30 to T-1).
        """
        # 1. Base row with target date and area
        base_df = pd.DataFrame([{'date': target_date, 'community_area': community_area}])

        # 2. Calendar features (12 features)
        cal_df = add_calendar_features(base_df)

        # 3. Lags (6 features) from target area history
        area_history = history_df[history_df['community_area'] == community_area].set_index('date')['crime_count']
        
        cal_df['lag_1'] = float(area_history.get(target_date - datetime.timedelta(days=1), np.nan))
        cal_df['lag_2'] = float(area_history.get(target_date - datetime.timedelta(days=2), np.nan))
        cal_df['lag_3'] = float(area_history.get(target_date - datetime.timedelta(days=3), np.nan))
        cal_df['lag_7'] = float(area_history.get(target_date - datetime.timedelta(days=7), np.nan))
        cal_df['lag_14'] = float(area_history.get(target_date - datetime.timedelta(days=14), np.nan))
        cal_df['lag_28'] = float(area_history.get(target_date - datetime.timedelta(days=28), np.nan))

        # 4. Rolling statistics (12 features) strictly over observed T-1 backwards
        counts_7 = [area_history[target_date - datetime.timedelta(days=d)] for d in range(1, 8)]
        counts_14 = [area_history[target_date - datetime.timedelta(days=d)] for d in range(1, 15)]
        counts_30 = [area_history[target_date - datetime.timedelta(days=d)] for d in range(1, 31)]

        cal_df['rolling_mean_7'] = float(np.mean(counts_7))
        cal_df['rolling_std_7'] = float(np.std(counts_7, ddof=1))
        cal_df['rolling_min_7'] = float(np.min(counts_7))
        cal_df['rolling_max_7'] = float(np.max(counts_7))

        cal_df['rolling_mean_14'] = float(np.mean(counts_14))
        cal_df['rolling_std_14'] = float(np.std(counts_14, ddof=1))
        cal_df['rolling_min_14'] = float(np.min(counts_14))
        cal_df['rolling_max_14'] = float(np.max(counts_14))

        cal_df['rolling_mean_30'] = float(np.mean(counts_30))
        cal_df['rolling_std_30'] = float(np.std(counts_30, ddof=1))
        cal_df['rolling_min_30'] = float(np.min(counts_30))
        cal_df['rolling_max_30'] = float(np.max(counts_30))

        # 5. Momentum ratio
        cal_df['ratio_lag1_to_rolling7'] = (cal_df['lag_1'] + 1.0) / (cal_df['rolling_mean_7'] + 1.0)

        # 6. Spatial context: citywide daily totals across all 77 areas
        citywide_daily = history_df.groupby('date')['crime_count'].sum()
        cal_df['citywide_lag_1'] = float(citywide_daily[target_date - datetime.timedelta(days=1)])
        
        citywide_7 = [citywide_daily[target_date - datetime.timedelta(days=d)] for d in range(1, 8)]
        cal_df['citywide_rolling_7'] = float(np.mean(citywide_7))

        # Ensure community_area column is int
        cal_df['community_area'] = int(community_area)

        # Validate that all 34 features exist and are non-null
        feature_row = cal_df[self.feature_names]
        if feature_row.isna().any().any():
            raise ValueError("Feature synthesis produced NaN values.")

        return feature_row

    def get_area_hotspots(self, community_area: int | str, limit: int = 10) -> Dict[str, Any]:
        """
        Queries localized DBSCAN clusters that touch the specified Community Area.
        Preserves cross-boundary multi-area breakdown counts.
        """
        ca_int, _ = self._validate_inputs(community_area, datetime.date(2024, 1, 1))
        ca_str = str(ca_int)

        # 1. Matching clusters: either dominant or in all_community_areas
        mask = self.hotspots_df['all_community_areas'].apply(lambda areas: ca_str in areas.split(','))
        matching_df = self.hotspots_df[mask].copy()

        total_matching = len(matching_df)
        if total_matching == 0:
            return {
                "total_matching_clusters": 0,
                "dominant_clusters_count": 0,
                "cross_boundary_clusters_count": 0,
                "clusters": []
            }

        dominant_count = int((matching_df['dominant_community_area'] == ca_int).sum())
        
        # A cluster is cross-boundary if it spans >1 area
        cross_boundary_mask = matching_df['all_community_areas'].apply(
            lambda a: len([x for x in str(a).split(',') if x.strip()]) > 1
        )
        cross_boundary_count = int(cross_boundary_mask.astype(int).sum())

        # Extract area-specific incident counts from JSON breakdown
        def get_area_count(breakdown_json: str) -> int:
            try:
                bd = json.loads(breakdown_json)
                return int(bd.get(ca_str, 0))
            except Exception:
                return 0

        matching_df['incident_count_in_area'] = matching_df['area_incident_breakdown'].apply(get_area_count)
        matching_df['is_dominant_in_this_area'] = matching_df['dominant_community_area'] == ca_int

        # Sort: incidents inside area descending, then cluster incident count descending
        sorted_df = matching_df.sort_values(
            by=['incident_count_in_area', 'incident_count'],
            ascending=[False, False]
        )

        clusters_output = []
        for _, row in sorted_df.head(limit).iterrows():
            clusters_output.append({
                "hotspot_rank": int(row['hotspot_rank']),
                "hotspot_id": int(row['hotspot_id']),
                "incident_count_total": int(row['incident_count']),
                "incident_count_in_area": int(row['incident_count_in_area']),
                "is_dominant_in_this_area": bool(row['is_dominant_in_this_area']),
                "centroid": {
                    "latitude": float(row['centroid_latitude']),
                    "longitude": float(row['centroid_longitude'])
                },
                "radius_meters": float(row['max_radius_meters']),
                "top_crime_types": str(row['top_crime_types'])
            })

        return {
            "total_matching_clusters": total_matching,
            "dominant_clusters_count": dominant_count,
            "cross_boundary_clusters_count": cross_boundary_count,
            "clusters": clusters_output
        }

    def get_kde_density_surface(self) -> Dict[str, Any]:
        """
        Returns the pre-computed macroscopic 50x50 continuous KDE density surface.
        """
        return {
            "lat_grid": self.kde_grid["lat_grid"],
            "lon_grid": self.kde_grid["lon_grid"],
            "density_grid": self.kde_grid["density_grid"],
            "bandwidth_meters": float(self.kde_grid["bandwidth_meters"]),
            "grid_size": int(self.kde_grid["grid_size"]),
            "target_year": int(self.kde_grid["target_year"])
        }

    def predict_area_risk(
        self,
        community_area: int | str,
        date: str | datetime.date,
        recent_history_df: Optional[pd.DataFrame] = None,
        include_hotspots: bool = True,
        hotspot_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Unified inference method for predicting daily crime risk and hotspots.
        """
        ca_int, query_date = self._validate_inputs(community_area, date)

        is_historical = (self.min_historical_date <= query_date <= self.max_historical_date)
        historical_actual = None
        prediction_error = None
        context_dict: Dict[str, Any] = {}

        if is_historical:
            # Historical fast-path
            idx = (ca_int, query_date)
            if idx not in self._historical_index.index:
                raise KeyError(
                    f"Requested historical record ({ca_int}, {query_date}) is missing from historical feature store."
                )

            row = self._historical_index.loc[idx]
            # Ensure row is a Series
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Extract 34 features in exact order
            feature_row = pd.DataFrame([row[self.feature_names]])
            historical_actual = float(row['crime_count'])

            # Baseline and context metrics
            context_dict = {
                "historical_actual_count": historical_actual,
                "baseline_prev_day": float(row['lag_1']),
                "baseline_prev_week": float(row['lag_7']),
                "recent_7d_mean": float(row['rolling_mean_7']),
                "citywide_trend_lag1": float(row['citywide_lag_1'])
            }
        else:
            # Future / unobserved date
            if recent_history_df is None:
                raise InsufficientHistoryError(
                    f"Requested date '{query_date}' is outside the historical store ({self.min_historical_date} to {self.max_historical_date}) "
                    f"and requires recent_history_df containing exactly 30 continuous observed days immediately preceding {query_date}."
                )

            valid_history = self._validate_recent_history_df(recent_history_df, query_date)
            feature_row = self._synthesize_future_features(ca_int, query_date, valid_history)

            # Historical actual is strictly None for future dates
            context_dict = {
                "historical_actual_count": None,
                "baseline_prev_day": float(feature_row['lag_1'].iloc[0]),
                "baseline_prev_week": float(feature_row['lag_7'].iloc[0]),
                "recent_7d_mean": float(feature_row['rolling_mean_7'].iloc[0]),
                "citywide_trend_lag1": float(feature_row['citywide_lag_1'].iloc[0])
            }

        # Model Inference
        raw_pred = float(self.model.predict(feature_row[self.feature_names])[0])
        pred_clipped = max(0.0, raw_pred)

        # Classification
        risk_tier = classify_risk_categories(pred_clipped, self.thresholds)
        tier_codes = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}

        if historical_actual is not None:
            prediction_error = round(pred_clipped - historical_actual, 4)
        context_dict["prediction_error"] = prediction_error

        # Hotspots
        hotspot_payload = self.get_area_hotspots(ca_int, limit=hotspot_limit) if include_hotspots else {}

        # Build query dict (no invented area names)
        query_dict: Dict[str, Any] = {
            "community_area": ca_int,
            "date": str(query_date),
            "is_historical": is_historical
        }

        # Build final standardized response
        response = {
            "query": query_dict,
            "prediction": {
                "predicted_crime_count": round(pred_clipped, 4),
                "risk_tier": risk_tier,
                "risk_tier_code": tier_codes.get(risk_tier, 0),
                "thresholds_used": {
                    "low": self.thresholds['low_threshold_q25'],
                    "moderate": self.thresholds['median_threshold_q50'],
                    "high": self.thresholds['high_threshold_q75']
                }
            },
            "context": context_dict,
            "hotspots": hotspot_payload,
            "model_info": {
                "model_name": self.model_name,
                "feature_count": len(self.feature_names)
            }
        }

        if self.model_version is not None:
            response["model_info"]["version"] = self.model_version

        return response
