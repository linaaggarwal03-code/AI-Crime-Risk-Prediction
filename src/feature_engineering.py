"""
Geographic Spatial Feature Engineering Module.
Investigates spatial predictors to support future supervised crime risk models
while enforcing strict point-in-time calculation to eliminate future-to-past data leakage.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EARTH_RADIUS_KM = 6371.0088


class SpatialFeatureExtractor:
    """
    Extracts spatial context features for urban crime risk prediction.
    Enforces temporal integrity: features for time T are calculated ONLY from records with t < T.
    """

    def __init__(
        self,
        cluster_summary_df: Optional[pd.DataFrame] = None,
        earth_radius_km: float = EARTH_RADIUS_KM,
    ):
        self.earth_radius_km = earth_radius_km
        self.cluster_summary = cluster_summary_df
        self.centroid_tree: Optional[BallTree] = None
        self.centroid_coords: Optional[np.ndarray] = None

        if cluster_summary_df is not None and not cluster_summary_df.empty:
            self._init_centroid_tree(cluster_summary_df)

    def _init_centroid_tree(self, cluster_summary_df: pd.DataFrame):
        coords = cluster_summary_df[["centroid_latitude", "centroid_longitude"]].to_numpy()
        self.centroid_coords = coords
        coords_rad = np.radians(coords)
        self.centroid_tree = BallTree(coords_rad, metric="haversine")

    def distance_to_nearest_hotspot_km(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """
        Computes the great-circle Haversine distance (in kilometers) from query locations
        to the nearest historically identified DBSCAN cluster centroid.
        """
        if self.centroid_tree is None:
            raise ValueError("Centroid BallTree is not initialized. Provide cluster_summary_df first.")

        query_rad = np.radians(np.column_stack([latitudes, longitudes]))
        dist_rad, _ = self.centroid_tree.query(query_rad, k=1)
        dist_km = dist_rad.flatten() * self.earth_radius_km
        return np.round(dist_km, 3)

    def compute_point_in_time_density(
        self,
        historical_df: pd.DataFrame,
        target_latitudes: np.ndarray,
        target_longitudes: np.ndarray,
        prediction_timestamp: pd.Timestamp,
        lookback_days: int = 30,
        radius_meters: float = 500.0,
        date_col: str = "Date",
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
    ) -> np.ndarray:
        """
        DATA LEAKAGE SAFEGUARD:
        Calculates historical incident counts within `radius_meters` strictly using events
        occurring in the window [T - lookback_days, T). Never includes events at or after T.
        """
        # Ensure dates are parsed
        if not pd.api.types.is_datetime64_any_dtype(historical_df[date_col]):
            hist_dates = pd.to_datetime(historical_df[date_col], errors="coerce")
        else:
            hist_dates = historical_df[date_col]

        window_start = prediction_timestamp - pd.Timedelta(days=lookback_days)
        window_end = prediction_timestamp

        # Strict past-only filter
        mask = (hist_dates >= window_start) & (hist_dates < window_end)
        past_records = historical_df[mask].dropna(subset=[lat_col, lon_col])

        if past_records.empty:
            return np.zeros(len(target_latitudes), dtype=int)

        past_coords_rad = np.radians(past_records[[lat_col, lon_col]].to_numpy())
        tree = BallTree(past_coords_rad, metric="haversine")

        radius_rad = (radius_meters / 1000.0) / self.earth_radius_km
        target_coords_rad = np.radians(np.column_stack([target_latitudes, target_longitudes]))

        counts = tree.query_radius(target_coords_rad, r=radius_rad, count_only=True)
        return counts

    def compute_community_area_historical_rates(
        self,
        historical_df: pd.DataFrame,
        area_col: str = "Community Area",
    ) -> Dict[str, float]:
        """
        Computes historical incident rate by Community Area as a baseline spatial prior.
        """
        area_series = historical_df[area_col].dropna().astype(str)
        total = len(area_series)
        rates = (area_series.value_counts() / total).to_dict()
        return {str(k): round(float(v), 5) for k, v in rates.items()}


def generate_feature_proposal_document(output_path: str = "docs/feature_proposal.md"):
    """
    Generates a formal spatial feature engineering proposal for Member 1 (Lina).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    content = r"""# Spatial Feature Engineering Proposal for Supervised Crime Risk Models

**Author:** Aditi Verma (Member 2, Geospatial Data Scientist)  
**Recipient:** Lina Aggarwal (Member 1, Lead ML Engineer)  
**Date:** October 2026  
**Status:** Exploratory / Proposed for Integration  

---

## 1. Executive Summary

To augment Lina's core predictive models (classification and crime count forecasting), this document defines four engineered spatial features derived from historical incident points and DBSCAN clustering outputs. 

These features capture spatial concentration, proximity to chronic high-crime corridors, and local geographic variance without causing target leakage.

---

## 2. Engineered Spatial Feature Definitions

| Feature Name | Definition | Metric / Units | Intended Model Use |
| :--- | :--- | :--- | :--- |
| `dist_to_nearest_hotspot_km` | Great-circle Haversine distance from prediction location $(lat, lon)$ to the nearest DBSCAN hotspot centroid | Kilometers ($km$) | Continuous predictor for classification & regression. Measures spatial exposure to chronic crime nodes. |
| `spatial_density_500m_30d` | Count of recorded historical crime events within a 500-meter radius over the preceding 30 days | Integer count | Captures localized micro-level spatial momentum. |
| `community_area_crime_rate` | Normalized historical incident proportion for the assigned Chicago Community Area | Unitless probability $[0, 1]$ | Static spatial prior representing neighborhood-level baseline volume. |
| `is_historical_hotspot_member` | Binary indicator ($1/0$) designating whether the location resides within the bounding envelope of a verified DBSCAN cluster | Binary flag | Categorical flag for high-density spatial zones. |

---

## 3. Critical Data Leakage Prevention Protocol

```
Timeline:
[================= Historical Window =================) | [==== Prediction Period ====]
(T - 30 days)                                         T  (T + 1 day / week)
                      ^                               |
                      |                               |
       Features calculated ONLY from events           Prediction targets evaluated
       where event_time < T                           ONLY where event_time >= T
```

### Protocol Requirements:
1. **Strict Temporal Partitioning:** For any prediction made for interval $[T, T + \Delta t]$, all spatial density metrics and rolling event counts must be calculated strictly from events where $\text{Date} < T$.
2. **Exclusion of Future Events:** Never compute spatial density across the entire static dataset prior to splitting.
3. **Quarantine of Post-Event Fields:** Under no circumstances should incident-level attributes (`Arrest`, `IUCR`, `FBI Code`, `Description`, `Updated On`) be used as spatial predictors, as they represent post-investigation police outcomes that are unknown at prediction time.
4. **Coordination with Lina:** These features are prepared in `src/feature_engineering.py`. Lina may evaluate their feature importance and SHAP values prior to final pipeline integration.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[FeatureProposal] Saved feature proposal to: {output_path}")


if __name__ == "__main__":
    generate_feature_proposal_document()
    print("[FeatureEngineering] Completed successfully.")
