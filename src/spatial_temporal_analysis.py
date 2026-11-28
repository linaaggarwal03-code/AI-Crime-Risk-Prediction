"""
Spatial-Temporal Crime Analysis Module (Milestone 3).
Computes time-of-day windows, hourly/daily distributions, and daytime vs. nighttime
spatial hotspot shifts from validated Chicago crime records and DBSCAN clusters.
"""

import os
import sys
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def classify_time_window(hour: int) -> str:
    """Maps 24-hour integer (0-23) to 4 interpretable urban time-of-day windows."""
    if 6 <= hour < 12:
        return "Morning (06:00-11:59)"
    elif 12 <= hour < 18:
        return "Afternoon (12:00-17:59)"
    elif 18 <= hour <= 23:
        return "Evening (18:00-23:59)"
    else:
        return "Night (00:00-05:59)"


def classify_day_night(hour: int) -> str:
    """Maps hour to binary Daytime (06:00-17:59) vs. Nighttime (18:00-05:59)."""
    return "Daytime" if 6 <= hour < 18 else "Nighttime"


class SpatialTemporalAnalyzer:
    """
    Analyzes temporal dynamics across DBSCAN spatial clusters and crime categories.
    """

    def __init__(self, hotspot_results_path: str = "outputs/hotspot_results.csv"):
        self.hotspot_results_path = hotspot_results_path
        self.df: Optional[pd.DataFrame] = None

    def load_and_enrich(self) -> pd.DataFrame:
        if not os.path.exists(self.hotspot_results_path):
            raise FileNotFoundError(
                f"Hotspot results not found at '{self.hotspot_results_path}'. "
                "Run src/dbscan_clustering.py first."
            )

        df = pd.read_csv(self.hotspot_results_path, low_memory=False)
        if "Date" not in df.columns:
            raise KeyError("Column 'Date' is required for spatial-temporal analysis.")

        parsed = pd.to_datetime(df["Date"], errors="coerce")
        df = df[parsed.notna()].copy()
        df["parsed_date"] = parsed[parsed.notna()]
        df["hour"] = df["parsed_date"].dt.hour
        df["day_of_week"] = df["parsed_date"].dt.day_name()
        df["time_window"] = df["hour"].apply(classify_time_window)
        df["day_night"] = df["hour"].apply(classify_day_night)

        self.df = df
        return df

    def compute_summary(self) -> Dict[str, Any]:
        if self.df is None:
            self.load_and_enrich()

        df = self.df
        assert df is not None

        # 1. Hourly distribution (0..23)
        hourly_counts = df["hour"].value_counts().reindex(range(24), fill_value=0)
        hourly_data = [
            {"hour": int(h), "label": f"{int(h):02d}:00", "count": int(c)}
            for h, c in hourly_counts.items()
        ]

        # 2. Time-of-day windows
        window_order = [
            "Morning (06:00-11:59)",
            "Afternoon (12:00-17:59)",
            "Evening (18:00-23:59)",
            "Night (00:00-05:59)",
        ]
        win_counts = df["time_window"].value_counts().reindex(window_order, fill_value=0)
        window_data = [
            {
                "window": str(w),
                "count": int(c),
                "percentage": round(int(c) / len(df) * 100, 2) if len(df) > 0 else 0.0,
            }
            for w, c in win_counts.items()
        ]

        # 3. Day-of-week distribution
        day_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        dow_counts = df["day_of_week"].value_counts().reindex(day_order, fill_value=0)
        dow_data = [
            {"day": str(d), "short": str(d)[:3], "count": int(c)}
            for d, c in dow_counts.items()
        ]

        # 4. Daytime vs Nighttime cluster shift (Top 10 non-noise clusters)
        clustered = df[df["cluster_id"] != -1]
        top_cids = clustered["cluster_id"].value_counts().head(10).index.tolist()
        cluster_shifts = []
        for cid in top_cids:
            c_sub = clustered[clustered["cluster_id"] == cid]
            day_cnt = int((c_sub["day_night"] == "Daytime").sum())
            night_cnt = int((c_sub["day_night"] == "Nighttime").sum())
            total_c = day_cnt + night_cnt
            cluster_shifts.append({
                "cluster_id": int(cid),
                "cluster_label": f"Cluster #{int(cid)}",
                "daytime_count": day_cnt,
                "nighttime_count": night_cnt,
                "daytime_pct": round(day_cnt / total_c * 100, 1) if total_c > 0 else 0.0,
                "nighttime_pct": round(night_cnt / total_c * 100, 1) if total_c > 0 else 0.0,
                "dominant_period": "Daytime" if day_cnt >= night_cnt else "Nighttime",
            })

        # 5. Category breakdown with clustered vs noise counts
        cat_col = "Primary Type" if "Primary Type" in df.columns else None
        category_profiles = []
        if cat_col:
            top_cats = df[cat_col].value_counts().head(10).index.tolist()
            for cat in top_cats:
                sub = df[df[cat_col] == cat]
                c_cnt = int((sub["cluster_id"] != -1).sum())
                n_cnt = int((sub["cluster_id"] == -1).sum())
                category_profiles.append({
                    "category": str(cat),
                    "total": int(len(sub)),
                    "clustered": c_cnt,
                    "noise": n_cnt,
                    "hotspot_concentration_pct": round(c_cnt / len(sub) * 100, 1) if len(sub) > 0 else 0.0,
                })

        return {
            "total_records_with_valid_timestamps": int(len(df)),
            "daytime_total": int((df["day_night"] == "Daytime").sum()),
            "nighttime_total": int((df["day_night"] == "Nighttime").sum()),
            "hourly_distribution": hourly_data,
            "time_windows": window_data,
            "day_of_week_distribution": dow_data,
            "cluster_day_night_shifts": cluster_shifts,
            "category_hotspot_profiles": category_profiles,
        }


def run_spatial_temporal_analysis(
    hotspot_results_path: str = "outputs/hotspot_results.csv",
    output_json_path: str = "outputs/spatial_temporal_summary.json",
) -> Dict[str, Any]:
    analyzer = SpatialTemporalAnalyzer(hotspot_results_path)
    analyzer.load_and_enrich()
    summary = analyzer.compute_summary()

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SpatialTemporal] Saved spatial-temporal summary to: {output_json_path}")
    return summary


if __name__ == "__main__":
    run_spatial_temporal_analysis()
