"""
Reproducible Artifact Serialization Pipeline
AI-Powered Urban Crime Risk & Hotspot Prediction System

Generates persistent, production-ready artifacts for the unified inference service:
1. models/random_forest_champion.joblib   - Champion RF Regressor (compressed joblib)
2. models/risk_thresholds.json             - Frozen four-tier risk classification quartiles
3. models/model_metadata.json              - Full model schema, environment versions, and metrics
4. data/processed/hotspot_clusters.parquet - 617 localized DBSCAN hotspot clusters with multi-area mappings
5. data/processed/kde_density_grid.npz     - Macroscopic 2D KDE continuous crime density surface (50x50)

Usage:
    python scripts/serialize_artifacts.py
"""

import os
import sys
import json
import platform
import argparse
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy

# Guarantee project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.random_forest_model import build_random_forest_model, get_feature_columns
from src.features import compute_training_risk_thresholds, FOUR_TIER_RISK_CATEGORIES
from src.hotspots import filter_valid_coordinates, detect_hotspots_dbscan, summarize_hotspots, compute_kde_density


def serialize_champion_model(
    train_parquet_path: str,
    output_model_path: str,
    compression_level: int = 3
) -> Dict[str, Any]:
    """
    Trains the champion Random Forest model strictly on the 2020-2023 training partition
    and serializes it using joblib with specified compression.
    """
    if not os.path.exists(train_parquet_path):
        raise FileNotFoundError(
            f"Training dataset not found at '{train_parquet_path}'. "
            f"Please ensure Milestone 2 preprocessing pipeline has been executed."
        )

    print(f"Loading training data from {train_parquet_path}...")
    train_df = pd.read_parquet(train_parquet_path)

    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)

    rf_params = {
        'n_estimators': 150,
        'max_depth': 12,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }

    print("Fitting champion RandomForestRegressor (150 trees, max_depth=12)...")
    rf_model = build_random_forest_model(**rf_params)
    rf_model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(rf_model, output_model_path, compress=compression_level)
    file_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"Champion model successfully saved to '{output_model_path}' ({file_size_mb:.2f} MB)")

    return {
        'model': rf_model,
        'params': rf_params,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'train_rows': len(train_df),
        'file_size_mb': file_size_mb
    }


def serialize_risk_thresholds(
    train_parquet_path: str,
    output_thresholds_path: str
) -> Dict[str, float]:
    """
    Derives and saves the frozen four-tier risk thresholds strictly from the training partition.
    Validation and test splits never influence these cutoffs.
    """
    train_df = pd.read_parquet(train_parquet_path)
    thresholds = compute_training_risk_thresholds(train_df)

    threshold_payload = {
        "low_threshold_q25": float(thresholds['low_threshold_q25']),
        "median_threshold_q50": float(thresholds['median_threshold_q50']),
        "high_threshold_q75": float(thresholds['high_threshold_q75']),
        "categories": FOUR_TIER_RISK_CATEGORIES,
        "boundary_rules": {
            "Low": "value <= low_threshold_q25",
            "Moderate": "low_threshold_q25 < value <= median_threshold_q50",
            "High": "median_threshold_q50 < value <= high_threshold_q75",
            "Very High": "value > high_threshold_q75"
        },
        "derivation_split": "2020-2023 Training Split (train_features.parquet)",
        "training_observations": int(len(train_df)),
        "derivation_date_range": ["2020-01-31", "2023-12-31"]
    }

    os.makedirs(os.path.dirname(output_thresholds_path), exist_ok=True)
    with open(output_thresholds_path, "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)

    print(f"Risk thresholds successfully saved to '{output_thresholds_path}'")
    return thresholds


def serialize_model_metadata(
    output_metadata_path: str,
    rf_params: Dict[str, Any],
    feature_cols: list,
    target_col: str,
    train_rows: int
) -> None:
    """
    Generates and saves the definitive model metadata JSON recording exact features,
    environment specifications, and validation/test performance.
    """
    metadata_payload = {
        "model_name": "Random Forest Regressor (Champion)",
        "model_type": "sklearn.ensemble.RandomForestRegressor",
        "hyperparameters": rf_params,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "target_variable": target_col,
        "training_environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform()
        },
        "training_dataset": {
            "filepath": "data/processed/train_features.parquet",
            "date_range": ["2020-01-31", "2023-12-31"],
            "observation_count": train_rows,
            "unique_community_areas": 77
        },
        "performance_metrics": {
            "validation_2024": {
                "dataset": "val_features.parquet",
                "observations": 28182,
                "mae": 2.5140,
                "rmse": 3.4717,
                "source_report": "reports/random_forest_evaluation.md"
            },
            "held_out_test_2025": {
                "dataset": "test_features.parquet",
                "observations": 28105,
                "mae": 2.4167,
                "rmse": 3.3365,
                "r2": 0.8081,
                "wape_pct": 28.58,
                "negative_predictions": 0,
                "classification_accuracy": 0.6141,
                "classification_macro_f1": 0.6190,
                "classification_weighted_f1": 0.6209,
                "source_report": "reports/final_test_evaluation.md & reports/risk_classification_evaluation.md"
            }
        },
        "serialization": {
            "model_file": "models/random_forest_champion.joblib",
            "format": "joblib",
            "compression": "zlib level 3"
        }
    }

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    print(f"Model metadata successfully saved to '{output_metadata_path}'")


def serialize_hotspots_and_kde(
    raw_csv_path: str,
    output_hotspots_path: str,
    output_kde_path: str,
    target_year: int = 2024,
    eps_meters: float = 100.0,
    min_samples: int = 50,
    kde_bandwidth_meters: float = 500.0,
    kde_grid_size: int = 50
) -> None:
    """
    Executes coordinate validation, localized DBSCAN clustering, multi-area breakdown extraction,
    and macroscopic KDE surface calculation, saving outputs to disk.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(
            f"Raw crime dataset not found at '{raw_csv_path}'. "
            f"Please download the Chicago crime incidents dataset to 'data/'."
        )

    print(f"Loading raw incident records from '{raw_csv_path}'...")
    usecols = ['ID', 'Date', 'Primary Type', 'Community Area', 'Latitude', 'Longitude']
    df_raw = pd.read_csv(raw_csv_path, usecols=usecols, low_memory=False)

    print("Filtering valid coordinates within Chicago bounding box...")
    valid_df, coord_audit = filter_valid_coordinates(df_raw)
    valid_df['parsed_date'] = pd.to_datetime(valid_df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    df_year = valid_df[valid_df['parsed_date'].dt.year == target_year].copy()

    total_year_incidents = len(df_year)
    print(f"Verified {target_year} incidents with coordinates: {total_year_incidents:,}")

    # 1. Run DBSCAN clustering
    print(f"Executing DBSCAN (eps={eps_meters}m, min_samples={min_samples}, metric=haversine)...")
    labels, _, run_stats = detect_hotspots_dbscan(df_year, eps_meters=eps_meters, min_samples=min_samples)
    print(f"DBSCAN detected: {run_stats['n_clusters']} clusters, {run_stats['n_hotspot_incidents']:,} clustered incidents")

    # 2. Summarize clusters
    hotspots_summary_df, summary_stats = summarize_hotspots(df_year, labels, top_n=None)

    # 3. Add cross-boundary multi-area breakdown
    print("Computing multi-area cross-boundary mappings for each cluster...")
    df_labeled = df_year.copy()
    df_labeled['hotspot_id'] = labels
    clustered_pts = df_labeled[df_labeled['hotspot_id'] != -1]

    area_breakdowns = []
    all_areas_list = []
    for h_id in hotspots_summary_df['hotspot_id']:
        c_pts = clustered_pts[clustered_pts['hotspot_id'] == h_id]
        areas_series = c_pts['Community Area'].dropna().astype(int).value_counts()
        area_dict = {str(k): int(v) for k, v in areas_series.items()}
        area_breakdowns.append(json.dumps(area_dict))
        all_areas_list.append(",".join(map(str, sorted([int(k) for k in area_dict.keys()]))))

    hotspots_summary_df['all_community_areas'] = all_areas_list
    hotspots_summary_df['area_incident_breakdown'] = area_breakdowns

    os.makedirs(os.path.dirname(output_hotspots_path), exist_ok=True)
    hotspots_summary_df.to_parquet(output_hotspots_path, index=False)
    print(f"Hotspot clusters successfully saved to '{output_hotspots_path}' ({len(hotspots_summary_df)} clusters)")

    # 4. Continuous KDE Density Surface
    print(f"Computing continuous KDE density surface (bandwidth={kde_bandwidth_meters}m, grid={kde_grid_size}x{kde_grid_size})...")
    lat_grid, lon_grid, density_grid, _ = compute_kde_density(
        df_year,
        bandwidth_meters=kde_bandwidth_meters,
        grid_size=kde_grid_size,
        sample_size=50000,
        random_state=42
    )

    os.makedirs(os.path.dirname(output_kde_path), exist_ok=True)
    np.savez_compressed(
        output_kde_path,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        density_grid=density_grid,
        bandwidth_meters=kde_bandwidth_meters,
        grid_size=kde_grid_size,
        target_year=target_year
    )
    print(f"KDE density grid successfully saved to '{output_kde_path}' (shape: {density_grid.shape})")


def main():
    parser = argparse.ArgumentParser(description="Serialize AI Crime Risk & Hotspot Prediction artifacts.")
    parser.add_argument("--train-data", default=os.path.join(PROJECT_ROOT, "data", "processed", "train_features.parquet"))
    parser.add_argument("--raw-csv", default=os.path.join(PROJECT_ROOT, "data", "Crimes_-_2001_to_Present_20260820.csv"))
    parser.add_argument("--model-out", default=os.path.join(PROJECT_ROOT, "models", "random_forest_champion.joblib"))
    parser.add_argument("--thresholds-out", default=os.path.join(PROJECT_ROOT, "models", "risk_thresholds.json"))
    parser.add_argument("--metadata-out", default=os.path.join(PROJECT_ROOT, "models", "model_metadata.json"))
    parser.add_argument("--hotspots-out", default=os.path.join(PROJECT_ROOT, "data", "processed", "hotspot_clusters.parquet"))
    parser.add_argument("--kde-out", default=os.path.join(PROJECT_ROOT, "data", "processed", "kde_density_grid.npz"))
    args = parser.parse_args()

    print("=== STARTING STAGE 1 REPRODUCIBLE ARTIFACT SERIALIZATION ===")
    
    # Task 1 & 2
    model_info = serialize_champion_model(args.train_data, args.model_out, compression_level=3)
    serialize_risk_thresholds(args.train_data, args.thresholds_out)
    
    # Task 3
    serialize_model_metadata(
        args.metadata_out,
        rf_params=model_info['params'],
        feature_cols=model_info['feature_cols'],
        target_col=model_info['target_col'],
        train_rows=model_info['train_rows']
    )
    
    # Task 4
    serialize_hotspots_and_kde(
        raw_csv_path=args.raw_csv,
        output_hotspots_path=args.hotspots_out,
        output_kde_path=args.kde_out,
        target_year=2024,
        eps_meters=100.0,
        min_samples=50,
        kde_bandwidth_meters=500.0,
        kde_grid_size=50
    )
    
    print("\n=== ALL STAGE 1 ARTIFACTS SUCCESSFULLY PRODUCED ===")


if __name__ == "__main__":
    main()
