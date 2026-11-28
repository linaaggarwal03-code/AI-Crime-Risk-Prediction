"""
Unit tests for Stage 1 Model & Hotspot Artifact Serialization.

Verifies:
1. Integrity and loadability of models/random_forest_champion.joblib
2. Prediction fidelity against freshly fitted in-memory model (zero drift)
3. Schema and feature ordering invariance
4. Integrity and exact numerical cutoffs of models/risk_thresholds.json
5. Accuracy and completeness of models/model_metadata.json
6. Integrity, schema, and coordinate bounds of data/processed/hotspot_clusters.parquet
7. Multi-area mapping columns in hotspot clusters
8. Dimensions, shape, and normalization of data/processed/kde_density_grid.npz
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.random_forest_model import build_random_forest_model, get_feature_columns
from src.features import classify_risk_categories


@pytest.fixture(scope="module")
def model_artifacts():
    model_path = os.path.join("models", "random_forest_champion.joblib")
    thresholds_path = os.path.join("models", "risk_thresholds.json")
    metadata_path = os.path.join("models", "model_metadata.json")

    assert os.path.exists(model_path), f"Missing {model_path}"
    assert os.path.exists(thresholds_path), f"Missing {thresholds_path}"
    assert os.path.exists(metadata_path), f"Missing {metadata_path}"

    model = joblib.load(model_path)
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "model": model,
        "thresholds": thresholds,
        "metadata": metadata
    }


def test_champion_model_structure_and_params(model_artifacts):
    """Verifies that loaded model matches exact champion hyperparameters."""
    model = model_artifacts["model"]
    assert model.n_estimators == 150
    assert model.max_depth == 12
    assert model.min_samples_leaf == 10
    assert model.max_features == 'sqrt'
    assert model.random_state == 42


def test_champion_model_prediction_fidelity(model_artifacts):
    """
    Verifies that the serialized model predictions on held-out test data
    match an in-memory re-trained model bitwise / within numerical precision.
    """
    loaded_model = model_artifacts["model"]

    train_df = pd.read_parquet("data/processed/train_features.parquet")
    test_df = pd.read_parquet("data/processed/test_features.parquet")

    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_test = test_df[feature_cols]

    # Re-train fresh in-memory model
    fresh_model = build_random_forest_model(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    fresh_model.fit(X_train, y_train)

    preds_loaded = np.maximum(0.0, loaded_model.predict(X_test))
    preds_fresh = np.maximum(0.0, fresh_model.predict(X_test))

    np.testing.assert_allclose(preds_loaded, preds_fresh, rtol=1e-7, atol=1e-7)


def test_risk_thresholds_json_content(model_artifacts):
    """Verifies frozen quartiles Q1=3.0, Q2=6.0, Q3=12.0 and category definitions."""
    t = model_artifacts["thresholds"]
    assert t["low_threshold_q25"] == 3.0
    assert t["median_threshold_q50"] == 6.0
    assert t["high_threshold_q75"] == 12.0
    assert t["categories"] == ["Low", "Moderate", "High", "Very High"]
    assert "derivation_split" in t

    # Verify classification function behaves identically with loaded dict
    assert classify_risk_categories(2.5, t) == "Low"
    assert classify_risk_categories(5.0, t) == "Moderate"
    assert classify_risk_categories(9.0, t) == "High"
    assert classify_risk_categories(15.0, t) == "Very High"


def test_model_metadata_json_content(model_artifacts):
    """Verifies metadata schema, exact 34 features, and performance records."""
    m = model_artifacts["metadata"]
    assert m["feature_count"] == 34
    num_cols, cat_cols, _, _ = get_feature_columns()
    assert m["feature_names"] == (num_cols + cat_cols)
    assert m["hyperparameters"]["n_estimators"] == 150
    assert m["performance_metrics"]["held_out_test_2025"]["r2"] == 0.8081
    assert m["performance_metrics"]["held_out_test_2025"]["mae"] == 2.4167
    assert m["training_environment"]["scikit_learn_version"] == "1.9.1"


def test_hotspot_clusters_parquet_artifact():
    """Verifies the pre-computed DBSCAN hotspot clusters parquet table."""
    hotspot_path = os.path.join("data", "processed", "hotspot_clusters.parquet")
    assert os.path.exists(hotspot_path), f"Missing {hotspot_path}"

    df = pd.read_parquet(hotspot_path)
    assert len(df) == 617, f"Expected 617 clusters, got {len(df)}"

    required_cols = [
        'hotspot_rank', 'hotspot_id', 'incident_count', 'pct_of_total_crime',
        'centroid_latitude', 'centroid_longitude', 'median_latitude', 'median_longitude',
        'min_latitude', 'max_latitude', 'min_longitude', 'max_longitude',
        'max_radius_meters', 'dominant_community_area', 'top_crime_types',
        'all_community_areas', 'area_incident_breakdown'
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column: {c}"

    # Verify coordinate bounds strictly within Chicago
    assert df['centroid_latitude'].between(41.60, 42.05).all()
    assert df['centroid_longitude'].between(-87.95, -87.50).all()

    # Verify rank 1 cluster properties
    top1 = df.iloc[0]
    assert top1['hotspot_rank'] == 1
    assert top1['incident_count'] == 17199
    assert top1['dominant_community_area'] == 8
    assert "32" in top1['all_community_areas']  # Confirms cross-boundary coverage


def test_kde_density_grid_npz_artifact():
    """Verifies the pre-computed macroscopic KDE density grid npz file."""
    kde_path = os.path.join("data", "processed", "kde_density_grid.npz")
    assert os.path.exists(kde_path), f"Missing {kde_path}"

    npz = np.load(kde_path)
    assert "lat_grid" in npz
    assert "lon_grid" in npz
    assert "density_grid" in npz

    lat_g = npz["lat_grid"]
    lon_g = npz["lon_grid"]
    dens_g = npz["density_grid"]

    assert lat_g.shape == (50, 50)
    assert lon_g.shape == (50, 50)
    assert dens_g.shape == (50, 50)

    # Normalized in [0, 1]
    assert dens_g.min() >= 0.0
    assert dens_g.max() <= 1.0
    assert dens_g.max() == pytest.approx(1.0)
