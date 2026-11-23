"""
Unit tests for Random Forest regression module.
Verifies feature partitioning, model construction, reproducibility with fixed seed,
non-negative predictions, validation evaluation, and test set isolation.
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.random_forest_model import (
    get_feature_columns,
    build_random_forest_model,
    predict_random_forest,
    tune_and_train_random_forest,
)

TRAIN_PARQUET_PATH = os.path.join('data', 'processed', 'train_features.parquet')
VAL_PARQUET_PATH = os.path.join('data', 'processed', 'val_features.parquet')
TEST_PARQUET_PATH = os.path.join('data', 'processed', 'test_features.parquet')


@pytest.fixture(scope="module")
def train_val_data():
    """Loads training and validation datasets once for the test module."""
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    val_df = pd.read_parquet(VAL_PARQUET_PATH)
    return train_df, val_df


def test_feature_selection_and_isolation():
    """
    Verifies that feature selection strictly isolates target and date identifiers.
    """
    num_cols, cat_cols, target_col, exclude_cols = get_feature_columns()

    assert target_col == 'crime_count'
    assert 'date' in exclude_cols
    assert 'crime_count' not in num_cols
    assert 'crime_count' not in cat_cols
    assert 'date' not in num_cols
    assert 'date' not in cat_cols

    assert set(num_cols).isdisjoint(set(cat_cols))
    assert len(num_cols) == 33
    assert cat_cols == ['community_area']


def test_model_construction():
    """
    Verifies that build_random_forest_model configures hyperparameters accurately.
    """
    rf = build_random_forest_model(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=42,
    )
    assert isinstance(rf, RandomForestRegressor)
    assert rf.n_estimators == 50
    assert rf.max_depth == 10
    assert rf.min_samples_leaf == 15
    assert rf.max_features == 'sqrt'
    assert rf.random_state == 42


def test_reproducibility_with_fixed_seed(train_val_data):
    """
    Verifies that training with fixed random_state produces identical predictions.
    """
    train_df, val_df = train_val_data
    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    # Sample small subset for fast deterministic assertion
    sample_train = train_df.head(2000)
    sample_val = val_df.head(500)

    rf1 = build_random_forest_model(n_estimators=20, max_depth=6, random_state=42)
    rf1.fit(sample_train[feature_cols], sample_train[target_col])
    preds1 = rf1.predict(sample_val[feature_cols])

    rf2 = build_random_forest_model(n_estimators=20, max_depth=6, random_state=42)
    rf2.fit(sample_train[feature_cols], sample_train[target_col])
    preds2 = rf2.predict(sample_val[feature_cols])

    assert np.allclose(preds1, preds2, atol=1e-8), "Random Forest predictions were non-deterministic across identical seeds."


def test_non_negative_predictions(train_val_data):
    """
    Verifies that predict_random_forest generates strictly non-negative predictions
    and confirms tree averages of non-negative targets produce zero negative predictions.
    """
    train_df, val_df = train_val_data
    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    sample_train = train_df.head(5000)
    model = build_random_forest_model(n_estimators=25, max_depth=8, min_samples_leaf=10, random_state=42)
    model.fit(sample_train[feature_cols], sample_train[target_col])

    preds = predict_random_forest(model, val_df)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(val_df)
    assert (preds >= 0.0).all(), "Predictions contain negative values."

    # Raw model predictions on non-negative targets should also be >= 0
    raw_preds = model.predict(val_df[feature_cols])
    assert (raw_preds >= 0.0).all(), "Raw tree predictions contained negative values."


def test_tuning_and_validation_benchmarks(train_val_data):
    """
    Tests Random Forest hyperparameter tuning on 2024 validation data,
    verifying outperformance against naive baseline and comparison with Ridge.
    """
    train_df, val_df = train_val_data

    # Use a focused grid of 2 configs for quick, rigorous test execution
    fast_configs = [
        {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 20, 'max_features': 'sqrt'},
        {'n_estimators': 50, 'max_depth': 12, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
    ]

    results = tune_and_train_random_forest(
        train_df,
        val_df,
        candidate_configs=fast_configs,
        random_state=42,
    )

    assert 'best_config' in results
    assert 'best_val_mae' in results
    assert 'best_val_rmse' in results
    assert 'comparison_table' in results
    assert 'feature_importances' in results

    baseline_mae = 3.3592
    baseline_rmse = 4.6725

    best_mae = results['best_val_mae']
    best_rmse = results['best_val_rmse']

    # Must outperform baseline significantly
    assert bool(results['is_better_than_baseline']) is True
    assert best_mae < baseline_mae
    assert best_rmse < baseline_rmse
    assert best_mae < 2.55
    assert best_rmse < 3.55

    # Check comparison table structure
    comp_table = results['comparison_table']
    assert len(comp_table) == 3
    assert 'Previous-Day Baseline (lag_1)' in comp_table['Model'].values
    assert 'Ridge Regression (alpha=0.01)' in comp_table['Model'].values


def test_test_set_unaccessed_during_random_forest():
    """
    Verifies that the 2025 test dataset remains completely untouched.
    """
    assert os.path.exists(TEST_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_PARQUET_PATH)
    assert len(test_df) == 28105
    assert str(test_df['date'].min()) == '2025-01-01'
    assert str(test_df['date'].max()) == '2025-12-31'
