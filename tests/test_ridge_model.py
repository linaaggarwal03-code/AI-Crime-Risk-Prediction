"""
Unit tests for Ridge Regression pipeline, hyperparameter tuning,
StandardScaler fitting constraints, and baseline comparisons.
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.ridge_model import (
    get_feature_columns,
    build_ridge_pipeline,
    predict_ridge,
    tune_and_train_ridge,
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


def test_feature_columns_partition():
    """
    Verifies that feature partitioning strictly isolates target and date identifiers.
    """
    num_cols, cat_cols, target_col, exclude_cols = get_feature_columns()

    assert target_col == 'crime_count'
    assert 'date' in exclude_cols
    assert 'crime_count' not in num_cols
    assert 'crime_count' not in cat_cols
    assert 'date' not in num_cols
    assert 'date' not in cat_cols

    # No overlapping columns
    assert set(num_cols).isdisjoint(set(cat_cols))
    assert len(num_cols) == 33
    assert cat_cols == ['community_area']


def test_ridge_pipeline_structure():
    """
    Verifies that the scikit-learn pipeline correctly incorporates
    StandardScaler for numericals, OneHotEncoder for community_area, and Ridge.
    """
    pipeline = build_ridge_pipeline(alpha=10.0)
    assert 'preprocessor' in pipeline.named_steps
    assert 'regressor' in pipeline.named_steps
    assert isinstance(pipeline.named_steps['regressor'], Ridge)
    assert pipeline.named_steps['regressor'].alpha == 10.0


def test_scaler_fitted_on_training_data_only(train_val_data):
    """
    Guarantees that StandardScaler parameters are strictly computed
    from the training split (2020-2023) and not contaminated by validation data.
    """
    train_df, _ = train_val_data
    num_cols, _, target_col, _ = get_feature_columns()

    pipeline = build_ridge_pipeline(alpha=1.0)
    pipeline.fit(train_df, train_df[target_col])

    # Extract fitted scaler from ColumnTransformer
    preprocessor = pipeline.named_steps['preprocessor']
    fitted_scaler = preprocessor.named_transformers_['num']

    assert isinstance(fitted_scaler, StandardScaler)
    assert len(fitted_scaler.mean_) == len(num_cols)

    # Compute expected training mean directly
    expected_means = train_df[num_cols].mean().to_numpy()
    assert np.allclose(fitted_scaler.mean_, expected_means, atol=1e-5)


def test_non_negative_predictions_clipping(train_val_data):
    """
    Verifies that predict_ridge() clips any negative predictions to zero.
    """
    train_df, val_df = train_val_data
    num_cols, cat_cols, target_col, _ = get_feature_columns()

    pipeline = build_ridge_pipeline(alpha=0.01)
    pipeline.fit(train_df[num_cols + cat_cols], train_df[target_col])

    preds = predict_ridge(pipeline, val_df)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(val_df)
    assert (preds >= 0.0).all(), "Predictions contain negative values."


def test_ridge_tuning_and_beats_baseline(train_val_data):
    """
    Verifies Ridge tuning across alphas on 2024 validation data
    and proves it significantly outperforms the Previous-Day baseline.
    """
    train_df, val_df = train_val_data
    results = tune_and_train_ridge(train_df, val_df, alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

    assert 'best_alpha' in results
    assert 'tuning_table' in results
    assert len(results['tuning_table']) == 6

    # Previous-day baseline was MAE=3.3592, RMSE=4.6725
    baseline_mae = 3.3592
    baseline_rmse = 4.6725

    best_mae = results['best_val_mae']
    best_rmse = results['best_val_rmse']

    # Ridge must beat baseline by at least 20%
    assert bool(results['is_better_than_baseline']) is True
    assert best_mae < 2.60, f"Expected MAE < 2.60, got {best_mae}"
    assert best_rmse < 3.60, f"Expected RMSE < 3.60, got {best_rmse}"
    assert best_mae < baseline_mae
    assert best_rmse < baseline_rmse


def test_test_set_unaccessed_during_ridge():
    """
    Verifies that the 2025 test dataset remains untouched.
    """
    assert os.path.exists(TEST_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_PARQUET_PATH)
    assert len(test_df) == 28105
    assert str(test_df['date'].min()) == '2025-01-01'
    assert str(test_df['date'].max()) == '2025-12-31'
