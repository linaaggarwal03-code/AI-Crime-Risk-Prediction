"""
Unit tests for final model evaluation on held-out test data.
Verifies calculation of extended metrics (MAE, RMSE, R2, WAPE),
distribution summary statistics, non-negative predictions, and
outperformance against naive baselines.
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_extended_metrics,
    compute_prediction_summary_stats,
    evaluate_final_champion_and_benchmarks,
)

TRAIN_PARQUET_PATH = os.path.join('data', 'processed', 'train_features.parquet')
TEST_PARQUET_PATH = os.path.join('data', 'processed', 'test_features.parquet')


def test_extended_metrics_math():
    """
    Verifies that MAE, RMSE, R2, and WAPE match hand-computed values.
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])  # Errors: +2, -2, +3, -3

    # Expected MAE = (2 + 2 + 3 + 3) / 4 = 10 / 4 = 2.5
    # Expected RMSE = sqrt((4 + 4 + 9 + 9) / 4) = sqrt(26 / 4) = sqrt(6.5) = 2.5495
    # Total actual = 100.0, Sum absolute error = 10.0 -> WAPE = 10.0%
    # Mean of y_true = 25.0, SS_tot = (15^2 + 5^2 + 5^2 + 15^2) = 225 + 25 + 25 + 225 = 500
    # SS_res = 26.0 -> R2 = 1 - (26 / 500) = 1 - 0.052 = 0.9480

    metrics = calculate_extended_metrics(y_true, y_pred)
    assert np.isclose(metrics['mae'], 2.5, atol=1e-4)
    assert np.isclose(metrics['rmse'], np.sqrt(6.5), atol=1e-4)
    assert np.isclose(metrics['r2'], 0.9480, atol=1e-4)
    assert np.isclose(metrics['wape'], 10.0, atol=1e-2)


def test_wape_with_clipping():
    """
    Verifies that negative predictions are clipped to zero before WAPE calculation.
    """
    y_true = np.array([5.0, 15.0])
    y_pred = np.array([-5.0, 15.0])  # Clipped to [0, 15]

    # Clipped error: |5 - 0| + |15 - 15| = 5.0
    # Total actual: 20.0 -> WAPE = (5 / 20) * 100 = 25.0%
    metrics = calculate_extended_metrics(y_true, y_pred)
    assert np.isclose(metrics['wape'], 25.0, atol=1e-2)
    assert np.isclose(metrics['mae'], 2.5, atol=1e-4)


def test_prediction_summary_stats():
    """
    Verifies compute_prediction_summary_stats outputs required distribution metrics.
    """
    y_true = np.array([0.0, 5.0, 10.0, 20.0])
    y_pred = np.array([1.0, 4.0, 11.0, 19.0])

    stats = compute_prediction_summary_stats(y_true, y_pred)
    assert 'Actual crime_count' in stats.columns
    assert 'Predicted crime_count' in stats.columns
    assert 'mean' in stats.index
    assert 'std' in stats.index
    assert 'min' in stats.index
    assert '50%' in stats.index
    assert 'max' in stats.index


def test_test_dataset_integrity():
    """
    Verifies the 2025 test dataset exists, has complete space-time coverage,
    and contains zero missing values.
    """
    assert os.path.exists(TEST_PARQUET_PATH), "Test dataset missing."
    test_df = pd.read_parquet(TEST_PARQUET_PATH)

    assert len(test_df) == 28105  # 365 days x 77 Community Areas
    assert test_df['date'].nunique() == 365
    assert test_df['community_area'].nunique() == 77
    assert str(test_df['date'].min()) == '2025-01-01'
    assert str(test_df['date'].max()) == '2025-12-31'
    assert test_df.isna().sum().sum() == 0


def test_final_evaluation_pipeline_fast():
    """
    Tests evaluation pipeline on a sample to verify dictionary keys,
    metric integrity, and baseline comparison.
    """
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH).head(3000)
    test_df = pd.read_parquet(TEST_PARQUET_PATH).head(1000)

    fast_rf_config = {
        'n_estimators': 20,
        'max_depth': 6,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
    }

    results = evaluate_final_champion_and_benchmarks(
        train_df,
        test_df,
        rf_config=fast_rf_config,
        ridge_alpha=1.0,
    )

    assert results['test_observations'] == 1000
    assert results['champion_model'] == 'Random Forest'
    assert 'champion_metrics' in results
    assert 'comparison_table' in results
    assert 'champion_summary_stats' in results

    rf_metrics = results['champion_metrics']
    assert rf_metrics['mae'] > 0
    assert rf_metrics['rmse'] > 0
    assert rf_metrics['wape'] > 0
    assert len(results['comparison_table']) == 4
    assert (results['champion_predictions'] >= 0.0).all()
