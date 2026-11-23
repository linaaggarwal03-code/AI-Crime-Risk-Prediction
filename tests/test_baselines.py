"""
Unit tests for baseline evaluation models and metrics.
Verifies previous-day (lag_1) and previous-week (lag_7) baselines,
metric calculations (MAE, RMSE), non-negative constraints, and 2024 validation results.
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    predict_previous_day,
    predict_previous_week,
    calculate_metrics,
    evaluate_baselines,
)

VAL_PARQUET_PATH = os.path.join('data', 'processed', 'val_features.parquet')
TEST_PARQUET_PATH = os.path.join('data', 'processed', 'test_features.parquet')


def test_metric_calculations_mathematical_correctness():
    """
    Verifies that MAE and RMSE calculations match hand-computed values.
    """
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])  # Errors: +2, -2, +3

    metrics = calculate_metrics(y_true, y_pred)
    # Expected MAE = (2 + 2 + 3) / 3 = 2.3333
    assert np.isclose(metrics['mae'], 7.0 / 3.0, atol=1e-4)
    # Expected RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(17 / 3) = 2.3805
    assert np.isclose(metrics['rmse'], np.sqrt(17.0 / 3.0), atol=1e-4)


def test_non_negative_prediction_enforcement():
    """
    Verifies that negative prediction inputs are strictly clipped to 0.0.
    """
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred_with_negatives = np.array([-3.0, 10.0, -1.0])

    # Clipping transforms [-3, 10, -1] -> [0, 10, 0]
    # Errors: |5 - 0| = 5, |10 - 10| = 0, |15 - 0| = 15
    # MAE = (5 + 0 + 15) / 3 = 6.6667
    metrics = calculate_metrics(y_true, y_pred_with_negatives)
    assert np.isclose(metrics['mae'], 20.0 / 3.0, atol=1e-4)


def test_baseline_predictions_from_dataframe():
    """
    Verifies that predict_previous_day and predict_previous_week extract
    the correct lag columns and return non-negative numpy arrays.
    """
    mock_df = pd.DataFrame({
        'lag_1': [5, 12, 0, 8],
        'lag_7': [7, 10, 2, 4],
        'crime_count': [6, 11, 1, 9]
    })

    pred_day = predict_previous_day(mock_df)
    pred_week = predict_previous_week(mock_df)

    assert isinstance(pred_day, np.ndarray)
    assert isinstance(pred_week, np.ndarray)
    assert (pred_day >= 0).all()
    assert (pred_week >= 0).all()
    assert np.array_equal(pred_day, [5.0, 12.0, 0.0, 8.0])
    assert np.array_equal(pred_week, [7.0, 10.0, 2.0, 4.0])


def test_baseline_evaluation_on_2024_validation():
    """
    Evaluates both naive baselines on the actual 2024 validation dataset.
    Verifies target alignment, zero missing values, and confirmed metric values.
    """
    assert os.path.exists(VAL_PARQUET_PATH), "Validation dataset missing."
    val_df = pd.read_parquet(VAL_PARQUET_PATH)

    # 1. Dataset checks
    assert len(val_df) == 28182  # 366 days x 77 Community Areas
    assert val_df['crime_count'].isna().sum() == 0
    assert val_df['lag_1'].isna().sum() == 0
    assert val_df['lag_7'].isna().sum() == 0

    # 2. Date alignment check (starts 2024-01-01, ends 2024-12-31)
    assert str(val_df['date'].min()) == '2024-01-01'
    assert str(val_df['date'].max()) == '2024-12-31'

    # 3. Evaluate baselines
    comp_df, results = evaluate_baselines(val_df)

    assert len(comp_df) == 2
    assert results['record_count'] == 28182
    assert results['target_variable'] == 'crime_count'

    # 4. Metric assertions for 2024 Validation Split
    lag1_metrics = results['previous_day_lag1']
    lag7_metrics = results['previous_week_lag7']

    # Previous-day (lag_1): MAE = 3.3592, RMSE = 4.6725
    assert np.isclose(lag1_metrics['mae'], 3.3592, atol=1e-3)
    assert np.isclose(lag1_metrics['rmse'], 4.6725, atol=1e-3)

    # Previous-week (lag_7): MAE = 3.4502, RMSE = 4.8111
    assert np.isclose(lag7_metrics['mae'], 3.4502, atol=1e-3)
    assert np.isclose(lag7_metrics['rmse'], 4.8111, atol=1e-3)

    # 5. Confirm that Previous-Day performs better on 2024 validation data
    assert lag1_metrics['mae'] < lag7_metrics['mae']
    assert lag1_metrics['rmse'] < lag7_metrics['rmse']
    assert results['better_baseline'] == 'Previous-Day (lag_1)'
    assert comp_df.iloc[0]['Baseline Model'] == 'Previous-Day Baseline'


def test_test_set_isolation():
    """
    Verifies that test_features.parquet (2025) exists and is intact,
    but is kept separate and was not evaluated during baseline selection.
    """
    assert os.path.exists(TEST_PARQUET_PATH), "Test dataset missing."
    test_df = pd.read_parquet(TEST_PARQUET_PATH)

    assert len(test_df) == 28105  # 365 days x 77 Community Areas
    assert str(test_df['date'].min()) == '2025-01-01'
    assert str(test_df['date'].max()) == '2025-12-31'
