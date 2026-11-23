"""
Baseline forecasting models for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Implements non-parametric naive persistence baselines (Previous-Day and Previous-Week)
and evaluation metrics (MAE, RMSE).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def predict_previous_day(df: pd.DataFrame) -> np.ndarray:
    """
    Previous-Day Baseline:
    Predicts crime_count on date t using the previous calendar day's count (lag_1).
    Enforces non-negative predictions.
    """
    if 'lag_1' not in df.columns:
        raise KeyError("Required feature 'lag_1' not found in DataFrame.")
    preds = df['lag_1'].to_numpy(dtype=float)
    return np.maximum(0.0, preds)


def predict_previous_week(df: pd.DataFrame) -> np.ndarray:
    """
    Previous-Week Baseline:
    Predicts crime_count on date t using the same day of the previous week (lag_7).
    Enforces non-negative predictions.
    """
    if 'lag_7' not in df.columns:
        raise KeyError("Required feature 'lag_7' not found in DataFrame.")
    preds = df['lag_7'].to_numpy(dtype=float)
    return np.maximum(0.0, preds)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE and RMSE.
    Enforces that predictions are non-negative.
    """
    y_true_clean = np.asarray(y_true, dtype=float)
    y_pred_clean = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    if len(y_true_clean) != len(y_pred_clean):
        raise ValueError("y_true and y_pred must have identical length.")

    if len(y_true_clean) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
    rmse = float(root_mean_squared_error(y_true_clean, y_pred_clean))

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
    }


def evaluate_baselines(val_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates both naive baselines on the validation dataset.
    Returns:
        summary_table: Comparison DataFrame of baseline metrics.
        results_dict: Detailed metrics and metadata dictionary.
    """
    if 'crime_count' not in val_df.columns:
        raise KeyError("Target column 'crime_count' missing from validation dataset.")

    y_true = val_df['crime_count'].to_numpy(dtype=float)

    # Predictions
    preds_lag1 = predict_previous_day(val_df)
    preds_lag7 = predict_previous_week(val_df)

    # Metrics
    metrics_lag1 = calculate_metrics(y_true, preds_lag1)
    metrics_lag7 = calculate_metrics(y_true, preds_lag7)

    # Determine better baseline on validation split
    better_baseline = (
        'Previous-Day (lag_1)' if metrics_lag1['mae'] < metrics_lag7['mae']
        else 'Previous-Week (lag_7)'
    )

    results_dict = {
        'dataset_split': '2024 Validation Split',
        'record_count': len(val_df),
        'target_variable': 'crime_count',
        'previous_day_lag1': metrics_lag1,
        'previous_week_lag7': metrics_lag7,
        'better_baseline': better_baseline,
    }

    comparison_df = pd.DataFrame([
        {
            'Baseline Model': 'Previous-Day Baseline',
            'Feature Used': 'lag_1',
            'Formula': 'y_hat(t) = y(t-1)',
            'MAE': metrics_lag1['mae'],
            'RMSE': metrics_lag1['rmse'],
            'Validation Rank': 1 if better_baseline == 'Previous-Day (lag_1)' else 2,
        },
        {
            'Baseline Model': 'Previous-Week Baseline',
            'Feature Used': 'lag_7',
            'Formula': 'y_hat(t) = y(t-7)',
            'MAE': metrics_lag7['mae'],
            'RMSE': metrics_lag7['rmse'],
            'Validation Rank': 2 if better_baseline == 'Previous-Day (lag_1)' else 1,
        }
    ]).sort_values(by='Validation Rank').reset_index(drop=True)

    return comparison_df, results_dict
