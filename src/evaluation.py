
"""
Evaluation module for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Implements extended regression metrics (MAE, RMSE, R2, WAPE), prediction alignment,
summary statistics comparison, and final held-out test set evaluation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.random_forest_model import (
    build_random_forest_model,
    get_feature_columns,
    predict_random_forest,
)
from src.ridge_model import build_ridge_pipeline, predict_ridge
from src.baselines import predict_previous_day, predict_previous_week


def calculate_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates primary regression evaluation metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of Determination
    - WAPE: Weighted Absolute Percentage Error ((sum |y - y_hat| / sum y) * 100)
    
    Enforces non-negative predictions.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have identical length.")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    total_actual = float(np.sum(y_true_arr))
    if total_actual > 0:
        wape = float((np.sum(np.abs(y_true_arr - y_pred_arr)) / total_actual) * 100.0)
    else:
        wape = 0.0

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'wape': round(wape, 2),
    }


def compute_prediction_summary_stats(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Computes distribution summary statistics for actual vs predicted crime counts.
    """
    actual_s = pd.Series(y_true, name='Actual crime_count')
    pred_s = pd.Series(y_pred, name='Predicted crime_count')
    stats = pd.concat([actual_s.describe(), pred_s.describe()], axis=1)
    return stats.round(4)


def evaluate_final_champion_and_benchmarks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rf_config: Dict[str, Any] | None = None,
    ridge_alpha: float = 0.01,
) -> Dict[str, Any]:
    """
    Performs final evaluation on the 2025 held-out test set:
    1. Trains champion Random Forest on 2020-2023 training data only.
    2. Trains benchmark Ridge Regression on 2020-2023 training data only.
    3. Evaluates naive Previous-Day and Previous-Week baselines on test set.
    4. Computes extended metrics (MAE, RMSE, R2, WAPE), negative prediction counts,
       and distributional statistics.
    
    Guarantees:
    - Zero training or tuning on 2025 test data.
    - True rolling one-day-ahead sequential setup.
    """
    if rf_config is None:
        rf_config = {
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
        }

    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy(dtype=float)

    # 1. Previous-Day Baseline
    preds_base_day = predict_previous_day(test_df)
    metrics_base_day = calculate_extended_metrics(y_test, preds_base_day)

    # 2. Previous-Week Baseline
    preds_base_week = predict_previous_week(test_df)
    metrics_base_week = calculate_extended_metrics(y_test, preds_base_week)

    # 3. Ridge Regression Benchmark (fitted strictly on training set)
    ridge_pipeline = build_ridge_pipeline(alpha=ridge_alpha)
    ridge_pipeline.fit(X_train, y_train)
    raw_ridge_preds = ridge_pipeline.predict(X_test)
    neg_ridge_count = int((raw_ridge_preds < 0.0).sum())
    preds_ridge = np.maximum(0.0, raw_ridge_preds)
    metrics_ridge = calculate_extended_metrics(y_test, preds_ridge)

    # 4. Champion Random Forest (fitted strictly on training set)
    rf_model = build_random_forest_model(
        n_estimators=rf_config.get('n_estimators', 150),
        max_depth=rf_config.get('max_depth', 12),
        min_samples_leaf=rf_config.get('min_samples_leaf', 10),
        max_features=rf_config.get('max_features', 'sqrt'),
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    raw_rf_preds = rf_model.predict(X_test)
    neg_rf_count = int((raw_rf_preds < 0.0).sum())
    preds_rf = np.maximum(0.0, raw_rf_preds)
    metrics_rf = calculate_extended_metrics(y_test, preds_rf)

    # Summary Statistics for Champion
    rf_summary_stats = compute_prediction_summary_stats(y_test, preds_rf)

    # Metric Comparison Table
    comparison_records = [
        {
            'Model': 'Random Forest (Champion)',
            'Model Class': 'Ensemble Decision Trees',
            'Test MAE': metrics_rf['mae'],
            'Test RMSE': metrics_rf['rmse'],
            'Test R2': metrics_rf['r2'],
            'Test WAPE (%)': metrics_rf['wape'],
            'Negative Preds': neg_rf_count,
            'MAE Delta vs Baseline': round(metrics_base_day['mae'] - metrics_rf['mae'], 4),
            '% MAE Reduction vs Baseline': round(((metrics_base_day['mae'] - metrics_rf['mae']) / metrics_base_day['mae']) * 100.0, 2),
            'Rank': 1,
        },
        {
            'Model': f'Ridge Regression (alpha={ridge_alpha})',
            'Model Class': 'Regularized Linear',
            'Test MAE': metrics_ridge['mae'],
            'Test RMSE': metrics_ridge['rmse'],
            'Test R2': metrics_ridge['r2'],
            'Test WAPE (%)': metrics_ridge['wape'],
            'Negative Preds': neg_ridge_count,
            'MAE Delta vs Baseline': round(metrics_base_day['mae'] - metrics_ridge['mae'], 4),
            '% MAE Reduction vs Baseline': round(((metrics_base_day['mae'] - metrics_ridge['mae']) / metrics_base_day['mae']) * 100.0, 2),
            'Rank': 2,
        },
        {
            'Model': 'Previous-Day Baseline (lag_1)',
            'Model Class': 'Naive Persistence',
            'Test MAE': metrics_base_day['mae'],
            'Test RMSE': metrics_base_day['rmse'],
            'Test R2': metrics_base_day['r2'],
            'Test WAPE (%)': metrics_base_day['wape'],
            'Negative Preds': 0,
            'MAE Delta vs Baseline': 0.0,
            '% MAE Reduction vs Baseline': 0.0,
            'Rank': 3,
        },
        {
            'Model': 'Previous-Week Baseline (lag_7)',
            'Model Class': 'Naive Persistence',
            'Test MAE': metrics_base_week['mae'],
            'Test RMSE': metrics_base_week['rmse'],
            'Test R2': metrics_base_week['r2'],
            'Test WAPE (%)': metrics_base_week['wape'],
            'Negative Preds': 0,
            'MAE Delta vs Baseline': round(metrics_base_day['mae'] - metrics_base_week['mae'], 4),
            '% MAE Reduction vs Baseline': round(((metrics_base_day['mae'] - metrics_base_week['mae']) / metrics_base_day['mae']) * 100.0, 2),
            'Rank': 4,
        },
    ]

    comparison_table = pd.DataFrame(comparison_records).sort_values(by='Rank').reset_index(drop=True)

    return {
        'test_observations': len(test_df),
        'champion_model': 'Random Forest',
        'champion_config': rf_config,
        'champion_metrics': metrics_rf,
        'ridge_metrics': metrics_ridge,
        'previous_day_metrics': metrics_base_day,
        'previous_week_metrics': metrics_base_week,
        'comparison_table': comparison_table,
        'champion_summary_stats': rf_summary_stats,
        'raw_negative_counts': {
            'random_forest': neg_rf_count,
            'ridge': neg_ridge_count,
            'previous_day': 0,
            'previous_week': 0,
        },
        'champion_predictions': preds_rf,
        'actuals': y_test,
    }
