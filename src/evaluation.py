
"""
Evaluation module for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Implements extended regression metrics (MAE, RMSE, R2, WAPE), prediction alignment,
summary statistics comparison, and final held-out test set evaluation.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.features import (
    FOUR_TIER_RISK_CATEGORIES,
    classify_risk_categories,
    compute_training_risk_thresholds,
)
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


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Dict[str, float],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculates four-tier risk classification metrics:
    - Accuracy
    - Macro F1-score
    - Weighted F1-score
    - Macro Precision
    - Macro Recall
    - 4x4 Confusion Matrix

    Both ground-truth counts (y_true) and continuous predictions (y_pred) are
    discretized into ['Low', 'Moderate', 'High', 'Very High'] using identical,
    frozen training-derived thresholds to prevent data leakage.
    """
    if labels is None:
        labels = FOUR_TIER_RISK_CATEGORIES

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have identical length.")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate classification metrics on empty arrays.")

    y_true_cls = classify_risk_categories(y_true_arr, thresholds)
    y_pred_cls = classify_risk_categories(y_pred_arr, thresholds)

    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    macro_f1 = float(f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0))
    macro_prec = float(precision_score(y_true_cls, y_pred_cls, average='macro', zero_division=0))
    macro_rec = float(recall_score(y_true_cls, y_pred_cls, average='macro', zero_division=0))
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)

    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_precision': round(macro_prec, 4),
        'macro_recall': round(macro_rec, 4),
        'confusion_matrix': cm,
        'labels': labels,
        'y_true_categories': y_true_cls,
        'y_pred_categories': y_pred_cls,
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

    # 5. Risk Classification Evaluation (frozen training thresholds)
    thresholds = compute_training_risk_thresholds(train_df)
    cls_rf = calculate_classification_metrics(y_test, preds_rf, thresholds)
    cls_ridge = calculate_classification_metrics(y_test, preds_ridge, thresholds)
    cls_base_day = calculate_classification_metrics(y_test, preds_base_day, thresholds)
    cls_base_week = calculate_classification_metrics(y_test, preds_base_week, thresholds)

    cls_records = [
        {
            'Model': 'Random Forest (Champion)',
            'Accuracy': cls_rf['accuracy'],
            'Macro F1': cls_rf['macro_f1'],
            'Weighted F1': cls_rf['weighted_f1'],
            'Macro Precision': cls_rf['macro_precision'],
            'Macro Recall': cls_rf['macro_recall'],
        },
        {
            'Model': f'Ridge Regression (alpha={ridge_alpha})',
            'Accuracy': cls_ridge['accuracy'],
            'Macro F1': cls_ridge['macro_f1'],
            'Weighted F1': cls_ridge['weighted_f1'],
            'Macro Precision': cls_ridge['macro_precision'],
            'Macro Recall': cls_ridge['macro_recall'],
        },
        {
            'Model': 'Previous-Day Baseline (lag_1)',
            'Accuracy': cls_base_day['accuracy'],
            'Macro F1': cls_base_day['macro_f1'],
            'Weighted F1': cls_base_day['weighted_f1'],
            'Macro Precision': cls_base_day['macro_precision'],
            'Macro Recall': cls_base_day['macro_recall'],
        },
        {
            'Model': 'Previous-Week Baseline (lag_7)',
            'Accuracy': cls_base_week['accuracy'],
            'Macro F1': cls_base_week['macro_f1'],
            'Weighted F1': cls_base_week['weighted_f1'],
            'Macro Precision': cls_base_week['macro_precision'],
            'Macro Recall': cls_base_week['macro_recall'],
        },
    ]
    cls_comparison_table = pd.DataFrame(cls_records)

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
        # Risk Classification Results
        'classification_thresholds': thresholds,
        'classification_rf': cls_rf,
        'classification_ridge': cls_ridge,
        'classification_previous_day': cls_base_day,
        'classification_previous_week': cls_base_week,
        'classification_comparison_table': cls_comparison_table,
    }


def evaluate_all_models_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rf_config: Optional[Dict[str, Any]] = None,
    ridge_alpha: float = 0.01,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Dedicated evaluation of 4-tier risk classification performance across all models:
    - Random Forest (Champion)
    - Ridge Regression (Benchmark)
    - Previous-Day Baseline
    - Previous-Week Baseline

    Uses identical frozen training thresholds to prevent test leakage.
    Returns individual model classification metrics, 4x4 confusion matrices,
    and a structured comparison summary table.
    """
    if thresholds is None:
        thresholds = compute_training_risk_thresholds(train_df)

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

    # 1. Predictions
    preds_day = predict_previous_day(test_df)
    preds_week = predict_previous_week(test_df)

    ridge_pipe = build_ridge_pipeline(alpha=ridge_alpha)
    ridge_pipe.fit(X_train, y_train)
    preds_ridge = np.maximum(0.0, ridge_pipe.predict(X_test))

    rf_model = build_random_forest_model(
        n_estimators=rf_config.get('n_estimators', 150),
        max_depth=rf_config.get('max_depth', 12),
        min_samples_leaf=rf_config.get('min_samples_leaf', 10),
        max_features=rf_config.get('max_features', 'sqrt'),
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    preds_rf = np.maximum(0.0, rf_model.predict(X_test))

    # 2. Compute classification metrics
    cls_rf = calculate_classification_metrics(y_test, preds_rf, thresholds)
    cls_ridge = calculate_classification_metrics(y_test, preds_ridge, thresholds)
    cls_day = calculate_classification_metrics(y_test, preds_day, thresholds)
    cls_week = calculate_classification_metrics(y_test, preds_week, thresholds)

    # 3. Ground truth class distribution
    y_true_cls = cls_rf['y_true_categories']
    gt_series = pd.Series(y_true_cls)
    class_dist_counts = gt_series.value_counts().reindex(FOUR_TIER_RISK_CATEGORIES).to_dict()
    class_dist_pct = (gt_series.value_counts(normalize=True).reindex(FOUR_TIER_RISK_CATEGORIES) * 100.0).round(2).to_dict()

    comparison_df = pd.DataFrame([
        {
            'Model': 'Random Forest (Champion)',
            'Accuracy': cls_rf['accuracy'],
            'Macro F1': cls_rf['macro_f1'],
            'Weighted F1': cls_rf['weighted_f1'],
            'Macro Precision': cls_rf['macro_precision'],
            'Macro Recall': cls_rf['macro_recall'],
        },
        {
            'Model': f'Ridge Regression (alpha={ridge_alpha})',
            'Accuracy': cls_ridge['accuracy'],
            'Macro F1': cls_ridge['macro_f1'],
            'Weighted F1': cls_ridge['weighted_f1'],
            'Macro Precision': cls_ridge['macro_precision'],
            'Macro Recall': cls_ridge['macro_recall'],
        },
        {
            'Model': 'Previous-Day Baseline',
            'Accuracy': cls_day['accuracy'],
            'Macro F1': cls_day['macro_f1'],
            'Weighted F1': cls_day['weighted_f1'],
            'Macro Precision': cls_day['macro_precision'],
            'Macro Recall': cls_day['macro_recall'],
        },
        {
            'Model': 'Previous-Week Baseline',
            'Accuracy': cls_week['accuracy'],
            'Macro F1': cls_week['macro_f1'],
            'Weighted F1': cls_week['weighted_f1'],
            'Macro Precision': cls_week['macro_precision'],
            'Macro Recall': cls_week['macro_recall'],
        },
    ])

    return {
        'thresholds': thresholds,
        'categories': FOUR_TIER_RISK_CATEGORIES,
        'class_distribution_counts': class_dist_counts,
        'class_distribution_pct': class_dist_pct,
        'models': {
            'random_forest': cls_rf,
            'ridge': cls_ridge,
            'previous_day': cls_day,
            'previous_week': cls_week,
        },
        'comparison_table': comparison_df,
    }
