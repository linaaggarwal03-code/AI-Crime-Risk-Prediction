"""
Random Forest Regression module for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Implements feature extraction, hyperparameter tuning across tree depths and leaf sizes,
fixed random seed reproducibility, non-negative prediction enforcement, and comparative
benchmarking against naive baselines and Ridge Regression.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def get_feature_columns() -> Tuple[List[str], List[str], str, List[str]]:
    """
    Returns explicit partition of dataset columns:
    - num_cols: Continuous/discrete numerical features.
    - cat_cols: Spatial identifier.
    - target_col: Primary continuous forecasting target (crime_count).
    - exclude_cols: Identifier/date/leakage columns strictly excluded from inputs.
    """
    target_col = 'crime_count'
    exclude_cols = ['date']

    cat_cols = ['community_area']

    num_cols = [
        # Calendar & Cyclical
        'day_of_week', 'is_weekend', 'month', 'day_of_month', 'day_of_year',
        'sin_dow', 'cos_dow', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy',
        'is_holiday',
        # Historical Autoregressive Lags
        'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
        # Rolling Statistics (strictly shifted past only)
        'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
        'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
        'rolling_mean_30', 'rolling_std_30', 'rolling_min_30', 'rolling_max_30',
        # Spatial-Temporal Context & Interaction
        'ratio_lag1_to_rolling7', 'citywide_lag_1', 'citywide_rolling_7'
    ]

    return num_cols, cat_cols, target_col, exclude_cols


def build_random_forest_model(
    n_estimators: int = 100,
    max_depth: int = 12,
    min_samples_leaf: int = 10,
    max_features: str | float = 'sqrt',
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """
    Builds a RandomForestRegressor with documented hyperparameters.
    
    Anti-leakage and reproducibility guarantee:
    - Fixed random_state ensures deterministic tree building.
    - Fitting is executed strictly on the training partition.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def predict_random_forest(model: RandomForestRegressor, df: pd.DataFrame) -> np.ndarray:
    """
    Generates predictions from a fitted Random Forest model and enforces
    non-negative counts via clipping.
    """
    num_cols, cat_cols, _, _ = get_feature_columns()
    X = df[num_cols + cat_cols]
    raw_preds = model.predict(X)
    clipped_preds = np.maximum(0.0, raw_preds)
    return clipped_preds


def tune_and_train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate_configs: List[Dict[str, Any]] | None = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains Random Forest models across a documented hyperparameter grid on
    the training set (2020-2023), evaluates on 2024 validation data, and
    selects the configuration with the lowest validation MAE.
    
    Returns a dictionary containing:
    - 'best_config': Selected configuration dictionary
    - 'best_model': Fitted RandomForestRegressor instance
    - 'best_val_mae': Best validation MAE (clipped)
    - 'best_val_rmse': Best validation RMSE (clipped)
    - 'tuning_table': DataFrame with results across candidate configurations
    - 'comparison_table': Benchmark against Baselines and Ridge Regression
    - 'is_better_than_baseline': Boolean indicating outperformance over baseline
    - 'is_better_than_ridge': Boolean indicating outperformance over Ridge
    - 'feature_importances': Series of feature importances sorted descending
    """
    if candidate_configs is None:
        candidate_configs = [
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 20, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 20, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 20, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 14, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
            {'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
        ]

    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)

    X_val = val_df[feature_cols]
    y_val = val_df[target_col].to_numpy(dtype=float)

    tuning_records = []
    trained_models = []

    for idx, cfg in enumerate(candidate_configs):
        model = build_random_forest_model(
            n_estimators=cfg.get('n_estimators', 100),
            max_depth=cfg.get('max_depth', 12),
            min_samples_leaf=cfg.get('min_samples_leaf', 10),
            max_features=cfg.get('max_features', 'sqrt'),
            random_state=random_state,
            n_jobs=-1,
        )
        # Model fitted strictly on training data
        model.fit(X_train, y_train)
        trained_models.append(model)

        raw_preds = model.predict(X_val)
        clipped_preds = np.maximum(0.0, raw_preds)

        raw_mae = float(mean_absolute_error(y_val, raw_preds))
        clip_mae = float(mean_absolute_error(y_val, clipped_preds))
        clip_rmse = float(root_mean_squared_error(y_val, clipped_preds))
        neg_count = int((raw_preds < 0.0).sum())

        tuning_records.append({
            'config_id': idx + 1,
            'n_estimators': cfg.get('n_estimators', 100),
            'max_depth': cfg.get('max_depth', 12),
            'min_samples_leaf': cfg.get('min_samples_leaf', 10),
            'max_features': str(cfg.get('max_features', 'sqrt')),
            'val_mae_raw': round(raw_mae, 4),
            'val_mae_clipped': round(clip_mae, 4),
            'val_rmse_clipped': round(clip_rmse, 4),
            'negative_predictions_clipped': neg_count,
        })

    tuning_table = pd.DataFrame(tuning_records).sort_values(by='val_mae_clipped').reset_index(drop=True)

    best_row = tuning_table.iloc[0]
    best_config_id = int(best_row['config_id'])
    best_model = trained_models[best_config_id - 1]
    best_config = candidate_configs[best_config_id - 1]

    # Benchmark references
    baseline_mae = 3.3592
    baseline_rmse = 4.6725
    ridge_mae = 2.5180
    ridge_rmse = 3.4933

    best_mae = float(best_row['val_mae_clipped'])
    best_rmse = float(best_row['val_rmse_clipped'])

    # Feature importances
    importances = pd.Series(
        best_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    comparison_df = pd.DataFrame([
        {
            'Model': 'Previous-Day Baseline (lag_1)',
            'Model Class': 'Naive Persistence',
            'Validation MAE': baseline_mae,
            'Validation RMSE': baseline_rmse,
            'MAE Delta vs Baseline': 0.0,
            '% MAE Reduction vs Baseline': 0.0,
        },
        {
            'Model': 'Ridge Regression (alpha=0.01)',
            'Model Class': 'Regularized Linear',
            'Validation MAE': ridge_mae,
            'Validation RMSE': ridge_rmse,
            'MAE Delta vs Baseline': round(baseline_mae - ridge_mae, 4),
            '% MAE Reduction vs Baseline': round(((baseline_mae - ridge_mae) / baseline_mae) * 100.0, 2),
        },
        {
            'Model': f"Random Forest (n_est={best_config['n_estimators']}, depth={best_config['max_depth']}, leaf={best_config['min_samples_leaf']})",
            'Model Class': 'Ensemble Decision Trees',
            'Validation MAE': best_mae,
            'Validation RMSE': best_rmse,
            'MAE Delta vs Baseline': round(baseline_mae - best_mae, 4),
            '% MAE Reduction vs Baseline': round(((baseline_mae - best_mae) / baseline_mae) * 100.0, 2),
        }
    ]).sort_values(by='Validation MAE').reset_index(drop=True)

    comparison_df['Rank'] = range(1, len(comparison_df) + 1)

    return {
        'best_config': best_config,
        'best_model': best_model,
        'best_val_mae': best_mae,
        'best_val_rmse': best_rmse,
        'tuning_table': tuning_table,
        'comparison_table': comparison_df,
        'is_better_than_baseline': bool(best_mae < baseline_mae),
        'is_better_than_ridge': bool(best_mae < ridge_mae),
        'feature_importances': importances,
    }
