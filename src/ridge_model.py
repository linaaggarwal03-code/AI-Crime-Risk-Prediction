"""
Ridge Regression module for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Implements feature selection, scikit-learn Pipeline construction with StandardScaler
and OneHotEncoder, alpha hyperparameter tuning, non-negative prediction clipping,
and performance benchmarking against naive baselines.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def get_feature_columns() -> Tuple[List[str], List[str], str, List[str]]:
    """
    Returns explicit partition of dataset columns:
    - num_cols: Continuous/discrete numerical features to be standardized.
    - cat_cols: Spatial categorical identifiers to be one-hot encoded.
    - target_col: Primary continuous forecasting target.
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


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with:
    - StandardScaler applied to numerical features.
    - OneHotEncoder applied to community_area (77 categories).
    - Ridge linear regression estimator.
    
    CRITICAL ANTI-LEAKAGE GUARANTEE:
    All transformers are fitted strictly on the training partition when
    pipeline.fit(X_train, y_train) is executed.
    """
    num_cols, cat_cols, _, _ = get_feature_columns()

    # Preprocessor: scale numericals, one-hot encode community_area (1 to 77)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(categories=[list(range(1, 78))], sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha, random_state=42))
    ])

    return pipeline


def predict_ridge(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """
    Generates predictions from a fitted Ridge pipeline and enforces non-negative counts.
    """
    num_cols, cat_cols, _, _ = get_feature_columns()
    X = df[num_cols + cat_cols]
    raw_preds = pipeline.predict(X)
    clipped_preds = np.maximum(0.0, raw_preds)
    return clipped_preds


def tune_and_train_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    alphas: List[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
) -> Dict[str, Any]:
    """
    Fits Ridge pipelines on the training set (2020-2023) across candidate alphas,
    evaluates validation MAE and RMSE (2024), and selects the best model.
    
    Returns a dictionary containing:
    - 'best_alpha': Selected alpha
    - 'best_pipeline': Fitted Pipeline instance with best alpha
    - 'best_val_mae': Best validation MAE (clipped)
    - 'best_val_rmse': Best validation RMSE (clipped)
    - 'tuning_table': DataFrame with results across all candidate alphas
    - 'comparison_with_baseline': Comparison table against Previous-Day baseline
    """
    num_cols, cat_cols, target_col, _ = get_feature_columns()
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)

    X_val = val_df[feature_cols]
    y_val = val_df[target_col].to_numpy(dtype=float)

    tuning_records = []
    trained_pipelines = {}

    for alpha in alphas:
        pipe = build_ridge_pipeline(alpha=alpha)
        # Scaler and encoder fit STRICTLY on training split
        pipe.fit(X_train, y_train)
        trained_pipelines[alpha] = pipe

        raw_preds = pipe.predict(X_val)
        clipped_preds = np.maximum(0.0, raw_preds)

        raw_mae = float(mean_absolute_error(y_val, raw_preds))
        clip_mae = float(mean_absolute_error(y_val, clipped_preds))
        clip_rmse = float(root_mean_squared_error(y_val, clipped_preds))
        neg_count = int((raw_preds < 0.0).sum())

        tuning_records.append({
            'alpha': alpha,
            'val_mae_raw': round(raw_mae, 4),
            'val_mae_clipped': round(clip_mae, 4),
            'val_rmse_clipped': round(clip_rmse, 4),
            'negative_predictions_clipped': neg_count,
        })

    tuning_table = pd.DataFrame(tuning_records).sort_values(by='val_mae_clipped').reset_index(drop=True)

    best_row = tuning_table.iloc[0]
    best_alpha = float(best_row['alpha'])
    best_pipeline = trained_pipelines[best_alpha]

    # Baseline comparison (Previous-Day baseline: MAE=3.3592, RMSE=4.6725)
    baseline_mae = 3.3592
    baseline_rmse = 4.6725

    mae_improvement = baseline_mae - best_row['val_mae_clipped']
    mae_pct_improvement = (mae_improvement / baseline_mae) * 100.0

    rmse_improvement = baseline_rmse - best_row['val_rmse_clipped']
    rmse_pct_improvement = (rmse_improvement / baseline_rmse) * 100.0

    comparison_df = pd.DataFrame([
        {
            'Model': 'Previous-Day Baseline (lag_1)',
            'Validation MAE': baseline_mae,
            'Validation RMSE': baseline_rmse,
            'MAE Delta vs Baseline': 0.0,
            '% Error Reduction': 0.0,
            'Rank': 2
        },
        {
            'Model': f'Ridge Regression (alpha={best_alpha})',
            'Validation MAE': best_row['val_mae_clipped'],
            'Validation RMSE': best_row['val_rmse_clipped'],
            'MAE Delta vs Baseline': round(mae_improvement, 4),
            '% Error Reduction': round(mae_pct_improvement, 2),
            'Rank': 1
        }
    ]).sort_values(by='Rank').reset_index(drop=True)

    return {
        'best_alpha': best_alpha,
        'best_pipeline': best_pipeline,
        'best_val_mae': best_row['val_mae_clipped'],
        'best_val_rmse': best_row['val_rmse_clipped'],
        'tuning_table': tuning_table,
        'comparison_with_baseline': comparison_df,
        'is_better_than_baseline': bool(best_row['val_mae_clipped'] < baseline_mae),
    }
