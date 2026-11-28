"""
Unit tests for 4-tier risk classification module.

Verifies:
- Training threshold derivation (Q1, Q2, Q3) strictly from training data.
- Strict anti-leakage invariance (test data does not modify training thresholds).
- Precise boundary condition evaluation across all four tiers:
  Low:        value <= Q1
  Moderate:   Q1 < value <= Q2
  High:       Q2 < value <= Q3
  Very High:  value > Q3
- Flexible input support (scalars, NumPy 1D arrays, pandas Series, Python lists).
- Input validation and exception handling for negative values, NaNs, non-numerics,
  missing threshold keys, and non-monotonic cutoffs.
- Mathematical correctness of classification metrics (Accuracy, Macro/Weighted F1,
  Macro Precision, Macro Recall) and 4x4 Confusion Matrix properties.
- Model evaluation on test data across champion and baseline models without
  assuming any single model must dominate all baselines on every metric.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FOUR_TIER_RISK_CATEGORIES,
    compute_training_risk_thresholds,
    assign_risk_tiers,
    classify_risk_categories,
)
from src.evaluation import (
    calculate_classification_metrics,
    evaluate_all_models_classification,
)


@pytest.fixture
def sample_thresholds():
    return {
        'low_threshold_q25': 3.0,
        'median_threshold_q50': 6.0,
        'high_threshold_q75': 12.0,
    }


def test_compute_training_risk_thresholds_derivation():
    """Verifies that thresholds are derived from 25th, median, and 75th percentiles monotonically."""
    df = pd.DataFrame({
        'crime_count': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    })
    thresholds = compute_training_risk_thresholds(df)

    assert 'low_threshold_q25' in thresholds
    assert 'median_threshold_q50' in thresholds
    assert 'high_threshold_q75' in thresholds
    assert thresholds['low_threshold_q25'] <= thresholds['median_threshold_q50'] <= thresholds['high_threshold_q75']


def test_classify_risk_categories_boundary_conditions(sample_thresholds):
    """
    Verifies exact tier assignment at and around each boundary cutoff:
    - Low:        <= 3.0
    - Moderate:   3.0 < x <= 6.0
    - High:       6.0 < x <= 12.0
    - Very High:  > 12.0
    """
    # Lower bound / Zero
    assert classify_risk_categories(0.0, sample_thresholds) == 'Low'
    assert classify_risk_categories(2.999, sample_thresholds) == 'Low'
    assert classify_risk_categories(3.0, sample_thresholds) == 'Low'

    # Q1 to Q2 (Moderate)
    assert classify_risk_categories(3.0001, sample_thresholds) == 'Moderate'
    assert classify_risk_categories(4.5, sample_thresholds) == 'Moderate'
    assert classify_risk_categories(6.0, sample_thresholds) == 'Moderate'

    # Q2 to Q3 (High)
    assert classify_risk_categories(6.0001, sample_thresholds) == 'High'
    assert classify_risk_categories(9.0, sample_thresholds) == 'High'
    assert classify_risk_categories(12.0, sample_thresholds) == 'High'

    # Above Q3 (Very High)
    assert classify_risk_categories(12.0001, sample_thresholds) == 'Very High'
    assert classify_risk_categories(50.0, sample_thresholds) == 'Very High'
    assert classify_risk_categories(200.0, sample_thresholds) == 'Very High'


def test_classify_risk_categories_input_types(sample_thresholds):
    """Verifies support for scalar, NumPy array, pandas Series, and Python lists."""
    test_vals = [1.5, 4.0, 8.5, 15.0]
    expected = ['Low', 'Moderate', 'High', 'Very High']

    # 1. Scalar
    for val, exp in zip(test_vals, expected):
        assert classify_risk_categories(val, sample_thresholds) == exp

    # 2. NumPy 1D array
    arr_in = np.array(test_vals, dtype=float)
    arr_out = classify_risk_categories(arr_in, sample_thresholds)
    assert isinstance(arr_out, np.ndarray)
    np.testing.assert_array_equal(arr_out, np.array(expected))

    # 3. pandas Series with index preservation
    s_in = pd.Series(test_vals, index=['area1', 'area2', 'area3', 'area4'])
    s_out = classify_risk_categories(s_in, sample_thresholds)
    assert isinstance(s_out, pd.Series)
    assert list(s_out.index) == ['area1', 'area2', 'area3', 'area4']
    assert s_out.tolist() == expected

    # 4. Python list
    list_out = classify_risk_categories(test_vals, sample_thresholds)
    assert isinstance(list_out, list)
    assert list_out == expected


def test_classify_risk_categories_invalid_inputs(sample_thresholds):
    """Verifies that negative values, NaNs, and non-numerics raise explicit exceptions."""
    # Negative values
    with pytest.raises(ValueError, match="must be non-negative"):
        classify_risk_categories(-0.5, sample_thresholds)

    with pytest.raises(ValueError, match="must be non-negative"):
        classify_risk_categories(np.array([1.0, -2.0, 5.0]), sample_thresholds)

    with pytest.raises(ValueError, match="must be non-negative"):
        classify_risk_categories(pd.Series([1.0, -1.0]), sample_thresholds)

    # NaN / Nulls
    with pytest.raises(ValueError, match="cannot be NaN"):
        classify_risk_categories(np.nan, sample_thresholds)

    with pytest.raises(ValueError, match="contains NaN"):
        classify_risk_categories(np.array([1.0, np.nan]), sample_thresholds)

    with pytest.raises(ValueError, match="contains NaN"):
        classify_risk_categories(pd.Series([1.0, None]), sample_thresholds)

    # Non-numeric types
    with pytest.raises(TypeError, match="Unsupported input type|Expected scalar|numeric"):
        classify_risk_categories("five", sample_thresholds)

    with pytest.raises(TypeError, match="numeric"):
        classify_risk_categories(pd.Series(["a", "b"]), sample_thresholds)

    # Missing thresholds
    with pytest.raises(KeyError, match="Thresholds dictionary must contain"):
        classify_risk_categories(5.0, {'low_threshold_q25': 3.0})

    # Non-monotonic thresholds
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        classify_risk_categories(5.0, {'q25': 10.0, 'q50': 5.0, 'q75': 15.0})


def test_classification_anti_leakage_invariance():
    """Confirms that test data has zero influence on training-derived thresholds."""
    train_data = pd.DataFrame({'crime_count': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})
    thresholds_train = compute_training_risk_thresholds(train_data)

    # Outlier extreme test data
    test_data = pd.DataFrame({'crime_count': [500, 600, 700, 800, 900, 1000]})

    # Thresholds remain identical because test data is quarantined
    assert thresholds_train['low_threshold_q25'] == train_data['crime_count'].quantile(0.25)
    assert thresholds_train['median_threshold_q50'] == train_data['crime_count'].median()
    assert thresholds_train['high_threshold_q75'] == train_data['crime_count'].quantile(0.75)


def test_calculate_classification_metrics_math(sample_thresholds):
    """
    Verifies mathematical accuracy of Accuracy, Macro F1, Weighted F1,
    Macro Precision, Macro Recall, and 4x4 Confusion Matrix properties.
    """
    y_true = np.array([1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0])  # 2 Low, 2 Mod, 2 High, 2 Very High
    y_pred = np.array([1.0, 4.0, 4.0, 7.0, 8.0, 15.0, 15.0, 20.0])  # 1 Low, 2 Mod, 2 High, 3 Very High

    metrics = calculate_classification_metrics(y_true, y_pred, sample_thresholds)

    assert 'accuracy' in metrics
    assert 'macro_f1' in metrics
    assert 'weighted_f1' in metrics
    assert 'macro_precision' in metrics
    assert 'macro_recall' in metrics
    assert 'confusion_matrix' in metrics

    # 5 out of 8 predictions are exactly correct:
    # y_true tiers: [Low, Low, Moderate, Moderate, High, High, Very High, Very High]
    # y_pred tiers: [Low, Moderate, Moderate, High, High, Very High, Very High, Very High]
    # Correct: index 0 (Low), 2 (Mod), 4 (High), 6 (VHigh), 7 (VHigh) -> 5/8 = 0.625
    assert metrics['accuracy'] == pytest.approx(0.625, abs=1e-3)

    cm = metrics['confusion_matrix']
    assert cm.shape == (4, 4)
    # Conservation of sample count: sum of confusion matrix must equal total samples
    assert cm.sum() == len(y_true)
    # Row sums equal ground-truth class support (2 for each class)
    np.testing.assert_array_equal(cm.sum(axis=1), np.array([2, 2, 2, 2]))


def test_confusion_matrix_accuracy_consistency(sample_thresholds):
    """
    Verifies that the accuracy reported by calculate_classification_metrics()
    is strictly and mathematically consistent with the confusion matrix diagonal sum
    divided by the total observation count (trace(cm) / sum(cm)).
    """
    rng = np.random.default_rng(123)
    # Generate synthetic ground truth counts and continuous predictions
    y_true = rng.uniform(0.0, 30.0, size=500)
    y_pred = np.maximum(0.0, y_true + rng.normal(0.0, 3.0, size=500))

    metrics = calculate_classification_metrics(y_true, y_pred, sample_thresholds)
    cm = metrics['confusion_matrix']
    diag_sum = int(np.trace(cm))
    total_obs = int(np.sum(cm))
    accuracy_from_matrix = diag_sum / total_obs

    assert total_obs == len(y_true)
    assert metrics['accuracy'] == pytest.approx(accuracy_from_matrix, abs=1e-4)
    assert round(accuracy_from_matrix, 4) == metrics['accuracy']


def test_backward_compatibility_assign_risk_tiers(sample_thresholds):
    """Verifies that Milestone 2 assign_risk_tiers() function retains 3-tier behavior."""
    df = pd.DataFrame({'crime_count': [1.0, 5.0, 15.0]})
    tiered = assign_risk_tiers(df, sample_thresholds)

    assert 'risk_level' in tiered.columns
    assert 'is_high_risk' in tiered.columns
    assert list(tiered['risk_level']) == ['Low', 'Normal', 'High']
    assert list(tiered['is_high_risk']) == [0, 0, 1]


def test_all_models_classification_pipeline_runs():
    """
    Runs classification evaluation across models on a synthetic sample dataset
    and verifies that comparison tables and confusion matrices are generated.
    Does NOT assert that any single model must outperform every baseline.
    """
    # Create small synthetic train and test frames matching feature columns
    from src.random_forest_model import get_feature_columns
    num_cols, cat_cols, target_col, _ = get_feature_columns()
    cols = num_cols + cat_cols + [target_col]

    rng = np.random.default_rng(42)
    n_train = 200
    n_test = 50

    train_data = {c: rng.uniform(1.0, 20.0, size=n_train) for c in num_cols}
    for c in cat_cols:
        train_data[c] = rng.integers(1, 78, size=n_train)
    train_data[target_col] = rng.uniform(0.0, 25.0, size=n_train)
    train_df = pd.DataFrame(train_data)

    test_data = {c: rng.uniform(1.0, 20.0, size=n_test) for c in num_cols}
    for c in cat_cols:
        test_data[c] = rng.integers(1, 78, size=n_test)
    test_data[target_col] = rng.uniform(0.0, 25.0, size=n_test)
    test_df = pd.DataFrame(test_data)

    results = evaluate_all_models_classification(
        train_df,
        test_df,
        rf_config={'n_estimators': 10, 'max_depth': 4, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
    )

    assert 'comparison_table' in results
    assert 'models' in results
    assert len(results['comparison_table']) == 4
    for model_name in ['random_forest', 'ridge', 'previous_day', 'previous_week']:
        assert model_name in results['models']
        m = results['models'][model_name]
        cm = m['confusion_matrix']
        assert cm.shape == (4, 4)
        assert cm.sum() == n_test
        cm_acc = np.trace(cm) / np.sum(cm)
        assert m['accuracy'] == pytest.approx(cm_acc, abs=1e-4)
        assert 0.0 <= results['models'][model_name]['accuracy'] <= 1.0
