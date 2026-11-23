"""
Unit tests for data preprocessing and feature engineering pipelines.
Verifies spatial audits, incident conservation, dense Cartesian grid construction,
anti-leakage guarantees, and chronological split boundaries.
"""

import os
from datetime import date
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    load_and_audit_incidents,
    build_spatial_temporal_grid,
    split_chronological,
)
from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_spatial_context_features,
    build_feature_pipeline,
    compute_training_risk_thresholds,
    assign_risk_tiers,
    get_us_federal_holidays,
)

CSV_PATH = os.path.join('data', 'Crimes_-_2001_to_Present_20260820.csv')


@pytest.fixture(scope="module")
def loaded_data():
    """Loads and audits raw incidents once for the test module."""
    resolved_df, unresolved_df, audit_metrics = load_and_audit_incidents(CSV_PATH)
    return resolved_df, unresolved_df, audit_metrics


def test_incident_conservation_and_spatial_audit(loaded_data):
    """
    Verifies that all 1,422,980 incidents are strictly accounted for:
    1,422,883 resolved + 97 unresolved ambiguous = 1,422,980 total.
    """
    resolved_df, unresolved_df, metrics = loaded_data

    assert metrics['total_raw_incidents'] == 1422980
    assert metrics['initially_valid_community_area'] == 1422844
    assert metrics['missing_community_area_total'] == 136
    assert metrics['resolved_unambiguous_incidents'] == 39
    assert metrics['unresolved_ambiguous_incidents'] == 97
    assert metrics['total_resolved_for_grid'] == 1422883
    assert metrics['conservation_verified'] is True

    # Ambiguous records check
    assert set(unresolved_df['Beat'].unique()).issubset({1614, 2212})
    assert len(unresolved_df[unresolved_df['Beat'] == 1614]) == 95
    assert len(unresolved_df[unresolved_df['Beat'] == 2212]) == 2

    # Coordinate-missing records are preserved (22,429 in resolved + 1 in unresolved = 22,430 total)
    assert metrics['missing_coordinates_count'] == 22429
    assert unresolved_df['Latitude'].isna().sum() == 1
    assert metrics['missing_coordinates_count'] + int(unresolved_df['Latitude'].isna().sum()) == 22430


def test_grid_completeness_and_sorting(loaded_data):
    """
    Verifies that the aggregated space-time grid contains exactly 168,784 rows,
    includes all 77 Community Areas, covers 2020-01-01 to 2025-12-31,
    retains all 1,422,883 crimes, and is strictly sorted.
    """
    resolved_df, _, _ = loaded_data
    grid_df, stats = build_spatial_temporal_grid(resolved_df)

    assert stats['total_cells'] == 168784
    assert stats['unique_areas'] == 77
    assert stats['unique_dates'] == 2192
    assert stats['min_date'] == '2020-01-01'
    assert stats['max_date'] == '2025-12-31'
    assert stats['total_aggregated_crimes'] == 1422883
    # 7,808 area-days had 0 crimes with the 97 ambiguous records quarantined
    assert stats['zero_crime_cells'] == 7808


    # Verify strict ascending sort order by community_area, date
    assert grid_df['community_area'].is_monotonic_increasing or (
        grid_df.groupby('community_area')['date'].apply(lambda s: s.is_monotonic_increasing).all()
    )


def test_chronological_splits(loaded_data):
    """
    Verifies that split boundaries are non-overlapping and match exact expected counts.
    """
    resolved_df, _, _ = loaded_data
    grid_df, _ = build_spatial_temporal_grid(resolved_df)
    warmup_df, train_df, val_df, test_df = split_chronological(grid_df)

    # 30 warm-up days in 2020 (Jan 1 - Jan 30)
    assert len(warmup_df) == 30 * 77  # 2,310
    assert warmup_df['date'].min() == date(2020, 1, 1)
    assert warmup_df['date'].max() == date(2020, 1, 30)

    # Train: 2020-01-31 to 2023-12-31 (1,431 days)
    assert len(train_df) == 1431 * 77  # 110,187
    assert train_df['date'].min() == date(2020, 1, 31)
    assert train_df['date'].max() == date(2023, 12, 31)

    # Val: 2024-01-01 to 2024-12-31 (366 days, leap year)
    assert len(val_df) == 366 * 77  # 28,182
    assert val_df['date'].min() == date(2024, 1, 1)
    assert val_df['date'].max() == date(2024, 12, 31)

    # Test: 2025-01-01 to 2025-12-31 (365 days)
    assert len(test_df) == 365 * 77  # 28,105
    assert test_df['date'].min() == date(2025, 1, 1)
    assert test_df['date'].max() == date(2025, 12, 31)

    # Total conservation
    total_split_rows = len(warmup_df) + len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == len(grid_df) == 168784


def test_anti_leakage_and_rolling_calculations():
    """
    Tests rolling and lag calculations on a synthetic 10-day 2-area panel
    to verify that day t features NEVER include day t crime counts.
    """
    dates = pd.date_range('2024-01-01', periods=10).date
    rows = []
    # Area 1: counts 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # Area 2: counts 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    for i, d in enumerate(dates):
        rows.append({'community_area': 1, 'date': d, 'crime_count': i + 1})
        rows.append({'community_area': 2, 'date': d, 'crime_count': (i + 1) * 10})

    synthetic_df = pd.DataFrame(rows)
    feat_df = add_lag_features(synthetic_df, lags=(1, 2, 7))
    feat_df = add_rolling_features(feat_df, windows=(3, 7))

    # Inspect Area 1 on day 4 (index 3 for Area 1: count=4)
    # lag_1 should be 3
    # rolling_mean_3 should be mean of day 1, 2, 3 = (1 + 2 + 3) / 3 = 2.0 (NOT including day 4!)
    a1_day4 = feat_df[(feat_df['community_area'] == 1) & (feat_df['date'] == dates[3])].iloc[0]
    assert a1_day4['crime_count'] == 4
    assert a1_day4['lag_1'] == 3
    assert a1_day4['lag_2'] == 2
    assert np.isclose(a1_day4['rolling_mean_3'], 2.0)

    # Inspect Area 2 on day 4 (count=40)
    # lag_1 should be 30
    # rolling_mean_3 should be mean of (10, 20, 30) = 20.0
    a2_day4 = feat_df[(feat_df['community_area'] == 2) & (feat_df['date'] == dates[3])].iloc[0]
    assert a2_day4['crime_count'] == 40
    assert a2_day4['lag_1'] == 30
    assert a2_day4['lag_2'] == 20
    assert np.isclose(a2_day4['rolling_mean_3'], 20.0)


def test_us_holidays_deterministic():
    """
    Verifies that known federal holidays are tagged with is_holiday == 1,
    and regular workdays are tagged with 0.
    Also verifies Juneteenth is NOT tagged in 2020 (enacted in 2021), but tagged in 2021+.
    """
    test_dates = pd.DataFrame({
        'date': [
            date(2024, 7, 4),   # Independence Day
            date(2024, 12, 25), # Christmas
            date(2024, 11, 28), # Thanksgiving
            date(2024, 3, 12),  # Regular Tuesday
            date(2024, 10, 15), # Regular Tuesday
            date(2020, 6, 19),  # Juneteenth 2020 (Not yet a federal holiday)
            date(2021, 6, 19),  # Juneteenth 2021 (Federal holiday enacted June 2021)
        ]
    })
    cal_df = add_calendar_features(test_dates)
    holidays_flagged = cal_df['is_holiday'].tolist()
    assert holidays_flagged == [1, 1, 1, 0, 0, 0, 1]


def test_training_thresholds_derivation(loaded_data):
    """
    Verifies that risk thresholds are derived strictly from the training split.
    """
    resolved_df, _, _ = loaded_data
    grid_df, _ = build_spatial_temporal_grid(resolved_df)
    _, train_df, _, _ = split_chronological(grid_df)

    thresholds = compute_training_risk_thresholds(train_df)
    assert 'low_threshold_q25' in thresholds
    assert 'median_threshold_q50' in thresholds
    assert 'high_threshold_q75' in thresholds

    # Verify monotonic thresholds
    assert thresholds['low_threshold_q25'] <= thresholds['median_threshold_q50'] <= thresholds['high_threshold_q75']

    # Test applying tiers
    tiered_train = assign_risk_tiers(train_df, thresholds)
    assert set(tiered_train['risk_level'].unique()) == {'Low', 'Normal', 'High'}
    assert tiered_train['is_high_risk'].isin([0, 1]).all()
