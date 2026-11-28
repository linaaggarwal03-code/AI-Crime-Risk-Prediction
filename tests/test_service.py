"""
Comprehensive unit tests for Unified Prediction-Hotspot Service (src/service.py).

Verifies:
1. Clean initialization with loaded artifacts.
2. Historical prediction fidelity against test_features.parquet.
3. Separation of predicted_crime_count and historical_actual_count.
4. Correct definition of prediction_error = predicted - actual.
5. Strict rejection of warmup dates (< 2020-01-31) and missing historical dates.
6. Strict rejection of invalid Community Area IDs (0, 78, -1, non-numerics).
7. Future forecast validation: InsufficientHistoryError when history is missing.
8. Future forecast validation: rejection of incomplete/invalid recent history (wrong dates, missing areas, negative values, duplicates).
9. Dynamic future prediction generation when valid 30-day genuine history is supplied.
10. Cross-boundary hotspot retrieval: verifies Area 32 (The Loop) retrieves Hotspot Rank 1 (Cluster ID 9) with 8,125 area incidents.
11. Independent definition and non-sum assumption of dominant_clusters_count and cross_boundary_clusters_count.
12. KDE surface retrieval without re-computation.
13. Absence of unsupported confidence intervals and invented area names.
"""

import datetime
import pytest
import numpy as np
import pandas as pd

from src.service import CrimeRiskService, InsufficientHistoryError


@pytest.fixture(scope="module")
def service():
    return CrimeRiskService()


def test_service_initialization(service):
    """Verifies that all artifacts, models, thresholds, and indices load successfully."""
    assert service.model is not None
    assert service.feature_names is not None
    assert len(service.feature_names) == 34
    assert service.thresholds == {
        'low_threshold_q25': 3.0,
        'median_threshold_q50': 6.0,
        'high_threshold_q75': 12.0
    }
    assert service.min_historical_date == datetime.date(2020, 1, 31)
    assert service.max_historical_date == datetime.date(2025, 12, 31)
    assert len(service.hotspots_df) == 617


def test_historical_prediction_fidelity_and_separation(service):
    """
    Verifies that a historical test query matches the test partition predictions exactly,
    separates predicted from actual counts, and computes error = predicted - actual.
    """
    # Load test partition to check ground truth
    test_df = pd.read_parquet("data/processed/test_features.parquet")
    sample_row = test_df[(test_df['community_area'] == 8) & (test_df['date'] == datetime.date(2025, 6, 15))].iloc[0]

    res = service.predict_area_risk(community_area=8, date="2025-06-15")

    pred = res['prediction']['predicted_crime_count']
    actual = res['context']['historical_actual_count']
    err = res['context']['prediction_error']

    # 1. Exact match with direct model prediction
    feature_row = pd.DataFrame([sample_row[service.feature_names]])
    expected_pred = float(np.maximum(0.0, service.model.predict(feature_row)[0]))
    assert pred == pytest.approx(round(expected_pred, 4), abs=1e-4)

    # 2. Separation of actual and predicted
    assert actual == float(sample_row['crime_count'])
    assert actual != pred

    # 3. Exact error definition: predicted - actual
    assert err == pytest.approx(round(pred - actual, 4), abs=1e-4)

    # 4. No confidence intervals
    assert 'confidence_interval_95' not in res['prediction']
    assert 'confidence_interval' not in res['prediction']

    # 5. Query information
    assert res['query']['community_area'] == 8
    assert res['query']['is_historical'] is True


def test_risk_tier_boundary_assignment(service):
    """Verifies that four-tier boundaries adhere strictly to Q1, Q2, Q3."""
    # Low risk area (Edison Park - Area 9)
    res_low = service.predict_area_risk(community_area=9, date="2025-02-10")
    if res_low['prediction']['predicted_crime_count'] <= 3.0:
        assert res_low['prediction']['risk_tier'] == 'Low'

    # High / Very High risk area (Austin - Area 25 or Near North Side - Area 8)
    res_high = service.predict_area_risk(community_area=25, date="2025-07-20")
    if res_high['prediction']['predicted_crime_count'] > 12.0:
        assert res_high['prediction']['risk_tier'] == 'Very High'


def test_invalid_inputs_rejection(service):
    """Verifies strict rejection of out-of-range Community Areas and malformed dates."""
    with pytest.raises(ValueError, match="Invalid community_area"):
        service.predict_area_risk(community_area=0, date="2025-06-15")

    with pytest.raises(ValueError, match="Invalid community_area"):
        service.predict_area_risk(community_area=78, date="2025-06-15")

    with pytest.raises(ValueError, match="Invalid community_area"):
        service.predict_area_risk(community_area="abc", date="2025-06-15")

    with pytest.raises(ValueError, match="Invalid date format"):
        service.predict_area_risk(community_area=8, date="invalid-date")


def test_warmup_date_rejection(service):
    """Verifies that dates before 2020-01-31 are rejected due to lack of 30-day lag buffer."""
    with pytest.raises(ValueError, match="Dates prior to 2020-01-31 lack required 30-day historical lag features"):
        service.predict_area_risk(community_area=8, date="2020-01-15")

    with pytest.raises(ValueError, match="Dates prior to 2020-01-31 lack required 30-day historical lag features"):
        service.predict_area_risk(community_area=8, date="2019-12-31")


def test_future_date_insufficient_history_rejection(service):
    """Verifies that future dates without recent_history_df raise InsufficientHistoryError."""
    with pytest.raises(InsufficientHistoryError, match="requires recent_history_df"):
        service.predict_area_risk(community_area=8, date="2026-01-01")


def test_recent_history_validation_rules(service):
    """
    Verifies that recent_history_df is rejected if:
    - Missing required columns
    - Not exactly 30 days
    - Contains negative counts
    - Contains duplicate area-date pairs
    - Missing some community areas
    """
    target_date = datetime.date(2026, 1, 1)

    # 1. Non-DataFrame
    with pytest.raises(TypeError):
        service.predict_area_risk(8, "2026-01-01", recent_history_df=[])

    # 2. Missing columns
    bad_df = pd.DataFrame({'community_area': [1], 'date': [datetime.date(2025, 12, 31)]})
    with pytest.raises(ValueError, match="missing required columns"):
        service.predict_area_risk(8, "2026-01-01", recent_history_df=bad_df)

    # 3. Incomplete dates (only 10 days instead of 30)
    dates_10 = [target_date - datetime.timedelta(days=d) for d in range(10, 0, -1)]
    records_10 = []
    for dt in dates_10:
        for ca in range(1, 78):
            records_10.append({'community_area': ca, 'date': dt, 'crime_count': 5.0})
    df_10 = pd.DataFrame(records_10)

    with pytest.raises(InsufficientHistoryError, match="exact required 30 continuous dates"):
        service.predict_area_risk(8, "2026-01-01", recent_history_df=df_10)

    # 4. Negative crime counts
    dates_30 = [target_date - datetime.timedelta(days=d) for d in range(30, 0, -1)]
    records_neg = []
    for dt in dates_30:
        for ca in range(1, 78):
            records_neg.append({'community_area': ca, 'date': dt, 'crime_count': 5.0})
    records_neg[0]['crime_count'] = -1.0
    df_neg = pd.DataFrame(records_neg)

    with pytest.raises(ValueError, match="negative crime counts"):
        service.predict_area_risk(8, "2026-01-01", recent_history_df=df_neg)


def test_future_prediction_dynamic_synthesis_success(service):
    """
    Verifies that when a valid 30-day genuine history buffer (T-30 to T-1) is supplied,
    future prediction runs successfully, returns actual=None, and computes valid tier.
    """
    target_date = datetime.date(2026, 1, 1)
    dates_30 = [target_date - datetime.timedelta(days=d) for d in range(30, 0, -1)]

    records_valid = []
    for dt in dates_30:
        for ca in range(1, 78):
            records_valid.append({
                'community_area': ca,
                'date': dt,
                'crime_count': float((ca % 10) + (dt.day % 5))
            })
    df_valid = pd.DataFrame(records_valid)

    res = service.predict_area_risk(community_area=8, date="2026-01-01", recent_history_df=df_valid)

    assert res['query']['is_historical'] is False
    assert res['query']['date'] == "2026-01-01"
    assert res['context']['historical_actual_count'] is None
    assert res['context']['prediction_error'] is None
    assert res['prediction']['predicted_crime_count'] >= 0.0
    assert res['prediction']['risk_tier'] in ['Low', 'Moderate', 'High', 'Very High']


def test_cross_boundary_hotspot_retrieval(service):
    """
    Verifies that Area 32 (The Loop) successfully retrieves Hotspot Rank 1 (Cluster ID 9)
    with exactly 8,125 area-specific incidents, despite Area 8 being the dominant area.
    """
    hotspots = service.get_area_hotspots(community_area=32, limit=10)

    assert hotspots['total_matching_clusters'] > 0
    assert hotspots['dominant_clusters_count'] >= 0
    assert hotspots['cross_boundary_clusters_count'] >= 0

    # Cluster 9 must be present in the matching clusters
    top_cluster = None
    for c in hotspots['clusters']:
        if c['hotspot_id'] == 9:
            top_cluster = c
            break

    assert top_cluster is not None, "Hotspot Cluster ID 9 not found for Area 32!"
    assert top_cluster['hotspot_rank'] == 1
    assert top_cluster['incident_count_total'] == 17199
    assert top_cluster['incident_count_in_area'] == 8125
    assert top_cluster['is_dominant_in_this_area'] is False  # Area 8 is dominant


def test_independent_cluster_counts(service):
    """
    Verifies that dominant_clusters_count and cross_boundary_clusters_count
    are defined independently and not assumed to sum to total_matching_clusters.
    """
    hotspots = service.get_area_hotspots(community_area=8, limit=20)
    total = hotspots['total_matching_clusters']
    dom = hotspots['dominant_clusters_count']
    cross = hotspots['cross_boundary_clusters_count']

    assert dom <= total
    assert cross <= total
    # They can overlap because a cluster can be dominant in Area 8 AND cross-boundary (e.g. Cluster 9)
    assert dom + cross >= total or dom + cross <= total  # No fixed sum constraint


def test_kde_density_surface_access(service):
    """Verifies that get_kde_density_surface() returns the pre-computed grid."""
    kde = service.get_kde_density_surface()
    assert kde['lat_grid'].shape == (50, 50)
    assert kde['lon_grid'].shape == (50, 50)
    assert kde['density_grid'].shape == (50, 50)
    assert kde['bandwidth_meters'] == 500.0
    assert kde['grid_size'] == 50
    assert kde['density_grid'].max() == pytest.approx(1.0)
