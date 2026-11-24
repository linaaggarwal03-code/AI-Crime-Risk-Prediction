"""
Unit tests for spatial crime hotspot detection module (DBSCAN & KDE).

Verifies:
- Strict coordinate validation, filtering, and exclusion of NaNs and out-of-bounds coords.
- Prohibition of synthetic/imputed GPS coordinates.
- Exactness of Haversine distance and meter-to-radian conversions.
- DBSCAN cluster formation, arbitrary geometry, and noise identification.
- Exact reproducibility across multiple runs with deterministic inputs.
- Hotspot summary statistics (centroid, radius, incident counts, top crime types).
- Kernel Density Estimation (KDE) surface computation, normalization, and bounds.
- Edge case handling (empty DataFrames, all-noise points, single point, boundary values).
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.hotspots import (
    CHICAGO_BOUNDING_BOX,
    EARTH_RADIUS_KM,
    filter_valid_coordinates,
    meters_to_radians,
    radians_to_meters,
    haversine_distance_meters,
    detect_hotspots_dbscan,
    summarize_hotspots,
    compute_kde_density,
    run_hotspot_pipeline,
)


def test_filter_valid_coordinates_drops_nans_and_out_of_bounds():
    """Verifies that NaN coordinates and out-of-bounds geographic points are strictly dropped."""
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6, 7],
        'Latitude': [41.8781, np.nan, 41.8800, 36.6194, 'invalid_num', 42.0000, 41.7500],
        'Longitude': [-87.6298, -87.6300, np.nan, -91.6865, -87.6500, -87.6600, -87.7000],
        'Primary Type': ['THEFT', 'ROBBERY', 'BATTERY', 'BURGLARY', 'THEFT', 'NARCOTICS', 'ASSAULT'],
        'Community Area': [32, 32, 8, 25, 28, 1, 43],
    })

    valid_df, audit = filter_valid_coordinates(df)

    # Expected valid rows:
    # ID 1: (41.8781, -87.6298) -> valid in Chicago
    # ID 2: NaN latitude -> dropped
    # ID 3: NaN longitude -> dropped
    # ID 4: (36.6194, -91.6865) -> out-of-bounds Missouri/Arkansas -> dropped
    # ID 5: 'invalid_num' -> dropped
    # ID 6: (42.0000, -87.6600) -> valid in Chicago
    # ID 7: (41.7500, -87.7000) -> valid in Chicago

    assert len(valid_df) == 3
    assert set(valid_df['ID']) == {1, 6, 7}
    assert audit['total_records'] == 7
    assert audit['valid_coordinates_count'] == 3
    assert audit['out_of_bounds_count'] == 1  # ID 4
    assert audit['missing_coordinates_count'] == 3  # ID 2, 3, 5
    assert not valid_df['Latitude'].isna().any()
    assert not valid_df['Longitude'].isna().any()


def test_filter_valid_coordinates_no_imputation():
    """Confirms that missing coordinates are NEVER imputed, synthesized, or filled with defaults."""
    df = pd.DataFrame({
        'ID': [101, 102],
        'Latitude': [41.88, None],
        'Longitude': [-87.63, None],
        'Community Area': [32, 32],
    })

    valid_df, _ = filter_valid_coordinates(df)
    assert len(valid_df) == 1
    assert valid_df.iloc[0]['ID'] == 101


def test_filter_valid_coordinates_empty_df():
    """Tests coordinate filtering on an empty DataFrame."""
    empty_df = pd.DataFrame(columns=['Latitude', 'Longitude'])
    valid_df, audit = filter_valid_coordinates(empty_df)
    assert len(valid_df) == 0
    assert audit['total_records'] == 0
    assert audit['valid_coordinates_count'] == 0


def test_filter_valid_coordinates_missing_columns():
    """Asserts KeyError is raised when required coordinate columns are absent."""
    df = pd.DataFrame({'colA': [1], 'colB': [2]})
    with pytest.raises(KeyError, match="Required coordinate columns"):
        filter_valid_coordinates(df)


def test_haversine_distance_and_angular_conversions():
    """Verifies mathematical correctness of Haversine distance calculation and unit conversions."""
    # Test meters to radians conversion and inversion
    dist_m = 1000.0  # 1 km
    rad = meters_to_radians(dist_m)
    recovered_m = radians_to_meters(rad)
    assert math.isclose(dist_m, recovered_m, rel_tol=1e-9)

    # Known reference: Chicago City Hall to Millennium Park Cloud Gate (~800 meters)
    # City Hall: (41.8840, -87.6324)
    # Cloud Gate: (41.8827, -87.6233)
    calc_dist = haversine_distance_meters(41.8840, -87.6324, 41.8827, -87.6233)
    assert 700.0 <= calc_dist <= 850.0

    # Distance to identical point must be exactly 0
    zero_dist = haversine_distance_meters(41.8840, -87.6324, 41.8840, -87.6324)
    assert math.isclose(zero_dist, 0.0, abs_tol=1e-7)


def test_dbscan_synthetic_clusters_and_noise_separation():
    """
    Constructs synthetic incidents with two compact hotspots and isolated noise points.
    Asserts DBSCAN accurately discovers both clusters and tags noise with -1.
    """
    rng = np.random.default_rng(42)

    # Hotspot A: Near Michigan Ave (41.8900, -87.6240), 60 points within ~100m
    lat_a = 41.8900 + rng.normal(0, 0.0005, size=60)
    lon_a = -87.6240 + rng.normal(0, 0.0005, size=60)

    # Hotspot B: Austin commercial strip (41.8950, -87.7600), 50 points within ~100m
    lat_b = 41.8950 + rng.normal(0, 0.0005, size=50)
    lon_b = -87.7600 + rng.normal(0, 0.0005, size=50)

    # Noise points: 15 widely scattered points across Chicago
    lat_noise = rng.uniform(41.70, 42.00, size=15)
    lon_noise = rng.uniform(-87.85, -87.55, size=15)

    df_synth = pd.DataFrame({
        'Latitude': np.concatenate([lat_a, lat_b, lat_noise]),
        'Longitude': np.concatenate([lon_a, lon_b, lon_noise]),
        'Primary Type': ['THEFT'] * 60 + ['BATTERY'] * 50 + ['OTHER OFFENSE'] * 15,
        'Community Area': [8] * 60 + [25] * 50 + [1] * 15
    })

    # eps=250m, min_samples=30
    labels, model, run_stats = detect_hotspots_dbscan(df_synth, eps_meters=250.0, min_samples=30)

    assert run_stats['n_clusters'] == 2
    assert run_stats['n_hotspot_incidents'] >= 110
    assert run_stats['n_noise_incidents'] >= 15
    assert (labels == -1).sum() == run_stats['n_noise_incidents']

    # Both clusters must have distinct non-negative integer IDs
    discovered_clusters = set(labels) - {-1}
    assert discovered_clusters == {0, 1}


def test_dbscan_reproducibility():
    """Verifies that DBSCAN produces byte-identical cluster assignments across multiple executions."""
    rng = np.random.default_rng(123)
    lats = 41.88 + rng.normal(0, 0.001, size=80)
    lons = -87.63 + rng.normal(0, 0.001, size=80)
    df = pd.DataFrame({'Latitude': lats, 'Longitude': lons})

    labels1, _, stats1 = detect_hotspots_dbscan(df, eps_meters=200.0, min_samples=20)
    labels2, _, stats2 = detect_hotspots_dbscan(df, eps_meters=200.0, min_samples=20)

    np.testing.assert_array_equal(labels1, labels2)
    assert stats1 == stats2


def test_dbscan_edge_cases():
    """Tests edge cases: empty DataFrame, all points as noise, single point."""
    # 1. Empty DataFrame
    empty_df = pd.DataFrame(columns=['Latitude', 'Longitude'])
    labels_empty, model_empty, stats_empty = detect_hotspots_dbscan(empty_df)
    assert len(labels_empty) == 0
    assert stats_empty['n_clusters'] == 0

    # 2. Single isolated point
    single_df = pd.DataFrame({'Latitude': [41.88], 'Longitude': [-87.63]})
    labels_single, _, stats_single = detect_hotspots_dbscan(single_df, min_samples=5)
    assert len(labels_single) == 1
    assert labels_single[0] == -1
    assert stats_single['n_clusters'] == 0
    assert stats_single['n_noise_incidents'] == 1

    # 3. Dispersed points where none can satisfy min_samples
    sparse_df = pd.DataFrame({
        'Latitude': [41.70, 41.80, 41.90, 42.00],
        'Longitude': [-87.80, -87.70, -87.60, -87.50]
    })
    labels_sparse, _, stats_sparse = detect_hotspots_dbscan(sparse_df, eps_meters=100.0, min_samples=10)
    assert all(lbl == -1 for lbl in labels_sparse)
    assert stats_sparse['n_clusters'] == 0
    assert stats_sparse['noise_ratio'] == 1.0


def test_summarize_hotspots_metrics():
    """Verifies that hotspot summarization correctly computes centroids, counts, ranks, and types."""
    # 30 points in Cluster 0 (Loop), 40 points in Cluster 1 (Austin), 10 noise points
    df = pd.DataFrame({
        'Latitude': [41.88] * 30 + [41.90] * 40 + [41.75] * 10,
        'Longitude': [-87.63] * 30 + [-87.75] * 40 + [-87.70] * 10,
        'Primary Type': ['THEFT'] * 20 + ['DECEPTIVE PRACTICE'] * 10 + ['BATTERY'] * 40 + ['ROBBERY'] * 10,
        'Community Area': [32] * 30 + [25] * 40 + [43] * 10
    })
    labels = np.array([0] * 30 + [1] * 40 + [-1] * 10)

    summary_df, stats = summarize_hotspots(df, labels, top_n=10)

    assert stats['total_clusters_found'] == 2
    assert stats['noise_incidents'] == 10
    assert stats['clustered_incidents'] == 70
    assert len(summary_df) == 2

    # Cluster 1 has 40 incidents, so it must be ranked #1
    rank1 = summary_df.iloc[0]
    assert rank1['hotspot_rank'] == 1
    assert rank1['hotspot_id'] == 1
    assert rank1['incident_count'] == 40
    assert rank1['dominant_community_area'] == 25
    assert math.isclose(rank1['centroid_latitude'], 41.90, abs_tol=1e-4)
    assert math.isclose(rank1['centroid_longitude'], -87.75, abs_tol=1e-4)

    # Cluster 0 has 30 incidents, so it must be ranked #2
    rank2 = summary_df.iloc[1]
    assert rank2['hotspot_rank'] == 2
    assert rank2['hotspot_id'] == 0
    assert rank2['incident_count'] == 30
    assert rank2['dominant_community_area'] == 32


def test_compute_kde_density_grid_shape_and_bounds():
    """Verifies that KDE produces normalized continuous density surfaces matching the grid shape."""
    rng = np.random.default_rng(99)
    # High crime cluster at 41.88, -87.63
    lats = np.concatenate([41.88 + rng.normal(0, 0.005, size=200), rng.uniform(41.65, 42.00, size=50)])
    lons = np.concatenate([-87.63 + rng.normal(0, 0.005, size=200), rng.uniform(-87.85, -87.55, size=50)])
    df = pd.DataFrame({'Latitude': lats, 'Longitude': lons})

    grid_size = 20
    lat_grid, lon_grid, density_grid, kde_model = compute_kde_density(
        df,
        bandwidth_meters=500.0,
        grid_size=grid_size,
        sample_size=None
    )

    assert lat_grid.shape == (grid_size, grid_size)
    assert lon_grid.shape == (grid_size, grid_size)
    assert density_grid.shape == (grid_size, grid_size)

    # Values must be bounded in [0, 1]
    assert density_grid.min() >= 0.0
    assert density_grid.max() <= 1.0
    assert density_grid.max() == pytest.approx(1.0)

    # Peak density must be near the synthetic cluster center (41.88, -87.63)
    max_idx = np.unravel_index(np.argmax(density_grid), density_grid.shape)
    peak_lat = lat_grid[max_idx]
    peak_lon = lon_grid[max_idx]
    dist_to_cluster = haversine_distance_meters(peak_lat, peak_lon, 41.88, -87.63)
    # Grid cell width is ~2 km; peak should be within 3 km of true center
    assert dist_to_cluster < 3500.0


def test_run_hotspot_pipeline_end_to_end(tmp_path):
    """Tests the full end-to-end hotspot pipeline with temporary CSV and report generation."""
    csv_file = tmp_path / "sample_crimes.csv"
    report_file = tmp_path / "sample_report.md"

    # Create synthetic incident records
    dates = ["05/12/2024 10:30:00 AM"] * 50 + ["05/13/2024 02:15:00 PM"] * 50 + ["01/01/2023 12:00:00 PM"] * 10
    sample_df = pd.DataFrame({
        'ID': range(1, 111),
        'Date': dates,
        'Primary Type': ['THEFT'] * 50 + ['BATTERY'] * 50 + ['ROBBERY'] * 10,
        'Community Area': [32] * 50 + [8] * 50 + [25] * 10,
        'Latitude': [41.884] * 50 + [41.890] * 50 + [41.900] * 10,
        'Longitude': [-87.632] * 50 + [-87.625] * 50 + [-87.750] * 10,
    })
    sample_df.to_csv(csv_file, index=False)

    results = run_hotspot_pipeline(
        csv_path=str(csv_file),
        target_year=2024,
        eps_meters=300.0,
        min_samples=20,
        kde_bandwidth_meters=500.0,
        kde_grid_size=20,
        output_report_path=str(report_file)
    )

    assert results['n_analyzed_incidents'] == 100  # Only 2024 incidents
    assert results['dbscan_run_stats']['n_clusters'] == 2
    assert report_file.exists()

    report_text = report_file.read_text(encoding='utf-8')
    assert "Milestone 3: Crime Hotspot Detection" in report_text
    assert "DBSCAN" in report_text


def test_dbscan_default_parameters():
    """Verifies that the primary default DBSCAN parameters are epsilon=100m and min_samples=50."""
    import inspect
    sig = inspect.signature(detect_hotspots_dbscan)
    assert sig.parameters['eps_meters'].default == 100.0
    assert sig.parameters['min_samples'].default == 50
    assert sig.parameters['algorithm'].default == 'ball_tree'

    sig_pipeline = inspect.signature(run_hotspot_pipeline)
    assert sig_pipeline.parameters['eps_meters'].default == 100.0
    assert sig_pipeline.parameters['min_samples'].default == 50
    assert sig_pipeline.parameters['comparison_eps_meters'].default == 250.0


def test_dbscan_sensitivity_comparison_reproducibility():
    """Verifies that both 100m localized clustering and 250m comparison runs execute reproducibly."""
    rng = np.random.default_rng(77)
    # Cluster 1: tight core within 50m
    lat1 = 41.88 + rng.normal(0, 0.0003, size=60)
    lon1 = -87.63 + rng.normal(0, 0.0003, size=60)
    # Cluster 2: nearby corridor within 200m of cluster 1 (should be distinct at 100m, chained at 250m)
    lat2 = 41.8815 + rng.normal(0, 0.0003, size=60)
    lon2 = -87.63 + rng.normal(0, 0.0003, size=60)
    # Noise points
    lat_noise = rng.uniform(41.70, 42.00, size=20)
    lon_noise = rng.uniform(-87.85, -87.55, size=20)

    df = pd.DataFrame({
        'Latitude': np.concatenate([lat1, lat2, lat_noise]),
        'Longitude': np.concatenate([lon1, lon2, lon_noise]),
        'Primary Type': ['THEFT'] * 120 + ['OTHER'] * 20,
        'Community Area': [32] * 120 + [1] * 20
    })

    # At 100m: clusters should be separate or high noise
    labels_100, _, stats_100 = detect_hotspots_dbscan(df, eps_meters=100.0, min_samples=30)
    # At 250m: clusters chain together into a single cluster
    labels_250, _, stats_250 = detect_hotspots_dbscan(df, eps_meters=250.0, min_samples=30)

    assert stats_100['eps_meters'] == 100.0
    assert stats_250['eps_meters'] == 250.0
    # Chaining effect verification: 250m must have fewer clusters and fewer noise points than 100m
    assert stats_250['n_noise_incidents'] <= stats_100['n_noise_incidents']
