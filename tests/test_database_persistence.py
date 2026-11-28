"""
Unit tests for DatabaseManager persistence layer (Milestone 3).
"""

from database.db_manager import DatabaseManager


def test_sqlite_persistence_save_and_retrieve(tmp_path):
    db_file = tmp_path / "test_crime.db"
    db = DatabaseManager(database_url=None, sqlite_path=db_file)

    assert db.backend_type == "SQLite"

    row_id = db.save_prediction_query(
        latitude=41.8781,
        longitude=-87.6298,
        target_timestamp="2026-11-25 14:30",
        risk_tier="High (Proximity to Historic Hotspot < 500m)",
        nearest_cluster_id=0,
        distance_km=0.25,
    )
    assert row_id >= 1

    history = db.get_recent_predictions(limit=5)
    assert len(history) == 1
    assert history[0]["latitude"] == 41.8781
    assert history[0]["nearest_cluster_id"] == 0
    assert history[0]["distance_to_hotspot_km"] == 0.25

    info = db.get_db_info()
    assert info["backend"] == "SQLite"
    assert info["predictions_logged"] == 1


def test_sqlite_non_destructive_migration_preserves_existing_records(tmp_path):
    """
    Verifies that migrating a legacy SQLite predictions table (where query_latitude,
    query_longitude, and model_version were NOT NULL) preserves 100% of existing rows
    and allows inserting ML forecasts with NULL coordinates.
    """
    import sqlite3

    db_file = tmp_path / "legacy_crime.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_latitude REAL NOT NULL,
            query_longitude REAL NOT NULL,
            target_prediction_timestamp TEXT NOT NULL,
            predicted_risk_tier TEXT NOT NULL,
            predicted_risk_score REAL,
            nearest_hotspot_id INTEGER,
            distance_to_hotspot_km REAL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        INSERT INTO predictions (
            id, query_latitude, query_longitude, target_prediction_timestamp,
            predicted_risk_tier, nearest_hotspot_id, distance_to_hotspot_km,
            model_name, model_version
        ) VALUES (7, 41.88, -87.63, '2026-11-25 12:00', 'High', 0, 0.2, 'DBSCAN_Spatial_Proximity', 'v0.3-provisional');
    """)
    conn.commit()
    conn.close()

    # Initialize DatabaseManager on legacy DB -> triggers non-destructive migration
    db = DatabaseManager(database_url=None, sqlite_path=db_file)
    ml_id = db.save_ml_prediction(
        community_area=8,
        prediction_date="2024-06-15",
        predicted_crime_count=18.42,
        risk_tier="Very High",
        model_name="Random Forest Regressor (Champion)",
        model_version=None,
        is_historical=True,
        historical_actual_count=20.0,
    )
    assert ml_id > 7

    all_rows = db.get_recent_predictions(limit=10)
    assert len(all_rows) == 2
    assert all_rows[0]["prediction_type"] == "ml_forecast"
    assert all_rows[0]["community_area"] == 8
    assert all_rows[0]["latitude"] is None
    assert all_rows[0]["longitude"] is None
    assert all_rows[0]["predicted_crime_count"] == 18.42

    legacy_row = all_rows[1]
    assert legacy_row["id"] == 7
    assert legacy_row["prediction_type"] == "spatial_proximity"
    assert legacy_row["latitude"] == 41.88


def test_ml_prediction_filtering_and_validation(tmp_path):
    import pytest

    db_file = tmp_path / "filter_crime.db"
    db = DatabaseManager(database_url=None, sqlite_path=db_file)

    db.save_prediction_query(
        latitude=41.88,
        longitude=-87.63,
        target_timestamp="2026-11-25 14:30",
        risk_tier="High",
        nearest_cluster_id=0,
        distance_km=0.15,
    )
    db.save_ml_prediction(
        community_area=32,
        prediction_date="2024-06-15",
        predicted_crime_count=29.68,
        risk_tier="Very High",
        model_name="Random Forest Regressor (Champion)",
        is_historical=True,
        historical_actual_count=43.0,
    )

    assert len(db.get_recent_predictions(limit=10)) == 2
    ml_only = db.get_recent_predictions(limit=10, prediction_type="ml_forecast")
    assert len(ml_only) == 1
    assert ml_only[0]["community_area"] == 32

    spatial_only = db.get_recent_predictions(limit=10, prediction_type="spatial_proximity")
    assert len(spatial_only) == 1
    assert spatial_only[0]["nearest_cluster_id"] == 0

    with pytest.raises(ValueError, match="community_area"):
        db.save_ml_prediction(
            community_area=0,
            prediction_date="2024-06-15",
            predicted_crime_count=5.0,
            risk_tier="Moderate",
            model_name="Random Forest Regressor (Champion)",
        )

    with pytest.raises(ValueError, match="predicted_crime_count"):
        db.save_ml_prediction(
            community_area=8,
            prediction_date="2024-06-15",
            predicted_crime_count=-1.0,
            risk_tier="Low",
            model_name="Random Forest Regressor (Champion)",
        )
