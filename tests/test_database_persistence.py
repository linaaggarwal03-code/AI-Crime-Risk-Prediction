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
