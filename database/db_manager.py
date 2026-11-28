"""
Live Database Persistence Manager (Milestone 3).
Supports PostgreSQL via DATABASE_URL (with psycopg2) and automatically falls back
to SQLite (database/crime_intelligence.db) for zero-configuration local execution.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = BASE_DIR / "database" / "crime_intelligence.db"


class DatabaseManager:
    """
    Unified relational persistence layer for predictions and hotspot metadata.
    """

    def __init__(self, database_url: Optional[str] = None, sqlite_path: Optional[Path] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.sqlite_path = sqlite_path or DEFAULT_SQLITE_PATH
        self.backend_type = "SQLite"
        self._init_backend()

    def _init_backend(self):
        if self.database_url and self.database_url.startswith("postgres"):
            try:
                import psycopg2  # noqa: F401
                conn = psycopg2.connect(self.database_url)
                conn.close()
                self.backend_type = "PostgreSQL"
            except Exception as e:
                print(f"[DatabaseManager] PostgreSQL unavailable ({e}); falling back to SQLite.")
                self.backend_type = "SQLite"
        else:
            self.backend_type = "SQLite"

        self.init_tables()

    def _get_connection(self):
        if self.backend_type == "PostgreSQL":
            import psycopg2
            return psycopg2.connect(self.database_url)
        else:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(str(self.sqlite_path))

    def init_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        if self.backend_type == "PostgreSQL":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    prediction_type VARCHAR(32) NOT NULL DEFAULT 'spatial_proximity',
                    community_area INTEGER,
                    query_latitude DOUBLE PRECISION,
                    query_longitude DOUBLE PRECISION,
                    target_prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    predicted_risk_tier VARCHAR(128) NOT NULL,
                    predicted_risk_score DOUBLE PRECISION,
                    predicted_crime_count DOUBLE PRECISION,
                    is_historical BOOLEAN,
                    historical_actual_count DOUBLE PRECISION,
                    nearest_hotspot_id INTEGER,
                    distance_to_hotspot_km DOUBLE PRECISION,
                    model_name VARCHAR(64) NOT NULL,
                    model_version VARCHAR(32),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Non-destructive schema migration for existing PostgreSQL tables
            pg_migrations = [
                "ALTER TABLE predictions ALTER COLUMN query_latitude DROP NOT NULL;",
                "ALTER TABLE predictions ALTER COLUMN query_longitude DROP NOT NULL;",
                "ALTER TABLE predictions ALTER COLUMN model_version DROP NOT NULL;",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prediction_type VARCHAR(32) NOT NULL DEFAULT 'spatial_proximity';",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS community_area INTEGER;",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS predicted_crime_count DOUBLE PRECISION;",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS is_historical BOOLEAN;",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS historical_actual_count DOUBLE PRECISION;",
            ]
            for stmt in pg_migrations:
                cursor.execute(stmt)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_type TEXT NOT NULL DEFAULT 'spatial_proximity',
                    community_area INTEGER,
                    query_latitude REAL,
                    query_longitude REAL,
                    target_prediction_timestamp TEXT NOT NULL,
                    predicted_risk_tier TEXT NOT NULL,
                    predicted_risk_score REAL,
                    predicted_crime_count REAL,
                    is_historical INTEGER,
                    historical_actual_count REAL,
                    nearest_hotspot_id INTEGER,
                    distance_to_hotspot_km REAL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self._migrate_sqlite_predictions_if_needed(cursor)
        conn.commit()
        conn.close()

    def _migrate_sqlite_predictions_if_needed(self, cursor) -> None:
        """
        Non-destructive SQLite migration: preserves all existing rows and IDs while
        making query_latitude/query_longitude/model_version nullable and adding ML columns.
        """
        cursor.execute("PRAGMA table_info(predictions)")
        cols_info = cursor.fetchall()
        col_map = {row[1]: {"notnull": row[3]} for row in cols_info}

        needs_migration = (
            "prediction_type" not in col_map
            or "community_area" not in col_map
            or "predicted_crime_count" not in col_map
            or "is_historical" not in col_map
            or "historical_actual_count" not in col_map
            or col_map.get("query_latitude", {}).get("notnull", 0) == 1
            or col_map.get("query_longitude", {}).get("notnull", 0) == 1
            or col_map.get("model_version", {}).get("notnull", 0) == 1
        )
        if not needs_migration:
            return

        cursor.execute("""
            CREATE TABLE predictions_migrated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_type TEXT NOT NULL DEFAULT 'spatial_proximity',
                community_area INTEGER,
                query_latitude REAL,
                query_longitude REAL,
                target_prediction_timestamp TEXT NOT NULL,
                predicted_risk_tier TEXT NOT NULL,
                predicted_risk_score REAL,
                predicted_crime_count REAL,
                is_historical INTEGER,
                historical_actual_count REAL,
                nearest_hotspot_id INTEGER,
                distance_to_hotspot_km REAL,
                model_name TEXT NOT NULL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            INSERT INTO predictions_migrated (
                id, prediction_type, community_area, query_latitude, query_longitude,
                target_prediction_timestamp, predicted_risk_tier, predicted_risk_score,
                predicted_crime_count, is_historical, historical_actual_count,
                nearest_hotspot_id, distance_to_hotspot_km, model_name, model_version, created_at
            )
            SELECT
                id, 'spatial_proximity', NULL, query_latitude, query_longitude,
                target_prediction_timestamp, predicted_risk_tier, predicted_risk_score,
                NULL, NULL, NULL,
                nearest_hotspot_id, distance_to_hotspot_km, model_name, model_version, created_at
            FROM predictions;
        """)
        cursor.execute("DROP TABLE predictions;")
        cursor.execute("ALTER TABLE predictions_migrated RENAME TO predictions;")

    def save_prediction_query(
        self,
        latitude: float,
        longitude: float,
        target_timestamp: str,
        risk_tier: str,
        nearest_cluster_id: Optional[int],
        distance_km: Optional[float],
        model_name: str = "DBSCAN_Spatial_Proximity",
        model_version: str = "spatial-proximity-v1",
        risk_score: Optional[float] = None,
    ) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholder = "%s" if self.backend_type == "PostgreSQL" else "?"
        created_at = datetime.now(timezone.utc).isoformat()

        if self.backend_type == "PostgreSQL":
            cursor.execute(
                f"""
                INSERT INTO predictions (
                    prediction_type, query_latitude, query_longitude, target_prediction_timestamp,
                    predicted_risk_tier, predicted_risk_score, nearest_hotspot_id,
                    distance_to_hotspot_km, model_name, model_version, created_at
                ) VALUES ({', '.join([placeholder]*11)}) RETURNING id
                """,
                (
                    "spatial_proximity", latitude, longitude, target_timestamp, risk_tier,
                    risk_score, nearest_cluster_id, distance_km,
                    model_name, model_version, created_at
                ),
            )
            new_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                f"""
                INSERT INTO predictions (
                    prediction_type, query_latitude, query_longitude, target_prediction_timestamp,
                    predicted_risk_tier, predicted_risk_score, nearest_hotspot_id,
                    distance_to_hotspot_km, model_name, model_version, created_at
                ) VALUES ({', '.join([placeholder]*11)})
                """,
                (
                    "spatial_proximity", latitude, longitude, target_timestamp, risk_tier,
                    risk_score, nearest_cluster_id, distance_km,
                    model_name, model_version, created_at
                ),
            )
            new_id = cursor.lastrowid

        conn.commit()
        conn.close()
        return int(new_id)

    def save_ml_prediction(
        self,
        community_area: int,
        prediction_date: str,
        predicted_crime_count: float,
        risk_tier: str,
        model_name: str,
        model_version: Optional[str] = None,
        is_historical: Optional[bool] = None,
        historical_actual_count: Optional[float] = None,
    ) -> int:
        """Persists a supervised Community Area ML forecast without fabricating coordinates."""
        if community_area < 1 or community_area > 77:
            raise ValueError(f"community_area must be between 1 and 77, got {community_area}.")
        if predicted_crime_count < 0.0:
            raise ValueError(f"predicted_crime_count must be >= 0, got {predicted_crime_count}.")

        conn = self._get_connection()
        cursor = conn.cursor()
        placeholder = "%s" if self.backend_type == "PostgreSQL" else "?"
        created_at = datetime.now(timezone.utc).isoformat()
        hist_val = (
            bool(is_historical)
            if self.backend_type == "PostgreSQL"
            else (1 if is_historical else 0) if is_historical is not None else None
        )

        if self.backend_type == "PostgreSQL":
            cursor.execute(
                f"""
                INSERT INTO predictions (
                    prediction_type, community_area, query_latitude, query_longitude,
                    target_prediction_timestamp, predicted_risk_tier, predicted_crime_count,
                    is_historical, historical_actual_count, model_name, model_version, created_at
                ) VALUES ({', '.join([placeholder]*12)}) RETURNING id
                """,
                (
                    "ml_forecast", int(community_area), None, None,
                    str(prediction_date), str(risk_tier), float(predicted_crime_count),
                    hist_val, historical_actual_count, str(model_name), model_version, created_at
                ),
            )
            new_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                f"""
                INSERT INTO predictions (
                    prediction_type, community_area, query_latitude, query_longitude,
                    target_prediction_timestamp, predicted_risk_tier, predicted_crime_count,
                    is_historical, historical_actual_count, model_name, model_version, created_at
                ) VALUES ({', '.join([placeholder]*12)})
                """,
                (
                    "ml_forecast", int(community_area), None, None,
                    str(prediction_date), str(risk_tier), float(predicted_crime_count),
                    hist_val, historical_actual_count, str(model_name), model_version, created_at
                ),
            )
            new_id = cursor.lastrowid

        conn.commit()
        conn.close()
        return int(new_id)

    def get_recent_predictions(
        self,
        limit: int = 20,
        prediction_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholder = "%s" if self.backend_type == "PostgreSQL" else "?"

        where_clause = ""
        params: List[Any] = []
        if prediction_type is not None:
            where_clause = f"WHERE prediction_type = {placeholder}"
            params.append(prediction_type)
        params.append(limit)

        cursor.execute(
            f"""
            SELECT id, prediction_type, community_area, query_latitude, query_longitude,
                   target_prediction_timestamp, predicted_risk_tier, predicted_crime_count,
                   is_historical, historical_actual_count, nearest_hotspot_id,
                   distance_to_hotspot_km, model_name, model_version, created_at
            FROM predictions
            {where_clause}
            ORDER BY id DESC
            LIMIT {placeholder}
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "id": int(r[0]),
                "prediction_type": str(r[1] or "spatial_proximity"),
                "community_area": int(r[2]) if r[2] is not None else None,
                "latitude": float(r[3]) if r[3] is not None else None,
                "longitude": float(r[4]) if r[4] is not None else None,
                "target_timestamp": str(r[5]),
                "risk_tier": str(r[6]),
                "predicted_crime_count": float(r[7]) if r[7] is not None else None,
                "is_historical": bool(r[8]) if r[8] is not None else None,
                "historical_actual_count": float(r[9]) if r[9] is not None else None,
                "nearest_cluster_id": int(r[10]) if r[10] is not None else None,
                "distance_to_hotspot_km": float(r[11]) if r[11] is not None else None,
                "model_name": str(r[12]),
                "model_version": str(r[13]) if r[13] is not None else None,
                "created_at": str(r[14]),
            })
        return results

    def get_db_info(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = int(cursor.fetchone()[0])
        conn.close()
        return {
            "backend": self.backend_type,
            "predictions_logged": pred_count,
        }
