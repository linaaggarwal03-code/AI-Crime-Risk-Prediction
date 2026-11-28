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
                    query_latitude DOUBLE PRECISION NOT NULL,
                    query_longitude DOUBLE PRECISION NOT NULL,
                    target_prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    predicted_risk_tier VARCHAR(128) NOT NULL,
                    predicted_risk_score DOUBLE PRECISION,
                    nearest_hotspot_id INTEGER,
                    distance_to_hotspot_km DOUBLE PRECISION,
                    model_name VARCHAR(64) NOT NULL,
                    model_version VARCHAR(32) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
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
        conn.commit()
        conn.close()

    def save_prediction_query(
        self,
        latitude: float,
        longitude: float,
        target_timestamp: str,
        risk_tier: str,
        nearest_cluster_id: Optional[int],
        distance_km: Optional[float],
        model_name: str = "DBSCAN_Spatial_Proximity",
        model_version: str = "v0.3-provisional",
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
                    query_latitude, query_longitude, target_prediction_timestamp,
                    predicted_risk_tier, predicted_risk_score, nearest_hotspot_id,
                    distance_to_hotspot_km, model_name, model_version, created_at
                ) VALUES ({', '.join([placeholder]*10)}) RETURNING id
                """,
                (
                    latitude, longitude, target_timestamp, risk_tier,
                    risk_score, nearest_cluster_id, distance_km,
                    model_name, model_version, created_at
                ),
            )
            new_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                f"""
                INSERT INTO predictions (
                    query_latitude, query_longitude, target_prediction_timestamp,
                    predicted_risk_tier, predicted_risk_score, nearest_hotspot_id,
                    distance_to_hotspot_km, model_name, model_version, created_at
                ) VALUES ({', '.join([placeholder]*10)})
                """,
                (
                    latitude, longitude, target_timestamp, risk_tier,
                    risk_score, nearest_cluster_id, distance_km,
                    model_name, model_version, created_at
                ),
            )
            new_id = cursor.lastrowid

        conn.commit()
        conn.close()
        return int(new_id)

    def get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholder = "%s" if self.backend_type == "PostgreSQL" else "?"
        cursor.execute(
            f"""
            SELECT id, query_latitude, query_longitude, target_prediction_timestamp,
                   predicted_risk_tier, nearest_hotspot_id, distance_to_hotspot_km,
                   model_name, model_version, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT {placeholder}
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "id": int(r[0]),
                "latitude": float(r[1]),
                "longitude": float(r[2]),
                "target_timestamp": str(r[3]),
                "risk_tier": str(r[4]),
                "nearest_cluster_id": int(r[5]) if r[5] is not None else None,
                "distance_to_hotspot_km": float(r[6]) if r[6] is not None else None,
                "model_name": str(r[7]),
                "model_version": str(r[8]),
                "created_at": str(r[9]),
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
