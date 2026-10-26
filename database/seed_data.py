"""
Database Seeding and Verification Script.
Demonstrates relational insertion and querying of crime incidents, DBSCAN hotspots,
and provisional prediction records. Uses SQLite as a local zero-dependency testing engine
compatible with the PostgreSQL schema specification.
"""

import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "crime_intelligence.db"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"


def init_database(db_path: Path = DB_PATH):
    """
    Initializes the relational schema in SQLite.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS crime_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_record_id TEXT UNIQUE NOT NULL,
        case_number TEXT,
        occurred_at TEXT NOT NULL,
        block_address TEXT,
        crime_type TEXT NOT NULL,
        location_description TEXT,
        arrest_made INTEGER DEFAULT 0,
        domestic_incident INTEGER DEFAULT 0,
        community_area_id INTEGER,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS hotspots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_id INTEGER NOT NULL,
        algorithm_name TEXT DEFAULT 'DBSCAN',
        distance_metric TEXT DEFAULT 'Haversine',
        eps_meters REAL NOT NULL,
        min_samples INTEGER NOT NULL,
        incident_count INTEGER NOT NULL,
        percentage_of_total REAL NOT NULL,
        representative_latitude REAL NOT NULL,
        representative_longitude REAL NOT NULL,
        min_latitude REAL NOT NULL,
        max_latitude REAL NOT NULL,
        min_longitude REAL NOT NULL,
        max_longitude REAL NOT NULL,
        primary_crime_type TEXT,
        dominant_community_area TEXT
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_latitude REAL NOT NULL,
        query_longitude REAL NOT NULL,
        target_prediction_timestamp TEXT NOT NULL,
        predicted_risk_tier TEXT NOT NULL,
        predicted_risk_score REAL,
        nearest_hotspot_id INTEGER REFERENCES hotspots(id),
        distance_to_hotspot_km REAL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print(f"[Database] Initialized relational tables in: {db_path}")


def seed_data(db_path: Path = DB_PATH):
    """
    Seeds hotspots and sample crime records from generated analytical outputs.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Seed Hotspots
    summary_csv = OUTPUTS_DIR / "cluster_summary.csv"
    if summary_csv.exists():
        df_clusters = pd.read_csv(summary_csv)
        cursor.execute("DELETE FROM hotspots")
        for _, r in df_clusters.iterrows():
            cursor.execute("""
                INSERT INTO hotspots (
                    cluster_id, eps_meters, min_samples, incident_count,
                    percentage_of_total, representative_latitude, representative_longitude,
                    min_latitude, max_latitude, min_longitude, max_longitude,
                    primary_crime_type, dominant_community_area
                ) VALUES (?, 500.0, 50, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(r["cluster_id"]),
                int(r["incident_count"]),
                float(r["percentage_of_total"]),
                float(r["centroid_latitude"]),
                float(r["centroid_longitude"]),
                float(r["min_latitude"]),
                float(r["max_latitude"]),
                float(r["min_longitude"]),
                float(r["max_longitude"]),
                str(r["primary_crime_type"]),
                str(r["dominant_community_area"]),
            ))
        print(f"[Database] Seeded {len(df_clusters)} hotspot clusters into 'hotspots'.")

    # 2. Seed Sample Crime Records
    sample_csv = DATA_DIR / "Crimes_Sample_2020_2025.csv"
    if sample_csv.exists():
        df_crimes = pd.read_csv(sample_csv, nrows=1000)
        cursor.execute("DELETE FROM crime_records")
        inserted = 0
        for _, r in df_crimes.iterrows():
            if pd.notnull(r.get("Latitude")) and pd.notnull(r.get("Longitude")):
                cursor.execute("""
                    INSERT OR IGNORE INTO crime_records (
                        source_record_id, case_number, occurred_at, block_address,
                        crime_type, location_description, arrest_made,
                        domestic_incident, community_area_id, latitude, longitude
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(r.get("ID", f"ID_{inserted}")),
                    str(r.get("Case Number", "")),
                    str(r.get("Date", "")),
                    str(r.get("Block", "")),
                    str(r.get("Primary Type", "THEFT")),
                    str(r.get("Location Description", "STREET")),
                    1 if r.get("Arrest") is True else 0,
                    1 if r.get("Domestic") is True else 0,
                    int(r.get("Community Area")) if pd.notnull(r.get("Community Area")) else None,
                    float(r["Latitude"]),
                    float(r["Longitude"]),
                ))
                inserted += 1
        print(f"[Database] Seeded {inserted} records into 'crime_records'.")

    # 3. Seed a provisional prediction record demonstration
    cursor.execute("""
        INSERT INTO predictions (
            query_latitude, query_longitude, target_prediction_timestamp,
            predicted_risk_tier, predicted_risk_score, nearest_hotspot_id,
            distance_to_hotspot_km, model_name, model_version
        ) VALUES (41.8781, -87.6298, '2026-10-27 12:00:00', 'High (Proximity < 500m)', 0.85, 1, 0.24, 'DBSCAN_Proximity_Heuristic', 'v0.1-provisional')
    """)

    conn.commit()

    # Verify query
    cursor.execute("SELECT COUNT(*) FROM hotspots")
    h_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM crime_records")
    c_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions")
    p_count = cursor.fetchone()[0]

    conn.close()
    print(f"[Database] Verification query results: {h_count} hotspots, {c_count} crime_records, {p_count} predictions.")
    return h_count, c_count, p_count


if __name__ == "__main__":
    init_database()
    seed_data()
