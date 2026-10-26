# PostgreSQL Database Architecture & Schema Specification

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Author:** Aditi Verma (Member 2, Geospatial Data Scientist & Full-Stack Developer)  
**Academic Milestone:** Milestone 2  
**Date:** October 2026  

---

## 1. Architectural Overview

The relational database architecture is designed to store, index, and query historical crime incidents, detected DBSCAN hotspot clusters, and future prospective model predictions. The schema balances analytical query performance with data normalization and PostGIS readiness.

```
+---------------------------------------------------------------------------------------+
|                                    RELATIONAL SCHEMA                                  |
+---------------------------------------------------------------------------------------+

   +------------------------+                    +----------------------------+
   |      crime_records     |                    |          hotspots          |
   +------------------------+                    +----------------------------+
   | PK id (BIGSERIAL)      |                    | PK id (SERIAL)             |
   |    source_record_id    |                    |    cluster_id (INT)        |
   |    occurred_at (TS)    |                    |    algorithm_name (TEXT)   |
   |    crime_type (TEXT)   |                    |    distance_metric (TEXT)  |
   |    community_area_id   |                    |    eps_meters (FLOAT)      |
   |    latitude (FLOAT)    |                    |    min_samples (INT)       |
   |    longitude (FLOAT)   |                    |    incident_count (INT)    |
   +-----------+------------+                    |    representative_lat      |
               |                                 |    representative_lon      |
               | 1:N                             +-------------+--------------+
               v                                               |
   +-----------+------------+                                  |
   |    spatial_features    |                                  | 1:N
   +------------------------+                                  |
   | PK id (BIGSERIAL)      |                                  |
   | FK crime_record_id     |                                  v
   |    feature_timestamp   |                    +-------------+--------------+
   |    spatial_density_500m|                    |        predictions         |
   |    dist_to_hotspot_km  |                    +----------------------------+
   |    community_crime_rate|                    | PK id (BIGSERIAL)          |
   +------------------------+                    |    query_latitude (FLOAT)  |
                                                 |    query_longitude (FLOAT) |
                                                 |    target_prediction_ts    |
                                                 |    predicted_risk_tier     |
                                                 |    predicted_risk_score    |
                                                 | FK nearest_hotspot_id      |
                                                 |    distance_to_hotspot_km  |
                                                 +----------------------------+
```

---

## 2. Table Specifications

### Table A: `crime_records`
Stores validated historical reported crime incidents.

| Column | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | `PRIMARY KEY` | Unique database identifier |
| `source_record_id` | `VARCHAR(64)` | `UNIQUE, NOT NULL` | Chicago Police Department incident ID |
| `case_number` | `VARCHAR(32)` | `NULLABLE` | Official Chicago Police docket number |
| `occurred_at` | `TIMESTAMP WITH TIME ZONE` | `NOT NULL` | Timestamp of recorded incident occurrence |
| `block_address` | `VARCHAR(128)` | `NULLABLE` | Anonymized hundred-block street address |
| `crime_type` | `VARCHAR(64)` | `NOT NULL` | Primary offense type (e.g. THEFT, BATTERY) |
| `crime_description` | `VARCHAR(128)` | `NULLABLE` | Detailed sub-classification description |
| `location_description` | `VARCHAR(64)` | `NULLABLE` | Physical venue (e.g. STREET, RESIDENCE) |
| `arrest_made` | `BOOLEAN` | `DEFAULT FALSE` | Legal arrest outcome |
| `domestic_incident` | `BOOLEAN` | `DEFAULT FALSE` | Domestic violence indicator |
| `community_area_id` | `INTEGER` | `NULLABLE` | Chicago Community Area (1 to 77) |
| `police_district_id` | `INTEGER` | `NULLABLE` | Chicago Police District (1 to 25) |
| `latitude` | `DOUBLE PRECISION` | `NOT NULL, CHECK` | Latitude ($41.60 \le \text{lat} \le 42.10$) |
| `longitude` | `DOUBLE PRECISION` | `NOT NULL, CHECK` | Longitude ($-88.05 \le \text{lon} \le -87.45$) |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | `DEFAULT CURRENT_TIMESTAMP` | Database insertion timestamp |

**Indexes:**
- `idx_crime_records_coords`: `(latitude, longitude)` for spatial bounding box filtering.
- `idx_crime_records_occurred_at`: `(occurred_at)` for rolling window temporal extraction.
- `idx_crime_records_type`: `(crime_type)` for categorical filtering.

---

### Table B: `hotspots`
Stores identified DBSCAN cluster centroids and spatial envelopes.

| Column | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `SERIAL` | `PRIMARY KEY` | Unique hotspot cluster identifier |
| `cluster_id` | `INTEGER` | `NOT NULL` | DBSCAN cluster ID (-1 reserved for noise) |
| `analysis_timestamp`| `TIMESTAMP WITH TIME ZONE` | `DEFAULT CURRENT_TIMESTAMP` | Execution timestamp of clustering run |
| `algorithm_name` | `VARCHAR(64)` | `DEFAULT 'DBSCAN'` | Clustering algorithm |
| `distance_metric` | `VARCHAR(64)` | `DEFAULT 'Haversine'` | Spatial metric |
| `eps_meters` | `DOUBLE PRECISION` | `NOT NULL` | Neighborhood radius in meters ($500.0\text{m}$) |
| `min_samples` | `INTEGER` | `NOT NULL` | Density sample threshold ($50$) |
| `incident_count` | `INTEGER` | `NOT NULL` | Total incidents assigned to cluster |
| `percentage_of_total` | `DOUBLE PRECISION`| `NOT NULL` | Share of total analyzed records |
| `representative_latitude` | `DOUBLE PRECISION` | `NOT NULL` | Mean cluster centroid latitude |
| `representative_longitude` | `DOUBLE PRECISION` | `NOT NULL` | Mean cluster centroid longitude |
| `min_latitude` | `DOUBLE PRECISION` | `NOT NULL` | Minimum latitude bounding box boundary |
| `max_latitude` | `DOUBLE PRECISION` | `NOT NULL` | Maximum latitude bounding box boundary |
| `min_longitude` | `DOUBLE PRECISION` | `NOT NULL` | Minimum longitude bounding box boundary |
| `max_longitude` | `DOUBLE PRECISION` | `NOT NULL` | Maximum longitude bounding box boundary |
| `primary_crime_type` | `VARCHAR(64)` | `NULLABLE` | Dominant crime category in cluster |
| `dominant_community_area` | `VARCHAR(32)` | `NULLABLE` | Dominant Chicago Community Area |

---

### Table C: `predictions`
Stores prospective risk predictions produced by supervised models.

| Column | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | `PRIMARY KEY` | Unique prediction record ID |
| `query_latitude` | `DOUBLE PRECISION` | `NOT NULL` | Query latitude coordinate |
| `query_longitude` | `DOUBLE PRECISION` | `NOT NULL` | Query longitude coordinate |
| `target_prediction_timestamp` | `TIMESTAMP WITH TIME ZONE` | `NOT NULL` | Target forecast timestamp |
| `horizon_hours` | `INTEGER` | `DEFAULT 24` | Lookahead horizon |
| `predicted_risk_tier` | `VARCHAR(32)` | `NOT NULL` | e.g. 'LOW', 'MODERATE', 'HIGH' |
| `predicted_risk_score` | `DOUBLE PRECISION` | `NULLABLE` | Calibrated classification probability $[0, 1]$ |
| `predicted_crime_count` | `DOUBLE PRECISION` | `NULLABLE` | Expected count from regression model |
| `nearest_hotspot_id` | `INTEGER` | `REFERENCES hotspots(id)` | Foreign key linking nearest cluster |
| `distance_to_hotspot_km` | `DOUBLE PRECISION` | `NULLABLE` | Haversine distance to nearest centroid ($km$) |
| `model_name` | `VARCHAR(64)` | `NOT NULL` | Supervised model identifier |
| `model_version` | `VARCHAR(32)` | `NOT NULL` | Model release version |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | `DEFAULT CURRENT_TIMESTAMP` | Inference log timestamp |

---

## 3. Example Queries

### Find Top 5 Hotspots by Incident Concentration:
```sql
SELECT cluster_id, incident_count, percentage_of_total, primary_crime_type, dominant_community_area
FROM hotspots
ORDER BY incident_count DESC
LIMIT 5;
```

### Proximity Query: Identify Incidents within 500m of Chicago Downtown:
```sql
SELECT source_record_id, crime_type, occurred_at, latitude, longitude
FROM crime_records
WHERE (6371.0088 * 2 * ASIN(SQRT(
    POWER(SIN(RADIANS(latitude - 41.8781) / 2), 2) +
    COS(RADIANS(41.8781)) * COS(RADIANS(latitude)) *
    POWER(SIN(RADIANS(longitude - (-87.6298)) / 2), 2)
))) <= 0.5;
```

---

## 4. Verification and Local Compatibility Layer

For testing environments lacking an active PostgreSQL daemon, `database/seed_data.py` provides an automated SQLite verification engine (`database/crime_intelligence.db`). In local execution, the script automatically generated the tables, populated 29 hotspot records, inserted 998 crime incident records, and executed successful relational verification queries.
