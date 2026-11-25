# Milestone 3 Standalone Technical Progress Report

**Project Title:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Contributor:** Aditi Verma (Member 2)  
**Working Branch:** `aditi`  
**Date:** November 2026  

---

## 1. Scope of Milestone 3 Deliverables

Building directly on the Milestone 2 spatial foundation (`src/data_loader.py`, `src/spatial_eda.py`, `src/dbscan_clustering.py`, `src/feature_engineering.py`, FastAPI backend, and Next.js 14 dashboard), Milestone 3 extends Member 2's standalone deliverables into a production-ready, database-backed, containerized spatial analytics system.

> **Integration Note:** Member 1's (Lina Aggarwal) supervised ML pipeline (`lina-ml` branch) remains decoupled at this stage; live ML model integration is intentionally deferred to a separate team integration step.

---

## 2. Implemented Milestone 3 Enhancements

### 2.1 Advanced Spatial-Temporal Analytics (`src/spatial_temporal_analysis.py`)
- Enriched the 24,960 validated Chicago crime records with parsed temporal dimensions (`hour`, `day_of_week`, 6-hour `time_window`, and binary `day_night`).
- Exported structured analytics to `outputs/spatial_temporal_summary.json` and `frontend/src/data/analysisData.json`:
  - **Daytime (06:00–17:59):** 13,310 incidents
  - **Nighttime (18:00–05:59):** 11,650 incidents
  - **24-Hour Diurnal Distribution:** Quantified hourly incident counts across 00:00–23:00.
  - **Cluster Day/Night Shifts:** Computed daytime vs. nighttime volume splits across the top 10 DBSCAN clusters.
  - **Category Hotspot Concentration:** Measured clustered vs. noise (-1) proportions for top offense types.

### 2.2 Live Database Persistence (`database/db_manager.py`)
- Implemented `DatabaseManager` supporting **PostgreSQL** (via `DATABASE_URL` and `psycopg2`) with automatic **SQLite** fallback (`database/crime_intelligence.db`) for zero-configuration local execution.
- Connected `POST /api/predict-risk` to persist every spatial risk query (coordinates, target timestamp, nearest cluster ID, Haversine distance, and provisional tier) into the `predictions` table.
- Added `GET /api/prediction-history` to retrieve persisted query logs.

### 2.3 Extended FastAPI Endpoints (`backend/main.py`)
- `GET /api/spatial-temporal`: Serves cached spatial-temporal analytics.
- `GET /api/prediction-history`: Returns recent database-persisted spatial risk queries.
- `GET /api/historical-incidents`: Extended with `time_window` (`Morning`, `Afternoon`, `Evening`, `Night`) and `community_area` query filters alongside `category` and `cluster_id`.

### 2.4 Interactive Frontend Upgrades (`frontend/`)
- **Historical Crime Map:** Added interactive dropdowns for **Crime Category**, **Time-of-Day Window**, and **Cluster ID**, dynamically rendering filtered incident markers on the Leaflet map.
- **Analytics Tab:** Added 4 interactive **Recharts** visualizations:
  1. 24-Hour Diurnal Crime Distribution (`AreaChart`)
  2. Daytime vs. Nighttime Hotspot Shifts (`BarChart`)
  3. Time-of-Day Window Breakdown (`BarChart`)
  4. Category Hotspot Concentration: Clustered vs. Noise (`BarChart`)
- **Risk Prediction Tab:** Added the **Recent Spatial Risk Queries (Persisted History)** table showing live database records, while preserving the transparent pending-ML-integration notice.

### 2.5 Containerization & Reproducible Packaging
- Added `backend/Dockerfile`, `frontend/Dockerfile`, and `docker-compose.yml` orchestrating PostgreSQL 15, FastAPI (`:8000`), and Next.js (`:3000`) with health checks and automatic schema initialization (`database/schema.sql`).

---

## 3. Verification & Test Outcomes

- **Pytest Suite:** `pytest tests -v` — **23/23 tests passing** (16 Milestone 2 tests + 7 new Milestone 3 tests covering `SpatialTemporalAnalyzer`, `DatabaseManager`, filtered `/api/historical-incidents`, `/api/spatial-temporal`, and `/api/prediction-history`).
- **Frontend Production Build:** `npm run build` in `frontend/` — Compiled and linted with 0 errors.

---

## 4. Remaining Team Integration Step

- Merging Lina's `lina-ml` branch models into `backend/` and replacing the provisional spatial proximity heuristic in `/api/predict-risk` with live supervised model inference.
