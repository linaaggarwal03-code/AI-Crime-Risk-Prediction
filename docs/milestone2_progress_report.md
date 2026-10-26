# Academic Milestone 2 Progress Report

**Project Title:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Institution / Course:** Academic Capstone / Milestone 2 Review  
**Student Name:** Aditi Verma (Member 2)  
**Teammate:** Lina Aggarwal (Member 1)  
**Role:** Senior Full-Stack Developer, Geospatial Data Scientist & ML Engineer  
**Date:** October 2026  
**Git Repository Branch:** `aditi`  

---

## 1. Overall Project Objective

The objective of this academic project is to build an integrated, AI-driven crime intelligence platform that combines **supervised prospective crime risk and volume forecasting** with **unsupervised geographic hotspot detection** across the urban domain of Chicago.

The platform assists municipal analysts and emergency dispatchers by uncovering spatial density patterns, forecasting future neighborhood risk tiers, and presenting interactive, explainable geospatial intelligence without relying on discriminatory or biased predictors.

---

## 2. Team Division of Responsibilities

- **Member 1 (Lina Aggarwal):**
  - Dataset collection and initial ingestion.
  - Exploratory analysis on temporal dynamics, crime categorization, and diurnal patterns.
  - Supervised machine learning model architecture (classification & regression).
  - Model validation on 2024 split (*final model testing on 2025 split pending*).
- **Member 2 (Aditi Verma - My Contribution):**
  - Robust spatial data ingestion and automated coordinate schema discovery.
  - Geographic boundary validation and coordinate auditing.
  - Spatial Exploratory Data Analysis (EDA) and publication-quality diagnostic visualizations (A through E).
  - Implementation of scikit-learn DBSCAN hotspot clustering using the spherical **Haversine** distance metric.
  - Multi-parameter sensitivity experiments across 25 configurations of $\varepsilon$ and $\text{min\_samples}$.
  - Analytical artifact generation (`hotspot_results.csv`, `cluster_summary.csv`, `parameter_experiments.csv`, `analysis_summary.json`).
  - Spatial feature engineering architecture with point-in-time calculation to eliminate future-to-past data leakage.
  - Full-stack interactive academic dashboard (Next.js 14, React, TypeScript, Tailwind CSS, Leaflet).
  - FastAPI backend REST architecture with precomputed spatial endpoints and provisional prediction handler.
  - Preliminary PostgreSQL database schema design (`schema.sql`) and SQLite verification migration (`seed_data.py`).
  - Automated unit and integration testing suite (`pytest` with 16 passing tests).
  - Academic methodology, limitations, and viva examination documentation.

---

## 3. Dataset Fields Relevant to Spatial Analysis

The spatial pipeline operates on the City of Chicago Reported Crime Data (2020–2025):

| Field Name | Type | Analytical Function in Spatial Module |
| :--- | :--- | :--- |
| `ID` / `source_record_id` | Integer / String | Unique primary identifier; preserved for assignment mapping |
| `Latitude` | Float | Geographic latitude in decimal degrees ($41.60^\circ\text{N} \le \phi \le 42.10^\circ\text{N}$) |
| `Longitude` | Float | Geographic longitude in decimal degrees ($-88.05^\circ\text{W} \le \lambda \le -87.45^\circ\text{W}$) |
| `Primary Type` | String | Offense classification; used to identify modal crime in detected clusters |
| `Community Area` | Float / Int | Chicago Community Area identifier (1 to 77) |
| `Date` | Datetime / Str | Occurrence timestamp; used for point-in-time rolling window filters |
| `Block` | String | Anonymized hundred-block address |

---

## 4. Geographic Preprocessing & Coordinate Auditing

The spatial data loader (`src/data_loader.py`) programmatically scans candidate fields without hardcoding assumptions. All records are validated against mathematical limits and Chicago jurisdictional boundaries:

- **Total records processed:** 25,000
- **Valid spatial coordinates:** 24,960 (99.84%)
- **Excluded missing coordinates:** 40 (0.16%)
- **Mathematical invalid / out-of-bounds:** 0 (0.00%)
- **Latitude Span:** $41.6458^\circ\text{N}$ to $42.0225^\circ\text{N}$ ($\Delta \phi = 0.3767^\circ$)
- **Longitude Span:** $-87.9096^\circ\text{W}$ to $-87.5246^\circ\text{W}$ ($\Delta \lambda = 0.3850^\circ$)
- **Geographic Mean Center:** $41.8461^\circ\text{N}, -87.6675^\circ\text{W}$

---

## 5. Spatial EDA Visualizations

Five diagnostic visualizations were programmatically generated and exported to `outputs/figures/`:
1. **Visualization A (`vis_a_historical_locations.png`):** Scatter distribution of historical crime incidents across Chicago.
2. **Visualization B (`vis_b_crime_density_heatmap.png`):** 2D hexagonal binning density heatmap on logarithmic scale.
3. **Visualization C (`vis_c_crime_category_distribution.png`):** Distribution of top 12 primary crime types (Theft: 23.5%, Battery: 17.8%, Criminal Damage: 11.4%).
4. **Visualization D (`vis_d_area_level_distribution.png`):** Volume distribution by Community Area (Near North Side, Austin, Near West Side, South Shore, Loop).
5. **Visualization E (`vis_e_geographic_coverage.png`):** Latitude and Longitude coordinate distributions with median lines.
6. **Interactive Spatial Map (`interactive_spatial_map.html`):** Standalone Leaflet map rendering incident points and tooltips.

---

## 6. DBSCAN Hotspot Detection & Distance Metric

### 6.1 Spherical Haversine Distance Formulation
Planar Euclidean coordinates introduce over 34% distortion error at Chicago's latitude ($41.88^\circ\text{N}$) because meridians converge toward the poles ($1^\circ\text{ lat} \approx 111\text{ km}$ vs $1^\circ\text{ lon} \approx 82.7\text{ km}$). We transform coordinates to radians and compute spherical distance via the **Haversine metric**:
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
where $R = 6,371.0088\text{ km}$. The neighborhood radius $\varepsilon$ is supplied in radians:
$$\varepsilon_{\text{rad}} = \frac{\varepsilon_{\text{km}}}{R}$$

### 6.2 Parameter Selection & Sensitivity Evaluation
We evaluated 25 parameter combinations across $\varepsilon \in [200\text{m}, 1000\text{m}]$ and $\text{min\_samples} \in [15, 100]$.

**Primary Selected Configuration:**
- $\varepsilon = 500\text{ meters}$ ($\varepsilon_{\text{rad}} \approx 7.848 \times 10^{-5}\text{ rad}$)
- $\text{min\_samples} = 50\text{ incidents}$

**Clustering Outcomes:**
- **Identified Hotspot Clusters:** 29
- **Clustered Incidents:** 19,351 (77.53%)
- **Noise / Dispersed Incidents:** 5,609 (22.47%)
- **Largest Hotspot Cluster:** 15,660 incidents (Cluster #0, central commercial/transit core)
- **Smallest Hotspot Cluster:** 24 incidents
- **Median Cluster Size:** 69 incidents

**Generated Visualizations:**
- `outputs/figures/dbscan_hotspot_clusters.png` (cluster map with noise in light gray and top centroids starred).
- `outputs/figures/dbscan_cluster_sizes.png` (incident counts by hotspot cluster).
- `outputs/figures/dbscan_parameter_tuning.png` (sensitivity curves of cluster count and noise fraction vs $\varepsilon$).
- `outputs/interactive_hotspot_map.html` (interactive Leaflet map with centroids and bounding envelopes).

---

## 7. Frontend Dashboard Implementation

A full-stack responsive web application was developed in `frontend/` using **Next.js 14, React, TypeScript, and Tailwind CSS**:

- **Architecture:** Client-side Leaflet mapping component with dynamic SSR handling, eliminating hydration errors.
- **6 Core Views:**
  1. *Dashboard Overview:* Real KPI summary cards, top hotspot rankings, category distribution bars, and community area breakdowns.
  2. *Historical Crime Map:* Full-screen interactive map with pan/zoom, cluster filter dropdown, centroid markers, and bounding boxes.
  3. *Hotspot Analysis:* Parameter overview, DBSCAN sensitivity matrix table, and cluster summary table.
  4. *Analytics:* Geographic bounding extents, temporal diurnal cycles, and generated visualization gallery.
  5. *Risk Prediction:* Interactive query form (lat, lon, date, time) with presets; computes distance to nearest hotspot ($d_{\text{cluster}}$) and provisional risk tier; displays transparent pending-integration badge.
  6. *Methodology:* Complete mathematical formulation of Haversine distance, DBSCAN rationale, and ethical safeguards.
- **Resilience:** Integrates with live FastAPI backend on `:8000` while automatically falling back to embedded analytical data (`analysisData.json`) if the backend is offline. Production build verified (`npm run build` succeeds).

---

## 8. Backend & Database Integration Status

- **FastAPI Backend (`backend/`):**
  - `GET /health`: Service health and active dataset volume.
  - `GET /api/summary`: Precomputed analysis metadata and parameters.
  - `GET /api/hotspots`: Top clusters and centroid list.
  - `GET /api/cluster-summary`: Raw cluster table records.
  - `GET /api/historical-incidents`: Sampled incident coordinates for mapping.
  - `POST /api/predict-risk`: Provisional spatial proximity endpoint with transparent status warning.
- **PostgreSQL Database (`database/schema.sql`):**
  - Fully normalized schema with PostGIS-ready coordinate types, constraints, and B-tree indexes for `crime_records`, `hotspots`, `predictions`, and `spatial_features`.
  - Local verification script (`database/seed_data.py`) executed successfully, inserting 29 hotspots and 998 crime records.

---

## 9. Status Summary: Completed vs. Pending

| Deliverable Component | Status | Responsible Member |
| :--- | :--- | :--- |
| Dataset Collection & Preprocessing | **Completed** | Member 1 (Lina) |
| Core Classification & Count Model Training | **Completed** | Member 1 (Lina) |
| Spatial EDA & Diagnostic Visualizations (A–E) | **Completed** | Member 2 (Aditi) |
| Haversine DBSCAN Hotspot Detection Engine | **Completed** | Member 2 (Aditi) |
| Hyperparameter Sensitivity Analysis Grid | **Completed** | Member 2 (Aditi) |
| Spatial Feature Engineering Architecture | **Completed** | Member 2 (Aditi) |
| PostgreSQL Relational Database Design | **Completed** | Member 2 (Aditi) |
| FastAPI Backend Integration | **Completed** | Member 2 (Aditi) |
| Full-Stack Next.js Analytics Dashboard | **Completed** | Member 2 (Aditi) |
| Automated Test Suite (16 pytest passing tests) | **Completed** | Member 2 (Aditi) |
| Final Supervised Model Testing (Held-Out 2025) | *Pending* | Member 1 (Lina) |
| Serialized Model Export (`.pkl` into `models/`) | *Pending* | Member 1 (Lina) |
| Live Supervised Model Inference Hook in API | *Pending Lina's Weights* | Team Integration |
