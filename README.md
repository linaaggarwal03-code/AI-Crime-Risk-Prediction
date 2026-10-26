# AI-Powered Urban Crime Risk & Hotspot Prediction System

**Academic Milestone 2 Implementation**  
**Team:**  
- **Member 1 (Lina Aggarwal):** Dataset collection, preprocessing, core ML classification & count prediction modeling.  
- **Member 2 (Aditi Verma):** Spatial Exploratory Data Analysis, Haversine DBSCAN Hotspot Detection, Spatial Feature Engineering, Full-Stack Next.js Dashboard, FastAPI Backend, PostgreSQL Database Schema, Testing & Academic Documentation.  

---

## 1. Project Overview & Architecture

This repository contains the complete implementation of the spatial analysis, density-based clustering, and geospatial intelligence platform for the City of Chicago Reported Crime Data (2020–2025).

```
AI-Crime-Risk-Prediction/
├── data/
│   ├── Crimes_Sample_2020_2025.csv       # Authentic 25,000 Chicago crime dataset
│   └── Crimes_-_2001_to_Present_20260820.csv # Lina's full 1.42M dataset (local)
│
├── src/
│   ├── data_loader.py                    # Programmatic column discovery & geo-boundary validation
│   ├── spatial_eda.py                    # Spatial descriptive statistics & visualizations A-E
│   ├── dbscan_clustering.py              # Haversine DBSCAN clustering & parameter sensitivity
│   ├── feature_engineering.py            # Point-in-time spatial feature extractor (leakage-free)
│   ├── download_sample_data.py           # Authentic Chicago Open Data API ingestion utility
│   └── generate_notebooks.py             # Reproducible Jupyter notebook generator
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Lina's temporal and categorical EDA
│   ├── 02_spatial_eda.ipynb              # Aditi's spatial exploratory data analysis
│   └── 03_dbscan_hotspot_analysis.ipynb  # Aditi's DBSCAN hotspot clustering & parameter tuning
│
├── backend/
│   ├── main.py                           # FastAPI REST application
│   ├── schemas.py                        # Pydantic request/response schemas
│   ├── config.py                         # File paths, CORS, and settings
│   └── requirements.txt                  # Backend dependencies
│
├── frontend/
│   ├── package.json                      # Next.js 14, React, TypeScript, Tailwind, Leaflet
│   └── src/
│       ├── app/
│       │   ├── page.tsx                  # 6-section analytics dashboard
│       │   ├── layout.tsx                # App layout & styling
│       │   └── globals.css               # Dark navy design & leaflet CSS
│       ├── components/
│       │   └── CrimeMap.tsx              # Interactive client-side Leaflet mapping component
│       └── data/
│           └── analysisData.json         # Precomputed analytical outputs for resilient fallback
│
├── database/
│   ├── schema.sql                        # PostgreSQL DDL with PostGIS-ready types & constraints
│   ├── seed_data.py                      # Relational migration & SQLite verification script
│   └── crime_intelligence.db             # Local zero-dependency SQLite database
│
├── outputs/
│   ├── hotspot_results.csv               # Incident-level cluster assignments (mapped by ID)
│   ├── cluster_summary.csv               # 29 detected DBSCAN hotspot clusters
│   ├── parameter_experiments.csv         # 25-configuration hyperparameter sensitivity grid
│   ├── analysis_summary.json             # Structured metadata & statistical summary
│   ├── spatial_eda_summary.json          # Spatial coordinate extents and category distributions
│   ├── interactive_hotspot_map.html      # Standalone interactive Leaflet HTML map
│   ├── interactive_spatial_map.html      # Standalone historical incidents Leaflet HTML map
│   └── figures/                          # Publication-quality static figures (PNG)
│       ├── vis_a_historical_locations.png
│       ├── vis_b_crime_density_heatmap.png
│       ├── vis_c_crime_category_distribution.png
│       ├── vis_d_area_level_distribution.png
│       ├── vis_e_geographic_coverage.png
│       ├── dbscan_hotspot_clusters.png
│       ├── dbscan_cluster_sizes.png
│       └── dbscan_parameter_tuning.png
│
├── docs/
│   ├── implementation_inventory.md       # Inventory of existing code vs new deliverables
│   ├── spatial_methodology.md            # Mathematical formulations (Haversine & DBSCAN)
│   ├── database_schema.md                # PostgreSQL schema & Entity-Relationship specification
│   ├── limitations.md                    # Data limitations, policing biases & ethical safeguards
│   ├── milestone2_progress_report.md     # Formal academic progress report for Milestone 2
│   ├── viva_questions.md                 # Technical viva Q&A preparation guide
│   └── feature_proposal.md               # Spatial feature engineering proposal for Member 1
│
├── tests/
│   ├── test_data_loader.py               # Unit tests for coordinate validation & column discovery
│   ├── test_dbscan_clustering.py         # Unit tests for Haversine metric & cluster assignment
│   ├── test_feature_engineering.py       # Unit tests for point-in-time calculation & data leakage
│   └── test_backend_api.py               # Integration tests for FastAPI endpoints
│
├── conftest.py                           # Pytest root configuration
└── README.md
```

---

## 2. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Node.js 18+ and npm 10+ (tested on Node v24.19.0)

### Step 1: Python Environment Setup
```bash
# Activate virtual environment or create one
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Step 2: Run End-to-End Spatial Analysis Pipeline
```bash
# 1. (Optional) Fetch fresh Chicago crime data slice (25,000 records)
python src/download_sample_data.py

# 2. Run Spatial Exploratory Data Analysis (generates figures A-E and summary JSON)
python src/spatial_eda.py

# 3. Run DBSCAN Hotspot Detection & Parameter Tuning
python src/dbscan_clustering.py

# 4. Generate Spatial Feature Engineering Proposal
python src/feature_engineering.py

# 5. Seed Relational Database
python database/seed_data.py
```

### Step 3: Run Automated Test Suite
```bash
pytest tests -v
```
*All 16 unit and integration tests execute with a 100% pass rate.*

### Step 4: Launch Backend API
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Swagger Interactive Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Detected Hotspots: `http://localhost:8000/api/hotspots`

### Step 5: Launch Next.js Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser to view the 6-view interactive dashboard.

---

## 3. Key Analytical Results (Milestone 2)

- **Analyzed Records:** 24,960 valid spatial records (99.84% completeness; 40 missing coordinates excluded).
- **Distance Metric:** Spherical Haversine ($R = 6,371.0088\text{ km}$) on coordinates in radians. Prevents 34% planar distortion error.
- **Selected DBSCAN Hyperparameters:** $\varepsilon = 500\text{ meters}$, $\text{min\_samples} = 50$.
- **Detected Hotspot Clusters:** **29 clusters**
- **Clustered Incident Share:** **19,351 incidents (77.53%)**
- **Dispersed Background Noise:** **5,609 incidents (22.47%)**
- **Top Concentration:** Cluster #0 centered in the Chicago Downtown / Near North Side / West Town commercial and transit corridor with 15,660 incidents.

---

## 4. Academic Documentation Links

1. [Implementation Inventory](docs/implementation_inventory.md)
2. [Spatial Methodology & Haversine Formulations](docs/spatial_methodology.md)
3. [Database Architecture & Schema Specification](docs/database_schema.md)
4. [System Limitations & Responsible AI](docs/limitations.md)
5. [Milestone 2 Academic Progress Report](docs/milestone2_progress_report.md)
6. [Academic Viva Questions & Answers](docs/viva_questions.md)
7. [Spatial Feature Engineering Proposal](docs/feature_proposal.md)