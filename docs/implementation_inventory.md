# Project Implementation Inventory

**Academic Project Title:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Team:**  
- **Member 1 (Lina Aggarwal):** Data collection, preprocessing, core ML classification/regression training, model validation.  
- **Member 2 (Aditi Verma):** Spatial Exploratory Data Analysis, Geographic DBSCAN Hotspot Detection, Spatial Feature Engineering, Full-Stack Dashboard, PostgreSQL Database Design, API Integration, Academic Documentation, Viva Preparation.  
**Date:** October 2026  
**Git Branch:** `aditi`  

---

## 1. What Already Exists in the Repository

Prior to Member 2's implementation, Member 1 initialized the project repository with exploratory analysis work:

1. **Jupyter Notebook (`notebooks/01_data_exploration.ipynb`):**
   - Ingestion and temporal/categorical exploration of Chicago Police Department Reported Crime Data spanning 2020 through 2025.
   - Analysis of 1,422,980 total crime records across 22 primary attributes.
   - Identified 22,430 records (1.58%) with missing spatial coordinates (`Latitude`, `Longitude`).
   - Analyzed yearly volume trends (troughs in 2020–2021 pandemic years, peak in 2023 with 263,431 incidents).
   - Documented monthly seasonality (February low of 101,689 incidents vs July high of 132,240 incidents; +30.0% variance).
   - Documented diurnal cycles (5:00 AM trough of 24,824 incidents; peak afternoon-to-evening; midnight reporting artifact).
   - Documented primary crime types (Theft: 21.8%, Battery: 18.0%, Criminal Damage: 11.4%, Assault: 8.9%).
   - Identified spatial concentration across Chicago's 77 Community Areas (Austin #25, Near North Side #8, Near West Side #28, South Shore #43, The Loop #32 account for ~20% of all crime).
   - Highlighted critical data leakage hazard: post-incident outcomes (`Arrest`, `IUCR`, `Description`, `FBI Code`, `Updated On`) must not be used as features in prospective risk prediction models.

2. **Generated Figures (`reports/figures/`):**
   - `01_yearly_crime_trend.png`
   - `02_top_crime_types.png`
   - `03_crime_by_hour_and_day.png`
   - `04_monthly_seasonality.png`
   - `05_arrest_rate_by_crime_type.png`
   - `06_top_community_areas.png`

3. **Repository Directory Layout:**
   - Scaffolded directories: `app/`, `data/`, `models/`, `notebooks/`, `reports/`, `src/`, `tests/`.
   - `requirements.txt`: base data science packages (`pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyter`).

---

## 2. What Works

- **Schema and Field Integrity:** The 22 attributes of the Chicago crime data have been verified and validated.
- **Temporal Analysis:** Seasonality, diurnal patterns, and yearly trajectories are established and documented.
- **Exploratory Visualizations:** Static distribution charts for temporal and categorical breakdowns are generated.
- **Quarantine Boundaries:** Data leakage rules are established to prevent post-event variables from contaminating future predictive features.

---

## 3. What My Part (Aditi, Member 2) Adds

To fulfill Milestone 2 deliverables and my assigned core responsibilities, I am implementing:

1. **Robust Data Ingestion & Validation Pipeline (`src/data_loader.py`):**
   - Universal data loader that dynamically discovers coordinate fields (`Latitude`/`Longitude` or `latitude`/`longitude`), handles encoding, and validates coordinate boundaries against standard Chicago spatial limits (Latitude: $41.64^\circ\text{N} - 42.03^\circ\text{N}$, Longitude: $-87.94^\circ\text{W} - -87.52^\circ\text{W}$).
   - Clear accounting and reporting of excluded records (missing coordinates vs out-of-bounds outliers).

2. **Spatial Exploratory Data Analysis Module (`src/spatial_eda.py` & `notebooks/02_spatial_eda.ipynb`):**
   - Spatial density calculations, bounding box coverage, coordinate summary metrics.
   - Five dedicated spatial visualizations:
     * **Visualization A:** Historical Incident Scatter Plot (spatial distribution across city grid).
     * **Visualization B:** Spatial Density Heatmap (2D Gaussian Kernel Density Estimation).
     * **Visualization C:** Top Crime Categories by Spatial Frequency.
     * **Visualization D:** Community Area Incident Volume distribution.
     * **Visualization E:** Spatial Coverage and Bounding Extent.

3. **Geographic Hotspot Clustering using DBSCAN (`src/dbscan_clustering.py` & `notebooks/03_dbscan_hotspot_analysis.ipynb`):**
   - Full scikit-learn DBSCAN clustering using the **Haversine** metric on spherical radians to avoid Cartesian distortion at Chicago's latitude ($\approx 41.88^\circ\text{N}$).
   - Parameter experiment suite testing multiple combinations of $\varepsilon$ ($200\text{ m} \le \varepsilon \le 1000\text{ m}$) and $\text{min\_samples}$ ($15 \le \text{min\_samples} \le 100$).
   - Extraction of cluster centroids, incident densities, bounding boxes, noise counts, and percentage clustered.
   - Generation of analytical artifacts:
     * `outputs/hotspot_results.csv`
     * `outputs/cluster_summary.csv`
     * `outputs/parameter_experiments.csv`
     * `outputs/analysis_summary.json`
     * `outputs/figures/` (cluster distribution map, cluster size bar chart, noise breakdown).
     * `outputs/interactive_hotspot_map.html` (interactive Folium/Leaflet map).

4. **Spatial Feature Engineering Architecture (`src/feature_engineering.py` & `docs/spatial_methodology.md`):**
   - Engineered spatial predictors for Lina's future supervised model:
     * Neighborhood incident density within $500\text{ m}$ radius.
     * Distance to nearest historic high-density cluster centroid ($d_{\text{cluster}}$).
     * Historical crime rate by Community Area.
     * Historical DBSCAN cluster membership flag.
   - Strict data leakage safeguards: point-in-time calculation enforcing $t_{\text{feature}} < t_{\text{prediction}}$.

5. **Preliminary PostgreSQL Database Architecture (`database/schema.sql` & `database/seed_data.py`):**
   - Normalized relational schema supporting raw incidents (`crime_records`), detected hotspots (`hotspots`), prospective model outputs (`predictions`), and spatial features (`spatial_features`).
   - Seeding and verification scripts.

6. **FastAPI Backend Integration (`backend/main.py`, `backend/schemas.py`, `backend/config.py`):**
   - High-performance REST API serving precomputed spatial summaries, hotspot clusters, historical incident samples, and health checks.
   - Structured provisional prediction endpoint that transparently documents model integration status.

7. **Full-Stack Academic Frontend Dashboard (`frontend/`):**
   - Next.js / React / TypeScript / Tailwind CSS interactive dashboard with 6 specialized views:
     1. Executive Dashboard (KPI metric cards, cluster distribution, category breakdown).
     2. Historical Crime Map (interactive Leaflet map with zoom, cluster overlays, and tooltips).
     3. Hotspot Analysis (DBSCAN parameter explorer, cluster metrics, cluster summary table).
     4. Analytics (temporal, categorical, and community-area distributions).
     5. Risk Prediction (interactive risk interface with transparent pending-integration banner).
     6. Methodology & Responsible AI (detailed mathematical explanation, ethical safeguards).

8. **Automated Unit & Integration Test Suite (`tests/`):**
   - Tests covering data loading, coordinate validation, haversine conversion, DBSCAN execution, noise handling, summary generation, and API endpoint contracts.

9. **Academic Documentation & Viva Preparation:**
   - Detailed methodology report (`docs/spatial_methodology.md`).
   - Database specification (`docs/database_schema.md`).
   - Limitations and ethical considerations (`docs/limitations.md`).
   - Milestone 2 comprehensive progress report (`docs/milestone2_progress_report.md`).
   - Comprehensive viva examination Q&A guide (`docs/viva_questions.md`).

---

## 4. What Remains Dependent on Lina (Member 1)

1. **Final Supervised Model Testing:**
   - Lina is finalizing model evaluation (accuracy, precision, recall, F1, RMSE/MAE) on the held-out 2025 test split.
2. **Model Artifact Export:**
   - Once testing completes, Lina will serialize the trained model artifacts (`.pkl` or `.joblib`) into `models/`.
3. **End-to-End Prediction Inference Hook:**
   - The FastAPI `/api/predict-risk` endpoint is architected with complete Pydantic request/response schemas; it currently returns an honest status indicating model weights are pending Lina's final validation. Once Lina exports the model weights, the inference function will be plugged in without requiring any API or frontend schema redesign.
