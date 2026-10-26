# Academic Viva Examination Preparation Guide

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Author:** Aditi Verma (Member 2, Geospatial Data Scientist & Full-Stack Developer)  
**Academic Milestone:** Milestone 2 Review  
**Date:** October 2026  

---

## Section 1 — Project Understanding & Collaboration

### Q1: What core problem does this project address?
**Answer:**  
The project addresses urban public safety analytics by integrating spatial exploratory analysis, historical density hotspot detection, and predictive machine learning. Traditional municipal policing and emergency dispatch often rely on retrospective aggregate crime reports tied to static administrative boundaries. This project provides a fine-grained, data-driven approach that uncovers true spatial concentration without boundary artifacts and prepares predictive features for prospective crime risk estimation.

### Q2: What is the fundamental difference between crime prediction and hotspot detection?
**Answer:**  
- **Hotspot Detection (DBSCAN - My Contribution):** An **unsupervised, retrospective** clustering technique that discovers where historical incident reports accumulated in continuous physical space. It identifies past density patterns without claiming that crime will recur at that exact location.
- **Crime Risk Prediction (Supervised ML - Lina's Contribution):** A **prospective forecasting** task that uses supervised classification and time-series regression to estimate the future probability or count of crime in a spatial unit over forward time horizons (e.g. next 24 hours), conditioning on calendar cycles, weather, land use, and historical lagged features.

### Q3: What is your individual contribution to this project as Member 2?
**Answer:**  
As Member 2, I am responsible for:
1. Programmatic data loading, automated column discovery, and coordinate boundary validation (`src/data_loader.py`).
2. Spatial Exploratory Data Analysis (EDA) and generating the 5 analytical figures (`src/spatial_eda.py`).
3. Implementation of the DBSCAN hotspot detection pipeline using spherical **Haversine** distance (`src/dbscan_clustering.py`).
4. Systematic hyperparameter tuning across 25 combinations of $\varepsilon$ and $\text{min\_samples}$ (`outputs/parameter_experiments.csv`).
5. Spatial feature engineering with strict point-in-time calculation to eliminate data leakage (`src/feature_engineering.py`).
6. Preliminary PostgreSQL relational database architecture (`database/schema.sql`) and verification seeding (`database/seed_data.py`).
7. FastAPI REST backend service with precomputed spatial endpoints (`backend/main.py`).
8. Full-stack interactive academic dashboard built with Next.js 14, React, TypeScript, Tailwind CSS, and Leaflet (`frontend/`).
9. Automated unit and integration testing suite (`tests/` with 16 passing tests).
10. Complete academic methodology, limitations, and progress report documentation.

### Q4: How does your work complement Lina's work?
**Answer:**  
Lina (Member 1) completed data collection, initial temporal EDA, and core supervised model training (classification and count forecasting). My spatial clustering and spatial feature extraction directly complement her models by translating unstructured $(lat, lon)$ points into structured spatial predictors—such as distance to the nearest historical hotspot ($d_{\text{cluster}}$) and localized 30-day neighborhood density. Furthermore, my full-stack dashboard and FastAPI backend provide the deployment vehicle to visualize both historical hotspots and future risk predictions.

---

## Section 2 — DBSCAN Algorithm & Spatial Clustering

### Q5: What is DBSCAN and why was it chosen over k-means?
**Answer:**  
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised density-based algorithm. We chose it over $k$-means for three reasons:
1. **Arbitrary Geometry:** $k$-means assumes spherical, convex clusters of similar variance. Urban crime concentrates along irregular linear streets, transit corridors, and arterial avenues. DBSCAN discovers arbitrary non-convex geometric shapes.
2. **No Pre-specified Cluster Count:** $k$-means requires guessing $k$ in advance. DBSCAN discovers the natural number of clusters based on local density.
3. **Explicit Noise Segregation:** $k$-means forces every outlier into a cluster, pulling centroids away from real density centers. DBSCAN explicitly classifies dispersed points as noise (label -1).

### Q6: What does $\varepsilon$ (epsilon) represent?
**Answer:**  
$\varepsilon$ defines the radius of the spherical neighborhood centered at each query point. In our implementation, $\varepsilon = 500\text{ meters}$, which is converted to spherical radians as $\varepsilon_{\text{rad}} = 0.5 / 6371.0088 \approx 7.848 \times 10^{-5}\text{ radians}$ for scikit-learn's Haversine metric.

### Q7: What does $\text{min\_samples}$ represent?
**Answer:**  
$\text{min\_samples}$ is the minimum number of data points (including the core point itself) required within the $\varepsilon$-neighborhood for that point to be classified as a **core point**. In our primary configuration, $\text{min\_samples} = 50$.

### Q8: What does a noise point (label -1) represent? Is it a "safe location"?
**Answer:**  
A noise point represents an isolated or dispersed historical incident that occurred in an area where incident density did not reach the threshold ($\ge 50$ incidents within 500m). **A noise point is emphatically NOT a "safe location"**; it simply indicates that past crime incidents in that location were spatially dispersed rather than heavily concentrated.

### Q9: How are clusters generated by DBSCAN?
**Answer:**  
1. Points with $\ge \text{min\_samples}$ within $\varepsilon$ are labeled **core points**.
2. If point $B$ is within $\varepsilon$ of core point $A$, $B$ is directly density-reachable from $A$.
3. If point $C$ is reachable via a chain of core points, $C$ is density-reachable.
4. A cluster is defined as the maximal set of mutually density-connected points. Points that are neither core points nor density-reachable from any core point are labeled as **noise (-1)**.

### Q10: How were your parameters selected?
**Answer:**  
We evaluated 25 parameter combinations ($\varepsilon \in [200\text{m}, 1000\text{m}]$, $\text{min\_samples} \in [15, 100]$):
- At $\varepsilon = 200\text{m}$, excessive fragmentation occurred, discarding up to 92.8% of records as noise.
- At $\varepsilon \ge 750\text{m}$, spatial chaining collapsed the city into 5 to 8 massive mega-clusters (>94% in a single cluster).
- **$\varepsilon = 500\text{m}$ with $\text{min\_samples} = 50$** was selected because it identified 29 distinct, interpretable urban clusters containing 77.53% of incidents while preserving 22.47% as genuine background noise.

### Q11: What are DBSCAN's primary limitations?
**Answer:**  
1. **Uniform Density Assumption:** DBSCAN uses a global $\varepsilon$ and $\text{min\_samples}$, making it struggle when clusters have widely varying densities (e.g. downtown high-density commercial zones vs suburban residential strips).
2. **Computational Complexity:** Pairwise distance calculation is $\mathcal{O}(n^2)$ without spatial indexing; using a BallTree index reduces this to $\mathcal{O}(n \log n)$.

---

## Section 3 — Geographic Distance & Spatial Mathematics

### Q12: Why are latitude and longitude not ordinary Cartesian coordinates?
**Answer:**  
Latitude and longitude are angular measurements on an oblate spheroid, not distances on a flat plane. The distance represented by one degree of longitude shrinks as latitude increases away from the equator:
$$\Delta x = \Delta \lambda \cdot R \cdot \cos(\phi)$$
At Chicago's latitude ($41.88^\circ\text{N}$), $1^\circ$ of latitude is $\approx 111.0\text{ km}$, but $1^\circ$ of longitude is $\approx 82.7\text{ km}$. Applying Euclidean distance directly on degrees introduces an asymmetric distortion error exceeding **34%**.

### Q13: What is Haversine distance and why convert coordinates to radians?
**Answer:**  
The Haversine formula calculates the shortest surface distance (great-circle distance) between two points on a sphere of radius $R$:
$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
Coordinates must be converted to radians ($\text{radians} = \text{degrees} \times \pi / 180$) because trigonometric functions in standard scientific libraries (including scikit-learn's `metric='haversine'`) operate exclusively on radian angles.

### Q14: How are cluster centroids calculated and what is the caveat?
**Answer:**  
Representative cluster centroids are computed as the arithmetic mean of the coordinates of all assigned incident points:
$$\bar{\phi} = \frac{1}{N} \sum_{i=1}^N \phi_i, \quad \bar{\lambda} = \frac{1}{N} \sum_{i=1}^N \lambda_i$$
**Caveat:** A mean coordinate centroid is purely a representative reference point. Because DBSCAN clusters are non-convex and irregularly shaped, the centroid may lie in an alley, intersection, or empty space, and does not represent the exact geographic boundary or peak density.

---

## Section 4 — Software Architecture & Full-Stack Implementation

### Q15: Why did you choose Next.js and Tailwind CSS for the frontend?
**Answer:**  
Next.js 14 with TypeScript and Tailwind CSS provides:
1. **Type Safety:** TypeScript enforces compile-time schema contracts between API responses and UI components.
2. **Component Modularity:** The interactive Leaflet map, KPI cards, and parameter tables are decoupled into responsive, reusable components.
3. **Resilience & Zero Hydration Crashes:** Client-side dynamic imports prevent server-side rendering crashes caused by Leaflet's reliance on browser `window`.
4. **Performance:** `npm run build` generates optimized, tree-shaken static bundles with sub-second page transitions.

### Q16: How does the frontend handle API connectivity?
**Answer:**  
The frontend connects dynamically to the FastAPI backend at `http://localhost:8000`. If the backend is running, live API endpoints (`/api/summary`, `/api/hotspots`, `/api/predict-risk`) are consumed. If the backend is offline, the frontend automatically falls back to pre-embedded analytical data (`analysisData.json`), ensuring 100% fail-safe operation during academic demonstrations.

### Q17: What does the FastAPI backend do?
**Answer:**  
`backend/main.py` provides a high-performance REST service. It caches precomputed DBSCAN outputs and spatial summaries in memory to prevent expensive recalculation on every request. It exposes endpoints for health, hotspots, cluster tables, incident points, and provisional risk prediction.

### Q18: What is the preliminary PostgreSQL schema used for?
**Answer:**  
The PostgreSQL schema (`database/schema.sql`) stores:
- `crime_records`: Raw validated historical incidents with coordinate boundary constraints.
- `hotspots`: Identified DBSCAN cluster metadata, centroids, and bounding extents.
- `predictions`: Prospective prediction requests, predicted risk tiers, and model versioning.
- `spatial_features`: Point-in-time engineered spatial features for model training.

---

## Section 5 — Responsible AI, Ethics & Data Leakage

### Q19: Does a historical hotspot prove future crime will occur there?
**Answer:**  
**No.** A historical hotspot is purely an empirical reflection of past reported incident density. Environmental changes, commercial turnover, proactive community interventions, or seasonal shifts alter crime patterns. Conflating historical clustering with guaranteed future crime constitutes an ecological fallacy.

### Q20: What is data leakage and how did you prevent it in spatial feature engineering?
**Answer:**  
Data leakage occurs when information from the target or from the future is inadvertently included in training features. If spatial density is calculated using the full static dataset, events occurring *after* prediction time $T$ leak into the model.  
**Safeguard:** In `src/feature_engineering.py`, we enforce strict point-in-time calculation:
$$\text{Density}(T) = \text{Count}(\{x_i \mid \text{Date}_i \in [T - 30\text{ days}, T)\})$$
Events occurring at or after $T$ are strictly excluded. Furthermore, post-event police outcomes (`Arrest`, `IUCR`, `FBI Code`) are quarantined.

### Q21: Can crime data contain reporting and policing biases?
**Answer:**  
Yes. Official crime records represent *reported and detected* crimes, not all crime. Proactive offenses (such as drug or weapons violations) reflect police patrol allocation rather than underlying community rates. In addition, minor offenses have lower reporting rates in historically disenfranchised communities. System outputs are therefore framed as decision-support metrics for municipal resource allocation, never for automated punitive measures.
