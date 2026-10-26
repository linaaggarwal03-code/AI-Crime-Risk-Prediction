# Spatial Analysis & DBSCAN Hotspot Detection Methodology

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Author:** Aditi Verma (Member 2, Geospatial Data Scientist & Full-Stack Developer)  
**Teammate:** Lina Aggarwal (Member 1, Machine Learning Engineer)  
**Academic Milestone:** Milestone 2  
**Date:** October 2026  

---

## 1. Introduction and Objectives

Spatial exploratory analysis and unsupervised clustering of urban crime records are critical components of a modern crime intelligence system. While supervised machine learning models predict incident probability or aggregate frequencies over forward temporal windows, spatial clustering fulfills a distinct analytical function: **identifying dense geographic concentrations of historical incident occurrences without imposing artificial administrative boundaries**.

This document outlines the end-to-end methodology developed for:
1. Programmatic dataset ingestion and coordinate validation.
2. Spatial exploratory data analysis (EDA) and density estimation.
3. Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using the spherical **Haversine** distance metric.
4. Systematic hyperparameter tuning and sensitivity analysis across neighborhood radii ($\varepsilon$) and density thresholds ($\text{min\_samples}$).
5. Spatial feature engineering with strict point-in-time calculation to eliminate future-to-past data leakage.

---

## 2. Geographic Preprocessing & Coordinate Boundary Validation

### 2.1 The Need for Robust Coordinate Auditing
Urban crime datasets commonly contain missing coordinates, unprojected addresses, geocoding failures, or records referencing incidents outside city boundaries. In our ingestion pipeline (`src/data_loader.py`), no assumptions are made about column casing or column names. The module discovers coordinate, category, date, and administrative identifier fields dynamically.

### 2.2 Exclusion Criteria & Boundary Audit
Each record passes through a three-stage geometric validation filter:
1. **Mathematical Validity:** Checks whether coordinates represent legitimate angles on a sphere:
   $$-90.0^\circ \le \text{Latitude} \le 90.0^\circ, \quad -180.0^\circ \le \text{Longitude} \le 180.0^\circ$$
2. **Missingness Identification:** Null, `NaN`, non-numeric, or placeholder values (such as $(0.0, 0.0)$) are excluded from spatial distance calculations and logged in an audit report.
3. **Study Area Jurisdictional Bounds:** Verified against the bounding envelope of the City of Chicago:
   $$41.60^\circ\text{N} \le \text{Latitude} \le 42.10^\circ\text{N}, \quad -88.05^\circ\text{W} \le \text{Longitude} \le -87.45^\circ\text{W}$$

**Observed Dataset Audit Metrics:**
- Total records evaluated: **25,000**
- Spatially valid records: **24,960 (99.84%)**
- Excluded due to missing coordinates: **40 (0.16%)**
- Invalid mathematical coordinates: **0 (0.00%)**
- Out-of-bounds geographic outliers: **0 (0.00%)**

Original data rows remain completely preserved in a clean analysis layer, ensuring Member 1's raw dataset is never altered or overwritten.

---

## 3. Spatial Exploratory Data Analysis (EDA)

Spatial EDA examines how incidents distribute across physical space and administrative units (Chicago's 77 Community Areas and 22 Police Districts):

1. **Visualization A (Historical Incident Scatter):**
   Demonstrates high incident concentration along the eastern commercial lakefront, Near North Side, Chicago Loop, and western arterial corridors (Austin, Garfield Park).
2. **Visualization B (Crime Density Heatmap):**
   Uses 2D hexagonal spatial binning on a logarithmic frequency scale to reveal non-linear density gradients and chronic crime nodes.
3. **Visualization C (Primary Crime Category Breakdown):**
   Confirms that **THEFT** (23.5%) and **BATTERY** (17.8%) dominate recorded crime volume, followed by **CRIMINAL DAMAGE** (11.4%), **ASSAULT** (8.7%), and **MOTOR VEHICLE THEFT** (8.2%).
4. **Visualization D (Community Area Volume):**
   Quantifies concentration in key community areas:
   - Area 8 (Near North Side): 1,209 incidents (4.8%)
   - Area 25 (Austin): 1,204 incidents (4.8%)
   - Area 28 (Near West Side): 1,042 incidents (4.2%)
   - Area 43 (South Shore): 892 incidents (3.6%)
   - Area 32 (The Loop): 884 incidents (3.5%)
5. **Visualization E (Geographic Extent and Coordinate Histograms):**
   Establishes the median geographic center of recorded crime activity at $41.8461^\circ\text{N}, -87.6675^\circ\text{W}$.

---

## 4. DBSCAN Hotspot Detection Pipeline

### 4.1 Why DBSCAN?
Traditional spatial clustering algorithms such as $k$-means enforce spherical cluster shapes and require pre-specifying the number of clusters $k$. Urban crime does not conform to circular geometries; criminal incidents concentrate along linear transportation corridors, commercial avenues, and irregularly shaped neighborhood blocks.

**DBSCAN** offers three fundamental advantages:
1. Discovers arbitrarily shaped, non-convex spatial clusters.
2. Automatically determines the optimal number of clusters from local density.
3. Explicitly segregates isolated, dispersed incidents into a **noise class (label -1)**, preventing outliers from distorting cluster centroids.

### 4.2 Distance Metric: The Spherical Haversine Formulation
Unprojected geographic coordinates $(\phi, \lambda)$ represent angular measurements on an oblate spheroid. At Chicago's latitude ($\phi \approx 41.88^\circ\text{N}$):
- $1^\circ$ of Latitude $\approx 111.0\text{ km}$
- $1^\circ$ of Longitude $\approx 111.0 \times \cos(41.88^\circ) \approx 82.7\text{ km}$

Applying standard Euclidean distance $\sqrt{(\Delta \phi)^2 + (\Delta \lambda)^2}$ treats latitude and longitude degrees as equal planar units, introducing an asymmetric distortion error exceeding **34%**.

To ensure spatial rigor, our implementation converts coordinates to spherical radians and utilizes the **Haversine great-circle metric**:
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
where $R = 6,371.0088\text{ km}$ is the volumetric mean Earth radius (IUGG standard).

The DBSCAN neighborhood radius parameter $\varepsilon$ is passed to scikit-learn in angular radians:
$$\varepsilon_{\text{rad}} = \frac{\varepsilon_{\text{km}}}{R} = \frac{\varepsilon_{\text{meters}} / 1000}{6371.0088}$$

For our primary configuration of $\varepsilon = 500\text{ meters}$:
$$\varepsilon_{\text{rad}} = \frac{0.5}{6371.0088} \approx 7.848 \times 10^{-5}\text{ radians}$$

---

## 5. Hyperparameter Tuning and Sensitivity Analysis

A rigorous machine learning workflow requires empirical parameter testing rather than arbitrary parameter selection. We evaluated 25 parameter configurations across:
- $\varepsilon \in \{200\text{m}, 350\text{m}, 500\text{m}, 750\text{m}, 1000\text{m}\}$
- $\text{min\_samples} \in \{15, 25, 50, 75, 100\}$

### 5.1 Sensitivity Findings (Summary of Tested Grid)

| $\varepsilon$ (meters) | $\text{min\_samples}$ | Detected Clusters | Clustered Points | Clustered % | Noise % | Mean Cluster Size |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 200m | 15 | 236 | 13,754 | 55.10% | 44.90% | 58.3 |
| 200m | 50 | 9 | 1,787 | 7.16% | 92.84% | 198.6 |
| 350m | 25 | 67 | 19,410 | 77.76% | 22.24% | 289.7 |
| **500m (Selected)** | **50** | **29** | **19,351** | **77.53%** | **22.47%** | **667.3** |
| 500m | 75 | 21 | 13,338 | 53.44% | 46.56% | 635.1 |
| 750m | 50 | 8 | 23,651 | 94.76% | 5.24% | 2,956.4 |
| 1000m | 50 | 5 | 24,658 | 98.79% | 1.21% | 4,931.6 |

### 5.2 Rationale for Primary Parameter Selection
- At $\varepsilon = 200\text{m}$, excessive spatial fragmentation occurs; requiring 50 samples discards 92.8% of records as noise.
- At $\varepsilon \ge 750\text{m}$, spatial chaining collapses the entire urban core into 5 to 8 massive mega-clusters, obliterating neighborhood-level distinction.
- **$\varepsilon = 500\text{ meters}$ with $\text{min\_samples} = 50$** yields 29 distinct, geographically interpretable clusters containing 77.53% of incidents while designating 22.47% of incidents as genuine, dispersed background noise.

---

## 6. Spatial Feature Engineering & Data Leakage Prevention

### 6.1 Engineered Spatial Predictors for Lina's Model
To support Member 1's supervised prediction models, `src/feature_engineering.py` extracts four spatial features:
1. `dist_to_nearest_hotspot_km`: Great-circle distance to the nearest historical DBSCAN cluster centroid.
2. `spatial_density_500m_30d`: Number of historical incidents within 500 meters during the preceding 30 days.
3. `community_area_crime_rate`: Normalized prior incident frequency by Community Area.
4. `is_historical_hotspot_member`: Binary flag indicating presence within a cluster envelope.

### 6.2 Data Leakage Prevention Protocol
Calculating spatial features across an entire static dataset before temporal splitting causes **severe target leakage**:
$$\text{Feature}(T) = f(\{x_i \mid t_i < T \text{ AND } t_i \ge T\}) \quad \longleftarrow \quad \text{FATAL LEAKAGE}$$

Our implementation strictly enforces point-in-time calculation:
$$\text{Feature}(T) = f(\{x_i \mid t_i \in [T - \Delta t, T)\})$$
All rolling density metrics are evaluated strictly on historical records preceding target prediction timestamp $T$.
Post-event outcome attributes (`Arrest`, `IUCR`, `FBI Code`, `Description`, `Updated On`) are quarantined.
