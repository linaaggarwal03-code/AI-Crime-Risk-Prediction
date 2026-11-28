# Milestone 3: Crime Hotspot Detection & Spatial Density Report

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Target Temporal Horizon:** 2024  
**Primary Methodology:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using Spherical Haversine Metric on BallTree  
**Continuous Intensity Surface:** Gaussian Kernel Density Estimation (KDE) with Spherical Metric  

---

## 1. Geographic Coordinate Integrity & Audit

Spatial hotspot detection requires exact, high-fidelity GPS point coordinates (`Latitude`, `Longitude`). Imputing or fabricating coordinates introduces severe geographic bias. Therefore, all records with null coordinates or coordinates outside the valid physical bounds of the City of Chicago were strictly audited and excluded.

- **Total Incidents in Dataset:** 1,422,980
- **Missing Coordinates (NaN):** 22,430 (1.58%)
- **Out-of-Bounds Coordinates:** 2 (0.0001%) *(e.g. 2 records logged at lat 36.62, lon -91.69 in Missouri/Arkansas)*
- **Verified Chicago Coordinates:** 1,400,548 (98.42%)
- **Incidents Analyzed in Evaluation Window (2024):** 257,526
- **Zero Imputation Policy:** Confirmed. Missing or corrupt coordinates are dropped without synthetic or centroid imputation.

---

## 2. Methodology & Mathematical Framework

### 2.1 Density-Based Spatial Clustering (DBSCAN)
Crime occurrences are non-uniformly distributed across urban street networks and do not conform to simple convex geometric shapes. Centroid-based algorithms like K-Means are unsuited because they force every point into a cluster and assume spherical partitions. In contrast, **DBSCAN** offers three decisive advantages for urban crime analysis:
1. **Arbitrary Geometry:** Detects linear transit corridors, commercial strips, and irregular neighborhood clusters.
2. **Built-in Noise Segregation:** Isolated, dispersed incidents are categorized as background noise (`label = -1`), preventing dilution of high-priority enforcement zones.
3. **Great-Circle Spherical Metric:** Measures true spatial distances over Earth's surface via Haversine distance:

$$d_H(p_1, p_2) = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

where $R = 6371.0088$ km, $\phi$ is latitude in radians, and $\lambda$ is longitude in radians.

### 2.2 Parameter Justification & Configuration
- **Primary Localized Epsilon ($\epsilon = 100.0$ meters):** Set to 100 meters, corresponding to approximately half of a standard Chicago city street block (a standard Chicago block is 660 ft / ~201m by 330 ft / ~100m). This search radius isolates true micro-places (specific intersections and discrete commercial corridors) while preventing cross-neighborhood density chaining.
- **Min Samples ($min\_samples = 50$ incidents):** Requires a minimum concentration of 50 reported incidents within the 100m radius over the calendar year (~1 crime per week at that micro-location). This isolates persistent hotspot locations from temporary, transient spikes.
- **Algorithm:** `ball_tree` indexing on coordinates in radians, providing exact spherical distance queries.

---

## 3. Primary Hotspot Clustering Results ($\epsilon = 100$m, $min\_samples = 50$)

- **Total Detected Localized Hotspot Clusters:** 617
- **Incidents in High-Density Hotspots:** 104,921 (40.74%)
- **Background Noise Incidents:** 152,605 (59.26%)

### Top 15 Detected Crime Hotspots (Ranked by Incident Volume)

| Rank | Hotspot ID | Crimes | % Total | Centroid (Lat, Lon) | Radius (m) | Dominant Area | Top Crime Types |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 9 | 17,199 | 6.68% | (41.8870, -87.6278) | 2718m | Area 8 | THEFT (6764); BATTERY (2312); DECEPTIVE PRACTICE (1913) |
| 2 | 12 | 3,290 | 1.28% | (41.7621, -87.5690) | 1326m | Area 43 | BATTERY (755); THEFT (513); CRIMINAL DAMAGE (408) |
| 3 | 35 | 2,076 | 0.81% | (41.7482, -87.6048) | 1041m | Area 44 | BATTERY (489); THEFT (280); CRIMINAL DAMAGE (268) |
| 4 | 19 | 1,580 | 0.61% | (41.9467, -87.6518) | 640m | Area 6 | THEFT (631); BATTERY (319); DECEPTIVE PRACTICE (159) |
| 5 | 20 | 1,320 | 0.51% | (41.8852, -87.6513) | 606m | Area 28 | THEFT (620); BATTERY (142); DECEPTIVE PRACTICE (102) |
| 6 | 62 | 1,269 | 0.49% | (41.8953, -87.7132) | 800m | Area 23 | NARCOTICS (645); BATTERY (149); THEFT (81) |
| 7 | 59 | 1,229 | 0.48% | (41.9066, -87.6724) | 825m | Area 24 | THEFT (534); BATTERY (142); CRIMINAL DAMAGE (120) |
| 8 | 61 | 1,068 | 0.41% | (41.9644, -87.6562) | 493m | Area 3 | THEFT (417); BATTERY (216); ASSAULT (96) |
| 9 | 67 | 999 | 0.39% | (41.7796, -87.6150) | 558m | Area 69 | BATTERY (317); ASSAULT (123); CRIMINAL DAMAGE (103) |
| 10 | 24 | 984 | 0.38% | (41.8814, -87.7534) | 603m | Area 25 | BATTERY (229); THEFT (143); NARCOTICS (117) |
| 11 | 149 | 862 | 0.33% | (41.9390, -87.6508) | 468m | Area 6 | THEFT (434); BATTERY (131); DECEPTIVE PRACTICE (54) |
| 12 | 21 | 845 | 0.33% | (41.9312, -87.6447) | 706m | Area 7 | THEFT (489); BATTERY (71); DECEPTIVE PRACTICE (59) |
| 13 | 27 | 780 | 0.30% | (41.8682, -87.6403) | 370m | Area 28 | THEFT (403); BATTERY (89); ASSAULT (73) |
| 14 | 41 | 778 | 0.30% | (41.8540, -87.6247) | 381m | Area 33 | THEFT (187); BATTERY (141); CRIMINAL DAMAGE (106) |
| 15 | 136 | 718 | 0.28% | (41.8026, -87.5885) | 552m | Area 41 | THEFT (349); BATTERY (74); MOTOR VEHICLE THEFT (65) |

---

## 4. Parameter Sensitivity Analysis: Density Chaining & The 250m Comparison

### 4.1 What is DBSCAN Density Chaining?
DBSCAN relies on transitive density-reachability: if Point A is density-reachable from Point B, and Point B from Point C, all three points are grouped into the same cluster. In densely populated urban environments with hundreds of thousands of crimes stacked across an entire 365-day year, an overly generous radius causes clusters to 'chain' across adjacent street blocks.

### 4.2 Why $\epsilon = 250$m Caused Density Chaining
When evaluated at $\epsilon = 250\text{ meters}$ across the unstratified 2024 dataset:
- **One giant mega-cluster (Cluster 0)** absorbed **236,725 incidents (91.92% of the city's crime)**.
- Cluster 0 spanned **69 out of Chicago's 77 Community Areas**, with a north-south extent of 40.5 km and an east-west extent of 22 km.
- A cluster containing 92% of all incidents across 69 neighborhoods is not a localized hotspot; it represents the contiguous urban street grid of Chicago.
- The only incidents not merged were enclaves physically separated by major geographic breaks (e.g. O'Hare airport terminals, Lake Calumet marshes, river channels).

### 4.3 Direct Comparative Evaluation: 100m vs. 250m

| Metric | Primary Config ($\epsilon = 100\text{m}$) | Baseline Comparison ($\epsilon = 250\text{m}$) | Methodological Assessment |
| :--- | :--- | :--- | :--- |
| **Epsilon (Search Radius)** | **100 meters** | 250 meters | 100m reflects intersection scale; 250m bridges blocks |
| **Total Clusters Discovered** | **617** | 60 | 100m isolates discrete corridors; 250m creates massive chaining |
| **Clustered Incidents** | **104,921 (40.74%)** | 249,961 (97.06%) | 100m reflects empirical micro-concentration; 250m over-clusters |
| **Noise Incidents (Background)**| **152,605 (59.26%)** | 7,565 (2.94%) | 59.3% background noise matches criminological crime concentration literature |
| **Largest Cluster Size** | **17,199 (6.68%)** | 236,725 (91.92%) | 100m confines largest cluster to Loop/Near North; 250m spans whole city |
| **Largest Cluster Community Areas**| **1–2 (Loop / Near North)** | 69 of 77 Community Areas | 250m destroys spatial specificity across neighborhood boundaries |

---

## 5. Continuous Spatial Modeling via Kernel Density Estimation (KDE)

### 5.1 Interpretation of Min-Max Normalization $[0, 1]$
> [!IMPORTANT]
> **Relative Grid Density, NOT Absolute Crime Rates:**
> The evaluated KDE density surface is min-max normalized to $[0, 1]$ across the spatial evaluation grid.
> - A value of `1.0` represents the **relative point of highest crime concentration** in the city (the downtown Loop / Michigan Avenue transit nexus).
> - A value of `0.0` represents the point of lowest observed density (e.g. over Lake Michigan shoreline or rural outer borders).
> - **These values do NOT represent absolute crime rates, incident counts, or probability density per square kilometer.** They represent relative spatial intensity for visual heatmap interpretation.

### 5.2 Grid Resolution & Bandwidth Constraints
- **Gaussian Smoothing Bandwidth ($\sigma = 500$ meters):** A 500m standard deviation smooths individual incidents over approximately 1 km ($2\sigma$). This acts as a spatial low-pass filter, capturing ambient regional risk but smoothing over fine-grained block differences.
- **Evaluation Grid Resolution ($50 \times 50$ cells):**
  - Latitude spacing: $\Delta \phi \approx 1,021\text{ meters}$
  - Longitude spacing: $\Delta \lambda \approx 760\text{ meters}$
  - Individual cell footprint: $\approx 0.78\text{ km}^2$
- **Resolution Disclaimer:** The $50 \times 50$ grid provides a citywide macroscopic density overview. **It does NOT provide street-block or intersection-level resolution.**

---

## 6. Key Findings & Spatial Alignments

1. **Validation of Crime Concentration Law:** Under $\epsilon = 100\text{m}$, 40.7% of all crimes occur in discrete localized hotspots, while 59.3% are dispersed background noise. This strongly corroborates Weisburd's (2015) 'Law of Crime Concentration at Places', which posits that a significant fraction of urban crime concentrates at a tiny proportion of micro-places.
2. **Specific High-Risk Corridors Identified:**
   - **Loop / Near North Side Commercial Transit Core (Rank 1):** 17,199 crimes (theft, deceptive practice, battery) centered along Michigan Avenue and State Street.
   - **South Shore Commercial Corridor (Rank 2):** 3,290 crimes centered around 71st St transit stations (battery, theft, damage).
   - **Chatham Commercial Hub (Rank 3):** 2,076 crimes along 79th/87th commercial arteries.
   - **Humboldt Park Open-Air Narcotics Market (Rank 6):** 1,269 crimes with 645 narcotics violations centered at (41.8953, -87.7132) along Chicago Avenue.
   - **Austin Madison Street Corridor (Rank 10):** 984 crimes (battery, theft, narcotics).
3. **Milestone 2 Area Model Alignment:** The localized hotspots are concentrated in the community areas flagged by our Milestone 2 Random Forest and Ridge regression models as high risk (Areas 25, 8, 32, 28, 43, 67, 49).

---

## 7. Limitations & Viva / Defense Preparation

1. **Density Chaining Vulnerability:** Global distance-threshold algorithms like DBSCAN are vulnerable to chaining on high-volume, temporally collapsed data. Mitigating this requires tuning $\epsilon$ to the micro-place scale ($100\text{m}$) or employing hierarchical clustering (HDBSCAN/OPTICS).
2. **Fixed Global Epsilon vs. Spatial Heterogeneity:** A uniform 100m radius is applied equally across downtown high-rises and single-family South/West Side residential areas. In high-rises, 100m covers thousands of vertical occupants, whereas in residential areas it covers a single street segment.
3. **Reporting Bias:** High crime counts in downtown and retail centers reflect store security enforcement and transit cameras rather than severe violent threat. Stratifying by offense category (violent vs. property) is essential for tactical resource allocation.
4. **KDE Grid Scale:** The $50 \times 50$ KDE grid gives macroscopic heatmaps, not street-level tactical dispatch maps.
