# Spatial Feature Engineering Proposal for Supervised Crime Risk Models

**Author:** Aditi Verma (Member 2, Geospatial Data Scientist)  
**Recipient:** Lina Aggarwal (Member 1, Lead ML Engineer)  
**Date:** October 2026  
**Status:** Exploratory / Proposed for Integration  

---

## 1. Executive Summary

To augment Lina's core predictive models (classification and crime count forecasting), this document defines four engineered spatial features derived from historical incident points and DBSCAN clustering outputs. 

These features capture spatial concentration, proximity to chronic high-crime corridors, and local geographic variance without causing target leakage.

---

## 2. Engineered Spatial Feature Definitions

| Feature Name | Definition | Metric / Units | Intended Model Use |
| :--- | :--- | :--- | :--- |
| `dist_to_nearest_hotspot_km` | Great-circle Haversine distance from prediction location $(lat, lon)$ to the nearest DBSCAN hotspot centroid | Kilometers ($km$) | Continuous predictor for classification & regression. Measures spatial exposure to chronic crime nodes. |
| `spatial_density_500m_30d` | Count of recorded historical crime events within a 500-meter radius over the preceding 30 days | Integer count | Captures localized micro-level spatial momentum. |
| `community_area_crime_rate` | Normalized historical incident proportion for the assigned Chicago Community Area | Unitless probability $[0, 1]$ | Static spatial prior representing neighborhood-level baseline volume. |
| `is_historical_hotspot_member` | Binary indicator ($1/0$) designating whether the location resides within the bounding envelope of a verified DBSCAN cluster | Binary flag | Categorical flag for high-density spatial zones. |

---

## 3. Critical Data Leakage Prevention Protocol

```
Timeline:
[================= Historical Window =================) | [==== Prediction Period ====]
(T - 30 days)                                         T  (T + 1 day / week)
                      ^                               |
                      |                               |
       Features calculated ONLY from events           Prediction targets evaluated
       where event_time < T                           ONLY where event_time >= T
```

### Protocol Requirements:
1. **Strict Temporal Partitioning:** For any prediction made for interval $[T, T + \Delta t]$, all spatial density metrics and rolling event counts must be calculated strictly from events where $	ext{Date} < T$.
2. **Exclusion of Future Events:** Never compute spatial density across the entire static dataset prior to splitting.
3. **Quarantine of Post-Event Fields:** Under no circumstances should incident-level attributes (`Arrest`, `IUCR`, `FBI Code`, `Description`, `Updated On`) be used as spatial predictors, as they represent post-investigation police outcomes that are unknown at prediction time.
4. **Coordination with Lina:** These features are prepared in `src/feature_engineering.py`. Lina may evaluate their feature importance and SHAP values prior to final pipeline integration.
