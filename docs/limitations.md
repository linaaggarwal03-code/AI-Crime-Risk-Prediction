# System Limitations and Responsible AI Considerations

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Author:** Aditi Verma (Member 2, Geospatial Data Scientist & Full-Stack Developer)  
**Academic Milestone:** Milestone 2  
**Date:** October 2026  

---

## 1. Spatial Analysis & Clustering Limitations

1. **Retrospective Nature of Unsupervised Clustering:**
   DBSCAN clusters reflect strictly *where recorded crime historically clustered* within the analyzed temporal span. A spatial hotspot is not an intrinsic property of a geographic neighborhood; it is an empirical summary of past reported events. Clustering alone does not account for demographic shifts, ongoing urban development, or dynamic temporal events.
2. **Cluster Centroid Representation:**
   DBSCAN discovers non-convex clusters of irregular geometric shapes (such as linear commercial corridors or L-shaped avenue blocks). A mean coordinate centroid $(\bar{\phi}, \bar{\lambda})$ is purely a summary location point and does not imply that the cluster is circular or that the centroid itself is the peak of crime activity.
3. **Boundary and Edge Effects:**
   Spatial clustering is bounded by the jurisdictional limits of the City of Chicago. Incidents occurring in adjacent municipalities (e.g. Evanston, Oak Park, Cicero) are absent from the Chicago Police Department records, potentially creating edge artifacts near city boundaries.
4. **Modifiable Areal Unit Problem (MAUP):**
   Aggregating crime counts into administrative units (Community Areas, Wards, or Police Districts) suffers from MAUP, where spatial boundaries are arbitrary and varying bin sizes yield differing apparent concentrations. Point-based DBSCAN mitigates MAUP by clustering raw coordinates directly, but district-level aggregations remain subject to this constraint.

---

## 2. Historical Crime Data Biases

1. **Reporting Disparities:**
   Recorded crime data reflects incidents reported to or detected by law enforcement, not total criminal activity. Certain categories (such as auto theft or commercial burglary) have near 100% reporting rates due to insurance mandates, whereas simple battery, domestic incidents, and minor thefts are systematically under-reported.
2. **Proactive vs. Reactive Policing Biases:**
   Certain offense types—predominantly `WEAPONS VIOLATION` (63.4% arrest rate) and `NARCOTICS`—are detected primarily through active police stops and patrol concentration rather than citizen dispatch calls. Consequently, higher recorded frequencies in specific neighborhoods may reflect elevated patrol deployment rather than higher underlying incidence.
3. **Midnight Recording Artifact:**
   Temporal audits reveal an artificial spike at hour 0 (00:00:00) with over 103,000 incidents. This reflects administrative data entry practices where incidents with unrecorded exact times are defaulted to midnight, rather than a genuine biological surge in late-night activity.

---

## 3. Ethical Safeguards & Responsible Use Guidelines

1. **Prohibition of Sensitive Personal Attributes:**
   The spatial and predictive pipelines strictly exclude race, ethnicity, religion, socioeconomic status, and personal identifiers.
2. **Anti-Stigmatization Safeguards:**
   System dashboards and analytical reports must never label residential communities as "dangerous" or "high-risk zones." Visualizations explicitly label features as *Historical Incident Density*, emphasizing that past concentration does not define community safety or future behavior.
3. **Quarantine of Post-Event Outcomes:**
   Incident-level post-investigation variables (`Arrest`, `IUCR`, `FBI Code`, `Description`, `Updated On`) are quarantined from prospective risk modeling features to prevent target data leakage.
4. **Decision-Support Context:**
   The system is engineered as an analytical decision-support tool for municipal resource allocation (e.g. streetlight placement, emergency medical stationing, community assistance dispatch), and must never be utilized to automate punitive enforcement actions.
