# Final Evaluation Report: Held-Out 2025 Test Set

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Repository:** `AI-Crime-Risk-Prediction`  
**Branch:** `lina-ml`  
**Target Variable:** `crime_count` (Daily aggregate reported crime incidents per Chicago Community Area)  
**Evaluation Split:** Held-out Calendar Year 2025 (`data/processed/test_features.parquet`, 28,105 observations)  
**Model Training Split:** 2020–2023 (`data/processed/train_features.parquet`, 110,187 observations)  
**Tuning & Model Selection Split:** 2024 Validation (`data/processed/val_features.parquet`, 28,182 observations)  

---

## 1. Executive Summary

This report documents the final evaluation of our predictive models on the strictly held-out **2025 test dataset**. Prior to this evaluation, the 2025 test dataset remained unread, unaccessed, and sealed.

**Preselected Champion:**
* **Random Forest Regressor** (`n_estimators=150, max_depth=12, min_samples_leaf=10, max_features='sqrt'`) was preselected as the champion model strictly based on its 2024 validation results (Validation MAE: **2.5140**, Validation RMSE: **3.4717**).
* Following the project's rigorous anti-leakage protocol, the champion model was fitted **exclusively on the 2020–2023 training data** (110,187 rows). Neither the 2024 validation split nor the 2025 test split were included in model training.

**Final 2025 Test Results:**
* **Test MAE:** **2.4167**
* **Test RMSE:** **3.3365**
* **Test $R^2$:** **0.8081** (explaining 80.81% of daily crime count variance across Chicago's 77 community areas)
* **Test WAPE:** **28.58%**
* **Negative Predictions:** **0 (0.00%)** before clipping
* **Baseline Outperformance:** Random Forest reduced MAE by **25.02%** and RMSE by **25.90%** compared to the Previous-Day baseline (`lag_1`).

---

## 2. Forecasting Setup & Operational Reality

### Nature of the Forecasting Task:
* **Sequential Rolling One-Day-Ahead Forecasting:**  
  On each calendar date $t$, the system predicts daily crime count using historical observed crime counts available up to $t-1$:
  * Autoregressive lags: $t-1, t-2, t-3, t-7, t-14, t-28$.
  * Shifted moving statistics: 7-day, 14-day, and 30-day rolling means, standard deviations, minimums, and maximums computed strictly using `.shift(1)`.
  * Context features: citywide lag and momentum ratio.
  * Calendar and cyclical coordinates for day $t$.
* **Not Year-in-Advance Forecasting:**  
  This evaluation does **not** represent a 365-day recursive trajectory generated without ground-truth updates. Rather, it models the operational deployment where police departments record actual incidents daily and predict resource needs for the upcoming 24 hours.

### Ethical & Sociotechnical Disclaimer:
* The predictions model **aggregate counts of officially reported incidents per community area and calendar date**.
* The model **does not** predict individual criminal behavior, individual recidivism, or personal demographic risk. It is an urban resource allocation and demand forecasting tool.

---

## 3. Final Test Metrics & Model Comparison

The final model and benchmark comparisons evaluated on all **28,105 test observations** in 2025:

| Model | Model Class | Test MAE | Test RMSE | Test $R^2$ | Test WAPE (%) | Raw Negative Preds | MAE Delta vs Baseline | % MAE Reduction vs Baseline | Validation Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | Ensemble Decision Trees | **2.4167** | **3.3365** | **0.8081** | **28.58%** | **0** | **-0.8065** | **-25.02%** | **1** |
| Ridge Regression ($\alpha=0.01$) | Regularized Linear | 2.4148 | 3.3518 | 0.8063 | 28.56% | 34 (0.12%) | -0.8084 | -25.08% | 2 |
| Previous-Day Baseline (`lag_1`) | Naive Persistence | 3.2232 | 4.5027 | 0.6505 | 38.12% | 0 | 0.0000 | 0.00% | 3 |
| Previous-Week Baseline (`lag_7`)| Naive Persistence | 3.3174 | 4.6242 | 0.6313 | 39.23% | 0 | +0.0942 | +2.92% | 4 |

### Metric Definitions:
* **MAE (Mean Absolute Error):** $\frac{1}{N} \sum |y_i - \hat{y}_i| = 2.4167$ crimes/day.
* **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2} = 3.3365$ crimes/day.
* **$R^2$ (Coefficient of Determination):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 0.8081$.
* **WAPE (Weighted Absolute Percentage Error):** $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\% = 28.58\%$.

---

## 4. Actual vs. Predicted Distribution Summary Statistics

Summary statistics comparing actual reported crime counts against champion Random Forest predictions on 2025 test data:

| Metric | Actual `crime_count` (2025) | Predicted `crime_count` (Random Forest) |
| :--- | :---: | :---: |
| **Count** | 28,105 | 28,105 |
| **Mean** | 8.4558 | 8.5139 |
| **Standard Deviation** | 7.6160 | 6.8793 |
| **Minimum** | 0.0000 | 0.6996 |
| **25th Percentile ($q_{25}$)** | 3.0000 | 3.3090 |
| **50th Percentile (Median)** | 6.0000 | 6.4285 |
| **75th Percentile ($q_{75}$)** | 11.0000 | 11.0596 |
| **Maximum** | 63.0000 | 43.1308 |

### Key Observations:
1. **Unbiased Central Tendency:** The predicted mean ($8.5139$) matches the actual test mean ($8.4558$) within **$0.058$ crimes/day** ($0.68\%$ error), demonstrating exceptional calibration across the entire annual cycle.
2. **Quartile Alignment:** The median (6.43 vs 6.00) and 75th percentile (11.06 vs 11.00) align closely with ground truth.
3. **Ensemble Variance Shrinkage:** As expected from bootstrap aggregation, extreme single-day outliers (such as the annual peak of 63 crimes in Austin/Area 25) are smoothed toward the conditional leaf mean, producing a predicted maximum of $43.13$.

---

## 5. Non-Negative Predictions & Clipping Analysis

* **Champion Random Forest:**
  * Tree leaves in the regressor compute the sample mean of non-negative training labels ($y \ge 0$).
  * Consequently, the champion model produced **0 negative predictions** out of 28,105 test rows. Non-negative clipping ($\hat{y} = \max(0.0, \hat{y})$) resulted in zero modifications.
* **Ridge Regression:**
  * Produced **34 negative predictions** ($0.12\%$) in low-crime peripheral areas (e.g., Edison Park / Area 9 and Hegewisch / Area 55) before clipping.
  * Clipping enforced physical realism ($\hat{y} \ge 0$) without skewing evaluation metrics.

---

## 6. Champion vs. Ridge Regression Analysis

* **Champion Retention:**
  Random Forest was retained as the champion model, adhering strictly to our validation-based protocol.
* **Empirical Comparison:**
  * Random Forest achieved superior variance explanation ($R^2 = 0.8081$ vs $0.8063$) and lower RMSE ($3.3365$ vs $3.3518$), indicating better management of high-crime peaks.
  * Ridge Regression achieved a virtually identical MAE ($2.4148$ vs $2.4167$, a difference of just $0.0019$ crimes/day).
* **Takeaway:**
  Both models generalize exceptionally well to held-out data, showing no signs of overfitting. In a production pipeline, Ridge provides near-instantaneous inference with minimal memory footprint, while Random Forest provides non-linear feature interaction modeling and tree-based explainability.

---

## 7. Limitations & Caveats

1. **Reporting Bias:** The dataset reflects reported crimes to law enforcement; under-reporting in certain neighborhoods or specific crime categories is not captured in administrative data.
2. **Aggregated Spatial Boundaries:** Predictions are generated at the Community Area level (77 areas in Chicago). Sub-neighborhood clustering requires spatial density estimation (DBSCAN/KDE, planned for Sprint 4).
3. **Exogenous Factors:** The model does not currently ingest real-time weather (temperature, precipitation) or municipal events, which can drive short-term crime fluctuation.

---

## 8. Anti-Leakage & Integrity Confirmation

* **Zero 2025 Contamination:** The 2025 test dataset was loaded strictly in read-only mode for evaluation and was never used during feature selection, model fitting, or hyperparameter tuning.
* **Zero Incident Loss:** All spatial audit rules, dense Cartesian grid guarantees, and boundary checks established in Sprint 2 were fully conserved.
