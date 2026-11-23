
# Sprint 3, Step 1: Baseline Forecasting Evaluation Report

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Task:** Implementation and validation of non-parametric naive baselines  
**Evaluation Split:** 2024 Validation Dataset (`data/processed/val_features.parquet`)  
**Record Count:** 28,182 observations (77 Community Areas $\times$ 366 calendar days in 2024)  
**Primary Target:** `crime_count` (reported daily crime incidents per Community Area)  

---

## 1. Executive Summary

Before training machine learning estimators (such as Ridge Regression or Random Forest), two empirical naive baselines were established on the **2024 Validation dataset**:
1. **Previous-Day Baseline:** Predicts today's crime volume using yesterday's observed crime volume ($\hat{y}_{i, t} = y_{i, t-1} = \text{lag\_1}$).
2. **Previous-Week Baseline:** Predicts today's crime volume using the same day of the prior week ($\hat{y}_{i, t} = y_{i, t-7} = \text{lag\_7}$).

Both baselines were evaluated on identical validation ground truth using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**. The **Previous-Day Baseline (`lag_1`) outperformed the Previous-Week Baseline (`lag_7`) across all metrics**, establishing the formal target benchmark that subsequent machine learning models must beat.

---

## 2. Validation Dataset & Target Alignment

* **Target Variable:** `crime_count` (Actual daily incidents in Community Area $i$ on date $t$).
  * Mean: $9.20$ crimes/day
  * Standard Deviation: $8.16$ crimes/day
  * Min: $0$, Max: $81$ crimes/day
* **Lag Inputs Verified:**
  * `lag_1`: Previous-day count ($t-1$ relative to target date $t$). Zero missing values across all 28,182 rows.
  * `lag_7`: Previous-week count ($t-7$ relative to target date $t$). Zero missing values across all 28,182 rows.
* **Temporal Alignment:** For every observation row $(i, t)$, the target date is strictly aligned with the prediction timestamp. The 2024 validation dataset spans from `2024-01-01` to `2024-12-31`.

---

## 3. Baseline Performance Comparison

| Baseline Model | Feature Used | Mathematical Definition | 2024 Val MAE | 2024 Val RMSE | Validation Rank |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Previous-Day Baseline** | `lag_1` | $\hat{y}_{i, t} = \max(0, y_{i, t-1})$ | **3.3592** | **4.6725** | **1 (Superior)** |
| **Previous-Week Baseline** | `lag_7` | $\hat{y}_{i, t} = \max(0, y_{i, t-7})$ | **3.4502** | **4.8111** | **2** |

*Absolute Advantage of Previous-Day over Previous-Week:*
* **$\Delta$ MAE:** $-0.0910$ crimes/day (2.6% lower absolute error)
* **$\Delta$ RMSE:** $-0.1386$ crimes/day (2.9% lower quadratic penalty)

---

## 4. Non-Negative Prediction Enforcement

* **Constraint:** Crime volume is strictly non-negative ($y \ge 0$). Predicted counts must satisfy $\hat{y}_{i, t} \ge 0$.
* **Implementation:** $\hat{y} = \max(0.0, \text{prediction})$ is enforced in [src/baselines.py](file:///c:/Users/Lina%20Aggarwal/Desktop/AI-Crime-Risk-Prediction/src/baselines.py).
* **Impact Analysis:** Because `lag_1` and `lag_7` are historical incident counts from our verified Cartesian grid, all lag values are naturally integers $\ge 0$. There were zero negative values in the raw validation features (`(val_df['lag_1'] < 0).sum() == 0`). Consequently, non-negative clipping leaves the baseline predictions unchanged, but guarantees safety against future model outputs that might predict negative numbers.

---

## 5. Methodological Insights & Next Steps

### Why Does Previous-Day Outperform Previous-Week?
While urban crime exhibits weekly periodicity (higher counts on Fridays and Saturdays), **immediate spatial momentum** is stronger than day-of-week recurrence. High-activity periods or localized crackdowns often persist across consecutive days within a community area, making yesterday's count ($t-1$) a closer numerical estimate than last week's count ($t-7$).

### Benchmark Target for Machine Learning:
To be considered successful in Sprint 3 Step 2:
$$\begin{aligned}
\text{Required MAE} &< \mathbf{3.3592} \text{ crimes/day} \\
\text{Required RMSE} &< \mathbf{4.6725} \text{ crimes/day}
\end{aligned}$$
Any multi-variable estimator (Ridge, Random Forest, LightGBM) that fails to surpass this baseline does not justify added model complexity.

### Test Set Preservation:
The 2025 Test Dataset (`data/processed/test_features.parquet`, 28,105 rows) remains completely untouched and was not used to select or tune the baselines.
