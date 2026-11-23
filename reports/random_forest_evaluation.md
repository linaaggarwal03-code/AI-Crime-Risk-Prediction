# Sprint 3, Step 3: Random Forest Model Evaluation Report

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Model:** Non-linear Ensemble Regressor (`sklearn.ensemble.RandomForestRegressor`)  
**Training Split:** 2020–2023 (`data/processed/train_features.parquet`, 110,187 observations)  
**Validation Split:** 2024 (`data/processed/val_features.parquet`, 28,182 observations)  
**Primary Target:** `crime_count` (reported daily crime incidents per Community Area)  

---

## 1. Executive Summary

In Step 3 of Sprint 3, a **Random Forest Regressor** was implemented, tuned, and evaluated against both the naive persistence baselines and our previously tuned Ridge Regression model.

The model was configured to test non-linear interactions across spatial indicators, autoregressive lags, rolling statistics, and calendar coordinates:
* The feature space strictly mirrors the anti-leakage partition established in Sprint 2 and Sprint 3.
* Hyperparameters were tuned across a documented grid covering tree depth (`max_depth`), leaf regularization (`min_samples_leaf`), feature subsampling (`max_features`), and ensemble size (`n_estimators`).
* Fixed random seeds (`random_state=42`) were applied throughout to ensure 100% deterministic reproducibility.
* Model selection was governed exclusively by **2024 validation MAE**. The 2025 test dataset remained completely untouched.

**Key Findings:**
1. Random Forest achieved a 2024 validation MAE of **2.5140** and RMSE of **3.4717** with configuration `n_estimators=150, max_depth=12, min_samples_leaf=10, max_features='sqrt'`.
2. Random Forest outperformed the Previous-Day baseline (MAE 3.3592) by **25.16%** MAE reduction and **25.70%** RMSE reduction.
3. Random Forest achieved a modest outperformance over Ridge Regression ($\text{MAE } 2.5140 \text{ vs } 2.5180$, a **$0.0040$** MAE reduction, and $\text{RMSE } 3.4717 \text{ vs } 3.4933$, a **$0.0216$** RMSE reduction).

---

## 2. Feature Selection & Input Partitioning

The model ingests **34 input features** (33 numerical features + 1 spatial identifier):

### Features Used:
1. **Historical Autoregressive Lags (6 features):**
   * `lag_1` (yesterday's count), `lag_2`, `lag_3`, `lag_7` (same day last week), `lag_14`, `lag_28`.
2. **Shifted Rolling Window Statistics (12 features):**
   * 7-day, 14-day, and 30-day moving averages (`rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`).
   * 7-day, 14-day, and 30-day volatility (`rolling_std_7`, `rolling_std_14`, `rolling_std_30`).
   * 7-day, 14-day, and 30-day extremes (`rolling_min_*`, `rolling_max_*`).
3. **Calendar & Cyclical Coordinates (12 features):**
   * `day_of_week`, `is_weekend`, `month`, `day_of_month`, `day_of_year`.
   * Continuous cyclical coordinates: `sin_dow`, `cos_dow`, `sin_month`, `cos_month`, `sin_doy`, `cos_doy`.
   * `is_holiday`: Deterministic US federal holiday indicator.
4. **Spatial Context & Interactions (3 features):**
   * `ratio_lag1_to_rolling7`: Momentum ratio comparing yesterday to the 7-day moving average.
   * `citywide_lag_1`: Citywide aggregate crime volume on day $t-1$.
   * `citywide_rolling_7`: Citywide 7-day moving average.
5. **Spatial Identifier (1 feature):**
   * `community_area`: Integer identifier (1 to 77) allowing tree splits to partition neighborhoods without sparse one-hot fragmentation.

### Features Strictly Excluded:
* `date`: Calendar date identifier (excluded to prevent index look-up leakage).
* `crime_count`: Target variable.
* **Quarantined Post-Event Attributes:** `ID`, `Case Number`, `Date`, `Block`, `IUCR`, `Primary Type`, `Description`, `Location Description`, `Arrest`, `Domestic`, `Beat`, `District`, `Ward`, `Latitude`, `Longitude` were quarantined during preprocessing and not included.

---

## 3. Hyperparameter Tuning Grid & Results

Six candidate hyperparameter configurations were evaluated on the 28,182 rows of the **2024 validation dataset**:

| Config ID | `n_estimators` | `max_depth` | `min_samples_leaf` | `max_features` | 2024 Val MAE (Raw) | 2024 Val MAE (Clipped $\ge 0$) | 2024 Val RMSE (Clipped) | Negative Predictions |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6** | **150** | **12** | **10** | **'sqrt'** | **2.5140** | **2.5140** | **3.4717** | **0 (0.00%)** |
| 4 | 100 | 12 | 10 | 'sqrt' | 2.5153 | 2.5153 | 3.4739 | 0 (0.00%) |
| 5 | 100 | 14 | 10 | 'sqrt' | 2.5181 | 2.5181 | 3.4760 | 0 (0.00%) |
| 3 | 100 | 12 | 20 | 'sqrt' | 2.5187 | 2.5187 | 3.4817 | 0 (0.00%) |
| 2 | 100 | 10 | 20 | 'sqrt' | 2.5202 | 2.5202 | 3.4813 | 0 (0.00%) |
| 1 | 100 | 8 | 20 | 'sqrt' | 2.5271 | 2.5271 | 3.4950 | 0 (0.00%) |

### Observations:
* **Selected Configuration:** Config 6 (`n_estimators=150, max_depth=12, min_samples_leaf=10, max_features='sqrt'`) achieved the lowest validation MAE of **2.5140** and lowest validation RMSE of **3.4717**.
* **Depth & Overfitting:** Setting `max_depth=14` began slightly degrading MAE (2.5181 vs 2.5153 at depth 12), demonstrating that depth 12 provides the optimal balance against overfitting on high-dimensional temporal features.
* **Feature Subsampling:** `max_features='sqrt'` (evaluating $\approx \sqrt{34} \approx 6$ candidate features per split) decorrelates individual trees effectively and significantly speeds up training compared to full feature scans.

---

## 4. Non-Negative Prediction Analysis

* **Theoretical Property:** In regression decision trees, prediction for any query instance is computed as the sample mean of training targets assigned to that leaf node. Because daily crime counts in the training set are strictly non-negative integers ($y \ge 0$), leaf means can never be mathematically negative.
* **Empirical Validation:** Across all 28,182 validation observations, exactly **0 negative predictions** were generated by any configuration.
* **Clipping Effect:** Applying $\hat{y} = \max(0.0, \hat{y})$ resulted in **zero modified values**, meaning raw and clipped MAE/RMSE are identical ($2.5140$).

---

## 5. Model Comparison: Baselines vs Ridge vs Random Forest

| Model | Model Class | 2024 Val MAE | 2024 Val RMSE | MAE Delta vs Baseline | % MAE Reduction vs Baseline | Validation Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (150 trees, depth 12)** | Ensemble Decision Trees | **2.5140** | **3.4717** | **-0.8452** | **-25.16%** | **1** |
| Ridge Regression ($\alpha=0.01$) | Regularized Linear | 2.5180 | 3.4933 | -0.8412 | -25.04% | 2 |
| Previous-Day Baseline (`lag_1`) | Naive Persistence | 3.3592 | 4.6725 | 0.0000 | 0.00% | 3 |
| Previous-Week Baseline (`lag_7`)| Naive Persistence | 3.4502 | 4.8111 | +0.0910 | +2.71% | 4 |

---

## 6. Feature Importances (MDI)

The Gini-based Mean Decrease in Impurity (MDI) highlights the dominant predictive signals in the fitted ensemble:

| Rank | Feature | Importance Share | Category |
| :---: | :--- | :---: | :--- |
| 1 | `rolling_mean_30` | 18.13% | 30-day baseline crime intensity |
| 2 | `rolling_mean_14` | 15.68% | 14-day moving average |
| 3 | `rolling_mean_7` | 11.77% | Short-term weekly moving average |
| 4 | `rolling_min_30` | 7.16% | Monthly floor baseline |
| 5 | `rolling_max_14` | 6.74% | Bi-weekly surge indicator |
| 6 | `rolling_min_14` | 6.65% | Bi-weekly floor indicator |
| 7 | `rolling_max_30` | 5.56% | Monthly peak indicator |
| 8 | `rolling_max_7` | 5.48% | Weekly peak indicator |
| 9 | `rolling_std_30` | 4.12% | Monthly neighborhood volatility |
| 10 | `rolling_min_7` | 3.56% | Weekly floor indicator |

**Key Insight:** Rolling statistics represent **over 80%** of total feature importance. Neighborhood crime forecasting is primarily anchored by multi-week baseline levels, with daily lags serving as high-frequency adjustments.

---

## 7. Runtime, Limitations & Operational Trade-offs

### Runtime:
* Training time for 150 trees on 110,187 observations using multi-threading (`n_jobs=-1`) is approximately **3.5 seconds**.
* Validation inference across all 28,182 rows requires **$\approx 0.15$ seconds**.

### Is Random Forest Worth Keeping Over Ridge Regression?
* **Statistical Performance:** Random Forest is technically the top-performing model, with a validation MAE of **2.5140** (vs Ridge **2.5180**) and validation RMSE of **3.4717** (vs Ridge **3.4933**).
* **Magnitude of Gain:** The outperformance is relatively small:
  * MAE reduction: $0.0040$ crimes/day ($0.16\%$ relative improvement).
  * RMSE reduction: $0.0216$ crimes/day ($0.62\%$ relative improvement).
* **Operational Overhead:** Ridge Regression consists of a single linear weight vector (110 coefficients) with virtually instant inference and microscopic serialization size. Random Forest requires serializing an ensemble of 150 trees (several megabytes).
* **Recommendation:**
  * **Keep Random Forest** as the current top-performing model in the model leaderboard.
  * Its non-linear structure and Gini importances provide rich inputs for explainability (SHAP analysis in Sprint 5).
  * Both Ridge and Random Forest remain valuable: Ridge as a low-latency baseline and Random Forest as a non-linear benchmark.

---

## 8. Test Set Isolation Guarantee

The 2025 Test Split (`data/processed/test_features.parquet`, 28,105 rows) remained strictly isolated and unaccessed during model tuning, fitting, and evaluation.
