# Sprint 3, Step 2: Ridge Regression Model Evaluation Report

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Model:** Regularized Linear Model (`scikit-learn.linear_model.Ridge`)  
**Training Split:** 2020–2023 (`data/processed/train_features.parquet`, 110,187 observations)  
**Validation Split:** 2024 (`data/processed/val_features.parquet`, 28,182 observations)  
**Primary Target:** `crime_count` (reported daily crime incidents per Community Area)  

---

## 1. Executive Summary

In Step 2 of Sprint 3, a **Ridge Regression** model was trained and evaluated to establish our first machine learning benchmark. 

The model was implemented as a strict scikit-learn `Pipeline`:
* Numerical features were standardized using `StandardScaler` fitted **only on 2020–2023 training data**.
* Spatial locations were one-hot encoded using `OneHotEncoder` across all 77 Chicago Community Areas.
* Regularization strength ($\alpha$) was tuned across six candidate values using **2024 validation MAE**.

**Key Finding:** Ridge Regression achieved a validation MAE of **2.5180** and an RMSE of **3.4933**, outperforming the Previous-Day baseline (MAE 3.3592) by **25.04%**. The 2025 test dataset remained completely untouched.

---

## 2. Feature Selection & Input Partitioning

The model ingests a total of **110 transformed input dimensions** ($33 \text{ numerical features} + 77 \text{ one-hot spatial indicators}$):

### Features Used:
1. **Historical Autoregressive Lags (6 features):**
   * `lag_1` (yesterday's count), `lag_2`, `lag_3`, `lag_7` (same day last week), `lag_14`, `lag_28`.
2. **Shifted Rolling Window Statistics (12 features):**
   * 7-day, 14-day, and 30-day moving averages (`rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`).
   * 7-day, 14-day, and 30-day volatility (`rolling_std_7`, `rolling_std_14`, `rolling_std_30`).
   * 7-day, 14-day, and 30-day extremes (`rolling_min_*`, `rolling_max_*`).
3. **Calendar & Cyclical Trigonometric Encodings (12 features):**
   * `day_of_week`, `is_weekend`, `month`, `day_of_month`, `day_of_year`.
   * Continuous cyclical coordinates: `sin_dow`, `cos_dow`, `sin_month`, `cos_month`, `sin_doy`, `cos_doy`.
   * `is_holiday`: Deterministic US federal holiday indicator.
4. **Spatial Context & Interactions (3 features):**
   * `ratio_lag1_to_rolling7`: Momentum ratio comparing yesterday to the 7-day rolling mean.
   * `citywide_lag_1`: Citywide aggregate crime volume on day $t-1$.
   * `citywide_rolling_7`: Citywide 7-day moving average.
5. **Spatial Categorical Encoding (77 binary dummy features):**
   * `community_area` was transformed via `OneHotEncoder(categories=[1..77])` to allow area-specific baseline intercepts without imposing artificial ordinality.

### Features Strictly Excluded:
* `date`: Calendar date identifier (excluded to prevent index look-up leakage).
* `crime_count`: Target variable.
* **Quarantined Post-Event Leakage Attributes:** `Arrest`, `IUCR`, `Primary Type`, `Description`, `FBI Code`, `Updated On`, `Case Number`, and `ID` were quarantined during preprocessing and not included.

---

## 3. Ridge Regularization Hyperparameter Tuning ($\alpha$)

Six candidate regularization values were evaluated on the 28,182 rows of the **2024 validation dataset**:

| Regularization $\alpha$ | Validation MAE (Raw) | Validation MAE (Clipped $\ge 0$) | Validation RMSE (Clipped) | Negative Predictions ($< 0$) |
| :---: | :---: | :---: | :---: | :---: |
| **0.01** | **2.5181** | **2.5180** | **3.4933** | **18 (0.06%)** |
| **0.10** | 2.5182 | 2.5181 | 3.4933 | 18 (0.06%) |
| **1.00** | 2.5190 | 2.5189 | 3.4932 | 16 (0.06%) |
| **10.00** | 2.5205 | 2.5204 | 3.4932 | 15 (0.05%) |
| **100.00** | 2.5246 | 2.5245 | 3.4970 | 16 (0.06%) |
| **1000.00** | 2.5334 | 2.5334 | 3.5074 | 7 (0.02%) |

* **Selected Parameter:** $\alpha = 0.01$ (achieves the minimum validation MAE of **2.5180**).
* **Sensitivity Analysis:** Performance is exceptionally stable between $\alpha = 0.01$ and $\alpha = 10.0$ (MAE varies by less than $0.0024$), indicating that the feature matrix is well-conditioned and collinearity is properly managed by L2 shrinkage.

---

## 4. Non-Negative Prediction Clipping Analysis

* **The Issue:** Linear models can mathematically produce negative predictions when multiple negative coefficients align with low input values.
* **Empirical Result:** Out of 28,182 predictions on 2024 validation data, exactly **18 predictions (0.06%)** were slightly below zero (the most negative predicted count was $-0.19$ crimes/day, occurring in low-crime outer residential areas).
* **Clipping Effect:** Applying $\hat{y} = \max(0.0, \text{prediction})$ slightly improved validation MAE from **2.5181** to **2.5180**. It mathematically enforces physical realism without distorting the model's predictive distribution.

---

## 5. Comparison Against Naive Baselines

| Model | Model Class | 2024 Val MAE | 2024 Val RMSE | MAE Reduction vs Baseline | % Error Reduction | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression ($\alpha=0.01$)** | L2 Linear Model | **2.5180** | **3.4933** | **-0.8412** | **-25.04%** | **1** |
| Previous-Day Baseline (`lag_1`) | Naive Persistence | 3.3592 | 4.6725 | 0.0000 | 0.00% | 2 |
| Previous-Week Baseline (`lag_7`)| Naive Persistence | 3.4502 | 4.8111 | +0.0910 | +2.71% | 3 |

### Interpretation:
1. **Significant Outperformance:** Ridge Regression beats the Previous-Day baseline by **$0.8412$ crimes per day on average** across all 77 Chicago Community Areas.
2. **Variance Reduction:** RMSE decreased from $4.6725$ to $3.4933$ (a **$25.24\%$ reduction**), demonstrating that Ridge significantly reduces severe forecasting errors during high-crime days.
3. **Linear Benchmark Established:** Any subsequent non-linear model (Random Forest, Gradient Boosting) must surpass an MAE of **2.5180** to justify its computational complexity.

---

## 6. Model Limitations & Observations

1. **Linear Additive Assumption:** Ridge cannot capture non-linear feature interactions (such as the interaction between warm summer months and high-density nightlife areas).
2. **Fixed Area Intercepts:** One-hot encoding creates separate base offsets for each area, but assumes that the effect of lagged crimes is uniform across all neighborhoods.
3. **Test Split Isolation:** The 2025 Test Split (`data/processed/test_features.parquet`, 28,105 rows) remains completely isolated and was not used for model selection or tuning.
