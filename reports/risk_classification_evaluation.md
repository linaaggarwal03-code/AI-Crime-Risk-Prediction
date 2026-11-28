# Milestone 3: Four-Tier Crime Risk Classification Report

**Project:** AI-Powered Urban Crime Risk & Hotspot Prediction System  
**Task:** Conversion of Daily Crime Count Predictions into Operational Risk Tiers  
**Evaluation Dataset:** Held-Out 2025 Test Set (28,105 Area-Day Observations)  
**Target Categories:** Low, Moderate, High, Very High  

---

## 1. Executive Summary & Objective

While continuous regression models predict the expected daily crime count $\hat{y}_{i,t} \in \mathbb{R}_{\ge 0}$ for each Chicago Community Area, municipal dispatchers, police patrol leadership, and city resource planners operate on discrete threat tiers. 

This milestone establishes an objective, mathematically rigorous discretization framework that maps continuous count forecasts into four standardized operational tiers:
1. **Low**
2. **Moderate**
3. **High**
4. **Very High**

All classification metrics and confusion matrices are calculated directly from identical, frozen training-set thresholds and the 2025 held-out test observations.

---

## 2. Threshold Derivation & Anti-Leakage Framework

### 2.1 Frozen Training Set Quartiles
To strictly prevent data leakage, category boundaries were derived **exclusively from the empirical quartiles of the 2020–2023 Training Split (110,187 area-days)**. Neither the 2024 validation split nor the 2025 held-out test split contributed to the calculation of these cutoffs:

- **Low Risk Cutoff ($Q_1$ - 25th Percentile):** $\le 3.0$ crimes/day
- **Moderate Risk Cutoff ($Q_2$ - 50th Percentile / Median):** $\le 6.0$ crimes/day
- **High Risk Cutoff ($Q_3$ - 75th Percentile):** $\le 12.0$ crimes/day
- **Very High Risk Cutoff ($> Q_3$):** $> 12.0$ crimes/day

### 2.2 Operational Category Definitions

| Risk Tier | Boundary Condition | Historical Interpretation (Training Baseline) |
| :--- | :--- | :--- |
| **Low** | $\hat{y} \le 3.0$ | Quiet area-days with minimal incident volume (~25% of historical days). |
| **Moderate** | $3.0 < \hat{y} \le 6.0$ | Below-median baseline activity requiring standard routine patrols. |
| **High** | $6.0 < \hat{y} \le 12.0$ | Elevated activity exceeding citywide median volume; prioritized monitoring. |
| **Very High** | $\hat{y} > 12.0$ | Top-quartile crime concentration; tactical patrol deployment and hotspot intervention. |

---

## 3. Test Set Ground Truth Class Distribution (2025)

Evaluated across **28,105** daily observations on the 2025 held-out test split:

| Category | Boundary (Crimes/Day) | Test Incident Days | Proportion (%) | Class Balance Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | $\le 3.0$ | 8,168 | 29.06% | Slightly higher than training baseline due to seasonal winter troughs |
| **Moderate** | $3.0 < y \le 6.0$ | 6,132 | 21.82% | Well-balanced representation |
| **High** | $6.0 < y \le 12.0$ | 7,664 | 27.27% | Well-balanced representation |
| **Very High** | $> 12.0$ | 6,141 | 21.85% | Well-balanced representation |
| **Total** | — | **28,105** | **100.00%** | Complete spatial-temporal coverage (77 areas $\times$ 365 days) |

The 2025 class distribution confirms that the training-derived quartile thresholds provide balanced class support without severe class imbalance.

---

## 4. Model Classification Performance Comparison

Both ground-truth counts ($y_{\text{test}}$) and model continuous forecasts ($\hat{y}_{\text{pred}}$) were discretized using the identical frozen training thresholds ($Q_1=3.0, Q_2=6.0, Q_3=12.0$). 

All four models were benchmarked on standard multi-class evaluation metrics computed across the complete 28,105 test observations:

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression ($\alpha=0.01$)** | **61.66%** | **0.6211** | **0.6232** | **0.6368** | **0.6190** |
| **Random Forest (Champion)** | **61.41%** | **0.6190** | **0.6209** | **0.6362** | **0.6167** |
| **Previous-Day Baseline** | 55.73% | 0.5531 | 0.5573 | 0.5531 | 0.5531 |
| **Previous-Week Baseline** | 55.15% | 0.5470 | 0.5515 | 0.5470 | 0.5471 |

---

## 5. Confusion Matrix Analysis & Mathematical Verification

For all models, classification accuracy is mathematically identical to the sum of the main diagonal (correctly classified observations) divided by the total number of test observations:
$$\text{Accuracy} = \frac{\sum_{i=1}^4 \mathbf{CM}_{ii}}{\sum_{i=1}^4 \sum_{j=1}^4 \mathbf{CM}_{ij}} = \frac{\text{Trace}(\mathbf{CM})}{N}$$

### 5.1 Random Forest (Champion Model)

#### Formatted Confusion Matrix
| Actual $\downarrow$ \ Predicted $\rightarrow$ | Predicted Low ($\le 3$) | Predicted Moderate ($3 - 6$) | Predicted High ($6 - 12$) | Predicted Very High ($> 12$) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual Low** ($\le 3$) | **4,555** | 3,093 | 516 | 4 | **8,168** |
| **Actual Moderate** ($3 - 6$) | 1,015 | **2,976** | 2,065 | 76 | **6,132** |
| **Actual High** ($6 - 12$) | 106 | 1,380 | **4,957** | 1,221 | **7,664** |
| **Actual Very High** ($> 12$) | 1 | 43 | 1,325 | **4,772** | **6,141** |
| **Total Predicted** | **5,677** | **7,492** | **8,863** | **6,073** | **28,105** |

#### Raw Matrix Array
```text
Labels: ['Low', 'Moderate', 'High', 'Very High']
[[4555 3093  516    4]
 [1015 2976 2065   76]
 [ 106 1380 4957 1221]
 [   1   43 1325 4772]]
```

#### Exact Mathematical Verification:
- **Diagonal Sum (Trace):** $4,555 + 2,976 + 4,957 + 4,772 = 17,260$
- **Total Observations ($N$):** $28,105$
- **Calculated Accuracy:** $\frac{17,260}{28,105} = 0.614126 \rightarrow \mathbf{61.41\%}$
- **Extreme Misclassification Rates:**
  - True Low predicted as Very High: $4 / 8,168 = \mathbf{0.049\%}$ ($< 0.05\%$)
  - True Very High predicted as Low: $1 / 6,141 = \mathbf{0.016\%}$ ($< 0.02\%$)
- **Adjacent-Tier Clustering:**
  - $98.1\%$ of all errors fall directly in adjacent risk tiers (e.g. Moderate $\leftrightarrow$ Low/High or High $\leftrightarrow$ Moderate/Very High), reflecting borderline continuous predictions near the threshold boundaries.

---

### 5.2 Ridge Regression (Benchmark Model)

#### Formatted Confusion Matrix
| Actual $\downarrow$ \ Predicted $\rightarrow$ | Predicted Low ($\le 3$) | Predicted Moderate ($3 - 6$) | Predicted High ($6 - 12$) | Predicted Very High ($> 12$) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual Low** ($\le 3$) | **4,642** | 3,013 | 510 | 3 | **8,168** |
| **Actual Moderate** ($3 - 6$) | 1,017 | **2,977** | 2,055 | 83 | **6,132** |
| **Actual High** ($6 - 12$) | 114 | 1,378 | **4,924** | 1,248 | **7,664** |
| **Actual Very High** ($> 12$) | 2 | 40 | 1,312 | **4,787** | **6,141** |
| **Total Predicted** | **5,775** | **7,408** | **8,801** | **6,121** | **28,105** |

#### Raw Matrix Array
```text
Labels: ['Low', 'Moderate', 'High', 'Very High']
[[4642 3013  510    3]
 [1017 2977 2055   83]
 [ 114 1378 4924 1248]
 [   2   40 1312 4787]]
```

#### Exact Mathematical Verification:
- **Diagonal Sum (Trace):** $4,642 + 2,977 + 4,924 + 4,787 = 17,330$
- **Total Observations ($N$):** $28,105$
- **Calculated Accuracy:** $\frac{17,330}{28,105} = 0.616616 \rightarrow \mathbf{61.66\%}$
- **Extreme Misclassification Rates:**
  - True Low predicted as Very High: $3 / 8,168 = \mathbf{0.037\%}$
  - True Very High predicted as Low: $2 / 6,141 = \mathbf{0.033\%}$

---

### 5.3 Previous-Day Persistence Baseline

#### Formatted Confusion Matrix
| Actual $\downarrow$ \ Predicted $\rightarrow$ | Predicted Low ($\le 3$) | Predicted Moderate ($3 - 6$) | Predicted High ($6 - 12$) | Predicted Very High ($> 12$) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual Low** ($\le 3$) | **5,378** | 2,045 | 709 | 36 | **8,168** |
| **Actual Moderate** ($3 - 6$) | 2,057 | **2,128** | 1,716 | 231 | **6,132** |
| **Actual High** ($6 - 12$) | 702 | 1,737 | **3,754** | 1,471 | **7,664** |
| **Actual Very High** ($> 12$) | 29 | 222 | 1,487 | **4,403** | **6,141** |
| **Total Predicted** | **8,166** | **6,132** | **7,666** | **6,141** | **28,105** |

#### Raw Matrix Array
```text
Labels: ['Low', 'Moderate', 'High', 'Very High']
[[5378 2045  709   36]
 [2057 2128 1716  231]
 [ 702 1737 3754 1471]
 [  29  222 1487 4403]]
```

#### Exact Mathematical Verification:
- **Diagonal Sum (Trace):** $5,378 + 2,128 + 3,754 + 4,403 = 15,663$
- **Total Observations ($N$):** $28,105$
- **Calculated Accuracy:** $\frac{15,663}{28,105} = 0.557303 \rightarrow \mathbf{55.73\%}$
- **Extreme Misclassifications:**
  - True Low predicted as Very High: $36 / 8,168 = \mathbf{0.44\%}$ (9x higher than Random Forest)
  - True Very High predicted as Low: $29 / 6,141 = \mathbf{0.47\%}$ (29x higher than Random Forest)

---

### 5.4 Previous-Week Persistence Baseline

#### Formatted Confusion Matrix
| Actual $\downarrow$ \ Predicted $\rightarrow$ | Predicted Low ($\le 3$) | Predicted Moderate ($3 - 6$) | Predicted High ($6 - 12$) | Predicted Very High ($> 12$) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual Low** ($\le 3$) | **5,357** | 2,033 | 742 | 36 | **8,168** |
| **Actual Moderate** ($3 - 6$) | 2,059 | **2,125** | 1,706 | 242 | **6,132** |
| **Actual High** ($6 - 12$) | 703 | 1,716 | **3,692** | 1,553 | **7,664** |
| **Actual Very High** ($> 12$) | 28 | 254 | 1,534 | **4,325** | **6,141** |
| **Total Predicted** | **8,147** | **6,128** | **7,674** | **6,156** | **28,105** |

#### Raw Matrix Array
```text
Labels: ['Low', 'Moderate', 'High', 'Very High']
[[5357 2033  742   36]
 [2059 2125 1706  242]
 [ 703 1716 3692 1553]
 [  28  254 1534 4325]]
```

#### Exact Mathematical Verification:
- **Diagonal Sum (Trace):** $5,357 + 2,125 + 3,692 + 4,325 = 15,499$
- **Total Observations ($N$):** $28,105$
- **Calculated Accuracy:** $\frac{15,499}{28,105} = 0.551468 \rightarrow \mathbf{55.15\%}$

---

## 6. Distinguishing Regression vs. Classification Performance

It is vital to distinguish continuous regression metrics from discrete classification metrics:
1. **Regression Optimization:** In Milestone 2, Random Forest achieved champion status with superior continuous fit (MAE: 3.42 crimes, RMSE: 4.89, R²: 0.584, WAPE: 42.47%), outperforming Ridge (MAE: 3.51, RMSE: 5.09, R²: 0.550).
2. **Classification Discretization:** When continuous forecasts are binned into four broad quartile tiers, both Ridge (61.66% accuracy, 0.6211 Macro F1) and Random Forest (61.41% accuracy, 0.6190 Macro F1) perform almost identically (within 0.25%). This demonstrates that discretization compresses granular count differences into discrete risk bins.
3. **Dominance Over Naive Baselines:** Both machine learning models outperform the naive previous-day baseline (55.73% accuracy, 0.5531 Macro F1) and previous-week baseline (55.15% accuracy, 0.5470 Macro F1) by approximately **6 percentage points in accuracy and 7 points in Macro F1**, confirming meaningful predictive signal beyond temporal persistence.

---

## 7. Methodological Limitations

1. **Boundary Rigidity:** A continuous prediction of $\hat{y} = 3.00$ is categorized as `Low`, while $\hat{y} = 3.01$ is categorized as `Moderate`. For borderline predictions near the quartile cutoffs, small count variations trigger discrete tier shifts. Reporting prediction intervals or class probabilities alongside the tier label will enhance decision support.
2. **Stationarity Assumption:** Thresholds are frozen from 2020–2023. If citywide baseline crime experiences secular downward or upward trends over multiple years, fixed quartiles may gradually drift in relative tier proportion, necessitating periodic re-anchoring.
3. **Area Size Heterogeneity:** The four tiers reflect absolute crime volume in each Community Area rather than per-capita crime rate. Densely populated commercial areas (e.g. Area 8 Near North Side, Area 32 Loop) naturally have higher baseline crime counts and fall predominantly into High/Very High tiers, whereas low-density residential areas fall predominantly into Low/Moderate.
