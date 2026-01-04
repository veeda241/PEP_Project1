# Evaluation Metrics Guide

## Complete Reference for Model Evaluation Metrics

---

# Table of Contents

1. [Regression Metrics](#regression-metrics)
2. [Classification Metrics](#classification-metrics)
3. [Time Series Metrics](#time-series-metrics)
4. [Choosing the Right Metric](#choosing-the-right-metric)

---

# Regression Metrics

## 1. R² Score (Coefficient of Determination)

### Formula
```
R² = 1 - (SS_res / SS_tot)
   = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

### Interpretation
| R² Value | Interpretation |
|----------|----------------|
| 1.0 | Perfect prediction |
| 0.9+ | Excellent fit |
| 0.7-0.9 | Good fit |
| 0.5-0.7 | Moderate fit |
| < 0.5 | Weak fit |
| < 0 | Worse than mean |

### Pros & Cons
| Pros | Cons |
|------|------|
| Scale-independent (0-1) | Can be negative |
| Easy to interpret | Increases with more features |
| Standard metric | Doesn't indicate prediction error |

### Adjusted R²
```
R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
```
Penalizes additional features.

---

## 2. Mean Squared Error (MSE)

### Formula
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

### Properties
- Always positive
- Heavily penalizes large errors
- Units are squared (less interpretable)
- Used as loss function in training

---

## 3. Root Mean Squared Error (RMSE)

### Formula
```
RMSE = √MSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

### Interpretation
- Same units as target variable
- Average prediction error magnitude
- More sensitive to outliers than MAE

### Example
If predicting house prices and RMSE = $30,000:
- On average, predictions are off by ~$30,000

---

## 4. Mean Absolute Error (MAE)

### Formula
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

### Comparison with RMSE
| Aspect | MAE | RMSE |
|--------|-----|------|
| Large error sensitivity | Equal weight | Higher weight |
| Outlier robustness | More robust | Less robust |
| Interpretation | Avg absolute error | Avg squared magnitude |

---

## 5. Mean Absolute Percentage Error (MAPE)

### Formula
```
MAPE = (100/n) × Σ|yᵢ - ŷᵢ| / |yᵢ|
```

### Interpretation
- Percentage error
- Scale-independent
- Undefined when y = 0

---

# Classification Metrics

## 1. Confusion Matrix

### Structure
```
                    Predicted
                |  Positive  |  Negative  |
    Actual Pos  |     TP     |     FN     |
    Actual Neg  |     FP     |     TN     |
```

### Components
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Type I Error (incorrectly predicted positive)
- **FN (False Negative)**: Type II Error (incorrectly predicted negative)

---

## 2. Accuracy

### Formula
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### When to Use
- Balanced classes
- All errors equally important

### When NOT to Use
- Imbalanced classes (e.g., 95% one class)
- Different error costs

---

## 3. Precision

### Formula
```
Precision = TP / (TP + FP)
```

### Question Answered
"Of all positive predictions, how many were correct?"

### When to Prioritize
- Cost of false positives is high
- Example: Spam detection (don't want to miss real emails)

---

## 4. Recall (Sensitivity, True Positive Rate)

### Formula
```
Recall = TP / (TP + FN)
```

### Question Answered
"Of all actual positives, how many did we catch?"

### When to Prioritize
- Cost of false negatives is high
- Example: Disease detection (don't want to miss sick patients)

---

## 5. F1 Score

### Formula
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Properties
- Harmonic mean of precision and recall
- Ranges from 0 to 1
- Balanced metric when both precision and recall matter

### Weighted F-beta Score
```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```
- β > 1: Weight recall more
- β < 1: Weight precision more

---

## 6. ROC Curve and AUC

### ROC Curve
Plot of True Positive Rate vs False Positive Rate at various thresholds.

```
TPR = TP / (TP + FN)  [Y-axis]
FPR = FP / (FP + TN)  [X-axis]
```

### AUC (Area Under Curve)
| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.9-1.0 | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.5-0.7 | Poor |
| 0.5 | Random classifier |
| < 0.5 | Worse than random |

### When to Use
- Compare models regardless of threshold
- Imbalanced classes
- Need probability rankings

---

## 7. Precision-Recall Curve

### When Preferred Over ROC
- Highly imbalanced classes
- Focus on positive class performance

---

## 8. Specificity (True Negative Rate)

### Formula
```
Specificity = TN / (TN + FP)
```

### Use Case
- When correctly identifying negatives is important
- Example: Confirming healthy patients are healthy

---

## 9. Log Loss (Cross-Entropy)

### Formula
```
Log Loss = -(1/n) × Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

### Properties
- Penalizes confident wrong predictions heavily
- Used for probability outputs
- Lower is better

---

# Time Series Metrics

## 1. RMSE (Root Mean Squared Error)
Same as regression. Most common for time series.

## 2. MAE (Mean Absolute Error)
Same as regression. Robust to outliers.

## 3. MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) × Σ|Actualᵢ - Forecastᵢ| / |Actualᵢ|
```

### Advantages
- Scale-independent (percentage)
- Easy to interpret

### Disadvantages
- Undefined when actual = 0
- Asymmetric (penalizes over-predictions less)

## 4. Symmetric MAPE (sMAPE)
```
sMAPE = (100/n) × Σ|Actualᵢ - Forecastᵢ| / ((|Actualᵢ| + |Forecastᵢ|)/2)
```

## 5. Mean Absolute Scaled Error (MASE)
```
MASE = MAE / MAE_naive
```
Where MAE_naive is from naive forecast (previous value).

### Interpretation
- MASE < 1: Better than naive
- MASE > 1: Worse than naive

---

# Choosing the Right Metric

## Regression

| Situation | Recommended Metric |
|-----------|-------------------|
| General comparison | R², RMSE |
| Outliers present | MAE |
| Business context needs % | MAPE |
| Different scales | R², MAPE |

## Classification

| Situation | Recommended Metric |
|-----------|-------------------|
| Balanced classes | Accuracy, F1 |
| Imbalanced classes | F1, ROC-AUC, PR-AUC |
| FP costly | Precision |
| FN costly | Recall |
| Probability ranking | ROC-AUC, Log Loss |

## Time Series

| Situation | Recommended Metric |
|-----------|-------------------|
| General forecasting | RMSE, MAE |
| Scale-independent comparison | MAPE, MASE |
| Intermittent demand | MASE (avoids division by zero) |

---

**Document Version:** 1.0
**Last Updated:** January 2026
