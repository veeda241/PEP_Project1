# Algorithm Reference Guide

## Complete Guide to Machine Learning Algorithms Used in This Project

---

# Table of Contents

1. [Regression Algorithms](#regression-algorithms)
2. [Classification Algorithms](#classification-algorithms)
3. [Time Series Algorithms](#time-series-algorithms)
4. [Algorithm Selection Guide](#algorithm-selection-guide)

---

# Regression Algorithms

## 1. Linear Regression

### Overview
The foundational regression algorithm that fits a linear relationship between features and target.

### Mathematical Model
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

### Cost Function (Ordinary Least Squares)
```
J(β) = Σᵢ(yᵢ - ŷᵢ)² = Σᵢ(yᵢ - β₀ - Σⱼβⱼxᵢⱼ)²
```

### Closed-Form Solution
```
β = (X'X)⁻¹X'y
```

### Assumptions
1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

### Pros & Cons
| Pros | Cons |
|------|------|
| Simple and interpretable | Assumes linear relationship |
| Fast training | Sensitive to outliers |
| No hyperparameters | Cannot capture complex patterns |
| Provides coefficient interpretation | May underfit complex data |

---

## 2. Ridge Regression (L2 Regularization)

### Overview
Adds L2 penalty to prevent overfitting by shrinking coefficients.

### Cost Function
```
J(β) = Σᵢ(yᵢ - ŷᵢ)² + α × Σⱼβⱼ²
```

### Effect of Alpha (α)
- α = 0: Equivalent to Linear Regression
- α → ∞: All coefficients → 0
- Optimal α: Found via cross-validation

### When to Use
- Multicollinearity present
- Many features relative to samples
- Want to prevent overfitting
- All features potentially relevant

---

## 3. Lasso Regression (L1 Regularization)

### Overview
Uses L1 penalty which can shrink coefficients to exactly zero.

### Cost Function
```
J(β) = Σᵢ(yᵢ - ŷᵢ)² + α × Σⱼ|βⱼ|
```

### Feature Selection Property
- L1 penalty can set coefficients exactly to 0
- Automatically performs feature selection
- Produces sparse models

### When to Use
- Many irrelevant features
- Need feature selection
- Want interpretable sparse model
- Suspect many coefficients should be zero

---

## 4. ElasticNet

### Overview
Combines L1 and L2 penalties for balanced regularization.

### Cost Function
```
J(β) = Σᵢ(yᵢ - ŷᵢ)² + α × (ρ × Σⱼ|βⱼ| + (1-ρ) × Σⱼβⱼ²)
```

### Hyperparameters
- **alpha (α)**: Overall regularization strength
- **l1_ratio (ρ)**: Balance between L1 and L2 (0=Ridge, 1=Lasso)

---

## 5. Decision Tree Regressor

### Overview
Non-linear model that recursively splits data based on feature thresholds.

### Splitting Criteria
```
MSE = (1/n) × Σᵢ(yᵢ - ȳ)²
```
Splits minimize weighted MSE of child nodes.

### Key Hyperparameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| max_depth | Maximum tree depth | 3-20, None |
| min_samples_split | Min samples to split | 2-20 |
| min_samples_leaf | Min samples in leaf | 1-10 |
| max_features | Features per split | 'sqrt', 'log2', int |

---

## 6. Random Forest Regressor

### Overview
Ensemble of decision trees using bagging (Bootstrap AGGregatING).

### Algorithm
```
1. Create B bootstrap samples from training data
2. For each sample, grow a decision tree:
   - At each node, consider random subset of features
   - Split using best feature from subset
3. Average predictions from all trees
```

### Why It Works
- Reduces variance through averaging
- Decorrelates trees via random features
- Robust to overfitting

### Key Hyperparameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| n_estimators | Number of trees | 100-500 |
| max_depth | Maximum tree depth | None, 10-30 |
| min_samples_split | Min samples to split | 2-10 |
| max_features | Features per split | 'sqrt', 0.3-0.7 |

---

## 7. Gradient Boosting Regressor

### Overview
Sequential ensemble where each tree corrects errors of previous trees.

### Algorithm
```
1. Initialize: F₀(x) = mean(y)
2. For m = 1 to M:
   a. Compute residuals: rᵢ = yᵢ - Fₘ₋₁(xᵢ)
   b. Fit tree hₘ to residuals
   c. Update: Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
3. Final: F(x) = F₀(x) + η × Σₘhₘ(x)
```

### Key Hyperparameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| n_estimators | Boosting stages | 100-500 |
| learning_rate | Shrinkage factor | 0.01-0.3 |
| max_depth | Tree depth (keep shallow) | 3-8 |
| subsample | Fraction for each tree | 0.5-1.0 |

---

# Classification Algorithms

## 1. Logistic Regression

### Overview
Despite its name, used for binary classification via sigmoid function.

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
```

### Cost Function (Log Loss)
```
J(β) = -Σᵢ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

### Decision Boundary
- Linear decision boundary in feature space
- Threshold typically at 0.5 probability

---

## 2. Decision Tree Classifier

### Splitting Criteria

**Gini Impurity:**
```
Gini = 1 - Σₖ pₖ²
```

**Entropy (Information Gain):**
```
Entropy = -Σₖ pₖ log₂(pₖ)
Information Gain = Entropy(parent) - Weighted_Entropy(children)
```

---

## 3. Random Forest Classifier

### Voting Mechanism
- Each tree makes a prediction
- Final class = majority vote
- Can also get probabilities by averaging

---

## 4. Gradient Boosting Classifier

### For Binary Classification
- Uses log loss as objective
- Trees predict log-odds, not classes
- Final prediction via sigmoid

---

## 5. Support Vector Machine (SVM)

### Overview
Finds hyperplane that maximizes margin between classes.

### Objective
```
minimize: (1/2)||w||² + C × Σᵢmax(0, 1 - yᵢ(w·xᵢ + b))
```

### Kernels
| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | K(x,y) = x·y | Linearly separable |
| RBF | K(x,y) = exp(-γ\|\|x-y\|\|²) | Non-linear, general |
| Polynomial | K(x,y) = (γx·y + r)^d | Polynomial boundary |

---

## 6. K-Nearest Neighbors (KNN)

### Algorithm
```
1. Compute distance to all training points
2. Find K nearest neighbors
3. Majority vote (classification) or average (regression)
```

### Distance Metrics
- **Euclidean**: √Σᵢ(xᵢ - yᵢ)²
- **Manhattan**: Σᵢ|xᵢ - yᵢ|
- **Minkowski**: (Σᵢ|xᵢ - yᵢ|^p)^(1/p)

---

## 7. Naive Bayes

### Bayes' Theorem
```
P(y|X) = P(X|y) × P(y) / P(X)
```

### "Naive" Assumption
Features are conditionally independent given class:
```
P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

---

# Time Series Algorithms

## 1. Simple Moving Average

### Formula
```
MA(t) = (Y(t-1) + Y(t-2) + ... + Y(t-n)) / n
```

### Use
- Baseline model
- Smoothing for visualization
- Removes short-term fluctuations

---

## 2. Exponential Smoothing

### Simple Exponential Smoothing (SES)
```
Ŷ(t+1) = α × Y(t) + (1-α) × Ŷ(t)
```
α = smoothing parameter (0 < α < 1)

### Holt's Linear Trend
Adds trend component:
```
Level: l(t) = α × Y(t) + (1-α) × (l(t-1) + b(t-1))
Trend: b(t) = β × (l(t) - l(t-1)) + (1-β) × b(t-1)
Forecast: Ŷ(t+h) = l(t) + h × b(t)
```

### Holt-Winters (Triple)
Adds seasonality:
```
Level: l(t) = α × (Y(t) - s(t-m)) + (1-α) × (l(t-1) + b(t-1))
Trend: b(t) = β × (l(t) - l(t-1)) + (1-β) × b(t-1)
Seasonal: s(t) = γ × (Y(t) - l(t)) + (1-γ) × s(t-m)
Forecast: Ŷ(t+h) = l(t) + h × b(t) + s(t+h-m)
```

---

## 3. ARIMA

### Components
- **AR(p)**: AutoRegressive - uses past values
- **I(d)**: Integrated - differencing order
- **MA(q)**: Moving Average - uses past errors

### Model
```
(1 - Σᵢφᵢ Lⁱ)(1-L)^d Y(t) = (1 + Σⱼθⱼ Lʲ) ε(t)
```

### Parameter Selection
- **p**: From PACF (Partial ACF) plot
- **d**: From stationarity tests (ADF)
- **q**: From ACF (Autocorrelation) plot

---

## 4. SARIMA

### Extension for Seasonality
SARIMA(p,d,q)(P,D,Q,m)

- Regular: (p, d, q)
- Seasonal: (P, D, Q, m)
- m = seasonal period (7 for weekly, 12 for monthly)

---

# Algorithm Selection Guide

## By Problem Type

| Problem | Best Algorithms |
|---------|-----------------|
| Linear relationship | Linear, Ridge, Lasso |
| Non-linear, many features | Random Forest, Gradient Boosting |
| Small dataset | KNN, SVM |
| Need interpretability | Decision Tree, Logistic Regression, Lasso |
| Time series with trend | ARIMA, Holt-Winters |
| Time series with seasonality | SARIMA, Holt-Winters |

## By Data Characteristics

| Characteristic | Recommended Approach |
|----------------|---------------------|
| Many features | Lasso, Random Forest |
| Multicollinearity | Ridge, PCA + any model |
| Imbalanced classes | SMOTE + any classifier, adjust threshold |
| Missing values | Tree-based models |
| Outliers present | Tree-based, MAE loss |

---

**Document Version:** 1.0
**Last Updated:** January 2026
