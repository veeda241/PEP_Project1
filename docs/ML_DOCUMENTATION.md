# Machine Learning Documentation

## Complete Technical Guide for ML Analytics Project

---

# Table of Contents

1. [Introduction to Machine Learning](#1-introduction-to-machine-learning)
2. [Regression Analysis](#2-regression-analysis)
3. [Classification Analysis](#3-classification-analysis)
4. [Time Series Analysis](#4-time-series-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Evaluation Metrics](#6-model-evaluation-metrics)
7. [Cross-Validation Techniques](#7-cross-validation-techniques)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Model Selection](#9-model-selection)
10. [Implementation Details](#10-implementation-details)

---

# 1. Introduction to Machine Learning

## 1.1 What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed. Instead of following static rules, ML algorithms improve their performance through experience.

### The ML Process

```
Data Collection → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Deployment
```

## 1.2 Types of Machine Learning

### 1.2.1 Supervised Learning
The algorithm learns from labeled training data to make predictions.

**Examples in this project:**
- Regression (predicting house prices)
- Classification (predicting customer churn)

### 1.2.2 Unsupervised Learning
The algorithm finds patterns in unlabeled data.

**Examples:** Clustering, dimensionality reduction

### 1.2.3 Time Series Analysis
Specialized techniques for sequential, time-dependent data.

**Example in this project:** Sales forecasting

## 1.3 The Machine Learning Pipeline

```python
# Typical ML Pipeline Structure
1. Load Data           → pd.read_csv('data.csv')
2. Explore Data        → df.describe(), df.info()
3. Clean Data          → Handle missing values, outliers
4. Feature Engineering → Create new features, encode categories
5. Split Data          → train_test_split(X, y, test_size=0.2)
6. Scale Features      → StandardScaler(), MinMaxScaler()
7. Train Models        → model.fit(X_train, y_train)
8. Evaluate            → model.score(), cross_val_score()
9. Tune Hyperparameters→ GridSearchCV, RandomizedSearchCV
10. Deploy             → Save model, create API
```

---

# 2. Regression Analysis

## 2.1 What is Regression?

Regression is a supervised learning technique used to predict **continuous numerical values**. The goal is to find the relationship between input features (X) and a target variable (y).

### Mathematical Foundation

The general form of linear regression:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- ŷ = Predicted value
- β₀ = Intercept (bias term)
- β₁...βₙ = Coefficients (weights)
- x₁...xₙ = Input features
- ε = Error term

## 2.2 Problem: House Price Prediction

### 2.2.1 Problem Statement
Given features about a house (size, location, amenities), predict its market price.

### 2.2.2 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| square_feet | Numeric | Total living area in sq ft |
| bedrooms | Numeric | Number of bedrooms |
| bathrooms | Numeric | Number of bathrooms |
| age_years | Numeric | Age of the house |
| distance_to_center_miles | Numeric | Distance to city center |
| has_pool | Binary | 1 if has pool, 0 otherwise |
| neighborhood_score | Numeric | Quality score (1-10) |

### 2.2.3 Target Variable
- **price**: House price in USD (continuous)

## 2.3 Regression Algorithms

### 2.3.1 Linear Regression

**Concept:** Fits a straight line through the data by minimizing the sum of squared residuals.

**Objective Function (Ordinary Least Squares):**
```
minimize: Σ(yᵢ - ŷᵢ)²
```

**Pros:**
- Simple and interpretable
- Fast training and prediction
- No hyperparameters to tune

**Cons:**
- Assumes linear relationship
- Sensitive to outliers
- Cannot capture complex patterns

**Implementation:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 2.3.2 Ridge Regression (L2 Regularization)

**Concept:** Adds a penalty term to prevent overfitting by shrinking coefficients.

**Objective Function:**
```
minimize: Σ(yᵢ - ŷᵢ)² + α × Σβⱼ²
```

Where α is the regularization strength.

**When to Use:**
- When you have multicollinearity
- When you want to prevent overfitting
- When all features are potentially relevant

**Hyperparameters:**
- `alpha`: Regularization strength (default: 1.0)

**Implementation:**
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### 2.3.3 Lasso Regression (L1 Regularization)

**Concept:** Uses L1 penalty which can shrink coefficients to exactly zero, performing feature selection.

**Objective Function:**
```
minimize: Σ(yᵢ - ŷᵢ)² + α × Σ|βⱼ|
```

**When to Use:**
- When you want automatic feature selection
- When you suspect many features are irrelevant
- When interpretability is important

**Implementation:**
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
```

### 2.3.4 ElasticNet (Combined L1/L2)

**Concept:** Combines both L1 and L2 penalties for balanced regularization.

**Objective Function:**
```
minimize: Σ(yᵢ - ŷᵢ)² + α × (ρ × Σ|βⱼ| + (1-ρ) × Σβⱼ²)
```

**Hyperparameters:**
- `alpha`: Overall regularization strength
- `l1_ratio`: Balance between L1 and L2 (0 to 1)

### 2.3.5 Random Forest Regressor

**Concept:** Ensemble of decision trees that averages predictions to reduce variance.

**How It Works:**
1. Create multiple bootstrap samples
2. Train a decision tree on each sample
3. Use random subset of features at each split
4. Average predictions from all trees

**Hyperparameters:**
- `n_estimators`: Number of trees (100-500)
- `max_depth`: Maximum tree depth (None for unlimited)
- `min_samples_split`: Minimum samples to split a node
- `max_features`: Features considered per split

**Pros:**
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Less prone to overfitting

**Cons:**
- Less interpretable
- Slower prediction time
- Memory intensive

**Implementation:**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```

### 2.3.6 Gradient Boosting Regressor

**Concept:** Builds trees sequentially, each correcting errors of the previous ones.

**How It Works:**
1. Start with an initial prediction (mean)
2. Calculate residuals (errors)
3. Fit a tree to the residuals
4. Update predictions with learning rate
5. Repeat until stopping criterion

**Hyperparameters:**
- `n_estimators`: Number of boosting stages
- `learning_rate`: Shrinkage factor (0.01-0.3)
- `max_depth`: Depth of individual trees (3-10)
- `subsample`: Fraction of samples per tree

**Implementation:**
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
```

### 2.3.7 Support Vector Regression (SVR)

**Concept:** Finds a hyperplane that best fits the data within a margin of tolerance.

**Kernel Options:**
- `linear`: For linear relationships
- `rbf`: For non-linear (most common)
- `poly`: For polynomial relationships

**Implementation:**
```python
from sklearn.svm import SVR

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_scaled, y_train)
```

### 2.3.8 K-Nearest Neighbors Regressor

**Concept:** Predicts based on the average of K nearest training examples.

**Implementation:**
```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=5, weights='distance')
model.fit(X_train_scaled, y_train)
```

## 2.4 Regression Metrics

### 2.4.1 R² Score (Coefficient of Determination)

**Formula:**
```
R² = 1 - (SS_res / SS_tot)
   = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

**Interpretation:**
- R² = 1: Perfect prediction
- R² = 0: Model predicts the mean
- R² < 0: Model worse than mean

**In Our Project:** Ridge Regression achieved R² = 0.9117 (91.17% variance explained)

### 2.4.2 Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
```

**Interpretation:**
- Same units as target variable
- Lower is better
- Penalizes large errors more heavily

**In Our Project:** RMSE = $30,129.76

### 2.4.3 Mean Absolute Error (MAE)

**Formula:**
```
MAE = Σ|yᵢ - ŷᵢ| / n
```

**Interpretation:**
- Average absolute prediction error
- More robust to outliers than RMSE
- Easy to interpret

---

# 3. Classification Analysis

## 3.1 What is Classification?

Classification is a supervised learning technique used to predict **categorical/discrete labels**. The goal is to assign input data to one of several predefined classes.

### Types of Classification

1. **Binary Classification:** Two classes (Yes/No, 0/1)
2. **Multi-class Classification:** More than two classes
3. **Multi-label Classification:** Multiple labels per sample

## 3.2 Problem: Customer Churn Prediction

### 3.2.1 Problem Statement
Given customer behavior and account information, predict whether a customer will leave (churn) the service.

### 3.2.2 Business Impact
- Customer acquisition costs 5-25x more than retention
- Early churn prediction enables proactive intervention
- Targeted retention campaigns reduce costs

### 3.2.3 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| tenure_months | Numeric | Months as customer |
| monthly_charges | Numeric | Monthly bill amount |
| total_charges | Numeric | Total amount paid |
| contract_type | Categorical | Month-to-month, 1yr, 2yr |
| payment_method | Categorical | Check, card, bank transfer |
| tech_support | Binary | Has tech support? |
| online_security | Binary | Has online security? |
| num_complaints | Numeric | Number of complaints filed |

### 3.2.4 Target Variable
- **churn**: 1 (churned) or 0 (retained)

## 3.3 Classification Algorithms

### 3.3.1 Logistic Regression

**Despite its name, logistic regression is used for classification!**

**Concept:** Uses the sigmoid function to predict probabilities.

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

Where z = β₀ + β₁x₁ + ... + βₙxₙ

**Decision Rule:**
- If P(y=1|X) >= 0.5 → Predict class 1
- If P(y=1|X) < 0.5 → Predict class 0

**Pros:**
- Probabilistic output
- Fast training
- Highly interpretable
- Works well for linearly separable data

**Cons:**
- Assumes linear decision boundary
- May underperform on complex problems

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

### 3.3.2 Decision Tree Classifier

**Concept:** Learns a series of if-then rules to classify data.

**How It Works:**
1. Select best feature to split on (using Gini impurity or entropy)
2. Create branches for each feature value
3. Repeat recursively until stopping criterion

**Gini Impurity:**
```
Gini = 1 - Σ(pᵢ)²
```

**Entropy:**
```
Entropy = -Σ(pᵢ × log₂(pᵢ))
```

**Pros:**
- Highly interpretable
- No scaling required
- Handles mixed data types

**Cons:**
- Prone to overfitting
- Unstable (small data changes = different trees)

### 3.3.3 Random Forest Classifier

**Concept:** Ensemble of decision trees using bagging.

**Key Differences from Decision Trees:**
- Uses bootstrap sampling (bagging)
- Random feature selection at splits
- Aggregates predictions (voting)

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
```

### 3.3.4 Gradient Boosting Classifier

**Concept:** Sequential ensemble where each tree corrects previous errors.

**Key Hyperparameters:**
- `learning_rate`: Step size (lower = more trees needed)
- `n_estimators`: Number of boosting stages
- `max_depth`: Usually shallow (3-8)

### 3.3.5 Support Vector Machine (SVM)

**Concept:** Finds the hyperplane that maximizes margin between classes.

**The Margin:** Distance between the hyperplane and nearest data points (support vectors).

**Kernel Trick:** Transforms data to higher dimensions for non-linear separation.

**Common Kernels:**
- `linear`: For linearly separable data
- `rbf` (Radial Basis Function): Most versatile
- `poly`: For polynomial boundaries

**Implementation:**
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
```

### 3.3.6 K-Nearest Neighbors (KNN)

**Concept:** Classifies based on majority vote of K nearest neighbors.

**Distance Metrics:**
- Euclidean: √Σ(xᵢ - yᵢ)²
- Manhattan: Σ|xᵢ - yᵢ|
- Minkowski: Generalization

**Pros:**
- Simple and intuitive
- No training phase
- Good for small datasets

**Cons:**
- Slow prediction for large datasets
- Sensitive to feature scaling
- Curse of dimensionality

### 3.3.7 Naive Bayes

**Concept:** Probabilistic classifier based on Bayes' theorem with independence assumption.

**Bayes' Theorem:**
```
P(y|X) = P(X|y) × P(y) / P(X)
```

**Types:**
- Gaussian NB: For continuous features
- Multinomial NB: For counts (text classification)
- Bernoulli NB: For binary features

### 3.3.8 AdaBoost (Adaptive Boosting)

**Concept:** Sequentially trains weak learners, giving more weight to misclassified samples.

**Implementation:**
```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 3.4 Classification Metrics

### 3.4.1 Confusion Matrix

```
                 Predicted
              |  Pos  |  Neg  |
Actual  Pos   |  TP   |  FN   |
        Neg   |  FP   |  TN   |
```

- **TP (True Positive):** Correctly predicted positive
- **TN (True Negative):** Correctly predicted negative
- **FP (False Positive):** Predicted positive, actually negative (Type I error)
- **FN (False Negative):** Predicted negative, actually positive (Type II error)

### 3.4.2 Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Limitation:** Misleading for imbalanced classes

### 3.4.3 Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Question it answers:** Of all positive predictions, how many were correct?

### 3.4.4 Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Question it answers:** Of all actual positives, how many did we catch?

### 3.4.5 F1 Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to use:** When you need balance between precision and recall.

**In Our Project:** Logistic Regression achieved F1 = 0.7254

### 3.4.6 ROC-AUC Score

**ROC Curve:** Plot of True Positive Rate vs False Positive Rate at various thresholds.

**AUC (Area Under Curve):**
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC < 0.5: Worse than random

---

# 4. Time Series Analysis

## 4.1 What is Time Series?

A time series is a sequence of data points collected over time at regular intervals. The goal is to understand patterns and forecast future values.

### Key Components

1. **Trend:** Long-term increase or decrease
2. **Seasonality:** Regular patterns that repeat (daily, weekly, yearly)
3. **Cyclical:** Long-term fluctuations (economic cycles)
4. **Noise:** Random, unpredictable variations

## 4.2 Problem: Sales Forecasting

### 4.2.1 Problem Statement
Given historical daily sales data, forecast future sales.

### 4.2.2 Data Characteristics
- 730 days (2 years) of daily sales
- Weekly seasonality (weekday vs weekend)
- Yearly seasonality (holidays, summer)
- Upward trend

## 4.3 Time Series Decomposition

**Additive Model:**
```
Y(t) = Trend(t) + Seasonal(t) + Residual(t)
```

**Multiplicative Model:**
```
Y(t) = Trend(t) × Seasonal(t) × Residual(t)
```

**Implementation:**
```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(series, model='additive', period=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

## 4.4 Time Series Algorithms

### 4.4.1 Moving Average

**Concept:** Average of the last N observations.

**Simple Moving Average (SMA):**
```
SMA(t) = (Y(t-1) + Y(t-2) + ... + Y(t-n)) / n
```

**Pros:** Simple, removes noise
**Cons:** Lags behind trends, no forecasting power

### 4.4.2 Exponential Smoothing

**Concept:** Weighted average where recent observations get more weight.

**Types:**

**Simple Exponential Smoothing (SES):**
```
Ŷ(t+1) = α × Y(t) + (1-α) × Ŷ(t)
```

**Holt's Linear Trend:**
Adds trend component.

**Holt-Winters (Triple Exponential Smoothing):**
Adds both trend and seasonality.

**Implementation:**
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    train,
    seasonal_periods=7,
    trend='add',
    seasonal='add'
)
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

### 4.4.3 ARIMA (AutoRegressive Integrated Moving Average)

**Components:**
- **AR (p):** Autoregressive - uses past values
- **I (d):** Integrated - differencing for stationarity
- **MA (q):** Moving Average - uses past forecast errors

**Model Notation:** ARIMA(p, d, q)

**Stationarity:** Time series should have constant mean and variance.

**Implementation:**
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(2, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

### 4.4.4 SARIMA (Seasonal ARIMA)

**Extends ARIMA with seasonal components:**
SARIMA(p, d, q)(P, D, Q, s)

Where:
- (p, d, q): Non-seasonal parameters
- (P, D, Q): Seasonal parameters
- s: Seasonal period (7 for weekly, 12 for monthly)

## 4.5 Time Series Metrics

### 4.5.1 RMSE
Same as regression - measures average error magnitude.

**In Our Project:** Exponential Smoothing achieved RMSE = 474.28

### 4.5.2 MAE
Measures average absolute error.

### 4.5.3 MAPE (Mean Absolute Percentage Error)

**Formula:**
```
MAPE = (100/n) × Σ|Actual - Forecast| / |Actual|
```

**Interpretation:** Average percentage error

---

# 5. Feature Engineering

## 5.1 What is Feature Engineering?

The process of using domain knowledge to create features that improve model performance.

## 5.2 Common Techniques

### 5.2.1 Handling Missing Values
```python
# Numerical: Mean/Median imputation
df['col'].fillna(df['col'].median(), inplace=True)

# Categorical: Mode imputation
df['col'].fillna(df['col'].mode()[0], inplace=True)
```

### 5.2.2 Encoding Categorical Variables

**Label Encoding:**
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])
```

**One-Hot Encoding:**
```python
df_encoded = pd.get_dummies(df, columns=['col'])
```

### 5.2.3 Feature Scaling

**StandardScaler (Z-score):**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**MinMaxScaler (0-1 range):**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 5.2.4 Creating New Features

```python
# Polynomial features
df['sqft_squared'] = df['sqft'] ** 2

# Interaction features
df['rooms_per_sqft'] = df['rooms'] / df['sqft']

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 10, 30, 50, 100])
```

---

# 6. Model Evaluation Metrics

## 6.1 Regression Metrics Summary

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| R² | 1 - SS_res/SS_tot | (-∞, 1] | Higher is better |
| RMSE | √(Σ(y-ŷ)²/n) | [0, ∞) | Lower is better, same units as target |
| MAE | Σ\|y-ŷ\|/n | [0, ∞) | Lower is better, robust to outliers |
| MAPE | 100×Σ\|y-ŷ\|/y | [0, ∞)% | Lower is better, percentage error |

## 6.2 Classification Metrics Summary

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | (TP+TN)/(All) | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| F1 | 2×Prec×Rec/(Prec+Rec) | Balance precision/recall |
| ROC-AUC | Area under ROC curve | Compare models |

---

# 7. Cross-Validation Techniques

## 7.1 K-Fold Cross-Validation

Splits data into K folds, trains on K-1, tests on 1. Repeats K times.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Mean: {scores.mean():.4f} (+/- {scores.std():.4f})')
```

## 7.2 Stratified K-Fold

Maintains class distribution in each fold. Essential for imbalanced data.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    # Train and evaluate
```

## 7.3 Time Series Cross-Validation

Rolling window approach that respects temporal order.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train on past, test on future
```

---

# 8. Hyperparameter Tuning

## 8.1 Grid Search

Exhaustive search over specified parameter values.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(X_train, y_train)
print(f'Best params: {grid_search.best_params_}')
```

## 8.2 Random Search

Samples from parameter distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=50,
    cv=5
)
```

---

# 9. Model Selection

## 9.1 Bias-Variance Tradeoff

- **Bias:** Error from overly simple models (underfitting)
- **Variance:** Error from overly complex models (overfitting)

**Goal:** Find the sweet spot with low bias AND low variance.

## 9.2 When to Use Which Model

| Scenario | Recommended Models |
|----------|-------------------|
| Linear relationship | Linear/Logistic Regression |
| Non-linear, many features | Random Forest, Gradient Boosting |
| Small dataset | KNN, SVM with RBF kernel |
| Interpretability required | Decision Trees, Logistic Regression |
| High-dimensional data | Lasso, PCA + any model |

---

# 10. Implementation Details

## 10.1 Project Results Summary

### Regression Results
| Model | R² Score | RMSE ($) | MAE ($) |
|-------|----------|----------|---------|
| Linear Regression | 0.9116 | 30,138 | 24,012 |
| **Ridge Regression** | **0.9117** | **30,130** | **24,008** |
| Lasso Regression | 0.9116 | 30,137 | 24,011 |
| ElasticNet | 0.8109 | 44,084 | 35,267 |
| Random Forest | 0.8658 | 37,132 | 29,706 |
| Gradient Boosting | 0.8880 | 33,925 | 27,140 |

### Classification Results
| Model | F1 Score | Accuracy | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | **0.7254** | **0.7400** | **0.7963** |
| Random Forest | 0.6877 | 0.7067 | 0.7741 |
| Gradient Boosting | 0.6877 | 0.7067 | 0.7763 |
| SVM | 0.6899 | 0.7100 | 0.7756 |

### Time Series Results
| Model | RMSE | MAE |
|-------|------|-----|
| Moving Average | 503.64 | 428.32 |
| **Exponential Smoothing** | **474.28** | **403.78** |
| ARIMA | 504.57 | 428.96 |

## 10.2 Code Quality Guidelines

1. **Modular Design:** Separate concerns into modules
2. **Type Hints:** Use Python type annotations
3. **Documentation:** Docstrings for all functions
4. **Error Handling:** Try-except blocks with logging
5. **Testing:** Unit tests for critical functions
6. **Version Control:** Git with meaningful commits

---

# Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Accuracy** | Proportion of correct predictions |
| **Bias** | Error from oversimplified assumptions |
| **Cross-Validation** | Technique to assess model generalization |
| **Ensemble** | Combining multiple models for better predictions |
| **Feature** | Input variable used for prediction |
| **Hyperparameter** | Parameter set before training |
| **Overfitting** | Model memorizes training data, poor generalization |
| **Regularization** | Penalty term to prevent overfitting |
| **Target** | Variable being predicted |
| **Variance** | Error from sensitivity to training data |

---

# Appendix B: References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
3. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
4. scikit-learn Documentation: https://scikit-learn.org/stable/
5. statsmodels Documentation: https://www.statsmodels.org/stable/

---

**Document Version:** 1.0
**Last Updated:** January 2026
**Author:** ML Analytics Team
