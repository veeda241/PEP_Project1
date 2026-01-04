"""
Configuration Settings for ML Analytics Project
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data Generation Settings
DATA_CONFIG = {
    'regression': {
        'n_samples': 2000,
        'test_size': 0.2,
        'random_state': 42
    },
    'classification': {
        'n_samples': 3000,
        'test_size': 0.2,
        'random_state': 42
    },
    'timeseries': {
        'n_days': 1095,  # 3 years
        'test_days': 90,
        'random_state': 42
    }
}

# Model Hyperparameters
REGRESSION_MODELS = {
    'Linear Regression': {},
    'Ridge Regression': {'alpha': 1.0},
    'Lasso Regression': {'alpha': 1.0},
    'ElasticNet': {'alpha': 1.0, 'l1_ratio': 0.5},
    'Random Forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
    'SVR': {'kernel': 'rbf', 'C': 1.0},
    'KNN': {'n_neighbors': 5, 'weights': 'distance'}
}

CLASSIFICATION_MODELS = {
    'Logistic Regression': {'max_iter': 1000, 'random_state': 42},
    'Decision Tree': {'max_depth': 10, 'random_state': 42},
    'Random Forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
    'AdaBoost': {'n_estimators': 100, 'random_state': 42},
    'SVM': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
    'KNN': {'n_neighbors': 5},
    'Naive Bayes': {},
    'Extra Trees': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'Bagging': {'n_estimators': 50, 'random_state': 42}
}

TIMESERIES_MODELS = {
    'Moving Average': {'window': 7},
    'Exponential Smoothing': {'seasonal_periods': 7, 'trend': 'add', 'seasonal': 'add'},
    'ARIMA': {'order': (2, 1, 2)},
    'SARIMA': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)},
    'Prophet-like': {'growth': 'linear'}
}

# Cross-Validation Settings
CV_FOLDS = 5
SCORING_REGRESSION = 'r2'
SCORING_CLASSIFICATION = 'f1'

# Visualization Settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 150
COLOR_PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
