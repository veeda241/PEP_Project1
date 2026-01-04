"""
Advanced Regression Module - House Price Prediction
8 Algorithms with Cross-Validation and Hyperparameter Analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import os
import json
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except:
    pass


class AdvancedHousePriceRegressor:
    """
    Advanced House Price Prediction with 8+ Regression Algorithms
    
    Features:
    - Multiple regression algorithms
    - Cross-validation
    - Feature importance analysis
    - Model persistence
    - Comprehensive visualizations
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'house_prices.csv'
        )
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'output', 'regression'
        )
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'output', 'models'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        self.df = None
        
    def load_data(self):
        """Load and prepare dataset with exploratory analysis"""
        print("\n[*] Loading House Price Dataset...")
        
        self.df = pd.read_csv(self.data_path)
        
        # Dataset info
        print(f"    Shape: {self.df.shape}")
        print(f"    Features: {list(self.df.columns[:-1])}")
        print(f"    Target: price")
        print(f"\n    Dataset Statistics:")
        print(f"    - Price Range: ${self.df['price'].min():,.0f} - ${self.df['price'].max():,.0f}")
        print(f"    - Mean Price: ${self.df['price'].mean():,.0f}")
        print(f"    - Median Price: ${self.df['price'].median():,.0f}")
        
        # Feature matrix and target
        self.X = self.df.drop('price', axis=1)
        self.y = self.df['price']
        self.feature_names = list(self.X.columns)
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n    Train: {len(self.X_train)} samples")
        print(f"    Test: {len(self.X_test)} samples")
        
        return self.df
    
    def _get_models(self):
        """Define all regression models"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance')
        }
    
    def train_models(self, parallel=True):
        """Train all regression models with cross-validation - optionally in parallel"""
        print("\n[*] Training Regression Models...")
        print("    " + "-"*50)
        
        models = self._get_models()
        
        # Determine which models need scaled data
        scaled_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                        'ElasticNet', 'SVR', 'KNN']
        
        if parallel:
            print(f"    [Parallel Mode: Using all available CPU cores]")
            from joblib import Parallel, delayed
            
            def train_single_model(name, model, X_train, X_test, y_train, y_test):
                """Train a single model and return results"""
                import numpy as np
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae = mean_absolute_error(y_test, y_pred_test)
                mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
                
                return {
                    'name': name,
                    'model': model,
                    'predictions': y_pred_test,
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                }
            
            # Prepare data for each model
            model_data = []
            for name, model in models.items():
                if name in scaled_models:
                    X_tr, X_te = self.X_train_scaled, self.X_test_scaled
                else:
                    X_tr, X_te = self.X_train.values, self.X_test.values
                model_data.append((name, model, X_tr, X_te, self.y_train, self.y_test))
            
            # Train all models in parallel
            results_list = Parallel(n_jobs=-1, verbose=0)(
                delayed(train_single_model)(name, model, X_tr, X_te, y_train, y_test)
                for name, model, X_tr, X_te, y_train, y_test in model_data
            )
            
            # Process results
            for result in results_list:
                name = result['name']
                self.models[name] = result['model']
                self.results[name] = {k: v for k, v in result.items() if k not in ['name', 'model']}
                overfit_gap = result['train_r2'] - result['test_r2']
                overfit_warning = " [!OVERFIT]" if overfit_gap > 0.1 else ""
                print(f"\n    [+] {name}: R2={result['test_r2']:.4f} | RMSE=${result['rmse']:,.0f}{overfit_warning}")
        else:
            # Sequential training (original method)
            for name, model in models.items():
                print(f"\n    Training {name}...")
                
                # Determine which data to use
                if name in scaled_models:
                    X_tr, X_te = self.X_train_scaled, self.X_test_scaled
                else:
                    X_tr, X_te = self.X_train.values, self.X_test.values
                
                # Train model
                model.fit(X_tr, self.y_train)
                
                # Predictions
                y_pred_train = model.predict(X_tr)
                y_pred_test = model.predict(X_te)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_tr, self.y_train, cv=5, scoring='r2')
                
                # Metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                mae = mean_absolute_error(self.y_test, y_pred_test)
                mape = mean_absolute_percentage_error(self.y_test, y_pred_test) * 100
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'predictions': y_pred_test,
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                }
                
                # Check for overfitting
                overfit_gap = train_r2 - test_r2
                overfit_warning = " [!OVERFIT]" if overfit_gap > 0.1 else ""
                
                print(f"      [+] Test R2: {test_r2:.4f} | RMSE: ${rmse:,.0f} | CV: {cv_scores.mean():.4f}{overfit_warning}")
        
        # Find best model
        best_r2 = max(r['test_r2'] for r in self.results.values())
        self.best_model_name = [n for n, r in self.results.items() if r['test_r2'] == best_r2][0]
        self.best_model = self.models[self.best_model_name]
        
        print("\n    " + "-"*50)
        print(f"    [*] Best Model: {self.best_model_name}")
        print(f"        R2 = {best_r2:.4f}, RMSE = ${self.results[self.best_model_name]['rmse']:,.0f}")
        
        return self.results
    
    def get_feature_importance(self):
        """Extract feature importance from models"""
        importance_data = {}
        
        # Tree-based models
        for name in ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'Decision Tree']:
            if name in self.models:
                model = self.models[name]
                importance = model.feature_importances_
                importance_data[name] = dict(zip(self.feature_names, importance.tolist()))
        
        # Linear models (coefficients)
        for name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            if name in self.models:
                model = self.models[name]
                coef_abs = np.abs(model.coef_)
                if coef_abs.sum() > 0:
                    coef_normalized = coef_abs / coef_abs.sum()
                    importance_data[name] = dict(zip(self.feature_names, coef_normalized.tolist()))
        
        return importance_data
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n[*] Creating Visualizations...")
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Model Comparison Chart
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        names = list(self.results.keys())
        r2_scores = [self.results[m]['test_r2'] for m in names]
        rmse_scores = [self.results[m]['rmse'] for m in names]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        
        # R2 comparison
        bars1 = axes[0].barh(names, r2_scores, color=colors)
        axes[0].set_xlabel('R2 Score', fontweight='bold')
        axes[0].set_title('Model Comparison - R2 Score', fontsize=14, fontweight='bold')
        axes[0].set_xlim(0, 1)
        for bar, score in zip(bars1, r2_scores):
            axes[0].text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', va='center', fontweight='bold', fontsize=9)
        
        # RMSE comparison
        bars2 = axes[1].barh(names, rmse_scores, color=colors)
        axes[1].set_xlabel('RMSE ($)', fontweight='bold')
        axes[1].set_title('Model Comparison - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
        for bar, score in zip(bars2, rmse_scores):
            axes[1].text(score + 500, bar.get_y() + bar.get_height()/2, 
                        f'${score:,.0f}', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Actual vs Predicted (Best Model)
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pred = self.results[self.best_model_name]['predictions']
        
        scatter = ax.scatter(self.y_test, y_pred, alpha=0.5, c='#3498db', edgecolors='white', s=50)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.best_model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        # Add metrics text
        r2 = self.results[self.best_model_name]['test_r2']
        rmse = self.results[self.best_model_name]['rmse']
        textstr = f'R2 = {r2:.4f}\nRMSE = ${rmse:,.0f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance
        if 'Random Forest' in self.models:
            fig, ax = plt.subplots(figsize=(10, 6))
            importance = self.models['Random Forest'].feature_importances_
            indices = np.argsort(importance)[::-1]
            
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(self.feature_names)))
            bars = ax.barh(range(len(indices)), importance[indices], color=colors[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([self.feature_names[i] for i in indices])
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Residuals Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        residuals = self.y_test.values - self.results[self.best_model_name]['predictions']
        
        # Histogram
        axes[0].hist(residuals, bins=50, color='#9b59b6', edgecolor='white', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residual (Actual - Predicted)', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        
        # Residuals vs Predicted
        axes[1].scatter(self.results[self.best_model_name]['predictions'], residuals, 
                       alpha=0.5, c='#3498db', edgecolors='white')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Price ($)', fontweight='bold')
        axes[1].set_ylabel('Residual ($)', fontweight='bold')
        axes[1].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'residuals_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Cross-Validation Scores
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cv_means = [self.results[m]['cv_r2_mean'] for m in names]
        cv_stds = [self.results[m]['cv_r2_std'] for m in names]
        
        x = np.arange(len(names))
        bars = ax.bar(x, cv_means, yerr=cv_stds, capsize=5, color=colors, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Cross-Validation R2 Score', fontweight='bold')
        ax.set_title('5-Fold Cross-Validation Scores (with Std Dev)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cross_validation_scores.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    [+] Saved 5 visualizations to {self.output_dir}")
    
    def save_models(self):
        """Save trained models"""
        print("\n[*] Saving Models...")
        
        for name, model in self.models.items():
            safe_name = name.lower().replace(' ', '_')
            filepath = os.path.join(self.models_dir, f'regression_{safe_name}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'regression_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"    [+] Saved {len(self.models)} models to {self.models_dir}")
    
    def get_summary(self):
        """Get comprehensive results summary"""
        return {
            'task': 'Regression',
            'problem': 'House Price Prediction',
            'dataset': {
                'total_samples': len(self.df),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features': self.feature_names,
                'target': 'price',
                'price_stats': {
                    'min': float(self.df['price'].min()),
                    'max': float(self.df['price'].max()),
                    'mean': float(self.df['price'].mean()),
                    'median': float(self.df['price'].median())
                }
            },
            'models': {n: {k: v for k, v in r.items() if k != 'predictions'} 
                      for n, r in self.results.items()},
            'best_model': {
                'name': self.best_model_name,
                'test_r2': self.results[self.best_model_name]['test_r2'],
                'rmse': self.results[self.best_model_name]['rmse'],
                'mae': self.results[self.best_model_name]['mae']
            },
            'feature_importance': self.get_feature_importance()
        }
    
    def save_results(self):
        """Save results to JSON"""
        summary = self.get_summary()
        filepath = os.path.join(self.output_dir, 'regression_results.json')
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"    [+] Results saved to {filepath}")
        return summary
    
    def run_full_pipeline(self):
        """Run complete regression pipeline"""
        print("\n" + "="*70)
        print("  HOUSE PRICE PREDICTION - ADVANCED REGRESSION ANALYSIS")
        print("  10 Algorithms | Cross-Validation | Feature Importance")
        print("="*70)
        
        self.load_data()
        self.train_models()
        self.create_visualizations()
        self.save_models()
        summary = self.save_results()
        
        print("\n" + "="*70)
        print("  [+] REGRESSION ANALYSIS COMPLETE!")
        print("="*70 + "\n")
        
        return summary


if __name__ == "__main__":
    from data_generator import DataGenerator
    DataGenerator().generate_house_price_data(n_samples=2000)
    regressor = AdvancedHousePriceRegressor()
    regressor.run_full_pipeline()
