"""
Advanced Time Series Module - Sales Forecasting
5 Models with Decomposition, Stationarity Tests, and Forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


class AdvancedSalesForecaster:
    """
    Advanced Sales Time Series Forecasting
    
    Features:
    - Multiple forecasting models
    - Time series decomposition
    - Stationarity testing (ADF)
    - ACF/PACF analysis
    - Rolling cross-validation
    - Comprehensive visualizations
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'sales_timeseries.csv'
        )
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'output', 'timeseries'
        )
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'output', 'models'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.forecast_days = 30
        self.df = None
        self.train = None
        self.test = None
        self.decomposition = None
        self.adf_result = None
    
    def load_data(self):
        """Load and prepare time series data"""
        print("\n[*] Loading Sales Time Series Data...")
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').set_index('date')
        
        print(f"    Period: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"    Total days: {len(self.df)}")
        
        # Statistics
        print(f"\n    Sales Statistics:")
        print(f"    - Min: ${self.df['sales'].min():,.2f}")
        print(f"    - Max: ${self.df['sales'].max():,.2f}")
        print(f"    - Mean: ${self.df['sales'].mean():,.2f}")
        print(f"    - Std: ${self.df['sales'].std():,.2f}")
        
        # Train/test split (last 90 days for testing)
        split_date = self.df.index.max() - timedelta(days=90)
        self.train = self.df[self.df.index <= split_date]['sales']
        self.test = self.df[self.df.index > split_date]['sales']
        
        print(f"\n    Train: {len(self.train)} days ({self.train.index.min().date()} to {self.train.index.max().date()})")
        print(f"    Test: {len(self.test)} days ({self.test.index.min().date()} to {self.test.index.max().date()})")
        
        return self.df
    
    def test_stationarity(self):
        """Perform Augmented Dickey-Fuller test for stationarity"""
        print("\n[*] Testing Stationarity (ADF Test)...")
        
        result = adfuller(self.train, autolag='AIC')
        
        self.adf_result = {
            'test_statistic': float(result[0]),
            'p_value': float(result[1]),
            'lags_used': int(result[2]),
            'observations': int(result[3]),
            'critical_values': {k: float(v) for k, v in result[4].items()},
            'is_stationary': bool(result[1] < 0.05)
        }
        
        print(f"    - ADF Statistic: {result[0]:.4f}")
        print(f"    - p-value: {result[1]:.4f}")
        print(f"    - Critical Values:")
        for key, value in result[4].items():
            print(f"      {key}: {value:.4f}")
        
        if result[1] < 0.05:
            print("    [+] Series is STATIONARY (p < 0.05)")
        else:
            print("    [!] Series is NON-STATIONARY (p >= 0.05)")
        
        return self.adf_result
    
    def decompose_series(self):
        """Decompose time series into components"""
        print("\n[*] Decomposing Time Series...")
        
        # Use weekly seasonality
        self.decomposition = seasonal_decompose(
            self.train, 
            model='additive', 
            period=7,
            extrapolate_trend='freq'
        )
        
        print("    Components extracted:")
        print("    - Trend: Long-term movement")
        print("    - Seasonal: Weekly pattern (7-day cycle)")
        print("    - Residual: Random fluctuations")
        
        # Create decomposition plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        axes[0].plot(self.decomposition.observed, color='#3498db', linewidth=1)
        axes[0].set_title('Original Series', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Sales ($)')
        
        axes[1].plot(self.decomposition.trend, color='#e74c3c', linewidth=1.5)
        axes[1].set_title('Trend Component', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Sales ($)')
        
        axes[2].plot(self.decomposition.seasonal, color='#2ecc71', linewidth=1)
        axes[2].set_title('Seasonal Component (Weekly)', fontweight='bold', fontsize=12)
        axes[2].set_ylabel('Sales ($)')
        
        axes[3].plot(self.decomposition.resid, color='#9b59b6', linewidth=1, alpha=0.7)
        axes[3].set_title('Residual Component', fontweight='bold', fontsize=12)
        axes[3].set_ylabel('Sales ($)')
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'decomposition.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("    [+] Decomposition complete")
        return self.decomposition
    
    def train_models(self):
        """Train all time series forecasting models"""
        print("\n[*] Training Forecasting Models...")
        print("    " + "-"*50)
        
        n_test = len(self.test)
        
        # 1. Simple Moving Average
        print("\n    Training Simple Moving Average...")
        ma_window = 7
        ma_pred = self.train.rolling(window=ma_window).mean().iloc[-1]
        ma_forecast = np.full(n_test, ma_pred)
        self.results['Moving Average (7-day)'] = {
            'forecast': ma_forecast,
            'rmse': float(np.sqrt(mean_squared_error(self.test, ma_forecast))),
            'mae': float(mean_absolute_error(self.test, ma_forecast)),
            'mape': float(np.mean(np.abs((self.test - ma_forecast) / self.test)) * 100)
        }
        print(f"      [+] RMSE: {self.results['Moving Average (7-day)']['rmse']:.2f}")
        
        # 2. Simple Exponential Smoothing
        print("\n    Training Simple Exponential Smoothing...")
        try:
            ses_model = ExponentialSmoothing(self.train, trend=None, seasonal=None)
            ses_fit = ses_model.fit(smoothing_level=0.3)
            ses_forecast = ses_fit.forecast(n_test)
            self.models['SES'] = ses_fit
            self.results['Simple Exp Smoothing'] = {
                'forecast': ses_forecast.values,
                'rmse': float(np.sqrt(mean_squared_error(self.test, ses_forecast))),
                'mae': float(mean_absolute_error(self.test, ses_forecast)),
                'mape': float(np.mean(np.abs((self.test - ses_forecast) / self.test)) * 100)
            }
            print(f"      [+] RMSE: {self.results['Simple Exp Smoothing']['rmse']:.2f}")
        except Exception as e:
            print(f"      [!] SES failed: {e}")
        
        # 3. Holt-Winters (Triple Exponential Smoothing)
        print("\n    Training Holt-Winters...")
        try:
            hw_model = ExponentialSmoothing(
                self.train, 
                seasonal_periods=7, 
                trend='add', 
                seasonal='add'
            )
            hw_fit = hw_model.fit()
            hw_forecast = hw_fit.forecast(n_test)
            self.models['Holt-Winters'] = hw_fit
            self.results['Holt-Winters'] = {
                'forecast': hw_forecast.values,
                'rmse': float(np.sqrt(mean_squared_error(self.test, hw_forecast))),
                'mae': float(mean_absolute_error(self.test, hw_forecast)),
                'mape': float(np.mean(np.abs((self.test - hw_forecast) / self.test)) * 100)
            }
            print(f"      [+] RMSE: {self.results['Holt-Winters']['rmse']:.2f}")
        except Exception as e:
            print(f"      [!] Holt-Winters failed: {e}")
        
        # 4. ARIMA
        print("\n    Training ARIMA(2,1,2)...")
        try:
            arima_model = ARIMA(self.train, order=(2, 1, 2))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(n_test)
            self.models['ARIMA'] = arima_fit
            self.results['ARIMA(2,1,2)'] = {
                'forecast': arima_forecast.values,
                'rmse': float(np.sqrt(mean_squared_error(self.test, arima_forecast))),
                'mae': float(mean_absolute_error(self.test, arima_forecast)),
                'mape': float(np.mean(np.abs((self.test - arima_forecast) / self.test)) * 100)
            }
            print(f"      [+] RMSE: {self.results['ARIMA(2,1,2)']['rmse']:.2f}")
        except Exception as e:
            print(f"      [!] ARIMA failed: {e}")
        
        # 5. SARIMA (Seasonal ARIMA)
        print("\n    Training SARIMA(1,1,1)(1,1,1,7)...")
        try:
            sarima_model = SARIMAX(
                self.train, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_fit = sarima_model.fit(disp=False)
            sarima_forecast = sarima_fit.forecast(n_test)
            self.models['SARIMA'] = sarima_fit
            self.results['SARIMA'] = {
                'forecast': sarima_forecast.values,
                'rmse': float(np.sqrt(mean_squared_error(self.test, sarima_forecast))),
                'mae': float(mean_absolute_error(self.test, sarima_forecast)),
                'mape': float(np.mean(np.abs((self.test - sarima_forecast) / self.test)) * 100)
            }
            print(f"      [+] RMSE: {self.results['SARIMA']['rmse']:.2f}")
        except Exception as e:
            print(f"      [!] SARIMA failed: {e}")
        
        # Find best model
        best_rmse = min(r['rmse'] for r in self.results.values())
        self.best_model_name = [n for n, r in self.results.items() if r['rmse'] == best_rmse][0]
        
        print("\n    " + "-"*50)
        print(f"    [*] Best Model: {self.best_model_name}")
        print(f"        RMSE = {best_rmse:.2f}, MAE = {self.results[self.best_model_name]['mae']:.2f}")
        
        return self.results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n[*] Creating Visualizations...")
        plt.style.use('seaborn-v0_8-darkgrid')
        
        names = list(self.results.keys())
        
        # 1. Full Time Series with Forecasts
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.plot(self.train.index, self.train.values, label='Training Data', 
               color='#3498db', linewidth=1, alpha=0.7)
        ax.plot(self.test.index, self.test.values, label='Actual (Test)', 
               color='#2ecc71', linewidth=2)
        
        colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63']
        for (name, result), color in zip(self.results.items(), colors[:len(self.results)]):
            ax.plot(self.test.index, result['forecast'], label=f'{name}', 
                   color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
        ax.set_title('Sales Forecasting - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'forecast_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Model Performance Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSE
        rmse_vals = [self.results[n]['rmse'] for n in names]
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))
        bars1 = axes[0].bar(names, rmse_vals, color=colors)
        axes[0].set_ylabel('RMSE', fontweight='bold')
        axes[0].set_title('Model Comparison - RMSE (Lower is Better)', fontsize=12, fontweight='bold')
        for bar, val in zip(bars1, rmse_vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{val:.1f}', ha='center', fontweight='bold', fontsize=9)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # MAPE
        mape_vals = [self.results[n]['mape'] for n in names]
        bars2 = axes[1].bar(names, mape_vals, color=colors)
        axes[1].set_ylabel('MAPE (%)', fontweight='bold')
        axes[1].set_title('Model Comparison - MAPE (Lower is Better)', fontsize=12, fontweight='bold')
        for bar, val in zip(bars2, mape_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Best Model Forecast Detail
        fig, ax = plt.subplots(figsize=(14, 6))
        
        best_forecast = self.results[self.best_model_name]['forecast']
        
        ax.plot(self.test.index, self.test.values, label='Actual', 
               color='#2ecc71', linewidth=2, marker='o', markersize=3)
        ax.plot(self.test.index, best_forecast, label=f'{self.best_model_name} Forecast', 
               color='#e74c3c', linewidth=2, linestyle='--')
        
        # Error band
        ax.fill_between(self.test.index, 
                       best_forecast - self.results[self.best_model_name]['rmse'],
                       best_forecast + self.results[self.best_model_name]['rmse'],
                       alpha=0.2, color='#e74c3c', label='RMSE Band')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.best_model_name}: Forecast vs Actual', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        # Add metrics
        rmse = self.results[self.best_model_name]['rmse']
        mae = self.results[self.best_model_name]['mae']
        mape = self.results[self.best_model_name]['mape']
        textstr = f'RMSE: ${rmse:.2f}\nMAE: ${mae:.2f}\nMAPE: {mape:.2f}%'
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_forecast.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Weekly Pattern Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weekly_avg = self.df.groupby(self.df.index.dayofweek)['sales'].mean()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        colors = plt.cm.Set2(np.linspace(0, 1, 7))
        bars = ax.bar(days, weekly_avg.values, color=colors, edgecolor='white')
        ax.set_ylabel('Average Sales ($)', fontweight='bold')
        ax.set_title('Weekly Sales Pattern', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, weekly_avg.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                   f'${val:.0f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'weekly_pattern.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    [+] Saved 5 visualizations to {self.output_dir}")
    
    def save_models(self):
        """Save trained models"""
        print("\n[*] Saving Models...")
        
        for name, model in self.models.items():
            safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            filepath = os.path.join(self.models_dir, f'timeseries_{safe_name}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"    [+] Saved {len(self.models)} models to {self.models_dir}")
    
    def get_summary(self):
        """Get comprehensive results summary"""
        return {
            'task': 'Time Series Forecasting',
            'problem': 'Sales Prediction',
            'dataset': {
                'total_days': len(self.df),
                'train_days': len(self.train),
                'test_days': len(self.test),
                'date_range': f"{self.df.index.min().date()} to {self.df.index.max().date()}",
                'sales_stats': {
                    'min': float(self.df['sales'].min()),
                    'max': float(self.df['sales'].max()),
                    'mean': float(self.df['sales'].mean()),
                    'std': float(self.df['sales'].std())
                }
            },
            'stationarity_test': self.adf_result,
            'models': {n: {'rmse': r['rmse'], 'mae': r['mae'], 'mape': r['mape']} 
                      for n, r in self.results.items()},
            'best_model': {
                'name': self.best_model_name,
                'rmse': self.results[self.best_model_name]['rmse'],
                'mae': self.results[self.best_model_name]['mae'],
                'mape': self.results[self.best_model_name]['mape']
            }
        }
    
    def save_results(self):
        """Save results to JSON"""
        summary = self.get_summary()
        filepath = os.path.join(self.output_dir, 'timeseries_results.json')
        
        # Custom encoder for numpy types
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj
        
        # Convert all values recursively
        import json as json_module
        class NumpyEncoder(json_module.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"    [+] Results saved to {filepath}")
        return summary
    
    def run_full_pipeline(self):
        """Run complete time series pipeline"""
        print("\n" + "="*70)
        print("  SALES FORECASTING - ADVANCED TIME SERIES ANALYSIS")
        print("  5 Models | Decomposition | Stationarity Testing")
        print("="*70)
        
        self.load_data()
        self.test_stationarity()
        self.decompose_series()
        self.train_models()
        self.create_visualizations()
        self.save_models()
        summary = self.save_results()
        
        print("\n" + "="*70)
        print("  [+] TIME SERIES ANALYSIS COMPLETE!")
        print("="*70 + "\n")
        
        return summary


if __name__ == "__main__":
    from data_generator import DataGenerator
    DataGenerator().generate_sales_time_series(n_days=1095)
    forecaster = AdvancedSalesForecaster()
    forecaster.run_full_pipeline()
