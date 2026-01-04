"""
Main Runner - Execute Complete ML Analytics Pipeline
"""

import sys
import os
import time
from datetime import datetime

# Fix encoding for Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except:
    pass

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import DataGenerator
from regression_model import AdvancedHousePriceRegressor
from classification_model import AdvancedChurnClassifier
from timeseries_model import AdvancedSalesForecaster


def print_header():
    """Print project header"""
    print("\n")
    print("=" * 80)
    print("   __  __ _         _                _       _   _          ")
    print("  |  \\/  | |       / \\   _ __   __ _| |_   _| |_(_) ___ ___ ")
    print("  | |\\/| | |      / _ \\ | '_ \\ / _` | | | | | __| |/ __/ __|")
    print("  | |  | | |___  / ___ \\| | | | (_| | | |_| | |_| | (__\\__ \\")
    print("  |_|  |_|_____|/_/   \\_\\_| |_|\\__,_|_|\\__, |\\__|_|\\___|___/")
    print("                                       |___/                ")
    print("=" * 80)
    print("  COMPREHENSIVE MACHINE LEARNING ANALYTICS PLATFORM")
    print("  Regression | Classification | Time Series Forecasting")
    print("=" * 80)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def run_all():
    """Run all ML pipelines"""
    start_time = time.time()
    
    print_header()
    
    # Step 1: Generate Data
    print("\n\n")
    print("*" * 80)
    print("  STEP 1: DATA GENERATION")
    print("*" * 80)
    
    generator = DataGenerator()
    datasets = generator.generate_all_datasets()
    
    # Step 2: Regression
    print("\n\n")
    print("*" * 80)
    print("  STEP 2: REGRESSION ANALYSIS")
    print("*" * 80)
    
    regressor = AdvancedHousePriceRegressor()
    regression_results = regressor.run_full_pipeline()
    
    # Step 3: Classification
    print("\n\n")
    print("*" * 80)
    print("  STEP 3: CLASSIFICATION ANALYSIS")
    print("*" * 80)
    
    classifier = AdvancedChurnClassifier()
    classification_results = classifier.run_full_pipeline()
    
    # Step 4: Time Series
    print("\n\n")
    print("*" * 80)
    print("  STEP 4: TIME SERIES ANALYSIS")
    print("*" * 80)
    
    forecaster = AdvancedSalesForecaster()
    timeseries_results = forecaster.run_full_pipeline()
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    
    # Final Summary
    print("\n")
    print("=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n  [1] REGRESSION - House Price Prediction")
    print("  " + "-"*50)
    print(f"      Dataset: {regression_results['dataset']['total_samples']} samples")
    print(f"      Features: {len(regression_results['dataset']['features'])}")
    print(f"      Models Trained: {len(regression_results['models'])}")
    print(f"      Best Model: {regression_results['best_model']['name']}")
    print(f"      R2 Score: {regression_results['best_model']['test_r2']:.4f}")
    print(f"      RMSE: ${regression_results['best_model']['rmse']:,.0f}")
    
    print("\n  [2] CLASSIFICATION - Customer Churn Prediction")
    print("  " + "-"*50)
    print(f"      Dataset: {classification_results['dataset']['total_samples']} samples")
    print(f"      Features: {len(classification_results['dataset']['features'])}")
    print(f"      Churn Rate: {classification_results['dataset']['class_distribution']['churn_rate']:.1f}%")
    print(f"      Models Trained: {len(classification_results['models'])}")
    print(f"      Best Model: {classification_results['best_model']['name']}")
    print(f"      F1 Score: {classification_results['best_model']['f1_score']:.4f}")
    print(f"      ROC-AUC: {classification_results['best_model']['roc_auc']:.4f}")
    
    print("\n  [3] TIME SERIES - Sales Forecasting")
    print("  " + "-"*50)
    print(f"      Dataset: {timeseries_results['dataset']['total_days']} days")
    print(f"      Period: {timeseries_results['dataset']['date_range']}")
    print(f"      Models Trained: {len(timeseries_results['models'])}")
    print(f"      Best Model: {timeseries_results['best_model']['name']}")
    print(f"      RMSE: {timeseries_results['best_model']['rmse']:.2f}")
    print(f"      MAPE: {timeseries_results['best_model']['mape']:.2f}%")
    
    print("\n  " + "="*50)
    print(f"  Total Execution Time: {elapsed_time:.1f} seconds")
    print("  " + "="*50)
    
    print("\n  OUTPUT DIRECTORIES:")
    print("  " + "-"*50)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    print(f"      Data:           {os.path.join(base_dir, 'data')}")
    print(f"      Regression:     {os.path.join(base_dir, 'output', 'regression')}")
    print(f"      Classification: {os.path.join(base_dir, 'output', 'classification')}")
    print(f"      Time Series:    {os.path.join(base_dir, 'output', 'timeseries')}")
    print(f"      Saved Models:   {os.path.join(base_dir, 'output', 'models')}")
    
    print("\n")
    print("=" * 80)
    print("  [+] ALL ANALYSES COMPLETE!")
    print("  [+] Start the dashboard: python src/api.py")
    print("  [+] Dashboard URL: http://localhost:5000")
    print("=" * 80)
    print("\n")
    
    return {
        'regression': regression_results,
        'classification': classification_results,
        'timeseries': timeseries_results,
        'execution_time': elapsed_time
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Analytics Pipeline')
    parser.add_argument('--parallel', action='store_true', 
                       help='Run all models in parallel for faster execution')
    args = parser.parse_args()
    
    if args.parallel:
        print("\n[*] Running in PARALLEL mode...")
        from main_parallel import run_all_parallel
        results = run_all_parallel()
    else:
        results = run_all()
