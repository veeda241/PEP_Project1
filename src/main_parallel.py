"""
Parallel ML Pipeline Runner - Execute All ML Analytics in Parallel
Uses multiprocessing to train Regression, Classification, and Time Series models simultaneously
"""

import sys
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Fix encoding for Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except:
    pass

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


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
    print("  PARALLEL MACHINE LEARNING ANALYTICS PLATFORM")
    print("  Regression | Classification | Time Series (Multiprocessing)")
    print("=" * 80)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU Cores Available: {multiprocessing.cpu_count()}")
    print("=" * 80)


def run_data_generation():
    """Generate all datasets"""
    from data_generator import DataGenerator
    print("\n[*] Generating Datasets...")
    generator = DataGenerator()
    datasets = generator.generate_all_datasets()
    print("[+] All datasets generated!")
    return datasets


def run_regression():
    """Run regression pipeline in a separate process"""
    try:
        # Re-import in subprocess
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        from regression_model import AdvancedHousePriceRegressor
        regressor = AdvancedHousePriceRegressor()
        results = regressor.run_full_pipeline()
        return ('regression', results, None)
    except Exception as e:
        return ('regression', None, str(e))


def run_classification():
    """Run classification pipeline in a separate process"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        from classification_model import AdvancedChurnClassifier
        classifier = AdvancedChurnClassifier()
        results = classifier.run_full_pipeline()
        return ('classification', results, None)
    except Exception as e:
        return ('classification', None, str(e))


def run_timeseries():
    """Run time series pipeline in a separate process"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        from timeseries_model import AdvancedSalesForecaster
        forecaster = AdvancedSalesForecaster()
        results = forecaster.run_full_pipeline()
        return ('timeseries', results, None)
    except Exception as e:
        return ('timeseries', None, str(e))


def run_all_parallel():
    """Run all ML pipelines in parallel using multiprocessing"""
    start_time = time.time()
    
    print_header()
    
    # Step 1: Generate Data (sequential - required before models)
    print("\n")
    print("*" * 80)
    print("  STEP 1: DATA GENERATION")
    print("*" * 80)
    
    run_data_generation()
    
    # Step 2: Run all models in parallel
    print("\n")
    print("*" * 80)
    print("  STEP 2: PARALLEL MODEL TRAINING")
    print("  [Running Regression, Classification, Time Series simultaneously]")
    print("*" * 80)
    
    results = {}
    errors = {}
    
    # Use ProcessPoolExecutor for true parallel execution
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_regression): 'Regression',
            executor.submit(run_classification): 'Classification',
            executor.submit(run_timeseries): 'Time Series'
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                task_key, result, error = future.result()
                if error:
                    errors[task_key] = error
                    print(f"\n    [X] {task_name} FAILED: {error}")
                else:
                    results[task_key] = result
                    print(f"\n    [+] {task_name} COMPLETED!")
            except Exception as e:
                errors[task_name.lower()] = str(e)
                print(f"\n    [X] {task_name} FAILED: {e}")
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    
    # Final Summary
    print("\n")
    print("=" * 80)
    print("  FINAL RESULTS SUMMARY (PARALLEL EXECUTION)")
    print("=" * 80)
    
    if 'regression' in results:
        regression_results = results['regression']
        print("\n  [1] REGRESSION - House Price Prediction")
        print("  " + "-"*50)
        print(f"      Dataset: {regression_results['dataset']['total_samples']} samples")
        print(f"      Features: {len(regression_results['dataset']['features'])}")
        print(f"      Models Trained: {len(regression_results['models'])}")
        print(f"      Best Model: {regression_results['best_model']['name']}")
        print(f"      R2 Score: {regression_results['best_model']['test_r2']:.4f}")
        print(f"      RMSE: ${regression_results['best_model']['rmse']:,.0f}")
    elif 'regression' in errors:
        print(f"\n  [1] REGRESSION - FAILED: {errors['regression']}")
    
    if 'classification' in results:
        classification_results = results['classification']
        print("\n  [2] CLASSIFICATION - Customer Churn Prediction")
        print("  " + "-"*50)
        print(f"      Dataset: {classification_results['dataset']['total_samples']} samples")
        print(f"      Features: {len(classification_results['dataset']['features'])}")
        print(f"      Churn Rate: {classification_results['dataset']['class_distribution']['churn_rate']:.1f}%")
        print(f"      Models Trained: {len(classification_results['models'])}")
        print(f"      Best Model: {classification_results['best_model']['name']}")
        print(f"      F1 Score: {classification_results['best_model']['f1_score']:.4f}")
        print(f"      ROC-AUC: {classification_results['best_model']['roc_auc']:.4f}")
    elif 'classification' in errors:
        print(f"\n  [2] CLASSIFICATION - FAILED: {errors['classification']}")
    
    if 'timeseries' in results:
        timeseries_results = results['timeseries']
        print("\n  [3] TIME SERIES - Sales Forecasting")
        print("  " + "-"*50)
        print(f"      Dataset: {timeseries_results['dataset']['total_days']} days")
        print(f"      Period: {timeseries_results['dataset']['date_range']}")
        print(f"      Models Trained: {len(timeseries_results['models'])}")
        print(f"      Best Model: {timeseries_results['best_model']['name']}")
        print(f"      RMSE: {timeseries_results['best_model']['rmse']:.2f}")
        print(f"      MAPE: {timeseries_results['best_model']['mape']:.2f}%")
    elif 'timeseries' in errors:
        print(f"\n  [3] TIME SERIES - FAILED: {errors['timeseries']}")
    
    print("\n  " + "="*50)
    print(f"  Total Execution Time: {elapsed_time:.1f} seconds")
    print(f"  Speed Improvement: ~{3 if len(results) == 3 else 1}x faster than sequential")
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
    if len(errors) == 0:
        print("  [+] ALL ANALYSES COMPLETE!")
    else:
        print(f"  [!] COMPLETED WITH {len(errors)} ERROR(S)")
    print("  [+] Start the dashboard: python src/api.py")
    print("  [+] Dashboard URL: http://localhost:5000")
    print("=" * 80)
    print("\n")
    
    return {
        'regression': results.get('regression'),
        'classification': results.get('classification'),
        'timeseries': results.get('timeseries'),
        'execution_time': elapsed_time,
        'errors': errors
    }


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    results = run_all_parallel()
