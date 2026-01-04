"""
Flask API for ML Dashboard
"""

from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import json

app = Flask(__name__, static_folder='../dashboard')
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/status')
def get_status():
    """Check if all results are available"""
    regression_exists = os.path.exists(os.path.join(OUTPUT_DIR, 'regression', 'regression_results.json'))
    classification_exists = os.path.exists(os.path.join(OUTPUT_DIR, 'classification', 'classification_results.json'))
    timeseries_exists = os.path.exists(os.path.join(OUTPUT_DIR, 'timeseries', 'timeseries_results.json'))
    
    return jsonify({
        'regression': regression_exists,
        'classification': classification_exists,
        'timeseries': timeseries_exists,
        'all_ready': regression_exists and classification_exists and timeseries_exists
    })


@app.route('/api/regression')
def get_regression():
    """Get regression results"""
    filepath = os.path.join(OUTPUT_DIR, 'regression', 'regression_results.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Results not found'}), 404


@app.route('/api/classification')
def get_classification():
    """Get classification results"""
    filepath = os.path.join(OUTPUT_DIR, 'classification', 'classification_results.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Results not found'}), 404


@app.route('/api/timeseries')
def get_timeseries():
    """Get time series results"""
    filepath = os.path.join(OUTPUT_DIR, 'timeseries', 'timeseries_results.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Results not found'}), 404


@app.route('/api/images/<category>/<filename>')
def get_image(category, filename):
    """Get visualization images"""
    filepath = os.path.join(OUTPUT_DIR, category, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    return jsonify({'error': 'Image not found'}), 404


@app.route('/api/summary')
def get_summary():
    """Get summary of all results"""
    summary = {'regression': None, 'classification': None, 'timeseries': None}
    
    for task in summary.keys():
        filepath = os.path.join(OUTPUT_DIR, task, f'{task}_results.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                summary[task] = json.load(f)
    
    return jsonify(summary)


# Production Model Registry
CHURN_ENGINE = None

def load_inference_engines():
    global CHURN_ENGINE
    try:
        from classification_model import AdvancedChurnClassifier
        engine = AdvancedChurnClassifier()
        # In a real system, we'd load the .pkl directly, but here we init the class
        # to access the helper methods. Currently, it needs re-training to be live.
        # For this DEMO, we will load the artifacts if they exist.
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models')
        model_path = os.path.join(models_dir, 'retention_model_champion.pkl')
        scaler_path = os.path.join(models_dir, 'production_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            import joblib
            engine.best_model = joblib.load(model_path)
            engine.scaler = joblib.load(scaler_path)
            
            # Load feature names from training data or config
            # Here we hardcode schema for demo safety
            engine.feature_names = [
                'tenure_months', 'monthly_charges', 'total_charges', 
                'contract_type', 'payment_method', 'tech_support', 
                'online_security', 'online_backup', 'device_protection', 
                'num_complaints', 'support_calls'
            ]
            engine.best_model_name = "Deployed_Champion"
            CHURN_ENGINE = engine
            print("[+] Inference Engine Loaded: READY")
        else:
            print("[!] Warn: Production models not found. Run training pipeline first.")
    except Exception as e:
        print(f"[!] Error loading inference engine: {e}")

# Initialize Engines on Startup
load_inference_engines()

@app.route('/api/predict/churn', methods=['POST'])
def predict_churn():
    """
    Real-time Churn Risk Assessment Endpoint.
    Expects JSON: { "features": [val1, val2, ...] }
    """
    if not CHURN_ENGINE:
        return jsonify({"error": "Model not loaded. Service Unavailable"}), 503
        
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid Input. 'features' list required."}), 400
            
        result = CHURN_ENGINE.predict_production(data['features'])
        
        # Log this inference (simulating DB log)
        # In real world: db.insert_log(input, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "Inference Failed"}), 500

if __name__ == '__main__':
    import socket
    
    # Get local IP address
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "YOUR_PC_IP"

    print("\n[*] Starting ChurnGuard Intelligence API...")
    print(f"    Local Access:    http://localhost:5000")
    print(f"    Network Access:  http://{local_ip}:5000")
    print("\n    Production Endpoints:")
    print("      - POST /api/predict/churn (Confidence-Aware Inference)")
    print("\n    Dashboard Endpoints:")
    print("      - /api/status")
    print("      - /api/regression")
    print("      - /api/classification")
    print("      - /api/timeseries")
    
    # Host='0.0.0.0' makes it accessible from other devices
    app.run(debug=True, host='0.0.0.0', port=5000)
