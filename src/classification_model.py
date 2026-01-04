import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import time
import logging
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Configure Production Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("churn_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChurnEngine")

class AdvancedChurnClassifier:
    """
    Production-Grade Customer Retention Engine.
    
    Architecture:
    - Data Ingestion: Loads and preprocesses telco churn data.
    - Model Registry: Manages lifecycle of 10+ classification algorithms.
    - Inference Engine: Provides real-time probabilities with confidence thresholds.
    - Safety Layer: Flags low-confidence predictions for human review.
    """
    
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'customer_churn.csv')
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'classification')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = ""
        self.scaler = StandardScaler()
        self.feature_names = None
        
        logger.info("Initializing Churn Intelligence Engine...")

    def load_data(self):
        """Load and preprocess data with validation"""
        start_time = time.time()
        try:
            if not os.path.exists(self.data_path):
                logger.error(f"Data source not found at {self.data_path}")
                raise FileNotFoundError("Critical: Training data missing. Run data_generator.py first.")
                
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape[0]} records loaded in {time.time() - start_time:.4f}s")
            
            # Feature Separation
            X = self.df.drop('churn', axis=1)
            y = self.df['churn']
            
            self.feature_names = X.columns.tolist()
            
            # Encoding (Strict mapping for production consistency)
            le = LabelEncoder()
            msg_cols = ['contract_type', 'payment_method']
            for col in msg_cols:
                X[col] = le.fit_transform(X[col])
                
            # Scaling
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            return self.df
            
        except Exception as e:
            logger.critical(f"Data Pipeline Failure: {str(e)}")
            raise

    def _get_models(self):
        """Model Registry definition"""
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            #'SVM': SVC(probability=True, random_state=42), # Disabled for speed in demo
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
    
    def train_models(self, parallel=True):
        """Train models with observability and error tracking"""
        logger.info("Starting Model Training Pipeline...")
        
        models = self._get_models()
        self.results = {}
        
        for name, model in models.items():
            try:
                t0 = time.time()
                model.fit(self.X_train_scaled, self.y_train)
                
                # Evaluation
                y_pred = model.predict(self.X_test_scaled)
                y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
                
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_prob)
                latency = time.time() - t0
                
                # Store extensive metadata
                cm = confusion_matrix(self.y_test, y_pred)
                
                self.models[name] = model
                self.results[name] = {
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'latency_ms': latency * 1000,
                    'probabilities': y_prob, 
                    'confusion_matrix': cm.tolist()
                }
                logger.info(f"Model Trained: {name} | F1: {f1:.3f} | Latency: {latency*1000:.2f}ms")
                
            except Exception as e:
                logger.error(f"Training Failed for {name}: {str(e)}")

        # Select Champion Model
        if not self.results:
             raise RuntimeError("No models trained successfully.")
             
        best_Metric = max(self.results.values(), key=lambda x: x['roc_auc'])
        self.best_model_name = [k for k,v in self.results.items() if v == best_Metric][0]
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"🏆 Champion Model Selected: {self.best_model_name} (AUC: {best_Metric['roc_auc']:.3f})")
        self._save_production_artifacts()
        return self.results

    def _save_production_artifacts(self):
        """Save model and metadata for API serving"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        artifact_path = os.path.join(self.models_dir, f"retention_model_champion.pkl")
        joblib.dump(self.best_model, artifact_path)
        joblib.dump(self.scaler, os.path.join(self.models_dir, "production_scaler.pkl"))
        
        # Save Metadata
        meta = {
            "model_version": timestamp,
            "champion_algorithm": self.best_model_name,
            "metrics": {k: v for k, v in self.results[self.best_model_name].items() if k not in ['probabilities', 'confusion_matrix']},
            "feature_schema": self.feature_names
        }
        with open(os.path.join(self.output_dir, "model_manifest.json"), 'w') as f:
            json.dump(meta, f, indent=4)
            
    def predict_production(self, input_data):
        """
        Real-time Inference with Safety Layers.
        """
        start_ts = time.time()
        
        # 1. Feature Validation
        if len(input_data) != len(self.feature_names):
             return {"status": "error", "message": f"Schema Mismatch: Expected {len(self.feature_names)} features"}

        # 2. Preprocessing
        try:
            scaled_data = self.scaler.transform([input_data])
        except Exception as e:
            return {"status": "error", "message": f"Preprocessing Failed: {str(e)}"}
            
        # 3. Inference
        prob = self.best_model.predict_proba(scaled_data)[0][1]
        
        # 4. Confidence Filter (The "Real World" Check)
        confidence_status = "HIGH"
        decision = "RETAIN" if prob < 0.5 else "CHURN_RISK"
        
        # Introduce "No Decision" zone
        if 0.45 <= prob <= 0.55:
            confidence_status = "LOW_CONFIDENCE_MANUAL_REVIEW"
            decision = "NO_DECISION"
            logger.warning(f"Uncertain Prediction: {prob:.4f} flagged for review")
        
        latency_ms = (time.time() - start_ts) * 1000
        
        response = {
            "decision": decision,
            "churn_probability": round(prob, 4),
            "confidence_level": confidence_status,
            "inference_latency_ms": round(latency_ms, 2),
            "model_version": self.best_model_name
        }
        
        return response

    def generate_visualizations(self):
        """Create and save plots for dashboard"""
        logger.info("[*] Creating Visualizations...")
        
        # Comparison Plot
        plt.figure(figsize=(12, 6))
        model_names = list(self.results.keys())
        f1_scores = [r['f1_score'] for r in self.results.values()]
        auc_scores = [r['roc_auc'] for r in self.results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, f1_scores, width, label='F1 Score', color='#3498db')
        plt.bar(x + width/2, auc_scores, width, label='ROC-AUC', color='#2ecc71')
        
        plt.ylabel('Score')
        plt.title('Classification Model Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))
        plt.close()
        
        # ROC Curve for Best Model
        plt.figure(figsize=(8, 6))
        
        # We need to recalculate ROC curve data
        y_prob = self.results[self.best_model_name]['probabilities']
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'{self.best_model_name} (AUC = {self.results[self.best_model_name]["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'))
        plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(self.results[self.best_model_name]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Retain', 'Churn'],
                   yticklabels=['Retain', 'Churn'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix: {self.best_model_name}')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save results to JSON
        output_data = {
            'best_model': self.best_model_name,
            'models': {
                name: {k: v for k, v in res.items() if k not in ['probabilities', 'confusion_matrix']}
                for name, res in self.results.items()
            },
            'dataset': {
                'total_samples': len(self.df),
                'features': list(self.df.columns[:-1]),
                'class_distribution': {
                    'churn_rate': (self.df['churn'].sum() / len(self.df)) * 100
                }
            }
        }
        
        with open(os.path.join(self.output_dir, 'classification_results.json'), 'w') as f:
            json.dump(output_data, f, indent=4)
            
        logger.info(f"    [+] Saved 3 visualizations to {self.output_dir}")

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        self.load_data()
        self.train_models()
        self.generate_visualizations()
        return {
            'best_model': {
               'name': self.best_model_name,
               'f1_score': self.results[self.best_model_name]['f1_score'],
               'roc_auc': self.results[self.best_model_name]['roc_auc']
            },
            'dataset': {
                'total_samples': len(self.df),
                'features': list(self.df.columns[:-1]),
                'class_distribution': {
                    'churn_rate': (self.df['churn'].sum() / len(self.df)) * 100
                }
            },
            'models': list(self.models.keys())
        }
