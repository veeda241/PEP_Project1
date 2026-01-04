# 🛡️ ChurnGuard: Enterprise Retention Intelligence

> **Predict. Protect. Retain.**  
> An AI-powered decision engine that identifies at-risk customers in real-time to prevent revenue leakage.

---

## 🚨 The Business Problem

Customer acquisition costs **5x more** than retention. For a mid-sized telecom company, a **2% monthly churn rate** can result in **$10M+ annual revenue loss**.

Existing rule-based systems (e.g., "Customer complained > 3 times") are reactive. By the time a rule triggers, the customer is already gone.

## 💡 The Solution

**ChurnGuard** is a proactive risk intelligence engine. It doesn't just "predict churn"; it acts as a **Revenue Firewall**.

*   **Real-time Risk Scoring**: Assessment in **<50ms latency**.
*   **Confidence-Aware AI**: Refuses to guess on uncertain cases ("Human-in-the-loop").
*   **Explainable Decisions**: Tells support agents *why* a customer is at risk.

---

## 🏗️ System Architecture

This is not a script. It is a microservice architecture designed for scale.

```mermaid
graph TD
    Client[Web/Mobile App] -->|JSON Request| API[Flask API Gateway]
    API -->|Schema Validation| Validator
    Validator -->|Valid Data| Model[Inference Engine (AdaBoost)]
    
    subgraph "Decision Engine"
        Model -->|Probabilities| Filter[Confidence Filter]
        Filter -->|High Confidence| Action[Automated Offer]
        Filter -->|Low Confidence (45-55%)| Review[Manual Review Queue]
        Filter -->|Low Risk| Log[Audit Log]
    end
    
    Action --> Database[(Operational DB)]
    Review --> Database
```

### Engineering Stack

| Component | Technology | Why? |
|-----------|------------|------|
| **API Gateway** | Flask (Python) | Lightweight, fast prototyping for REST endpoints. |
| **Inference** | scikit-learn | Low latency (<50ms) compared to deep learning overkill. |
| **Processing** | Multiprocessing | Parallel training for rapid model iteration. |
| **Safety** | Confidence Thresholds | Prevents False Positives (customer annoyance). |

---

## ⚡ Key Capabilities

### 1. Confidence-Aware Inference
The system knows when it doesn't know.

```json
// Response for Uncertain Case
{
    "decision": "NO_DECISION",
    "churn_probability": 0.5213,
    "confidence_level": "LOW_CONFIDENCE_MANUAL_REVIEW",
    "inference_latency_ms": 12.45
}
```

### 2. Multi-Paradigm Intelligence
While **Retention (Classification)** is the core module, the platform scales to:
*   **Revenue Forecasting** (Time Series Module)
*   **Asset Valuation** (Regression Module)

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   1GB RAM (Inference is lightweight)

### Quick Start (Production Mode)

```bash
# 1. Clone Repository
git clone https://github.com/your-repo/churn-guard.git
cd churn-guard

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Train Champion Models (Parallel)
python src/main_parallel.py

# 4. Start API Gateway
python src/api.py
```

### Testing the API

```bash
curl -X POST http://localhost:5000/api/predict/churn \
     -H "Content-Type: application/json" \
     -d '{ "features": [12, 65.5, 780.0, 0, 2, 0, 0, 1, 1, 2, 1] }'
```

---

## ⚖️ Ethics & Compliance

We take AI safety seriously. This system includes:
*   **Bias Auditing**: Protected attributes (race, gender) are excluded.
*   **Fail-Safe Mode**: API degrades gracefully if model artifacts are missing.
*   See `docs/ETHICS.md` for our full risk framework.

---

## 📈 Performance

| Metric | Score | Business Impact |
|--------|-------|-----------------|
| **ROC-AUC** | **0.725** | Strong ability to discriminate risk profiles. |
| **F1-Score** | **0.638** | Balanced precision/recall to minimize wasted offers. |
| **Latency** | **~15ms** | Real-time decisioning during customer calls. |

---

**© 2026 ChurnGuard Intelligence Team**