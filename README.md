## Credit Scoring Model (Internship Task)

This project implements an end-to-end credit scoring workflow with:
- model training (`logistic`, `decision_tree`, `random_forest`)
- model evaluation and metrics export
- FastAPI inference service (`/predict`)
- Streamlit dashboard for monitoring + manual predictions

## Project Structure

- `data/german_credit_data.csv` - input dataset
- `src/train_credit_model.py` - training + evaluation pipeline
- `src/api.py` - FastAPI prediction API (uses `models/random_forest.joblib`)
- `app/dashboard.py` - Streamlit monitoring and prediction UI
- `models/` - trained model artifacts
- `reports/` - exported metrics per model
- `requirements.txt` - dependencies

## Target Label Note

The dataset may not always include a true repayment/default label.

Supported modes:
1. **Real target mode (recommended):** provide a valid target column such as `Risk`.
2. **Demo mode:** generate a proxy label with `--create-proxy-target` using `Credit amount` and `--proxy-threshold`.

Proxy mode is for learning only and is not production-grade risk modeling.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models

### Logistic regression (real target)

```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --target-column Risk --model-type logistic
```

### Decision tree (proxy target demo)

```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --create-proxy-target --model-type decision_tree
```

### Random forest (proxy target demo)

```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --create-proxy-target --model-type random_forest
```

### Optional threshold tuning at train time

```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --create-proxy-target --model-type random_forest --decision-threshold 0.45
```

## Artifacts Generated

Per model:
- `models/logistic.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`

Per model metrics:
- `reports/metrics_logistic.json`
- `reports/metrics_decision_tree.json`
- `reports/metrics_random_forest.json`

## Run Inference API

Start FastAPI:

```bash
python src/api.py
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

Predict endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"Age\":35,\"Sex\":\"male\",\"Job\":2,\"Housing\":\"own\",\"Saving_accounts\":\"little\",\"Checking_account\":\"moderate\",\"Credit_amount\":5000,\"Duration\":24,\"Purpose\":\"radio/TV\"}"
```

Response shape:

```json
{
  "risk_probability": 0.3124,
  "decision": "Approve"
}
```

## Run Dashboard

Start Streamlit:

```bash
streamlit run app/dashboard.py
```

The dashboard supports:
- model selection (`logistic`, `decision_tree`, `random_forest`)
- threshold control from sidebar
- performance cards and confusion matrix
- class distribution and CV stability view
- manual prediction form (calls API at `http://127.0.0.1:8000/predict`)
- downloadable metrics JSON and HTML snapshot report

## Evaluation Metrics

`train_credit_model.py` exports:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report
- CV ROC-AUC mean/std

## Recommended Local Run Order

1. Train at least one model (preferably all 3 for dashboard switching).
2. Start API server: `python src/api.py`.
3. Start dashboard: `streamlit run app/dashboard.py`.
4. Open Streamlit UI and test the Prediction page.
