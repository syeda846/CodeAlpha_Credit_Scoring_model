## Credit Scoring Model (Internship Task)

This project implements an end-to-end **credit scoring classification workflow** using:
- Logistic Regression
- Decision Tree
- Random Forest

It includes:
- data loading and validation
- preprocessing with a leakage-safe `Pipeline`
- stratified train/test split and cross-validation
- evaluation with Precision, Recall, F1, ROC-AUC, and confusion matrix
- model and metrics artifact export

## Project Structure

- `data/german_credit_data.csv` - dataset
- `notebooks/credit_model.ipynb` - exploration notebook
- `src/train_credit_model.py` - professional training script
- `reports/metrics.json` - exported run metrics
- `models/` - saved model artifacts

## Important Note About Target

The current dataset does not contain a true repayment/default label by default.  
So the script supports two modes:

1. **Real mode (recommended):** use a real target column (e.g., `Risk`) if available.
2. **Demo mode:** create a proxy target using credit amount threshold (`--create-proxy-target`).

Demo mode is only for learning and does **not** represent real-world credit risk modeling.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### PowerShell (Windows) recommended

If you keep hitting PowerShell command issues (like `&&` not working), use the included runner:

```powershell
.\run.ps1 -ModelType random_forest -DataPath data/german_credit_data.csv -CreateProxyTarget
```

## Training Examples

### 1) Logistic Regression (with real target column)
```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --target-column Risk --model-type logistic
```

### 2) Decision Tree (demo proxy target)
```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --create-proxy-target --model-type decision_tree
```

### 3) Random Forest (demo proxy target)
```bash
python src/train_credit_model.py --data-path data/german_credit_data.csv --create-proxy-target --model-type random_forest
```

## Output Files

After training:
- model pipeline (per model):
  - `models/logistic.joblib`
  - `models/decision_tree.joblib`
  - `models/random_forest.joblib`
- metrics report (per model):
  - `reports/metrics_logistic.json`
  - `reports/metrics_decision_tree.json`
  - `reports/metrics_random_forest.json`

Dashboard supports live selection of model + threshold from the sidebar.

## Evaluation Metrics

The script reports:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report
- Cross-validation ROC-AUC (mean/std)

## Internship Task Mapping

This repository satisfies the requested items:
- creditworthiness prediction via classification
- feature preprocessing and engineering pipeline
- multiple model options (Logistic Regression, Decision Tree, Random Forest)
- metric-based performance evaluation (Precision, Recall, F1, ROC-AUC)

## 🚀 Quick Start
To launch the interactive dashboard and test the models:
```powershell
streamlit run app/main.py
