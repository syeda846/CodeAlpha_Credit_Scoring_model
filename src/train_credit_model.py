"""
Professional credit scoring training script.

Usage examples:
  python src/train_credit_model.py --data-path data/german_credit_data.csv --target-column Risk --model-type logistic
  python src/train_credit_model.py --data-path data/german_credit_data.csv --create-proxy-target --model-type random_forest
"""
#Necessary imports
from __future__ import annotations

import argparse
import json
from pathlib import Path

#Necessary libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


#Function to parse arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a professional credit scoring baseline model.")
    #`--data-path` is required so runs are explicit and reproducible (no hidden hardcoded paths).
    parser.add_argument("--data-path", type=str, required=True, help="Path to input CSV data.")
    parser.add_argument(
        "--target-column",
        type=str,
        default="Risk",
        help="Name of target column. Defaults to 'Risk'.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "decision_tree", "random_forest"],
        help="Classifier type to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size. Defaults to 0.2.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classification. Defaults to 0.5.",
    )
    parser.add_argument(
        "--create-proxy-target",
        action="store_true",
        help=(
            "Create a synthetic target from credit amount only for learning/demo. "
            "Not suitable for real credit scoring."
        ),
    )
    parser.add_argument(
        "--proxy-threshold",
        type=float,
        default=5000.0,
        help="Threshold for synthetic target if --create-proxy-target is enabled.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/credit_scoring_pipeline.joblib",
        help="Path for serialized pipeline. Default auto-routes to models/<model_type>.joblib.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="reports/metrics.json",
        help="Path for model metrics JSON report. Default auto-routes to reports/metrics_<model_type>.json.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="learning rate of the logistic model. Defaults to 0.01."
    )
    return parser.parse_args()


def normalize_binary_target(y: pd.Series) -> tuple[pd.Series, dict[str, int] | None]:
    """
    Ensure binary targets are numeric 0/1.

    If the dataset already has numeric labels, we keep them as-is.
    If it uses common credit labels like {'good','bad'}, we map them to 0/1
    so that probability-thresholding also produces compatible labels.
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int), None

    # Normalize to lowercase strings for robust matching.
    y_str = y.astype(str).str.strip().str.lower()
    unique = set(y_str.dropna().unique().tolist())

    # Common credit target conventions.
    if unique.issubset({"good", "bad"}) and len(unique) == 2:
        mapping = {"good": 0, "bad": 1}  # 1 = higher risk (bad), 0 = lower risk (good)
        return y_str.map(mapping).astype(int), mapping

    if unique.issubset({"0", "1"}) and len(unique) == 2:
        mapping = {"0": 0, "1": 1}
        return y_str.map(mapping).astype(int), mapping

    raise ValueError(
        f"Unsupported target labels: {sorted(unique)}. "
        "Provide a numeric 0/1 target or use common labels like 'good'/'bad'."
    )


def build_pipeline(X: pd.DataFrame, model_type: str, random_state: int) -> Pipeline:
    # Split features by dtype so we can apply the right preprocessing to each group.
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Ignore unseen categories at inference-time instead of crashing.
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_type == "logistic":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def validate_or_create_target(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.target_column in df.columns:
        return df

    if not args.create_proxy_target:
        # Professional default: don't silently invent a target.
        raise ValueError(
            f"Target column '{args.target_column}' not found in dataset. "
            "Provide a dataset with a true outcome label, or run with "
            "--create-proxy-target for a demo-only synthetic target."
        )

    if "Credit amount" not in df.columns:
        raise ValueError(
            "Cannot create proxy target because 'Credit amount' column is missing."
        )

    print(
        "[WARNING] Using synthetic proxy target from 'Credit amount'. "
        "This is demo-only and not a professional credit risk target."
    )
    df = df.copy()
    # Demo-only proxy target: this is NOT a real default/creditworthiness outcome label.
    df[args.target_column] = (df["Credit amount"] > args.proxy_threshold).astype(int)
    return df


def evaluate_at_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = validate_or_create_target(df, args)

    drop_columns = [col for col in ["Unnamed: 0"] if col in df.columns]
    df = df.drop(columns=drop_columns)

    X = df.drop(columns=[args.target_column])
    y_raw = df[args.target_column]
    y, target_mapping = normalize_binary_target(y_raw)

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        # Stratification preserves class balance in both splits (important for imbalanced targets).
        stratify=y,
    )

    pipeline = build_pipeline(
        X=X_train,
        model_type=args.model_type,
        random_state=args.random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_roc_auc = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        # ROC-AUC evaluates ranking quality across thresholds (common in credit scoring).
        scoring="roc_auc",
        n_jobs=None,
    )

    pipeline.fit(X_train, y_train)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_at_threshold(y_test, y_prob_test, args.decision_threshold)
    test_metrics["cv_roc_auc_mean"] = float(cv_roc_auc.mean())
    test_metrics["cv_roc_auc_std"] = float(cv_roc_auc.std())
    test_metrics["target_column"] = args.target_column
    test_metrics["model_type"] = args.model_type
    test_metrics["target_mapping"] = target_mapping
    test_metrics["rows"] = int(len(df))
    test_metrics["features"] = int(X.shape[1])
    test_metrics["class_distribution"] = {
        str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()
    }
    test_metrics["professional_note"] = (
        "If this model was trained with a proxy target, results are not "
        "representative of real-world credit risk."
    )

    # If default output names are used, save per-model artifacts so dashboard can switch models without retraining.
    if args.model_output == "models/credit_scoring_pipeline.joblib":
        model_output = Path(f"models/{args.model_type}.joblib")
    else:
        model_output = Path(args.model_output)

    if args.metrics_output == "reports/metrics.json":
        metrics_output = Path(f"reports/metrics_{args.model_type}.json")
    else:
        metrics_output = Path(args.metrics_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output)
    with metrics_output.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {model_output}")
    print(f"Metrics saved to: {metrics_output}")
    print(f"CV ROC-AUC mean/std: {test_metrics['cv_roc_auc_mean']:.4f} / {test_metrics['cv_roc_auc_std']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Threshold: {test_metrics['threshold']}")


if __name__ == "__main__":
    main()
