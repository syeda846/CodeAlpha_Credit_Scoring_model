from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_PATH = ROOT / "data" / "german_credit_data.csv"
MODEL_ARTIFACTS = {
    "logistic": {
        "model": MODELS_DIR / "logistic.joblib",
        "metrics": REPORTS_DIR / "metrics_logistic.json",
    },
    "decision_tree": {
        "model": MODELS_DIR / "decision_tree.joblib",
        "metrics": REPORTS_DIR / "metrics_decision_tree.json",
    },
    "random_forest": {
        "model": MODELS_DIR / "random_forest.joblib",
        "metrics": REPORTS_DIR / "metrics_random_forest.json",
    },
}


def resolve_paths(model_type: str) -> tuple[Path, Path]:
    model_path = MODEL_ARTIFACTS[model_type]["model"]
    metrics_path = MODEL_ARTIFACTS[model_type]["metrics"]

    # Backward-compatibility with older single-artifact naming.
    if not model_path.exists():
        legacy_model = MODELS_DIR / "credit_scoring_pipeline.joblib"
        if legacy_model.exists():
            model_path = legacy_model
    if not metrics_path.exists():
        legacy_metrics = REPORTS_DIR / "metrics.json"
        if legacy_metrics.exists():
            metrics_path = legacy_metrics

    return model_path, metrics_path


@st.cache_data
def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def label_from_prediction(
    pred_int: int,
    target_mapping: dict[str, int] | None,
) -> str:
    if not target_mapping:
        return str(pred_int)

    inverse = {v: k for k, v in target_mapping.items()}
    return inverse.get(pred_int, str(pred_int))


def risk_tier(prob: float, threshold: float) -> tuple[str, str]:
    if prob >= threshold + 0.15:
        return "HIGH RISK", "error"
    if prob >= threshold:
        return "MEDIUM RISK", "warning"
    return "LOW RISK", "success"


def status_chip(value: float) -> str:
    if value >= 0.70:
        return "GOOD"
    if value >= 0.50:
        return "WATCH"
    return "WEAK"


def status_color(value: float) -> str:
    if value >= 0.70:
        return "green"
    if value >= 0.50:
        return "orange"
    return "red"


def render_header(metrics_path: Path) -> None:
    st.title("Credit Scoring Dashboard")
    last_updated = datetime.fromtimestamp(metrics_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Professional model monitoring and prediction view. Last updated: {last_updated}")


def apply_theme_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 12px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.01);
            min-height: 100px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
        }
        .kpi-chip {
            font-size: 0.80rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .panel-subtitle {
            color: #9aa4b2;
            font-size: 0.92rem;
            margin-top: -4px;
            margin-bottom: 10px;
        }
        @media (max-width: 900px) {
            h1 { font-size: 1.8rem !important; }
            h2, h3 { font-size: 1.25rem !important; }
            div[data-testid="stMetric"] { min-height: 90px; padding: 8px 10px; }
            button[kind="primary"], button[kind="secondary"] {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_model_summary(metrics: dict, active_threshold: float) -> None:
    st.subheader("Model Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Type", str(metrics.get("model_type", "N/A")))
    c2.metric("Threshold", f"{active_threshold:.2f}")
    c3.metric("Target Column", str(metrics.get("target_column", "N/A")))

    st.markdown("**Target Mapping**")
    st.json(metrics.get("target_mapping", {}))


def render_performance_cards(metrics: dict) -> None:
    st.subheader("Performance Cards")
    #st.markdown("<div class='panel-subtitle'>Color context: GREEN=GOOD, ORANGE=WATCH, RED=WEAK</div>", unsafe_allow_html=True)
    cards = [
        ("Accuracy", float(metrics.get("accuracy", 0.0))),
        ("Precision", float(metrics.get("precision", 0.0))),
        ("Recall", float(metrics.get("recall", 0.0))),
        ("F1", float(metrics.get("f1", 0.0))),
        ("ROC-AUC", float(metrics.get("roc_auc", 0.0))),
        ("PR-AUC", float(metrics.get("pr_auc", 0.0))),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)

    for idx, (label, value) in enumerate(cards):
        col = row1[idx] if idx < 3 else row2[idx - 3]
        col.metric(label, f"{value:.3f}")
        chip = status_chip(value)
        color = status_color(value)
        col.markdown(f"<span class='kpi-chip' style='color:{color}'>{chip}</span>", unsafe_allow_html=True)

    if float(metrics.get("recall", 0.0)) >= 0.85 and float(metrics.get("precision", 0.0)) < 0.50:
        st.warning("Model is recall-heavy: catches more bad-risk cases, but with more false positives.")


def build_confusion_figure(metrics: dict):
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Good (0)", "Actual Bad (1)"],
        columns=["Pred Good (0)", "Pred Bad (1)"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix: Good vs Bad")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    return fig


def render_confusion_matrix(metrics: dict) -> None:
    st.subheader("Confusion Matrix")
    fig = build_confusion_figure(metrics)
    st.pyplot(fig)
    plt.close(fig)


def build_class_distribution_figure(metrics: dict):
    dist = metrics.get("class_distribution", {})
    if not dist:
        return None

    dist_series = pd.Series({str(k): int(v) for k, v in dist.items()})
    dist_df = dist_series.rename("count").reset_index().rename(columns={"index": "class"})
    dist_df["class"] = dist_df["class"].map({"0": "Good (0)", "1": "Bad (1)"}).fillna(dist_df["class"])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.barplot(data=dist_df, x="class", y="count", palette="Blues_d", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def render_class_distribution(metrics: dict) -> None:
    st.subheader("Class Distribution")
    fig = build_class_distribution_figure(metrics)
    if fig is None:
        st.info("No class distribution found in metrics.json.")
        return
    st.pyplot(fig)
    plt.close(fig)


def render_model_stability(metrics: dict) -> None:
    st.subheader("Model Stability")
    cv_mean = float(metrics.get("cv_roc_auc_mean", 0.0))
    cv_std = float(metrics.get("cv_roc_auc_std", 0.0))
    c1, c2 = st.columns(2)
    c1.metric("CV ROC-AUC Mean", f"{cv_mean:.3f}")
    c2.metric("CV ROC-AUC Std", f"{cv_std:.3f}")
    if cv_std > 0.08:
        st.warning("Higher CV standard deviation suggests less stable performance across folds.")
    else:
        st.info("CV variance looks acceptable for this baseline model.")


def render_prediction_form(data: pd.DataFrame) -> None:
    st.subheader("Prediction Form")
    st.caption("Enter applicant features manually to get probability + final class.")

    # API prediction should use only feature columns; remove known non-feature columns if present.
    drop_cols = [c for c in ["Unnamed: 0", "Risk"] if c in data.columns]
    features_df = data.drop(columns=drop_cols).copy()
    form_key = f"prediction_form_{st.session_state.get('form_version', 0)}"

    with st.form(form_key):
        st.markdown("#### Applicant Inputs")
        user_input: dict[str, object] = {}
        left_col, right_col = st.columns(2)

        for idx, col in enumerate(features_df.columns):
            col_series = features_df[col]
            input_col = left_col if idx % 2 == 0 else right_col
            if pd.api.types.is_numeric_dtype(col_series):
                default_value = float(col_series.median())
                min_value = float(col_series.min())
                max_value = float(col_series.max())
                help_text = "Value in months." if col.lower() == "duration" else None
                user_input[col] = input_col.number_input(
                    label=col,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    help=help_text,
                )
            else:
                values = sorted(col_series.dropna().astype(str).unique().tolist())
                default_idx = 0
                user_input[col] = input_col.selectbox(col, options=values, index=default_idx)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    # 1) Build the payload with the exact field names expected by FastAPI.
    # We map UI columns (which contain spaces) to API keys (which use underscores).
    payload = {
        "Age": int(user_input["Age"]),
        "Sex": str(user_input["Sex"]),
        "Job": int(user_input["Job"]),
        "Housing": str(user_input["Housing"]),
        "Saving_accounts": str(user_input["Saving accounts"]),
        "Checking_account": str(user_input["Checking account"]),
        "Credit_amount": int(user_input["Credit amount"]),
        "Duration": int(user_input["Duration"]),
        "Purpose": str(user_input["Purpose"]),
    }

    # 2) Send data to the running FastAPI service and parse its JSON response.
    input_df = pd.DataFrame([user_input])
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException:
        st.error("Connection Error: Is your FastAPI server running at http://127.0.0.1:8000?")
        st.write("Input prepared for API:")
        st.dataframe(input_df)
        return
    except ValueError:
        st.error("API Error: Received an invalid JSON response from the prediction endpoint.")
        st.write("Input prepared for API:")
        st.dataframe(input_df)
        return

    # 3) Read API output and present decision + probability in the dashboard.
    decision = str(result.get("decision", "Unknown"))
    risk_probability = float(result.get("risk_probability", 0.0))

    st.markdown("### Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Risk Probability", f"{risk_probability:.4f}")
    c2.metric("Decision", decision)

    if decision.lower() == "approve":
        st.success(f"✅ Approved! Risk: {risk_probability:.4f}")
    elif decision.lower() == "reject":
        st.error(f"❌ Rejected. Risk: {risk_probability:.4f}")
    else:
        st.warning(f"Decision from API: {decision} | Risk: {risk_probability:.4f}")

    st.write("Input used for prediction:")
    st.dataframe(input_df)

    prediction_export = pd.DataFrame(
        [
            {
                **user_input,
                "risk_probability": round(risk_probability, 6),
                "decision": decision,
            }
        ]
    )
    st.download_button(
        label="Download Prediction CSV",
        data=prediction_export.to_csv(index=False),
        file_name="prediction_result.csv",
        mime="text/csv",
    )

    if st.button("Reset form"):
        st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
        st.rerun()


def render_download_metrics(metrics: dict) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    st.download_button(
        label="Download metrics.json",
        data=metrics_json,
        file_name="metrics.json",
        mime="application/json",
    )


def figure_to_base64_png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_snapshot_report_html(metrics: dict, active_threshold: float) -> str:
    conf_fig = build_confusion_figure(metrics)
    class_fig = build_class_distribution_figure(metrics)
    conf_b64 = figure_to_base64_png(conf_fig)
    plt.close(conf_fig)
    class_img_html = "<p>Class distribution not available.</p>"
    if class_fig is not None:
        class_b64 = figure_to_base64_png(class_fig)
        plt.close(class_fig)
        class_img_html = f"<img src='data:image/png;base64,{class_b64}' alt='Class Distribution' style='max-width:100%;border-radius:8px;'/>"

    target_mapping_html = json.dumps(metrics.get("target_mapping", {}), indent=2)
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Credit Scoring Snapshot Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
        h1, h2 {{ margin-bottom: 8px; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
        .metric {{ font-size: 1.4rem; font-weight: bold; }}
        .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        pre {{ background: #f6f8fa; padding: 8px; border-radius: 8px; }}
      </style>
    </head>
    <body>
      <h1>Credit Scoring Dashboard Snapshot</h1>
      <p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
      <div class="grid">
        <div class="card"><div>Model Type</div><div class="metric">{metrics.get("model_type", "N/A")}</div></div>
        <div class="card"><div>Threshold</div><div class="metric">{float(active_threshold):.2f}</div></div>
        <div class="card"><div>Target</div><div class="metric">{metrics.get("target_column", "N/A")}</div></div>
      </div>
      <div class="grid">
        <div class="card"><div>Accuracy</div><div class="metric">{float(metrics.get("accuracy", 0.0)):.3f}</div></div>
        <div class="card"><div>Precision</div><div class="metric">{float(metrics.get("precision", 0.0)):.3f}</div></div>
        <div class="card"><div>Recall</div><div class="metric">{float(metrics.get("recall", 0.0)):.3f}</div></div>
        <div class="card"><div>F1</div><div class="metric">{float(metrics.get("f1", 0.0)):.3f}</div></div>
        <div class="card"><div>ROC-AUC</div><div class="metric">{float(metrics.get("roc_auc", 0.0)):.3f}</div></div>
        <div class="card"><div>PR-AUC</div><div class="metric">{float(metrics.get("pr_auc", 0.0)):.3f}</div></div>
      </div>
      <h2>Target Mapping</h2>
      <pre>{target_mapping_html}</pre>
      <h2>Charts</h2>
      <div class="row">
        <div><img src="data:image/png;base64,{conf_b64}" alt="Confusion Matrix" style="max-width:100%;border-radius:8px;"/></div>
        <div>{class_img_html}</div>
      </div>
      <h2>Model Stability</h2>
      <p>CV ROC-AUC Mean: {float(metrics.get("cv_roc_auc_mean", 0.0)):.3f} | CV ROC-AUC Std: {float(metrics.get("cv_roc_auc_std", 0.0)):.3f}</p>
      <h2>Limitations</h2>
      <p>{metrics.get("professional_note", "")}</p>
    </body>
    </html>
    """


def render_snapshot_export(metrics: dict, active_threshold: float) -> None:
    st.subheader("Export")
    st.caption("One-click downloadable snapshot report of current dashboard state.")
    report_html = build_snapshot_report_html(metrics, active_threshold)
    st.download_button(
        label="Export Dashboard Screenshot Report (.html)",
        data=report_html,
        file_name="dashboard_snapshot_report.html",
        mime="text/html",
    )


def render_trust_panel(metrics: dict) -> None:
    with st.expander("Assumptions, Limits, and Responsible Use"):
        st.markdown(
            """
            - This dashboard is an educational baseline and not financial advice.
            - Model quality depends on dataset quality, representativeness, and label correctness.
            - Threshold settings control trade-offs (false positives vs false negatives).
            - Monitor model drift and re-train periodically in real deployments.
            """
        )
        st.caption(metrics.get("professional_note", ""))


def main() -> None:
    st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
    apply_theme_css()

    if not DATA_PATH.exists():
        st.error(f"Missing dataset file: {DATA_PATH}")
        st.stop()

    if "form_version" not in st.session_state:
        st.session_state["form_version"] = 0

    model_type = st.sidebar.selectbox(
        "Model Type",
        options=["logistic", "decision_tree", "random_forest"],
        index=2,
    )
    _, metrics_path = resolve_paths(model_type)
    if not metrics_path.exists():
        st.error(
            f"Missing metrics for '{model_type}'. Expected: {metrics_path}. "
            f"Run training for this model first."
        )
        st.stop()
    render_header(metrics_path)
    metrics = load_metrics(metrics_path)
    data = load_dataset(DATA_PATH)
    default_threshold = float(metrics.get("threshold", 0.5))
    active_threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, default_threshold, 0.01)

    section = st.sidebar.radio(
        "Navigation",
        ["Overview", "Performance", "Prediction", "About"],
        index=0,
    )

    if section == "Overview":
        # Row 1: summary cards
        render_model_summary(metrics, active_threshold)
        st.divider()
        # Row 2: KPI strip with status context
        render_performance_cards(metrics)
        st.divider()
        # Row 3: charts + stability
        left, right = st.columns(2)
        with left:
            render_confusion_matrix(metrics)
        with right:
            render_class_distribution(metrics)
            st.markdown("")
            render_model_stability(metrics)
        st.divider()
        render_download_metrics(metrics)
        render_snapshot_export(metrics, active_threshold)

    elif section == "Performance":
        render_performance_cards(metrics)
        st.divider()
        left, right = st.columns(2)
        with left:
            render_confusion_matrix(metrics)
        with right:
            render_class_distribution(metrics)
            st.markdown("")
            render_model_stability(metrics)
        st.divider()
        render_download_metrics(metrics)
        render_snapshot_export(metrics, active_threshold)

    elif section == "Prediction":
        render_prediction_form(data)

    else:
        st.subheader("About")
        st.write("This dashboard helps review model quality and run manual credit-risk predictions.")
        render_model_summary(metrics, active_threshold)
        st.divider()
        render_trust_panel(metrics)
        st.divider()
        render_download_metrics(metrics)
        render_snapshot_export(metrics, active_threshold)


if __name__ == "__main__":
    main()
