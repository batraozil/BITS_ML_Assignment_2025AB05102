# app.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

ART_DIR = Path("models/artifacts")

st.set_page_config(page_title="HAR Classification Models", layout="wide")

@st.cache_resource
def load_artifacts():
    metrics = json.loads((ART_DIR / "metrics.json").read_text())
    feature_cols = json.loads((ART_DIR / "feature_columns.json").read_text())
    le = joblib.load(ART_DIR / "label_encoder.pkl")

    # Load all models
    models = {}
    for p in ART_DIR.glob("*.pkl"):
        if p.name in ["label_encoder.pkl"]:
            continue
        models[p.stem] = joblib.load(p)

    return metrics, feature_cols, le, models

def pretty_name(stem: str) -> str:
    mapping = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "kNN",
        "naive_bayes_gaussian": "Naive Bayes (Gaussian)",
        "random_forest_ensemble": "Random Forest (Ensemble)",
        "xgboost_ensemble": "XGBoost (Ensemble)",
    }
    return mapping.get(stem, stem)

def plot_confusion(cm, class_names):
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    plt.colorbar(ax.images[0], ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # annotate counts
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def main():
    st.title("Human Activity Recognition (HAR) – Multi-Model Classification")

    if not ART_DIR.exists():
        st.error("Missing models/artifacts. Run training locally and commit artifacts to GitHub.")
        st.stop()

    metrics, feature_cols, le, models = load_artifacts()

    # ---- Metrics Table ----
    st.subheader("Model Comparison (Evaluation Metrics)")
    metrics_df = pd.DataFrame(metrics).T[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].sort_values("F1", ascending=False)
    st.dataframe(metrics_df, use_container_width=True)

    st.divider()

    # ---- Model Selection ----
    st.subheader("Test a Model")
    model_keys = sorted(models.keys())
    selected_stem = st.selectbox(
        "Choose a model",
        model_keys,
        format_func=pretty_name
    )
    model = models[selected_stem]

    st.write("Upload a CSV with the **same feature columns** as training data (Refer to the Git data folder for test file - File Name - har_test.csv).")
    st.markdown("Check out the sample test file here (https://github.com/batraozil/BITS_ML_Assignment_2025AB05102/blob/769be21c51a7fd6857b5509ba50f0315b14a1512/data/har_test.csv). Download the file and use it for model testing")


    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a test CSV to run predictions.")
        st.stop()

    df = pd.read_csv(uploaded)

    has_label = "Activity" in df.columns
    if has_label:
        y_true_text = df["Activity"].astype(str)
        df_features = df.drop(columns=["Activity"])
    else:
        df_features = df

    # Validate columns
    missing = [c for c in feature_cols if c not in df_features.columns]
    extra = [c for c in df_features.columns if c not in feature_cols]

    if missing:
        st.error(f"Missing required columns ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
        st.stop()

    # Keep only expected columns in correct order
    X = df_features[feature_cols].copy()

    # Predict
    y_pred_int = model.predict(X)
    y_pred_text = le.inverse_transform(y_pred_int)

    out = pd.DataFrame({"PredictedActivity": y_pred_text})
    st.subheader("Predictions")
    st.dataframe(out.head(30), use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    # If user provided labels, show confusion matrix + report
    if has_label:
        try:
            y_true_int = le.transform(y_true_text)
        except Exception:
            st.error("Your `Activity` labels don’t match training labels. Use the same activity names as dataset.")
            st.stop()

        cm = confusion_matrix(y_true_int, y_pred_int)
        st.subheader("Confusion Matrix")
        fig = plot_confusion(cm, class_names=list(le.classes_))
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true_int, y_pred_int, target_names=list(le.classes_), zero_division=0)
        st.code(report)

if __name__ == "__main__":
    main()
