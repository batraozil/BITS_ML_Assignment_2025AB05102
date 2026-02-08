# models/train_models.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)

from xgboost import XGBClassifier

DATA_CSV = Path("data/har.csv")
ART_DIR = Path("models/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def multiclass_auc_ovr(y_true_int, y_proba, n_classes: int) -> float:
    # roc_auc_score for multiclass requires binarized y
    Y = label_binarize(y_true_int, classes=list(range(n_classes)))
    return roc_auc_score(Y, y_proba, average="weighted", multi_class="ovr")

def evaluate_model(model, X_test, y_test_int, n_classes: int):
    y_pred = model.predict(X_test)

    # predict_proba exists for all our chosen models
    y_proba = model.predict_proba(X_test)
    auc = multiclass_auc_ovr(y_test_int, y_proba, n_classes)

    return {
        "Accuracy": float(accuracy_score(y_test_int, y_pred)),
        "AUC": float(auc),
        "Precision": float(precision_score(y_test_int, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test_int, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_test_int, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test_int, y_pred)),
    }

def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing {DATA_CSV}. Run: python models/build_dataset.py")

    df = pd.read_csv(DATA_CSV)
    if "Activity" not in df.columns:
        raise ValueError("Expected target column named 'Activity' in har.csv")

    X = df.drop(columns=["Activity"])
    y = df["Activity"].astype(str)

    # Encode labels to ints for AUC + MCC stability
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=RANDOM_STATE, stratify=y_int
    )

    # Preprocessor: scaling helps LR + KNN + NB; harmless for trees
    preprocessor = StandardScaler()

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", preprocessor),
            ("clf", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": Pipeline([
            ("scaler", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes (Gaussian)": Pipeline([
            ("scaler", preprocessor),
            ("clf", GaussianNB())
        ]),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = evaluate_model(model, X_test, y_test, n_classes)
        metrics[name] = scores

        # Save model
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        joblib.dump(model, ART_DIR / f"{safe_name}.pkl")

    # Save label encoder
    joblib.dump(le, ART_DIR / "label_encoder.pkl")

    # Save metrics
    with open(ART_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature columns (for upload validation)
    with open(ART_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    print("Training done. Artifacts saved in models/artifacts/")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
