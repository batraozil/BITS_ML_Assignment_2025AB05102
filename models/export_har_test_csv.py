# models/export_har_test_csv.py
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/UCI HAR Dataset")
OUT_CSV = Path("data/har_test.csv")

def main():
    features = pd.read_csv(RAW_DIR / "features.txt", sep=r"\s+", header=None)[1].tolist()
    labels = pd.read_csv(RAW_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
    id_to_activity = dict(zip(labels["id"], labels["activity"]))

    X_test = pd.read_csv(RAW_DIR / "test/X_test.txt", sep=r"\s+", header=None)
    y_test = pd.read_csv(RAW_DIR / "test/y_test.txt", sep=r"\s+", header=None, names=["ActivityId"])

    X_test.columns = features
    y_test["Activity"] = y_test["ActivityId"].map(id_to_activity)

    df = pd.concat([X_test, y_test["Activity"]], axis=1)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved test CSV: {OUT_CSV}")

if __name__ == "__main__":
    main()
