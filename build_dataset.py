# models/build_dataset.py
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/UCI HAR Dataset")
OUT_CSV = Path("data/har.csv")

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Missing dataset folder: {RAW_DIR}\n"
            "Download & unzip the UCI HAR Dataset into data/raw/"
        )

    # Feature names
    features = pd.read_csv(RAW_DIR / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    feature_names = features["name"].tolist()

    # Activity labels mapping
    act_labels = pd.read_csv(RAW_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
    id_to_activity = dict(zip(act_labels["id"], act_labels["activity"]))

    def load_split(split: str) -> pd.DataFrame:
        X = pd.read_csv(RAW_DIR / split / f"X_{split}.txt", sep=r"\s+", header=None)
        y = pd.read_csv(RAW_DIR / split / f"y_{split}.txt", sep=r"\s+", header=None, names=["ActivityId"])
        X.columns = feature_names
        df = pd.concat([X, y], axis=1)
        df["Activity"] = df["ActivityId"].map(id_to_activity)
        df.drop(columns=["ActivityId"], inplace=True)
        return df

    train_df = load_split("train")
    test_df = load_split("test")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} | shape={full_df.shape}")

if __name__ == "__main__":
    main()
