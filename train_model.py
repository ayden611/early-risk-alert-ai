import json
import os
from typing import Optional, List

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


CSV_PATH = "data/health_data.csv"      # keep this path
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _make_exercise_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Converts exercise column to numeric 0/1/2.
    Accepts:
      - already numeric
      - strings: Low/Moderate/High (case-insensitive)
    """
    s = df[col]
    if s.dtype == "object":
        s2 = (
            s.astype(str)
            .str.strip()
            .str.lower()
            .map({"low": 0, "moderate": 1, "medium": 1, "high": 2})
        )
        return pd.to_numeric(s2, errors="coerce").fillna(0).clip(0, 2).astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 2).astype(int)


def _make_target_binary(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Converts target to 0/1.
    Accepts:
      - numeric 0/1
      - strings like: "low risk"/"high risk", "low"/"high", "0"/"1", "false"/"true"
    """
    y = df[col]
    if y.dtype == "object":
        s = y.astype(str).str.strip().str.lower()
        s = s.replace(
            {
                "high_risk": "high risk",
                "low_risk": "low risk",
                "high": "high risk",
                "low": "low risk",
                "1": "high risk",
                "0": "low risk",
                "true": "high risk",
                "false": "low risk",
            }
        )
        mapped = s.map({"low risk": 0, "high risk": 1})
        y_num = pd.to_numeric(mapped, errors="coerce")
    else:
        y_num = pd.to_numeric(y, errors="coerce")

    y_num = y_num.fillna(0).astype(int)
    # force anything not 0 to 1 (in case values like 2 appear)
    y_num = (y_num > 0).astype(int)
    return y_num


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = _normalize_cols(df)

    print("\n✅ Columns in dataset:", df.columns.tolist())

    # --- detect columns (these cover most datasets) ---
    age_c = _first_existing(df, ["age"])
    bmi_c = _first_existing(df, ["bmi"])
    ex_c = _first_existing(df, ["exercise_level", "exercise", "activity_level", "activity"])
    sys_c = _first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = _first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c  = _first_existing(df, ["heart_rate", "hr", "pulse"])

    # target column: supports your custom csv AND common cardio datasets
    y_c = _first_existing(df, ["target", "risk", "label", "risk_label", "cardio"])

    missing = [name for name, col in {
        "age": age_c,
        "bmi": bmi_c,
        "exercise": ex_c,
        "systolic_bp": sys_c,
        "diastolic_bp": dia_c,
        "heart_rate": hr_c,
        "target": y_c,
    }.items() if col is None]

    if missing:
        raise ValueError(
            "Missing required columns in CSV: " + ", ".join(missing) +
            "\nFix: open data/health_data.csv and make sure it contains these fields "
            "(or rename the header to match one of the supported names)."
        )

    # --- clean minimal ---
    df = df.copy()
    df["exercise_level"] = _make_exercise_numeric(df, ex_c)
    df = _to_numeric(df, [age_c, bmi_c, sys_c, dia_c, hr_c])
    df["target"] = _make_target_binary(df, y_c)

    # keep only needed cols and drop rows with missing values
    keep = [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, "target"]
    df = df[keep].dropna()

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].values
    y = df["target"].values

    counts = pd.Series(y).value_counts().to_dict()
    print("✅ Class counts (0=low, 1=high):", counts)

    # must have both classes
    if len(counts) < 2:
        raise ValueError(
            f"Your dataset only has ONE class after processing: {counts}. "
            "You need both low-risk (0) and high-risk (1) rows to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # simple strong baseline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    # metrics
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    metrics = {
        "rows_used": int(len(df)),
        "class_counts": counts,
        "roc_auc": auc,
        "classification_report": report
    }

    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved model -> {MODEL_OUT}")
    print(f"✅ Saved metrics -> {METRICS_OUT}")
    print(f"✅ ROC AUC -> {auc:.4f}\n")


if __name__ == "__main__":
    main()
