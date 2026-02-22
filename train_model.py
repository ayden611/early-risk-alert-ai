# train_model.py
# Clean, reliable trainer for Early Risk Alert AI
# - Reads data/health_data.csv
# - Auto-detects common column names
# - Builds numeric exercise_level (Low/Moderate/High -> 0/1/2)
# - Builds binary target (low risk=0, high risk=1) from common labels
# - Trains Logistic Regression pipeline (scaler + classifier)
# - Saves demo_model.pkl + model_metrics.json
# - No warning spam

import json
import os
import joblib
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

CSV_PATH = os.path.join("data", "health_data.csv")
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has numeric exercise_level: Low=0, Moderate/Medium=1, High=2
    Accepts: exercise_level / exercise / activity / activity_level
    If missing, creates exercise_level=0 (Low).
    """
    df = df.copy()
    src = first_existing(df, ["exercise_level", "exercise", "activity", "activity_level"])

    if src is None:
        df["exercise_level"] = 0
        return df

    if src != "exercise_level":
        df["exercise_level"] = df[src]
    else:
        df["exercise_level"] = df["exercise_level"]

    # Convert string labels
    if df["exercise_level"].dtype == "object":
        s = (
            df["exercise_level"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["exercise_level"] = s.map(
            {"low": 0, "moderate": 1, "medium": 1, "high": 2}
        )

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0)
    df["exercise_level"] = df["exercise_level"].clip(0, 2).astype(int)
    return df


def make_binary_target(y_raw: pd.Series) -> pd.Series:
    """
    Converts common target formats to 0/1.
    Accepts:
      - numeric 0/1
      - strings like "low risk"/"high risk", "low"/"high", "true"/"false"
      - column names like risk/label/target/risk_label/cardio
    """
    y = y_raw.copy()

    if y.dtype == "object":
        s = y.astype(str).str.strip().str.lower()

        # Normalize common variants
        s = s.replace(
            {
                "high_risk": "high risk",
                "low_risk": "low risk",
                "1": "high risk",
                "0": "low risk",
                "true": "high risk",
                "false": "low risk",
                "high": "high risk",
                "low": "low risk",
                "yes": "high risk",
                "no": "low risk",
            }
        )

        mapped = s.map({"low risk": 0, "high risk": 1})
        # If mapping fails (NaN), try numeric coercion
        y = mapped

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    # Force to 0/1
    y = (y > 0).astype(int)
    return y


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = normalize_cols(df)
    df = build_exercise_level(df)

    # Feature columns (supports multiple naming styles)
    age_c = first_existing(df, ["age"])
    bmi_c = first_existing(df, ["bmi"])
    sys_c = first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c = first_existing(df, ["heart_rate", "hr"])

    # Target column (supports several naming styles)
    y_c = first_existing(df, ["target", "risk", "label", "risk_label", "cardio"])

    missing = [name for name, col in {
        "age": age_c,
        "bmi": bmi_c,
        "systolic_bp": sys_c,
        "diastolic_bp": dia_c,
        "heart_rate": hr_c,
        "target": y_c,
    }.items() if col is None]

    if missing:
        raise ValueError(
            "Missing required columns in CSV: "
            + ", ".join(missing)
            + "\nTip: open data/health_data.csv and check the header names."
        )

    use_cols = [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c]
    df = df[use_cols].copy()

    # Convert numeric + drop rows with missing essentials
    df = to_numeric(df, [age_c, bmi_c, sys_c, dia_c, hr_c])
    df = df.dropna(subset=[age_c, bmi_c, sys_c, dia_c, hr_c, y_c])

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].values
    y = make_binary_target(df[y_c])

    counts = y.value_counts().to_dict()
    print("TARGET COUNTS:", counts)

    # If only 1 class exists, you can't train a classifier properly
    if len(counts) < 2:
        raise ValueError(
            f"Your dataset only has ONE class after cleaning (counts={counts}). "
            "You need both low + high risk rows to train."
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Pipeline model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        )),
    ])

    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    proba = None
    auc = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None

    report = classification_report(y_test, preds, zero_division=0, output_dict=True)

    metrics = {
        "rows_used": int(len(df)),
        "feature_columns": [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c],
        "target_column": y_c,
        "target_counts_after_cleaning": counts,
        "roc_auc": auc,
        "classification_report": report,
    }

    # Save
    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ TRAIN OK")
    print("✅ Saved:", MODEL_OUT)
    print("✅ Saved metrics:", METRICS_OUT)
    print("✅ ROC AUC:", auc)


if __name__ == "__main__":
    main()
