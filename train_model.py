import json
import os
from typing import Optional, List, Tuple

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# ---- Config ----
CSV_PATH = "data/health_data.csv"
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


# ---- Helpers ----
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has numeric exercise_level (0=Low,1=Moderate,2=High)
    Accepts: exercise_level / exercise / activity / activity_level
    """
    df = df.copy()
    src = first_col(df, ["exercise_level", "exercise", "activity", "activity_level"])

    if src is None:
        df["exercise_level"] = 0
        return df

    if src != "exercise_level":
        df["exercise_level"] = df[src]

    # Convert strings like "Low", "Moderate", "High"
    if df["exercise_level"].dtype == "object":
        s = df["exercise_level"].astype(str).str.strip().str.lower()
        df["exercise_level"] = s.map(
            {"low": 0, "moderate": 1, "medium": 1, "high": 2}
        )

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0)
    df["exercise_level"] = df["exercise_level"].clip(0, 2)
    return df


def make_binary_target(y_raw: pd.Series) -> pd.Series:
    """
    Converts common target formats into 0/1.
    Handles:
      - numeric already 0/1
      - numeric like 1/2 -> (y>1)
      - strings like "low risk"/"high risk", "low"/"high", "0"/"1", "true"/"false"
      - numeric scores with many unique values -> split by median (top half = 1)
    """
    y = y_raw.copy()

    if y.dtype == "object":
        s = y.astype(str).str.strip().str.lower()
        # normalize some variations
        s = s.replace(
            {
                "high_risk": "high risk",
                "low_risk": "low risk",
                "yes": "high risk",
                "no": "low risk",
            }
        )
        mapped = s.map(
            {
                "low risk": 0,
                "high risk": 1,
                "low": 0,
                "high": 1,
                "0": 0,
                "1": 1,
                "false": 0,
                "true": 1,
            }
        )
        y = mapped

    y = pd.to_numeric(y, errors="coerce")

    # If still mostly NaN, force to 0
    y = y.fillna(0)

    uniq = sorted(y.dropna().unique().tolist())

    # If already binary-ish
    if set(uniq).issubset({0, 1}):
        return y.astype(int)

    # If looks like 1/2 labels
    if set(uniq).issubset({1, 2}):
        return (y > 1).astype(int)

    # Otherwise treat as risk score: split by median
    med = float(y.median())
    return (y > med).astype(int)


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = norm_cols(df)
    df = build_exercise_level(df)

    # ---- Required columns (auto-detect) ----
    age_c = first_col(df, ["age"])
    bmi_c = first_col(df, ["bmi"])
    sys_c = first_col(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = first_col(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c  = first_col(df, ["heart_rate", "hr", "pulse"])
    y_c   = first_col(df, ["target", "risk", "label", "risk_label", "cardio"])

    missing = [name for name, col in {
        "age": age_c,
        "bmi": bmi_c,
        "systolic_bp": sys_c,
        "diastolic_bp": dia_c,
        "heart_rate": hr_c,
        "target": y_c,
    }.items() if col is None]

    if missing:
        print("❌ Missing required columns:", missing)
        print("✅ Columns found in your CSV are:")
        print(df.columns.tolist())
        print("\nFix your CSV headers OR tell me what your column names are.")
        raise SystemExit(1)

    # ---- Build model dataset ----
    use_cols = [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c]
    work = df[use_cols].copy()

    work = to_numeric(work, [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c])
    work = work.dropna(subset=[age_c, bmi_c, sys_c, dia_c, hr_c, y_c])

    X = work[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].values
    y = make_binary_target(work[y_c])

    counts = y.value_counts().to_dict()
    print("✅ Rows used:", int(len(work)))
    print("✅ Target column:", y_c)
    print("✅ Class counts:", counts)

    # Need at least 2 classes
    if y.nunique() < 2:
        print("\n❌ Your dataset ended up with ONLY ONE class after cleaning.")
        print("That means your target column doesn't contain both low/high values.")
        raise SystemExit(1)

    # If a class is tiny, stratify can fail — handle safely
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=stratify
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Probabilities + AUC (only if possible)
    auc = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        pass

    report = classification_report(
        y_test, preds,
        output_dict=True,
        zero_division=0  # <- stops warning spam
    )
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "rows_used": int(len(work)),
        "target_column": y_c,
        "features_used": [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c],
        "class_counts": counts,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved model -> {MODEL_OUT}")
    print(f"✅ Saved metrics -> {METRICS_OUT}")
    if auc is not None:
        print(f"✅ ROC_AUC -> {auc:.4f}")


if __name__ == "__main__":
    main()
