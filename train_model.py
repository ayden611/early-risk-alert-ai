# train_model.py
# Drop-in replacement: copy/paste this whole file, then run:
#   python3 train_model.py
#
# Expects your CSV at: data/health_data.csv
# Outputs:
#   demo_model.pkl
#   model_metrics.json

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CSV_PATH = "data/health_data.csv"
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


# ---- helpers ----
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates numeric exercise_level (0=Low, 1=Moderate, 2=High).
    Accepts columns: exercise_level / exercise / activity / activity_level
    Accepts string values like Low/Moderate/High (case-insensitive) or numeric.
    """
    df = df.copy()
    src = _first_existing(df, ["exercise_level", "exercise", "activity", "activity_level"])

    if src is None:
        df["exercise_level"] = 0
        return df

    if src != "exercise_level":
        df["exercise_level"] = df[src]
    # Convert strings -> numbers if needed
    if df["exercise_level"].dtype == "object":
        s = df["exercise_level"].astype(str).str.strip().str.lower()
        df["exercise_level"] = s.map(
            {
                "low": 0,
                "moderate": 1,
                "medium": 1,
                "high": 2,
                "0": 0,
                "1": 1,
                "2": 2,
            }
        )
    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0)
    df["exercise_level"] = df["exercise_level"].clip(0, 2)
    return df


def _make_binary_target(y_raw: pd.Series) -> pd.Series:
    """
    Converts common target formats -> 0/1.
    Accepts:
      - numeric 0/1
      - strings: "low risk"/"high risk", "low"/"high", "false"/"true", etc.
      - anything else: numeric coercion then (y > 0) -> 1
    """
    y = y_raw.copy()

    if y.dtype == "object":
        s = y.astype(str).str.strip().str.lower()
        s = s.replace(
            {
                "high_risk": "high risk",
                "low_risk": "low risk",
                "1": "high risk",
                "0": "low risk",
                "true": "high risk",
                "false": "low risk",
                "yes": "high risk",
                "no": "low risk",
                "high": "high risk",
                "low": "low risk",
                "risk": "high risk",
                "no_risk": "low risk",
            }
        )
        mapped = s.map({"low risk": 0, "high risk": 1})
        # if still unmapped, try numeric
        y = mapped

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    # Force to 0/1 even if weird values show up
    y = (y > 0).astype(int)
    return y


def _safe_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Stratify only if both classes have >= 2 samples (otherwise stratify will error).
    """
    y_series = pd.Series(y)
    vc = y_series.value_counts()
    can_stratify = (len(vc) == 2) and (vc.min() >= 2)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = _normalize_cols(df)
    df = _ensure_exercise_level(df)

    # Find columns (supports multiple naming styles)
    age_c = _first_existing(df, ["age"])
    bmi_c = _first_existing(df, ["bmi"])
    sys_c = _first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = _first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c = _first_existing(df, ["heart_rate", "hr", "pulse"])

    # Target candidates (add more if you want)
    y_c = _first_existing(df, ["target", "risk", "label", "risk_label", "cardio"])

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

    # Numeric conversion + drop bad rows
    df = _to_numeric(df, [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c])
    df = df.dropna(subset=[age_c, bmi_c, sys_c, dia_c, hr_c, y_c])

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].to_numpy(dtype=float)
    y = _make_binary_target(df[y_c])

    counts = y.value_counts().to_dict()
    print("Target column:", y_c)
    print("Class counts:", counts)

    if len(counts) < 2:
        raise ValueError(
            f"Your dataset only has ONE class after cleaning (counts={counts}). "
            "You need both low + high risk rows to train."
        )

    # Train/test split (stratify only when safe)
    X_train, X_test, y_train, y_test = _safe_split(X, y.to_numpy())

    # Simple, stable model (balanced to help imbalanced datasets)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
            )),
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # AUC only if we can compute probabilities
    auc = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    metrics = {
        "rows_used": int(len(df)),
        "target_col": y_c,
        "class_counts_after_cleaning": counts,
        "roc_auc": auc,
        "classification_report": report,
        "feature_order": ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"],
        "resolved_feature_columns": {
            "age": age_c,
            "bmi": bmi_c,
            "systolic_bp": sys_c,
            "diastolic_bp": dia_c,
            "heart_rate": hr_c,
        },
    }

    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved model -> {MODEL_OUT}")
    print(f"✅ Saved metrics -> {METRICS_OUT}")
    if auc is not None:
        print(f"✅ ROC AUC -> {auc:.4f}")


if __name__ == "__main__":
    main()
