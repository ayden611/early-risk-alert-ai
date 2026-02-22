#!/usr/bin/env python3
"""
Early Risk Alert AI - Training Script (CLEAN)

- Reads data/health_data.csv
- Normalizes column names
- Builds/infers a binary target if needed
- Trains a calibrated Logistic Regression pipeline
- Saves model to demo_model.pkl
- Prints a clean report without UndefinedMetricWarning spam
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CSV_PATH = os.environ.get("CSV_PATH", "data/health_data.csv")
MODEL_OUT = os.environ.get("MODEL_OUT", "demo_model.pkl")
METRICS_OUT = os.environ.get("METRICS_OUT", "metrics.json")

RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))


# ----------------------------
# Helpers
# ----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def map_exercise_level(series: pd.Series) -> pd.Series:
    """
    Map exercise strings -> numeric:
      low -> 0, moderate/medium -> 1, high -> 2
    If already numeric, keep it.
    """
    s = series.copy()

    # If it's already numeric, just coerce.
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # Otherwise, map common labels.
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "low": 0,
        "l": 0,
        "0": 0,
        "moderate": 1,
        "medium": 1,
        "med": 1,
        "m": 1,
        "1": 1,
        "high": 2,
        "h": 2,
        "2": 2,
    }
    return s.map(mapping)


def infer_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Find target column if present. Otherwise infer a reasonable binary target.
    Preferred target names: risk, target, label, outcome, high_risk, cardiovascular_risk, cvd, disease

    If not found, infer:
      high_risk = 1 if (systolic>=140 OR diastolic>=90 OR bmi>=30 OR heart_rate>=100) else 0
    """
    df = df.copy()

    target_col = first_existing(
        df,
        [
            "target",
            "label",
            "risk",
            "outcome",
            "high_risk",
            "cardiovascular_risk",
            "cvd",
            "disease",
            "has_disease",
        ],
    )

    if target_col:
        y_raw = df[target_col]
        # Make it binary 0/1
        if pd.api.types.is_numeric_dtype(y_raw):
            y = pd.to_numeric(y_raw, errors="coerce")
            # If values are like 0/1 already, keep; otherwise treat >0 as 1
            y = (y > 0).astype(int)
        else:
            s = y_raw.astype(str).str.strip().str.lower()
            y = s.isin(["1", "true", "yes", "y", "high", "high_risk", "risk", "positive"]).astype(int)
        return df, y, target_col

    # Infer from features (requires these columns or we fail gracefully later)
    # We'll build with what we can find.
    sys_col = first_existing(df, ["systolic_bp", "sys_bp", "systolic", "sbp"])
    dia_col = first_existing(df, ["diastolic_bp", "dia_bp", "diastolic", "dbp"])
    bmi_col = first_existing(df, ["bmi", "body_mass_index"])
    hr_col = first_existing(df, ["heart_rate", "hr", "pulse"])

    # Create numeric temp columns if they exist
    for c in [sys_col, dia_col, bmi_col, hr_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    conditions = []
    if sys_col:
        conditions.append(df[sys_col] >= 140)
    if dia_col:
        conditions.append(df[dia_col] >= 90)
    if bmi_col:
        conditions.append(df[bmi_col] >= 30)
    if hr_col:
        conditions.append(df[hr_col] >= 100)

    if not conditions:
        raise ValueError(
            "No target column found AND could not infer target (missing BP/BMI/HR columns). "
            "Add a target column (risk/target/label) to your CSV."
        )

    y = np.zeros(len(df), dtype=int)
    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined | cond
    y = combined.astype(int)

    inferred_name = "high_risk_inferred"
    return df, pd.Series(y, name=inferred_name), inferred_name


@dataclass
class FeatureCols:
    age: str
    bmi: str
    exercise_level: str
    systolic_bp: str
    diastolic_bp: str
    heart_rate: str


def resolve_feature_cols(df: pd.DataFrame) -> FeatureCols:
    age = first_existing(df, ["age", "years"])
    bmi = first_existing(df, ["bmi", "body_mass_index"])
    ex = first_existing(df, ["exercise_level", "exercise", "activity", "activity_level"])
    sys_bp = first_existing(df, ["systolic_bp", "sys_bp", "systolic", "sbp"])
    dia_bp = first_existing(df, ["diastolic_bp", "dia_bp", "diastolic", "dbp"])
    hr = first_existing(df, ["heart_rate", "hr", "pulse"])

    missing = [name for name, col in [
        ("age", age),
        ("bmi", bmi),
        ("exercise_level", ex),
        ("systolic_bp", sys_bp),
        ("diastolic_bp", dia_bp),
        ("heart_rate", hr),
    ] if col is None]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}. "
            "Your CSV must include age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate "
            "(names can vary, but must be recognizable)."
        )

    return FeatureCols(
        age=age, bmi=bmi, exercise_level=ex, systolic_bp=sys_bp, diastolic_bp=dia_bp, heart_rate=hr
    )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = normalize_cols(df)

    # Target
    df, y, target_name = infer_target(df)

    # Features
    cols = resolve_feature_cols(df)

    # Build feature matrix
    work = df[[cols.age, cols.bmi, cols.exercise_level, cols.systolic_bp, cols.diastolic_bp, cols.heart_rate]].copy()

    # Exercise mapping
    work[cols.exercise_level] = map_exercise_level(work[cols.exercise_level])

    # Numeric coercion
    work = coerce_numeric(work, [cols.age, cols.bmi, cols.exercise_level, cols.systolic_bp, cols.diastolic_bp, cols.heart_rate])
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with missing values
    full = pd.concat([work, y.rename("target")], axis=1)
    full = full.dropna()
    X = full.drop(columns=["target"])
    y = full["target"].astype(int)

    # Safety: need both classes
    vc = y.value_counts().to_dict()
    if len(vc) < 2:
        raise ValueError(
            f"Your target only has one class after cleaning: {vc}. "
            "You need both 0 and 1 examples to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Model: balanced LR + calibration for better probabilities
    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("cal", CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)),
        ]
    )

    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    # Metrics (no warning spam)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC AUC can fail if a fold collapses; handle safely
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None

    # Save model
    joblib.dump(clf, MODEL_OUT)

    # Save metrics json
    metrics = {
        "csv_path": CSV_PATH,
        "model_out": MODEL_OUT,
        "target_name": target_name,
        "target_counts": vc,
        "features_used": list(X.columns),
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_rows_after_cleaning": int(len(full)),
        "random_state": RANDOM_STATE,
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ TRAIN OK")
    print("✅ Saved:", MODEL_OUT)
    print("✅ Metrics:", METRICS_OUT)
    print("✅ Target counts:", vc)
    if auc is not None:
        print("✅ ROC AUC:", round(auc, 4))
    print("\n--- Classification report ---")
    print(report)
    print("\n--- Confusion matrix [ [tn, fp], [fn, tp] ] ---")
    print(cm)


if __name__ == "__main__":
    main()
