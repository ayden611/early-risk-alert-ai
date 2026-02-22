import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# --------- CONFIG ----------
CSV_PATH = os.path.join("data", "health_data.csv")
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# We train on these "standard" features (your app form uses these)
FEATURES = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]

# Target column candidates
TARGET_CANDIDATES = ["target", "risk", "label", "high_risk", "cardio"]


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
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


def _build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has numeric exercise_level (0=Low,1=Moderate,2=High).
    Accepts strings like Low/Moderate/High OR numeric already.
    """
    df = df.copy()

    # If exercise_level missing, try to create from common alternatives
    if "exercise_level" not in df.columns:
        alt = _first_existing(df, ["exercise", "activity", "activity_level"])
        if alt is not None:
            df["exercise_level"] = df[alt]
        else:
            df["exercise_level"] = np.nan

    # If it's object/text, map it
    if df["exercise_level"].dtype == "object":
        s = df["exercise_level"].astype(str).str.lower().str.strip()
        mapping = {
            "low": 0, "l": 0, "0": 0,
            "moderate": 1, "medium": 1, "m": 1, "1": 1,
            "high": 2, "h": 2, "2": 2,
        }
        df["exercise_level"] = s.map(mapping)

    # Coerce numeric and fill missing with median (or 1 if all missing)
    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce")
    if df["exercise_level"].notna().any():
        df["exercise_level"] = df["exercise_level"].fillna(df["exercise_level"].median())
    else:
        df["exercise_level"] = df["exercise_level"].fillna(1)

    return df


def _pick_target(df: pd.DataFrame) -> str:
    t = _first_existing(df, TARGET_CANDIDATES)
    if t is None:
        raise ValueError(
            f"Could not find a target column. Tried: {TARGET_CANDIDATES}. "
            f"Available columns: {list(df.columns)}"
        )
    return t


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = _normalize_cols(df)

    # Build / normalize exercise_level
    df = _build_exercise_level(df)

    # Pick target
    target_col = _pick_target(df)

    # Ensure numeric on core columns
    df = _to_numeric(df, FEATURES + [target_col])

    # Drop rows missing required cols
    keep_cols = FEATURES + [target_col]
    df = df[keep_cols].dropna()

    # Ensure target is 0/1 int
    # If target contains {0,1} or {1,2} or floats -> coerce and binarize if needed.
    y = df[target_col].astype(float)

    # If y looks like >1 labels (e.g., 1/2), convert to 0/1 by thresholding at median
    unique_vals = sorted(pd.unique(y))
    if set(unique_vals).issubset({0.0, 1.0}):
        y_bin = y.astype(int)
    else:
        # common case: cardio dataset uses 0/1 already, but this keeps it robust
        y_bin = (y >= np.median(y)).astype(int)

    df[target_col] = y_bin.astype(int)
    return df, target_col


@dataclass
class TrainMetrics:
    csv_path: str
    model_out: str
    metrics_out: str
    target_name: str
    target_counts: dict
    features_used: list
    roc_auc: float | None
    confusion_matrix: list
    classification_report: dict


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    df_raw = pd.read_csv(CSV_PATH)
    df, target_col = _clean_dataframe(df_raw)

    X = df[FEATURES]
    y = df[target_col].astype(int)

    # Stratify only if both classes exist
    stratify = y if y.nunique() == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    # ---- Calibration (SAFE + version compatible) ----
    # CalibratedClassifierCV in your sklearn expects `estimator=`, not `base_estimator=`.
    # Also: calibration CV needs enough samples per class.
    min_class = int(y_train.value_counts().min()) if y_train.nunique() == 2 else 0

    if y_train.nunique() == 2 and min_class >= 3:
        model = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=3)
    elif y_train.nunique() == 2 and min_class >= 2:
        model = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=2)
    else:
        # If only one class or too few samples, skip calibration
        model = base_pipeline

    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)

    roc = None
    if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            roc = float(roc_auc_score(y_test, proba))
        except Exception:
            roc = None

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ---- Save model ----
    joblib.dump(model, MODEL_OUT)

    metrics = TrainMetrics(
        csv_path=CSV_PATH,
        model_out=MODEL_OUT,
        metrics_out=METRICS_OUT,
        target_name=target_col,
        target_counts=y.value_counts().to_dict(),
        features_used=FEATURES,
        roc_auc=roc,
        confusion_matrix=cm,
        classification_report=report,
    )

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    print("✅ TRAIN OK")
    print("✅ Saved:", MODEL_OUT)
    print("✅ Metrics:", METRICS_OUT)
    print("✅ Target counts:", y.value_counts().to_dict())
    if roc is not None:
        print("✅ ROC AUC:", round(roc, 4))


if __name__ == "__main__":
    main()
