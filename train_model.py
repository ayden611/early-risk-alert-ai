import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --------- CONFIG ----------
CSV_PATH = os.path.join("data", "health_data.csv")
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
TARGET_CANDIDATES = ["target", "risk", "label", "high_risk", "cardio"]


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric exercise_level (0=Low,1=Moderate,2=High).
    Accepts strings or numeric.
    """
    df = df.copy()

    if "exercise_level" not in df.columns:
        alt = first_existing(df, ["exercise", "activity", "activity_level"])
        df["exercise_level"] = df[alt] if alt else np.nan

    if df["exercise_level"].dtype == "object":
        s = df["exercise_level"].astype(str).str.lower().str.strip()
        mapping = {
            "low": 0, "l": 0, "0": 0,
            "moderate": 1, "medium": 1, "m": 1, "1": 1,
            "high": 2, "h": 2, "2": 2,
        }
        df["exercise_level"] = s.map(mapping)

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce")

    if df["exercise_level"].notna().any():
        df["exercise_level"] = df["exercise_level"].fillna(df["exercise_level"].median())
    else:
        df["exercise_level"] = df["exercise_level"].fillna(1)

    return df


def pick_target(df: pd.DataFrame) -> str:
    t = first_existing(df, TARGET_CANDIDATES)
    if not t:
        raise ValueError(
            "Could not find target column. Tried: {}. Available: {}".format(
                TARGET_CANDIDATES, list(df.columns)
            )
        )
    return t


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = normalize_cols(df)
    df = build_exercise_level(df)

    target_col = pick_target(df)

    df = to_numeric(df, FEATURES + [target_col])
    df = df[FEATURES + [target_col]].dropna()

    y = df[target_col].astype(float)
    unique_vals = sorted(pd.unique(y))

    if set(unique_vals).issubset({0.0, 1.0}):
        y_bin = y.astype(int)
    else:
        y_bin = (y >= np.median(y)).astype(int)

    df[target_col] = y_bin.astype(int)
    return df, target_col


@dataclass
class TrainMetrics:
    csv_path: str
    model_out: str
    metrics_out: str
    target_name: str
    target_counts: Dict[str, int]
    features_used: List[str]
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


def make_calibrated_model(base_pipeline, y_train: pd.Series):
    """
    Works with both sklearn APIs:
    - Newer: CalibratedClassifierCV(estimator=..., ...)
    - Older: CalibratedClassifierCV(base_estimator=..., ...)
    Also avoids calibration when class counts are too small.
    """
    if y_train.nunique() != 2:
        return base_pipeline

    min_class = int(y_train.value_counts().min())
    if min_class < 2:
        return base_pipeline

    cv = 3 if min_class >= 3 else 2

    # Try newer param name first, then fallback.
    try:
        return CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_pipeline, method="sigmoid", cv=cv)


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("CSV not found at: {}".format(CSV_PATH))

    df_raw = pd.read_csv(CSV_PATH)
    df, target_col = clean_dataframe(df_raw)

    X = df[FEATURES]
    y = df[target_col].astype(int)

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

    model = make_calibrated_model(base_pipeline, y_train)
    model.fit(X_train, y_train)

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

    joblib.dump(model, MODEL_OUT)

    metrics = TrainMetrics(
        csv_path=CSV_PATH,
        model_out=MODEL_OUT,
        metrics_out=METRICS_OUT,
        target_name=target_col,
        target_counts={str(k): int(v) for k, v in y.value_counts().to_dict().items()},
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
