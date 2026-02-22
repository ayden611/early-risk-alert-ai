import json
import os
from typing import Optional

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


CSV_PATH = "data/health_data.csv"     # keep this path
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has numeric exercise_level (0=Low,1=Moderate,2=High).
    Accepts: exercise_level / exercise / activity / activity_level, and string values too.
    """
    df = df.copy()
    src = _first_existing(df, ["exercise_level", "exercise", "activity", "activity_level"])

    if src is None:
        df["exercise_level"] = 0
        return df

    if src != "exercise_level":
        df["exercise_level"] = df[src]

    # Convert strings like "Low", "Moderate", "High"
    if df["exercise_level"].dtype == "object":
        df["exercise_level"] = (
            df["exercise_level"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"low": 0, "moderate": 1, "medium": 1, "high": 2})
        )

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0)
    df["exercise_level"] = df["exercise_level"].clip(0, 2)
    return df


def _make_binary_target(y_raw: pd.Series) -> pd.Series:
    """
    Convert common target formats to 0/1.
    Accepts:
      - numeric 0/1
      - strings like "low risk"/"high risk"
      - strings like "low"/"high"
    """
    y = y_raw.copy()

    if y.dtype == "object":
        s = y.astype(str).str.strip().str.lower()
        # Map a bunch of likely labels
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
        y = s.map({"low risk": 0, "high risk": 1})

    y = pd.to_numeric(y, errors="coerce")
    y = y.fillna(0).astype(int)
    # Force into 0/1 if weird values appear
    y = (y > 0).astype(int)
    return y


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = _normalize_cols(df)
    df = _build_exercise_level(df)

    # Find columns (supports several naming styles)
    age_c = _first_existing(df, ["age"])
    bmi_c = _first_existing(df, ["bmi"])
    sys_c = _first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = _first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c  = _first_existing(df, ["heart_rate", "hr"])
    y_c   = _first_existing(df, ["target", "risk", "label", "risk_label"])

    missing = [name for name, col in {
        "age": age_c, "bmi": bmi_c, "systolic_bp": sys_c, "diastolic_bp": dia_c, "heart_rate": hr_c, "target": y_c
    }.items() if col is None]

    if missing:
        raise ValueError(
            "Missing required columns in CSV: "
            + ", ".join(missing)
            + "\nTip: open data/health_data.csv and check the header names."
        )

    use_cols = [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c]
    df = df[use_cols].copy()

    df = _to_numeric(df, [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c])
    df = df.dropna(subset=[age_c, bmi_c, sys_c, dia_c, hr_c, y_c])

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].values
    y = _make_binary_target(df[y_c])

    # Must have at least 2 classes
    counts = y.value_counts()
    if counts.shape[0] < 2:
        raise ValueError(
            f"Your dataset only has ONE class after cleaning (counts={counts.to_dict()}). "
            "You need both low + high risk rows to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # --- KEY FIX: choose a safe CV for calibration ---
    min_class_train = int(y_train.value_counts().min())
    cv_splits = min(5, min_class_train)

    if cv_splits >= 2:
        # Use calibration only if there are enough samples per class
        try:
            model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv_splits)
        except TypeError:
            # For older scikit-learn versions
            model = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=cv_splits)
    else:
        # Not enough data for calibration; train base model
        model = base

    model.fit(X_train, y_train)

    # Evaluate
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None

    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    metrics = {
        "rows_used": int(len(df)),
        "class_counts_after_cleaning": counts.to_dict(),
        "min_class_train": min_class_train,
        "calibration_cv_used": int(cv_splits) if cv_splits >= 2 else None,
        "roc_auc": auc,
        "classification_report": report,
    }

    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved model -> {MODEL_OUT}")
    print(f"✅ Saved metrics -> {METRICS_OUT}")
    print(f"✅ calibration_cv_used = {metrics['calibration_cv_used']}")


if __name__ == "__main__":
    main()
