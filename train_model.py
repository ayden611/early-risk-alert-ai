import json
import pandas as pd
import joblib

from typing import Optional, List

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


CSV_PATH = "data/health_data.csv"
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


def _build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    col = _first_existing(
        df,
        ["exercise_level", "exercise", "activity", "activity_level"]
    )

    if not col:
        df["exercise_level"] = 0
        return df

    if col != "exercise_level":
        df["exercise_level"] = df[col]

    if df["exercise_level"].dtype == "object":
        df["exercise_level"] = df["exercise_level"].map({
            "Low": 0,
            "Moderate": 1,
            "High": 2,
            "low": 0,
            "moderate": 1,
            "high": 2
        })

    df["exercise_level"] = pd.to_numeric(
        df["exercise_level"], errors="coerce"
    ).fillna(0)

    return df


def main():
    df = pd.read_csv(CSV_PATH)
    df = _normalize_cols(df)
    df = _build_exercise_level(df)

    age_c = _first_existing(df, ["age"])
    bmi_c = _first_existing(df, ["bmi"])
    sys_c = _first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = _first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c = _first_existing(df, ["heart_rate", "hr"])
    y_c = _first_existing(df, ["target", "risk", "label"])

    required = [age_c, bmi_c, sys_c, dia_c, hr_c, y_c]
    if any(c is None for c in required):
        raise ValueError("Missing required columns in dataset.")

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].copy()
    y = df[y_c].copy()

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(y, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model = CalibratedClassifierCV(pipe)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    joblib.dump(model, MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model trained successfully.")
    print("ROC-AUC:", metrics["roc_auc"])


if __name__ == "__main__":
    main()


