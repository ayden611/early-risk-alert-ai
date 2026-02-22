# train_model.py
import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Kill sklearn warning spam
warnings.filterwarnings("ignore")

CSV_PATH = "data/health_data.csv"
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


def normalize_columns(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def find_column(df, options):
    for o in options:
        if o in df.columns:
            return o
    return None


def build_exercise(df):
    src = find_column(df, ["exercise_level", "exercise", "activity", "activity_level"])
    if src is None:
        df["exercise_level"] = 0
        return df

    if src != "exercise_level":
        df["exercise_level"] = df[src]

    if df["exercise_level"].dtype == "object":
        s = df["exercise_level"].astype(str).str.lower()
        df["exercise_level"] = s.map({
            "low": 0,
            "moderate": 1,
            "medium": 1,
            "high": 2
        })

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0)
    return df


def build_target(series):
    y = series.copy()

    if y.dtype == "object":
        s = y.astype(str).str.lower().str.strip()
        s = s.replace({
            "low risk": 0,
            "high risk": 1,
            "low": 0,
            "high": 1,
            "false": 0,
            "true": 1,
            "0": 0,
            "1": 1
        })
        y = s

    y = pd.to_numeric(y, errors="coerce").fillna(0)
    y = (y > 0).astype(int)
    return y


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("health_data.csv not found.")

    df = pd.read_csv(CSV_PATH)
    df = normalize_columns(df)
    df = build_exercise(df)

    age = find_column(df, ["age"])
    bmi = find_column(df, ["bmi"])
    sys = find_column(df, ["systolic_bp", "sys_bp", "systolic"])
    dia = find_column(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr = find_column(df, ["heart_rate", "hr", "pulse"])
    target = find_column(df, ["target", "risk", "label", "risk_label", "cardio"])

    if None in [age, bmi, sys, dia, hr, target]:
        print("Missing required columns.")
        print("Columns found:", df.columns.tolist())
        return

    df = df[[age, bmi, "exercise_level", sys, dia, hr, target]].dropna()

    X = df[[age, bmi, "exercise_level", sys, dia, hr]].astype(float).values
    y = build_target(df[target])

    if y.nunique() < 2:
        print("Dataset only has one class after cleaning.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    auc = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except:
        pass

    report = classification_report(y_test, preds, zero_division=0, output_dict=True)

    metrics = {
        "rows_used": len(df),
        "class_counts": y.value_counts().to_dict(),
        "roc_auc": auc,
        "report": report
    }

    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print("Model saved as demo_model.pkl")
    print("Metrics saved as model_metrics.json")


if __name__ == "__main__":
    main()
