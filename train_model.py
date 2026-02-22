import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

CSV_PATH = "data/health_data.csv"
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


def main():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print("Columns in dataset:", df.columns.tolist())

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_cols = [
        "age",
        "bmi",
        "exercise_level",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "risk"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols].dropna()

    # Convert target to 0 / 1
    df["risk"] = df["risk"].astype(str).str.lower().map({
        "low": 0,
        "low risk": 0,
        "0": 0,
        "high": 1,
        "high risk": 1,
        "1": 1
    })

    df = df.dropna()

    X = df[["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]].values
    y = df["risk"].astype(int)

    print("Class counts:", y.value_counts().to_dict())

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both low and high risk samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
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
    probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds, output_dict=True)
    auc = float(roc_auc_score(y_test, probs))

    metrics = {
        "class_counts": y.value_counts().to_dict(),
        "roc_auc": auc,
        "classification_report": report
    }

    joblib.dump(model, MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved →", MODEL_OUT)
    print("Metrics saved →", METRICS_OUT)
    print("ROC AUC:", auc)


if __name__ == "__main__":
    main()
