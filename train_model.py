import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


CSV_PATH = "data/health_data.csv"   # <-- keep this path
MODEL_OUT = "demo_model.pkl"
METRICS_OUT = "model_metrics.json"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = _first_existing(df, ["exercise_level", "exercise", "activity", "activity_level"])
    if not col:
        # if missing, create neutral default
        df["exercise_level"] = 0
        return df

    if col != "exercise_level":
        df["exercise_level"] = df[col]

    if df["exercise_level"].dtype == "object":
        df["exercise_level"] = df["exercise_level"].map({
            "Low": 0, "Moderate": 1, "High": 2,
            "low": 0, "moderate": 1, "high": 2
        })

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0)
    return df


def main():
    df = pd.read_csv(CSV_PATH)
    df = _normalize_cols(df)
    df = _build_exercise_level(df)

    # expected feature columns
    age_c = _first_existing(df, ["age"])
    bmi_c = _first_existing(df, ["bmi"])
    sys_c = _first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = _first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c  = _first_existing(df, ["heart_rate", "hr"])
    y_c   = _first_existing(df, ["target", "risk", "label"])

    missing = [n for n, c in [("age", age_c), ("bmi", bmi_c), ("systolic", sys_c),
                             ("diastolic", dia_c), ("heart_rate", hr_c), ("target", y_c)] if c is None]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}. Found: {list(df.columns)}")

    # numeric coerce
    for c in [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c])

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].astype(float)
    y = df[y_c].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # base pipeline
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    # calibrated model = smoother probabilities
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    report = classification_report(y_test, (probs >= 0.5).astype(int))

    joblib.dump(clf, MODEL_OUT)

    metrics = {
        "roc_auc": round(float(auc), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classification_report": report
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved:", MODEL_OUT)
    print("Metrics saved:", METRICS_OUT)
    print("ROC AUC:", auc)


if __name__ == "__main__":
    main()
