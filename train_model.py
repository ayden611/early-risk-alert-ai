import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ========= CONFIG =========
CSV_PATH = os.path.join("data", "health_data.csv")
MODEL_OUT = "demo_model.pkl"

# Silence that UndefinedMetricWarning noise
warnings.filterwarnings("ignore")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric exercise_level exists:
    Low=0, Moderate/Medium=1, High=2
    """
    df = df.copy()

    # If a variant column exists, rename it to exercise_level
    if "exercise_level" not in df.columns:
        alt = first_existing(df, ["exercise", "activity", "activity_level"])
        if alt:
            df = df.rename(columns={alt: "exercise_level"})

    if "exercise_level" not in df.columns:
        # default if missing
        df["exercise_level"] = 1

    # Map strings -> numbers
    if df["exercise_level"].dtype == object:
        s = df["exercise_level"].astype(str).str.strip().str.lower()
        mapping = {
            "low": 0,
            "l": 0,
            "moderate": 1,
            "medium": 1,
            "m": 1,
            "high": 2,
            "h": 2,
        }
        df["exercise_level"] = s.map(mapping)

    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(1).astype(int)
    df["exercise_level"] = df["exercise_level"].clip(0, 2)
    return df


def make_binary_target(series: pd.Series) -> pd.Series:
    """
    Accepts targets like:
    - 0/1
    - Low/High
    - True/False
    - risk/target/label columns
    Returns 0/1 int series.
    """
    s = series.copy()

    if s.dtype == object:
        t = s.astype(str).str.strip().str.lower()
        mapping = {
            "low": 0,
            "low_risk": 0,
            "low risk": 0,
            "0": 0,
            "false": 0,
            "no": 0,
            "negative": 0,

            "high": 1,
            "high_risk": 1,
            "high risk": 1,
            "1": 1,
            "true": 1,
            "yes": 1,
            "positive": 1,
        }
        s = t.map(mapping)

    s = pd.to_numeric(s, errors="coerce")
    # force to 0/1
    s = (s > 0).astype(int)
    return s


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = normalize_cols(df)
    df = build_exercise_level(df)

    # Find required columns (supports different naming styles)
    age_c = first_existing(df, ["age"])
    bmi_c = first_existing(df, ["bmi"])
    sys_c = first_existing(df, ["systolic_bp", "sys_bp", "systolic"])
    dia_c = first_existing(df, ["diastolic_bp", "dia_bp", "diastolic"])
    hr_c  = first_existing(df, ["heart_rate", "hr"])
    y_c   = first_existing(df, ["target", "risk", "label", "risk_label", "cardio"])

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
            "Missing required columns in CSV: " + ", ".join(missing) +
            "\nTip: open data/health_data.csv and check header names."
        )

    use_cols = [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c]
    df = df[use_cols].copy()

    # Numeric conversion
    for c in [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c])

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].values
    y = make_binary_target(df[y_c])

    counts = y.value_counts().to_dict()
    if len(counts) < 2:
        raise ValueError(
            f"Your dataset only has ONE class after cleaning (counts={counts}). "
            "You need both low + high risk rows to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
    ])

    model.fit(X_train, y_train)

    # Evaluate (safe metrics)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("\n✅ TRAIN OK")
    print("✅ ROC AUC:", round(float(auc), 4))
    print("✅ Target counts:", counts)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_OUT)
    print("\n✅ Saved:", MODEL_OUT)


if __name__ == "__main__":
    main()
    main()
    main()
    main()
    main()
    main()
