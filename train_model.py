import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# ====== CONFIG ======
CSV_PATH = os.path.join("data", "health_data.csv")   # change if your file name differs
MODEL_OUT = "demo_model.pkl"
RANDOM_STATE = 42
# ====================


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has numeric exercise_level:
      Low=0, Moderate/Medium=1, High=2
    Accepts: exercise_level / exercise / activity / activity_level
    """
    df = df.copy()

    src = first_existing(df, ["exercise_level", "exercise", "activity", "activity_level"])
    if src is None:
        # If missing, create a default (0)
        df["exercise_level"] = 0
        return df

    s = df[src]
    # If numeric already
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() > 0.8:
        df["exercise_level"] = sn.fillna(sn.median())
        # clamp 0-2 if it looks like that scale
        df["exercise_level"] = df["exercise_level"].clip(0, 2)
        return df

    # Otherwise map strings
    s2 = s.astype(str).str.strip().str.lower()
    mapping = {
        "low": 0, "l": 0, "0": 0,
        "moderate": 1, "medium": 1, "mod": 1, "m": 1, "1": 1,
        "high": 2, "h": 2, "2": 2,
    }
    df["exercise_level"] = s2.map(mapping)
    df["exercise_level"] = pd.to_numeric(df["exercise_level"], errors="coerce").fillna(0).astype(int)
    df["exercise_level"] = df["exercise_level"].clip(0, 2)
    return df


def make_binary_target(series: pd.Series) -> pd.Series:
    """
    Converts common target label formats into 0/1.
    Accepts:
      - 0/1 ints
      - "low risk" / "high risk"
      - "low"/"high"
      - true/false
      - yes/no
      - cardio dataset style: cardio 0/1
    """
    s = series.copy()

    # Try numeric first
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() > 0.8:
        y = sn.fillna(0).astype(int)
        # force into 0/1
        y = (y > 0).astype(int)
        return y

    # Otherwise map strings
    st = s.astype(str).str.strip().str.lower()
    mapping = {
        "low risk": 0, "low": 0, "0": 0, "false": 0, "no": 0, "n": 0,
        "high risk": 1, "high": 1, "1": 1, "true": 1, "yes": 1, "y": 1,
    }
    y = st.map(mapping)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    y = (y > 0).astype(int)
    return y


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = normalize_cols(df)
    df = build_exercise_level(df)

    # Find feature columns (supports several naming styles)
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
            "\nTip: open data/health_data.csv and check your header names."
        )

    # Convert numeric fields
    df = to_numeric(df, [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c])

    # Drop rows with missing required values
    df = df.dropna(subset=[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c, y_c]).copy()

    X = df[[age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c]].values
    y = make_binary_target(df[y_c])

    counts = y.value_counts().to_dict()
    print("Target counts:", counts)

    # Need at least 2 classes to train properly
    if len(counts) < 2:
        raise ValueError(
            f"Your dataset only has ONE class after cleaning: {counts}. "
            "You need both low + high risk rows to train."
        )

    # Split (use stratify only if both classes have enough samples)
    stratify = y if min(counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print("ROC AUC:", round(float(auc), 4))
    except Exception as e:
        print("ROC AUC not available:", e)

    # Save model + feature order (important for your Flask app)
    payload = {
        "model": model,
        "feature_order": [age_c, bmi_c, "exercise_level", sys_c, dia_c, hr_c],
    }
    joblib.dump(payload, MODEL_OUT)

    size = os.path.getsize(MODEL_OUT)
    print("\n✅ TRAIN OK")
    print("✅ Saved:", MODEL_OUT, f"({size} bytes)")
    print("✅ Feature order:", payload["feature_order"])


if __name__ == "__main__":
    main()
