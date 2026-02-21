import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


CSV_PATH = "data/health_data.csv"
MODEL_OUT = "demo_model.pkl"


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
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_exercise_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has numeric exercise_level (0=Low,1=Moderate,2=High).
    If a matching column exists, we convert it.
    If not, we try to derive it from minutes if available.
    Otherwise we default to Moderate (1).
    """
    df = df.copy()

    # Direct exercise/activity column candidates
    ex_col = _first_existing(df, [
        "exercise_level", "exercise", "activity_level", "activity", "exerciselevel"
    ])

    if ex_col is not None:
        # If it's text (low/moderate/high), map it
        if df[ex_col].dtype == "object":
            mapping = {
                "low": 0, "moderate": 1, "medium": 1, "high": 2,
                "light": 0, "none": 0, "sedentary": 0,
                "active": 2, "very_active": 2, "very active": 2
            }
            df["exercise_level"] = (
                df[ex_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
            )
        else:
            # If numeric already, keep it but clamp to 0-2
            df["exercise_level"] = pd.to_numeric(df[ex_col], errors="coerce")

        # Clamp into 0..2
        df["exercise_level"] = df["exercise_level"].clip(lower=0, upper=2)
        return df

    # Try to derive from minutes/week
    mins_col = _first_existing(df, [
        "exercise_minutes", "exercise_mins", "minutes_exercised", "weekly_exercise_minutes",
        "activity_minutes", "weekly_activity_minutes"
    ])

    if mins_col is not None:
        df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")
        # Simple bins: <60 low, 60-149 moderate, >=150 high
        df["exercise_level"] = pd.cut(
            df[mins_col],
            bins=[-1, 59, 149, 10**9],
            labels=[0, 1, 2]
        ).astype(float)
        return df

    # Fallback: default Moderate
    df["exercise_level"] = 1.0
    print("WARN: No exercise column found. Defaulting exercise_level to 1 (Moderate).")
    return df


def main():
    df = pd.read_csv(CSV_PATH)
    df = _normalize_cols(df)

    print("\nLoaded columns:")
    print(list(df.columns))

    # Find core feature columns (with synonym support)
    col_age = _first_existing(df, ["age", "years"])
    col_bmi = _first_existing(df, ["bmi", "body_mass_index"])
    col_sys = _first_existing(df, ["systolic_bp", "systolic", "sbp", "sys_bp"])
    col_dia = _first_existing(df, ["diastolic_bp", "diastolic", "dbp", "dia_bp"])
    col_hr  = _first_existing(df, ["heart_rate", "hr", "pulse", "resting_heart_rate"])

    # Target column candidates
    col_y = _first_existing(df, ["target", "label", "risk", "outcome", "cardio", "high_risk"])

    missing = [name for name, col in {
        "age": col_age,
        "bmi": col_bmi,
        "systolic_bp": col_sys,
        "diastolic_bp": col_dia,
        "heart_rate": col_hr,
        "target": col_y
    }.items() if col is None]

    if missing:
        print("\nERROR: Missing required columns:", missing)
        print("Fix by renaming columns in the CSV or tell me the column list above and I’ll map them.")
        return

    # Build exercise_level safely (creates df["exercise_level"])
    df = _build_exercise_level(df)

    # Rename feature columns into expected names
    df = df.rename(columns={
        col_age: "age",
        col_bmi: "bmi",
        col_sys: "systolic_bp",
        col_dia: "diastolic_bp",
        col_hr: "heart_rate",
        col_y: "target"
    })

    # Make sure target is 0/1
    # If target is text, try to map common values
    if df["target"].dtype == "object":
        ymap = {
            "0": 0, "1": 1,
            "low": 0, "low_risk": 0, "no": 0, "false": 0,
            "high": 1, "high_risk": 1, "yes": 1, "true": 1
        }
        df["target"] = df["target"].astype(str).str.strip().str.lower().map(ymap)

    df = _to_numeric(df, ["age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "exercise_level", "target"])
    df = df.dropna(subset=["age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "exercise_level", "target"])

    # Force int labels
    df["target"] = df["target"].astype(int)

    X = df[["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]]
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # Pipeline + calibration (better probability outputs)
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        print("ROC AUC skipped (not available).")

    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
