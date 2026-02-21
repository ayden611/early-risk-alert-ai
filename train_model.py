import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


# ============================================================
# LOAD DATA
# ============================================================

# Update path if needed
df = pd.read_csv("data/health_data.csv")

# Expected columns:
# age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, target

# Convert exercise_level text to numeric if needed
if df["exercise_level"].dtype == "object":
    df["exercise_level"] = df["exercise_level"].map({
        "Low": 0,
        "Moderate": 1,
        "High": 2,
        "low": 0,
        "moderate": 1,
        "high": 2
    })

# Features
X = df[[
    "age",
    "bmi",
    "exercise_level",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate"
]]

# Target (0 = low risk, 1 = high risk)
y = df["target"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# BASE MODEL (WITH SCALING)
# ============================================================

base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])


# ============================================================
# CALIBRATION
# ============================================================

model = CalibratedClassifierCV(
    base_pipeline,
    method="sigmoid",
    cv=5
)

model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "demo_model.pkl")

print("\nModel retrained, calibrated, and saved as demo_model.pkl")
