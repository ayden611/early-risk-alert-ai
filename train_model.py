import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

# --------------------------------
# LOAD DATA
# --------------------------------

# Update this path if needed
df = pd.read_csv("data/cardiovascular_disease.csv")

# Example expected columns:
# age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, target

# If exercise_level is text, convert to numeric
if df["exercise_level"].dtype == "object":
    df["exercise_level"] = df["exercise_level"].map({
        "Low": 0,
        "Moderate": 1,
        "High": 2
    })

X = df[[
    "age",
    "bmi",
    "exercise_level",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate"
]]

y = df["target"]  # 0 = low risk, 1 = high risk

# --------------------------------
# TRAIN / TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------
# BASE MODEL
# --------------------------------

base_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# --------------------------------
# CALIBRATION
# --------------------------------

model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=5
)

model.fit(X_train, y_train)

# --------------------------------
# EVALUATION
# --------------------------------

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------
# SAVE MODEL
# --------------------------------

joblib.dump(model, "demo_model.pkl")

print("\nModel saved as demo_model.pkl")
