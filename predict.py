import os
import numpy as np
import joblib

# Load model safely (works local + Render)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
_model = joblib.load(MODEL_PATH)

# Map exercise strings to numeric if your model expects numbers
_EX_MAP = {"low": 0, "moderate": 1, "high": 2}

def predict_risk(*, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    """
    Returns: (risk_label, probability)
      - risk_label: "High Risk" or "Low Risk"
      - probability: float 0..1 (probability of High Risk)
    """
    ex = str(exercise_level).strip().lower()
    ex_num = _EX_MAP.get(ex, 0)  # default Low if unknown

    X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

    pred = int(_model.predict(X)[0])
    prob_high = float(_model.predict_proba(X)[0][1])

    risk_label = "High Risk" if pred == 1 else "Low Risk"
    return risk_label, prob_high
