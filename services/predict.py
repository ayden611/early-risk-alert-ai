import os
import math

import joblib
import numpy as np
from flask import current_app

# demo_model.pkl is stored at repo root (same level as app.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "demo_model.pkl")

_model = joblib.load(MODEL_PATH)


def _exercise_to_num(level: str) -> float:
    m = {"Low": 0.0, "Moderate": 1.0, "High": 2.0}
    return m.get((level or "").strip(), 0.0)


def _smooth_probability(p: float) -> float:
    """
    Fast, stable "calibration-lite" so probabilities don't slam to 0% / 100%.
    Uses temperature scaling on the logit, then clips.
    """
    temp = float(current_app.config.get("PROB_TEMP", 2.0))
    floor = float(current_app.config.get("PROB_FLOOR", 0.02))
    ceil = float(current_app.config.get("PROB_CEIL", 0.98))

    p = float(p)
    p = min(max(p, 1e-6), 1 - 1e-6)  # avoid log(0)

    logit = math.log(p / (1 - p))
    logit_scaled = logit / max(temp, 1e-6)
    p2 = 1 / (1 + math.exp(-logit_scaled))

    return float(min(max(p2, floor), ceil))


def predict_risk(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    """
    Returns:
      label: "High" / "Low"
      p: smoothed probability in [0,1]
      raw_p: original model probability in [0,1]
    """
    ex = _exercise_to_num(exercise_level)

    X = np.array([[age, bmi, ex, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

    raw_p = float(_model.predict_proba(X)[0][1])  # class 1 = High
    p = _smooth_probability(raw_p)

    label = "High" if p >= 0.5 else "Low"
    return label, p, raw_p
