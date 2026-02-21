from flask import Blueprint, request, jsonify

from era.extensions import db
from era.models import Prediction
from era.services.predict_risk import predict_risk

bp = Blueprint("api", __name__, url_prefix="/api")


def _num(value, field):
    """Parse a number safely; raise ValueError with a helpful message."""
    if value is None or value == "":
        raise ValueError(f"'{field}' is required")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number")


def _exercise(value):
    """Normalize exercise level to one of: Low, Moderate, High"""
    if value is None or value == "":
        raise ValueError("'exercise_level' is required")
    v = str(value).strip().lower()
    if v in ("low", "0"):
        return "Low"
    if v in ("moderate", "1", "medium"):
        return "Moderate"
    if v in ("high", "2"):
        return "High"
    raise ValueError("'exercise_level' must be Low, Moderate, or High")


@bp.get("/health")
def health():
    return jsonify({"ok": True}), 200


@bp.post("/predict")
def api_predict():
    """
    JSON in:
      {
        "age": 42,
        "bmi": 28.0,
        "exercise_level": "Low",
        "systolic_bp": 170,
        "diastolic_bp": 90,
        "heart_rate": 100
      }

    JSON out:
      {
        "risk": "High",
        "probability": 0.94,
        "probability_pct": 94.0
      }
    """
    data = request.get_json(silent=True) or {}

    try:
        age = _num(data.get("age"), "age")
        bmi = _num(data.get("bmi"), "bmi")
        exercise_level = _exercise(data.get("exercise_level"))
        systolic_bp = _num(data.get("systolic_bp"), "systolic_bp")
        diastolic_bp = _num(data.get("diastolic_bp"), "diastolic_bp")
        heart_rate = _num(data.get("heart_rate"), "heart_rate")
    except ValueError as e:
        return jsonify({"error": "validation_error", "message": str(e)}), 400

    # Predict (service should return: (risk_label_str, probability_float_0_to_1))
    risk_label, prob = predict_risk(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
    )

    # Save to DB
    row = Prediction(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=float(prob),
    )
    db.session.add(row)
    db.session.commit()

    return jsonify(
        {
            "risk": risk_label,
            "probability": float(prob),
            "probability_pct": round(float(prob) * 100, 1),
        }
    ), 200
