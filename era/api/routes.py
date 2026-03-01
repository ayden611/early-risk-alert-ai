import os
from datetime import datetime, timedelta, timezone
import jwt
from flask import Blueprint,request, jsonify, current_app

from era.extensions import db
from era.models import Prediction

# IMPORTANT:
# You deleted the services/ folder, so import predict_risk from the root predict.py file.
from predict import predict_risk

# make sure Blueprint is imported

api_bp = Blueprint("api", __name__)

@api_bp.get("/")
def root():
    return jsonify({"ok": True, "service": "early-risk-alert-mobile-api"}), 200


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


@api_bp.get("/health")
def health():
    return jsonify({"ok": True}), 200


@api_bp.post("/predict")
def api_predict():
    data = request.get_json(silent=True) or {}

    try:
        age = _num(data.get("age"), "age")
        bmi = _num(data.get("bmi"), "bmi")
        exercise_level = _exercise(data.get("exercise_level"))
        systolic_bp = _num(data.get("systolic_bp"), "systolic_bp")
        diastolic_bp = _num(data.get("diastolic_bp"), "diastolic_bp")
        heart_rate = _num(data.get("heart_rate"), "heart_rate")
    except ValueError as e:
        return jsonify({"error": "validation", "message": str(e)}), 400

    # Run model
    risk_label, probability = predict_risk(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
    )

    # Save to DB (if your model + DB table exists)
    pred = Prediction(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=probability,
    )
    db.session.add(pred)
    db.session.commit()

    return jsonify(
        {
            "risk_label": risk_label,
            "probability": probability,
        }
    ), 200

# ----------------------------
# Mobile auth: refresh token
# ----------------------------

def _jwt_secret():
    # Prefer config; fallback to env (won't break existing setups)
    return current_app.config.get("JWT_SECRET") or os.getenv("JWT_SECRET", "dev-change-me")


def _issue_access_token(user_id: str):
    now = datetime.now(timezone.utc)
    minutes = int(current_app.config.get("JWT_ACCESS_MINUTES", 30))
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _issue_refresh_token(user_id: str):
    now = datetime.now(timezone.utc)
    days = int(current_app.config.get("JWT_REFRESH_DAYS", 14))
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=days)).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _decode_token(token: str):
    return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])


@api_bp.post("/auth/refresh")
def auth_refresh():
    """
    Mobile flow:
    - app stores refresh_token (Keychain/Keystore)
    - when access token expires -> call /auth/refresh -> get new access_token
    """
    data = request.get_json(silent=True) or {}
    refresh_token = data.get("refresh_token")

    if not refresh_token:
        return jsonify({"error": "missing_refresh_token"}), 400

    try:
        payload = _decode_token(refresh_token)
        if payload.get("type") != "refresh":
            return jsonify({"error": "invalid_token_type"}), 401

        user_id = payload.get("sub")
        new_access = _issue_access_token(user_id)

        rotate = str(current_app.config.get("JWT_ROTATE_REFRESH", "true")).lower() == "true"
        if rotate:
            new_refresh = _issue_refresh_token(user_id)
            return jsonify({"access_token": new_access, "refresh_token": new_refresh}), 200

        return jsonify({"access_token": new_access}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "refresh_expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "refresh_invalid"}), 401
