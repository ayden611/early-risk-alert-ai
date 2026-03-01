import os
from datetime import datetime, timedelta, timezone

import jwt
from flask import Blueprint, request, jsonify

from era.extensions import db
from era.models import Prediction

# IMPORTANT:
# You deleted/moved services/, so import predict_risk from root predict.py file.
from predict import predict_risk

api_bp = Blueprint("api", __name__)


# ----------------------------
# Helpers
# ----------------------------
def _json_error(code: str, message: str, status: int = 400):
    return jsonify({"error": code, "message": message}), status


def _num(value, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"'{field}' is required")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number")


def _exercise(value) -> str:
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


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "dev-secret-change-me")


def _issue_access_token(user_id: str) -> str:
    minutes = int(os.getenv("ACCESS_TOKEN_MINUTES", "30"))
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _issue_refresh_token(user_id: str) -> str:
    days = int(os.getenv("REFRESH_TOKEN_DAYS", "14"))
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=days)).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _decode_token(token: str) -> dict:
    return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])


def _bearer_token() -> str | None:
    h = request.headers.get("Authorization", "")
    if h.lower().startswith("bearer "):
        return h.split(" ", 1)[1].strip()
    return None


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/")
def root():
    # Helps Render + quick sanity check
    return jsonify({"ok": True, "service": "early-risk-alert-mobile-api"}), 200


@api_bp.get("/health")
def health():
    # Optional DB check (won’t crash if DB is down)
    db_ok = True
    db_error = None
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return (
        jsonify(
            {
                "ok": True,
                "db_ok": db_ok,
                "db_error": db_error,
            }
        ),
        (200 if db_ok else 503),
    )


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
        return _json_error("validation", str(e), 400)

    # Run model
    try:
        risk_label, probability = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )
    except Exception as e:
        return _json_error("model_error", f"Prediction failed: {e}", 500)

    # Save to DB (safe: won’t break deploy if table isn’t ready)
    try:
        rec = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=str(risk_label),
            probability=float(probability),
        )
        db.session.add(rec)
        db.session.commit()
    except Exception:
        # Don’t crash request if DB/table isn’t set up yet
        try:
            db.session.rollback()
        except Exception:
            pass

    return (
        jsonify(
            {
                "risk_label": str(risk_label),
                "probability": float(probability),
            }
        ),
        200,
    )


# ----------------------------
# Auth (simple JWT)
# ----------------------------
@api_bp.post("/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")

    if not user_id:
        return _json_error("user_id_required", "Provide {\"user_id\":\"123\"}", 400)

    return (
        jsonify(
            {
                "access_token": _issue_access_token(str(user_id)),
                "refresh_token": _issue_refresh_token(str(user_id)),
                "token_type": "Bearer",
            }
        ),
        200,
    )


@api_bp.post("/auth/refresh")
def auth_refresh():
    data = request.get_json(silent=True) or {}
    refresh_token = data.get("refresh_token") or _bearer_token()

    if not refresh_token:
        return _json_error(
            "refresh_token_required",
            "Provide refresh_token in JSON or Authorization: Bearer <token>",
            400,
        )

    try:
        payload = _decode_token(refresh_token)
        if payload.get("type") != "refresh":
            return _json_error("invalid_token", "Not a refresh token", 401)

        user_id = payload.get("sub")
        if not user_id:
            return _json_error("invalid_token", "Missing sub", 401)

        return (
            jsonify(
                {
                    "access_token": _issue_access_token(str(user_id)),
                    "token_type": "Bearer",
                }
            ),
            200,
        )

    except jwt.ExpiredSignatureError:
        return _json_error("expired_token", "Refresh token expired", 401)
    except Exception as e:
        return _json_error("invalid_token", f"Could not decode token: {e}", 401)
