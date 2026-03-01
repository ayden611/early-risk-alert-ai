import os
import time
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from flask import Blueprint, request, jsonify, current_app

from era.extensions import db

# Optional DB model (we will try to save if it exists)
try:
    from era.models import Prediction  # type: ignore
except Exception:
    Prediction = None  # type: ignore

# IMPORTANT: your project imports predict_risk from root predict.py
from predict import predict_risk  # noqa: E402


api_bp = Blueprint("api", __name__)

# ----------------------------
# In-memory magic codes (simple + stable)
# ----------------------------
_MAGIC_CODES = {}  # user_id -> {"code": "123456", "exp": epoch_seconds}
MAGIC_CODE_TTL_SECONDS = 10 * 60  # 10 minutes


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_prod() -> bool:
    # Render sets RENDER / you may set ENV=production
    env = (os.getenv("ENV") or os.getenv("FLASK_ENV") or "").lower()
    return env == "production" or bool(os.getenv("RENDER"))


def _num(value, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"'{field}' is required")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number")


def _exercise(value) -> str:
    """
    Normalize exercise_level to one of: Low, Moderate, High
    Accepts: "low"/"0", "moderate"/"1"/"medium", "high"/"2"
    """
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


def _auth_secret() -> str:
    secret = current_app.config.get("SECRET_KEY") or os.getenv("SECRET_KEY")
    if not secret:
        # fallback (keeps app from crashing, but you SHOULD set SECRET_KEY in Render)
        secret = "dev-secret-change-me"
    return secret


def _make_token(user_id: str) -> str:
    payload = {
        "sub": str(user_id),
        "iat": int(_now_utc().timestamp()),
        "exp": int((_now_utc() + timedelta(days=30)).timestamp()),
    }
    return jwt.encode(payload, _auth_secret(), algorithm="HS256")


def _get_bearer_token() -> str | None:
    h = request.headers.get("Authorization", "")
    if h.startswith("Bearer "):
        return h.split(" ", 1)[1].strip()
    return None


def _decode_token(token: str) -> dict:
    return jwt.decode(token, _auth_secret(), algorithms=["HS256"])


# ----------------------------
# Basic routes
# ----------------------------
@api_bp.get("/")
def api_root():
    # Remember: this is mounted at /api/v1/
    return jsonify(
        {
            "ok": True,
            "service": "early-risk-alert-mobile-api",
            "base_path": "/api/v1",
            "endpoints": [
                "GET  /api/v1/health",
                "POST /api/v1/predict",
                "POST /api/v1/auth/login",
                "POST /api/v1/auth/verify",
                "POST /api/v1/auth/refresh",
            ],
        }
    ), 200


@api_bp.get("/health")
def health():
    return jsonify({"ok": True}), 200


# ----------------------------
# Prediction
# ----------------------------
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

    # Save to DB if Prediction model exists + table exists
    if Prediction is not None:
        try:
            rec = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=risk_label,
                probability=probability,
                created_at=_now_utc(),
            )
            db.session.add(rec)
            db.session.commit()
        except Exception:
            # Don't crash API if DB/table/columns differ
            db.session.rollback()

    return jsonify({"risk_label": risk_label, "probability": probability}), 200


# ----------------------------
# Auth (Magic code -> JWT)
# ----------------------------
@api_bp.post("/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "'user_id' is required"}), 400

    # 6-digit code
    code = f"{secrets.randbelow(1_000_000):06d}"
    exp = int(time.time()) + MAGIC_CODE_TTL_SECONDS
    _MAGIC_CODES[user_id] = {"code": code, "exp": exp}

    resp = {
        "ok": True,
        "sent": True,
        "expires_in": MAGIC_CODE_TTL_SECONDS,
    }

    # For development/testing only so you can move forward fast
    if not _is_prod():
        resp["dev_code"] = code

    return jsonify(resp), 200


@api_bp.post("/auth/verify")
def auth_verify():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id") or "").strip()
    code = str(data.get("code") or "").strip()

    if not user_id or not code:
        return jsonify(
            {"error": "validation", "message": "'user_id' and 'code' are required"}
        ), 400

    entry = _MAGIC_CODES.get(user_id)
    if not entry:
        return jsonify({"error": "auth", "message": "No code requested"}), 401

    if int(time.time()) > int(entry["exp"]):
        _MAGIC_CODES.pop(user_id, None)
        return jsonify({"error": "auth", "message": "Code expired"}), 401

    if code != entry["code"]:
        return jsonify({"error": "auth", "message": "Invalid code"}), 401

    # Success -> issue JWT
    _MAGIC_CODES.pop(user_id, None)
    token = _make_token(user_id)

    return jsonify({"ok": True, "token": token}), 200


@api_bp.post("/auth/refresh")
def auth_refresh():
    token = _get_bearer_token()
    if not token:
        return jsonify({"error": "auth", "message": "Missing Bearer token"}), 401

    try:
        payload = _decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return jsonify({"error": "auth", "message": "Invalid token"}), 401
    except Exception:
        return jsonify({"error": "auth", "message": "Invalid token"}), 401

    new_token = _make_token(str(user_id))
    return jsonify({"ok": True, "token": new_token}), 200
