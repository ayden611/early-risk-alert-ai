# era/api/routes.py
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from flask import Blueprint, current_app, jsonify, request

from era.extensions import db

# Optional DB model (save predictions if it exists)
try:
    from era.models import Prediction  # type: ignore
except Exception:
    Prediction = None  # type: ignore

# Your model function lives at the repo root predict.py (per your setup)
try:
    from predict import predict_risk  # type: ignore
except Exception:
    predict_risk = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# Simple in-memory auth store
# NOTE: This works best with WEB_CONCURRENCY=1 on Render.
# For multi-worker scaling, store codes in DB/Redis instead.
# ----------------------------
_AUTH_CODES: dict[str, dict] = {}  # user_id -> {"code": str, "exp": datetime}
_REFRESH_ALLOW: dict[str, dict] = {}  # refresh_jti -> {"user_id": str, "exp": datetime}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _jwt_secret() -> str:
    # Uses Flask SECRET_KEY if present, else env JWT_SECRET, else fallback (not recommended for prod)
    return (
        current_app.config.get("SECRET_KEY")
        or os.getenv("JWT_SECRET")
        or "dev-secret-change-me"
    )


def _jwt_alg() -> str:
    return os.getenv("JWT_ALG", "HS256")


def _json_error(kind: str, msg: str, status: int):
    return jsonify({"error": kind, "message": msg}), status


def _require_json() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _issue_tokens(user_id: str) -> dict:
    secret = _jwt_secret()
    alg = _jwt_alg()

    access_exp = _now() + timedelta(minutes=int(os.getenv("ACCESS_TOKEN_MIN", "15")))
    refresh_exp = _now() + timedelta(days=int(os.getenv("REFRESH_TOKEN_DAYS", "7")))

    refresh_jti = secrets.token_urlsafe(16)

    access_payload = {
        "sub": user_id,
        "type": "access",
        "exp": int(access_exp.timestamp()),
        "iat": int(_now().timestamp()),
    }
    refresh_payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": refresh_jti,
        "exp": int(refresh_exp.timestamp()),
        "iat": int(_now().timestamp()),
    }

    access_token = jwt.encode(access_payload, secret, algorithm=alg)
    refresh_token = jwt.encode(refresh_payload, secret, algorithm=alg)

    # allow refresh token usage (basic allowlist)
    _REFRESH_ALLOW[refresh_jti] = {"user_id": user_id, "exp": refresh_exp}

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "access_expires_in": int((access_exp - _now()).total_seconds()),
    }


def _decode_token(token: str) -> dict:
    secret = _jwt_secret()
    alg = _jwt_alg()
    return jwt.decode(token, secret, algorithms=[alg])


def _exercise_norm(value):
    if value is None:
        raise ValueError("'exercise_level' is required")
    v = str(value).strip().lower()
    if v in ("low", "0"):
        return "Low"
    if v in ("moderate", "1", "medium"):
        return "Moderate"
    if v in ("high", "2"):
        return "High"
    raise ValueError("'exercise_level' must be Low, Moderate, or High")


def _num(value, field):
    if value is None or value == "":
        raise ValueError(f"'{field}' is required")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number")


# ----------------------------
# Routes
# ----------------------------

@api_bp.get("/")
def api_root():
    # This prevents 404s when you hit /api/v1/
    return jsonify({"ok": True, "service": "early-risk-alert-mobile-api"}), 200


@api_bp.get("/health")
def health():
    return jsonify({"ok": True}), 200


@api_bp.post("/auth/login")
def auth_login():
    """
    Body: {"user_id":"123"}
    Creates a 6-digit code (expires in 10 min) and logs it in Render logs.
    """
    data = _require_json()
    user_id = str(data.get("user_id") or "").strip()
    if not user_id:
        return _json_error("validation", "'user_id' is required", 400)

    code = f"{secrets.randbelow(1_000_000):06d}"
    exp = _now() + timedelta(seconds=int(os.getenv("AUTH_CODE_TTL_SEC", "600")))

    _AUTH_CODES[user_id] = {"code": code, "exp": exp}

    # In real life you would send SMS/email.
    # For now: log it so you can verify immediately.
    current_app.logger.info("LOGIN_CODE user_id=%s code=%s exp=%s", user_id, code, exp)

    return jsonify({"ok": True, "sent": True, "expires_in": int((exp - _now()).total_seconds())}), 200


@api_bp.post("/auth/verify")
def auth_verify():
    """
    Body: {"user_id":"123", "code":"123456"}
    Verifies code, returns access + refresh tokens.
    """
    data = _require_json()
    user_id = str(data.get("user_id") or "").strip()
    code = str(data.get("code") or "").strip()

    if not user_id or not code:
        return _json_error("validation", "'user_id' and 'code' are required", 400)

    rec = _AUTH_CODES.get(user_id)
    if not rec:
        return _json_error("auth", "No active code. Call /auth/login again.", 401)

    if _now() > rec["exp"]:
        _AUTH_CODES.pop(user_id, None)
        return _json_error("auth", "Code expired", 401)

    if code != rec["code"]:
        return _json_error("auth", "Invalid code", 401)

    # one-time use
    _AUTH_CODES.pop(user_id, None)

    tokens = _issue_tokens(user_id)
    return jsonify({"ok": True, **tokens}), 200


@api_bp.post("/auth/refresh")
def auth_refresh():
    """
    Header: Authorization: Bearer <refresh_token>
    Returns a new access token (and optionally a new refresh token if you want later).
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return _json_error("auth", "Missing Bearer token", 401)

    token = auth.split(" ", 1)[1].strip()
    try:
        payload = _decode_token(token)
    except Exception:
        return _json_error("auth", "Invalid token", 401)

    if payload.get("type") != "refresh":
        return _json_error("auth", "Wrong token type", 401)

    user_id = str(payload.get("sub") or "").strip()
    jti = str(payload.get("jti") or "").strip()
    if not user_id or not jti:
        return _json_error("auth", "Invalid refresh token payload", 401)

    allow = _REFRESH_ALLOW.get(jti)
    if not allow:
        return _json_error("auth", "Refresh token revoked/unknown", 401)

    if _now() > allow["exp"]:
        _REFRESH_ALLOW.pop(jti, None)
        return _json_error("auth", "Refresh token expired", 401)

    tokens = _issue_tokens(user_id)
    return jsonify({"ok": True, **tokens}), 200


@api_bp.post("/predict")
def api_predict():
    """
    Body:
    {
      "age": 45,
      "bmi": 27.5,
      "exercise_level": "Moderate",
      "systolic_bp": 130,
      "diastolic_bp": 85,
      "heart_rate": 72
    }
    """
    if predict_risk is None:
        return _json_error("server", "Model not available (predict_risk import failed)", 500)

    data = _require_json()

    try:
        age = _num(data.get("age"), "age")
        bmi = _num(data.get("bmi"), "bmi")
        exercise_level = _exercise_norm(data.get("exercise_level"))
        systolic_bp = _num(data.get("systolic_bp"), "systolic_bp")
        diastolic_bp = _num(data.get("diastolic_bp"), "diastolic_bp")
        heart_rate = _num(data.get("heart_rate"), "heart_rate")
    except ValueError as e:
        return _json_error("validation", str(e), 400)

    risk_label, probability = predict_risk(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
    )

    # Save if DB model exists
    if Prediction is not None:
        try:
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
        except Exception:
            db.session.rollback()
            # don't fail prediction if DB save fails

    return jsonify({"risk_label": risk_label, "probability": probability}), 200
