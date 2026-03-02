import os
import time
import json
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from flask import Blueprint, request, jsonify, current_app

from era.extensions import db

# DB model is optional (app still runs without it)
try:
    from era.models import Prediction  # type: ignore
except Exception:
    Prediction = None  # type: ignore

# Your prediction function (root-level predict.py)
# IMPORTANT: this must exist at repo root as predict.py with predict_risk(...)
from predict import predict_risk  # type: ignore


api_bp = Blueprint("api", __name__)


# ----------------------------
# Helpers
# ----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _json(data, code=200):
    return jsonify(data), code


def _get_jwt_secret() -> str:
    # MUST be set on Render as env var (recommended), but fallback works locally
    return os.getenv("JWT_SECRET", current_app.config.get("SECRET_KEY", "dev-secret-change-me"))


def _issue_token(user_id: str, minutes: int = 60) -> str:
    payload = {
        "sub": user_id,
        "iat": int(_now_utc().timestamp()),
        "exp": int((_now_utc() + timedelta(minutes=minutes)).timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm="HS256")


def _decode_token(token: str) -> dict:
    return jwt.decode(token, _get_jwt_secret(), algorithms=["HS256"])


def _bearer_token() -> str | None:
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer "):
        return None
    return h.split(" ", 1)[1].strip() or None


def require_auth():
    token = _bearer_token()
    if not token:
        return None, _json({"error": "auth", "message": "Missing Bearer token"}, 401)
    try:
        payload = _decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None, _json({"error": "auth", "message": "Invalid token payload"}, 401)
        return user_id, None
    except jwt.ExpiredSignatureError:
        return None, _json({"error": "auth", "message": "Token expired"}, 401)
    except Exception:
        return None, _json({"error": "auth", "message": "Invalid token"}, 401)


def _num(value, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"'{field}' is required")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number")


def _exercise(value) -> str:
    # Normalize exercise level to Low/Moderate/High for predict_risk
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


def _log_event(event: str, extra: dict | None = None):
    # Simple structured logs to Render
    payload = {"event": event, "ts": int(_now_utc().timestamp())}
    if extra:
        payload.update(extra)
    current_app.logger.info(json.dumps(payload))


# ----------------------------
# Basic service endpoints
# ----------------------------
@api_bp.get("/")
def root():
    return _json({"ok": True, "service": "early-risk-alert-api"})


@api_bp.get("/health")
def health():
    db_ok = True
    db_error = None
    try:
        db.session.execute(db.text("SELECT 1"))  # type: ignore[attr-defined]
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return _json(
        {
            "ok": True,
            "db_ok": db_ok,
            "db_error": db_error,
            "time_utc": _now_utc().isoformat(),
        },
        200 if db_ok else 503,
    )


# ----------------------------
# Auth (simple OTP demo)
# ----------------------------
# In-memory OTP store (works on single instance; good for demo)
# For real production, store in DB or Redis.
_OTP = {}  # user_id -> {"code": "123456", "exp": epoch_seconds}


def _otp_issue(user_id: str, ttl_seconds: int = 600) -> str:
    code = f"{secrets.randbelow(1_000_000):06d}"
    _OTP[user_id] = {"code": code, "exp": int(time.time()) + ttl_seconds}
    return code


def _otp_verify(user_id: str, code: str) -> bool:
    rec = _OTP.get(user_id)
    if not rec:
        return False
    if int(time.time()) > int(rec["exp"]):
        return False
    return str(rec["code"]) == str(code)


@api_bp.post("/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "")).strip()
    if not user_id:
        return _json({"error": "validation", "message": "'user_id' is required"}, 400)

    # Issue OTP (demo). In real life you'd send SMS/email.
    code = _otp_issue(user_id, ttl_seconds=600)

    _log_event("auth_login", {"user_id": user_id})
    # We do NOT return the code to client in production.
    # For demo/testing, you can optionally return it if DEMO_RETURN_OTP=1
    if os.getenv("DEMO_RETURN_OTP", "0") == "1":
        return _json({"ok": True, "sent": True, "expires_in": 600, "code": code})

    return _json({"ok": True, "sent": True, "expires_in": 600})


@api_bp.post("/auth/verify")
def auth_verify():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "")).strip()
    code = str(data.get("code", "")).strip()

    if not user_id or not code:
        return _json({"error": "validation", "message": "'user_id' and 'code' are required"}, 400)

    if not _otp_verify(user_id, code):
        # differentiate expired vs wrong for your logs
        rec = _OTP.get(user_id)
        if rec and int(time.time()) > int(rec["exp"]):
            return _json({"error": "auth", "message": "Code expired"}, 401)
        return _json({"error": "auth", "message": "Invalid code"}, 401)

    token = _issue_token(user_id, minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "60")))
    _log_event("auth_verify_ok", {"user_id": user_id})
    return _json({"ok": True, "token": token, "token_type": "Bearer"})


@api_bp.post("/auth/refresh")
def auth_refresh():
    # Refresh simply re-issues if current token valid (demo style)
    user_id, err = require_auth()
    if err:
        return err
    token = _issue_token(user_id, minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "60")))
    return _json({"ok": True, "token": token, "token_type": "Bearer"})


# ----------------------------
# Predict (JWT protected)
# ----------------------------
@api_bp.post("/predict")
def api_predict():
    user_id, err = require_auth()
    if err:
        return err

    data = request.get_json(silent=True) or {}

    try:
        age = _num(data.get("age"), "age")
        bmi = _num(data.get("bmi"), "bmi")
        exercise_level = _exercise(data.get("exercise_level"))
        systolic_bp = _num(data.get("systolic_bp"), "systolic_bp")
        diastolic_bp = _num(data.get("diastolic_bp"), "diastolic_bp")
        heart_rate = _num(data.get("heart_rate"), "heart_rate")
    except ValueError as e:
        return _json({"error": "validation", "message": str(e)}, 400)

    # Run model
    risk_label, probability = predict_risk(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
    )

    # Save (if DB model exists)
    saved = False
    if Prediction is not None:
        try:
            pred = Prediction(
                user_id=str(user_id),
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
            db.session.add(pred)
            db.session.commit()
            saved = True
        except Exception as e:
            # Don’t break predict if DB fails
            db.session.rollback()
            _log_event("db_save_failed", {"error": str(e)})

    _log_event(
        "predict",
        {"user_id": user_id, "risk_label": risk_label, "probability": probability, "saved": saved},
    )

    return _json(
        {
            "ok": True,
            "risk_label": risk_label,
            "probability": probability,
            "saved": saved,
        }
    )


# ----------------------------
# History (JWT protected)
# ----------------------------
@api_bp.get("/history")
def history():
    user_id, err = require_auth()
    if err:
        return err

    if Prediction is None:
        return _json({"ok": True, "items": [], "note": "Prediction model/table not available"}, 200)

    try:
        q = (
            db.session.query(Prediction)  # type: ignore
            .filter(getattr(Prediction, "user_id") == str(user_id))  # type: ignore
            .order_by(getattr(Prediction, "created_at").desc())  # type: ignore
            .limit(int(os.getenv("HISTORY_LIMIT", "50")))
        )
        items = []
        for p in q:
            items.append(
                {
                    "id": getattr(p, "id", None),
                    "created_at": getattr(p, "created_at", None).isoformat() if getattr(p, "created_at", None) else None,
                    "age": getattr(p, "age", None),
                    "bmi": getattr(p, "bmi", None),
                    "exercise_level": getattr(p, "exercise_level", None),
                    "systolic_bp": getattr(p, "systolic_bp", None),
                    "diastolic_bp": getattr(p, "diastolic_bp", None),
                    "heart_rate": getattr(p, "heart_rate", None),
                    "risk_label": getattr(p, "risk_label", None),
                    "probability": getattr(p, "probability", None),
                }
            )
        return _json({"ok": True, "items": items})
    except Exception as e:
        return _json({"error": "server", "message": str(e)}, 500)
