# era/api/routes.py
import os
import time
import json
import secrets
from datetime import datetime, timedelta, timezone

import jwt
import numpy as np
from flask import Blueprint, request, jsonify, current_app

from era.extensions import db

# Optional DB model (we will try to save if it exists)
try:
    from era.models import Prediction  # type: ignore
except Exception:
    Prediction = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# Config helpers
# ----------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_jwt_secret() -> str:
    # Prefer explicit JWT_SECRET; fall back to Flask SECRET_KEY; then env SECRET_KEY; then a dev default.
    return (
        os.getenv("JWT_SECRET")
        or (current_app.config.get("JWT_SECRET") if current_app else None)
        or os.getenv("SECRET_KEY")
        or (current_app.config.get("SECRET_KEY") if current_app else None)
        or "dev-secret-change-me"
    )


def _jwt_access_minutes() -> int:
    try:
        return int(os.getenv("JWT_ACCESS_MINUTES", current_app.config.get("JWT_ACCESS_MINUTES", 60)))
    except Exception:
        return 60


def _magic_code_ttl_seconds() -> int:
    try:
        return int(os.getenv("MAGIC_CODE_TTL", current_app.config.get("MAGIC_CODE_TTL", 600)))
    except Exception:
        return 600


def _demo_return_otp_enabled() -> bool:
    # Your Render env var should be: DEMO_RETURN_OTP = 1
    return _env_bool("DEMO_RETURN_OTP", default=False)


# ----------------------------
# In-memory OTP store (demo)
# NOTE: resets on deploy/restart (OK for demo)
# ----------------------------
_OTP_STORE: dict[str, dict] = {}
# structure: { user_id: {"code": "123456", "expires_at": epoch_seconds} }


def _issue_code(user_id: str) -> dict:
    ttl = _magic_code_ttl_seconds()
    code = f"{secrets.randbelow(1000000):06d}"
    _OTP_STORE[user_id] = {"code": code, "expires_at": time.time() + ttl}
    return {"code": code, "expires_in": ttl}


def _check_code(user_id: str, code: str) -> tuple[bool, str]:
    rec = _OTP_STORE.get(user_id)
    if not rec:
        return False, "No code requested (call /auth/login first)"
    if time.time() > float(rec.get("expires_at", 0)):
        return False, "Code expired"
    if str(rec.get("code")) != str(code):
        return False, "Invalid code"
    return True, "ok"


# ----------------------------
# Model loading (lazy)
# ----------------------------
_MODEL = None


def _project_root() -> str:
    # routes.py is at era/api/routes.py -> go up two levels to project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # If you store model path in env/model config, respect it.
    model_path = os.getenv("MODEL_PATH") or current_app.config.get("MODEL_PATH") or "demo_model.pkl"
    if not os.path.isabs(model_path):
        model_path = os.path.join(_project_root(), model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    import joblib  # local import so app can boot even if joblib missing in some envs

    _MODEL = joblib.load(model_path)
    return _MODEL


def _exercise_to_num(exercise_level) -> float:
    # Accept numeric already
    try:
        if exercise_level is None:
            return 0.0
        if isinstance(exercise_level, (int, float)):
            return float(exercise_level)
        s = str(exercise_level).strip().lower()
        mapping = {"low": 0.0, "moderate": 1.0, "medium": 1.0, "high": 2.0}
        if s in mapping:
            return mapping[s]
        # allow "0", "1", "2"
        return float(s)
    except Exception:
        return 0.0


def _get_payload() -> dict:
    return request.get_json(silent=True) or {}


def _require_fields(payload: dict, fields: list[str]) -> list[str]:
    missing = []
    for f in fields:
        v = payload.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(f)
    return missing


# ----------------------------
# JWT helpers
# ----------------------------
def _make_token(user_id: str) -> str:
    secret = _get_jwt_secret()
    minutes = _jwt_access_minutes()
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=minutes)

    claims = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "type": "access",
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def _read_bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "") or ""
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _auth_user_id() -> tuple[bool, str | None, str | None]:
    token = _read_bearer_token()
    if not token:
        return False, None, "Missing Bearer token"

    try:
        secret = _get_jwt_secret()
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
        user_id = decoded.get("sub")
        if not user_id:
            return False, None, "Invalid token (missing sub)"
        return True, str(user_id), None
    except jwt.ExpiredSignatureError:
        return False, None, "Token expired"
    except Exception:
        return False, None, "Invalid token"


# ----------------------------
# Health Risk "Intelligence Engine"
# ----------------------------
def _health_risk_intelligence(payload: dict, prob_high: float, risk_label: str) -> dict:
    """
    Lightweight "intelligence layer" that turns raw model output into:
    - risk_score (0-100)
    - risk_factors (human-readable)
    - recommendations (actionable)

    prob_high: model probability for High Risk (0..1)
    risk_label: "High Risk" or "Low Risk"
    """
    # ---- robust reads (handles missing keys) ----
    age = float(payload.get("age", 0) or 0)
    bmi = float(payload.get("bmi", 0) or 0)
    exercise = str(payload.get("exercise_level", "") or "").strip().lower()
    sbp = float(payload.get("systolic_bp", payload.get("sys_bp", 0)) or 0)
    dbp = float(payload.get("diastolic_bp", payload.get("dia_bp", 0)) or 0)
    hr = float(payload.get("heart_rate", 0) or 0)

    # ---- base score primarily from model probability ----
    score = int(round(max(0.0, min(1.0, prob_high)) * 100))

    # Small “nudges” (keep model as main signal)
    if age >= 55:
        score += 5
    if bmi >= 30:
        score += 7
    elif bmi >= 25:
        score += 3
    if sbp >= 140 or dbp >= 90:
        score += 7
    elif sbp >= 130 or dbp >= 80:
        score += 3
    if hr >= 90:
        score += 2
    if exercise in ("low", "none", "sedentary"):
        score += 3

    score = max(0, min(100, score))

    # ---- human-readable factors ----
    factors = []
    if age:
        factors.append(f"Age: {int(age)}")
    if bmi:
        factors.append(f"BMI: {bmi:.1f}")
    if sbp or dbp:
        factors.append(f"Blood pressure: {int(sbp)}/{int(dbp)}")
    if hr:
        factors.append(f"Heart rate: {int(hr)}")
    if exercise:
        factors.append(f"Exercise level: {exercise}")

    # ---- recommendations (simple + safe) ----
    recs = []
    if sbp >= 140 or dbp >= 90:
        recs.append("Consider checking your blood pressure again and discussing persistent high readings with a clinician.")
    elif sbp >= 130 or dbp >= 80:
        recs.append("If you consistently see elevated blood pressure, consider lifestyle changes and periodic monitoring.")

    if bmi >= 30:
        recs.append("A gradual, sustainable weight plan (nutrition + activity) can lower long-term cardiovascular risk.")
    elif bmi >= 25:
        recs.append("Small changes in diet and activity can improve BMI over time.")

    if exercise in ("low", "none", "sedentary"):
        recs.append("Aim for regular light-to-moderate activity (e.g., walking) and increase gradually.")
    else:
        recs.append("Keep up consistent activity; consistency is more important than intensity.")

    if hr >= 90:
        recs.append("If resting heart rate is frequently high, consider hydration, sleep, stress management, and medical guidance if persistent.")

    # Safety note
    recs.append("This is not a medical diagnosis. If you have symptoms or concerns, seek professional medical advice.")

    return {
        "risk_score": score,
        "risk_factors": factors,
        "recommendations": recs,
        "model_risk_label": risk_label,
    }


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/")
def root():
    return jsonify(
        {
            "ok": True,
            "service": "early-risk-alert-mobile-api",
            "version": "v1",
            "endpoints": ["/health", "/auth/login", "/auth/verify", "/predict", "/demo/predict"],
        }
    )


@api_bp.get("/health")
def health():
    # quick DB ping if available
    db_ok = True
    db_error = None
    try:
        db.session.execute(db.text("SELECT 1"))  # type: ignore[attr-defined]
    except Exception as e:
        db_ok = False
        db_error = str(e)

    model_ok = True
    model_error = None
    try:
        _load_model()
    except Exception as e:
        model_ok = False
        model_error = str(e)

    return jsonify(
        {
            "ok": True,
            "db_ok": db_ok,
            "db_error": db_error,
            "model_ok": model_ok,
            "model_error": model_error,
            "demo_return_otp": _demo_return_otp_enabled(),
        }
    )


@api_bp.post("/auth/login")
def auth_login():
    payload = _get_payload()
    missing = _require_fields(payload, ["user_id"])
    if missing:
        return jsonify({"error": "validation", "message": f"{missing[0]} is required"}), 400

    user_id = str(payload["user_id"]).strip()
    issued = _issue_code(user_id)

    # In real life you'd email/sms; for demo we can return it if enabled
    resp = {"ok": True, "sent": True, "expires_in": issued["expires_in"]}
    if _demo_return_otp_enabled():
        resp["code"] = issued["code"]

    return jsonify(resp), 200


@api_bp.post("/auth/verify")
def auth_verify():
    payload = _get_payload()
    missing = _require_fields(payload, ["user_id", "code"])
    if missing:
        return jsonify({"error": "validation", "message": "user_id and code are required"}), 400

    user_id = str(payload["user_id"]).strip()
    code = str(payload["code"]).strip()

    ok, msg = _check_code(user_id, code)
    if not ok:
        return jsonify({"error": "auth", "message": msg}), 401

    token = _make_token(user_id)

    # Optional: clear OTP after success
    try:
        _OTP_STORE.pop(user_id, None)
    except Exception:
        pass

    return jsonify({"ok": True, "token": token, "token_type": "Bearer"}), 200


@api_bp.post("/predict")
def predict():
    # JWT required
    ok, user_id, err = _auth_user_id()
    if not ok:
        return jsonify({"error": "auth", "message": err}), 401

    payload = _get_payload()

    # Allow either systolic_bp/diastolic_bp or sys_bp/dia_bp
    if "systolic_bp" not in payload and "sys_bp" in payload:
        payload["systolic_bp"] = payload.get("sys_bp")
    if "diastolic_bp" not in payload and "dia_bp" in payload:
        payload["diastolic_bp"] = payload.get("dia_bp")

    required = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
    missing = _require_fields(payload, required)
    if missing:
        return jsonify({"error": "validation", "message": f"Missing: {', '.join(missing)}"}), 400

    # Build model input
    try:
        age = float(payload["age"])
        bmi = float(payload["bmi"])
        ex = _exercise_to_num(payload["exercise_level"])
        sbp = float(payload["systolic_bp"])
        dbp = float(payload["diastolic_bp"])
        hr = float(payload["heart_rate"])
    except Exception:
        return jsonify({"error": "validation", "message": "Invalid numeric input"}), 400

    model = _load_model()

    X = np.array([[age, bmi, ex, sbp, dbp, hr]], dtype=float)

    # Predict
    pred = int(model.predict(X)[0])
    risk_label = "High Risk" if pred == 1 else "Low Risk"

    prob_high = None
    try:
        prob_high = float(model.predict_proba(X)[0][1])  # class 1 = High Risk
    except Exception:
        prob_high = 1.0 if pred == 1 else 0.0

    # Intelligence Engine
    intel = _health_risk_intelligence(payload, prob_high=prob_high, risk_label=risk_label)

    # Optional save
    saved = False
    if Prediction is not None:
        try:
            row = Prediction()  # type: ignore[call-arg]
            # Fill fields safely if they exist on the model
            for k, v in {
                "user_id": user_id,
                "age": age,
                "bmi": bmi,
                "exercise_level": str(payload.get("exercise_level", "")),
                "systolic_bp": sbp,
                "diastolic_bp": dbp,
                "heart_rate": hr,
                "risk_label": risk_label,
                "probability": prob_high,
                "created_at": datetime.now(timezone.utc),
            }.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            db.session.add(row)
            db.session.commit()
            saved = True
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
            saved = False

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "risk_label": risk_label,
            "probability": prob_high,
            "saved": saved,
            **intel,
        }
    ), 200


@api_bp.post("/demo/predict")
def demo_predict():
    """
    Single-call demo endpoint: no token juggling.

    Enabled ONLY when DEMO_RETURN_OTP=1.
    Client sends:
      { "user_id":"123", "age":..., "bmi":..., "exercise_level":"Moderate", "systolic_bp":..., "diastolic_bp":..., "heart_rate":... }

    Server will:
      - issue OTP
      - verify OTP internally
      - run predict internally
      - return prediction + (optionally) token
    """
    if not _demo_return_otp_enabled():
        return jsonify({"error": "disabled", "message": "Demo endpoint disabled"}), 403

    payload = _get_payload()
    missing = _require_fields(payload, ["user_id"])
    if missing:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    user_id = str(payload["user_id"]).strip()
    issued = _issue_code(user_id)

    # internal verify
    ok, msg = _check_code(user_id, issued["code"])
    if not ok:
        return jsonify({"error": "auth", "message": msg}), 401

    token = _make_token(user_id)

    # Run same predict logic without forcing client auth
    # Reuse payload conversion
    if "systolic_bp" not in payload and "sys_bp" in payload:
        payload["systolic_bp"] = payload.get("sys_bp")
    if "diastolic_bp" not in payload and "dia_bp" in payload:
        payload["diastolic_bp"] = payload.get("dia_bp")

    required = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
    missing2 = _require_fields(payload, required)
    if missing2:
        return jsonify({"error": "validation", "message": f"Missing: {', '.join(missing2)}"}), 400

    try:
        age = float(payload["age"])
        bmi = float(payload["bmi"])
        ex = _exercise_to_num(payload["exercise_level"])
        sbp = float(payload["systolic_bp"])
        dbp = float(payload["diastolic_bp"])
        hr = float(payload["heart_rate"])
    except Exception:
        return jsonify({"error": "validation", "message": "Invalid numeric input"}), 400

    model = _load_model()
    X = np.array([[age, bmi, ex, sbp, dbp, hr]], dtype=float)
    pred = int(model.predict(X)[0])
    risk_label = "High Risk" if pred == 1 else "Low Risk"

    try:
        prob_high = float(model.predict_proba(X)[0][1])
    except Exception:
        prob_high = 1.0 if pred == 1 else 0.0

    intel = _health_risk_intelligence(payload, prob_high=prob_high, risk_label=risk_label)

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "risk_label": risk_label,
            "probability": prob_high,
            "token_type": "Bearer",
            "token": token,  # optional to show; client can ignore
            **intel,
        }
    ), 200
