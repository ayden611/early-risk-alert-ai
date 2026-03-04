# era/api/routes.py
import os
import time
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
import numpy as np
from flask import Blueprint, request, jsonify, current_app

from era.extensions import db

try:
    # Optional DB model (we will use it if it exists)
    from era.models import Prediction  # type: ignore
except Exception:
    Prediction = None  # type: ignore

try:
    from sqlalchemy import text  # type: ignore
except Exception:
    text = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# In-memory OTP store (demo / simple auth)
# NOTE: resets on deploy/restart (fine for demo)
# ----------------------------
_OTP_STORE: Dict[str, Dict[str, Any]] = {}  # user_id -> {"code": str, "exp": float}


# ----------------------------
# Config helpers
# ----------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _jwt_secret() -> str:
    # Prefer explicit JWT_SECRET, then Flask SECRET_KEY, then SECRET_KEY env, then fallback
    return (
        os.getenv("JWT_SECRET")
        or (current_app.config.get("JWT_SECRET") if current_app else None)
        or current_app.config.get("SECRET_KEY", None)
        if current_app
        else None
        or os.getenv("SECRET_KEY")
        or "dev-jwt-secret-change-me"
    )


JWT_ACCESS_MINUTES = _env_int("JWT_ACCESS_MINUTES", 60)
MAGIC_CODE_TTL = _env_int("MAGIC_CODE_TTL", 600)  # seconds
DEMO_RETURN_OTP = _env_bool("DEMO_RETURN_OTP", False)

MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")


# ----------------------------
# DB: lightweight table creation (no migrations needed)
# ----------------------------
def _ensure_tables() -> None:
    """
    Creates a simple health_readings table if it does not exist.
    Works for Postgres + SQLite.
    Safe to call repeatedly.
    """
    if text is None:
        return

    try:
        dialect = db.engine.dialect.name  # type: ignore
    except Exception:
        return

    if dialect == "postgresql":
        sql = """
        CREATE TABLE IF NOT EXISTS health_readings (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            age DOUBLE PRECISION,
            bmi DOUBLE PRECISION,
            exercise_level TEXT,
            systolic_bp DOUBLE PRECISION,
            diastolic_bp DOUBLE PRECISION,
            heart_rate DOUBLE PRECISION,
            risk_label TEXT,
            probability DOUBLE PRECISION,
            risk_score INTEGER,
            factors_json TEXT,
            recommendations_json TEXT,
            insights_json TEXT,
            raw_transcript TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_health_readings_user_time
            ON health_readings (user_id, created_at DESC);
        """
    else:
        # SQLite / others
        sql = """
        CREATE TABLE IF NOT EXISTS health_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            age REAL,
            bmi REAL,
            exercise_level TEXT,
            systolic_bp REAL,
            diastolic_bp REAL,
            heart_rate REAL,
            risk_label TEXT,
            probability REAL,
            risk_score INTEGER,
            factors_json TEXT,
            recommendations_json TEXT,
            insights_json TEXT,
            raw_transcript TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_health_readings_user_time
            ON health_readings (user_id, created_at);
        """
    try:
        db.session.execute(text(sql))
        db.session.commit()
    except Exception:
        db.session.rollback()


@api_bp.before_app_request
def _boot_once() -> None:
    # Ensure DB table exists for history/trends (safe to run often)
    _ensure_tables()


# ----------------------------
# Model loading
# ----------------------------
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Optional: allow disabling ML model in dev
    if not os.path.exists(MODEL_PATH):
        _MODEL = None
        return _MODEL

    try:
        import joblib  # type: ignore

        _MODEL = joblib.load(MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def _exercise_to_num(exercise_level: str) -> float:
    v = (exercise_level or "").strip().lower()
    if v in ("low", "sedentary", "none"):
        return 0.0
    if v in ("moderate", "medium", "avg", "average"):
        return 1.0
    if v in ("high", "active", "heavy"):
        return 2.0
    # If user sends numeric string, accept it
    try:
        return float(v)
    except Exception:
        return 1.0


# ----------------------------
# JWT helpers
# ----------------------------
def _make_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_ACCESS_MINUTES)).timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _get_bearer_token() -> Optional[str]:
    h = request.headers.get("Authorization", "")
    if not h:
        return None
    parts = h.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _require_user() -> Tuple[Optional[str], Optional[Tuple[Dict[str, Any], int]]]:
    token = _get_bearer_token()
    if not token:
        return None, ({"error": "auth", "message": "Missing Bearer token"}, 401)
    try:
        decoded = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
        user_id = str(decoded.get("sub") or "")
        if not user_id:
            return None, ({"error": "auth", "message": "Invalid token (no subject)"}, 401)
        return user_id, None
    except jwt.ExpiredSignatureError:
        return None, ({"error": "auth", "message": "Token expired"}, 401)
    except Exception:
        return None, ({"error": "auth", "message": "Invalid token"}, 401)


# ----------------------------
# Parsing / Validation
# ----------------------------
def _json() -> Dict[str, Any]:
    try:
        return request.get_json(force=True) or {}
    except Exception:
        return {}


def _missing(payload: dict, required: List[str]) -> List[str]:
    miss = []
    for k in required:
        if payload.get(k) in (None, ""):
            miss.append(k)
    return miss


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _canonicalize_readings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts flexible keys; returns canonical readings.
    """
    age = _safe_float(payload.get("age"))
    bmi = _safe_float(payload.get("bmi"))

    # exercise
    ex_raw = payload.get("exercise_level", payload.get("exercise"))
    exercise_level = str(ex_raw) if ex_raw is not None else ""

    # BP keys
    sbp = _safe_float(payload.get("systolic_bp", payload.get("sys_bp")))
    dbp = _safe_float(payload.get("diastolic_bp", payload.get("dia_bp")))
    hr = _safe_float(payload.get("heart_rate", payload.get("hr")))

    out = {
        "age": age,
        "bmi": bmi,
        "exercise_level": exercise_level,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "heart_rate": hr,
    }
    return out


# ----------------------------
# Voice transcript parsing (simple but effective)
# ----------------------------
_RE_BP_SLASH = re.compile(r"(?P<s>\d{2,3})\s*/\s*(?P<d>\d{2,3})")
_RE_BP_OVER = re.compile(r"(?P<s>\d{2,3})\s*(?:over)\s*(?P<d>\d{2,3})", re.IGNORECASE)
_RE_HR = re.compile(r"(?:heart\s*rate|hr)\s*(?:is\s*)?(?P<hr>\d{2,3})", re.IGNORECASE)
_RE_BMI = re.compile(r"(?:bmi)\s*(?:is\s*)?(?P<bmi>\d{1,2}(?:\.\d+)?)", re.IGNORECASE)
_RE_AGE = re.compile(r"(?:age)\s*(?:is\s*)?(?P<age>\d{1,3})", re.IGNORECASE)
_RE_EX = re.compile(r"(?:exercise|activity)\s*(?:level\s*)?(?:is\s*)?(?P<ex>low|moderate|high)", re.IGNORECASE)


def _parse_transcript(transcript: str) -> Dict[str, Any]:
    t = (transcript or "").strip()
    out: Dict[str, Any] = {}

    m = _RE_BP_SLASH.search(t) or _RE_BP_OVER.search(t)
    if m:
        out["systolic_bp"] = _safe_float(m.group("s"))
        out["diastolic_bp"] = _safe_float(m.group("d"))

    m = _RE_HR.search(t)
    if m:
        out["heart_rate"] = _safe_float(m.group("hr"))

    m = _RE_BMI.search(t)
    if m:
        out["bmi"] = _safe_float(m.group("bmi"))

    m = _RE_AGE.search(t)
    if m:
        out["age"] = _safe_float(m.group("age"))

    m = _RE_EX.search(t)
    if m:
        out["exercise_level"] = m.group("ex")

    return out


# ----------------------------
# Intelligence Layer (score + factors + recommendations + insights)
# ----------------------------
def _risk_score(prob_high: float, readings: Dict[str, Any]) -> int:
    """
    Score 0-100 using model probability + small nudges from readings.
    Keeps the model as the primary signal.
    """
    score = int(round(float(prob_high) * 100))

    age = readings.get("age") or 0
    bmi = readings.get("bmi") or 0
    sbp = readings.get("systolic_bp") or 0
    dbp = readings.get("diastolic_bp") or 0
    hr = readings.get("heart_rate") or 0
    ex = str(readings.get("exercise_level") or "").lower().strip()

    # gentle nudges
    if age >= 55:
        score += 5
    if bmi >= 30:
        score += 7
    elif bmi >= 25:
        score += 3
    if sbp >= 140 or dbp >= 90:
        score += 8
    elif sbp >= 130 or dbp >= 80:
        score += 4
    if hr >= 90:
        score += 3
    if ex in ("low", "sedentary", "none"):
        score += 3

    return max(0, min(100, score))


def _factors(readings: Dict[str, Any]) -> List[str]:
    facts = []
    if readings.get("age") is not None:
        facts.append(f"Age: {int(readings['age'])}")
    if readings.get("bmi") is not None:
        facts.append(f"BMI: {readings['bmi']:.1f}")
    if readings.get("systolic_bp") is not None and readings.get("diastolic_bp") is not None:
        facts.append(f"Blood pressure: {int(readings['systolic_bp'])}/{int(readings['diastolic_bp'])}")
    if readings.get("heart_rate") is not None:
        facts.append(f"Heart rate: {int(readings['heart_rate'])}")
    ex = (readings.get("exercise_level") or "").strip()
    if ex:
        facts.append(f"Exercise level: {ex}")
    return facts


def _recommendations(readings: Dict[str, Any], risk_score: int) -> List[str]:
    recs: List[str] = []

    sbp = readings.get("systolic_bp")
    dbp = readings.get("diastolic_bp")
    bmi = readings.get("bmi")
    ex = str(readings.get("exercise_level") or "").lower().strip()

    if sbp is not None and dbp is not None and (sbp >= 130 or dbp >= 80):
        recs.append("If you consistently see elevated blood pressure, consider lifestyle changes and periodic monitoring.")
    if bmi is not None and bmi >= 25:
        recs.append("Small changes in diet and activity can improve BMI over time.")
    if ex in ("low", "sedentary", "none"):
        recs.append("Aim for light-to-moderate activity most days (consistency matters more than intensity).")

    if risk_score >= 70:
        recs.append("Consider discussing these numbers with a healthcare professional, especially if symptoms are present.")
    recs.append("This is not a medical diagnosis. If you have symptoms or concerns, seek professional medical advice.")
    return recs


def _trend_engine(recent: List[Dict[str, Any]], current: Dict[str, Any], current_score: int) -> Dict[str, Any]:
    """
    Compares current values vs the most recent prior reading (if any),
    and gives a simple direction signal.
    """
    if not recent:
        return {"has_history": False}

    prev = recent[0]  # most recent previous
    def delta(key: str) -> Optional[float]:
        a = _safe_float(current.get(key))
        b = _safe_float(prev.get(key))
        if a is None or b is None:
            return None
        return a - b

    d_sbp = delta("systolic_bp")
    d_dbp = delta("diastolic_bp")
    d_bmi = delta("bmi")
    d_hr = delta("heart_rate")
    d_score = _safe_float(current_score) - _safe_float(prev.get("risk_score") or 0)

    def dir_label(d: Optional[float], good_when_down: bool = True) -> str:
        if d is None:
            return "unknown"
        if abs(d) < 0.0001:
            return "stable"
        if d < 0:
            return "improving" if good_when_down else "worsening"
        return "worsening" if good_when_down else "improving"

    return {
        "has_history": True,
        "vs_last": {
            "systolic_bp_delta": d_sbp,
            "diastolic_bp_delta": d_dbp,
            "bmi_delta": d_bmi,
            "heart_rate_delta": d_hr,
            "risk_score_delta": d_score,
            "bp_trend": dir_label((d_sbp or 0) + (d_dbp or 0), good_when_down=True),
            "bmi_trend": dir_label(d_bmi, good_when_down=True),
            "risk_trend": dir_label(d_score, good_when_down=True),
        },
    }


def _personal_insights(readings: Dict[str, Any], risk_score: int, trends: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    sbp = readings.get("systolic_bp")
    dbp = readings.get("diastolic_bp")
    bmi = readings.get("bmi")
    ex = str(readings.get("exercise_level") or "").lower().strip()

    if trends.get("has_history"):
        vs = trends.get("vs_last", {})
        if vs.get("risk_trend") == "improving":
            out.append("Your overall risk score improved compared to your last entry.")
        elif vs.get("risk_trend") == "worsening":
            out.append("Your overall risk score increased compared to your last entry.")

        if vs.get("bp_trend") == "improving":
            out.append("Blood pressure looks improved vs your last reading.")
        elif vs.get("bp_trend") == "worsening":
            out.append("Blood pressure looks higher vs your last reading.")

    if sbp is not None and dbp is not None and (sbp >= 140 or dbp >= 90):
        out.append("Your blood pressure is in a higher range; consider re-checking and tracking over several days.")
    elif sbp is not None and dbp is not None and (sbp >= 130 or dbp >= 80):
        out.append("Your blood pressure is slightly elevated; small habit changes can have a big effect over time.")

    if bmi is not None and bmi >= 30:
        out.append("BMI is in a higher range; gradual, sustainable changes tend to work best.")
    elif bmi is not None and bmi >= 25:
        out.append("BMI is slightly elevated; adding daily walking and modest diet adjustments can help.")

    if ex in ("low", "sedentary", "none"):
        out.append("Try a simple goal: 20–30 minutes of walking most days.")

    # Keep it short: 2–4 insights
    return out[:4]


# ----------------------------
# DB read/write for history
# ----------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_reading(
    user_id: str,
    readings: Dict[str, Any],
    risk_label: str,
    prob_high: float,
    risk_score: int,
    factors: List[str],
    recommendations: List[str],
    insights: List[str],
    raw_transcript: Optional[str] = None,
) -> bool:
    """
    Saves into health_readings table (created automatically) for consistent history/trends.
    If Prediction model exists, we ALSO try to save there (best effort).
    """
    saved = False

    # Save to our stable table
    if text is not None:
        try:
            dialect = db.engine.dialect.name  # type: ignore
            if dialect == "postgresql":
                created_at_val = None  # use default NOW()
                sql = """
                INSERT INTO health_readings
                    (user_id, created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                     risk_label, probability, risk_score, factors_json, recommendations_json, insights_json, raw_transcript)
                VALUES
                    (:user_id, NOW(), :age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate,
                     :risk_label, :probability, :risk_score, :factors_json, :recommendations_json, :insights_json, :raw_transcript)
                """
            else:
                sql = """
                INSERT INTO health_readings
                    (user_id, created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                     risk_label, probability, risk_score, factors_json, recommendations_json, insights_json, raw_transcript)
                VALUES
                    (:user_id, :created_at, :age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate,
                     :risk_label, :probability, :risk_score, :factors_json, :recommendations_json, :insights_json, :raw_transcript)
                """
                created_at_val = _now_iso()

            db.session.execute(
                text(sql),
                {
                    "user_id": user_id,
                    "created_at": created_at_val,
                    "age": readings.get("age"),
                    "bmi": readings.get("bmi"),
                    "exercise_level": readings.get("exercise_level"),
                    "systolic_bp": readings.get("systolic_bp"),
                    "diastolic_bp": readings.get("diastolic_bp"),
                    "heart_rate": readings.get("heart_rate"),
                    "risk_label": risk_label,
                    "probability": float(prob_high),
                    "risk_score": int(risk_score),
                    "factors_json": json.dumps(factors),
                    "recommendations_json": json.dumps(recommendations),
                    "insights_json": json.dumps(insights),
                    "raw_transcript": raw_transcript,
                },
            )
            db.session.commit()
            saved = True
        except Exception:
            db.session.rollback()

    # Best-effort save to optional Prediction model (if your schema supports it)
    if Prediction is not None:
        try:
            row = Prediction()  # type: ignore
            # These attribute sets are best-effort (won't crash if missing)
            for k, v in readings.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            if hasattr(row, "user_id"):
                setattr(row, "user_id", user_id)
            if hasattr(row, "risk_label"):
                setattr(row, "risk_label", risk_label)
            if hasattr(row, "probability"):
                setattr(row, "probability", float(prob_high))
            if hasattr(row, "risk_score"):
                setattr(row, "risk_score", int(risk_score))
            if hasattr(row, "created_at"):
                setattr(row, "created_at", datetime.now(timezone.utc))

            db.session.add(row)
            db.session.commit()
            saved = True
        except Exception:
            db.session.rollback()

    return saved


def _fetch_recent(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    if text is None:
        return []
    try:
        dialect = db.engine.dialect.name  # type: ignore
        if dialect == "postgresql":
            sql = """
            SELECT user_id, created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                   risk_label, probability, risk_score
            FROM health_readings
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
        else:
            sql = """
            SELECT user_id, created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                   risk_label, probability, risk_score
            FROM health_readings
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
        rows = db.session.execute(text(sql), {"user_id": user_id, "limit": int(limit)}).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/api/v1")
def root():
    return {"ok": True, "service": "early-risk-alert-mobile-api"}, 200


@api_bp.get("/api/v1/health")
def health():
    db_ok = True
    err = None
    try:
        if text is not None:
            db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        err = str(e)
    model = _load_model()
    return {
        "ok": True,
        "db_ok": db_ok,
        "db_error": err,
        "model_loaded": model is not None,
        "demo_return_otp": DEMO_RETURN_OTP,
    }, (200 if db_ok else 503)


@api_bp.post("/api/v1/auth/login")
def auth_login():
    payload = _json()
    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        return {"error": "validation", "message": "'user_id' is required"}, 400

    code = f"{secrets.randbelow(1000000):06d}"
    exp = time.time() + float(MAGIC_CODE_TTL)
    _OTP_STORE[user_id] = {"code": code, "exp": exp}

    resp = {"ok": True, "sent": True, "expires_in": MAGIC_CODE_TTL}
    if DEMO_RETURN_OTP:
        resp["code"] = code
    return resp, 200


@api_bp.post("/api/v1/auth/verify")
def auth_verify():
    payload = _json()
    user_id = str(payload.get("user_id") or "").strip()
    code = str(payload.get("code") or "").strip()

    if not user_id or not code:
        return {"error": "validation", "message": "'user_id' and 'code' are required"}, 400

    rec = _OTP_STORE.get(user_id)
    if not rec:
        return {"error": "auth", "message": "No code requested. Call /auth/login first."}, 401

    if time.time() > float(rec.get("exp", 0)):
        return {"error": "auth", "message": "Code expired"}, 401

    if str(rec.get("code")) != code:
        return {"error": "auth", "message": "Invalid code"}, 401

    token = _make_access_token(user_id)
    return {"ok": True, "token": token, "token_type": "Bearer"}, 200


@api_bp.post("/api/v1/predict")
def predict():
    user_id, err = _require_user()
    if err:
        return err[0], err[1]

    payload = _json()
    readings = _canonicalize_readings(payload)

    required = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
    missing = [k for k in required if readings.get(k) in (None, "", [])]
    if missing:
        return {"error": "validation", "message": f"Missing: {', '.join(missing)}"}, 400

    model = _load_model()
    if model is None:
        return {"error": "server", "message": f"Model not found at {MODEL_PATH}"}, 503

    age = float(readings["age"])
    bmi = float(readings["bmi"])
    ex = float(_exercise_to_num(str(readings["exercise_level"])))
    sbp = float(readings["systolic_bp"])
    dbp = float(readings["diastolic_bp"])
    hr = float(readings["heart_rate"])

    X = np.array([[age, bmi, ex, sbp, dbp, hr]], dtype=float)
    pred = int(model.predict(X)[0])
    risk_label = "High Risk" if pred == 1 else "Low Risk"
    try:
        prob_high = float(model.predict_proba(X)[0][1])
    except Exception:
        prob_high = 1.0 if pred == 1 else 0.0

    return {
        "ok": True,
        "user_id": user_id,
        "risk_label": risk_label,
        "probability": prob_high,
        "saved": False,
    }, 200


@api_bp.get("/api/v1/history")
def history():
    user_id, err = _require_user()
    if err:
        return err[0], err[1]

    limit = _env_int("HISTORY_LIMIT_DEFAULT", 20)
    try:
        limit = int(request.args.get("limit", limit))
    except Exception:
        pass
    limit = max(1, min(200, limit))

    rows = _fetch_recent(user_id, limit=limit)
    return {"ok": True, "user_id": user_id, "count": len(rows), "items": rows}, 200


@api_bp.post("/api/v1/voice/parse")
def voice_parse():
    payload = _json()
    transcript = str(payload.get("transcript") or "")
    if not transcript.strip():
        return {"error": "validation", "message": "'transcript' is required"}, 400

    parsed = _parse_transcript(transcript)
    canonical = _canonicalize_readings(parsed)
    missing = [k for k in ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"] if canonical.get(k) in (None, "", [])]
    return {
        "ok": True,
        "transcript": transcript,
        "parsed": canonical,
        "missing": missing,
    }, 200


@api_bp.post("/api/v1/insights/predict")
def insights_predict():
    """
    One-call endpoint:
    - Accepts either readings JSON or { "transcript": "..." }
    - Predicts
    - Builds intelligence (score, factors, recommendations, insights)
    - Saves to history
    - Returns trends
    """
    user_id, err = _require_user()
    if err:
        return err[0], err[1]

    payload = _json()

    raw_transcript = None
    if payload.get("transcript"):
        raw_transcript = str(payload.get("transcript") or "")
        parsed = _parse_transcript(raw_transcript)
        # Merge: explicit JSON fields override transcript-derived fields
        merged = dict(parsed)
        merged.update({k: v for k, v in payload.items() if k != "transcript"})
        readings = _canonicalize_readings(merged)
    else:
        readings = _canonicalize_readings(payload)

    required = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
    missing = [k for k in required if readings.get(k) in (None, "", [])]
    if missing:
        return {"error": "validation", "message": f"Missing: {', '.join(missing)}"}, 400

    model = _load_model()
    if model is None:
        return {"error": "server", "message": f"Model not found at {MODEL_PATH}"}, 503

    age = float(readings["age"])
    bmi = float(readings["bmi"])
    ex = float(_exercise_to_num(str(readings["exercise_level"])))
    sbp = float(readings["systolic_bp"])
    dbp = float(readings["diastolic_bp"])
    hr = float(readings["heart_rate"])

    X = np.array([[age, bmi, ex, sbp, dbp, hr]], dtype=float)
    pred = int(model.predict(X)[0])
    risk_label = "High Risk" if pred == 1 else "Low Risk"
    try:
        prob_high = float(model.predict_proba(X)[0][1])
    except Exception:
        prob_high = 1.0 if pred == 1 else 0.0

    score = _risk_score(prob_high, readings)
    factors = _factors(readings)
    recs = _recommendations(readings, score)

    # trends need prior history (exclude current; so fetch recent first)
    recent = _fetch_recent(user_id, limit=10)
    trends = _trend_engine(recent, readings, score)
    insights = _personal_insights(readings, score, trends)

    saved = _save_reading(
        user_id=user_id,
        readings=readings,
        risk_label=risk_label,
        prob_high=prob_high,
        risk_score=score,
        factors=factors,
        recommendations=recs,
        insights=insights,
        raw_transcript=raw_transcript,
    )

    return {
        "ok": True,
        "user_id": user_id,
        "risk_label": risk_label,
        "probability": prob_high,
        "risk_score": score,
        "risk_factors": factors,
        "recommendations": recs,
        "personal_insights": insights,
        "trends": trends,
        "saved": saved,
    }, 200


# Demo shortcut (optional): no JWT required for quick demos
# Enable with DEMO_ALLOW_OPEN_PREDICT=1 and use /api/v1/demo/predict
DEMO_ALLOW_OPEN_PREDICT = _env_bool("DEMO_ALLOW_OPEN_PREDICT", False)


@api_bp.post("/api/v1/demo/predict")
def demo_predict():
    if not DEMO_ALLOW_OPEN_PREDICT:
        return {"error": "auth", "message": "Demo endpoint disabled"}, 403

    payload = _json()
    user_id = str(payload.get("user_id") or "demo").strip() or "demo"

    if payload.get("transcript"):
        raw_transcript = str(payload.get("transcript") or "")
        parsed = _parse_transcript(raw_transcript)
        merged = dict(parsed)
        merged.update({k: v for k, v in payload.items() if k != "transcript"})
        readings = _canonicalize_readings(merged)
    else:
        readings = _canonicalize_readings(payload)
        raw_transcript = None

    required = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
    missing = [k for k in required if readings.get(k) in (None, "", [])]
    if missing:
        return {"error": "validation", "message": f"Missing: {', '.join(missing)}"}, 400

    model = _load_model()
    if model is None:
        return {"error": "server", "message": f"Model not found at {MODEL_PATH}"}, 503

    age = float(readings["age"])
    bmi = float(readings["bmi"])
    ex = float(_exercise_to_num(str(readings["exercise_level"])))
    sbp = float(readings["systolic_bp"])
    dbp = float(readings["diastolic_bp"])
    hr = float(readings["heart_rate"])

    X = np.array([[age, bmi, ex, sbp, dbp, hr]], dtype=float)
    pred = int(model.predict(X)[0])
    risk_label = "High Risk" if pred == 1 else "Low Risk"
    try:
        prob_high = float(model.predict_proba(X)[0][1])
    except Exception:
        prob_high = 1.0 if pred == 1 else 0.0

    score = _risk_score(prob_high, readings)
    factors = _factors(readings)
    recs = _recommendations(readings, score)

    recent = _fetch_recent(user_id, limit=10)
    trends = _trend_engine(recent, readings, score)
    insights = _personal_insights(readings, score, trends)

    saved = _save_reading(
        user_id=user_id,
        readings=readings,
        risk_label=risk_label,
        prob_high=prob_high,
        risk_score=score,
        factors=factors,
        recommendations=recs,
        insights=insights,
        raw_transcript=raw_transcript,
    )

    return {
        "ok": True,
        "user_id": user_id,
        "risk_label": risk_label,
        "probability": prob_high,
        "risk_score": score,
        "risk_factors": factors,
        "recommendations": recs,
        "personal_insights": insights,
        "trends": trends,
        "saved": saved,
        "model_risk_label": risk_label,  # compatibility
    }, 200
