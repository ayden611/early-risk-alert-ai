# era/api/routes.py
import os
import time
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
import numpy as np
import joblib
from flask import Blueprint, request, jsonify, current_app
import re
from flask import Response, stream_with_context
from sqlalchemy import text


_BP_RE = re.compile(r"\b(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.IGNORECASE)
_HR_RE = re.compile(r"(?:heart rate|pulse)\s*(?:is|was)?\s*(\d{2,3})", re.IGNORECASE)

def _parse_voice_transcript(transcript: str):
    out = {}

    m = _BP_RE.search(transcript)
    if m:
        out["systolic_bp"] = float(m.group(1))
        out["diastolic_bp"] = float(m.group(2))

    m = _HR_RE.search(transcript)
    if m:
        out["heart_rate"] = float(m.group(1))

    return out

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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _jwt_secret() -> str:
    # Prefer explicit JWT_SECRET, fallback to Flask SECRET_KEY, fallback dev default
    return (
        os.getenv("JWT_SECRET")
        or (current_app.config.get("JWT_SECRET") if current_app else None)
        or os.getenv("SECRET_KEY")
        or (current_app.config.get("SECRET_KEY") if current_app else None)
        or "dev-jwt-secret-change-me"
    )


def _jwt_access_minutes() -> int:
    try:
        return int(os.getenv("JWT_ACCESS_MINUTES", "60"))
    except Exception:
        return 60


def _issue_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(_now_utc().timestamp()),
        "exp": int((_now_utc() + timedelta(minutes=_jwt_access_minutes())).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _require_bearer_token() -> Tuple[Optional[str], Optional[dict], Optional[Tuple[dict, int]]]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, None, ({"error": "auth", "message": "Missing Bearer token"}, 401)
    token = auth.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
        user_id = str(claims.get("sub") or "")
        if not user_id:
            return None, None, ({"error": "auth", "message": "Invalid token subject"}, 401)
        return user_id, claims, None
    except Exception:
        return None, None, ({"error": "auth", "message": "Invalid token"}, 401)


# ----------------------------
# OTP demo store (in-memory)
# ----------------------------
_OTP_STORE: Dict[str, Dict[str, Any]] = {}
# shape: { user_id: { "code": "123456", "expires_at": epoch_seconds } }

def _otp_ttl_seconds() -> int:
    try:
        return int(os.getenv("MAGIC_CODE_TTL", "600"))
    except Exception:
        return 600


def _set_otp(user_id: str, code: str) -> None:
    _OTP_STORE[user_id] = {"code": code, "expires_at": time.time() + _otp_ttl_seconds()}


def _get_otp(user_id: str) -> Optional[dict]:
    rec = _OTP_STORE.get(user_id)
    if not rec:
        return None
    if time.time() > float(rec.get("expires_at", 0)):
        _OTP_STORE.pop(user_id, None)
        return None
    return rec


# ----------------------------
# Model loading (cached)
# ----------------------------
_MODEL = None
_MODEL_META: Dict[str, Any] = {}

def _model_path() -> str:
    # In repo root usually: demo_model.pkl
    # You can set MODEL_PATH in Render if needed
    return os.getenv("MODEL_PATH", "demo_model.pkl")


def _load_model():
    global _MODEL, _MODEL_META
    if _MODEL is not None:
        return _MODEL

    path = _model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    _MODEL = joblib.load(path)

    meta_path = os.getenv("MODEL_META_PATH", "model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                _MODEL_META = json.load(f)
        except Exception:
            _MODEL_META = {}
    return _MODEL


# ----------------------------
# Input parsing
# ----------------------------
def _exercise_to_num(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    mapping = {"low": 0.0, "moderate": 1.0, "medium": 1.0, "high": 2.0}
    return float(mapping.get(s, 0.0))


def _first(payload: dict, keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return default


def _validate_payload(payload: dict) -> Optional[Tuple[dict, int]]:
    required = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]
    missing = [k for k in required if k not in payload]
    if missing:
        return {"error": "validation", "message": f"Missing: {', '.join(missing)}"}, 400
    return None


def _coerce_inputs(payload: dict) -> Tuple[float, float, float, float, float, float]:
    age = float(payload["age"])
    bmi = float(payload["bmi"])
    ex = _exercise_to_num(payload["exercise_level"])
    sbp = float(_first(payload, ["systolic_bp", "sys_bp"], 0))
    dbp = float(_first(payload, ["diastolic_bp", "dia_bp"], 0))
    hr = float(payload["heart_rate"])
    return age, bmi, ex, sbp, dbp, hr


# ----------------------------
# “Health Risk Intelligence Engine”
# ----------------------------
def _health_risk_intelligence(payload: dict, prob_high: float, risk_label: str, trend: Optional[dict]) -> dict:
    age = float(payload.get("age", 0) or 0)
    bmi = float(payload.get("bmi", 0) or 0)
    exercise = str(payload.get("exercise_level", "") or "").strip().lower()
    sbp = float(_first(payload, ["systolic_bp", "sys_bp"], 0) or 0)
    dbp = float(_first(payload, ["diastolic_bp", "dia_bp"], 0) or 0)
    hr = float(payload.get("heart_rate", 0) or 0)

    score = int(round(prob_high * 100))
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
    score = max(0, min(100, score))

    factors: List[str] = []
    if age >= 55:
        factors.append(f"Age {age:g} (higher baseline risk)")
    if bmi >= 25:
        factors.append(f"BMI {bmi:g} (overweight range)")
    if bmi >= 30:
        factors.append(f"BMI {bmi:g} (obesity range)")
    if sbp >= 130 or dbp >= 80:
        factors.append(f"Blood pressure {sbp:g}/{dbp:g} (elevated range)")
    if sbp >= 140 or dbp >= 90:
        factors.append(f"Blood pressure {sbp:g}/{dbp:g} (high range)")
    if hr >= 90:
        factors.append(f"Heart rate {hr:g} (elevated)")
    if exercise in ("low", ""):
        factors.append("Exercise level (low/unknown)")

    recs: List[str] = []
    recs.append("This is not medical advice. If you have symptoms or concerns, seek professional care.")
    if sbp >= 130 or dbp >= 80:
        recs.append("If you consistently see elevated blood pressure, consider lifestyle changes and periodic monitoring.")
    if bmi >= 25:
        recs.append("Small changes in diet and activity can improve BMI over time.")
    if exercise in ("low", ""):
        recs.append("Aim for consistent activity; consistency matters more than intensity at first.")
    recs.append("Track readings over time — trends are often more meaningful than one measurement.")

    out = {
        "risk_score": score,
        "risk_factors": factors,
        "recommendations": recs,
    }
    if trend:
        out["trend"] = trend
    return out


def _trend_from_history(user_id: str, limit: int = 14) -> Optional[dict]:
    if Prediction is None:
        return None
    try:
        rows = (
            Prediction.query.filter_by(user_id=user_id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return None

    if not rows or len(rows) < 3:
        return None

    probs: List[float] = []
    for r in reversed(rows):
        p = getattr(r, "probability", None)
        if p is None:
            lbl = (getattr(r, "risk_label", "") or "").lower()
            p = 1.0 if "high" in lbl else 0.0
        try:
            probs.append(float(p))
        except Exception:
            probs.append(0.0)

    first = float(np.mean(probs[: max(1, len(probs)//2)]))
    last = float(np.mean(probs[len(probs)//2 :]))
    delta = last - first

    direction = "up" if delta > 0.03 else "down" if delta < -0.03 else "flat"
    return {
        "window": len(probs),
        "avg_prob_early": round(first, 4),
        "avg_prob_recent": round(last, 4),
        "direction": direction,
    }


def _try_save_prediction(user_id: str, payload: dict, risk_label: str, probability: float) -> bool:
    if Prediction is None:
        return False
    try:
        rec = Prediction()
        for k, v in {
            "user_id": user_id,
            "age": payload.get("age"),
            "bmi": payload.get("bmi"),
            "exercise_level": payload.get("exercise_level"),
            "systolic_bp": _first(payload, ["systolic_bp", "sys_bp"]),
            "diastolic_bp": _first(payload, ["diastolic_bp", "dia_bp"]),
            "heart_rate": payload.get("heart_rate"),
            "risk_label": risk_label,
            "probability": probability,
            "created_at": _now_utc().replace(tzinfo=None),
        }.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        db.session.add(rec)
        db.session.commit()
        return True
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
        return False


@api_bp.get("/")
def api_root():
    return jsonify({"ok": True, "service": "early-risk-alert-mobile-api"})


@api_bp.post("/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    demo_return = _env_bool("DEMO_RETURN_OTP", False)
    code = f"{secrets.randbelow(1000000):06d}"
    _set_otp(user_id, code)

    resp = {"ok": True, "expires_in": _otp_ttl_seconds(), "sent": True}
    if demo_return:
        resp["code"] = code
    return jsonify(resp)


@api_bp.post("/auth/verify")
def auth_verify():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id") or "").strip()
    code = str(data.get("code") or "").strip()
    if not user_id or not code:
        return jsonify({"error": "validation", "message": "user_id and code are required"}), 400

    rec = _get_otp(user_id)
    if not rec or rec.get("code") != code:
        return jsonify({"error": "auth", "message": "Invalid code"}), 401

    token = _issue_token(user_id)
    return jsonify({"ok": True, "token": token, "token_type": "Bearer"})


@api_bp.post("/predict")
def predict():
    user_id, _claims, err = _require_bearer_token()
    if err:
        return jsonify(err[0]), err[1]

    payload = request.get_json(silent=True) or {}
    bad = _validate_payload(payload)
    if bad:
        return jsonify(bad[0]), bad[1]

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(payload)
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

    saved = _try_save_prediction(user_id, payload, risk_label, prob_high)
    trend = _trend_from_history(user_id)
    intel = _health_risk_intelligence(payload, prob_high=prob_high, risk_label=risk_label, trend=trend)

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "risk_label": risk_label,
            "probability": prob_high,
            "saved": saved,
            **intel,
        }
    )


@api_bp.get("/history")
def history():
    user_id, _claims, err = _require_bearer_token()
    if err:
        return jsonify(err[0]), err[1]
    if Prediction is None:
        return jsonify({"ok": True, "items": [], "note": "Prediction model not configured"}), 200

    try:
        limit = int(request.args.get("limit", "50"))
        limit = max(1, min(200, limit))
    except Exception:
        limit = 50

    try:
        rows = (
            Prediction.query.filter_by(user_id=user_id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for r in rows:
            items.append(
                {
                    "created_at": getattr(r, "created_at", None).isoformat() if getattr(r, "created_at", None) else None,
                    "risk_label": getattr(r, "risk_label", None),
                    "probability": getattr(r, "probability", None),
                    "age": getattr(r, "age", None),
                    "bmi": getattr(r, "bmi", None),
                    "exercise_level": getattr(r, "exercise_level", None),
                    "systolic_bp": getattr(r, "systolic_bp", None),
                    "diastolic_bp": getattr(r, "diastolic_bp", None),
                    "heart_rate": getattr(r, "heart_rate", None),
                }
            )
        return jsonify({"ok": True, "user_id": user_id, "items": items}), 200
    except Exception as e:
        return jsonify({"error": "db", "message": str(e)}), 500


@api_bp.post("/demo/predict")
def demo_predict():
    if not _env_bool("DEMO_RETURN_OTP", False):
        return jsonify({"error": "disabled", "message": "demo endpoint disabled"}), 403

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    bad = _validate_payload(payload)
    if bad:
        return jsonify(bad[0]), bad[1]

    token = _issue_token(user_id)

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(payload)
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

    saved = _try_save_prediction(user_id, payload, risk_label, prob_high)
    trend = _trend_from_history(user_id)
    intel = _health_risk_intelligence(payload, prob_high=prob_high, risk_label=risk_label, trend=trend)

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "risk_label": risk_label,
            "probability": prob_high,
            "saved": saved,
            "token_type": "Bearer",
            "token": token,
            **intel,
        }
    )

@api_bp.post("/voice/predict")
def voice_predict():
    payload = request.get_json(silent=True) or {}
    transcript = payload.get("transcript")

    if not transcript:
        return jsonify({"error": "transcript required"}), 400

    extracted = _parse_voice_transcript(transcript)

    merged = {**extracted, **payload}

    required = ["age","bmi","exercise_level","systolic_bp","diastolic_bp","heart_rate"]
    missing = [k for k in required if k not in merged]

    if missing:
        return jsonify({"error": f"Missing {missing}"}), 400

    age = float(merged["age"])
    bmi = float(merged["bmi"])
    ex = _exercise_to_num(merged["exercise_level"])
    sbp = float(merged["systolic_bp"])
    dbp = float(merged["diastolic_bp"])
    hr = float(merged["heart_rate"])

    model = _load_model()

    X = np.array([[age,bmi,ex,sbp,dbp,hr]])

    pred = int(model.predict(X)[0])
    risk_label = "High Risk" if pred else "Low Risk"

    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = 1.0 if pred else 0.0

    intel = _health_risk_intelligence(
        merged,
        prob_high=prob,
        risk_label=risk_label
    )

    return jsonify({
        "ok":True,
        "mode":"voice",
        "transcript": transcript,
        "extracted": extracted,
        "risk_label":risk_label,
        "probability":prob,
        **intel
    })

from sqlalchemy import text
import json

@api_bp.get("/summary")
def summary():
    _ensure_pipeline_tables()  # from your event-pipeline block
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id required"}), 400

    # Pull latest generated summary (worker writes these)
    row = db.session.execute(text("""
        SELECT created_at, summary_text, model_version, meta_json
        FROM health_summary
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT 1
    """), {"user_id": user_id}).fetchone()

    if not row:
        return jsonify({
            "ok": True,
            "user_id": user_id,
            "summary": "No summary yet. Submit a reading to /api/v1/intake first."
        }), 200

    return jsonify({
        "ok": True,
        "user_id": user_id,
        "created_at": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
        "model_version": row[2],
        "summary": row[1],
        "meta": json.loads(row[3]) if row[3] else {}
    }), 200

# ============================================================
# Real-time Monitor (SSE) + AI Assistant
# ============================================================

def _get_latest_context(user_id: str) -> dict:
    """
    Pull a compact 'patient context' that the assistant can reason over.
    Keep it fast: 1-3 queries max.
    """
    # Latest summary (written by worker)
    summary_row = db.session.execute(text("""
        SELECT created_at, summary_text, model_version
        FROM health_summary
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT 1
    """), {"user_id": user_id}).fetchone()

    # Recent timeline/events (last 10)
    events = db.session.execute(text("""
        SELECT id, created_at, event_type, payload_json
        FROM health_event
        WHERE user_id = :user_id
        ORDER BY id DESC
        LIMIT 10
    """), {"user_id": user_id}).fetchall()

    # Recent anomalies (last 10)
    anomalies = db.session.execute(text("""
        SELECT id, created_at, kind, severity, details_json
        FROM health_anomaly
        WHERE user_id = :user_id
        ORDER BY id DESC
        LIMIT 10
    """), {"user_id": user_id}).fetchall()

    return {
        "user_id": user_id,
        "latest_summary": {
            "created_at": summary_row[0].isoformat() if summary_row else None,
            "text": summary_row[1] if summary_row else None,
            "model_version": summary_row[2] if summary_row else None,
        },
        "recent_events": [
            {
                "id": r[0],
                "created_at": r[1].isoformat() if r[1] else None,
                "event_type": r[2],
                "payload": json.loads(r[3]) if r[3] else {},
            }
            for r in events
        ],
        "recent_anomalies": [
            {
                "id": r[0],
                "created_at": r[1].isoformat() if r[1] else None,
                "kind": r[2],
                "severity": r[3],
                "details": json.loads(r[4]) if r[4] else {},
            }
            for r in anomalies
        ],
    }


def _assistant_answer(question: str, ctx: dict) -> dict:
    """
    'Small change' assistant: deterministic, reliable, and fast.
    It uses the user's real data (summary/events/anomalies) to answer.
    You can later swap this function to a real LLM call without changing the API contract.
    """
    q = (question or "").strip().lower()
    summary = (ctx.get("latest_summary") or {}).get("text") or "No summary yet. Record an intake to generate one."
    anomalies = ctx.get("recent_anomalies") or []
    events = ctx.get("recent_events") or []

    # Extract last intake vitals if present
    last_intake = None
    for e in events:
        if (e.get("event_type") or "").lower() in ("intake", "health_intake"):
            last_intake = e.get("payload") or {}
            break

    # Quick helpers
    def fmt_bp(p):
        sbp = p.get("systolic_bp") or p.get("sys_bp")
        dbp = p.get("diastolic_bp") or p.get("dia_bp")
        if sbp is None or dbp is None:
            return None
        return f"{sbp}/{dbp}"

    # Simple intent routing
    if any(k in q for k in ["summary", "overview", "how am i doing", "status"]):
        return {
            "answer": summary,
            "suggestions": [
                "Ask: 'Any anomalies lately?'",
                "Ask: 'Show my latest vitals'",
                "Ask: 'What changed this week?'",
            ],
        }

    if any(k in q for k in ["anomaly", "alert", "anything wrong", "red flag"]):
        if not anomalies:
            return {"answer": "No anomalies detected in your recent readings.", "suggestions": ["Ask: 'Show my latest vitals'"]}
        top = anomalies[0]
        return {
            "answer": f"Most recent anomaly: {top.get('kind')} (severity: {top.get('severity')}).",
            "details": top.get("details", {}),
            "suggestions": ["Ask: 'What should I do next?'", "Ask: 'Explain this anomaly in plain English'"],
        }

    if any(k in q for k in ["latest", "vitals", "blood pressure", "heart rate", "my readings"]):
        if not last_intake:
            return {"answer": "I don't see a recent intake event yet. Send /intake first.", "suggestions": []}
        bp = fmt_bp(last_intake)
        hr = last_intake.get("heart_rate")
        bmi = last_intake.get("bmi")
        parts = []
        if bp: parts.append(f"BP: {bp}")
        if hr is not None: parts.append(f"HR: {hr}")
        if bmi is not None: parts.append(f"BMI: {bmi}")
        return {"answer": "Latest readings — " + ", ".join(parts) if parts else "Latest intake found, but readings were incomplete.", "suggestions": ["Ask: 'Any anomalies lately?'"]}

    # Default: return a helpful data-grounded response
    return {
        "answer": "I can help with your health timeline. Try: 'summary', 'latest vitals', or 'anomalies'.",
        "context_hint": {
            "has_summary": bool(ctx.get("latest_summary", {}).get("text")),
            "events_seen": len(events),
            "anomalies_seen": len(anomalies),
        },
    }


@api_bp.post("/assistant/ask")
def assistant_ask():
    """
    Minimal AI assistant endpoint.
    Body: { "user_id": "123", "question": "How am I doing this week?" }
    """
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "")).strip()
    question = str(data.get("question", "")).strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id required"}), 400
    if not question:
        return jsonify({"error": "validation", "message": "question required"}), 400

    ctx = _get_latest_context(user_id)
    out = _assistant_answer(question, ctx)
    return jsonify({"ok": True, "user_id": user_id, "question": question, **out}), 200


@api_bp.get("/monitor/stream")
def monitor_stream():
    """
    Real-time monitor via Server-Sent Events (SSE).
    Client connects and receives events like:
      event: alert
      data: {...json...}

    Query params:
      user_id=123
      after_id=0   (optional) start after this anomaly id
    """
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id required"}), 400

    try:
        after_id = int(request.args.get("after_id", "0"))
    except Exception:
        after_id = 0

    def gen():
        nonlocal after_id
        # Initial hello (helps proxies keep the connection)
        yield "event: hello\ndata: {\"ok\":true}\n\n"

        while True:
            rows = db.session.execute(text("""
                SELECT id, created_at, kind, severity, details_json
                FROM health_anomaly
                WHERE user_id = :user_id AND id > :after_id
                ORDER BY id ASC
                LIMIT 25
            """), {"user_id": user_id, "after_id": after_id}).fetchall()

            for r in rows:
                after_id = max(after_id, int(r[0]))
                payload = {
                    "id": r[0],
                    "created_at": r[1].isoformat() if r[1] else None,
                    "kind": r[2],
                    "severity": r[3],
                    "details": json.loads(r[4]) if r[4] else {},
                }
                yield "event: alert\n"
                yield "data: " + json.dumps(payload) + "\n\n"

            # Keepalive ping every ~10s so Render/proxies don’t drop it
            yield "event: ping\ndata: {}\n\n"
            time.sleep(10)

    return Response(stream_with_context(gen()), mimetype="text/event-stream")

@api_bp.post("/assistant/ask")
def assistant_ask():
    try:
        data = request.get_json(silent=True) or {}
        user_id = str(data.get("user_id","")).strip()
        question = str(data.get("question","")).strip()

        if not user_id:
            return jsonify({"error":"user_id required"}),400

        return jsonify({
            "ok": True,
            "user_id": user_id,
            "answer": "Your AI health assistant is active. A full summary will appear after your first intake."
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }),500
