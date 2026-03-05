# era/api/routes.py
import json
import re
import time
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db
from flask import request, jsonify


@api_bp.post("/api/v1/events")
def ingest_event():
    body = request.get_json(force=True) or {}
    tenant_id = body.get("tenant_id", "demo-tenant")
    patient_id = str(body.get("patient_id", "unknown"))
    payload = body.get("payload", {})

    db.session.execute(
        text("""
        INSERT INTO risk_jobs (tenant_id, patient_id, payload_json, status)
        VALUES (:t, :p, :j, 'pending')
        """),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(payload)},
    )
    db.session.commit()
    return jsonify({"ok": True, "tenant_id": tenant_id, "patient_id": patient_id}), 200

# Optional ORM model (if you have it)
try:
    from era.models import HealthEvent  # type: ignore
except Exception:
    HealthEvent = None  # type: ignore

# Optional AI reasoning engine (if present)
try:
    from era.ai.health_reasoning import generate_health_reasoning  # type: ignore
except Exception:
    generate_health_reasoning = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# JSON error handler (ONE)
# ----------------------------
@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500
    
@api_bp.get("/routes")
def list_routes():
    from flask import current_app
    return {
        "routes": [
            str(r) for r in current_app.url_map.iter_rules()
        ]
    }


# ----------------------------
# Helpers
# ----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _exercise_to_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("0", "low"):
        return 0.0
    if s in ("1", "moderate", "medium"):
        return 1.0
    if s in ("2", "high"):
        return 2.0
    return _safe_float(x)


# Regexes for voice transcript parsing
_BP_RE = re.compile(r"\b(?:bp|blood\s*pressure)\s*(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.I)
_HR_RE = re.compile(r"\b(?:hr|heart\s*rate)\s*(\d{2,3})\b", re.I)
_BMI_RE = re.compile(r"\bbmi\s*(\d{1,2}(?:\.\d+)?)\b", re.I)
_AGE_RE = re.compile(r"\bage\s*(\d{1,3})\b", re.I)
_EX_RE = re.compile(r"\bexercise\s*(low|moderate|high)\b", re.I)


def _parse_transcript(transcript: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = transcript or ""

    m = _BP_RE.search(t)
    if m:
        out["systolic_bp"] = float(m.group(1))
        out["diastolic_bp"] = float(m.group(2))

    m = _HR_RE.search(t)
    if m:
        out["heart_rate"] = float(m.group(1))

    m = _BMI_RE.search(t)
    if m:
        out["bmi"] = float(m.group(1))

    m = _AGE_RE.search(t)
    if m:
        out["age"] = float(m.group(1))

    m = _EX_RE.search(t)
    if m:
        out["exercise_level"] = m.group(1).capitalize()

    return out


def _coerce_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize keys + cast types. Keeps raw fields too.
    """
    out = dict(payload or {})
    out["user_id"] = str(out.get("user_id") or "").strip()

    out["age"] = _safe_float(out.get("age"))
    out["bmi"] = _safe_float(out.get("bmi"))
    out["systolic_bp"] = _safe_float(out.get("systolic_bp"))
    out["diastolic_bp"] = _safe_float(out.get("diastolic_bp"))
    out["heart_rate"] = _safe_float(out.get("heart_rate"))
    out["exercise_level"] = out.get("exercise_level")
    out["exercise_num"] = _exercise_to_num(out.get("exercise_level"))

    return out


# ----------------------------
# Minimal anomaly detection (rules)
# ----------------------------
def _detect_anomalies(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    sbp = v.get("systolic_bp")
    dbp = v.get("diastolic_bp")
    hr = v.get("heart_rate")
    bmi = v.get("bmi")

    anomalies: List[Dict[str, Any]] = []

    # Hypertensive crisis
    if (sbp is not None and sbp >= 180) or (dbp is not None and dbp >= 120):
        anomalies.append({
            "kind": "hypertensive_crisis",
            "severity": 3,
            "details": {"sbp": sbp, "dbp": dbp, "hr": hr, "bmi": bmi},
        })
        return anomalies  # highest priority

    # Stage 2 hypertension
    if (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90):
        anomalies.append({
            "kind": "stage2_hypertension",
            "severity": 2,
            "details": {"sbp": sbp, "dbp": dbp, "hr": hr, "bmi": bmi},
        })

    # Tachy / brady (simple)
    if hr is not None and hr >= 120:
        anomalies.append({"kind": "tachycardia", "severity": 2, "details": {"hr": hr}})
    if hr is not None and hr < 50:
        anomalies.append({"kind": "bradycardia", "severity": 2, "details": {"hr": hr}})

    # BMI flag (non-emergency)
    if bmi is not None and bmi >= 35:
        anomalies.append({"kind": "high_bmi", "severity": 1, "details": {"bmi": bmi}})

    return anomalies


# ----------------------------
# Storage: ensure event table exists (works even without ORM)
# ----------------------------
def _ensure_health_event_table() -> None:
    """
    Creates a simple table if ORM isn't present.
    Safe to call repeatedly.
    """
    if HealthEvent is not None:
        return

    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS health_event (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data_json TEXT
            );
        """))
        db.session.commit()
    except Exception:
        db.session.rollback()


def _insert_event(user_id: str, event_type: str, data: Dict[str, Any]) -> Optional[int]:
    created_at = _now_utc()
    payload_json = json.dumps(data, separators=(",", ":"), default=str)

    if HealthEvent is not None:
        try:
            ev = HealthEvent(  # type: ignore
                user_id=user_id,
                event_type=event_type,
                created_at=created_at,
                data_json=payload_json,
            )
            db.session.add(ev)
            db.session.commit()
            return getattr(ev, "id", None)
        except Exception:
            db.session.rollback()
            # fall through to raw insert attempt

    _ensure_health_event_table()
    try:
        res = db.session.execute(
            text("""
                INSERT INTO health_event (created_at, user_id, event_type, data_json)
                VALUES (:created_at, :user_id, :event_type, :data_json)
                RETURNING id
            """),
            {
                "created_at": created_at,
                "user_id": user_id,
                "event_type": event_type,
                "data_json": payload_json,
            },
        )
        new_id = res.scalar()
        db.session.commit()
        return int(new_id) if new_id is not None else None
    except Exception:
        db.session.rollback()
        return None


def _get_recent_events(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    try:
        rows = db.session.execute(
            text("""
                SELECT created_at, event_type, data_json
                FROM health_event
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"user_id": user_id, "limit": limit},
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            data_json = r[2]
            try:
                data = json.loads(data_json) if data_json else {}
            except Exception:
                data = {"raw": data_json}
            out.append({"created_at": str(r[0]), "event_type": r[1], "data": data})
        return out
    except Exception:
        return []


# ----------------------------
# REAL-TIME STREAM BUS (the “12-line upgrade” concept)
# ----------------------------
# In-memory pub/sub (works on one instance). Billion-dollar version swaps this with Redis Streams.
_subscribers: Dict[str, List["queue.Queue[str]"]] = {}
_sub_lock = threading.Lock()

def _publish(user_id: str, event: Dict[str, Any]) -> None:
    msg = json.dumps(event, separators=(",", ":"), default=str)
    with _sub_lock:
        qs = list(_subscribers.get(user_id, []))
    for q in qs:
        try:
            q.put_nowait(msg)
        except Exception:
            pass


def _sse_stream(user_id: str):
    q: "queue.Queue[str]" = queue.Queue(maxsize=200)
    with _sub_lock:
        _subscribers.setdefault(user_id, []).append(q)

    try:
        # initial ping so client sees connection
        yield "event: ping\ndata: {}\n\n"
        while True:
            try:
                msg = q.get(timeout=20)
                yield f"event: anomaly\ndata: {msg}\n\n"
            except queue.Empty:
                yield f"event: ping\ndata: {json.dumps({'ts': str(_now_utc())})}\n\n"
    finally:
        with _sub_lock:
            if user_id in _subscribers and q in _subscribers[user_id]:
                _subscribers[user_id].remove(q)


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


@api_bp.get("/api/v1/stream/anomalies")
def stream_anomalies():
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    return Response(
        stream_with_context(_sse_stream(user_id)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_bp.post("/api/v1/demo/intake")
def demo_intake():
    payload = request.get_json(silent=True) or {}
    v = _coerce_inputs(payload)
    user_id = v.get("user_id") or ""
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    # store intake event
    event_id = _insert_event(user_id, "health_intake", v)

    # continuous “analysis” on every new event (smallest real-time engine)
    anomalies = _detect_anomalies(v)
    published = []
    for a in anomalies:
        evt = {
            "id": event_id,
            "user_id": user_id,
            "kind": a["kind"],
            "severity": a["severity"],
            "details": a.get("details", {}),
            "created_at": str(_now_utc()),
        }
        _publish(user_id, evt)
        published.append(evt)

    return jsonify({"ok": True, "user_id": user_id, "event_id": event_id, "anomalies": published}), 200


@api_bp.post("/api/v1/voice/intake")
def voice_intake():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    transcript = str(payload.get("transcript") or "")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not transcript:
        return jsonify({"error": "transcript required"}), 400

    extracted = _parse_transcript(transcript)
    merged = dict(payload)
    merged.update(extracted)
    merged["user_id"] = user_id
    merged["transcript"] = transcript

    # route through the same engine
    v = _coerce_inputs(merged)
    event_id = _insert_event(user_id, "voice_intake", {"transcript": transcript, "extracted": extracted})

    anomalies = _detect_anomalies(v)
    published = []
    for a in anomalies:
        evt = {
            "id": event_id,
            "user_id": user_id,
            "kind": a["kind"],
            "severity": a["severity"],
            "details": a.get("details", {}),
            "created_at": str(_now_utc()),
        }
        _publish(user_id, evt)
        published.append(evt)

    return jsonify({
        "ok": True,
        "user_id": user_id,
        "event_id": event_id,
        "extracted": extracted,
        "anomalies": published,
    }), 200


@api_bp.get("/ai/health/reasoning")
def ai_health_reasoning():
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    # If your real AI engine exists, use it.
    if generate_health_reasoning is not None:
        result = generate_health_reasoning(user_id)
        return jsonify({"user_id": user_id, "ai_analysis": result}), 200

    # Fallback reasoning: summarize recent events
    recent = _get_recent_events(user_id, limit=12)
    summary = {
        "note": "AI engine not available; returning recent event summary.",
        "recent_events": recent,
        "tip": "Add era/ai/health_reasoning.py with generate_health_reasoning(user_id) to enable the full AI assistant.",
    }
    return jsonify({"user_id": user_id, "ai_analysis": summary}), 200
