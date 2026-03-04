# era/api/routes.py
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# Optional model (save events if it exists)
try:
    from era.models import HealthEvent  # type: ignore
except Exception:
    HealthEvent = None  # type: ignore

# ------------------------------------------------------------
# Blueprint MUST be defined before decorators
# ------------------------------------------------------------
api_bp = Blueprint("api", __name__)

# ------------------------------------------------------------
# Global error handler (prevents HTML 500 pages)
# ------------------------------------------------------------
@api_bp.errorhandler(Exception)
def _handle_error(e: Exception):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _first(payload: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return default


def _exercise_to_num(v: Any) -> float:
    # accepts: Low/Moderate/High OR 0/1/2
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("low", "l", "0"):
        return 0.0
    if s in ("moderate", "mod", "m", "1"):
        return 1.0
    if s in ("high", "h", "2"):
        return 2.0
    # fallback: try numeric
    try:
        return float(s)
    except Exception:
        return 0.0


def _coerce_inputs(payload: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    age = float(_first(payload, ["age"], 0) or 0)
    bmi = float(_first(payload, ["bmi"], 0) or 0)
    ex = _exercise_to_num(_first(payload, ["exercise_level", "exercise"], 0))
    sbp = float(_first(payload, ["systolic_bp", "sys_bp", "sbp"], 0) or 0)
    dbp = float(_first(payload, ["diastolic_bp", "dia_bp", "dbp"], 0) or 0)
    hr = float(_first(payload, ["heart_rate", "hr"], 0) or 0)
    return age, bmi, ex, sbp, dbp, hr


# ------------------------------------------------------------
# Voice transcript parsing
# ------------------------------------------------------------
_BP_RE = re.compile(r"\b(?:bp\s*)?(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.IGNORECASE)
_HR_RE = re.compile(r"\b(?:hr|heart\s*rate)\s*(\d{2,3})\b", re.IGNORECASE)
_BMI_RE = re.compile(r"\bbmi\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
_AGE_RE = re.compile(r"\bage\s*(\d{1,3})\b", re.IGNORECASE)
_EX_RE = re.compile(r"\b(low|moderate|high)\b", re.IGNORECASE)


def _parse_transcript(transcript: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = (transcript or "").strip()

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


# ------------------------------------------------------------
# DB table for anomalies (no migrations required)
# ------------------------------------------------------------
def _ensure_pipeline_tables() -> None:
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS health_anomaly (
              id SERIAL PRIMARY KEY,
              user_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              severity INT NOT NULL,
              details_json TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )
    db.session.commit()


def _run_anomaly_detection(user_id: str, age: float, bmi: float, sbp: float, dbp: float, hr: float) -> Optional[Dict[str, Any]]:
    # Simple, clear, "real-time monitoring" rules
    severity, kind = 0, None

    if sbp >= 180 or dbp >= 120:
        severity, kind = 3, "hypertensive_crisis"
    elif sbp >= 160 or dbp >= 100:
        severity, kind = 2, "stage2_hypertension"
    elif sbp >= 140 or dbp >= 90:
        severity, kind = 1, "stage1_hypertension"

    if hr >= 120:
        severity, kind = max(severity, 2), kind or "tachycardia"
    if 0 < hr <= 45:
        severity, kind = max(severity, 2), kind or "bradycardia"

    if bmi >= 35:
        severity, kind = max(severity, 1), kind or "bmi_high"

    if severity <= 0 or not kind:
        return None

    details = {"age": age, "bmi": bmi, "sbp": sbp, "dbp": dbp, "hr": hr}
    _ensure_pipeline_tables()
    db.session.execute(
        text(
            """
            INSERT INTO health_anomaly (user_id, kind, severity, details_json, created_at)
            VALUES (:user_id, :kind, :severity, :details_json, NOW());
            """
        ),
        {"user_id": user_id, "kind": kind, "severity": severity, "details_json": json.dumps(details)},
    )
    db.session.commit()

    return {"user_id": user_id, "kind": kind, "severity": severity, "details": details}


def _save_event(user_id: str, event_type: str, data: Dict[str, Any]) -> bool:
    if HealthEvent is None:
        return False
    try:
        evt = HealthEvent(user_id=user_id, event_type=event_type, data=data)
        db.session.add(evt)
        db.session.commit()
        return True
    except Exception:
        db.session.rollback()
        return False


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@api_bp.get("/")
def index():
    return {"ok": True, "service": "early-risk-alert-ai", "version": "v1"}, 200


@api_bp.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        return {"status": "ok", "db_ok": True}, 200
    except Exception as e:
        return {"status": "degraded", "db_ok": False, "db_error": str(e)}, 503


@api_bp.post("/demo/intake")
def demo_intake():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(payload)
    except Exception:
        return jsonify({"error": "validation", "message": "Invalid numeric input"}), 400

    saved = _save_event(
        user_id=user_id,
        event_type="health_intake",
        data={"age": age, "bmi": bmi, "exercise_level": ex, "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": hr},
    )

    anomaly = _run_anomaly_detection(user_id, age, bmi, sbp, dbp, hr)

    return jsonify(
        {
            "ok": True,
            "saved": saved,
            "user_id": user_id,
            "vitals": {"age": age, "bmi": bmi, "exercise_level": ex, "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": hr},
            "anomaly": anomaly,
        }
    ), 200


@api_bp.post("/voice/intake")
def voice_intake():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    transcript = str(payload.get("transcript") or "").strip()

    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400
    if not transcript:
        return jsonify({"error": "validation", "message": "transcript is required"}), 400

    parsed = _parse_transcript(transcript)

    # merge parsed values into a new payload for the same intake pipeline
    merged = {"user_id": user_id, **parsed}

    # Save the raw voice event (optional)
    _save_event(user_id=user_id, event_type="voice_intake", data={"transcript": transcript, "parsed": parsed})

    # Reuse demo_intake logic by calling the same coercion + anomaly detection
    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(merged)
    except Exception:
        return jsonify({"error": "validation", "message": "Could not parse numeric values from transcript"}), 400

    saved = _save_event(
        user_id=user_id,
        event_type="health_intake",
        data={"age": age, "bmi": bmi, "exercise_level": ex, "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": hr},
    )

    anomaly = _run_anomaly_detection(user_id, age, bmi, sbp, dbp, hr)

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "parsed": parsed,
            "saved": saved,
            "vitals": {"age": age, "bmi": bmi, "exercise_level": ex, "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": hr},
            "anomaly": anomaly,
        }
    ), 200


@api_bp.get("/stream/anomalies")
def stream_anomalies():
    """
    Server-Sent Events stream:
    GET /api/v1/stream/anomalies?user_id=123
    """
    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    _ensure_pipeline_tables()

    def gen():
        last_id = 0
        while True:
            rows = db.session.execute(
                text(
                    """
                    SELECT id, kind, severity, details_json, created_at
                    FROM health_anomaly
                    WHERE user_id = :user_id AND id > :last_id
                    ORDER BY id ASC
                    LIMIT 50
                    """
                ),
                {"user_id": user_id, "last_id": last_id},
            ).fetchall()

            for r in rows:
                last_id = int(r[0])
                payload = {
                    "id": last_id,
                    "user_id": user_id,
                    "kind": r[1],
                    "severity": int(r[2]),
                    "details": json.loads(r[3]),
                    "created_at": str(r[4]),
                }
                yield f"event: anomaly\ndata: {json.dumps(payload)}\n\n"

            # heartbeat so Render/proxies keep connection open
            yield f"event: ping\ndata: {json.dumps({'ts': _now_utc().isoformat()})}\n\n"
            time.sleep(2)

    return Response(stream_with_context(gen()), mimetype="text/event-stream")
