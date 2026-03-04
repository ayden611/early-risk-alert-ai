# era/api/routes.py
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# Optional ORM model (if you have it)
try:
    from era.models import HealthEvent  # type: ignore
except Exception:
    HealthEvent = None  # type: ignore

# Optional reasoning engine (if present)
try:
    from era.ai.health_reasoning import generate_health_reasoning  # type: ignore
except Exception:
    generate_health_reasoning = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# Error handler (JSON)
# ----------------------------
@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500


# ----------------------------
# Helpers
# ----------------------------
_BP_RE = re.compile(r"\b(?:bp|blood\s*pressure)\s*(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.I)
_HR_RE = re.compile(r"\b(?:hr|heart\s*rate|pulse)\s*(\d{2,3})\b", re.I)
_BMI_RE = re.compile(r"\b(?:bmi)\s*(\d{2,3}(?:\.\d+)?)\b", re.I)
_AGE_RE = re.compile(r"\b(?:age)\s*(\d{1,3})\b", re.I)
_EX_RE = re.compile(r"\b(?:exercise|activity)\s*(low|moderate|high)\b", re.I)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _exercise_to_num(level: Any) -> Optional[int]:
    if level is None:
        return None
    s = str(level).strip().lower()
    if s in ("0", "low"):
        return 0
    if s in ("1", "moderate", "mod"):
        return 1
    if s in ("2", "high"):
        return 2
    return None


def _parse_transcript(transcript: str) -> Dict[str, Any]:
    """
    Parses a simple transcript like:
    "BP 140 over 90 heart rate 88 bmi 27.5 age 45 exercise moderate"
    """
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
    Accepts either:
      - systolic_bp or sbp
      - diastolic_bp or dbp
      - exercise_level or exercise
    """
    p = dict(payload or {})

    # Normalize keys
    if "sbp" in p and "systolic_bp" not in p:
        p["systolic_bp"] = p.get("sbp")
    if "dbp" in p and "diastolic_bp" not in p:
        p["diastolic_bp"] = p.get("dbp")
    if "exercise" in p and "exercise_level" not in p:
        p["exercise_level"] = p.get("exercise")

    # Coerce numeric
    for k in ("age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate"):
        if k in p:
            p[k] = _safe_float(p.get(k))

    return p


def _ensure_pipeline_tables() -> None:
    """
    Creates/patches the minimal tables needed for:
      - health events
      - storing JSON as text (data_json)
    Works on Postgres.
    """
    # Create table if not exists
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS health_event (
              id BIGSERIAL PRIMARY KEY,
              user_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              data_json TEXT
            );
            """
        )
    )
    # Ensure column exists (fixes your UndefinedColumn: data_json)
    db.session.execute(text("ALTER TABLE health_event ADD COLUMN IF NOT EXISTS data_json TEXT;"))
    db.session.execute(text("CREATE INDEX IF NOT EXISTS ix_health_event_user_id_id ON health_event(user_id, id);"))
    db.session.execute(text("CREATE INDEX IF NOT EXISTS ix_health_event_type ON health_event(event_type);"))
    db.session.commit()


def _insert_event(user_id: str, event_type: str, data: Dict[str, Any]) -> int:
    _ensure_pipeline_tables()
    created_at = _now_utc()

    if HealthEvent is not None:
        row = HealthEvent(  # type: ignore
            user_id=user_id,
            event_type=event_type,
            created_at=created_at,
            data_json=json.dumps(data),
        )
        db.session.add(row)
        db.session.commit()
        return int(getattr(row, "id"))
    else:
        res = db.session.execute(
            text(
                """
                INSERT INTO health_event (user_id, event_type, created_at, data_json)
                VALUES (:user_id, :event_type, :created_at, :data_json)
                RETURNING id
                """
            ),
            {
                "user_id": user_id,
                "event_type": event_type,
                "created_at": created_at,
                "data_json": json.dumps(data),
            },
        )
        new_id = int(res.scalar() or 0)
        db.session.commit()
        return new_id


def _fetch_events(user_id: str, event_types: List[str], after_id: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_pipeline_tables()
    res = db.session.execute(
        text(
            """
            SELECT id, user_id, event_type, created_at, data_json
            FROM health_event
            WHERE user_id = :user_id
              AND event_type = ANY(:types)
              AND id > :after_id
            ORDER BY id ASC
            LIMIT :limit
            """
        ),
        {"user_id": user_id, "types": event_types, "after_id": after_id, "limit": limit},
    )
    out: List[Dict[str, Any]] = []
    for r in res.mappings():
        payload = {}
        try:
            payload = json.loads(r["data_json"] or "{}")
        except Exception:
            payload = {"raw": r["data_json"]}
        out.append(
            {
                "id": int(r["id"]),
                "user_id": r["user_id"],
                "event_type": r["event_type"],
                "created_at": (r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"])),
                "data": payload,
            }
        )
    return out


# ----------------------------
# 20-line anomaly detection drop-in (clean + fast)
# ----------------------------
def detect_anomalies(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    sbp = _safe_float(v.get("systolic_bp"))
    dbp = _safe_float(v.get("diastolic_bp"))
    hr = _safe_float(v.get("heart_rate"))
    bmi = _safe_float(v.get("bmi"))
    out = []
    if sbp is not None and dbp is not None:
        if sbp >= 180 or dbp >= 120: out.append({"kind":"hypertensive_crisis","severity":3})
        elif sbp >= 140 or dbp >= 90: out.append({"kind":"stage2_hypertension","severity":2})
        elif sbp >= 130 or dbp >= 80: out.append({"kind":"stage1_hypertension","severity":1})
        elif sbp < 90 or dbp < 60: out.append({"kind":"hypotension","severity":1})
    if hr is not None:
        if hr >= 120: out.append({"kind":"tachycardia","severity":2})
        elif hr <= 45: out.append({"kind":"bradycardia","severity":2})
    if bmi is not None and bmi >= 35: out.append({"kind":"very_high_bmi","severity":1})
    return out


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    db_ok = True
    err = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        err = str(e)

    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "db_ok": db_ok,
            "db_error": err,
        }
    ), (200 if db_ok else 503)


# Typed intake
@api_bp.post("/api/v1/demo/intake")
def demo_intake():
    payload = request.get_json(silent=True) or {}
    payload = _coerce_inputs(payload)

    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    # Store intake
    intake_id = _insert_event(user_id, "health_intake", payload)

    # Detect anomalies + store them as events
    anomalies = detect_anomalies(payload)
    saved = []
    for a in anomalies:
        a_payload = {
            "source_event_id": intake_id,
            "details": {
                "age": payload.get("age"),
                "bmi": payload.get("bmi"),
                "sbp": payload.get("systolic_bp"),
                "dbp": payload.get("diastolic_bp"),
                "hr": payload.get("heart_rate"),
            },
            **a,
        }
        aid = _insert_event(user_id, "anomaly", a_payload)
        saved.append({**a_payload, "id": aid})

    return jsonify({"ok": True, "user_id": user_id, "intake_id": intake_id, "anomalies": saved}), 200


# Voice transcript intake (you send transcript text; app parses vitals)
@api_bp.post("/api/v1/voice/intake")
def voice_intake():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    transcript = str(payload.get("transcript") or "")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not transcript.strip():
        return jsonify({"error": "transcript required"}), 400

    parsed = _parse_transcript(transcript)
    merged = _coerce_inputs({**parsed, "user_id": user_id, "transcript": transcript})

    voice_id = _insert_event(user_id, "voice_intake", merged)

    anomalies = detect_anomalies(merged)
    saved = []
    for a in anomalies:
        a_payload = {
            "source_event_id": voice_id,
            "details": {
                "age": merged.get("age"),
                "bmi": merged.get("bmi"),
                "sbp": merged.get("systolic_bp"),
                "dbp": merged.get("diastolic_bp"),
                "hr": merged.get("heart_rate"),
            },
            **a,
        }
        aid = _insert_event(user_id, "anomaly", a_payload)
        saved.append({**a_payload, "id": aid})

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "voice_id": voice_id,
            "parsed": {k: merged.get(k) for k in ("age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate")},
            "anomalies": saved,
        }
    ), 200


# SSE: stream anomaly events in real time
@api_bp.get("/api/v1/stream/anomalies")
def stream_anomalies():
    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    retry_ms = 2000

    def gen():
        last_id = 0

        # Send a header retry directive
        yield f"retry: {retry_ms}\n\n"

        # On connect, send last few anomalies (optional)
        try:
            recent = _fetch_events(user_id, ["anomaly"], after_id=0, limit=5)
            for ev in recent:
                last_id = max(last_id, ev["id"])
                yield "event: anomaly\n"
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield "event: error\n"
            yield f"data: {json.dumps({'message': str(e)})}\n\n"

        # Poll loop
        while True:
            time.sleep(1.0)
            try:
                new_events = _fetch_events(user_id, ["anomaly"], after_id=last_id, limit=50)
                for ev in new_events:
                    last_id = max(last_id, ev["id"])
                    yield "event: anomaly\n"
                    yield f"data: {json.dumps(ev)}\n\n"
                # keepalive ping
                yield "event: ping\n"
                yield f"data: {json.dumps({'ts': _now_utc().isoformat()})}\n\n"
            except Exception as e:
                yield "event: error\n"
                yield f"data: {json.dumps({'message': str(e)})}\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


# AI reasoning engine endpoint
@api_bp.get("/api/v1/ai/health/reasoning")
def ai_health_reasoning():
    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    if generate_health_reasoning is None:
        return jsonify({"error": "ai", "message": "generate_health_reasoning not available"}), 501

    result = generate_health_reasoning(user_id)
    return jsonify({"user_id": user_id, "ai_analysis": result}), 200
