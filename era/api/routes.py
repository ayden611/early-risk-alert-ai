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

api_bp = Blueprint("api", __name__)

# ----------------------------
# Optional imports (don't crash if missing)
# ----------------------------
try:
    from era.models import HealthEvent  # type: ignore
except Exception:
    HealthEvent = None  # type: ignore

try:
    from era.ai.health_reasoning import generate_health_reasoning  # type: ignore
except Exception:
    generate_health_reasoning = None  # type: ignore


@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500


# ----------------------------
# Helpers: parsing + anomaly rules
# ----------------------------
_BP_RE = re.compile(r"\b(?:bp|blood\s*pressure)\s*(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.I)
_HR_RE = re.compile(r"\b(?:hr|heart\s*rate)\s*(\d{2,3})\b", re.I)
_BMI_RE = re.compile(r"\b(?:bmi)\s*([0-9]+(?:\.[0-9]+)?)\b", re.I)
_AGE_RE = re.compile(r"\b(?:age)\s*(\d{1,3})\b", re.I)
_EX_RE = re.compile(r"\b(?:exercise)\s*(low|moderate|high)\b", re.I)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_transcript(t: str) -> Dict[str, Any]:
    t = (t or "").strip()
    out: Dict[str, Any] = {"transcript": t}

    m = _BP_RE.search(t)
    if m:
        out["systolic_bp"] = int(m.group(1))
        out["diastolic_bp"] = int(m.group(2))

    m = _HR_RE.search(t)
    if m:
        out["heart_rate"] = int(m.group(1))

    m = _BMI_RE.search(t)
    if m:
        out["bmi"] = float(m.group(1))

    m = _AGE_RE.search(t)
    if m:
        out["age"] = int(m.group(1))

    m = _EX_RE.search(t)
    if m:
        out["exercise_level"] = m.group(1).capitalize()

    return out


def _rule_anomalies(details: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    sbp = _safe_float(details.get("systolic_bp") or details.get("sbp"))
    dbp = _safe_float(details.get("diastolic_bp") or details.get("dbp"))
    hr = _safe_float(details.get("heart_rate") or details.get("hr"))

    kind = None
    severity = 0

    # Demo rules (NOT medical advice)
    if sbp is not None and dbp is not None:
        if sbp >= 180 or dbp >= 120:
            kind = "hypertensive_crisis"
            severity = 3
        elif sbp >= 140 or dbp >= 90:
            kind = "stage1_hypertension"
            severity = 2
        elif sbp >= 130 or dbp >= 80:
            kind = "elevated_bp"
            severity = 1

    if hr is not None:
        if hr >= 130:
            kind = kind or "tachycardia"
            severity = max(severity, 2)
        elif hr <= 45:
            kind = kind or "bradycardia"
            severity = max(severity, 2)

    if not kind:
        return None, {"status": "ok", "note": "No rule-based anomaly detected (demo rules)."}

    anomaly = {
        "kind": kind,
        "severity": severity,
        "details": {
            "sbp": sbp,
            "dbp": dbp,
            "hr": hr,
            "age": details.get("age"),
            "bmi": details.get("bmi"),
            "exercise_level": details.get("exercise_level"),
        },
        "ts": _now_utc().isoformat(),
    }
    summary = {"status": "anomaly", "kind": kind, "severity": severity, "note": "Rule-based anomaly detected (demo rules)."}
    return anomaly, summary


# ----------------------------
# DB bootstrap (no manual ALTER TABLE needed)
# ----------------------------
def _ensure_pipeline_tables() -> None:
    dialect = db.engine.dialect.name

    if dialect == "postgresql":
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_event (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    data_json TEXT
                );
                """
            )
        )
        db.session.execute(text("ALTER TABLE health_event ADD COLUMN IF NOT EXISTS data_json TEXT;"))
    else:
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT
                );
                """
            )
        )
        cols = db.session.execute(text("PRAGMA table_info(health_event);")).fetchall()
        col_names = {c[1] for c in cols}
        if "data_json" not in col_names:
            db.session.execute(text("ALTER TABLE health_event ADD COLUMN data_json TEXT;"))

    db.session.commit()


def _insert_event(user_id: str, event_type: str, payload: Dict[str, Any]) -> int:
    _ensure_pipeline_tables()
    created_at = _now_utc()

    if db.engine.dialect.name == "postgresql":
        row = db.session.execute(
            text(
                """
                INSERT INTO health_event (user_id, event_type, created_at, data_json)
                VALUES (:user_id, :event_type, :created_at, :data_json)
                RETURNING id;
                """
            ),
            {"user_id": user_id, "event_type": event_type, "created_at": created_at, "data_json": json.dumps(payload)},
        ).fetchone()
        db.session.commit()
        return int(row[0])

    db.session.execute(
        text(
            """
            INSERT INTO health_event (user_id, event_type, created_at, data_json)
            VALUES (:user_id, :event_type, :created_at, :data_json);
            """
        ),
        {"user_id": user_id, "event_type": event_type, "created_at": created_at.isoformat(), "data_json": json.dumps(payload)},
    )
    db.session.commit()
    rid = db.session.execute(text("SELECT last_insert_rowid();")).fetchone()
    return int(rid[0])


def _fetch_events_since(user_id: str, since_id: int, only_type: Optional[str] = None):
    _ensure_pipeline_tables()
    q = """
        SELECT id, user_id, event_type, created_at, data_json
        FROM health_event
        WHERE user_id = :user_id AND id > :since_id
    """
    params = {"user_id": user_id, "since_id": since_id}
    if only_type:
        q += " AND event_type = :event_type"
        params["event_type"] = only_type
    q += " ORDER BY id ASC LIMIT 200;"
    return db.session.execute(text(q), params).fetchall()


# ----------------------------
# Core endpoints
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return jsonify({"status": "ok" if db_ok else "degraded", "db_ok": db_ok, "service": "early-risk-alert"}), (200 if db_ok else 503)


@api_bp.get("/routes")
def list_routes():
    return jsonify({"routes": [str(r) for r in current_app.url_map.iter_rules()]}), 200


@api_bp.post("/api/v1/demo/intake")
def demo_intake():
    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    event_id = _insert_event(user_id, "health_intake", body)

    anomaly, summary = _rule_anomalies(body)
    anomaly_id = _insert_event(user_id, "anomaly", anomaly) if anomaly else None

    return jsonify({"ok": True, "user_id": user_id, "event_id": event_id, "anomaly_event_id": anomaly_id, "summary": summary}), 200


@api_bp.post("/api/v1/voice/intake")
def voice_intake():
    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id") or "").strip()
    transcript = str(body.get("transcript") or "").strip()

    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not transcript:
        return jsonify({"error": "transcript required"}), 400

    parsed = _parse_transcript(transcript)
    payload = {**body, **parsed}

    event_id = _insert_event(user_id, "voice_intake", payload)

    anomaly, summary = _rule_anomalies(payload)
    anomaly_id = _insert_event(user_id, "anomaly", anomaly) if anomaly else None

    return jsonify({"ok": True, "user_id": user_id, "event_id": event_id, "parsed": parsed, "anomaly_event_id": anomaly_id, "summary": summary}), 200


@api_bp.get("/api/v1/stream/anomalies")
def stream_anomalies():
    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    _ensure_pipeline_tables()
    latest = db.session.execute(
        text(
            """
            SELECT COALESCE(MAX(id), 0)
            FROM health_event
            WHERE user_id = :user_id AND event_type = 'anomaly';
            """
        ),
        {"user_id": user_id},
    ).fetchone()
    last_id = int(latest[0] or 0)

    def sse():
        nonlocal last_id
        yield "event: ping\ndata: " + json.dumps({"ts": _now_utc().isoformat(), "hello": True}) + "\n\n"

        while True:
            try:
                rows = _fetch_events_since(user_id, last_id, only_type="anomaly")
                for r in rows:
                    rid = int(r[0])
                    data_json = r[4] or "{}"
                    last_id = max(last_id, rid)
                    yield "event: anomaly\ndata: " + json.dumps({"id": rid, "user_id": user_id, "data": json.loads(data_json)}) + "\n\n"

                yield "event: ping\ndata: " + json.dumps({"ts": _now_utc().isoformat()}) + "\n\n"
                time.sleep(2)
            except GeneratorExit:
                return
            except Exception as e:
                yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"
                time.sleep(2)

    return Response(stream_with_context(sse()), mimetype="text/event-stream")


@api_bp.get("/ai/health/reasoning")
def ai_health_reasoning():
    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    if generate_health_reasoning is None:
        return jsonify({"user_id": user_id, "ai_analysis": {"status": "unavailable", "message": "Reasoning engine not installed in this deploy."}}), 200

    result = generate_health_reasoning(user_id)
    return jsonify({"user_id": user_id, "ai_analysis": result}), 200
