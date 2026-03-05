# era/api/routes.py
import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# Optional Redis stream support
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


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
# Config / Redis helpers
# ----------------------------
STREAM_KEY = os.getenv("VITALS_STREAM_KEY", "vitals:events")
DLQ_KEY = os.getenv("VITALS_DLX_KEY", "vitals:dlq")

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _parse_ts(s: Optional[str]) -> datetime:
    if not s:
        return _utcnow()
    try:
        # Accept ISO strings
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return _utcnow()

def _redis_client():
    url = os.getenv("REDIS_URL")
    if not url or redis is None:
        return None
    try:
        return redis.Redis.from_url(url, decode_responses=True)
    except Exception:
        current_app.logger.exception("Redis init failed")
        return None


# ----------------------------
# Lightweight anomaly detection
# (fast rules; worker is source of truth)
# ----------------------------
def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def detect_anomalies(vitals: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Simple, explainable rules for demo + safety.
    """
    alerts: list[Dict[str, Any]] = []

    hr = _as_float(vitals.get("heart_rate"))
    sys = _as_float(vitals.get("systolic_bp"))
    dia = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    # Heart rate
    if hr is not None and hr >= 130:
        alerts.append({"alert_type": "tachycardia", "severity": "warn", "message": f"High heart rate: {hr}"})
    if hr is not None and hr <= 40:
        alerts.append({"alert_type": "bradycardia", "severity": "warn", "message": f"Low heart rate: {hr}"})

    # Blood pressure
    if sys is not None and dia is not None and (sys >= 180 or dia >= 120):
        alerts.append({"alert_type": "hypertensive_crisis", "severity": "high", "message": f"Very high BP: {sys}/{dia}"})
    elif sys is not None and dia is not None and (sys >= 140 or dia >= 90):
        alerts.append({"alert_type": "hypertension", "severity": "warn", "message": f"High BP: {sys}/{dia}"})

    # SpO2
    if spo2 is not None and spo2 < 90:
        alerts.append({"alert_type": "low_spo2", "severity": "high", "message": f"Low SpO2: {spo2}"})
    elif spo2 is not None and spo2 < 92:
        alerts.append({"alert_type": "borderline_spo2", "severity": "warn", "message": f"Borderline SpO2: {spo2}"})

    return alerts


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    # DB check
    db_ok = True
    db_error = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    # Redis check
    r = _redis_client()
    redis_ok = False
    if r:
        try:
            r.ping()
            redis_ok = True
        except Exception:
            redis_ok = False

    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "db_ok": db_ok,
            "db_error": db_error,
            "redis_ok": redis_ok,
            "stream_key": STREAM_KEY,
        }
    ), (200 if db_ok else 503)


@api_bp.post("/vitals")
def ingest_vitals():
    """
    Stream ingestion endpoint:
    - Always returns quickly
    - Publishes to Redis Stream if available
    - If Redis not available, writes to DB directly (still works)
    """
    body = request.get_json(force=True) or {}

    tenant_id = str(body.get("tenant_id") or "demo")
    patient_id = str(body.get("patient_id") or "unknown")
    source = str(body.get("source") or "api")
    vitals = body.get("vitals") or {}
    event_ts = _parse_ts(body.get("event_ts"))
    event_id = str(body.get("event_id") or uuid.uuid4().hex)

    payload = {
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "source": source,
        "event_ts": event_ts.isoformat(),
        "event_id": event_id,
        "vitals": vitals,
        "received_at": _utcnow().isoformat(),
    }

    r = _redis_client()
    if r:
        # Push to stream
        msg_id = r.xadd(STREAM_KEY, {"payload": json.dumps(payload)}, maxlen=200000, approximate=True)
        return jsonify({"ok": True, "accepted": True, "message_id": msg_id, "event_id": event_id}), 202

    # Fallback: write directly to DB + immediate alerts (not ideal but safe)
    alerts = detect_anomalies(vitals)

    db.session.execute(
        text(
            """
            INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, event_id, payload_json)
            VALUES (:t, :p, :s, :ets, :eid, :j::jsonb)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "ets": event_ts,
            "eid": event_id,
            "j": json.dumps(vitals),
        },
    )

    for a in alerts:
        db.session.execute(
            text(
                """
                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, details_json)
                VALUES (:t, :p, :atype, :sev, :msg, :d::jsonb)
                """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "atype": a.get("alert_type", "anomaly"),
                "sev": a.get("severity", "info"),
                "msg": a.get("message", ""),
                "d": json.dumps({"vitals": vitals}),
            },
        )

    # latest snapshot
    db.session.execute(
        text(
            """
            INSERT INTO patient_latest_vitals (tenant_id, patient_id, updated_at, latest_json)
            VALUES (:t, :p, NOW(), :j::jsonb)
            ON CONFLICT (tenant_id, patient_id)
            DO UPDATE SET updated_at = NOW(), latest_json = EXCLUDED.latest_json
            """
        ),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(vitals)},
    )

    db.session.commit()
    return jsonify({"ok": True, "accepted": True, "mode": "db_fallback", "event_id": event_id}), 202


@api_bp.get("/alerts")
def get_alerts():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "unknown")
    limit = int(request.args.get("limit", "50"))

    rows = db.session.execute(
        text(
            """
            SELECT id, tenant_id, patient_id, created_at, alert_type, severity, message, details_json
            FROM alerts
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {"t": tenant_id, "p": patient_id, "lim": limit},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "alerts": list(rows)}), 200


@api_bp.get("/vitals/latest")
def latest_vitals():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "unknown")

    row = db.session.execute(
        text(
            """
            SELECT tenant_id, patient_id, updated_at, latest_json
            FROM patient_latest_vitals
            WHERE tenant_id = :t AND patient_id = :p
            """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    if not row:
        return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "latest": None}), 200

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "updated_at": row["updated_at"], "latest": row["latest_json"]}), 200


@api_bp.get("/rollups/15m")
def rollups_15m():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "unknown")
    since_minutes = int(request.args.get("since_minutes", "180"))  # last 3h default
    limit = int(request.args.get("limit", "200"))

    since_ts = _utcnow() - timedelta(minutes=since_minutes)

    rows = db.session.execute(
        text(
            """
            SELECT *
            FROM vitals_rollups_15m
            WHERE tenant_id = :t AND patient_id = :p
              AND bucket_start >= :since
            ORDER BY bucket_start DESC
            LIMIT :lim
            """
        ),
        {"t": tenant_id, "p": patient_id, "since": since_ts, "lim": limit},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "rollups": list(rows)}), 200


@api_bp.post("/admin/replay-dlq")
def replay_dlq():
    """
    Moves the most recent N DLQ items back into the main stream.
    This is a simple admin tool to recover from transient worker bugs.
    """
    body = request.get_json(force=True) or {}
    n = int(body.get("n", 50))

    r = _redis_client()
    if not r:
        return jsonify({"ok": False, "error": "redis_not_configured"}), 400

    items = r.xrevrange(DLQ_KEY, count=n)
    moved = 0
    for msg_id, fields in items:
        payload = fields.get("payload")
        if payload:
            r.xadd(STREAM_KEY, {"payload": payload}, maxlen=200000, approximate=True)
            moved += 1

    return jsonify({"ok": True, "moved": moved, "from": DLQ_KEY, "to": STREAM_KEY}), 200
