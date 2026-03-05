# era/api/routes.py
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# Optional: Redis stream support (won't crash if redis isn't installed)
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# JSON Error handler
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
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _ensure_tables() -> None:
    """
    Creates tables used by streaming/monitoring if they don't exist.
    Safe to call on every request (but we keep it lightweight).
    """
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS vitals_events (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                source TEXT DEFAULT 'api',
                event_ts TIMESTAMPTZ NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )

    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                alert_type TEXT NOT NULL DEFAULT 'anomaly',
                severity TEXT NOT NULL DEFAULT 'info',
                message TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )

    # Indexes (Postgres supports IF NOT EXISTS)
    db.session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_vitals_events_tenant_patient_created
            ON vitals_events (tenant_id, patient_id, created_at DESC);
            """
        )
    )
    db.session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
            ON alerts (tenant_id, patient_id, created_at DESC);
            """
        )
    )
    db.session.commit()


# ----------------------------
# Redis stream (partitioned) — optional
# ----------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
STREAM_NAME_PREFIX = os.getenv("STREAM_NAME_PREFIX", "vitals")
STREAM_SHARDS = int(os.getenv("STREAM_SHARDS", "16"))
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "200000"))

_redis_client = None


def _redis() -> Optional["redis.Redis"]:
    global _redis_client
    if not REDIS_URL or redis is None:
        return None
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _shard_for(tenant_id: str, patient_id: str) -> int:
    # Stable, cheap shard: hash modulo
    s = f"{tenant_id}:{patient_id}"
    return (hash(s) % STREAM_SHARDS + STREAM_SHARDS) % STREAM_SHARDS


def _stream_name(tenant_id: str, patient_id: str) -> str:
    shard = _shard_for(tenant_id, patient_id)
    return f"{STREAM_NAME_PREFIX}:{shard:03d}"


def publish_to_stream(tenant_id: str, patient_id: str, vitals: Dict[str, Any], source: str, event_ts: Optional[str]):
    r = _redis()
    if r is None:
        return {"streaming": False, "message_id": None, "stream": None}

    sname = _stream_name(tenant_id, patient_id)
    payload = {
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "source": source,
        "event_ts": event_ts or "",
        "vitals_json": json.dumps(vitals, separators=(",", ":")),
        "published_at": str(time.time()),
    }
    msg_id = r.xadd(sname, payload, maxlen=STREAM_MAXLEN, approximate=True)
    return {"streaming": True, "message_id": msg_id, "stream": sname}


# ----------------------------
# Anomaly detection (fast rules)
# ----------------------------
def detect_anomalies(vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    hr = _as_float(vitals.get("heart_rate"))
    sys = _as_float(vitals.get("systolic_bp"))
    dia = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    if hr is not None and (hr >= 120 or hr <= 40):
        alerts.append({"alert_type": "anomaly", "severity": "warn", "message": f"Heart rate out of range: {hr}"})

    if sys is not None and sys >= 160:
        alerts.append({"alert_type": "anomaly", "severity": "warn", "message": f"Systolic BP high: {sys}"})

    if dia is not None and dia >= 100:
        alerts.append({"alert_type": "anomaly", "severity": "warn", "message": f"Diastolic BP high: {dia}"})

    if spo2 is not None and spo2 <= 92:
        alerts.append({"alert_type": "anomaly", "severity": "critical", "message": f"SpO2 low: {spo2}"})

    return alerts


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        return jsonify({"status": "degraded", "db_ok": False, "db_error": str(e)}), 503

    # Try to ensure tables; if it fails, still return health but note it.
    tables_ok = True
    tables_error = None
    try:
        _ensure_tables()
    except Exception as e:
        tables_ok = False
        tables_error = str(e)

    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "db_ok": db_ok,
            "tables_ok": tables_ok,
            "tables_error": tables_error,
            "utc": _utcnow_iso(),
        }
    ), (200 if db_ok else 503)


@api_bp.post("/vitals")
def ingest_vitals():
    """
    Device/app posts vitals here.
    - Stores raw event in DB
    - Runs fast anomaly detection (and writes alerts)
    - Optionally publishes to Redis Streams if REDIS_URL is configured
    """
    _ensure_tables()

    body = request.get_json(force=True) or {}
    tenant_id = str(body.get("tenant_id", "demo"))
    patient_id = str(body.get("patient_id", "unknown"))
    source = str(body.get("source", "api"))
    vitals = body.get("vitals") or {}
    event_ts_raw = body.get("event_ts")  # optional ISO string

    # Store event in DB
    db.session.execute(
        text(
            """
            INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, payload_json)
            VALUES (:t, :p, :s, :ets, :j::jsonb)
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "ets": event_ts_raw,  # Postgres will parse ISO string if valid; null ok
            "j": json.dumps(vitals),
        },
    )

    # Detect anomalies now (quick) and write alerts
    alerts = detect_anomalies(vitals)
    for a in alerts:
        db.session.execute(
            text(
                """
                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, created_at)
                VALUES (:t, :p, :at, :sev, :msg, NOW())
                """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "at": a.get("alert_type", "anomaly"),
                "sev": a.get("severity", "info"),
                "msg": a.get("message", ""),
            },
        )

    db.session.commit()

    # Optional stream publish (won't fail deploy if Redis isn't set)
    stream_ack = publish_to_stream(
        tenant_id=tenant_id,
        patient_id=patient_id,
        vitals=vitals,
        source=source,
        event_ts=event_ts_raw,
    )

    return jsonify(
        {
            "ok": True,
            "accepted": True,
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "alerts_written": len(alerts),
            **stream_ack,
        }
    ), 200
@api
