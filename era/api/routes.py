# era/api/routes.py
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# Optional Redis stream support (won't crash if redis isn't installed)
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
# Helpers
# ----------------------------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept "Z"
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _redis_client():
    url = os.getenv("REDIS_URL")
    if not url or redis is None:
        return None
    # Render Redis URLs are usually rediss:// (TLS) or redis://
    return redis.Redis.from_url(url, decode_responses=True)


def _stream_key(tenant_id: str) -> str:
    # One stream per tenant (keeps it scalable + easy to shard later)
    return f"vitals_stream:{tenant_id}"


def _partition_key(tenant_id: str, patient_id: str, partitions: int = 128) -> int:
    # Stable partition: tenant+patient -> partition number
    s = f"{tenant_id}:{patient_id}"
    return (abs(hash(s)) % partitions)


def _ensure_tables():
    """
    Creates tables if they do not exist.
    Safe to run on every request (IF NOT EXISTS).
    """
    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS vitals_events (
            id BIGSERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            source TEXT DEFAULT 'unknown',
            event_ts TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            payload_json JSONB NOT NULL
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
            alert_type TEXT DEFAULT 'anomaly',
            severity TEXT DEFAULT 'info',
            message TEXT DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        )
    )

    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS vitals_rollups_15m (
            tenant_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            bucket_start TIMESTAMPTZ NOT NULL,
            count_events INTEGER NOT NULL DEFAULT 0,
            avg_hr DOUBLE PRECISION,
            min_spo2 DOUBLE PRECISION,
            max_sys DOUBLE PRECISION,
            max_dia DOUBLE PRECISION,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, patient_id, bucket_start)
        );
        """
        )
    )

    # Helpful indexes
    db.session.execute(
        text(
            """
        CREATE INDEX IF NOT EXISTS idx_vitals_events_tenant_patient_created
        ON vitals_events(tenant_id, patient_id, created_at DESC);
        """
        )
    )
    db.session.execute(
        text(
            """
        CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
        ON alerts(tenant_id, patient_id, created_at DESC);
        """
        )
    )

    db.session.commit()


def detect_anomalies(vitals: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Lightweight rules-based anomaly detection (fast).
    You can later replace with ML model inference.
    """
    alerts: List[Dict[str, str]] = []

    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    if hr is not None and hr >= 130:
        alerts.append({"severity": "warn", "message": f"High heart rate: {hr:.0f}"})
    if sys_bp is not None and sys_bp >= 160:
        alerts.append({"severity": "warn", "message": f"High systolic BP: {sys_bp:.0f}"})
    if dia_bp is not None and dia_bp >= 100:
        alerts.append({"severity": "warn", "message": f"High diastolic BP: {dia_bp:.0f}"})
    if spo2 is not None and spo2 <= 92:
        alerts.append({"severity": "warn", "message": f"Low SpO2: {spo2:.0f}"})

    return alerts


# ----------------------------
# Health
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    db_ok = True
    db_err = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_err = str(e)

    r_ok = False
    r_err = None
    try:
        r = _redis_client()
        if r is not None:
            r.ping()
            r_ok = True
    except Exception as e:
        r_ok = False
        r_err = str(e)

    return (
        jsonify(
            {
                "status": "ok" if db_ok else "degraded",
                "db_ok": db_ok,
                "db_error": db_err,
                "redis_ok": r_ok,
                "redis_error": r_err,
                "time_utc": _utcnow_iso(),
            }
        ),
        (200 if db_ok else 503),
    )


# ----------------------------
# Stream ingestion endpoint
# ----------------------------
@api_bp.post("/v1/vitals")
def ingest_vitals():
    """
    Device/app posts vitals here.
    We always write raw event to Postgres.
    If Redis is configured, we ALSO publish to a tenant stream for worker processing.

    Payload example:
    {
      "tenant_id":"demo",
      "patient_id":"123",
      "source":"watch",
      "event_ts":"2026-03-05T17:00:00Z",
      "event_id":"evt-123-0001",
      "vitals":{"heart_rate":132,"systolic_bp":165,"diastolic_bp":100,"spo2":91}
    }
    """
    _ensure_tables()

    body = request.get_json(force=True) or {}
    tenant_id = str(body.get("tenant_id") or "demo")
    patient_id = str(body.get("patient_id") or "unknown")
    source = str(body.get("source") or "unknown")
    event_ts = _parse_iso(body.get("event_ts"))
    event_id = str(body.get("event_id") or "")

    vitals = body.get("vitals") or {}
    if not isinstance(vitals, dict):
        return jsonify({"error": "bad_request", "message": "vitals must be an object"}), 400

    # Save raw event
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
            "ets": event_ts,
            "j": json.dumps(
                {
                    "tenant_id": tenant_id,
                    "patient_id": patient_id,
                    "source": source,
                    "event_ts": body.get("event_ts"),
                    "event_id": event_id,
                    "vitals": vitals,
                }
            ),
        },
    )
    db.session.commit()

    # Publish to Redis stream (queue) for scalable processing
    message_id = None
    pushed = False
    try:
        r = _redis_client()
        if r is not None:
            partitions = int(os.getenv("STREAM_PARTITIONS", "128"))
            part = _partition_key(tenant_id, patient_id, partitions=partitions)
            stream = _stream_key(tenant_id)

            message = {
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "source": source,
                "event_ts": body.get("event_ts") or "",
                "event_id": event_id,
                "partition": str(part),
                "payload": json.dumps(vitals),
            }
            message_id = r.xadd(stream, message, maxlen=100000, approximate=True)
            pushed = True
    except Exception as e:
        current_app.logger.warning(f"Redis publish failed: {e}")

    # Optional immediate detection (tiny) so you see alerts even if worker lags
    # You can set INLINE_DETECT=0 to disable
    inline_detect = os.getenv("INLINE_DETECT", "1") == "1"
    inline_alerts = []
    if inline_detect:
        for a in detect_anomalies(vitals):
            inline_alerts.append(a)
            db.session.execute(
                text(
                    """
                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message)
                VALUES (:t,:p,'anomaly',:sev,:msg)
                """
                ),
                {"t": tenant_id, "p": patient_id, "sev": a["severity"], "msg": a["message"]},
            )
        db.session.commit()

    return (
        jsonify(
            {
                "ok": True,
                "accepted": True,
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "pushed_to_queue": pushed,
                "message_id": message_id,
                "inline_alerts": inline_alerts,
            }
        ),
        200,
    )


# ----------------------------
# Read endpoints
# ----------------------------
@api_bp.get("/v1/alerts")
def get_alerts():
    _ensure_tables()

    tenant_id = str(request.args.get("tenant_id") or "demo")
    patient_id = str(request.args.get("patient_id") or "unknown")
    limit = int(request.args.get("limit") or 50)

    rows = db.session.execute(
        text(
            """
        SELECT id, alert_type, severity, message, created_at
        FROM alerts
        WHERE tenant_id=:t AND patient_id=:p
        ORDER BY created_at DESC
        LIMIT :lim
        """
        ),
        {"t": tenant_id, "p": patient_id, "lim": limit},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "alerts": list(rows)}), 200


@api_bp.get("/v1/vitals/latest")
def latest_vitals():
    _ensure_tables()

    tenant_id = str(request.args.get("tenant_id") or "demo")
    patient_id = str(request.args.get("patient_id") or "unknown")

    row = db.session.execute(
        text(
            """
        SELECT payload_json, created_at
        FROM vitals_events
        WHERE tenant_id=:t AND patient_id=:p
        ORDER BY created_at DESC
        LIMIT 1
        """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    if not row:
        return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "latest": None}), 200

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "latest": row}), 200


@api_bp.get("/v1/rollups/15m")
def rollups_15m():
    _ensure_tables()

    tenant_id = str(request.args.get("tenant_id") or "demo")
    patient_id = str(request.args.get("patient_id") or "unknown")
    since_minutes = int(request.args.get("since_minutes") or 120)

    rows = db.session.execute(
        text(
            """
        SELECT bucket_start, count_events, avg_hr, min_spo2, max_sys, max_dia, updated_at
        FROM vitals_rollups_15m
        WHERE tenant_id=:t AND patient_id=:p
          AND bucket_start >= NOW() - (:mins || ' minutes')::interval
        ORDER BY bucket_start DESC
        LIMIT 200
        """
        ),
        {"t": tenant_id, "p": patient_id, "mins": since_minutes},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "rollups": list(rows)}), 200
