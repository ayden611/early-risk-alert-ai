# era/api/routes.py
import json
import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# Optional Redis Streams (queue)
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


api_bp = Blueprint("api", __name__)


# ----------------------------
# JSON error handler
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


def _as_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _redis_client():
    """
    Returns a redis client if REDIS_URL is set and redis lib is available.
    Never raises on failure (keeps API up even if redis is down).
    """
    url = os.getenv("REDIS_URL", "").strip()
    if not url or redis is None:
        return None
    try:
        r = redis.Redis.from_url(url, decode_responses=True)
        # lightweight ping
        r.ping()
        return r
    except Exception:
        current_app.logger.warning("Redis not available; continuing without queue.")
        return None


def _partition_for_patient(patient_id: str, shards: int) -> int:
    """
    Stable partition id based on patient_id.
    """
    if shards <= 1:
        return 0
    h = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % shards


def _stream_key(tenant_id: str, shard: int) -> str:
    # One stream per tenant+shard gives you horizontal scaling without hot keys.
    return f"vitals:{tenant_id}:shard:{shard}"


def _ensure_tables():
    """
    Minimal schema (no migrations tool required).
    Safe to call multiple times.
    """
    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS vitals_events (
          id BIGSERIAL PRIMARY KEY,
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'manual',
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
          details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
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
          cnt INTEGER NOT NULL DEFAULT 0,
          avg_hr DOUBLE PRECISION NULL,
          min_spo2 DOUBLE PRECISION NULL,
          max_systolic DOUBLE PRECISION NULL,
          max_diastolic DOUBLE PRECISION NULL,
          last_event_ts TIMESTAMPTZ NULL,
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


def _detect_anomalies(vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lightweight rule-based anomalies (fast, cheap, safe).
    You can replace with model inference later.
    """
    alerts: List[Dict[str, Any]] = []

    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    # Example thresholds (adjust later)
    if hr is not None and (hr < 45 or hr > 120):
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn" if hr <= 40 or hr >= 130 else "info",
                "message": f"Abnormal heart rate: {hr}",
                "details": {"metric": "heart_rate", "value": hr},
            }
        )

    if sys_bp is not None and sys_bp >= 160:
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn",
                "message": f"High systolic BP: {sys_bp}",
                "details": {"metric": "systolic_bp", "value": sys_bp},
            }
        )

    if dia_bp is not None and dia_bp >= 100:
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn",
                "message": f"High diastolic BP: {dia_bp}",
                "details": {"metric": "diastolic_bp", "value": dia_bp},
            }
        )

    if spo2 is not None and spo2 <= 92:
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn" if spo2 <= 90 else "info",
                "message": f"Low SpO2: {spo2}",
                "details": {"metric": "spo2", "value": spo2},
            }
        )

    return alerts


def _parse_event_ts(event_ts: Optional[str]) -> Optional[datetime]:
    if not event_ts:
        return None
    try:
        # ISO-8601 string expected
        dt = datetime.fromisoformat(event_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    db_ok = True
    db_error = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return (
        jsonify(
            {
                "status": "ok" if db_ok else "degraded",
                "db_ok": db_ok,
                "db_error": db_error,
                "time": _utcnow_iso(),
            }
        ),
        (200 if db_ok else 503),
    )


@api_bp.post("/vitals")
def ingest_vitals():
    """
    Ingest vitals:
      1) persist raw event in Postgres (durable)
      2) push to Redis Streams (real-time queue) if available
      3) (optional) run ultra-fast anomaly detection inline (cheap)
    """
    _ensure_tables()

    body = request.get_json(force=True, silent=True) or {}

    tenant_id = str(body.get("tenant_id") or "demo")
    patient_id = str(body.get("patient_id") or "unknown")
    source = str(body.get("source") or "manual")
    vitals = body.get("vitals") or {}
    event_ts = body.get("event_ts")  # optional iso string
    event_dt = _parse_event_ts(event_ts)

    if not isinstance(vitals, dict) or not patient_id:
        return jsonify({"ok": False, "error": "invalid payload"}), 400

    # Persist raw event
    db.session.execute(
        text(
            """
            INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, payload_json)
            VALUES (:t, :p, :s, :ts, :j::jsonb)
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "ts": event_dt,
            "j": json.dumps(vitals),
        },
    )

    # Inline quick anomalies (optional) -> still also processed by worker later
    inline_alerts = _detect_anomalies(vitals)
    for a in inline_alerts:
        db.session.execute(
            text(
                """
                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, details_json)
                VALUES (:t, :p, :at, :sev, :msg, :d::jsonb)
                """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "at": a["alert_type"],
                "sev": a["severity"],
                "msg": a["message"],
                "d": json.dumps(a.get("details") or {}),
            },
        )

    db.session.commit()

    # Push to Redis stream (partitioned)
    r = _redis_client()
    message_id = None
    shards = _env_int("STREAM_SHARDS", 8)
    if r is not None:
        try:
            shard = _partition_for_patient(patient_id, shards)
            skey = _stream_key(tenant_id, shard)
            message_id = r.xadd(
                skey,
                {
                    "tenant_id": tenant_id,
                    "patient_id": patient_id,
                    "source": source,
                    "event_ts": event_ts or "",
                    "vitals_json": json.dumps(vitals),
                },
                maxlen=_env_int("STREAM_MAXLEN", 200_000),
                approximate=True,
            )
        except Exception:
            current_app.logger.exception("Failed to publish to Redis stream")

    return (
        jsonify(
            {
                "ok": True,
                "accepted": True,
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "message_id": message_id,
                "inline_alerts": len(inline_alerts),
            }
        ),
        200,
    )


@api_bp.get("/alerts")
def get_alerts():
    _ensure_tables()

    tenant_id = str(request.args.get("tenant_id") or "demo")
    patient_id = str(request.args.get("patient_id") or "")
    limit = _env_int("limit", 50)
    limit = min(max(limit, 1), 200)

    params = {"t": tenant_id, "lim": limit}
    where = "WHERE tenant_id = :t"
    if patient_id:
        where += " AND patient_id = :p"
        params["p"] = patient_id

    rows = db.session.execute(
        text(
            f"""
            SELECT tenant_id, patient_id, alert_type, severity, message, details_json, created_at
            FROM alerts
            {where}
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        params,
    ).mappings().all()

    alerts = []
    for r in rows:
        alerts.append(
            {
                "tenant_id": r["tenant_id"],
                "patient_id": r["patient_id"],
                "alert_type": r["alert_type"],
                "severity": r["severity"],
                "message": r["message"],
                "details": r["details_json"] or {},
                "created_at": (r["created_at"].isoformat() if r["created_at"] else None),
            }
        )

    return jsonify({"ok": True, "tenant_id": tenant_id, "patient_id": patient_id, "alerts": alerts}), 200


@api_bp.get("/rollups/15m")
def get_rollups_15m():
    _ensure_tables()

    tenant_id = str(request.args.get("tenant_id") or "demo")
    patient_id = str(request.args.get("patient_id") or "")
    limit = _env_int("limit", 50)
    limit = min(max(limit, 1), 200)

    params = {"t": tenant_id, "lim": limit}
    where = "WHERE tenant_id = :t"
    if patient_id:
        where += " AND patient_id = :p"
        params["p"] = patient_id

    rows = db.session.execute(
        text(
            f"""
            SELECT tenant_id, patient_id, bucket_start, cnt, avg_hr, min_spo2, max_systolic, max_diastolic, last_event_ts, updated_at
            FROM vitals_rollups_15m
            {where}
            ORDER BY bucket_start DESC
            LIMIT :lim
            """
        ),
        params,
    ).mappings().all()

    out = []
    for r in rows:
        out.append(
            {
                "tenant_id": r["tenant_id"],
                "patient_id": r["patient_id"],
                "bucket_start": r["bucket_start"].isoformat(),
                "cnt": r["cnt"],
                "avg_hr": r["avg_hr"],
                "min_spo2": r["min_spo2"],
                "max_systolic": r["max_systolic"],
                "max_diastolic": r["max_diastolic"],
                "last_event_ts": (r["last_event_ts"].isoformat() if r["last_event_ts"] else None),
                "updated_at": r["updated_at"].isoformat(),
            }
        )

    return jsonify({"ok": True, "rollups": out}), 200
