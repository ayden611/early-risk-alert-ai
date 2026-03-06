import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

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
    return datetime.now(timezone.utc).isoformat()


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# ----------------------------
# Health
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

    return {
        "status": "ok" if db_ok else "degraded",
        "db_ok": db_ok,
        "db_error": db_error,
        "service": "early-risk-alert-api",
    }, (200 if db_ok else 503)


# ----------------------------
# Ingest vitals
# ----------------------------
@api_bp.post("/vitals")
def ingest_vitals():
    data: Dict[str, Any] = request.get_json(force=True)

    tenant_id = data.get("tenant_id", "demo")
    patient_id = data.get("patient_id")
    source = data.get("source", "unknown")
    event_id = data.get("event_id")
    event_ts = data.get("event_ts")

    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    payload_json = json.dumps(data.get("vitals", data))

    db.session.execute(
        text(
            """
            INSERT INTO vitals_events
                (tenant_id, patient_id, source, event_id, event_ts, payload_json)
            VALUES
                (:t, :p, :s, :e, COALESCE(CAST(:ts AS timestamptz), NOW()), CAST(:payload AS jsonb))
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "e": event_id,
            "ts": event_ts,
            "payload": payload_json,
        },
    )

    db.session.commit()

    return jsonify(
        {
            "accepted": True,
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "received_at": _utcnow_iso(),
        }
    ), 202


# ----------------------------
# Alerts list
# ----------------------------
@api_bp.get("/alerts")
def get_alerts():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")
    limit = int(request.args.get("limit", 50))

    if patient_id:
        rows = db.session.execute(
            text(
                """
                SELECT id, patient_id, alert_type, severity, status, message, created_at
                FROM alerts
                WHERE tenant_id = :t AND patient_id = :p
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"t": tenant_id, "p": patient_id, "lim": limit},
        ).mappings()
    else:
        rows = db.session.execute(
            text(
                """
                SELECT id, patient_id, alert_type, severity, status, message, created_at
                FROM alerts
                WHERE tenant_id = :t
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"t": tenant_id, "lim": limit},
        ).mappings()

    return jsonify({"alerts": [dict(r) for r in rows]}), 200


# ----------------------------
# Patient rollups
# ----------------------------
@api_bp.get("/patients/<patient_id>/rollups")
def patient_rollups(patient_id: str):
    tenant_id = request.args.get("tenant_id", "demo")

    rows = db.session.execute(
        text(
            """
            SELECT
                AVG((payload_json->>'heart_rate')::float) AS avg_heart_rate,
                AVG((payload_json->>'systolic_bp')::float) AS avg_systolic_bp,
                AVG((payload_json->>'diastolic_bp')::float) AS avg_diastolic_bp,
                AVG((payload_json->>'spo2')::float) AS avg_spo2
            FROM vitals_events
            WHERE tenant_id = :t AND patient_id = :p
            """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "rollups": dict(rows) if rows else {},
        }
    ), 200


# ----------------------------
# Stream Health
# ----------------------------
@api_bp.get("/stream/health")
def stream_health():
    try:
        import os
        import redis as redis_lib
    except Exception:
        redis_lib = None

    redis_ok = False
    if redis_lib is not None:
        url = __import__("os").getenv("REDIS_URL")
        if url:
            try:
                client = redis_lib.Redis.from_url(url, decode_responses=True)
                redis_ok = bool(client.ping())
            except Exception:
                redis_ok = False

    return jsonify(
        {
            "status": "ok",
            "redis_ok": redis_ok,
            "mode": "pubsub" if redis_ok else "polling_fallback",
        }
    ), 200


# ----------------------------
# Stream Channels
# ----------------------------
@api_bp.get("/stream/channels")
def stream_channels():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")

    channels = [
        "stream:vitals",
        f"stream:vitals:{tenant_id}",
        "stream:alerts",
        f"stream:alerts:{tenant_id}",
    ]

    if patient_id:
        channels.extend(
            [
                f"stream:vitals:{tenant_id}:{patient_id}",
                f"stream:alerts:{tenant_id}:{patient_id}",
            ]
        )

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "channels": channels,
        }
    ), 200
