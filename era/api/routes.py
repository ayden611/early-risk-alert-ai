# era/api/routes.py
import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from flask import Blueprint, jsonify, request, current_app, Response
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db
from era.streaming import get_stream_client

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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_tables() -> None:
    # Minimal schema needed for streaming + alerts
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS vitals_events (
              id BIGSERIAL PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              patient_id TEXT NOT NULL,
              source TEXT,
              event_ts TIMESTAMPTZ,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              payload_json JSONB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
              id BIGSERIAL PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              patient_id TEXT NOT NULL,
              alert_type TEXT DEFAULT 'anomaly',
              severity TEXT DEFAULT 'info',
              message TEXT DEFAULT '',
              created_at TIMESTAMPTZ DEFAULT NOW(),
              meta JSONB
            );
            CREATE INDEX IF NOT EXISTS idx_vitals_tenant_patient_created
              ON vitals_events(tenant_id, patient_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
              ON alerts(tenant_id, patient_id, created_at DESC);
            """
        )
    )
    db.session.commit()


@api_bp.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
        db_error = None
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "db_ok": db_ok,
            "db_error": db_error,
            "ts": _utcnow_iso(),
        }
    ), (200 if db_ok else 503)


@api_bp.post("/vitals")
def ingest_vitals():
    _ensure_tables()

    body = request.get_json(force=True) or {}
    tenant_id = str(body.get("tenant_id", "demo"))
    patient_id = str(body.get("patient_id", "unknown"))
    source = str(body.get("source", "manual"))
    vitals = body.get("vitals", {}) or {}
    event_ts = body.get("event_ts")  # optional ISO

    # Save raw event (audit/history)
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
            "ts": event_ts,
            "j": json.dumps(vitals),
        },
    )
    db.session.commit()

    # Publish to stream (for worker processing)
    accepted = True
    message_id = None
    try:
        client = get_stream_client()
        message_id = client.publish_vitals(tenant_id, patient_id, {"vitals": vitals, "event_ts": event_ts, "source": source})
    except Exception as e:
        # If stream is down, still accept (you can switch this to fail-closed if you want)
        current_app.logger.warning(f"stream publish failed: {e}")
        accepted = False

    return jsonify(
        {
            "ok": True,
            "accepted": accepted,
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "message_id": message_id,
        }
    ), 200


@api_bp.get("/alerts")
def list_alerts():
    _ensure_tables()
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "unknown")
    limit = int(request.args.get("limit", "50"))

    rows = db.session.execute(
        text(
            """
            SELECT id, tenant_id, patient_id, alert_type, severity, message, created_at, meta
            FROM alerts
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {"t": tenant_id, "p": patient_id, "lim": limit},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "alerts": [dict(r) for r in rows]}), 200


@api_bp.get("/stream/alerts")
def stream_alerts_sse():
    """
    Server-Sent Events endpoint.
    Browser connects: /api/v1/stream/alerts?tenant_id=demo&patient_id=123
    Requires REDIS_URL. If Redis missing, we fallback to polling DB (works but not realtime-perfect).
    """
    _ensure_tables()
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "unknown")

    def sse_format(obj: Dict[str, Any]) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    # Try Redis pubsub
    try:
        client = get_stream_client()
        redis = client.redis  # type: ignore[attr-defined]
        channel = f"alerts:{tenant_id}:{patient_id}"
        pubsub = redis.pubsub()
        pubsub.subscribe(channel)

        def gen():
            yield "retry: 1500\n\n"
            yield sse_format({"type": "connected", "tenant_id": tenant_id, "patient_id": patient_id, "ts": _utcnow_iso()})
            for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                data = msg.get("data")
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {"raw": data}
                yield sse_format(payload)

        return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as e:
        current_app.logger.warning(f"SSE Redis fallback (polling): {e}")

        # Fallback: poll DB every 2s for newest alerts
        def gen_poll():
            yield "retry: 2000\n\n"
            last_id = 0
            while True:
                rows = db.session.execute(
                    text(
                        """
                        SELECT id, alert_type, severity, message, created_at, meta
                        FROM alerts
                        WHERE tenant_id=:t AND patient_id=:p AND id > :last
                        ORDER BY id ASC
                        LIMIT 50
                        """
                    ),
                    {"t": tenant_id, "p": patient_id, "last": last_id},
                ).mappings().all()

                for r in rows:
                    last_id = max(last_id, int(r["id"]))
                    yield sse_format({"type": "alert", **dict(r)})
                time.sleep(2)

        return Response(gen_poll(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
