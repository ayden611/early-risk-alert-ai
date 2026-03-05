# era/api/routes.py
import json
import time
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import HTTPException
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from era.ai.anomaly import detect_anomalies
from era.ai.reasoning import risk_score_snapshot

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
# Health check
# IMPORTANT: create_app already prefixes /api/v1
# so this becomes /api/v1/healthz
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
                "ts": int(time.time()),
            }
        ),
        (200 if db_ok else 503),
    )


# ----------------------------
# Create table if missing
# ----------------------------
def _ensure_risk_jobs_table():
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS risk_jobs (
              id BIGSERIAL PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              patient_id TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              started_at TIMESTAMPTZ,
              finished_at TIMESTAMPTZ,
              result_json TEXT,
              error TEXT
            );
            """
        )
    )
    db.session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_risk_jobs_status_id
            ON risk_jobs(status, id);
            """
        )
    )
    db.session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_risk_jobs_tenant_patient
            ON risk_jobs(tenant_id, patient_id);
            """
        )
    )
    db.session.commit()


# ----------------------------
# Ingest an event (creates a job row)
# POST /api/v1/events
# ----------------------------
@api_bp.post("/events")
def ingest_event():
    _ensure_risk_jobs_table()

    body: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    tenant_id = str(body.get("tenant_id") or body.get("tenant") or "demo-tenant")
    patient_id = str(body.get("patient_id") or "unknown")
    payload = body.get("payload") or {}

    # store the event as a "job" row (worker can process later)
    row = db.session.execute(
        text(
            """
            INSERT INTO risk_jobs (tenant_id, patient_id, payload_json, status)
            VALUES (:t, :p, :j, 'pending')
            RETURNING id;
            """
        ),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(payload)},
    ).fetchone()

    db.session.commit()

    job_id = int(row[0]) if row else None
    return jsonify({"ok": True, "tenant_id": tenant_id, "patient_id": patient_id, "job_id": job_id}), 201


# ----------------------------
# List patients seen for a tenant
# GET /api/v1/clinician/patients?tenant_id=demo-tenant
# ----------------------------
@api_bp.get("/clinician/patients")
def clinician_patients():
    _ensure_risk_jobs_table()

    tenant_id = request.args.get("tenant_id", "demo-tenant")

    rows = db.session.execute(
        text(
            """
            SELECT patient_id, COUNT(*) AS events
            FROM risk_jobs
            WHERE tenant_id = :t
            GROUP BY patient_id
            ORDER BY events DESC
            LIMIT 500;
            """
        ),
        {"t": tenant_id},
    ).fetchall()

    return jsonify(
        {
            "tenant_id": tenant_id,
            "count": len(rows),
            "patients": [{"patient_id": r[0], "events": int(r[1])} for r in rows],
        }
    )


# ----------------------------
# Get job status
# GET /api/v1/jobs/<id>
# ----------------------------
@api_bp.get("/jobs/<int:job_id>")
def get_job(job_id: int):
    _ensure_risk_jobs_table()

    row = db.session.execute(
        text(
            """
            SELECT id, tenant_id, patient_id, status, created_at, started_at, finished_at, result_json, error
            FROM risk_jobs
            WHERE id = :id
            """
        ),
        {"id": job_id},
    ).fetchone()

    if not row:
        return jsonify({"error": "not_found", "job_id": job_id}), 404

    return jsonify(
        {
            "id": int(row[0]),
            "tenant_id": row[1],
            "patient_id": row[2],
            "status": row[3],
            "created_at": str(row[4]) if row[4] else None,
            "started_at": str(row[5]) if row[5] else None,
            "finished_at": str(row[6]) if row[6] else None,
            "result": json.loads(row[7]) if row[7] else None,
            "error": row[8],
        }
    )

@api_bp.post("/vitals")
def ingest_vitals():
    """
    Ingests a vital payload and immediately runs lightweight anomaly detection.
    This is the "stream ingestion" endpoint (devices/apps post here).
    """
    body = request.get_json(force=True) or {}
    tenant_id = body.get("tenant_id", "demo-tenant")
    patient_id = str(body.get("patient_id", "unknown"))
    source = body.get("source", "manual")
    vitals = body.get("vitals", {}) or {}
    event_ts = body.get("event_ts")  # optional ISO string

    # Save raw vitals event
    db.session.execute(
        text("""
            INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, payload_json)
            VALUES (:t, :p, :s, :ts, :j::jsonb)
        """),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "ts": event_ts,
            "j": json.dumps(vitals),
        },
    )

    # Run anomaly detection now (fast) and write alerts
    alerts = detect_anomalies(vitals)
    for a in alerts:
        db.session.execute(
            text("""
                INSERT INTO alerts (tenant_id, patient_id, severity, kind, title, details)
                VALUES (:t, :p, :sev, :kind, :title, :details::jsonb)
            """),
            {
                "t": tenant_id,
                "p": patient_id,
                "sev": a.severity,
                "kind": a.kind,
                "title": a.title,
                "details": json.dumps(a.details),
            },
        )

    db.session.commit()

    return jsonify(
        {
            "ok": True,
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "alerts_written": len(alerts),
        }
    ), 200


@api_bp.get("/patients/latest")
def patient_latest():
    """
    Returns latest vitals + last N alerts for a patient.
    """
    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id", "unknown")

    latest = db.session.execute(
        text("""
            SELECT received_at, payload_json
            FROM vitals_events
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY received_at DESC
            LIMIT 1
        """),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    alerts = db.session.execute(
        text("""
            SELECT created_at, severity, kind, title, details
            FROM alerts
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY created_at DESC
            LIMIT 25
        """),
        {"t": tenant_id, "p": patient_id},
    ).mappings().all()

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "latest_vitals": latest["payload_json"] if latest else None,
            "latest_vitals_at": latest["received_at"].isoformat() if latest else None,
            "alerts": [
                {
                    "created_at": a["created_at"].isoformat(),
                    "severity": a["severity"],
                    "kind": a["kind"],
                    "title": a["title"],
                    "details": a["details"],
                }
                for a in alerts
            ],
        }
    ), 200


@api_bp.get("/ai/reasoning")
def ai_reasoning():
    """
    AI reasoning engine v1:
      - pulls latest vitals
      - looks at recent alerts
      - outputs risk score + clinician-facing notes
    """
    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id", "unknown")

    latest = db.session.execute(
        text("""
            SELECT payload_json
            FROM vitals_events
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY received_at DESC
            LIMIT 1
        """),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_alerts = db.session.execute(
        text("""
            SELECT severity, kind, title, details
            FROM alerts
            WHERE tenant_id=:t AND patient_id=:p AND created_at >= :since
            ORDER BY created_at DESC
            LIMIT 50
        """),
        {"t": tenant_id, "p": patient_id, "since": since},
    ).mappings().all()

    latest_vitals = latest["payload_json"] if latest else {}
    snapshot = risk_score_snapshot(latest_vitals, [dict(r) for r in recent_alerts])

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "latest_vitals": latest_vitals,
            "ai_analysis": snapshot,
        }
    ), 200
    @api_bp.post("/vitals")
    def ingest_vitals():
    """
    Ingests a vital payload and immediately runs lightweight anomaly detection.
    This is the "stream ingestion" endpoint (devices/apps post here).
    """
    body = request.get_json(force=True) or {}
    tenant_id = body.get("tenant_id", "demo-tenant")
    patient_id = str(body.get("patient_id", "unknown"))
    source = body.get("source", "manual")
    vitals = body.get("vitals", {}) or {}
    event_ts = body.get("event_ts")  # optional ISO string

    # Save raw vitals event
    db.session.execute(
        text("""
            INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, payload_json)
            VALUES (:t, :p, :s, :ts, :j::jsonb)
        """),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "ts": event_ts,
            "j": json.dumps(vitals),
        },
    )

    # Run anomaly detection now (fast) and write alerts
    alerts = detect_anomalies(vitals)
    for a in alerts:
        db.session.execute(
            text("""
                INSERT INTO alerts (tenant_id, patient_id, severity, kind, title, details)
                VALUES (:t, :p, :sev, :kind, :title, :details::jsonb)
            """),
            {
                "t": tenant_id,
                "p": patient_id,
                "sev": a.severity,
                "kind": a.kind,
                "title": a.title,
                "details": json.dumps(a.details),
            },
        )

    db.session.commit()

    return jsonify(
        {
            "ok": True,
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "alerts_written": len(alerts),
        }
    ), 200


@api_bp.get("/patients/latest")
def patient_latest():
    """
    Returns latest vitals + last N alerts for a patient.
    """
    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id", "unknown")

    latest = db.session.execute(
        text("""
            SELECT received_at, payload_json
            FROM vitals_events
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY received_at DESC
            LIMIT 1
        """),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    alerts = db.session.execute(
        text("""
            SELECT created_at, severity, kind, title, details
            FROM alerts
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY created_at DESC
            LIMIT 25
        """),
        {"t": tenant_id, "p": patient_id},
    ).mappings().all()

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "latest_vitals": latest["payload_json"] if latest else None,
            "latest_vitals_at": latest["received_at"].isoformat() if latest else None,
            "alerts": [
                {
                    "created_at": a["created_at"].isoformat(),
                    "severity": a["severity"],
                    "kind": a["kind"],
                    "title": a["title"],
                    "details": a["details"],
                }
                for a in alerts
            ],
        }
    ), 200


@api_bp.get("/ai/reasoning")
def ai_reasoning():
    """
    AI reasoning engine v1:
      - pulls latest vitals
      - looks at recent alerts
      - outputs risk score + clinician-facing notes
    """
    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id", "unknown")

    latest = db.session.execute(
        text("""
            SELECT payload_json
            FROM vitals_events
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY received_at DESC
            LIMIT 1
        """),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_alerts = db.session.execute(
        text("""
            SELECT severity, kind, title, details
            FROM alerts
            WHERE tenant_id=:t AND patient_id=:p AND created_at >= :since
            ORDER BY created_at DESC
            LIMIT 50
        """),
        {"t": tenant_id, "p": patient_id, "since": since},
    ).mappings().all()

    latest_vitals = latest["payload_json"] if latest else {}
    snapshot = risk_score_snapshot(latest_vitals, [dict(r) for r in recent_alerts])

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "latest_vitals": latest_vitals,
            "ai_analysis": snapshot,
        }
    ), 200


@api_bp.get("/stream/alerts")
def stream_alerts():
    """
    Simple server-sent events (SSE) stream of alerts.
    This is a stepping stone toward WebSockets + pubsub for scale.
    """
    tenant_id = request.args.get("tenant_id", "demo-tenant")
    after_id = int(request.args.get("after_id", "0"))

    def gen():
        nonlocal after_id
        # NOTE: polling loop; good for demos, not for 1M scale
        while True:
            rows = db.session.execute(
                text("""
                    SELECT id, created_at, patient_id, severity, kind, title, details
                    FROM alerts
                    WHERE tenant_id=:t AND id > :after
                    ORDER BY id ASC
                    LIMIT 100
                """),
                {"t": tenant_id, "after": after_id},
            ).mappings().all()

            for r in rows:
                after_id = int(r["id"])
                payload = {
                    "id": int(r["id"]),
                    "created_at": r["created_at"].isoformat(),
                    "patient_id": r["patient_id"],
                    "severity": r["severity"],
                    "kind": r["kind"],
                    "title": r["title"],
                    "details": r["details"],
                }
                yield f"event: alert\ndata: {json.dumps(payload)}\n\n"

            time.sleep(2)

    return Response(gen(), mimetype="text/event-stream")


