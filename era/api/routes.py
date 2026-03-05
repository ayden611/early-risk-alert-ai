# era/api/routes.py
import json
import time
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, request
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
PY
