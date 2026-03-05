# era/api/routes.py
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

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
# Health check
# ----------------------------
@api_bp.get("/api/v1/healthz")
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
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ),
        200 if db_ok else 503,
    )


# ----------------------------
# Event ingest (creates a "job" row)
# ----------------------------
@api_bp.post("/api/v1/events")
def ingest_event():
    body = request.get_json(force=True) or {}

    tenant_id = str(body.get("tenant_id", "demo-tenant"))
    patient_id = str(body.get("patient_id", "unknown"))
    payload = body.get("payload", {})

    if not isinstance(payload, dict):
        return jsonify({"error": "payload must be an object"}), 400

    # minimal DB insert (risk_jobs table must exist)
    db.session.execute(
        text(
            """
            INSERT INTO risk_jobs (tenant_id, patient_id, payload_json, status, created_at)
            VALUES (:t, :p, :j, 'pending', NOW())
            """
        ),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(payload)},
    )
    db.session.commit()

    return jsonify({"ok": True, "tenant_id": tenant_id, "patient_id": patient_id}), 200


# ----------------------------
# Simple clinician view (latest patients + statuses)
# ----------------------------
@api_bp.get("/api/v1/clinician/patients")
def clinician_patients():
    tenant_id = request.args.get("tenant_id", "demo-tenant")

    rows = db.session.execute(
        text(
            """
            SELECT tenant_id, patient_id, status, created_at, started_at, finished_at
            FROM risk_jobs
            WHERE tenant_id = :t
            ORDER BY created_at DESC
            LIMIT 100
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patients": list(rows)}), 200


# ----------------------------
# Latest risk result for a patient
# ----------------------------
@api_bp.get("/api/v1/patient/latest")
def patient_latest():
    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    row = db.session.execute(
        text(
            """
            SELECT tenant_id, patient_id, status, result_json, error, finished_at
            FROM risk_jobs
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"t": tenant_id, "p": str(patient_id)},
    ).mappings().first()

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "latest": row}), 200
