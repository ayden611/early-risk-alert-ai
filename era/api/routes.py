# era/api/routes.py
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import Blueprint, Response, current_app, jsonify, request
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
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_tables():
    """
    Minimal tables required for:
    - ingesting events -> risk_jobs
    - showing dashboard data (patient_risk_cache, alerts, clinician_notes) if you created them
    """
    # risk_jobs queue (your worker polls this)
    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS risk_jobs (
          id BIGSERIAL PRIMARY KEY,
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', -- pending|processing|done|failed
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

    # Optional “dashboard” tables (safe to create even if unused)
    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS patient_risk_cache (
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          risk_score FLOAT,
          risk_level TEXT,
          summary TEXT,
          model_version TEXT,
          last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          PRIMARY KEY (tenant_id, patient_id)
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
          severity TEXT NOT NULL, -- info|warn|high
          title TEXT NOT NULL,
          body TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          acknowledged_at TIMESTAMPTZ
        );
        """
        )
    )

    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS clinician_notes (
          id BIGSERIAL PRIMARY KEY,
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          author_user_id TEXT,
          note TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        )
    )

    db.session.commit()


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
                "utc_now": _utc_now_iso(),
            }
        ),
        (200 if db_ok else 503),
    )


# ----------------------------
# Event ingestion -> queue job
# ----------------------------
@api_bp.post("/api/v1/events")
def ingest_event():
    """
    Pushes an event into risk_jobs for the worker to process.
    """
    _ensure_tables()

    body: Dict[str, Any] = request.get_json(force=True) or {}
    tenant_id = (body.get("tenant_id") or "demo-tenant").strip()
    patient_id = str(body.get("patient_id") or "unknown").strip()
    payload = body.get("payload") or {}

    if not isinstance(payload, dict):
        return jsonify({"error": "payload must be an object"}), 400

    db.session.execute(
        text(
            """
            INSERT INTO risk_jobs (tenant_id, patient_id, payload_json, status)
            VALUES (:t, :p, :j, 'pending')
            """
        ),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(payload)},
    )
    db.session.commit()

    return jsonify({"ok": True, "tenant_id": tenant_id, "patient_id": patient_id}), 202


# ----------------------------
# Clinician dashboard (JSON APIs)
# ----------------------------
@api_bp.get("/api/v1/clinician/patients")
def clinician_patients():
    """
    List patients for a tenant from patient_risk_cache (fast).
    """
    _ensure_tables()
    tenant_id = (request.args.get("tenant_id") or request.headers.get("X-Tenant-Id") or "demo-tenant").strip()

    rows = db.session.execute(
        text(
            """
            SELECT patient_id, risk_score, risk_level, last_updated
            FROM patient_risk_cache
            WHERE tenant_id=:t
            ORDER BY last_updated DESC
            LIMIT 500
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "patients": list(rows)}), 200


@api_bp.get("/api/v1/clinician/patients/<patient_id>")
def clinician_patient_detail(patient_id: str):
    """
    Patient detail: risk cache + recent notes + alerts
    """
    _ensure_tables()
    tenant_id = (request.args.get("tenant_id") or request.headers.get("X-Tenant-Id") or "demo-tenant").strip()

    risk = db.session.execute(
        text(
            """
            SELECT patient_id, risk_score, risk_level, summary, model_version, last_updated
            FROM patient_risk_cache
            WHERE tenant_id=:t AND patient_id=:p
            """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    notes = db.session.execute(
        text(
            """
            SELECT note, author_user_id, created_at
            FROM clinician_notes
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY created_at DESC
            LIMIT 50
            """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().all()

    alerts = db.session.execute(
        text(
            """
            SELECT id, severity, title, body, created_at, acknowledged_at
            FROM alerts
            WHERE tenant_id=:t AND patient_id=:p
            ORDER BY created_at DESC
            LIMIT 50
            """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().all()

    return jsonify(
        {"tenant_id": tenant_id, "patient_id": patient_id, "risk": risk, "notes": list(notes), "alerts": list(alerts)}
    ), 200


@api_bp.post("/api/v1/clinician/patients/<patient_id>/notes")
def clinician_add_note(patient_id: str):
    """
    Add a clinician note (no auth in this demo version).
    """
    _ensure_tables()
    tenant_id = (request.args.get("tenant_id") or request.headers.get("X-Tenant-Id") or "demo-tenant").strip()
    body: Dict[str, Any] = request.get_json(force=True) or {}
    note = (body.get("note") or "").strip()

    if not note:
        return jsonify({"error": "note required"}), 400

    author_user_id = (body.get("author_user_id") or "").strip() or None

    db.session.execute(
        text(
            """
            INSERT INTO clinician_notes (tenant_id, patient_id, author_user_id, note)
            VALUES (:t, :p, :u, :n)
            """
        ),
        {"t": tenant_id, "p": patient_id, "u": author_user_id, "n": note},
    )
    db.session.commit()

    return jsonify({"ok": True}), 201


@api_bp.get("/api/v1/clinician/alerts")
def clinician_alerts():
    _ensure_tables()
    tenant_id = (request.args.get("tenant_id") or request.headers.get("X-Tenant-Id") or "demo-tenant").strip()

    rows = db.session.execute(
        text(
            """
            SELECT id, patient_id, severity, title, body, created_at, acknowledged_at
            FROM alerts
            WHERE tenant_id=:t
            ORDER BY created_at DESC
            LIMIT 200
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    return jsonify({"tenant_id": tenant_id, "alerts": list(rows)}), 200


# ----------------------------
# Simple HTML redirect / placeholder
# ----------------------------
@api_bp.get("/dashboard")
def dashboard_hint():
    """
    If your HTML dashboard template lives in the web blueprint instead, this just guides you.
    """
    return Response(
        "Dashboard route is active. If you have an HTML dashboard, wire it in your web blueprint.\n"
        "Try: GET /api/v1/clinician/patients?tenant_id=demo-tenant",
        mimetype="text/plain",
    )
