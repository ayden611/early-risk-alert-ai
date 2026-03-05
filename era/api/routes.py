# era/api/routes.py
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text
from werkzeug.exceptions import HTTPException
from redis import Redis
from era.streaming import REDIS_URL, VITALS_STREAM_KEY

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


def detect_anomalies(vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lightweight rule-based anomaly detection (fast, safe, explainable).
    For production at scale, keep this cheap at ingest, then do heavier scoring async in worker.
    """
    alerts: List[Dict[str, Any]] = []

    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp") or vitals.get("sys_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp") or vitals.get("dia_bp"))
    spo2 = _as_float(vitals.get("spo2"))
    temp_c = _as_float(vitals.get("temp_c"))
    temp_f = _as_float(vitals.get("temp_f"))

    if temp_f is None and temp_c is not None:
        temp_f = (temp_c * 9.0 / 5.0) + 32.0

    # Heart rate
    if hr is not None:
        if hr >= 130:
            alerts.append({"type": "tachycardia", "severity": "high", "message": f"Heart rate very high ({hr})."})
        elif hr >= 110:
            alerts.append({"type": "tachycardia", "severity": "medium", "message": f"Heart rate elevated ({hr})."})
        elif hr <= 40:
            alerts.append({"type": "bradycardia", "severity": "high", "message": f"Heart rate very low ({hr})."})
        elif hr <= 50:
            alerts.append({"type": "bradycardia", "severity": "medium", "message": f"Heart rate low ({hr})."})

    # Blood pressure
    if sys_bp is not None and dia_bp is not None:
        if sys_bp >= 180 or dia_bp >= 120:
            alerts.append({"type": "hypertensive_crisis", "severity": "high",
                           "message": f"BP crisis range ({sys_bp}/{dia_bp})."})
        elif sys_bp >= 160 or dia_bp >= 100:
            alerts.append({"type": "high_bp", "severity": "medium", "message": f"High BP ({sys_bp}/{dia_bp})."})
        elif sys_bp <= 90 or dia_bp <= 60:
            alerts.append({"type": "low_bp", "severity": "medium", "message": f"Low BP ({sys_bp}/{dia_bp})."})

    # Oxygen
    if spo2 is not None:
        if spo2 < 90:
            alerts.append({"type": "low_spo2", "severity": "high", "message": f"Low oxygen saturation ({spo2}%)."})
        elif spo2 < 93:
            alerts.append({"type": "low_spo2", "severity": "medium", "message": f"Borderline oxygen saturation ({spo2}%)."})

    # Temperature
    if temp_f is not None:
        if temp_f >= 103:
            alerts.append({"type": "fever", "severity": "high", "message": f"High fever ({round(temp_f, 1)}F)."})
        elif temp_f >= 100.4:
            alerts.append({"type": "fever", "severity": "medium", "message": f"Fever ({round(temp_f, 1)}F)."})
        elif temp_f <= 95:
            alerts.append({"type": "hypothermia", "severity": "high", "message": f"Low temperature ({round(temp_f, 1)}F)."})

    return alerts


def build_reasoning_snapshot(vitals: Dict[str, Any], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    "AI health reasoning engine" (starter): produces a structured explanation.
    In production, this becomes:
      - cheap online summary (this)
      - deeper async reasoning (worker) w/ guidelines + citations + care pathways.
    """
    summary = []
    if not vitals:
        summary.append("No vitals received yet for this patient.")
    else:
        summary.append("Latest vitals ingested and evaluated for anomalies.")

    if alerts:
        high = [a for a in alerts if a.get("severity") == "high"]
        med = [a for a in alerts if a.get("severity") == "medium"]
        if high:
            summary.append(f"{len(high)} high-severity alert(s) detected requiring prompt review.")
        if med:
            summary.append(f"{len(med)} medium-severity alert(s) detected; monitor and re-check trend.")
    else:
        summary.append("No anomalies detected by lightweight rules at this time.")

    # Very simple risk score (0-100) based on alerts
    score = 5
    for a in alerts:
        if a["severity"] == "high":
            score += 30
        elif a["severity"] == "medium":
            score += 15
    score = max(0, min(100, score))

    return {
        "generated_at": _utcnow_iso(),
        "risk_score_0_100": score,
        "summary": " ".join(summary),
        "alerts": alerts,
        "next_actions": [
            "Verify device signal quality and measurement context (resting, post-exertion, etc.).",
            "If repeated high-severity alerts occur, escalate per care protocol.",
            "Trend review recommended (last 24-72 hours) rather than a single reading."
        ],
        "disclaimer": "Not medical advice. For clinical decisions, follow your organization’s protocols."
    }


# ----------------------------
# DB bootstrap (optional but helpful)
# ----------------------------
def ensure_tables() -> None:
    """
    Creates tables if they don't exist.
    Safe to call repeatedly.
    """
    db.session.execute(text("""
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
    """))
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_risk_jobs_status_id ON risk_jobs(status, id);
    """))
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_risk_jobs_tenant_patient ON risk_jobs(tenant_id, patient_id);
    """))

    db.session.execute(text("""
        CREATE TABLE IF NOT EXISTS vitals_events (
          id BIGSERIAL PRIMARY KEY,
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          source TEXT,
          event_ts TIMESTAMPTZ,
          payload_json TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_vitals_events_tenant_patient_created ON vitals_events(tenant_id, patient_id, created_at DESC);
    """))

    db.session.execute(text("""
        CREATE TABLE IF NOT EXISTS alerts (
          id BIGSERIAL PRIMARY KEY,
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          alert_type TEXT NOT NULL,
          severity TEXT NOT NULL,
          message TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created ON alerts(tenant_id, patient_id, created_at DESC);
    """))

    db.session.commit()


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

    return jsonify({
        "status": "ok" if db_ok else "degraded",
        "db_ok": db_ok,
        "db_error": db_error,
        "ts": _utcnow_iso(),
    }), (200 if db_ok else 503)


@api_bp.post("/admin/ensure_tables")
def admin_ensure_tables():
    ensure_tables()
    return jsonify({"ok": True, "message": "Tables ensured"}), 200


# ----------------------------
# Risk jobs ingestion
# ----------------------------
@api_bp.post("/events")
def ingest_event():
    ensure_tables()

    body = request.get_json(force=True) or {}
    tenant_id = body.get("tenant_id", "demo-tenant")
    patient_id = str(body.get("patient_id", "unknown"))
    payload = body.get("payload", {}) or {}

    db.session.execute(
        text("""
            INSERT INTO risk_jobs (tenant_id, patient_id, payload_json, status)
            VALUES (:t, :p, :j, 'pending')
        """),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(payload)},
    )
    db.session.commit()

    return jsonify({
        "ok": True,
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "status": "pending",
        "ts": _utcnow_iso()
    }), 200


# ----------------------------
# Vitals streaming ingestion
# ----------------------------
@api_bp.post("/vitals")
def ingest_vitals():
    """
    Stream ingestion endpoint: devices/apps post vitals here.
    Now pushes to Redis Stream instead of doing heavy work.
    """

    if not REDIS_URL:
        return jsonify({"error": "REDIS_URL not set"}), 500

    body = request.get_json(force=True) or {}

    tenant_id = body.get("tenant_id", "demo-tenant")
    patient_id = str(body.get("patient_id", "unknown"))
    source = body.get("source", "manual")
    vitals = body.get("vitals", {}) or {}
    event_ts = body.get("event_ts")

    event = {
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "source": source,
        "event_ts": event_ts or "",
        "vitals_json": json.dumps(vitals),
        "ingested_at": _utcnow_iso(),
    }

    r = Redis.from_url(REDIS_URL, decode_responses=True)

    msg_id = r.xadd(
        VITALS_STREAM_KEY,
        event,
        maxlen=2000000,
        approximate=True
    )

    return jsonify({
        "ok": True,
        "accepted": True,
        "message_id": msg_id,
        "tenant_id": tenant_id,
        "patient_id": patient_id
    }), 202


# ----------------------------
# AI reasoning snapshot
# ----------------------------
@api_bp.get("/ai/health/reasoning")
def ai_health_reasoning():
    ensure_tables()

    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    row = db.session.execute(
        text("""
            SELECT payload_json, created_at
            FROM vitals_events
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"t": tenant_id, "p": str(patient_id)},
    ).fetchone()

    latest_vitals = {}
    latest_vitals_ts = None
    if row:
        latest_vitals = json.loads(row[0] or "{}")
        latest_vitals_ts = row[1].isoformat() if row[1] else None

    arows = db.session.execute(
        text("""
            SELECT alert_type, severity, message, created_at
            FROM alerts
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT 10
        """),
        {"t": tenant_id, "p": str(patient_id)},
    ).fetchall()

    recent_alerts = [{
        "type": r[0],
        "severity": r[1],
        "message": r[2],
        "created_at": r[3].isoformat() if r[3] else None
    } for r in arows]

    snapshot = build_reasoning_snapshot(latest_vitals, recent_alerts)

    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": str(patient_id),
        "latest_vitals_ts": latest_vitals_ts,
        "latest_vitals": latest_vitals,
        "ai_analysis": snapshot
    }), 200


# ----------------------------
# Clinician patients view
# ----------------------------
@api_bp.get("/clinician/patients")
def clinician_patients():
    ensure_tables()

    tenant_id = request.args.get("tenant_id", "demo-tenant")
    limit = int(request.args.get("limit", "50"))

    rows = db.session.execute(
        text("""
            WITH p AS (
              SELECT patient_id, MAX(created_at) AS last_seen
              FROM vitals_events
              WHERE tenant_id = :t
              GROUP BY patient_id
            )
            SELECT patient_id, last_seen
            FROM p
            ORDER BY last_seen DESC
            LIMIT :lim
        """),
        {"t": tenant_id, "lim": limit},
    ).fetchall()

    patients = []
    for patient_id, last_seen in rows:
        vrow = db.session.execute(
            text("""
                SELECT payload_json, created_at
                FROM vitals_events
                WHERE tenant_id = :t AND patient_id = :p
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"t": tenant_id, "p": patient_id},
        ).fetchone()

        latest_vitals = json.loads(vrow[0] or "{}") if vrow else {}
        latest_vitals_ts = vrow[1].isoformat() if (vrow and vrow[1]) else None

        arow = db.session.execute(
            text("""
                SELECT alert_type, severity, message, created_at
                FROM alerts
                WHERE tenant_id = :t AND patient_id = :p
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"t": tenant_id, "p": patient_id},
        ).fetchone()

        last_alert = None
        if arow:
            last_alert = {
                "type": arow[0],
                "severity": arow[1],
                "message": arow[2],
                "created_at": arow[3].isoformat() if arow[3] else None
            }

        patients.append({
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "last_seen": last_seen.isoformat() if last_seen else None,
            "latest_vitals_ts": latest_vitals_ts,
            "latest_vitals": latest_vitals,
            "last_alert": last_alert,
        })

    return jsonify({"tenant_id": tenant_id, "patients": patients}), 200


@api_bp.get("/alerts")
def get_alerts():
    ensure_tables()

    tenant_id = request.args.get("tenant_id", "demo-tenant")
    patient_id = request.args.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    limit = int(request.args.get("limit", "50"))

    rows = db.session.execute(
        text("""
            SELECT alert_type, severity, message, created_at
            FROM alerts
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT :lim
        """),
        {"t": tenant_id, "p": str(patient_id), "lim": limit},
    ).fetchall()

    alerts = [{
        "type": r[0],
        "severity": r[1],
        "message": r[2],
        "created_at": r[3].isoformat() if r[3] else None
    } for r in rows]

    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": str(patient_id),
        "alerts": alerts
    }), 200
