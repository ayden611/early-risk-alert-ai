import json
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

api_bp = Blueprint("api", __name__)


# --------------------------------------------------
# Error handling
# --------------------------------------------------
@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled API error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _tenant_id() -> str:
    return (
        request.headers.get("X-Tenant-Id")
        or request.args.get("tenant_id")
        or (request.get_json(silent=True) or {}).get("tenant_id")
        or "demo"
    )


def _actor_id() -> str:
    return request.headers.get("X-User-Id", "system")


def _actor_role() -> str:
    return request.headers.get("X-Role", "admin")


def _request_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def require_roles(*roles: str):
    allowed = {r.lower() for r in roles}

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            role = _actor_role().lower()
            if role not in allowed:
                return (
                    jsonify(
                        {
                            "error": "forbidden",
                            "message": f"Role '{role}' is not allowed for this endpoint",
                            "allowed_roles": sorted(list(allowed)),
                        }
                    ),
                    403,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# --------------------------------------------------
# Schema bootstrap
# --------------------------------------------------
def ensure_platform_tables() -> None:
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS vitals_events (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                source TEXT,
                event_id TEXT,
                event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                payload_json JSONB NOT NULL
            );
            """
        )
    )

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
            CREATE TABLE IF NOT EXISTS alerts (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                alert_type TEXT NOT NULL DEFAULT 'anomaly',
                severity TEXT NOT NULL DEFAULT 'info',
                status TEXT NOT NULL DEFAULT 'open',
                message TEXT NOT NULL DEFAULT '',
                payload_json JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                acknowledged_at TIMESTAMPTZ,
                acknowledged_by TEXT
            );
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

    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS risk_scores (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                risk_score NUMERIC(6,3) NOT NULL,
                risk_level TEXT NOT NULL,
                summary TEXT NOT NULL,
                findings_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )

    db.session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_risk_scores_tenant_patient_created
            ON risk_scores (tenant_id, patient_id, created_at DESC);
            """
        )
    )

    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                ip_address TEXT,
                meta_json JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )

    db.session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_created
            ON audit_logs (tenant_id, created_at DESC);
            """
        )
    )

    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS mobile_devices (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                platform TEXT,
                app_version TEXT,
                push_token TEXT,
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )

    db.session.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_mobile_devices_tenant_patient_device
            ON mobile_devices (tenant_id, patient_id, device_id);
            """
        )
    )

    db.session.commit()


def _audit(
    action: str,
    resource_type: str,
    resource_id: Optional[str],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    db.session.execute(
        text(
            """
            INSERT INTO audit_logs
                (tenant_id, actor_id, actor_role, action, resource_type, resource_id, ip_address, meta_json)
            VALUES
                (:tenant_id, :actor_id, :actor_role, :action, :resource_type, :resource_id, :ip_address, CAST(:meta_json AS jsonb))
            """
        ),
        {
            "tenant_id": _tenant_id(),
            "actor_id": _actor_id(),
            "actor_role": _actor_role(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": _request_ip(),
            "meta_json": _json_dumps(meta or {}),
        },
    )


# --------------------------------------------------
# AI scoring and anomaly logic
# --------------------------------------------------
def _normalize_vitals(payload: Dict[str, Any]) -> Dict[str, Any]:
    nested = payload.get("vitals")
    if isinstance(nested, dict):
        return nested

    keys = [
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "spo2",
        "respiratory_rate",
        "temperature_f",
    ]
    return {k: payload.get(k) for k in keys if k in payload}


def _score_vitals(vitals: Dict[str, Any]) -> Dict[str, Any]:
    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))
    rr = _as_float(vitals.get("respiratory_rate"))
    temp = _as_float(vitals.get("temperature_f"))

    findings: List[str] = []
    alerts: List[Dict[str, str]] = []
    score = 0.08

    if hr is not None and hr >= 120:
        findings.append(f"Tachycardia detected ({hr:.0f} bpm)")
        alerts.append(
            {
                "alert_type": "tachycardia",
                "severity": "high",
                "message": f"Heart rate elevated at {hr:.0f} bpm",
            }
        )
        score += 0.22
    elif hr is not None and hr >= 100:
        findings.append(f"Elevated heart rate ({hr:.0f} bpm)")
        score += 0.10

    if sys_bp is not None and sys_bp >= 180:
        findings.append(f"Severely elevated systolic blood pressure ({sys_bp:.0f})")
        alerts.append(
            {
                "alert_type": "hypertensive_crisis",
                "severity": "critical",
                "message": f"Systolic blood pressure critically high at {sys_bp:.0f}",
            }
        )
        score += 0.30
    elif sys_bp is not None and sys_bp >= 160:
        findings.append(f"High systolic blood pressure ({sys_bp:.0f})")
        alerts.append(
            {
                "alert_type": "hypertension",
                "severity": "high",
                "message": f"Systolic blood pressure elevated at {sys_bp:.0f}",
            }
        )
        score += 0.18

    if dia_bp is not None and dia_bp >= 110:
        findings.append(f"Severely elevated diastolic blood pressure ({dia_bp:.0f})")
        alerts.append(
            {
                "alert_type": "hypertensive_crisis",
                "severity": "critical",
                "message": f"Diastolic blood pressure critically high at {dia_bp:.0f}",
            }
        )
        score += 0.25
    elif dia_bp is not None and dia_bp >= 100:
        findings.append(f"High diastolic blood pressure ({dia_bp:.0f})")
        alerts.append(
            {
                "alert_type": "hypertension",
                "severity": "high",
                "message": f"Diastolic blood pressure elevated at {dia_bp:.0f}",
            }
        )
        score += 0.16

    if spo2 is not None and spo2 < 90:
        findings.append(f"Critical low oxygen saturation ({spo2:.0f}%)")
        alerts.append(
            {
                "alert_type": "low_oxygen",
                "severity": "critical",
                "message": f"SpO2 critically low at {spo2:.0f}%",
            }
        )
        score += 0.30
    elif spo2 is not None and spo2 < 92:
        findings.append(f"Low oxygen saturation ({spo2:.0f}%)")
        alerts.append(
            {
                "alert_type": "low_oxygen",
                "severity": "high",
                "message": f"SpO2 low at {spo2:.0f}%",
            }
        )
        score += 0.20

    if rr is not None and rr >= 24:
        findings.append(f"Elevated respiratory rate ({rr:.0f})")
        score += 0.08

    if temp is not None and temp >= 101.5:
        findings.append(f"Fever detected ({temp:.1f}F)")
        score += 0.08

    score = min(score, 0.99)

    if score >= 0.75:
        risk_level = "high"
    elif score >= 0.40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    summary = "No major abnormalities detected." if not findings else " | ".join(findings)

    return {
        "risk_score": round(score, 3),
        "risk_level": risk_level,
        "summary": summary,
        "findings": findings,
        "alerts": alerts,
    }


# --------------------------------------------------
# Core platform endpoints
# --------------------------------------------------
@api_bp.get("/healthz")
def api_healthz():
    try:
        ensure_platform_tables()
        db.session.execute(text("SELECT 1"))
        return (
            jsonify(
                {
                    "status": "ok",
                    "service": "early-risk-alert-api",
                    "db_ok": True,
                    "db_error": None,
                }
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "degraded",
                    "service": "early-risk-alert-api",
                    "db_ok": False,
                    "db_error": str(e),
                }
            ),
            503,
        )


@api_bp.post("/mobile/connect")
def mobile_connect():
    ensure_platform_tables()
    body = request.get_json(force=True) or {}

    tenant_id = str(body.get("tenant_id", _tenant_id()))
    patient_id = str(body.get("patient_id", "unknown"))
    device_id = str(body.get("device_id", "device-unknown"))
    platform = str(body.get("platform", "unknown"))
    app_version = str(body.get("app_version", "unknown"))
    push_token = body.get("push_token")

    db.session.execute(
        text(
            """
            INSERT INTO mobile_devices (tenant_id, patient_id, device_id, platform, app_version, push_token, last_seen_at)
            VALUES (:t, :p, :d, :platform, :app_version, :push_token, NOW())
            ON CONFLICT (tenant_id, patient_id, device_id)
            DO UPDATE SET
                platform = EXCLUDED.platform,
                app_version = EXCLUDED.app_version,
                push_token = EXCLUDED.push_token,
                last_seen_at = NOW()
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "d": device_id,
            "platform": platform,
            "app_version": app_version,
            "push_token": push_token,
        },
    )
    _audit("mobile_connect", "mobile_device", device_id, body)
    db.session.commit()

    return (
        jsonify(
            {
                "ok": True,
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "device_id": device_id,
                "connected_at": _utcnow_iso(),
            }
        ),
        200,
    )


@api_bp.post("/vitals")
def ingest_vitals():
    ensure_platform_tables()

    body = request.get_json(force=True) or {}
    tenant_id = str(body.get("tenant_id", _tenant_id()))
    patient_id = str(body.get("patient_id", "unknown"))
    source = str(body.get("source", "manual"))
    event_id = body.get("event_id")
    event_ts = body.get("event_ts")
    vitals = _normalize_vitals(body)

    ai = _score_vitals(vitals)

    db.session.execute(
        text(
            """
            INSERT INTO vitals_events
                (tenant_id, patient_id, source, event_id, event_ts, payload_json)
            VALUES
                (:t, :p, :s, :eid, COALESCE(CAST(:ets AS timestamptz), NOW()), CAST(:payload AS jsonb))
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "eid": event_id,
            "ets": event_ts,
            "payload": _json_dumps(vitals),
        },
    )

    db.session.execute(
        text(
            """
            INSERT INTO risk_scores
                (tenant_id, patient_id, risk_score, risk_level, summary, findings_json)
            VALUES
                (:t, :p, :score, :level, :summary, CAST(:findings AS jsonb))
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "score": ai["risk_score"],
            "level": ai["risk_level"],
            "summary": ai["summary"],
            "findings": _json_dumps(ai["findings"]),
        },
    )

    for alert in ai["alerts"]:
        db.session.execute(
            text(
                """
                INSERT INTO alerts
                    (tenant_id, patient_id, alert_type, severity, status, message, payload_json)
                VALUES
                    (:t, :p, :atype, :sev, 'open', :msg, CAST(:payload AS jsonb))
                """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "atype": alert["alert_type"],
                "sev": alert["severity"],
                "msg": alert["message"],
                "payload": _json_dumps(vitals),
            },
        )

    _audit(
        "vitals_ingested",
        "patient",
        patient_id,
        {"source": source, "risk_level": ai["risk_level"], "alerts_created": len(ai["alerts"])},
    )
    db.session.commit()

    return (
        jsonify(
            {
                "ok": True,
                "accepted": True,
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "received_at": _utcnow_iso(),
                "risk_score": ai["risk_score"],
                "risk_level": ai["risk_level"],
                "alerts_created": len(ai["alerts"]),
                "findings": ai["findings"],
            }
        ),
        200,
    )


# --------------------------------------------------
# A) Live anomaly alerts dashboard
# --------------------------------------------------
@api_bp.get("/alerts/live")
@require_roles("admin", "clinician", "insurer")
def alerts_live():
    ensure_platform_tables()

    tenant_id = _tenant_id()
    limit = _as_int(request.args.get("limit"), 100)
    severity = request.args.get("severity")

    sql = """
        SELECT id, patient_id, alert_type, severity, status, message, created_at, payload_json
        FROM alerts
        WHERE tenant_id = :t
    """
    params: Dict[str, Any] = {"t": tenant_id, "lim": limit}

    if severity:
        sql += " AND severity = :sev "
        params["sev"] = severity

    sql += " ORDER BY created_at DESC LIMIT :lim "

    rows = db.session.execute(text(sql), params).mappings().all()

    alerts = []
    for row in rows:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        alerts.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "alert_type": row["alert_type"],
                "severity": row["severity"],
                "status": row["status"],
                "message": row["message"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "payload": payload,
            }
        )

    return jsonify({"tenant_id": tenant_id, "alerts": alerts}), 200


@api_bp.post("/alerts/<int:alert_id>/ack")
@require_roles("admin", "clinician")
def ack_alert(alert_id: int):
    ensure_platform_tables()

    updated = db.session.execute(
        text(
            """
            UPDATE alerts
            SET status = 'acknowledged',
                acknowledged_at = NOW(),
                acknowledged_by = :actor
            WHERE id = :id AND tenant_id = :t
            """
        ),
        {"id": alert_id, "t": _tenant_id(), "actor": _actor_id()},
    )

    if updated.rowcount == 0:
        return jsonify({"error": "not_found"}), 404

    _audit("alert_acknowledged", "alert", str(alert_id), {})
    db.session.commit()
    return jsonify({"ok": True, "alert_id": alert_id, "status": "acknowledged"}), 200


# --------------------------------------------------
# B) AI health risk scoring per patient
# --------------------------------------------------
@api_bp.get("/patients/<patient_id>/risk")
@require_roles("admin", "clinician", "insurer")
def patient_risk(patient_id: str):
    ensure_platform_tables()

    row = db.session.execute(
        text(
            """
            SELECT risk_score, risk_level, summary, findings_json, created_at
            FROM risk_scores
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"t": _tenant_id(), "p": patient_id},
    ).mappings().first()

    if not row:
        return jsonify({"error": "not_found"}), 404

    findings = row["findings_json"]
    if isinstance(findings, str):
        findings = json.loads(findings)

    return (
        jsonify(
            {
                "tenant_id": _tenant_id(),
                "patient_id": patient_id,
                "risk_score": float(row["risk_score"]),
                "risk_level": row["risk_level"],
                "summary": row["summary"],
                "findings": findings,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
        ),
        200,
    )


@api_bp.get("/patients/<patient_id>/timeline")
@require_roles("admin", "clinician")
def patient_timeline(patient_id: str):
    ensure_platform_tables()

    rows = db.session.execute(
        text(
            """
            SELECT event_ts, created_at, payload_json
            FROM vitals_events
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT 100
            """
        ),
        {"t": _tenant_id(), "p": patient_id},
    ).mappings().all()

    timeline = []
    for row in rows:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        timeline.append(
            {
                "event_ts": row["event_ts"].isoformat() if row["event_ts"] else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "vitals": payload,
            }
        )

    return jsonify({"tenant_id": _tenant_id(), "patient_id": patient_id, "timeline": timeline}), 200


# --------------------------------------------------
# C) Hospital admin analytics portal
# --------------------------------------------------
@api_bp.get("/admin/analytics/overview")
@require_roles("admin")
def admin_analytics():
    ensure_platform_tables()

    tenant_id = _tenant_id()

    kpis = db.session.execute(
        text(
            """
            SELECT
                (SELECT COUNT(DISTINCT patient_id) FROM vitals_events WHERE tenant_id = :t) AS unique_patients,
                (SELECT COUNT(*) FROM vitals_events WHERE tenant_id = :t AND created_at >= NOW() - INTERVAL '24 hours') AS vitals_24h,
                (SELECT COUNT(*) FROM alerts WHERE tenant_id = :t AND status = 'open') AS open_alerts,
                (SELECT COUNT(*) FROM alerts WHERE tenant_id = :t AND severity = 'critical' AND created_at >= NOW() - INTERVAL '24 hours') AS critical_alerts_24h
            """
        ),
        {"t": tenant_id},
    ).mappings().first()

    risk_mix = db.session.execute(
        text(
            """
            SELECT risk_level, COUNT(*) AS count
            FROM (
                SELECT DISTINCT ON (patient_id) patient_id, risk_level
                FROM risk_scores
                WHERE tenant_id = :t
                ORDER BY patient_id, created_at DESC
            ) x
            GROUP BY risk_level
            ORDER BY risk_level
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    return (
        jsonify(
            {
                "tenant_id": tenant_id,
                "overview": {
                    "unique_patients": int(kpis["unique_patients"] or 0),
                    "vitals_24h": int(kpis["vitals_24h"] or 0),
                    "open_alerts": int(kpis["open_alerts"] or 0),
                    "critical_alerts_24h": int(kpis["critical_alerts_24h"] or 0),
                },
                "risk_mix": [{"risk_level": r["risk_level"], "count": int(r["count"])} for r in risk_mix],
            }
        ),
        200,
    )


# --------------------------------------------------
# D) Insurance risk pool monitoring
# --------------------------------------------------
@api_bp.get("/insurer/pool/overview")
@require_roles("insurer", "admin")
def insurer_pool_overview():
    ensure_platform_tables()

    tenant_id = _tenant_id()

    rows = db.session.execute(
        text(
            """
            SELECT risk_level, COUNT(*) AS members
            FROM (
                SELECT DISTINCT ON (patient_id) patient_id, risk_level
                FROM risk_scores
                WHERE tenant_id = :t
                ORDER BY patient_id, created_at DESC
            ) latest
            GROUP BY risk_level
            ORDER BY risk_level
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    alert_rows = db.session.execute(
        text(
            """
            SELECT severity, COUNT(*) AS count
            FROM alerts
            WHERE tenant_id = :t
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY severity
            ORDER BY severity
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    return (
        jsonify(
            {
                "tenant_id": tenant_id,
                "member_risk_distribution": [
                    {"risk_level": row["risk_level"], "members": int(row["members"])} for row in rows
                ],
                "alerts_last_24h": [
                    {"severity": row["severity"], "count": int(row["count"])} for row in alert_rows
                ],
            }
        ),
        200,
    )


# --------------------------------------------------
# E) Investor-ready system architecture diagram
# --------------------------------------------------
@api_bp.get("/architecture/diagram")
@require_roles("admin", "insurer")
def architecture_diagram():
    mermaid = """
flowchart LR
    mobile[Mobile App / Device SDK] --> api[Mobile API / Flask]
    api --> pg[(Postgres)]
    api --> redis[(Redis / Valkey)]
    api --> worker[Background Worker]
    worker --> pg
    worker --> alerts[Live Alerts Stream]
    api --> admin[Hospital Admin Portal]
    api --> insurer[Insurance Risk Portal]
    api --> audit[Audit / RBAC Layer]
    admin --> api
    insurer --> api
"""
    return (
        jsonify(
            {
                "title": "Early Risk Alert Platform Architecture",
                "diagram_type": "mermaid",
                "mermaid": mermaid.strip(),
                "notes": [
                    "API tier handles ingestion, scoring, analytics APIs, and mobile app connectivity.",
                    "Postgres is the system of record for vitals, alerts, scores, audit logs, and devices.",
                    "Redis/Valkey supports queueing, caching, and fan-out for real-time alert streams.",
                    "Background workers scale horizontally for anomaly detection and batch scoring.",
                    "Role-based access and audit logs support HIPAA-oriented operational controls.",
                ],
            }
        ),
        200,
    )


# --------------------------------------------------
# F) HIPAA-grade audit logging & RBAC security
# --------------------------------------------------
@api_bp.get("/security/me")
def security_me():
    return (
        jsonify(
            {
                "tenant_id": _tenant_id(),
                "actor_id": _actor_id(),
                "role": _actor_role(),
                "permissions": {
                    "admin": ["all"],
                    "clinician": ["read_patients", "read_alerts", "ack_alerts", "read_analytics"],
                    "insurer": ["read_pool_monitoring", "read_risk", "read_architecture"],
                    "mobile": ["connect_device", "submit_vitals"],
                }.get(_actor_role(), []),
            }
        ),
        200,
    )


@api_bp.get("/audit/logs")
@require_roles("admin")
def audit_logs():
    ensure_platform_tables()

    limit = _as_int(request.args.get("limit"), 100)
    rows = db.session.execute(
        text(
            """
            SELECT actor_id, actor_role, action, resource_type, resource_id, ip_address, meta_json, created_at
            FROM audit_logs
            WHERE tenant_id = :t
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {"t": _tenant_id(), "lim": limit},
    ).mappings().all()

    items = []
    for row in rows:
        meta = row["meta_json"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        items.append(
            {
                "actor_id": row["actor_id"],
                "actor_role": row["actor_role"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "ip_address": row["ip_address"],
                "meta": meta,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
        )

    return jsonify({"tenant_id": _tenant_id(), "logs": items}), 200


# --------------------------------------------------
# G) Million-patient scaling architecture metadata
# --------------------------------------------------
@api_bp.get("/scale/readiness")
@require_roles("admin", "insurer")
def scale_readiness():
    return (
        jsonify(
            {
                "status": "ready_for_next_phase",
                "current_pattern": {
                    "api": "stateless Flask web tier",
                    "db": "Postgres system of record",
                    "cache_queue": "Redis / Valkey",
                    "worker": "horizontal background processors",
                },
                "million_patient_plan": [
                    "Partition vitals events by tenant_id and time window",
                    "Move real-time alert fan-out to Redis streams or Kafka",
                    "Scale workers horizontally by patient shard",
                    "Add read replicas for analytics-heavy dashboards",
                    "Cache live dashboard aggregates in Redis",
                    "Adopt time-series retention and archival strategy for historical events",
                ],
            }
        ),
        200,
    )


# --------------------------------------------------
# H) Mobile app connection layer
# --------------------------------------------------
@api_bp.get("/mobile/config")
def mobile_config():
    return (
        jsonify(
            {
                "tenant_id": _tenant_id(),
                "api_version": "v1",
                "submit_vitals_path": "/api/v1/vitals",
                "device_connect_path": "/api/v1/mobile/connect",
                "health_check_path": "/api/v1/healthz",
                "recommended_headers": ["X-Tenant-Id", "X-User-Id", "X-Role"],
                "auth_mode": "header-based placeholder; replace with JWT for production",
            }
        ),
        200,
    )
