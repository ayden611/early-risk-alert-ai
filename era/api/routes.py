import json
from datetime import datetime, timezone
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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


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
# DB bootstrap
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

    db.session.commit()


# --------------------------------------------------
# Health
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


# --------------------------------------------------
# Real-time patient monitoring dashboard
# --------------------------------------------------
@api_bp.get("/dashboard/overview")
def dashboard_overview():
    ensure_platform_tables()

    tenant_id = request.args.get("tenant_id", "demo")

    stats = db.session.execute(
        text(
            """
            SELECT
                (SELECT COUNT(DISTINCT patient_id) FROM vitals_events WHERE tenant_id = :t) AS patient_count,
                (SELECT COUNT(*) FROM alerts WHERE tenant_id = :t AND status = 'open') AS open_alerts,
                (SELECT COUNT(*) FROM alerts WHERE tenant_id = :t AND severity = 'critical' AND status = 'open') AS critical_alerts,
                (SELECT COUNT(*) FROM vitals_events WHERE tenant_id = :t AND created_at >= NOW() - INTERVAL '1 hour') AS events_last_hour
            """
        ),
        {"t": tenant_id},
    ).mappings().first()

    return (
        jsonify(
            {
                "tenant_id": tenant_id,
                "patient_count": int(stats["patient_count"] or 0),
                "open_alerts": int(stats["open_alerts"] or 0),
                "critical_alerts": int(stats["critical_alerts"] or 0),
                "events_last_hour": int(stats["events_last_hour"] or 0),
                "dashboard_mode": "real_time_monitoring",
            }
        ),
        200,
    )


# --------------------------------------------------
# Live streaming vitals ingestion
# --------------------------------------------------
@api_bp.post("/vitals")
def ingest_vitals():
    ensure_platform_tables()

    body = request.get_json(force=True) or {}
    tenant_id = str(body.get("tenant_id", "demo"))
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


@api_bp.get("/vitals/latest")
def latest_vitals():
    ensure_platform_tables()

    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")

    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    row = db.session.execute(
        text(
            """
            SELECT tenant_id, patient_id, source, event_id, event_ts, created_at, payload_json
            FROM vitals_events
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    if not row:
        return jsonify({"error": "not_found"}), 404

    payload = row["payload_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    return (
        jsonify(
            {
                "tenant_id": row["tenant_id"],
                "patient_id": row["patient_id"],
                "source": row["source"],
                "event_id": row["event_id"],
                "event_ts": row["event_ts"].isoformat() if row["event_ts"] else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "vitals": payload,
            }
        ),
        200,
    )


# --------------------------------------------------
# AI anomaly alerts
# --------------------------------------------------
@api_bp.get("/alerts/live")
def alerts_live():
    ensure_platform_tables()

    tenant_id = request.args.get("tenant_id", "demo")
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

    items = []
    for row in rows:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)

        items.append(
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

    return jsonify({"tenant_id": tenant_id, "alerts": items}), 200


@api_bp.get("/patients/<patient_id>/risk")
def patient_risk(patient_id: str):
    ensure_platform_tables()

    tenant_id = request.args.get("tenant_id", "demo")

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
        {"t": tenant_id, "p": patient_id},
    ).mappings().first()

    if not row:
        return jsonify({"error": "not_found"}), 404

    findings = row["findings_json"]
    if isinstance(findings, str):
        findings = json.loads(findings)

    return (
        jsonify(
            {
                "tenant_id": tenant_id,
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


# --------------------------------------------------
# Command center UI API
# --------------------------------------------------
@api_bp.get("/command-center/summary")
def command_center_summary():
    ensure_platform_tables()

    tenant_id = request.args.get("tenant_id", "demo")

    rows = db.session.execute(
        text(
            """
            SELECT severity, COUNT(*) AS count
            FROM alerts
            WHERE tenant_id = :t AND status = 'open'
            GROUP BY severity
            ORDER BY severity
            """
        ),
        {"t": tenant_id},
    ).mappings().all()

    risk_rows = db.session.execute(
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
                "open_alerts_by_severity": [
                    {"severity": row["severity"], "count": int(row["count"])} for row in rows
                ],
                "patient_risk_mix": [
                    {"risk_level": row["risk_level"], "count": int(row["count"])} for row in risk_rows
                ],
                "ui_mode": "command_center",
            }
        ),
        200,
    )


# --------------------------------------------------
# Million-patient scaling mode
# --------------------------------------------------
@api_bp.get("/scale/readiness")
def scale_readiness():
    return (
        jsonify(
            {
                "status": "ready_for_next_phase",
                "scaling_mode": "million_patient",
                "current_pattern": {
                    "api": "stateless Flask web tier",
                    "db": "Postgres system of record",
                    "workers": "background workers can scale horizontally",
                    "cache_queue": "Redis/Valkey ready",
                },
                "next_steps": [
                    "Partition vitals events by tenant and time window",
                    "Move hot alert fan-out to Redis streams",
                    "Shard workers by patient or tenant hash",
                    "Add read replicas for analytics",
                    "Cache dashboard aggregates in Redis",
                    "Archive cold historical vitals to lower-cost storage",
                ],
            }
        ),
        200,
    )


# --------------------------------------------------
# Rollups for monitoring analytics
# --------------------------------------------------
@api_bp.get("/rollups/15m")
def rollups_15m():
    ensure_platform_tables()

    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")
    since_minutes = _as_int(request.args.get("since_minutes"), 120)

    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    rows = db.session.execute(
        text(
            """
            SELECT
                to_timestamp(floor(extract(epoch from event_ts) / 900) * 900) AT TIME ZONE 'UTC' AS bucket_start,
                COUNT(*) AS samples,
                ROUND(AVG((payload_json->>'heart_rate')::numeric), 2) AS avg_heart_rate,
                ROUND(AVG((payload_json->>'systolic_bp')::numeric), 2) AS avg_systolic_bp,
                ROUND(AVG((payload_json->>'diastolic_bp')::numeric), 2) AS avg_diastolic_bp,
                ROUND(AVG((payload_json->>'spo2')::numeric), 2) AS avg_spo2
            FROM vitals_events
            WHERE tenant_id = :t
              AND patient_id = :p
              AND created_at >= NOW() - (:mins || ' minutes')::interval
            GROUP BY 1
            ORDER BY 1 DESC
            """
        ),
        {"t": tenant_id, "p": patient_id, "mins": since_minutes},
    ).mappings().all()

    items = []
    for row in rows:
        items.append(
            {
                "bucket_start": row["bucket_start"].isoformat() if row["bucket_start"] else None,
                "samples": int(row["samples"] or 0),
                "avg_heart_rate": float(row["avg_heart_rate"]) if row["avg_heart_rate"] is not None else None,
                "avg_systolic_bp": float(row["avg_systolic_bp"]) if row["avg_systolic_bp"] is not None else None,
                "avg_diastolic_bp": float(row["avg_diastolic_bp"]) if row["avg_diastolic_bp"] is not None else None,
                "avg_spo2": float(row["avg_spo2"]) if row["avg_spo2"] is not None else None,
            }
        )

    return (
        jsonify(
            {
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "since_minutes": since_minutes,
                "rollups": items,
            }
        ),
        200,
    )
