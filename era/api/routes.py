import json
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

api_bp = Blueprint("api", __name__)


@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def _as_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def ensure_api_tables():
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
                message TEXT NOT NULL DEFAULT '',
                payload_json JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

    db.session.commit()


def _detect_alerts(vitals):
    alerts = []

    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    if hr is not None and hr >= 120:
        alerts.append(
            {
                "alert_type": "tachycardia",
                "severity": "high",
                "message": f"Heart rate elevated at {hr:.0f} bpm",
            }
        )

    if sys_bp is not None and sys_bp >= 160:
        alerts.append(
            {
                "alert_type": "hypertension",
                "severity": "high",
                "message": f"Systolic blood pressure elevated at {sys_bp:.0f}",
            }
        )

    if dia_bp is not None and dia_bp >= 100:
        alerts.append(
            {
                "alert_type": "hypertension",
                "severity": "high",
                "message": f"Diastolic blood pressure elevated at {dia_bp:.0f}",
            }
        )

    if spo2 is not None and spo2 < 92:
        alerts.append(
            {
                "alert_type": "low_oxygen",
                "severity": "critical",
                "message": f"SpO2 low at {spo2:.0f}%",
            }
        )

    return alerts


@api_bp.get("/healthz")
def api_healthz():
    db_ok = True
    db_error = None

    try:
        ensure_api_tables()
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "service": "early-risk-alert-api",
            "db_ok": db_ok,
            "db_error": db_error,
        }
    ), (200 if db_ok else 503)


@api_bp.post("/vitals")
def ingest_vitals():
    ensure_api_tables()

    body = request.get_json(force=True) or {}

    tenant_id = str(body.get("tenant_id", "demo"))
    patient_id = str(body.get("patient_id", "unknown"))
    source = str(body.get("source", "manual"))
    event_id = body.get("event_id")
    event_ts = body.get("event_ts")
    vitals = body.get("vitals") or {}

    db.session.execute(
        text(
            """
            INSERT INTO vitals_events
                (tenant_id, patient_id, source, event_id, event_ts, payload_json)
            VALUES
                (:t, :p, :s, :eid, COALESCE(CAST(:ets AS timestamptz), NOW()), CAST(:j AS jsonb))
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "eid": event_id,
            "ets": event_ts,
            "j": json.dumps(vitals),
        },
    )

    detected = _detect_alerts(vitals)

    for alert in detected:
        db.session.execute(
            text(
                """
                INSERT INTO alerts
                    (tenant_id, patient_id, alert_type, severity, message, payload_json)
                VALUES
                    (:t, :p, :atype, :sev, :msg, CAST(:payload AS jsonb))
                """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "atype": alert["alert_type"],
                "sev": alert["severity"],
                "msg": alert["message"],
                "payload": json.dumps(vitals),
            },
        )

    db.session.commit()

    return jsonify(
        {
            "ok": True,
            "accepted": True,
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "received_at": _utcnow_iso(),
            "alerts_created": len(detected),
        }
    ), 200


@api_bp.get("/vitals/latest")
def latest_vitals():
    ensure_api_tables()

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

    return jsonify(
        {
            "tenant_id": row["tenant_id"],
            "patient_id": row["patient_id"],
            "source": row["source"],
            "event_id": row["event_id"],
            "event_ts": row["event_ts"].isoformat() if row["event_ts"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "vitals": payload,
        }
    ), 200


@api_bp.get("/alerts")
def get_alerts():
    ensure_api_tables()

    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")
    limit = int(request.args.get("limit", 50))

    if not patient_id:
        return jsonify({"error": "patient_id required"}), 400

    rows = db.session.execute(
        text(
            """
            SELECT alert_type, severity, message, created_at, payload_json
            FROM alerts
            WHERE tenant_id = :t AND patient_id = :p
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {"t": tenant_id, "p": patient_id, "lim": limit},
    ).mappings().all()

    items = []
    for row in rows:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)

        items.append(
            {
                "alert_type": row["alert_type"],
                "severity": row["severity"],
                "message": row["message"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "payload": payload,
            }
        )

    return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "alerts": items}), 200


@api_bp.get("/rollups/15m")
def rollups_15m():
    ensure_api_tables()

    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")
    since_minutes = int(request.args.get("since_minutes", 120))

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

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "since_minutes": since_minutes,
            "rollups": items,
        }
    ), 200
