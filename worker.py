import json
import os
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def as_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def ensure_tables():
    with engine.begin() as conn:
        conn.execute(
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

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_vitals_events_tenant_patient_created
                ON vitals_events (tenant_id, patient_id, created_at DESC);
                """
            )
        )

        conn.execute(
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

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
                ON alerts (tenant_id, patient_id, created_at DESC);
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS risk_jobs (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    result_json JSONB,
                    error TEXT
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_risk_jobs_status_id
                ON risk_jobs (status, id);
                """
            )
        )


def detect_alerts(vitals):
    alerts = []

    hr = as_float(vitals.get("heart_rate"))
    sys_bp = as_float(vitals.get("systolic_bp"))
    dia_bp = as_float(vitals.get("diastolic_bp"))
    spo2 = as_float(vitals.get("spo2"))

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


def build_health_reasoning(payload):
    vitals = payload.get("vitals") or payload
    hr = as_float(vitals.get("heart_rate"))
    sys_bp = as_float(vitals.get("systolic_bp"))
    dia_bp = as_float(vitals.get("diastolic_bp"))
    spo2 = as_float(vitals.get("spo2"))

    findings = []
    risk_score = 0.1

    if hr is not None and hr >= 120:
        findings.append("tachycardia")
        risk_score += 0.25

    if sys_bp is not None and sys_bp >= 160:
        findings.append("high systolic blood pressure")
        risk_score += 0.25

    if dia_bp is not None and dia_bp >= 100:
        findings.append("high diastolic blood pressure")
        risk_score += 0.2

    if spo2 is not None and spo2 < 92:
        findings.append("low oxygen saturation")
        risk_score += 0.3

    risk_score = min(risk_score, 0.99)

    if risk_score >= 0.75:
        level = "high"
    elif risk_score >= 0.4:
        level = "moderate"
    else:
        level = "low"

    summary = (
        "No major abnormalities detected."
        if not findings
        else "Detected " + ", ".join(findings) + "."
    )

    return {
        "generated_at": utcnow_iso(),
        "risk_score": round(risk_score, 3),
        "risk_level": level,
        "summary": summary,
        "findings": findings,
    }


def fetch_next_job():
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, tenant_id, patient_id, payload_json
                FROM risk_jobs
                WHERE status = 'pending'
                ORDER BY id
                LIMIT 1
                """
            )
        ).mappings().first()

        if not row:
            return None

        updated = conn.execute(
            text(
                """
                UPDATE risk_jobs
                SET status = 'processing',
                    started_at = NOW()
                WHERE id = :id AND status = 'pending'
                """
            ),
            {"id": row["id"]},
        )

        if updated.rowcount == 0:
            return None

        return row


def insert_alerts(tenant_id, patient_id, vitals, alerts):
    if not alerts:
        return

    with engine.begin() as conn:
        for alert in alerts:
            conn.execute(
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


def complete_job(job_id, result):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE risk_jobs
                SET status = 'done',
                    finished_at = NOW(),
                    result_json = CAST(:result AS jsonb),
                    error = NULL
                WHERE id = :id
                """
            ),
            {"id": job_id, "result": json.dumps(result)},
        )


def fail_job(job_id, error_message):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE risk_jobs
                SET status = 'failed',
                    finished_at = NOW(),
                    error = :err
                WHERE id = :id
                """
            ),
            {"id": job_id, "err": error_message[:4000]},
        )


def process_job(job):
    payload = job["payload_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    vitals = payload.get("vitals") or payload
    alerts = detect_alerts(vitals)
    reasoning = build_health_reasoning(payload)

    insert_alerts(job["tenant_id"], job["patient_id"], vitals, alerts)

    result = {
        "ok": True,
        "tenant_id": job["tenant_id"],
        "patient_id": job["patient_id"],
        "alerts_created": len(alerts),
        "alerts": alerts,
        "ai_analysis": reasoning,
    }

    complete_job(job["id"], result)
    print(f"[{utcnow_iso()}] processed job {job['id']} for patient {job['patient_id']}")


def main():
    print(f"[{utcnow_iso()}] worker starting")
    ensure_tables()

    poll_seconds = float(os.getenv("WORKER_POLL_SECONDS", "2"))

    while True:
        try:
            job = fetch_next_job()
            if not job:
                time.sleep(poll_seconds)
                continue

            try:
                process_job(job)
            except Exception as e:
                fail_job(job["id"], str(e))
                print(f"[{utcnow_iso()}] job {job['id']} failed: {e}")

        except Exception as e:
            print(f"[{utcnow_iso()}] worker loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
