import os
import json
import time
import random
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

TENANT_ID = "demo"

PATIENTS = [
    {"patient_id": "p101", "baseline_hr": 82, "baseline_sys": 126, "baseline_dia": 82, "baseline_spo2": 97},
    {"patient_id": "p202", "baseline_hr": 76, "baseline_sys": 122, "baseline_dia": 79, "baseline_spo2": 98},
    {"patient_id": "p303", "baseline_hr": 88, "baseline_sys": 132, "baseline_dia": 86, "baseline_spo2": 96},
    {"patient_id": "p404", "baseline_hr": 72, "baseline_sys": 118, "baseline_dia": 76, "baseline_spo2": 99},
]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def clamp(value, low, high):
    return max(low, min(high, value))

def generate_vitals(p):
    event_mode = random.choices(
        ["normal", "watch", "critical"],
        weights=[60, 28, 12],
        k=1
    )[0]

    hr = p["baseline_hr"] + random.randint(-8, 8)
    sys_bp = p["baseline_sys"] + random.randint(-10, 10)
    dia_bp = p["baseline_dia"] + random.randint(-6, 6)
    spo2 = p["baseline_spo2"] + random.randint(-2, 1)

    alert_type = None
    alert_message = None
    severity = None

    if event_mode == "watch":
        bump = random.choice(["hr", "bp", "spo2"])
        if bump == "hr":
            hr += random.randint(18, 28)
            alert_type, alert_message, severity = "tachycardia", "Heart rate elevated", "high"
        elif bump == "bp":
            sys_bp += random.randint(18, 30)
            dia_bp += random.randint(10, 18)
            alert_type, alert_message, severity = "hypertension", "Blood pressure elevated", "high"
        else:
            spo2 -= random.randint(4, 7)
            alert_type, alert_message, severity = "low_spo2", "Oxygen saturation dropping", "high"

    elif event_mode == "critical":
        bump = random.choice(["hr", "bp", "spo2"])
        if bump == "hr":
            hr += random.randint(30, 45)
            alert_type, alert_message, severity = "tachycardia", "Heart rate critical", "critical"
        elif bump == "bp":
            sys_bp += random.randint(28, 40)
            dia_bp += random.randint(16, 24)
            alert_type, alert_message, severity = "hypertension", "Blood pressure critical", "critical"
        else:
            spo2 -= random.randint(8, 12)
            alert_type, alert_message, severity = "low_spo2", "Oxygen saturation critical", "critical"

    hr = clamp(hr, 45, 180)
    sys_bp = clamp(sys_bp, 90, 210)
    dia_bp = clamp(dia_bp, 50, 130)
    spo2 = clamp(spo2, 78, 100)

    return {
        "heart_rate": hr,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "spo2": spo2,
        "alert_type": alert_type,
        "alert_message": alert_message,
        "severity": severity,
    }

def ensure_tables():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vitals_events (
            id BIGSERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            payload_json JSONB NOT NULL
        );
        """))

        conn.execute(text("""
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

def insert_event(patient_id, payload):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO vitals_events (tenant_id, patient_id, payload_json)
            VALUES (:tenant_id, :patient_id, CAST(:payload_json AS JSONB))
            """),
            {
                "tenant_id": TENANT_ID,
                "patient_id": patient_id,
                "payload_json": json.dumps(payload),
            }
        )

def insert_alert(patient_id, alert_type, severity, message):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message)
            VALUES (:tenant_id, :patient_id, :alert_type, :severity, :message)
            """),
            {
                "tenant_id": TENANT_ID,
                "patient_id": patient_id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
            }
        )

def trim_old_data():
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM alerts
            WHERE id NOT IN (
                SELECT id FROM alerts
                ORDER BY created_at DESC
                LIMIT 100
            )
        """))
        conn.execute(text("""
            DELETE FROM vitals_events
            WHERE id NOT IN (
                SELECT id FROM vitals_events
                ORDER BY created_at DESC
                LIMIT 500
            )
        """))

def main():
    ensure_tables()
    print("Live dashboard feeder started...")

    while True:
        for p in PATIENTS:
            result = generate_vitals(p)
            payload = {
                "event_ts": utc_now(),
                "heart_rate": result["heart_rate"],
                "systolic_bp": result["systolic_bp"],
                "diastolic_bp": result["diastolic_bp"],
                "spo2": result["spo2"],
            }

            insert_event(p["patient_id"], payload)

            if result["alert_type"]:
                insert_alert(
                    p["patient_id"],
                    result["alert_type"],
                    result["severity"],
                    result["alert_message"],
                )

        trim_old_data()
        print("Inserted live vitals + alerts...")
        time.sleep(6)

if __name__ == "__main__":
    main()
