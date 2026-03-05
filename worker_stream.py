import json
import os
import time
from redis import Redis

from era import create_app
from era.extensions import db
from sqlalchemy import text

from era.streaming import REDIS_URL, VITALS_STREAM_KEY, VITALS_CONSUMER_GROUP

# Import helpers from routes.py (or duplicate them here if you prefer)
# We’ll duplicate tiny essentials here to avoid circular imports.

def _as_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def detect_anomalies(vitals):
    alerts = []
    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp") or vitals.get("sys_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp") or vitals.get("dia_bp"))
    spo2 = _as_float(vitals.get("spo2"))
    temp_c = _as_float(vitals.get("temp_c"))
    temp_f = _as_float(vitals.get("temp_f"))
    if temp_f is None and temp_c is not None:
        temp_f = (temp_c * 9.0 / 5.0) + 32.0

    if hr is not None:
        if hr >= 130:
            alerts.append({"type": "tachycardia", "severity": "high", "message": f"Heart rate very high ({hr})."})
        elif hr >= 110:
            alerts.append({"type": "tachycardia", "severity": "medium", "message": f"Heart rate elevated ({hr})."})
        elif hr <= 40:
            alerts.append({"type": "bradycardia", "severity": "high", "message": f"Heart rate very low ({hr})."})
        elif hr <= 50:
            alerts.append({"type": "bradycardia", "severity": "medium", "message": f"Heart rate low ({hr})."})

    if sys_bp is not None and dia_bp is not None:
        if sys_bp >= 180 or dia_bp >= 120:
            alerts.append({"type": "hypertensive_crisis", "severity": "high",
                           "message": f"BP crisis range ({sys_bp}/{dia_bp})."})
        elif sys_bp >= 160 or dia_bp >= 100:
            alerts.append({"type": "high_bp", "severity": "medium", "message": f"High BP ({sys_bp}/{dia_bp})."})
        elif sys_bp <= 90 or dia_bp <= 60:
            alerts.append({"type": "low_bp", "severity": "medium", "message": f"Low BP ({sys_bp}/{dia_bp})."})

    if spo2 is not None:
        if spo2 < 90:
            alerts.append({"type": "low_spo2", "severity": "high", "message": f"Low oxygen saturation ({spo2}%)."})
        elif spo2 < 93:
            alerts.append({"type": "low_spo2", "severity": "medium", "message": f"Borderline oxygen saturation ({spo2}%)."})

    if temp_f is not None:
        if temp_f >= 103:
            alerts.append({"type": "fever", "severity": "high", "message": f"High fever ({round(temp_f, 1)}F)."})
        elif temp_f >= 100.4:
            alerts.append({"type": "fever", "severity": "medium", "message": f"Fever ({round(temp_f, 1)}F)."})
        elif temp_f <= 95:
            alerts.append({"type": "hypothermia", "severity": "high", "message": f"Low temperature ({round(temp_f, 1)}F)."})

    return alerts


def ensure_tables():
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
        CREATE INDEX IF NOT EXISTS idx_vitals_events_tenant_patient_created
        ON vitals_events(tenant_id, patient_id, created_at DESC);
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
        CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
        ON alerts(tenant_id, patient_id, created_at DESC);
    """))
    db.session.commit()


def main():
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL not set")

    app = create_app()
    r = Redis.from_url(REDIS_URL, decode_responses=True)

    consumer_name = os.getenv("WORKER_NAME") or f"worker-{os.getpid()}"

    with app.app_context():
        ensure_tables()

        # Create consumer group if missing
        try:
            r.xgroup_create(VITALS_STREAM_KEY, VITALS_CONSUMER_GROUP, id="0-0", mkstream=True)
        except Exception as e:
            # BUSYGROUP is expected if already exists
            if "BUSYGROUP" not in str(e):
                raise

        print(f"[stream-worker] group={VITALS_CONSUMER_GROUP} consumer={consumer_name} stream={VITALS_STREAM_KEY}")

        while True:
            # Read up to N messages, block briefly
            resp = r.xreadgroup(
                groupname=VITALS_CONSUMER_GROUP,
                consumername=consumer_name,
                streams={VITALS_STREAM_KEY: ">"},
                count=100,
                block=2000,
            )

            if not resp:
                continue

            for stream_name, messages in resp:
                for msg_id, fields in messages:
                    try:
                        tenant_id = fields.get("tenant_id", "demo-tenant")
                        patient_id = fields.get("patient_id", "unknown")
                        source = fields.get("source", "manual")
                        event_ts = fields.get("event_ts") or None
                        vitals = json.loads(fields.get("vitals_json") or "{}")

                        # Save vitals
                        db.session.execute(
                            text("""
                                INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, payload_json)
                                VALUES (:t, :p, :s, :ts, :j)
                            """),
                            {"t": tenant_id, "p": patient_id, "s": source, "ts": event_ts, "j": json.dumps(vitals)},
                        )

                        # Detect + save alerts
                        alerts = detect_anomalies(vitals)
                        for a in alerts:
                            db.session.execute(
                                text("""
                                    INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message)
                                    VALUES (:t, :p, :at, :sev, :msg)
                                """),
                                {
                                    "t": tenant_id,
                                    "p": patient_id,
                                    "at": a.get("type", "unknown"),
                                    "sev": a.get("severity", "medium"),
                                    "msg": a.get("message", ""),
                                },
                            )

                        db.session.commit()

                        # Ack message (remove from pending list for this group)
                        r.xack(VITALS_STREAM_KEY, VITALS_CONSUMER_GROUP, msg_id)

                    except Exception as e:
                        db.session.rollback()
                        # Don’t ack so it remains pending; you can add retries/DLQ later
                        print(f"[stream-worker] ERROR msg_id={msg_id} err={e}")

if __name__ == "__main__":
    main()
