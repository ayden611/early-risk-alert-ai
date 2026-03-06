import json
import os
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

try:
    import redis
except Exception:
    redis = None


DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL and redis is not None:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        redis_client = None


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
                    status TEXT NOT NULL DEFAULT 'open',
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
                CREATE TABLE IF NOT EXISTS stream_offsets (
                    stream_name TEXT PRIMARY KEY,
                    last_event_id BIGINT NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
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


def publish_event(channel, payload):
    if redis_client is None:
        return
    try:
        redis_client.publish(channel, json.dumps(payload, default=str))
    except Exception:
        pass


def get_last_offset(stream_name):
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT last_event_id
                FROM stream_offsets
                WHERE stream_name = :name
                """
            ),
            {"name": stream_name},
        ).mappings().first()

        return int(row["last_event_id"]) if row else 0


def set_last_offset(stream_name, last_event_id):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO stream_offsets (stream_name, last_event_id, updated_at)
                VALUES (:name, :last_event_id, NOW())
                ON CONFLICT (stream_name)
                DO UPDATE SET
                    last_event_id = EXCLUDED.last_event_id,
                    updated_at = NOW()
                """
            ),
            {"name": stream_name, "last_event_id": last_event_id},
        )


def fetch_new_vitals(last_id, batch_size=100):
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, tenant_id, patient_id, source, event_id, event_ts, created_at, payload_json
                FROM vitals_events
                WHERE id > :last_id
                ORDER BY id ASC
                LIMIT :lim
                """
            ),
            {"last_id": last_id, "lim": batch_size},
        ).mappings().all()
        return rows


def insert_alert(tenant_id, patient_id, alert, vitals):
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO alerts
                    (tenant_id, patient_id, alert_type, severity, status, message, payload_json)
                VALUES
                    (:t, :p, :atype, :sev, 'open', :msg, CAST(:payload AS jsonb))
                RETURNING id, created_at
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
        ).mappings().first()
        return result


def process_vitals_stream():
    stream_name = "vitals_events"
    last_id = get_last_offset(stream_name)
    rows = fetch_new_vitals(last_id)

    if not rows:
        return 0

    processed = 0
    newest_id = last_id

    for row in rows:
        newest_id = max(newest_id, int(row["id"]))

        vitals = row["payload_json"]
        if isinstance(vitals, str):
            vitals = json.loads(vitals)

        event_payload = {
            "id": row["id"],
            "tenant_id": row["tenant_id"],
            "patient_id": row["patient_id"],
            "source": row["source"],
            "event_id": row["event_id"],
            "event_ts": row["event_ts"].isoformat() if row["event_ts"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "vitals": vitals,
        }

        publish_event("stream:vitals", event_payload)
        publish_event(f"stream:vitals:{row['tenant_id']}", event_payload)
        publish_event(f"stream:vitals:{row['tenant_id']}:{row['patient_id']}", event_payload)

        alerts = detect_alerts(vitals)
        for alert in alerts:
            saved = insert_alert(row["tenant_id"], row["patient_id"], alert, vitals)
            alert_payload = {
                "id": saved["id"] if saved else None,
                "tenant_id": row["tenant_id"],
                "patient_id": row["patient_id"],
                "alert_type": alert["alert_type"],
                "severity": alert["severity"],
                "message": alert["message"],
                "created_at": saved["created_at"].isoformat() if saved and saved["created_at"] else utcnow_iso(),
                "payload": vitals,
            }
            publish_event("stream:alerts", alert_payload)
            publish_event(f"stream:alerts:{row['tenant_id']}", alert_payload)
            publish_event(f"stream:alerts:{row['tenant_id']}:{row['patient_id']}", alert_payload)

        processed += 1

    set_last_offset(stream_name, newest_id)
    return processed


def main():
    print(f"[{utcnow_iso()}] stream worker starting")
    ensure_tables()

    poll_seconds = float(os.getenv("STREAM_POLL_SECONDS", "2"))

    while True:
        try:
            processed = process_vitals_stream()
            if processed:
                print(f"[{utcnow_iso()}] processed {processed} new vitals events")
            else:
                time.sleep(poll_seconds)
        except Exception as e:
            print(f"[{utcnow_iso()}] worker loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
