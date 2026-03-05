# worker.py
import json
import os
import time
from datetime import datetime, timezone

from sqlalchemy import text

from era import create_app
from era.extensions import db

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


def _parse_iso(ts: str):
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def detect_anomalies(vitals):
    alerts = []
    try:
        hr = float(vitals.get("heart_rate")) if vitals.get("heart_rate") is not None else None
        sys_bp = float(vitals.get("systolic_bp")) if vitals.get("systolic_bp") is not None else None
        dia_bp = float(vitals.get("diastolic_bp")) if vitals.get("diastolic_bp") is not None else None
        spo2 = float(vitals.get("spo2")) if vitals.get("spo2") is not None else None
    except Exception:
        hr = sys_bp = dia_bp = spo2 = None

    if hr is not None and hr >= 130:
        alerts.append(("warn", f"High heart rate: {hr:.0f}"))
    if sys_bp is not None and sys_bp >= 160:
        alerts.append(("warn", f"High systolic BP: {sys_bp:.0f}"))
    if dia_bp is not None and dia_bp >= 100:
        alerts.append(("warn", f"High diastolic BP: {dia_bp:.0f}"))
    if spo2 is not None and spo2 <= 92:
        alerts.append(("warn", f"Low SpO2: {spo2:.0f}"))

    return alerts


def ensure_tables():
    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS vitals_rollups_15m (
            tenant_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            bucket_start TIMESTAMPTZ NOT NULL,
            count_events INTEGER NOT NULL DEFAULT 0,
            avg_hr DOUBLE PRECISION,
            min_spo2 DOUBLE PRECISION,
            max_sys DOUBLE PRECISION,
            max_dia DOUBLE PRECISION,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, patient_id, bucket_start)
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
            alert_type TEXT DEFAULT 'anomaly',
            severity TEXT DEFAULT 'info',
            message TEXT DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        )
    )
    db.session.execute(
        text(
            """
        CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
        ON alerts(tenant_id, patient_id, created_at DESC);
        """
        )
    )
    db.session.commit()


def redis_client():
    url = os.getenv("REDIS_URL")
    if not url or redis is None:
        return None
    return redis.Redis.from_url(url, decode_responses=True)


def stream_key(tenant_id: str) -> str:
    return f"vitals_stream:{tenant_id}"


def bucket_15m(dt: datetime) -> datetime:
    # floor to 15m
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def update_rollup(tenant_id: str, patient_id: str, event_ts: datetime, vitals: dict):
    b = bucket_15m(event_ts)

    hr = vitals.get("heart_rate")
    spo2 = vitals.get("spo2")
    sys_bp = vitals.get("systolic_bp")
    dia_bp = vitals.get("diastolic_bp")

    # Upsert rollup — keep it simple and fast
    db.session.execute(
        text(
            """
        INSERT INTO vitals_rollups_15m (tenant_id, patient_id, bucket_start, count_events, avg_hr, min_spo2, max_sys, max_dia, updated_at)
        VALUES (:t,:p,:b, 1,
                CASE WHEN :hr IS NULL THEN NULL ELSE :hr::double precision END,
                CASE WHEN :spo2 IS NULL THEN NULL ELSE :spo2::double precision END,
                CASE WHEN :sys IS NULL THEN NULL ELSE :sys::double precision END,
                CASE WHEN :dia IS NULL THEN NULL ELSE :dia::double precision END,
                NOW()
        )
        ON CONFLICT (tenant_id, patient_id, bucket_start)
        DO UPDATE SET
            count_events = vitals_rollups_15m.count_events + 1,
            avg_hr = CASE
                WHEN EXCLUDED.avg_hr IS NULL THEN vitals_rollups_15m.avg_hr
                WHEN vitals_rollups_15m.avg_hr IS NULL THEN EXCLUDED.avg_hr
                ELSE ((vitals_rollups_15m.avg_hr * vitals_rollups_15m.count_events) + EXCLUDED.avg_hr) / (vitals_rollups_15m.count_events + 1)
            END,
            min_spo2 = CASE
                WHEN EXCLUDED.min_spo2 IS NULL THEN vitals_rollups_15m.min_spo2
                WHEN vitals_rollups_15m.min_spo2 IS NULL THEN EXCLUDED.min_spo2
                ELSE LEAST(vitals_rollups_15m.min_spo2, EXCLUDED.min_spo2)
            END,
            max_sys = CASE
                WHEN EXCLUDED.max_sys IS NULL THEN vitals_rollups_15m.max_sys
                WHEN vitals_rollups_15m.max_sys IS NULL THEN EXCLUDED.max_sys
                ELSE GREATEST(vitals_rollups_15m.max_sys, EXCLUDED.max_sys)
            END,
            max_dia = CASE
                WHEN EXCLUDED.max_dia IS NULL THEN vitals_rollups_15m.max_dia
                WHEN vitals_rollups_15m.max_dia IS NULL THEN EXCLUDED.max_dia
                ELSE GREATEST(vitals_rollups_15m.max_dia, EXCLUDED.max_dia)
            END,
            updated_at = NOW()
        """
        ),
        {"t": tenant_id, "p": patient_id, "b": b, "hr": hr, "spo2": spo2, "sys": sys_bp, "dia": dia_bp},
    )
    db.session.commit()


def write_alert(tenant_id: str, patient_id: str, severity: str, message: str):
    db.session.execute(
        text(
            """
        INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message)
        VALUES (:t,:p,'anomaly',:sev,:msg)
        """
        ),
        {"t": tenant_id, "p": patient_id, "sev": severity, "msg": message},
    )
    db.session.commit()


def main():
    app = create_app()
    with app.app_context():
        ensure_tables()

        r = redis_client()
        if r is None:
            print("REDIS_URL not set or redis not installed. Worker idle.")
            while True:
                time.sleep(10)

        tenant_id = os.getenv("WORKER_TENANT", "demo")
        stream = stream_key(tenant_id)

        group = os.getenv("STREAM_GROUP", "vitals_workers")
        consumer = os.getenv("STREAM_CONSUMER", f"c-{os.getpid()}")
        block_ms = int(os.getenv("STREAM_BLOCK_MS", "2000"))
        batch = int(os.getenv("STREAM_BATCH", "50"))

        # Create consumer group (safe if already exists)
        try:
            r.xgroup_create(stream, group, id="0-0", mkstream=True)
        except Exception:
            pass

        print(f"Worker consuming stream={stream} group={group} consumer={consumer}")

        # Consume loop
        while True:
            try:
                resp = r.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream: ">"},
                    count=batch,
                    block=block_ms,
                )
                if not resp:
                    continue

                # resp: [(stream, [(id, {fields})...])]
                for _, messages in resp:
                    for msg_id, fields in messages:
                        tenant = fields.get("tenant_id", tenant_id)
                        patient = fields.get("patient_id", "unknown")
                        event_ts = _parse_iso(fields.get("event_ts", "")) or datetime.now(timezone.utc)

                        payload = fields.get("payload", "{}")
                        try:
                            vitals = json.loads(payload) if payload else {}
                        except Exception:
                            vitals = {}

                        # 1) rollup
                        update_rollup(tenant, patient, event_ts, vitals)

                        # 2) anomaly detect
                        for sev, msg in detect_anomalies(vitals):
                            write_alert(tenant, patient, sev, msg)

                        # ACK
                        r.xack(stream, group, msg_id)

            except Exception as e:
                print(f"Worker loop error: {e}")
                time.sleep(2)


if __name__ == "__main__":
    main()
