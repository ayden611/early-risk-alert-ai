# worker.py
import os
import json
import time
from typing import Any, Dict, Optional, List, Tuple

from sqlalchemy import text

from era import create_app
from era.extensions import db

# Optional Redis stream support (won't crash if redis isn't installed)
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


REDIS_URL = os.getenv("REDIS_URL", "")
STREAM_NAME_PREFIX = os.getenv("STREAM_NAME_PREFIX", "vitals")
STREAM_SHARDS = int(os.getenv("STREAM_SHARDS", "16"))
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "200000"))

CONSUMER_GROUP = os.getenv("STREAM_CONSUMER_GROUP", "vitals-workers")
CONSUMER_NAME = os.getenv("STREAM_CONSUMER_NAME", f"worker-{os.getpid()}")
BLOCK_MS = int(os.getenv("STREAM_BLOCK_MS", "2000"))
BATCH_COUNT = int(os.getenv("STREAM_BATCH_COUNT", "100"))

_r = None


def _redis() -> Optional["redis.Redis"]:
    global _r
    if not REDIS_URL or redis is None:
        return None
    if _r is None:
        _r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _r


def _ensure_tables() -> None:
    db.session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS vitals_events (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                source TEXT DEFAULT 'api',
                event_ts TIMESTAMPTZ NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                alert_type TEXT NOT NULL DEFAULT 'anomaly',
                severity TEXT NOT NULL DEFAULT 'info',
                message TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
            CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
            ON alerts (tenant_id, patient_id, created_at DESC);
            """
        )
    )
    db.session.commit()


def ensure_groups_exist() -> None:
    r = _redis()
    if r is None:
        return

    for shard in range(STREAM_SHARDS):
        sname = f"{STREAM_NAME_PREFIX}:{shard:03d}"
        try:
            r.xgroup_create(name=sname, groupname=CONSUMER_GROUP, id="$", mkstream=True)
        except Exception as e:
            # BUSYGROUP is ok
            if "BUSYGROUP" not in str(e):
                raise


def consume_batch() -> List[Tuple[str, str, Dict[str, str]]]:
    r = _redis()
    if r is None:
        return []

    streams = {f"{STREAM_NAME_PREFIX}:{shard:03d}": ">" for shard in range(STREAM_SHARDS)}
    resp = r.xreadgroup(
        groupname=CONSUMER_GROUP,
        consumername=CONSUMER_NAME,
        streams=streams,
        count=BATCH_COUNT,
        block=BLOCK_MS,
    )

    out: List[Tuple[str, str, Dict[str, str]]] = []
    for sname, messages in resp:
        for msg_id, fields in messages:
            out.append((sname, msg_id, fields))
    return out


def ack(stream_name: str, message_id: str) -> None:
    r = _redis()
    if r is None:
        return
    r.xack(stream_name, CONSUMER_GROUP, message_id)


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_anomalies(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    hr = _as_float(v.get("heart_rate"))
    sys = _as_float(v.get("systolic_bp"))
    dia = _as_float(v.get("diastolic_bp"))
    spo2 = _as_float(v.get("spo2"))

    if hr is not None and (hr >= 120 or hr <= 40):
        alerts.append({"alert_type": "anomaly", "severity": "warn", "message": f"Heart rate out of range: {hr}"})

    if sys is not None and sys >= 160:
        alerts.append({"alert_type": "anomaly", "severity": "warn", "message": f"Systolic BP high: {sys}"})

    if dia is not None and dia >= 100:
        alerts.append({"alert_type": "anomaly", "severity": "warn", "message": f"Diastolic BP high: {dia}"})

    if spo2 is not None and spo2 <= 92:
        alerts.append({"alert_type": "anomaly", "severity": "critical", "message": f"SpO2 low: {spo2}"})

    return alerts


def main():
    app = create_app()

    with app.app_context():
        _ensure_tables()

        # Redis is optional — if not set, worker stays alive but idle (won't crash deploy)
        if not REDIS_URL or redis is None:
            print("⚠️ Worker running without Redis streaming (set REDIS_URL + install redis to enable).")
            while True:
                time.sleep(5)

        ensure_groups_exist()
        print(f"✅ Worker streaming ready: shards={STREAM_SHARDS} group={CONSUMER_GROUP} name={CONSUMER_NAME}")

        while True:
            batch = consume_batch()
            if not batch:
                continue

            for stream_name, msg_id, fields in batch:
                try:
                    tenant_id = fields.get("tenant_id", "demo")
                    patient_id = fields.get("patient_id", "unknown")
                    vitals_json = fields.get("vitals_json", "{}")
                    vitals = json.loads(vitals_json) if vitals_json else {}

                    alerts = detect_anomalies(vitals)

                    for a in alerts:
                        db.session.execute(
                            text(
                                """
                                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, created_at)
                                VALUES (:t, :p, :at, :sev, :msg, NOW())
                                """
                            ),
                            {
                                "t": tenant_id,
                                "p": patient_id,
                                "at": a.get("alert_type", "anomaly"),
                                "sev": a.get("severity", "info"),
                                "msg": a.get("message", ""),
                            },
                        )

                    db.session.commit()
                    ack(stream_name, msg_id)

                except Exception as e:
                    db.session.rollback()
                    print(f"❌ Failed processing {stream_name} {msg_id}: {e}")

            time.sleep(0.01)


if __name__ == "__main__":
    main()
