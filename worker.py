# worker.py
import json
import os
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text

# Optional Redis Streams
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if not url:
        raise RuntimeError("DATABASE_URL is required for worker")
    return url


def _engine():
    return create_engine(
        _db_url(),
        pool_pre_ping=True,
        pool_size=_env_int("DB_POOL_SIZE", 5),
        max_overflow=_env_int("DB_MAX_OVERFLOW", 10),
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_event_ts(event_ts: str) -> Optional[datetime]:
    if not event_ts:
        return None
    try:
        dt = datetime.fromisoformat(event_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _partition_for_patient(patient_id: str, shards: int) -> int:
    if shards <= 1:
        return 0
    h = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % shards


def _stream_key(tenant_id: str, shard: int) -> str:
    return f"vitals:{tenant_id}:shard:{shard}"


def _redis_client():
    url = os.getenv("REDIS_URL", "").strip()
    if not url or redis is None:
        return None
    r = redis.Redis.from_url(url, decode_responses=True)
    r.ping()
    return r


def _ensure_tables(conn):
    conn.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS vitals_events (
          id BIGSERIAL PRIMARY KEY,
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'manual',
          event_ts TIMESTAMPTZ NULL,
          payload_json JSONB NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
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
          details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        )
    )
    conn.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS vitals_rollups_15m (
          tenant_id TEXT NOT NULL,
          patient_id TEXT NOT NULL,
          bucket_start TIMESTAMPTZ NOT NULL,
          cnt INTEGER NOT NULL DEFAULT 0,
          avg_hr DOUBLE PRECISION NULL,
          min_spo2 DOUBLE PRECISION NULL,
          max_systolic DOUBLE PRECISION NULL,
          max_diastolic DOUBLE PRECISION NULL,
          last_event_ts TIMESTAMPTZ NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          PRIMARY KEY (tenant_id, patient_id, bucket_start)
        );
        """
        )
    )
    conn.execute(
        text(
            """
        CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
        ON alerts(tenant_id, patient_id, created_at DESC);
        """
        )
    )
    conn.commit()


def _detect_anomalies(vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    if hr is not None and (hr < 45 or hr > 120):
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn" if hr <= 40 or hr >= 130 else "info",
                "message": f"Abnormal heart rate: {hr}",
                "details": {"metric": "heart_rate", "value": hr},
            }
        )
    if sys_bp is not None and sys_bp >= 160:
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn",
                "message": f"High systolic BP: {sys_bp}",
                "details": {"metric": "systolic_bp", "value": sys_bp},
            }
        )
    if dia_bp is not None and dia_bp >= 100:
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn",
                "message": f"High diastolic BP: {dia_bp}",
                "details": {"metric": "diastolic_bp", "value": dia_bp},
            }
        )
    if spo2 is not None and spo2 <= 92:
        alerts.append(
            {
                "alert_type": "anomaly",
                "severity": "warn" if spo2 <= 90 else "info",
                "message": f"Low SpO2: {spo2}",
                "details": {"metric": "spo2", "value": spo2},
            }
        )
    return alerts


def _bucket_start_15m(dt: datetime) -> datetime:
    # floor to 15-minute boundary
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def _update_rollup(conn, tenant_id: str, patient_id: str, event_dt: datetime, vitals: Dict[str, Any]):
    bucket = _bucket_start_15m(event_dt)

    hr = _as_float(vitals.get("heart_rate"))
    sys_bp = _as_float(vitals.get("systolic_bp"))
    dia_bp = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    # Upsert with incremental updates
    conn.execute(
        text(
            """
        INSERT INTO vitals_rollups_15m
          (tenant_id, patient_id, bucket_start, cnt, avg_hr, min_spo2, max_systolic, max_diastolic, last_event_ts, updated_at)
        VALUES
          (:t, :p, :b, 1, :hr, :spo2, :sys, :dia, :evt, NOW())
        ON CONFLICT (tenant_id, patient_id, bucket_start)
        DO UPDATE SET
          cnt = vitals_rollups_15m.cnt + 1,
          avg_hr = CASE
            WHEN :hr IS NULL THEN vitals_rollups_15m.avg_hr
            WHEN vitals_rollups_15m.avg_hr IS NULL THEN :hr
            ELSE (vitals_rollups_15m.avg_hr * vitals_rollups_15m.cnt + :hr) / (vitals_rollups_15m.cnt + 1)
          END,
          min_spo2 = CASE
            WHEN :spo2 IS NULL THEN vitals_rollups_15m.min_spo2
            WHEN vitals_rollups_15m.min_spo2 IS NULL THEN :spo2
            ELSE LEAST(vitals_rollups_15m.min_spo2, :spo2)
          END,
          max_systolic = CASE
            WHEN :sys IS NULL THEN vitals_rollups_15m.max_systolic
            WHEN vitals_rollups_15m.max_systolic IS NULL THEN :sys
            ELSE GREATEST(vitals_rollups_15m.max_systolic, :sys)
          END,
          max_diastolic = CASE
            WHEN :dia IS NULL THEN vitals_rollups_15m.max_diastolic
            WHEN vitals_rollups_15m.max_diastolic IS NULL THEN :dia
            ELSE GREATEST(vitals_rollups_15m.max_diastolic, :dia)
          END,
          last_event_ts = GREATEST(COALESCE(vitals_rollups_15m.last_event_ts, :evt), :evt),
          updated_at = NOW()
        """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "b": bucket,
            "hr": hr,
            "spo2": spo2,
            "sys": sys_bp,
            "dia": dia_bp,
            "evt": event_dt,
        },
    )


def _write_alerts(conn, tenant_id: str, patient_id: str, alerts: List[Dict[str, Any]]):
    for a in alerts:
        conn.execute(
            text(
                """
            INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, details_json)
            VALUES (:t, :p, :at, :sev, :msg, :d::jsonb)
            """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "at": a["alert_type"],
                "sev": a["severity"],
                "msg": a["message"],
                "d": json.dumps(a.get("details") or {}),
            },
        )


def _process_event(conn, tenant_id: str, patient_id: str, vitals: Dict[str, Any], event_ts: Optional[str]):
    event_dt = _parse_event_ts(event_ts or "") or _utcnow()

    alerts = _detect_anomalies(vitals)
    if alerts:
        _write_alerts(conn, tenant_id, patient_id, alerts)

    _update_rollup(conn, tenant_id, patient_id, event_dt, vitals)


def main():
    print("Worker starting...")
    eng = _engine()

    with eng.connect() as conn:
        _ensure_tables(conn)

    r = None
    try:
        r = _redis_client()
    except Exception:
        r = None

    tenant_id = os.getenv("TENANT_ID", "demo")
    shards = _env_int("STREAM_SHARDS", 8)
    group = os.getenv("CONSUMER_GROUP", "vitals-cg")
    consumer = os.getenv("CONSUMER_NAME", f"worker-{os.getpid()}")
    block_ms = _env_int("STREAM_BLOCK_MS", 5000)
    count = _env_int("STREAM_READ_COUNT", 100)

    if r is None:
        # If no redis, worker can still run; but it won't have a queue to read.
        # You can add DB-queue fallback later if you want.
        print("REDIS_URL not set or redis not available. Worker will idle.")
        while True:
            time.sleep(5)

    # Create consumer groups per shard (safe if exists)
    for shard in range(shards):
        key = _stream_key(tenant_id, shard)
        try:
            r.xgroup_create(key, group, id="0-0", mkstream=True)
        except Exception:
            # group probably exists
            pass

    print(f"Consuming tenant={tenant_id} shards={shards} group={group} consumer={consumer}")

    while True:
        streams = {_stream_key(tenant_id, s): ">" for s in range(shards)}
        try:
            resp = r.xreadgroup(group, consumer, streams, count=count, block=block_ms)
        except Exception as e:
            print(f"xreadgroup error: {e}")
            time.sleep(2)
            continue

        if not resp:
            continue

        # resp format: [(stream_name, [(msg_id, {field:val}) ...]) ...]
        for stream_name, messages in resp:
            for msg_id, fields in messages:
                try:
                    t = fields.get("tenant_id") or tenant_id
                    p = fields.get("patient_id") or "unknown"
                    vitals_json = fields.get("vitals_json") or "{}"
                    event_ts = fields.get("event_ts") or ""

                    vitals = json.loads(vitals_json) if isinstance(vitals_json, str) else {}
                    if not isinstance(vitals, dict):
                        vitals = {}

                    with eng.begin() as conn:
                        _process_event(conn, t, p, vitals, event_ts)

                    # Ack only after DB commit
                    r.xack(stream_name, group, msg_id)
                except Exception as e:
                    # Don't ack on failure -> message can be retried
                    print(f"process error stream={stream_name} id={msg_id}: {e}")
                    time.sleep(0.1)


if __name__ == "__main__":
    main()
