# worker.py
import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text

from era import create_app
from era.extensions import db

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


STREAM_KEY = os.getenv("VITALS_STREAM_KEY", "vitals:events")
DLQ_KEY = os.getenv("VITALS_DLX_KEY", "vitals:dlq")
GROUP = os.getenv("VITALS_CONSUMER_GROUP", "vitals-workers")
CONSUMER = os.getenv("VITALS_CONSUMER_NAME", f"worker-{os.getpid()}")
BLOCK_MS = int(os.getenv("WORKER_BLOCK_MS", "5000"))
BATCH = int(os.getenv("WORKER_BATCH", "50"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: Optional[str]) -> datetime:
    if not s:
        return _utcnow()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return _utcnow()


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _bucket_15m(ts: datetime) -> Tuple[datetime, datetime]:
    # floor to 15-minute boundary
    minute = (ts.minute // 15) * 15
    start = ts.replace(minute=minute, second=0, microsecond=0)
    end = start + timedelta(minutes=15)
    return start, end


def detect_anomalies(vitals: Dict[str, Any]) -> list[Dict[str, Any]]:
    alerts: list[Dict[str, Any]] = []
    hr = _as_float(vitals.get("heart_rate"))
    sys = _as_float(vitals.get("systolic_bp"))
    dia = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    if hr is not None and hr >= 130:
        alerts.append({"alert_type": "tachycardia", "severity": "warn", "message": f"High heart rate: {hr}"})
    if hr is not None and hr <= 40:
        alerts.append({"alert_type": "bradycardia", "severity": "warn", "message": f"Low heart rate: {hr}"})

    if sys is not None and dia is not None and (sys >= 180 or dia >= 120):
        alerts.append({"alert_type": "hypertensive_crisis", "severity": "high", "message": f"Very high BP: {sys}/{dia}"})
    elif sys is not None and dia is not None and (sys >= 140 or dia >= 90):
        alerts.append({"alert_type": "hypertension", "severity": "warn", "message": f"High BP: {sys}/{dia}"})

    if spo2 is not None and spo2 < 90:
        alerts.append({"alert_type": "low_spo2", "severity": "high", "message": f"Low SpO2: {spo2}"})
    elif spo2 is not None and spo2 < 92:
        alerts.append({"alert_type": "borderline_spo2", "severity": "warn", "message": f"Borderline SpO2: {spo2}"})

    return alerts


def redis_client():
    url = os.getenv("REDIS_URL")
    if not url or redis is None:
        return None
    return redis.Redis.from_url(url, decode_responses=True)


def ensure_group(r):
    try:
        r.xgroup_create(name=STREAM_KEY, groupname=GROUP, id="0-0", mkstream=True)
    except Exception as e:
        # Group exists is OK
        if "BUSYGROUP" in str(e):
            return
        raise


def write_dlq(r, payload: str, error: str):
    r.xadd(DLQ_KEY, {"payload": payload, "error": error, "ts": _utcnow().isoformat()}, maxlen=200000, approximate=True)


def process_one(message_id: str, payload_str: str) -> None:
    """
    Core worker logic:
    - idempotent event write
    - alerts
    - latest snapshot
    - 15m rollup upsert
    """
    payload = json.loads(payload_str)

    tenant_id = str(payload.get("tenant_id") or "demo")
    patient_id = str(payload.get("patient_id") or "unknown")
    source = str(payload.get("source") or "stream")
    event_id = str(payload.get("event_id") or message_id)
    event_ts = _parse_ts(payload.get("event_ts"))
    vitals = payload.get("vitals") or {}

    # 1) Raw event (deduped by unique index when event_id present)
    db.session.execute(
        text(
            """
            INSERT INTO vitals_events (tenant_id, patient_id, source, event_ts, event_id, payload_json)
            VALUES (:t, :p, :s, :ets, :eid, :j::jsonb)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "s": source,
            "ets": event_ts,
            "eid": event_id,
            "j": json.dumps(vitals),
        },
    )

    # 2) Latest snapshot
    db.session.execute(
        text(
            """
            INSERT INTO patient_latest_vitals (tenant_id, patient_id, updated_at, latest_json)
            VALUES (:t, :p, NOW(), :j::jsonb)
            ON CONFLICT (tenant_id, patient_id)
            DO UPDATE SET updated_at = NOW(), latest_json = EXCLUDED.latest_json
            """
        ),
        {"t": tenant_id, "p": patient_id, "j": json.dumps(vitals)},
    )

    # 3) Alerts
    alerts = detect_anomalies(vitals)
    for a in alerts:
        db.session.execute(
            text(
                """
                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, details_json)
                VALUES (:t, :p, :atype, :sev, :msg, :d::jsonb)
                """
            ),
            {
                "t": tenant_id,
                "p": patient_id,
                "atype": a.get("alert_type", "anomaly"),
                "sev": a.get("severity", "info"),
                "msg": a.get("message", ""),
                "d": json.dumps({"vitals": vitals, "event_ts": event_ts.isoformat(), "event_id": event_id}),
            },
        )

    # 4) 15-minute rollup (upsert)
    b_start, b_end = _bucket_15m(event_ts)

    hr = _as_float(vitals.get("heart_rate"))
    sys = _as_float(vitals.get("systolic_bp"))
    dia = _as_float(vitals.get("diastolic_bp"))
    spo2 = _as_float(vitals.get("spo2"))

    # Use upsert that updates min/max/last + increments n_events
    db.session.execute(
        text(
            """
            INSERT INTO vitals_rollups_15m (
              tenant_id, patient_id, bucket_start, bucket_end,
              n_events,
              hr_min, hr_max, hr_last,
              sys_min, sys_max, sys_last,
              dia_min, dia_max, dia_last,
              spo2_min, spo2_max, spo2_last,
              updated_at
            )
            VALUES (
              :t, :p, :bs, :be,
              1,
              :hr, :hr, :hr,
              :sys, :sys, :sys,
              :dia, :dia, :dia,
              :spo2, :spo2, :spo2,
              NOW()
            )
            ON CONFLICT (tenant_id, patient_id, bucket_start)
            DO UPDATE SET
              bucket_end = EXCLUDED.bucket_end,
              n_events = vitals_rollups_15m.n_events + 1,

              hr_min = CASE
                WHEN EXCLUDED.hr_last IS NULL THEN vitals_rollups_15m.hr_min
                WHEN vitals_rollups_15m.hr_min IS NULL THEN EXCLUDED.hr_last
                ELSE LEAST(vitals_rollups_15m.hr_min, EXCLUDED.hr_last)
              END,
              hr_max = CASE
                WHEN EXCLUDED.hr_last IS NULL THEN vitals_rollups_15m.hr_max
                WHEN vitals_rollups_15m.hr_max IS NULL THEN EXCLUDED.hr_last
                ELSE GREATEST(vitals_rollups_15m.hr_max, EXCLUDED.hr_last)
              END,
              hr_last = COALESCE(EXCLUDED.hr_last, vitals_rollups_15m.hr_last),

              sys_min = CASE
                WHEN EXCLUDED.sys_last IS NULL THEN vitals_rollups_15m.sys_min
                WHEN vitals_rollups_15m.sys_min IS NULL THEN EXCLUDED.sys_last
                ELSE LEAST(vitals_rollups_15m.sys_min, EXCLUDED.sys_last)
              END,
              sys_max = CASE
                WHEN EXCLUDED.sys_last IS NULL THEN vitals_rollups_15m.sys_max
                WHEN vitals_rollups_15m.sys_max IS NULL THEN EXCLUDED.sys_last
                ELSE GREATEST(vitals_rollups_15m.sys_max, EXCLUDED.sys_last)
              END,
              sys_last = COALESCE(EXCLUDED.sys_last, vitals_rollups_15m.sys_last),

              dia_min = CASE
                WHEN EXCLUDED.dia_last IS NULL THEN vitals_rollups_15m.dia_min
                WHEN vitals_rollups_15m.dia_min IS NULL THEN EXCLUDED.dia_last
                ELSE LEAST(vitals_rollups_15m.dia_min, EXCLUDED.dia_last)
              END,
              dia_max = CASE
                WHEN EXCLUDED.dia_last IS NULL THEN vitals_rollups_15m.dia_max
                WHEN vitals_rollups_15m.dia_max IS NULL THEN EXCLUDED.dia_last
                ELSE GREATEST(vitals_rollups_15m.dia_max, EXCLUDED.dia_last)
              END,
              dia_last = COALESCE(EXCLUDED.dia_last, vitals_rollups_15m.dia_last),

              spo2_min = CASE
                WHEN EXCLUDED.spo2_last IS NULL THEN vitals_rollups_15m.spo2_min
                WHEN vitals_rollups_15m.spo2_min IS NULL THEN EXCLUDED.spo2_last
                ELSE LEAST(vitals_rollups_15m.spo2_min, EXCLUDED.spo2_last)
              END,
              spo2_max = CASE
                WHEN EXCLUDED.spo2_last IS NULL THEN vitals_rollups_15m.spo2_max
                WHEN vitals_rollups_15m.spo2_max IS NULL THEN EXCLUDED.spo2_last
                ELSE GREATEST(vitals_rollups_15m.spo2_max, EXCLUDED.spo2_last)
              END,
              spo2_last = COALESCE(EXCLUDED.spo2_last, vitals_rollups_15m.spo2_last),

              updated_at = NOW()
            """
        ),
        {
            "t": tenant_id,
            "p": patient_id,
            "bs": b_start,
            "be": b_end,
            "hr": hr,
            "sys": sys,
            "dia": dia,
            "spo2": spo2,
        },
    )

    db.session.commit()


def main():
    app = create_app()
    r = redis_client()

    with app.app_context():
        if not r:
            app.logger.error("REDIS_URL not set or redis not installed. Worker will idle.")
            while True:
                time.sleep(10)

        ensure_group(r)
        app.logger.info(f"Worker online. stream={STREAM_KEY} group={GROUP} consumer={CONSUMER}")

        while True:
            try:
                resp = r.xreadgroup(
                    groupname=GROUP,
                    consumername=CONSUMER,
                    streams={STREAM_KEY: ">"},
                    count=BATCH,
                    block=BLOCK_MS,
                )
                if not resp:
                    continue

                # resp format: [(stream, [(id, {field: value})...])]
                for _stream, messages in resp:
                    for msg_id, fields in messages:
                        payload_str = fields.get("payload")
                        if not payload_str:
                            # ack and skip
                            r.xack(STREAM_KEY, GROUP, msg_id)
                            continue

                        try:
                            process_one(msg_id, payload_str)
                            r.xack(STREAM_KEY, GROUP, msg_id)
                        except Exception as e:
                            app.logger.exception("Process failed; moving to DLQ")
                            try:
                                write_dlq(r, payload_str, error=str(e))
                                r.xack(STREAM_KEY, GROUP, msg_id)
                            except Exception:
                                app.logger.exception("DLQ write failed; leaving message pending")
                                # do not ack, so it stays pending
                                time.sleep(1)

            except Exception:
                app.logger.exception("Worker loop error")
                time.sleep(2)


if __name__ == "__main__":
    main()
