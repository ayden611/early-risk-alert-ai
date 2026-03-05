# worker.py
import os
import json
import time
import argparse
import multiprocessing as mp
from typing import Dict, Any, List

from sqlalchemy import text

from era import create_app
from era.extensions import db
from era.streaming import get_stream_client

PARTITIONS = int(os.getenv("STREAM_PARTITIONS", "16"))
GROUP = os.getenv("STREAM_GROUP", "vitals-workers")

def detect_anomalies(vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fast rules-based detector (you can swap this with ML later).
    Return list of alerts dicts.
    """
    alerts = []
    hr = vitals.get("heart_rate")
    sbp = vitals.get("systolic_bp")
    dbp = vitals.get("diastolic_bp")
    spo2 = vitals.get("spo2")

    def add(sev, msg):
        alerts.append({"alert_type": "anomaly", "severity": sev, "message": msg, "meta": vitals})

    try:
        if hr is not None and float(hr) >= 120:
            add("warn", f"High heart rate detected: {hr}")
        if sbp is not None and float(sbp) >= 160:
            add("warn", f"High systolic BP detected: {sbp}")
        if dbp is not None and float(dbp) >= 100:
            add("warn", f"High diastolic BP detected: {dbp}")
        if spo2 is not None and float(spo2) <= 92:
            add("warn", f"Low SpO2 detected: {spo2}")
    except Exception:
        pass

    return alerts


def ensure_tables():
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
              created_at TIMESTAMPTZ DEFAULT NOW(),
              meta JSONB
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_tenant_patient_created
              ON alerts(tenant_id, patient_id, created_at DESC);
            """
        )
    )
    db.session.commit()


def worker_loop(consumer_name: str):
    app = create_app()
    with app.app_context():
        ensure_tables()
        client = get_stream_client()

        stream_keys = [f"vitals:demo:p{i}" for i in range(PARTITIONS)]
        # NOTE: if you have multiple tenants, you can either:
        # - use one "demo" tenant for now, OR
        # - store tenant list and build stream_keys dynamically

        while True:
            try:
                msgs = client.consume_vitals(stream_keys, group=GROUP, consumer=consumer_name, block_ms=5000, count=50)
                if not msgs:
                    continue

                for stream, msg_id, payload in msgs:
                    tenant_id = payload.get("tenant_id") or "demo"
                    patient_id = payload.get("patient_id") or "unknown"
                    vitals_obj = payload.get("vitals") or payload.get("payload", {}).get("vitals") or payload.get("meta") or payload
                    # payload from streaming.py is {"vitals":..., "event_ts":..., "source":...}
                    if isinstance(payload.get("vitals"), dict):
                        vitals_obj = payload["vitals"]

                    alerts = detect_anomalies(vitals_obj if isinstance(vitals_obj, dict) else {})

                    for a in alerts:
                        db.session.execute(
                            text(
                                """
                                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, meta)
                                VALUES (:t, :p, :at, :sev, :msg, :meta::jsonb)
                                """
                            ),
                            {
                                "t": tenant_id,
                                "p": patient_id,
                                "at": a.get("alert_type", "anomaly"),
                                "sev": a.get("severity", "info"),
                                "msg": a.get("message", ""),
                                "meta": json.dumps(a.get("meta", {})),
                            },
                        )
                        db.session.commit()

                        # realtime publish to dashboard
                        client.publish_alert_realtime(
                            tenant_id,
                            patient_id,
                            {
                                "type": "alert",
                                "tenant_id": tenant_id,
                                "patient_id": patient_id,
                                "alert_type": a.get("alert_type", "anomaly"),
                                "severity": a.get("severity", "info"),
                                "message": a.get("message", ""),
                                "ts": time.time(),
                            },
                        )

                    client.ack(stream, GROUP, msg_id)

            except Exception as e:
                # don't die, keep running
                print(f"[{consumer_name}] error: {e}")
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=int(os.getenv("WORKER_PROCS", "1")))
    args = parser.parse_args()

    procs = []
    for i in range(args.procs):
        name = f"c{i}"
        p = mp.Process(target=worker_loop, args=(name,), daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
