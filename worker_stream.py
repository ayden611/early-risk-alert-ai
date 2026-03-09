import os
import time
import random
from datetime import datetime, timezone

from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


PATIENTS = ["P-1001", "P-1002", "P-1003", "P-1004", "P-1005"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rand_vitals():
    return {
        "heart_rate": random.randint(55, 130),
        "systolic_bp": random.randint(95, 160),
        "diastolic_bp": random.randint(60, 100),
        "spo2": random.randint(88, 100),
        "resp_rate": random.randint(10, 26),
        "temp": round(random.uniform(97.0, 103.0), 1),
    }


def risk_level(v):
    if v["spo2"] < 92 or v["heart_rate"] > 120 or v["systolic_bp"] > 150:
        return "HIGH"
    if v["heart_rate"] > 105 or v["resp_rate"] > 22:
        return "MEDIUM"
    return "NORMAL"


def insert_event(pid, vitals):
    payload = {"event_ts": utc_now(), **vitals}
    level = risk_level(vitals)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO vitals_events (tenant_id, patient_id, payload_json)
                VALUES ('demo', :pid, CAST(:payload AS JSONB))
                """
            ),
            {"pid": pid, "payload": str(payload).replace("'", '"')},
        )

        if level != "NORMAL":
            conn.execute(
                text(
                    """
                    INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message)
                    VALUES ('demo', :pid, 'anomaly', :sev, :msg)
                    """
                ),
                {
                    "pid": pid,
                    "sev": level,
                    "msg": f"{level} risk detected",
                },
            )


def main():
    print("Live dashboard feeder running...", flush=True)

    while True:
        for pid in PATIENTS:
            vitals = rand_vitals()
            insert_event(pid, vitals)
            print(f"Inserted data for {pid}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()