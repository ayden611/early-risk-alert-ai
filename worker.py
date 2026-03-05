import os
import json
import time
from typing import Any, Dict, Optional

from sqlalchemy import text

from era import create_app
from era.extensions import db
from era.streaming import ensure_groups_exist, consume_batch, ack


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_anomalies(v: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Lightweight rules. Replace later with model-based detection.
    Returns list of alert dicts.
    """
    alerts = []
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
        ensure_groups_exist()
        print("✅ Worker streaming groups ready.")

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

                    # Write alerts to DB
                    for a in alerts:
                        db.session.execute(
                            text("""
                                INSERT INTO alerts (tenant_id, patient_id, alert_type, severity, message, created_at)
                                VALUES (:t, :p, :at, :sev, :msg, NOW())
                            """),
                            {
                                "t": tenant_id,
                                "p": patient_id,
                                "at": a.get("alert_type", "anomaly"),
                                "sev": a.get("severity", "info"),
                                "msg": a.get("message", ""),
                            },
                        )

                    db.session.commit()

                    # Ack message so it won't be reprocessed
                    ack(stream_name, msg_id)

                except Exception as e:
                    # Don't ack on failure so it can be retried
                    db.session.rollback()
                    print(f"❌ Failed processing {stream_name} {msg_id}: {e}")

            # tiny breather for CPU
            time.sleep(0.01)


if __name__ == "__main__":
    main()
