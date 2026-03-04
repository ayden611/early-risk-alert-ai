import json
from sqlalchemy import text
from era.extensions import db


def generate_health_reasoning(user_id):

    rows = db.session.execute(
        text("""
        SELECT data
        FROM health_event
        WHERE user_id = :user_id
        ORDER BY id DESC
        LIMIT 5
        """),
        {"user_id": user_id}
    ).fetchall()

    if not rows:
        return {"message": "No health data available"}

    events = [json.loads(r[0]) for r in rows]

    bp_values = [e.get("systolic_bp", 0) for e in events]
    hr_values = [e.get("heart_rate", 0) for e in events]

    latest_bp = bp_values[0]
    latest_hr = hr_values[0]

    reasoning = "Vitals appear stable."

    if latest_bp >= 180:
        reasoning = "Critical blood pressure detected."

    elif bp_values[0] > bp_values[-1]:
        reasoning = "Blood pressure trend increasing."

    if latest_hr > 120:
        reasoning += " Heart rate elevated."

    return {
        "bp": latest_bp,
        "hr": latest_hr,
        "ai_reasoning": reasoning
    }
