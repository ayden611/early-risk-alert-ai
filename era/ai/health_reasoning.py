from __future__ import annotations
# era/ai/reasoning.py
import json
from sqlalchemy import text
from era.extensions import db

from typing import Any, Dict, List


def risk_score_snapshot(
    latest_vitals: Dict[str, Any],
    recent_alerts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Produces:
      - risk_score: 0..100
      - risk_band: low | moderate | high
      - notes: clinician-facing explanations (not medical advice)
    """

    score = 10
    notes: List[str] = []

    hr = latest_vitals.get("heart_rate")
    sys = latest_vitals.get("systolic_bp")
    dia = latest_vitals.get("diastolic_bp")
    spo2 = latest_vitals.get("spo2")

    # ----------------------------
    # Heart rate
    # ----------------------------
    if isinstance(hr, (int, float)):
        if hr >= 130:
            score += 30
            notes.append("HR is very high relative to typical resting ranges.")
        elif hr >= 110:
            score += 15
            notes.append("HR is elevated.")
        elif hr <= 50:
            score += 10
            notes.append("HR is low.")

    # ----------------------------
    # Blood pressure
    # ----------------------------
    if isinstance(sys, (int, float)) and isinstance(dia, (int, float)):
        if sys >= 180 or dia >= 120:
            score += 35
            notes.append("BP is in a very high range.")
        elif sys >= 160 or dia >= 100:
            score += 20
            notes.append("BP is high.")
        elif sys <= 90 or dia <= 60:
            score += 10
            notes.append("BP is low.")

    # ----------------------------
    # Oxygen saturation
    # ----------------------------
    if isinstance(spo2, (int, float)):
        if spo2 < 90:
            score += 35
            notes.append("SpO2 is low.")
        elif spo2 < 93:
            score += 15
            notes.append("SpO2 is borderline low.")

    # ----------------------------
    # Recent alerts influence risk
    # ----------------------------
    critical = sum(1 for a in recent_alerts if a.get("severity") == "critical")
    warn = sum(1 for a in recent_alerts if a.get("severity") == "warn")

    score += critical * 15
    score += warn * 6

    if critical:
        notes.append(f"{critical} critical alert(s) in the recent window.")
    if warn:
        notes.append(f"{warn} warning alert(s) in the recent window.")

    # ----------------------------
    # Clamp score
    # ----------------------------
    score = max(0, min(100, score))

    # ----------------------------
    # Risk band classification
    # ----------------------------
    if score < 30:
        band = "low"
    elif score < 65:
        band = "moderate"
    else:
        band = "high"

    return {
        "risk_score": score,
        "risk_band": band,
        "notes": notes[:8],
        "disclaimer": "Automated monitoring heuristic — not a medical diagnosis.",
    }


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
