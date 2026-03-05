# era/ai/anomaly.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Alert:
    severity: str   # "info"|"warn"|"critical"
    kind: str       # "anomaly"|"threshold"|"trend"
    title: str
    details: Dict[str, Any]


def _num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_anomalies(vitals: Dict[str, Any]) -> List[Alert]:
    """
    Lightweight clinical-style threshold checks + basic combinations.
    NOT medical advice; a safety monitoring heuristic.

    Expected keys (optional):
      heart_rate, systolic_bp, diastolic_bp, spo2, resp_rate, temp_c, glucose
    """
    alerts: List[Alert] = []

    hr = _num(vitals.get("heart_rate"))
    sys = _num(vitals.get("systolic_bp"))
    dia = _num(vitals.get("diastolic_bp"))
    spo2 = _num(vitals.get("spo2"))
    rr = _num(vitals.get("resp_rate"))
    temp_c = _num(vitals.get("temp_c"))
    glucose = _num(vitals.get("glucose"))

    # Heart rate
    if hr is not None:
        if hr >= 130:
            alerts.append(Alert("critical", "threshold", "Very high heart rate", {"heart_rate": hr}))
        elif hr >= 110:
            alerts.append(Alert("warn", "threshold", "High heart rate", {"heart_rate": hr}))
        elif hr <= 40:
            alerts.append(Alert("critical", "threshold", "Very low heart rate", {"heart_rate": hr}))
        elif hr <= 50:
            alerts.append(Alert("warn", "threshold", "Low heart rate", {"heart_rate": hr}))

    # Blood pressure
    if sys is not None and dia is not None:
        if sys >= 180 or dia >= 120:
            alerts.append(Alert("critical", "threshold", "Hypertensive crisis range BP", {"systolic_bp": sys, "diastolic_bp": dia}))
        elif sys >= 160 or dia >= 100:
            alerts.append(Alert("warn", "threshold", "High blood pressure", {"systolic_bp": sys, "diastolic_bp": dia}))
        elif sys <= 90 or dia <= 60:
            alerts.append(Alert("warn", "threshold", "Low blood pressure", {"systolic_bp": sys, "diastolic_bp": dia}))

    # Oxygen saturation
    if spo2 is not None:
        if spo2 < 90:
            alerts.append(Alert("critical", "threshold", "Low oxygen saturation", {"spo2": spo2}))
        elif spo2 < 93:
            alerts.append(Alert("warn", "threshold", "Borderline low oxygen saturation", {"spo2": spo2}))

    # Respiratory rate
    if rr is not None:
        if rr > 30:
            alerts.append(Alert("critical", "threshold", "Very high respiratory rate", {"resp_rate": rr}))
        elif rr > 22:
            alerts.append(Alert("warn", "threshold", "High respiratory rate", {"resp_rate": rr}))
        elif rr < 8:
            alerts.append(Alert("critical", "threshold", "Very low respiratory rate", {"resp_rate": rr}))

    # Temperature (C)
    if temp_c is not None:
        if temp_c >= 39.0:
            alerts.append(Alert("warn", "threshold", "High temperature", {"temp_c": temp_c}))
        elif temp_c <= 35.0:
            alerts.append(Alert("warn", "threshold", "Low temperature", {"temp_c": temp_c}))

    # Glucose (mg/dL) – generic thresholds
    if glucose is not None:
        if glucose < 54:
            alerts.append(Alert("critical", "threshold", "Very low glucose", {"glucose": glucose}))
        elif glucose < 70:
            alerts.append(Alert("warn", "threshold", "Low glucose", {"glucose": glucose}))
        elif glucose > 300:
            alerts.append(Alert("critical", "threshold", "Very high glucose", {"glucose": glucose}))
        elif glucose > 180:
            alerts.append(Alert("warn", "threshold", "High glucose", {"glucose": glucose}))

    # Combination signal (simple)
    if hr is not None and spo2 is not None and hr >= 120 and spo2 < 93:
        alerts.append(
            Alert(
                "critical",
                "anomaly",
                "High HR + low SpO2 combination",
                {"heart_rate": hr, "spo2": spo2},
            )
        )

    return alerts
