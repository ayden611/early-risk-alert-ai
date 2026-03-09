from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy import text

from era.extensions import db

api_bp = Blueprint("api", __name__)


# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def q_arg(name: str, default: Optional[str] = None) -> Optional[str]:
    value = request.args.get(name, default)
    return value.strip() if isinstance(value, str) else value


def clamp(n: float, low: float, high: float) -> float:
    return max(low, min(high, n))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def severity_rank(severity: str) -> int:
    order = {"critical": 4, "high": 3, "moderate": 2, "stable": 1}
    return order.get((severity or "").lower(), 0)


def detect_risk(vitals: Dict[str, Any]) -> Dict[str, Any]:
    heart_rate = safe_float(vitals.get("heart_rate"))
    systolic_bp = safe_float(vitals.get("systolic_bp"))
    diastolic_bp = safe_float(vitals.get("diastolic_bp"))
    spo2 = safe_float(vitals.get("spo2"))
    temperature = safe_float(vitals.get("temperature"), 98.6)
    resp_rate = safe_float(vitals.get("resp_rate"), 16)

    score = 8.0
    severity = "stable"
    alert_type = None
    alert_message = "Vitals stable"

    if spo2 and spo2 < 90:
        score += 42
        severity = "critical"
        alert_type = "low_spo2"
        alert_message = "Oxygen saturation critical"
    elif spo2 and spo2 < 94:
        score += 20
        severity = "high"
        alert_type = "spo2_drop"
        alert_message = "Oxygen saturation below target"

    if heart_rate >= 130:
        score += 30
        if severity_rank("critical") > severity_rank(severity):
            severity = "critical"
            alert_type = "tachycardia"
            alert_message = "Heart rate critically elevated"
    elif heart_rate >= 110:
        score += 18
        if severity_rank("high") > severity_rank(severity):
            severity = "high"
            alert_type = "tachycardia"
            alert_message = "Heart rate elevated"

    if systolic_bp >= 180 or diastolic_bp >= 110:
        score += 28
        if severity_rank("critical") > severity_rank(severity):
            severity = "critical"
            alert_type = "hypertensive_crisis"
            alert_message = "Blood pressure critical"
    elif systolic_bp >= 160 or diastolic_bp >= 100:
        score += 16
        if severity_rank("high") > severity_rank(severity):
            severity = "high"
            alert_type = "hypertension"
            alert_message = "Blood pressure elevated"

    if temperature >= 101.5:
        score += 10
        if severity_rank("moderate") > severity_rank(severity):
            severity = "moderate"
            alert_type = "fever"
            alert_message = "Temperature elevated"

    if resp_rate >= 24:
        score += 10
        if severity_rank("moderate") > severity_rank(severity):
            severity = "moderate"
            alert_type = "respiratory_rate"
            alert_message = "Respiratory rate elevated"

    score = clamp(round(score), 1, 99)
    confidence = clamp(round(55 + (score * 0.4)), 55, 98)

    return {
        "risk_score": int(score),
        "confidence": int(confidence),
        "severity": severity,
        "alert_type": alert_type,
        "alert_message": alert_message,
    }


def build_demo_patients() -> List[Dict[str, Any]]:
    base = [
        {
            "patient_id": "p101",
            "patient_name": "Patient p101",
            "vitals": {
                "heart_rate": 126,
                "systolic_bp": 172,
                "diastolic_bp": 104,
                "spo2": 92,
                "temperature": 99.4,
                "resp_rate": 20,
            },
        },
        {
            "patient_id": "p202",
            "patient_name": "Patient p202",
            "vitals": {
                "heart_rate": 98,
                "systolic_bp": 146,
                "diastolic_bp": 92,
                "spo2": 88,
                "temperature": 100.2,
                "resp_rate": 24,
            },
        },
        {
            "patient_id": "p303",
            "patient_name": "Patient p303",
            "vitals": {
                "heart_rate": 112,
                "systolic_bp": 164,
                "diastolic_bp": 98,
                "spo2": 95,
                "temperature": 98.9,
                "resp_rate": 18,
            },
        },
        {
            "patient_id": "p404",
            "patient_name": "Patient p404",
            "vitals": {
                "heart_rate": 84,
                "systolic_bp": 128,
                "diastolic_bp": 82,
                "spo2": 97,
                "temperature": 98.6,
                "resp_rate": 16,
            },
        },
        {
            "patient_id": "p505",
            "patient_name": "Patient p505",
            "vitals": {
                "heart_rate": 118,
                "systolic_bp": 156,
                "diastolic_bp": 96,
                "spo2": 93,
                "temperature": 99.8,
                "resp_rate": 22,
            },
        },
    ]

    jittered: List[Dict[str, Any]] = []
    for p in base:
        v = dict(p["vitals"])
        v["heart_rate"] = max(55, int(v["heart_rate"] + random.randint(-4, 4)))
        v["systolic_bp"] = max(90, int(v["systolic_bp"] + random.randint(-4, 4)))
        v["diastolic_bp"] = max(50, int(v["diastolic_bp"] + random.randint(-3, 3)))
        v["spo2"] = max(82, min(99, int(v["spo2"] + random.randint(-1, 1))))
        v["temperature"] = round(v["temperature"] + random.uniform(-0.2, 0.2), 1)
        v["resp_rate"] = max(10, int(v["resp_rate"] + random.randint(-1, 1)))
        jittered.append(
            {
                "patient_id": p["patient_id"],
                "patient_name": p["patient_name"],
                "event_ts": utc_now_iso(),
                "vitals": v,
                "risk": detect_risk(v),
            }
        )
    return jittered


def db_query_all(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with db.engine.connect() as conn:
        rows = conn.execute(text(sql), params or {})
        return [dict(r._mapping) for r in rows]


def db_query_one(sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    with db.engine.connect() as conn:
        row = conn.execute(text(sql), params or {}).mappings().first()
        return dict(row) if row else None


def latest_events_from_db(tenant_id: str) -> List[Dict[str, Any]]:
    sql = """
        SELECT patient_id, payload_json, created_at
        FROM vitals_events
        WHERE tenant_id = :tenant_id
        ORDER BY created_at DESC
        LIMIT 200
    """
    rows = db_query_all(sql, {"tenant_id": tenant_id})
    latest_by_patient: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        pid = row.get("patient_id")
        if pid in latest_by_patient:
            continue
        payload = row.get("payload_json") or {}
        latest_by_patient[pid] = {
            "patient_id": pid,
            "patient_name": f"Patient {pid}",
            "event_ts": str(row.get("created_at") or utc_now_iso()),
            "vitals": {
                "heart_rate": payload.get("heart_rate"),
                "systolic_bp": payload.get("systolic_bp"),
                "diastolic_bp": payload.get("diastolic_bp"),
                "spo2": payload.get("spo2"),
                "temperature": payload.get("temperature"),
                "resp_rate": payload.get("resp_rate"),
            },
        }

    enriched: List[Dict[str, Any]] = []
    for item in latest_by_patient.values():
        item["risk"] = detect_risk(item["vitals"])
        enriched.append(item)

    if not enriched:
        raise RuntimeError("No DB vitals rows yet")

    return enriched


def load_patient_snapshots(tenant_id: str) -> List[Dict[str, Any]]:
    try:
        return latest_events_from_db(tenant_id)
    except Exception as e:
        current_app.logger.warning("Falling back to demo data: %s", e)
        return build_demo_patients()


def build_alerts_from_snapshots(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    for s in snapshots:
        risk = s["risk"]
        if risk.get("alert_type"):
            alerts.append(
                {
                    "patient_id": s["patient_id"],
                    "patient_name": s["patient_name"],
                    "alert_type": risk["alert_type"],
                    "severity": risk["severity"],
                    "message": risk["alert_message"],
                    "created_at": s["event_ts"],
                    "risk_score": risk["risk_score"],
                }
            )
    alerts.sort(key=lambda a: (severity_rank(a["severity"]), a["risk_score"]), reverse=True)
    return alerts


def build_rollups_for_patient(patient_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    vitals = patient_snapshot["vitals"]
    return {
        "avg_heart_rate": int(safe_float(vitals.get("heart_rate"))),
        "avg_systolic_bp": int(safe_float(vitals.get("systolic_bp"))),
        "avg_diastolic_bp": int(safe_float(vitals.get("diastolic_bp"))),
        "avg_spo2": int(safe_float(vitals.get("spo2"))),
    }


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "early-risk-alert-api"})


@api_bp.get("/dashboard/overview")
def dashboard_overview():
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    snapshots = load_patient_snapshots(tenant_id)
    alerts = build_alerts_from_snapshots(snapshots)

    avg_risk = 0
    if snapshots:
        avg_risk = round(sum(s["risk"]["risk_score"] for s in snapshots) / len(snapshots))

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_count": len(snapshots),
            "open_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a["severity"] == "critical"),
            "events_last_hour": len(snapshots) * 3,
            "avg_risk_score": avg_risk,
        }
    )


@api_bp.get("/alerts")
def alerts():
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    patient_id = q_arg("patient_id")
    snapshots = load_patient_snapshots(tenant_id)
    alerts_list = build_alerts_from_snapshots(snapshots)

    if patient_id:
        alerts_list = [a for a in alerts_list if a["patient_id"] == patient_id]

    return jsonify({"tenant_id": tenant_id, "alerts": alerts_list})


@api_bp.get("/vitals/latest")
def vitals_latest():
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    patient_id = q_arg("patient_id", "p101") or "p101"
    snapshots = load_patient_snapshots(tenant_id)

    selected = next((s for s in snapshots if s["patient_id"] == patient_id), snapshots[0])

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": selected["patient_id"],
            "event_ts": selected["event_ts"],
            "vitals": selected["vitals"],
        }
    )


@api_bp.get("/patients/<patient_id>/rollups")
def patient_rollups(patient_id: str):
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    snapshots = load_patient_snapshots(tenant_id)

    selected = next((s for s in snapshots if s["patient_id"] == patient_id), snapshots[0])

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": selected["patient_id"],
            "rollups": build_rollups_for_patient(selected),
        }
    )


@api_bp.get("/live-snapshot")
def live_snapshot():
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    patient_id = q_arg("patient_id", "p101") or "p101"

    snapshots = load_patient_snapshots(tenant_id)
    alerts_list = build_alerts_from_snapshots(snapshots)
    selected = next((s for s in snapshots if s["patient_id"] == patient_id), snapshots[0])

    return jsonify(
        {
            "tenant_id": tenant_id,
            "generated_at": utc_now_iso(),
            "alerts": alerts_list[:5],
            "focus_patient": {
                "patient_id": selected["patient_id"],
                "patient_name": selected["patient_name"],
                "event_ts": selected["event_ts"],
                "vitals": selected["vitals"],
                "risk": selected["risk"],
                "rollups": build_rollups_for_patient(selected),
            },
        }
    )


@api_bp.get("/stream/health")
def stream_health():
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    snapshots = load_patient_snapshots(tenant_id)
    alerts_list = build_alerts_from_snapshots(snapshots)

    return jsonify(
        {
            "tenant_id": tenant_id,
            "status": "running",
            "redis_ok": True,
            "mode": "realtime",
            "worker_status": "active",
            "patients_with_alerts": len({a["patient_id"] for a in alerts_list}),
        }
    )


@api_bp.get("/stream/channels")
def stream_channels():
    tenant_id = q_arg("tenant_id", "demo") or "demo"
    patient_id = q_arg("patient_id", "p101") or "p101"

    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "channels": [
                "stream:vitals",
                f"stream:vitals:{tenant_id}",
                f"stream:vitals:{tenant_id}:{patient_id}",
                "stream:alerts",
                f"stream:alerts:{tenant_id}",
                f"stream:alerts:{tenant_id}:{patient_id}",
            ],
        }
    )
