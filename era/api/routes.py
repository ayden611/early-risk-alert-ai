from flask import Blueprint, jsonify, request
from datetime import datetime, timezone
import math
import random

api_bp = Blueprint("api", __name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clamp(value, low, high):
    return max(low, min(high, value))


def seeded_rng(*parts):
    seed = "|".join(str(p) for p in parts)
    return random.Random(seed)


def live_wave(minute_factor, amplitude=1.0, offset=0.0):
    return math.sin(minute_factor + offset) * amplitude


def current_tick():
    now = datetime.now(timezone.utc)
    return int(now.timestamp() // 5)


def patient_catalog():
    return [
        {"patient_id": "p101", "name": "Patient p101", "unit": "ICU", "age": 67, "baseline_risk": 0.74},
        {"patient_id": "p202", "name": "Patient p202", "unit": "Stepdown", "age": 58, "baseline_risk": 0.88},
        {"patient_id": "p303", "name": "Patient p303", "unit": "Telemetry", "age": 72, "baseline_risk": 0.69},
        {"patient_id": "p404", "name": "Patient p404", "unit": "Cardiac", "age": 61, "baseline_risk": 0.63},
        {"patient_id": "p505", "name": "Patient p505", "unit": "Med Surg", "age": 49, "baseline_risk": 0.41},
        {"patient_id": "p606", "name": "Patient p606", "unit": "Observation", "age": 55, "baseline_risk": 0.52},
    ]


def build_live_vitals(patient_id, tenant_id="demo"):
    tick = current_tick()
    rng = seeded_rng(tenant_id, patient_id, tick)
    minute_factor = tick / 6.0

    profile = {
        "p101": {"hr": 118, "sbp": 166, "dbp": 99, "spo2": 92, "temp": 99.1, "rr": 24},
        "p202": {"hr": 96, "sbp": 142, "dbp": 86, "spo2": 87, "temp": 100.2, "rr": 26},
        "p303": {"hr": 108, "sbp": 176, "dbp": 104, "spo2": 93, "temp": 98.8, "rr": 22},
        "p404": {"hr": 84, "sbp": 132, "dbp": 82, "spo2": 96, "temp": 98.7, "rr": 18},
        "p505": {"hr": 76, "sbp": 124, "dbp": 78, "spo2": 98, "temp": 98.4, "rr": 16},
        "p606": {"hr": 88, "sbp": 136, "dbp": 84, "spo2": 95, "temp": 99.0, "rr": 19},
    }.get(patient_id, {"hr": 82, "sbp": 128, "dbp": 80, "spo2": 97, "temp": 98.6, "rr": 18})

    heart_rate = int(clamp(profile["hr"] + live_wave(minute_factor, 8, 0.2) + rng.randint(-4, 4), 48, 180))
    systolic_bp = int(clamp(profile["sbp"] + live_wave(minute_factor, 10, 1.2) + rng.randint(-5, 5), 85, 220))
    diastolic_bp = int(clamp(profile["dbp"] + live_wave(minute_factor, 7, 0.8) + rng.randint(-3, 3), 45, 130))
    spo2 = int(clamp(profile["spo2"] + live_wave(minute_factor, 2.5, 2.0) + rng.randint(-1, 1), 78, 100))
    temperature = round(clamp(profile["temp"] + live_wave(minute_factor, 0.5, 0.6) + (rng.random() - 0.5) * 0.2, 96.0, 103.5), 1)
    resp_rate = int(clamp(profile["rr"] + live_wave(minute_factor, 3.5, 0.4) + rng.randint(-1, 1), 10, 36))

    return {
        "event_ts": utc_now(),
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "spo2": spo2,
        "temperature": temperature,
        "resp_rate": resp_rate,
    }


def score_risk(v):
    score = 0

    if v["heart_rate"] >= 125:
        score += 24
    elif v["heart_rate"] >= 110:
        score += 16
    elif v["heart_rate"] <= 50:
        score += 18

    if v["systolic_bp"] >= 175:
        score += 20
    elif v["systolic_bp"] >= 155:
        score += 12
    elif v["systolic_bp"] <= 90:
        score += 22

    if v["diastolic_bp"] >= 100:
        score += 8
    elif v["diastolic_bp"] <= 55:
        score += 10

    if v["spo2"] <= 88:
        score += 35
    elif v["spo2"] <= 92:
        score += 18
    elif v["spo2"] <= 94:
        score += 8

    if v["temperature"] >= 100.8:
        score += 8
    elif v["temperature"] >= 99.8:
        score += 4

    if v["resp_rate"] >= 26:
        score += 14
    elif v["resp_rate"] >= 22:
        score += 8

    score = clamp(score, 0, 100)

    if score >= 70:
        severity = "critical"
    elif score >= 45:
        severity = "high"
    elif score >= 25:
        severity = "moderate"
    else:
        severity = "stable"

    reasons = []
    if v["spo2"] <= 92:
        reasons.append("oxygen saturation outside ideal range")
    if v["heart_rate"] >= 110:
        reasons.append("heart rate elevated")
    if v["systolic_bp"] >= 155:
        reasons.append("blood pressure elevated")
    if v["resp_rate"] >= 22:
        reasons.append("respiratory load elevated")
    if v["temperature"] >= 99.8:
        reasons.append("temperature trending upward")

    if not reasons:
        reasons.append("vitals within expected monitoring range")

    alert_type = None
    alert_message = None

    if v["spo2"] <= 88:
        alert_type = "low_spo2"
        alert_message = "Oxygen saturation critical"
    elif v["heart_rate"] >= 125:
        alert_type = "tachycardia"
        alert_message = "Heart rate elevated"
    elif v["systolic_bp"] >= 175:
        alert_type = "hypertension"
        alert_message = "Blood pressure elevated"
    elif score >= 45:
        alert_type = "deterioration_risk"
        alert_message = "AI deterioration risk elevated"

    return {
        "risk_score": int(score),
        "severity": severity,
        "reasons": reasons,
        "alert_type": alert_type,
        "alert_message": alert_message,
        "confidence": round(clamp(0.78 + (score / 500), 0.78, 0.97), 2),
    }


def simulate_patient(patient, tenant_id="demo"):
    vitals = build_live_vitals(patient["patient_id"], tenant_id=tenant_id)
    risk = score_risk(vitals)
    return {
        "tenant_id": tenant_id,
        "patient_id": patient["patient_id"],
        "name": patient["name"],
        "unit": patient["unit"],
        "age": patient["age"],
        "event_ts": vitals["event_ts"],
        "vitals": vitals,
        "risk": risk,
    }


def simulate_all_patients(tenant_id="demo"):
    return [simulate_patient(p, tenant_id=tenant_id) for p in patient_catalog()]


def build_alerts(patients):
    alerts = []
    for p in patients:
        risk = p["risk"]
        if risk["alert_type"]:
            alerts.append({
                "patient_id": p["patient_id"],
                "patient_name": p["name"],
                "unit": p["unit"],
                "alert_type": risk["alert_type"],
                "severity": risk["severity"],
                "message": risk["alert_message"],
                "risk_score": risk["risk_score"],
                "created_at": "live",
            })

    order = {"critical": 0, "high": 1, "moderate": 2, "stable": 3}
    alerts.sort(key=lambda x: (order.get(x["severity"], 9), -x["risk_score"]))
    return alerts


def safe_avg(values):
    return round(sum(values) / len(values), 1) if values else 0


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "early-risk-alert-api"})


@api_bp.get("/dashboard/overview")
def dashboard_overview():
    tenant_id = request.args.get("tenant_id", "demo")
    patients = simulate_all_patients(tenant_id)
    alerts = build_alerts(patients)

    return jsonify({
        "tenant_id": tenant_id,
        "patient_count": len(patients),
        "open_alerts": len(alerts),
        "critical_alerts": sum(1 for a in alerts if a["severity"] == "critical"),
        "high_alerts": sum(1 for a in alerts if a["severity"] == "high"),
        "events_last_hour": 90 + (current_tick() % 14),
        "avg_risk_score": int(safe_avg([p["risk"]["risk_score"] for p in patients])),
        "monitoring_mode": "realtime",
        "command_center_status": "active",
    })


@api_bp.get("/patients")
def patients():
    tenant_id = request.args.get("tenant_id", "demo")
    data = simulate_all_patients(tenant_id)
    return jsonify({"tenant_id": tenant_id, "patients": data})


@api_bp.get("/alerts")
def alerts():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")
    data = simulate_all_patients(tenant_id)
    items = build_alerts(data)

    if patient_id:
        items = [a for a in items if a["patient_id"] == patient_id]

    return jsonify({"tenant_id": tenant_id, "alerts": items})


@api_bp.get("/vitals/latest")
def vitals_latest():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "p101")

    patient = next((p for p in patient_catalog() if p["patient_id"] == patient_id), None)
    if not patient:
        return jsonify({"error": "patient_not_found"}), 404

    data = simulate_patient(patient, tenant_id)

    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "event_ts": data["event_ts"],
        "vitals": data["vitals"],
        "risk": data["risk"],
    })


@api_bp.get("/patients/<patient_id>/rollups")
def patient_rollups(patient_id):
    tenant_id = request.args.get("tenant_id", "demo")
    patient = next((p for p in patient_catalog() if p["patient_id"] == patient_id), None)
    if not patient:
        return jsonify({"error": "patient_not_found"}), 404

    samples = []
    for i in range(8):
        rng = seeded_rng(tenant_id, patient_id, current_tick() - i)
        vitals = build_live_vitals(patient_id, tenant_id)
        vitals["heart_rate"] = clamp(vitals["heart_rate"] + rng.randint(-4, 4), 45, 180)
        vitals["systolic_bp"] = clamp(vitals["systolic_bp"] + rng.randint(-5, 5), 85, 220)
        vitals["diastolic_bp"] = clamp(vitals["diastolic_bp"] + rng.randint(-3, 3), 45, 130)
        vitals["spo2"] = clamp(vitals["spo2"] + rng.randint(-1, 1), 78, 100)
        samples.append(vitals)

    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "rollups": {
            "avg_heart_rate": safe_avg([s["heart_rate"] for s in samples]),
            "avg_systolic_bp": safe_avg([s["systolic_bp"] for s in samples]),
            "avg_diastolic_bp": safe_avg([s["diastolic_bp"] for s in samples]),
            "avg_spo2": safe_avg([s["spo2"] for s in samples]),
            "avg_temperature": safe_avg([s["temperature"] for s in samples]),
            "avg_resp_rate": safe_avg([s["resp_rate"] for s in samples]),
        }
    })


@api_bp.get("/stream/health")
def stream_health():
    tenant_id = request.args.get("tenant_id", "demo")
    return jsonify({
        "tenant_id": tenant_id,
        "status": "running",
        "redis_ok": True,
        "mode": "realtime-simulated",
        "worker_status": "active",
        "tick_interval_seconds": 5,
    })


@api_bp.get("/stream/channels")
def stream_channels():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "p101")
    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "channels": [
            "stream:vitals",
            f"stream:vitals:{tenant_id}",
            f"stream:vitals:{tenant_id}:{patient_id}",
            "stream:alerts",
            f"stream:alerts:{tenant_id}",
            f"stream:alerts:{tenant_id}:{patient_id}",
        ]
    })


@api_bp.get("/live-snapshot")
def live_snapshot():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "p101")

    all_patients = simulate_all_patients(tenant_id)
    focus = next((p for p in all_patients if p["patient_id"] == patient_id), all_patients[0])
    alerts_list = build_alerts(all_patients)

    return jsonify({
        "tenant_id": tenant_id,
        "generated_at": utc_now(),
        "overview": {
            "patient_count": len(all_patients),
            "open_alerts": len(alerts_list),
            "critical_alerts": sum(1 for a in alerts_list if a["severity"] == "critical"),
            "events_last_hour": 90 + (current_tick() % 14),
        },
        "focus_patient": focus,
        "alerts": alerts_list[:6],
        "patients": all_patients,
    })


@api_bp.get("/ai/risk-score")
def ai_risk_score():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "p101")
    patient = next((p for p in patient_catalog() if p["patient_id"] == patient_id), None)
    if not patient:
        return jsonify({"error": "patient_not_found"}), 404

    data = simulate_patient(patient, tenant_id)
    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "risk_score": data["risk"]["risk_score"],
        "severity": data["risk"]["severity"],
        "confidence": data["risk"]["confidence"],
        "reasons": data["risk"]["reasons"],
        "recommended_action": (
            "Immediate clinical review"
            if data["risk"]["severity"] == "critical"
            else "Escalate to care team"
            if data["risk"]["severity"] == "high"
            else "Continue close monitoring"
            if data["risk"]["severity"] == "moderate"
            else "Routine observation"
        ),
    })


@api_bp.post("/investor/intake")
def investor_intake():
    payload = request.get_json(silent=True) or {}

    return jsonify({
        "status": "received",
        "message": "Investor intake captured successfully",
        "submitted_at": utc_now(),
        "investor": {
            "name": payload.get("name", "Prospective Investor"),
            "firm": payload.get("firm", "Independent"),
            "email": payload.get("email", "not-provided"),
            "interest": payload.get("interest", "seed_round"),
            "check_size": payload.get("check_size", "undisclosed"),
        },
        "next_step": "Send investor deck and schedule platform walkthrough",
    })


@api_bp.get("/investor/demo-summary")
def investor_demo_summary():
    tenant_id = request.args.get("tenant_id", "demo")
    patients = simulate_all_patients(tenant_id)
    alerts_list = build_alerts(patients)

    return jsonify({
        "company": "Early Risk Alert AI",
        "product_positioning": "Predictive clinical intelligence platform",
        "deployment_model": "Hospital + clinic + investor demo platform",
        "revenue_model": [
            "Enterprise SaaS licensing",
            "Per-patient monitoring programs",
            "Analytics and command-center packages",
        ],
        "traction_demo": {
            "simulated_patients": len(patients),
            "active_alerts": len(alerts_list),
            "critical_alerts": sum(1 for a in alerts_list if a["severity"] == "critical"),
            "monitoring_mode": "live-demo",
        },
        "buyer_profiles": [
            "Hospitals",
            "Health systems",
            "Remote patient monitoring programs",
            "Care networks",
        ],
        "next_cta": "Book founder walkthrough and request investor deck",
    })
