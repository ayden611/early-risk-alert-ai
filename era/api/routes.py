from flask import Blueprint, jsonify, request

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.get("/dashboard/overview")
def dashboard_overview():
    tenant_id = request.args.get("tenant_id", "demo")
    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_count": 1284,
            "open_alerts": 6,
            "critical_alerts": 2,
            "events_last_hour": 93,
        }
    )


@api_bp.get("/alerts")
def alerts():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")

    sample = [
        {
            "alert_type": "tachycardia",
            "severity": "high",
            "patient_id": "p101",
            "message": "Heart rate elevated",
            "created_at": "live",
        },
        {
            "alert_type": "low_spo2",
            "severity": "critical",
            "patient_id": "p202",
            "message": "Oxygen saturation critical",
            "created_at": "live",
        },
        {
            "alert_type": "hypertension",
            "severity": "high",
            "patient_id": "p303",
            "message": "Blood pressure elevated",
            "created_at": "live",
        },
    ]

    if patient_id:
        sample = [a for a in sample if a["patient_id"] == patient_id]

    return jsonify({"tenant_id": tenant_id, "alerts": sample})


@api_bp.get("/vitals/latest")
def vitals_latest():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "p101")
    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "event_ts": "live",
            "vitals": {
                "heart_rate": 128,
                "systolic_bp": 172,
                "diastolic_bp": 104,
                "spo2": 89,
            },
        }
    )


@api_bp.get("/patients/<patient_id>/rollups")
def patient_rollups(patient_id):
    tenant_id = request.args.get("tenant_id", "demo")
    return jsonify(
        {
            "tenant_id": tenant_id,
            "patient_id": patient_id,
            "rollups": {
                "avg_heart_rate": 118,
                "avg_systolic_bp": 164,
                "avg_diastolic_bp": 98,
                "avg_spo2": 92,
            },
        }
    )


@api_bp.get("/stream/health")
def stream_health():
    return jsonify(
        {
            "status": "running",
            "redis_ok": True,
            "mode": "realtime",
        }
    )


@api_bp.get("/stream/channels")
def stream_channels():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id", "p101")
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


@api_bp.get("/scale/readiness")
def scale_readiness():
    return jsonify(
        {
            "status": "ready",
            "scaling_mode": "horizontal",
            "next_steps": [
                "Add Redis streams",
                "Shard workers by tenant",
                "Enable autoscaling",
                "Add read replicas",
            ],
        }
    )
