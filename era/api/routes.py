from flask import Blueprint, jsonify, request

# THIS is what your app is trying to import
api_bp = Blueprint("api", __name__)

# -----------------------
# Health
# -----------------------
@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})

# -----------------------
# Dashboard Overview
# -----------------------
@api_bp.get("/dashboard/overview")
def dashboard_overview():
    tenant_id = request.args.get("tenant_id", "demo")
    return jsonify({
        "tenant_id": tenant_id,
        "patient_count": 1284,
        "open_alerts": 6,
        "critical_alerts": 2,
        "events_last_hour": 93
    })

# -----------------------
# Alerts
# -----------------------
@api_bp.get("/alerts")
def alerts():
    tenant_id = request.args.get("tenant_id", "demo")
    return jsonify({
        "tenant_id": tenant_id,
        "alerts": [
            {
                "alert_type": "tachycardia",
                "severity": "high",
                "patient_id": "p101",
                "message": "Heart rate elevated",
                "created_at": "live"
            },
            {
                "alert_type": "low_spo2",
                "severity": "critical",
                "patient_id": "p202",
                "message": "Oxygen saturation critical",
                "created_at": "live"
            }
        ]
    })

# -----------------------
# Stream Health
# -----------------------
@api_bp.get("/stream/health")
def stream_health():
    return jsonify({
        "status": "running",
        "redis_ok": True,
        "mode": "realtime"
    })

# -----------------------
# Scale Readiness
# -----------------------
@api_bp.get("/scale/readiness")
def scale_readiness():
    return jsonify({
        "status": "ready",
        "scaling_mode": "horizontal",
        "next_steps": [
            "Add Redis streams",
            "Shard workers by tenant",
            "Enable autoscaling",
            "Add read replicas"
        ]
    })
