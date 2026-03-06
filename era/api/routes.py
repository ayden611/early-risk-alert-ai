from flask import Blueprint, jsonify, request

api_bp = Blueprint("api", __name__)


@api_bp.get("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "service": "early-risk-alert",
        "db_ok": True
    })


@api_bp.get("/api/v1/dashboard/overview")
def dashboard_overview():
    return jsonify({
        "patient_count": 128,
        "open_alerts": 5,
        "critical_alerts": 2,
        "events_last_hour": 87
    })


@api_bp.get("/api/v1/alerts/live")
def alerts_live():
    return jsonify({
        "alerts": [
            {
                "patient_id": "p100",
                "alert_type": "High Heart Rate",
                "severity": "high",
                "message": "Heart rate exceeded safe threshold",
                "created_at": "just now"
            }
        ]
    })


@api_bp.get("/api/v1/patients/<patient_id>/risk")
def patient_risk(patient_id):
    return jsonify({
        "patient_id": patient_id,
        "risk_level": "moderate",
        "risk_score": 62,
        "summary": "Elevated cardiovascular stress detected"
    })


@api_bp.get("/api/v1/vitals/latest")
def latest_vitals():
    return jsonify({
        "vitals": {
            "heart_rate": 88,
            "systolic_bp": 122,
            "diastolic_bp": 79,
            "spo2": 97
        },
        "event_ts": "just now"
    })


@api_bp.get("/api/v1/rollups/15m")
def rollups():
    return jsonify({
        "rollups": [
            {
                "bucket_start": "10:00",
                "samples": 12,
                "avg_heart_rate": 85,
                "avg_systolic_bp": 120,
                "avg_diastolic_bp": 78,
                "avg_spo2": 98
            }
        ]
    })


@api_bp.get("/api/v1/command-center/summary")
def command_center():
    return jsonify({
        "open_alerts_by_severity": [
            {"severity": "high", "count": 2},
            {"severity": "moderate", "count": 3}
        ],
        "patient_risk_mix": [
            {"risk_level": "high", "count": 12},
            {"risk_level": "moderate", "count": 34},
            {"risk_level": "low", "count": 82}
        ]
    })


@api_bp.get("/api/v1/scale/readiness")
def scale_readiness():
    return jsonify({
        "scaling_mode": "Million-patient ready",
        "next_steps": [
            "Enable Redis Streams",
            "Enable Worker Autoscaling",
            "Add WebSocket fanout",
            "Add multi-region replicas"
        ]
    })
