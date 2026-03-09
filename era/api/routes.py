from flask import Blueprint, jsonify, request, render_template
from sqlalchemy import text
from era.extensions import db

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


# =========================
# LIVE DASHBOARD OVERVIEW
# =========================
@api_bp.get("/dashboard/overview")
def dashboard_overview():
    tenant_id = request.args.get("tenant_id", "demo")

    with db.engine.begin() as conn:
        patient_count = conn.execute(text("""
            SELECT COUNT(DISTINCT patient_id)
            FROM vitals_events
            WHERE tenant_id = :tenant
        """), {"tenant": tenant_id}).scalar() or 0

        open_alerts = conn.execute(text("""
            SELECT COUNT(*) FROM alerts
            WHERE tenant_id = :tenant
        """), {"tenant": tenant_id}).scalar() or 0

        critical_alerts = conn.execute(text("""
            SELECT COUNT(*) FROM alerts
            WHERE tenant_id = :tenant AND severity = 'critical'
        """), {"tenant": tenant_id}).scalar() or 0

        events_last_hour = conn.execute(text("""
            SELECT COUNT(*) FROM vitals_events
            WHERE tenant_id = :tenant
            AND created_at > NOW() - INTERVAL '1 hour'
        """), {"tenant": tenant_id}).scalar() or 0

    return jsonify({
        "tenant_id": tenant_id,
        "patient_count": patient_count,
        "open_alerts": open_alerts,
        "critical_alerts": critical_alerts,
        "events_last_hour": events_last_hour,
    })


# =========================
# LIVE ALERTS FEED
# =========================
@api_bp.get("/alerts")
def alerts():
    tenant_id = request.args.get("tenant_id", "demo")

    with db.engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT patient_id, alert_type, severity, message, created_at
            FROM alerts
            WHERE tenant_id = :tenant
            ORDER BY created_at DESC
            LIMIT 50
        """), {"tenant": tenant_id}).mappings().all()

    return jsonify({"tenant_id": tenant_id, "alerts": list(rows)})


# =========================
# LATEST VITALS
# =========================
@api_bp.get("/vitals/latest")
def vitals_latest():
    tenant_id = request.args.get("tenant_id", "demo")
    patient_id = request.args.get("patient_id")

    with db.engine.begin() as conn:
        row = conn.execute(text("""
            SELECT patient_id, payload_json, created_at
            FROM vitals_events
            WHERE tenant_id = :tenant
            ORDER BY created_at DESC
            LIMIT 1
        """), {"tenant": tenant_id}).mappings().first()

    if not row:
        return jsonify({"error": "no data"})

    return jsonify({
        "tenant_id": tenant_id,
        "patient_id": row["patient_id"],
        "event_ts": row["created_at"],
        "vitals": row["payload_json"]
    })


# =========================
# STREAM HEALTH
# =========================
@api_bp.get("/stream/health")
def stream_health():
    return jsonify({
        "status": "running",
        "mode": "realtime"
    })


# =========================
# PAGES
# =========================
@api_bp.route("/login")
def login():
    return render_template("login.html")


@api_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
