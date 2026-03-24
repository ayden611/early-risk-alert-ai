from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List
from werkzeug.security import check_password_hash


from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

from era.web.command_center import COMMAND_CENTER_HTML


INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder & CEO, Early Risk Alert AI"


LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Secure Pilot Login</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --line:rgba(255,255,255,.08);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
    }
    *{box-sizing:border-box}
    body{
      margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at 10% 10%, rgba(91,212,255,.12), transparent 20%),
        linear-gradient(180deg, #07101c, #0b1528);
    }
    .card{
      width:min(560px,100%);
      border:1px solid var(--line);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    .k{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px}
    h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 18px;color:var(--muted);line-height:1.6}
    label{display:block;font-size:13px;font-weight:900;margin-bottom:8px}
    input, select{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid var(--line);
      background:#0d1728;color:var(--text);font:inherit;margin-bottom:14px;
    }
    button{
      width:100%;padding:14px 18px;border:none;border-radius:16px;cursor:pointer;
      font:inherit;font-weight:1000;color:#07101c;
      background:linear-gradient(135deg,var(--blue),var(--blue2));
    }
    .links{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
    a{color:#cfe7ff;text-decoration:none;font-weight:800}
    .error{
      margin-bottom:14px;padding:12px 14px;border-radius:14px;
      background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="k">Secure pilot operating layer</div>
    <h1>Early Risk Alert AI Login</h1>
    <p>Assign a role, hospital brand, and unit scope. Admin sees all units. Pilot users can be restricted to a single unit.</p>
    __ERROR__
    <form method="post" action="/login">
      <label>Full Name</label>
      <input name="full_name" placeholder="Your full name" required>

      <label>Email</label>
      <input name="email" type="email" placeholder="you@example.com" required>

      <label>Role</label>
      <select name="user_role" required>
        <option value="viewer">Viewer</option>
        <option value="operator">Operator</option>
        <option value="physician">Physician</option>
        <option value="admin">Admin</option>
      </select>

      <label>Hospital Brand</label>
      <select name="hospital_brand" required>
        <option value="early-risk-alert-ai">Early Risk Alert AI</option>
        <option value="north-star-medical">North Star Medical Center</option>
        <option value="summit-regional">Summit Regional Hospital</option>
        <option value="blue-valley-health">Blue Valley Health</option>
      </select>

      <label>Assigned Unit</label>
      <select name="assigned_unit" required>
        <option value="all">All Units</option>
        <option value="icu">ICU</option>
        <option value="telemetry">Telemetry</option>
        <option value="stepdown">Stepdown</option>
        <option value="ward">Ward</option>
        <option value="rpm">RPM / Home</option>
      </select>

      <button type="submit">Enter Command Center</button>
    </form>

    <div class="links">
      <a href="/pilot-access">Pilot Access</a>
      <a href="/hospital-demo">Hospital Demo</a>
      <a href="/investor-intake">Investor Access</a>
      <a href="/executive-walkthrough">Executive Walkthrough</a>
    </div>
  </div>
</body>
</html>
"""


PILOT_ACCESS_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pilot Access — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{
      margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .card{
      width:min(520px,100%);
      border:1px solid rgba(255,255,255,.08);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    h1{margin:0 0 10px;font-size:38px;line-height:.95;letter-spacing:-.05em}
    p{color:#9fb4d6;line-height:1.6}
    input{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.08);
      background:#0d1728;color:#eef4ff;font:inherit;margin-bottom:14px;
    }
    button{
      width:100%;padding:14px 18px;border:none;border-radius:16px;cursor:pointer;
      font:inherit;font-weight:1000;color:#07101c;
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
    }
    .error{
      margin-bottom:14px;padding:12px 14px;border-radius:14px;
      background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Branded Pilot Access</h1>
    <p>Enter your pilot access email to load your branded command center and assigned unit.</p>
    __ERROR__
    <form method="post" action="/pilot-access">
      <input name="pilot_email" type="email" placeholder="pilot@hospital.com" required>
      <button type="submit">Enter Pilot Account</button>
    </form>
  </div>
</body>
</html>
"""


FORM_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --line:rgba(255,255,255,.08);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
    }
    *{box-sizing:border-box}
    body{
      margin:0;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .wrap{max-width:860px;margin:0 auto}
    .card{
      border:1px solid var(--line);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    .k{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px}
    h1{margin:0 0 10px;font-size:44px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 20px;color:var(--muted);line-height:1.6}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .field{margin-bottom:14px}
    label{display:block;font-size:13px;font-weight:900;margin-bottom:8px}
    input, select, textarea{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid var(--line);
      background:#0d1728;color:var(--text);font:inherit;
    }
    textarea{min-height:120px;resize:vertical}
    button{
      padding:14px 18px;border:none;border-radius:16px;cursor:pointer;
      font:inherit;font-weight:1000;color:#07101c;
      background:linear-gradient(135deg,var(--blue),var(--blue2));
    }
    .links{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
    a{color:#cfe7ff;text-decoration:none;font-weight:800}
    @media (max-width:700px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="k">Early Risk Alert AI</div>
      <h1>__HEADING__</h1>
      <p>__COPY__</p>
      <form method="post">
        <div class="grid">
          __FIELDS__
        </div>
        <button type="submit">__BUTTON__</button>
      </form>
      <div class="links">
        <a href="/command-center">Command Center</a>
        <a href="/admin/review">Admin Review</a>
      </div>
    </div>
  </div>
</body>
</html>
"""


THANK_YOU_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Request Received</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{
      margin:0;padding:24px;display:grid;place-items:center;min-height:100vh;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .card{
      width:min(760px,100%);
      border:1px solid rgba(255,255,255,.08);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 18px;color:#9fb4d6;line-height:1.6}
    .box{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.03);
      padding:16px;line-height:1.8;color:#dce9ff;margin:16px 0;
    }
    a{
      display:inline-flex;align-items:center;justify-content:center;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Thank You</h1>
    <p>__MESSAGE__</p>
    <div class="box">__DETAILS__</div>
    <a href="/command-center">Return to Command Center</a>
  </div>
</body>
</html>
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY", "early-risk-alert-dev-secret")

    # =========================
    # 🔐 PILOT SECURITY LAYER
    # =========================


    # In-memory stores (safe for pilot — upgrade later to DB)
    AUDIT_LOGS = []
    USER_SESSIONS = {}

    # -------------------------
    # 🔐 ROLE DEFINITIONS
    # -------------------------
    ROLES = {
        "admin": ["view", "ack", "assign", "escalate", "resolve"],
        "nurse": ["view", "ack", "assign"],
        "viewer": ["view"],
}

    # -------------------------
    # 🔐 LOGIN SESSION TRACKING
    # -------------------------
    def current_user():
        return session.get("full_name", "guest")

    def current_role():
        role = str(session.get("user_role", "viewer")).strip().lower()
        return role if role in ROLES else "viewer"

    def current_unit():
        return str(session.get("assigned_unit", "all")).strip().lower() or "all"


    # -------------------------
    # 🔐 ROLE CHECK DECORATOR
    # -------------------------
    def require_permission(action):
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                role = current_role()
                if action not in ROLES.get(role, []):
                    return jsonify({"ok": False, "error": "permission denied"}), 403
                return f(*args, **kwargs)
            return wrapper
        return decorator


    # -------------------------
    # 🧾 AUDIT LOGGING
    # -------------------------
    def utc_now_iso():
        return datetime.now(timezone.utc).isoformat()


    def log_action(user, role, action, patient_id):
        AUDIT_LOGS.append({
            "timestamp": utc_now_iso(),
            "user": user,
            "role": role,
            "action": action,
            "patient_id": patient_id,
            "unit": current_unit(),
    })


    @app.get("/api/audit")
    def get_audit_logs():
        role = current_role()
        if "view" not in ROLES.get(role, []):
            return jsonify({"ok": False, "error": "permission denied"}), 403

        unit = current_unit()
        logs = [row for row in AUDIT_LOGS if row.get("unit") == unit]
        return jsonify(logs[-100:])


    # -------------------------
    # 🏥 UNIT FILTERING
    # -------------------------
    def filter_by_unit(records):
        unit = current_unit()
        return [r for r in records if r.get("unit") == unit]


    # -------------------------
    # 🔁 SYSTEM HEALTH
    # -------------------------
    LAST_UPDATE = datetime.now(timezone.utc)


    @app.get("/api/system/health")
    def system_health():
        now = datetime.now(timezone.utc)
        delta = (now - LAST_UPDATE).total_seconds()
        status = "connected" if delta < 10 else "degraded"

        return jsonify({
            "status": status,
            "last_update": LAST_UPDATE.isoformat(),
            "seconds_since_update": round(delta, 2),
    })


    # -------------------------
    # 🔄 UPDATE HEARTBEAT
    # -------------------------
    def touch_system():
        global LAST_UPDATE
        LAST_UPDATE = datetime.now(timezone.utc)


    # -------------------------
    # 🔐 SAFE LOGIN ROUTE
    # -------------------------


    # -------------------------
    # 🧠 ACTION ENDPOINTS (AUDITED)
    # -------------------------
    @app.post("/api/action/<action>/<patient_id>")
    def take_action(action, patient_id):
        action = str(action).strip().lower()
        allowed_actions = {"ack", "assign", "escalate", "resolve"}

        if action not in allowed_actions:
            return jsonify({"ok": False, "error": "invalid action"}), 400

        role = current_role()
        user = current_user()

        if action not in ROLES.get(role, []):
            return jsonify({"ok": False, "error": "permission denied"}), 403

        log_action(user, role, action.upper(), patient_id)
        touch_system()

        return jsonify({
            "ok": True,
            "action": action,
            "patient_id": patient_id,
            "user": user,
            "role": role,
            "unit": current_unit(),
    })


    # -------------------------
    # 🔐 SECURITY HEADER HARDENING
    # -------------------------
    @app.after_request
    def add_security_headers(response):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store"
        # Legacy header; harmless for older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


    # =========================    
    # ⚠️ PILOT DISCLAIMER
    # =========================
    PILOT_BANNER = """
    <div style="
    position:fixed;
    bottom:0;
    left:0;
    right:0;
    background:#0b1b2b;
    color:#fff;
    padding:10px;
    text-align:center;
    font-size:12px;
    border-top:1px solid rgba(255,255,255,0.1);
    z-index:9999;">
⚠️ Pilot Environment — Decision Support Only. Not a Medical Device. Clinical decisions remain with licensed professionals.
    </div>
    """

    data_dir = Path(app.instance_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    hospital_file = data_dir / "hospital_demo_requests.json"
    exec_file = data_dir / "executive_walkthrough_requests.json"
    investor_file = data_dir / "investor_intake_requests.json"
    workflow_file = data_dir / "command_workflow.json"

    ROLE_PERMISSIONS = {
        "viewer": {"read"},
        "operator": {"read", "ack", "assign"},
        "physician": {"read", "ack", "assign", "escalate", "resolve"},
        "admin": {"read", "ack", "assign", "escalate", "resolve", "admin"},
    }

    HOSPITAL_BRANDS = {
        "early-risk-alert-ai": {
            "hospital_name": "Early Risk Alert AI",
            "brand_name": "Early Risk Alert AI",
            "brand_tagline": "Clinical Command Center",
            "brand_primary": "#7aa2ff",
            "brand_secondary": "#5bd4ff",
        },
        "north-star-medical": {
            "hospital_name": "North Star Medical Center",
            "brand_name": "North Star Medical",
            "brand_tagline": "ICU Pilot Command Center",
            "brand_primary": "#5bd4ff",
            "brand_secondary": "#8fdcff",
        },
        "summit-regional": {
            "hospital_name": "Summit Regional Hospital",
            "brand_name": "Summit Regional",
            "brand_tagline": "Telemetry Pilot Center",
            "brand_primary": "#f4bd6a",
            "brand_secondary": "#ffdba5",
        },
        "blue-valley-health": {
            "hospital_name": "Blue Valley Health",
            "brand_name": "Blue Valley Health",
            "brand_tagline": "Clinical Operations Pilot",
            "brand_primary": "#3ad38f",
            "brand_secondary": "#8ff3c1",
        },
    }

    PILOT_ACCOUNTS = {
        "icu@northstarpilot.com": {
            "full_name": "North Star ICU Pilot",
            "email": "icu@northstarpilot.com",
            "user_role": "operator",
            "assigned_unit": "icu",
            "hospital_brand": "north-star-medical",
        },
        "telemetry@summitpilot.com": {
            "full_name": "Summit Telemetry Pilot",
            "email": "telemetry@summitpilot.com",
            "user_role": "viewer",
            "assigned_unit": "telemetry",
            "hospital_brand": "summit-regional",
        },
        "ops@bluevalleypilot.com": {
            "full_name": "Blue Valley Pilot User",
            "email": "ops@bluevalleypilot.com",
            "user_role": "operator",
            "assigned_unit": "ward",
            "hospital_brand": "blue-valley-health",
        },
        "admin@erapilot.com": {
            "full_name": "ERA Pilot Admin",
            "email": "admin@erapilot.com",
            "user_role": "admin",
            "assigned_unit": "all",
            "hospital_brand": "early-risk-alert-ai",
        },
    }

    def _read_json(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def _write_json(path: Path, payload: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_workflow() -> Dict[str, Any]:
        if not workflow_file.exists():
            default = {"pilot_mode": True, "records": {}, "audit_log": []}
            workflow_file.write_text(json.dumps(default, indent=2), encoding="utf-8")
            return default
        try:
            data = json.loads(workflow_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("invalid store")
            data.setdefault("pilot_mode", True)
            data.setdefault("records", {})
            data.setdefault("audit_log", [])
            return data
        except Exception:
            return {"pilot_mode": True, "records": {}, "audit_log": []}

    def _save_workflow(store: Dict[str, Any]) -> None:
        workflow_file.write_text(json.dumps(store, indent=2), encoding="utf-8")

    def _logged_in() -> bool:
        return bool(session.get("logged_in"))

    def _current_role() -> str:
        role = str(session.get("user_role") or "viewer").strip().lower()
        return role if role in ROLE_PERMISSIONS else "viewer"

    def _current_user() -> str:
        return str(session.get("full_name", "Pilot User")).strip() or "Pilot User"

    def _current_hospital_brand_key() -> str:
        brand = str(session.get("hospital_brand", "early-risk-alert-ai")).strip().lower()
        return brand if brand in HOSPITAL_BRANDS else "early-risk-alert-ai"

    def _current_brand() -> Dict[str, str]:
        return HOSPITAL_BRANDS[_current_hospital_brand_key()]

    def _current_unit_access() -> str:
        unit = str(session.get("assigned_unit", "all")).strip().lower()
        valid_units = {"all", "icu", "telemetry", "stepdown", "ward", "rpm"}
        return unit if unit in valid_units else "all"

    def _has_permission(action: str) -> bool:
        return action in ROLE_PERMISSIONS.get(_current_role(), {"read"})

    def _login_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _logged_in():
                return redirect(url_for("login"))
            return fn(*args, **kwargs)
        return wrapper

    def _admin_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _logged_in():
                return redirect(url_for("login"))
            if _current_role() != "admin":
                return redirect(url_for("command_center"))
            return fn(*args, **kwargs)
        return wrapper

    def _lead_score(payload: Dict[str, Any], lead_type: str) -> int:
        score = 0
        timeline = str(payload.get("timeline", "")).strip().lower()
        if "immediate" in timeline:
            score += 3
        elif "30-60" in timeline:
            score += 2
        elif "quarter" in timeline:
            score += 1

        if lead_type == "investor":
            stage = str(payload.get("stage", "")).strip().lower()
            if "institutional" in stage or "strategic" in stage:
                score += 3
            elif "angel" in stage or "seed" in stage:
                score += 2

        return score

    def _append_row(path: Path, row: Dict[str, Any]) -> None:
        rows = _read_json(path)
        rows.append(row)
        _write_json(path, rows)

    def _detail_html(payload: Dict[str, Any], keys: List[str]) -> str:
        return "<br>".join(
            f"<strong>{k.replace('_', ' ').title()}:</strong> {payload.get(k, '')}"
            for k in keys
        )

    def _normalize_room_to_unit(room: str) -> str:
        r = str(room or "").lower()
        if "icu" in r:
            return "icu"
        if "telemetry" in r:
            return "telemetry"
        if "stepdown" in r:
            return "stepdown"
        if "ward" in r:
            return "ward"
        if "rpm" in r or "home" in r:
            return "rpm"
        return "telemetry"

    def _thresholds_for_unit(unit: str) -> Dict[str, float]:
        defaults = {
            "icu": {"spo2_low": 92, "hr_high": 120, "sbp_high": 160},
            "telemetry": {"spo2_low": 93, "hr_high": 110, "sbp_high": 150},
            "stepdown": {"spo2_low": 93, "hr_high": 115, "sbp_high": 155},
            "ward": {"spo2_low": 94, "hr_high": 110, "sbp_high": 150},
            "rpm": {"spo2_low": 94, "hr_high": 105, "sbp_high": 145},
        }
        return defaults.get(unit, defaults["telemetry"])

    def _simulated_patients() -> List[Dict[str, Any]]:
        tick = int(time.time() // 5)

        base = [
        {
                "patient_id": "p101",
                "patient_name": "Patient 1042",
                "room": "ICU-12",
                "program": "Cardiac",
                "baseline": {"heart_rate": 124, "systolic_bp": 160, "diastolic_bp": 98, "spo2": 90},
                "volatility": {"heart_rate": 8, "systolic_bp": 10, "diastolic_bp": 6, "spo2": 2},
                "trend_bias": {"heart_rate": 4, "systolic_bp": 5, "diastolic_bp": 2, "spo2": -1},
        },
        {
                "patient_id": "p202",
                "patient_name": "Patient 2188",
                "room": "Telemetry-04",
                "program": "Pulmonary",
                "baseline": {"heart_rate": 101, "systolic_bp": 149, "diastolic_bp": 89, "spo2": 93},
                "volatility": {"heart_rate": 6, "systolic_bp": 8, "diastolic_bp": 5, "spo2": 2},
                "trend_bias": {"heart_rate": 2, "systolic_bp": 2, "diastolic_bp": 1, "spo2": -1},
        },
        {
                "patient_id": "p303",
                "patient_name": "Patient 3045",
                "room": "Stepdown-09",
                "program": "Cardiac",
                "baseline": {"heart_rate": 110, "systolic_bp": 156, "diastolic_bp": 95, "spo2": 95},
                "volatility": {"heart_rate": 7, "systolic_bp": 9, "diastolic_bp": 6, "spo2": 1},
                "trend_bias": {"heart_rate": 3, "systolic_bp": 4, "diastolic_bp": 2, "spo2": 0},
        },
        {
                "patient_id": "p404",
                "patient_name": "Patient 4172",
                "room": "Ward-21",
                "program": "Recovery",
                "baseline": {"heart_rate": 84, "systolic_bp": 128, "diastolic_bp": 82, "spo2": 97},
                "volatility": {"heart_rate": 4, "systolic_bp": 6, "diastolic_bp": 4, "spo2": 1},
                "trend_bias": {"heart_rate": 0, "systolic_bp": 0, "diastolic_bp": 0, "spo2": 0},
        },
        {
                "patient_id": "p505",
                "patient_name": "Patient 5117",
                "room": "RPM-Home-03",
                "program": "Home Monitoring",
                "baseline": {"heart_rate": 106, "systolic_bp": 146, "diastolic_bp": 87, "spo2": 93},
                "volatility": {"heart_rate": 5, "systolic_bp": 7, "diastolic_bp": 4, "spo2": 2},
                "trend_bias": {"heart_rate": 1, "systolic_bp": 1, "diastolic_bp": 1, "spo2": -1},
        },
    ]

        def _bounded(value: float, low: int, high: int) -> int:
            return int(max(low, min(high, round(value))))

        out = []

        for item in base:
            unit = _normalize_room_to_unit(item["room"])
            thresholds = _thresholds_for_unit(unit)

            seed = f"{item['patient_id']}:{tick}"
            rng = random.Random(seed)

            phase = tick % 6
            drift_multiplier = {
                0: -0.45,
                1: -0.15,
                2: 0.10,
                3: 0.35,
                4: 0.20,
                5: -0.05,
            }[phase]

            baseline = item["baseline"]
            volatility = item["volatility"]
            trend_bias = item["trend_bias"]

            hr = _bounded(
                baseline["heart_rate"]
                + (trend_bias["heart_rate"] * drift_multiplier)
                + rng.randint(-volatility["heart_rate"], volatility["heart_rate"]),
                58,
                155,
        )

            sbp = _bounded(
                baseline["systolic_bp"]
                + (trend_bias["systolic_bp"] * drift_multiplier)
                + rng.randint(-volatility["systolic_bp"], volatility["systolic_bp"]),
                95,
                210,
        )

            dbp = _bounded(
                baseline["diastolic_bp"]
                + (trend_bias["diastolic_bp"] * drift_multiplier)
                + rng.randint(-volatility["diastolic_bp"], volatility["diastolic_bp"]),
                55,
                130,
        )

            spo2 = _bounded(
                baseline["spo2"]
                + (trend_bias["spo2"] * drift_multiplier)
                + rng.randint(-volatility["spo2"], volatility["spo2"]),
                84,
                100,
        )

            score = 0.0
            reasons = []

            if spo2 < thresholds["spo2_low"]:
                gap = thresholds["spo2_low"] - spo2
                score += 3.2 + min(gap * 0.7, 2.2)
                reasons.append(f"SpO₂ below {thresholds['spo2_low']}")

            if hr > thresholds["hr_high"]:
                gap = hr - thresholds["hr_high"]
                score += 1.8 + min(gap * 0.08, 1.8)
                reasons.append(f"HR above {thresholds['hr_high']}")

            if sbp > thresholds["sbp_high"]:
                gap = sbp - thresholds["sbp_high"]
                score += 1.9 + min(gap * 0.06, 1.9)
                reasons.append(f"SBP above {thresholds['sbp_high']}")

            if spo2 <= max(thresholds["spo2_low"] - 3, 88) and hr >= thresholds["hr_high"]:
                score += 0.8
                reasons.append("combined oxygen and heart rate deterioration pattern")

            if spo2 <= max(thresholds["spo2_low"] - 2, 89) and sbp >= thresholds["sbp_high"]:
                score += 0.6
                reasons.append("combined oxygen and blood pressure pressure pattern")

            score = round(min(score, 9.7), 1)

            if score >= 7:
                severity = "critical"
                action = "Escalate immediately to bedside reassessment and senior clinician review."
            elif score >= 4:
                severity = "high"
                action = "Assign nurse and reassess within the next monitoring interval."
            elif score >= 2:
                severity = "moderate"
                action = "Continue monitoring and review trend changes."
            else:
                severity = "stable"
                action = "Continue routine monitoring."

            if not reasons:
                reasons = ["Combined signal pattern within acceptable thresholds"]

            story_map = {
                "critical": "Predictive monitoring indicates active deterioration risk with supportive escalation priority.",
                "high": "Supportive AI logic shows elevated deterioration attention and recommends rapid reassessment.",
                "moderate": "Trend changes are being monitored for progression beyond routine workflow.",
                "stable": "Signals remain within monitored range for current workflow visibility.",
        }

            alert_message = "Clinical alert surfaced" if severity != "stable" else "Vitals stable"

            out.append({
                "patient_id": item["patient_id"],
                "patient_name": item["patient_name"],
                "room": item["room"],
                "program": item["program"],
                "heart_rate": hr,
                "bp_systolic": sbp,
                "bp_diastolic": dbp,
                "spo2": spo2,
                "vitals": {
                    "heart_rate": hr,
                    "systolic_bp": sbp,
                    "diastolic_bp": dbp,
                    "spo2": spo2,
            },
                "story": story_map[severity],
                "risk": {
                    "risk_score": score,
                    "severity": severity,
                    "alert_message": alert_message,
                    "recommended_action": action,
                    "reasons": reasons,
            }
        })

        return out

    def _simulated_snapshot() -> Dict[str, Any]:
        patients = _simulated_patients()
        alerts = []
        for patient in patients:
            risk = patient["risk"]
            if risk["severity"] != "stable":
                alerts.append({
                    "patient_id": patient["patient_id"],
                    "severity": risk["severity"],
                    "message": risk["alert_message"],
                    "room": patient["room"],
                })

        avg = sum(_safe_float(p["risk"]["risk_score"]) for p in patients) / len(patients) if patients else 0.0
        return {
            "generated_at": _utc_now_iso(),
            "patients": patients,
            "alerts": alerts,
            "summary": {
                "patient_count": len(patients),
                "open_alerts": len(alerts),
                "critical_alerts": sum(1 for a in alerts if a["severity"] == "critical"),
                "avg_risk_score": round(avg, 1),
            },
        }

    def _workflow_record(store: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        if patient_id not in store["records"]:
            store["records"][patient_id] = {
                "ack": False,
                "assigned": False,
                "escalated": False,
                "resolved": False,
                "state": "new",
                "assigned_label": "",
                "updated_at": _utc_now_iso(),
                "role": "",
            }
        return store["records"][patient_id]

    def _audit(store: Dict[str, Any], patient_id: str, action: str, role: str, note: str = "") -> None:
        store["audit_log"].append({
            "time": _utc_now_iso(),
            "patient_id": patient_id,
            "action": action,
            "role": role,
            "note": note,
        })
        store["audit_log"] = store["audit_log"][-200:]

    @app.get("/")
    def home():
        return redirect("/command-center")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            form = request.form or {}
            full_name = str(form.get("full_name", "")).strip()
            email = str(form.get("email", "")).strip().lower()
            user_role = str(form.get("user_role", "viewer")).strip().lower()
            hospital_brand = str(form.get("hospital_brand", "early-risk-alert-ai")).strip().lower()
            assigned_unit = str(form.get("assigned_unit", "all")).strip().lower()
            admin_password = str(form.get("admin_password", "")).strip()

            admin_email = str(os.getenv("ADMIN_EMAIL", "")).strip().lower()
            allowed_admin_emails = {
                e.strip().lower()
                for e in os.getenv("ALLOWED_ADMIN_EMAILS", admin_email).split(",")
                if e.strip()
}
            admin_password_hash = str(os.getenv("ADMIN_PASSWORD_HASH", "")).strip()

            if user_role == "admin":
                if not email or email not in allowed_admin_emails:
                    return render_template_string(
                        LOGIN_HTML.replace(
                            "__ERROR__",
                            "<div class='error'>Admin access denied for this email.</div>"
            )
        )
                if not admin_password_hash or not admin_password:
                    return render_template_string(
                        LOGIN_HTML.replace(
                            "__ERROR__",
                            "<div class='error'>Admin password is required.</div>"
            )
        )
                if not check_password_hash(admin_password_hash, admin_password):
                    return render_template_string(
                        LOGIN_HTML.replace(
                            "__ERROR__",
                            "<div class='error'>Invalid admin password.</div>"
            )
        )
            if user_role not in ROLE_PERMISSIONS:
                user_role = "viewer"
            if hospital_brand not in HOSPITAL_BRANDS:
                hospital_brand = "early-risk-alert-ai"

            valid_units = {"all", "icu", "telemetry", "stepdown", "ward", "rpm"}
            if assigned_unit not in valid_units:
                assigned_unit = "all"

            if not full_name or not email:
                return render_template_string(LOGIN_HTML.replace("__ERROR__", "<div class='error'>Full name and email are required.</div>"))

            session["logged_in"] = True
            session["full_name"] = full_name
            session["email"] = email
            session["user_role"] = user_role
            session["hospital_brand"] = hospital_brand
            session["assigned_unit"] = "all" if user_role == "admin" else assigned_unit
            return redirect("/command-center")

        return render_template_string(LOGIN_HTML.replace("__ERROR__", ""))

    @app.route("/pilot-access", methods=["GET", "POST"])
    def pilot_access():
        if request.method == "POST":
            pilot_email = str(request.form.get("pilot_email", "")).strip().lower()
            account = PILOT_ACCOUNTS.get(pilot_email)

            if not account:
                return render_template_string(PILOT_ACCESS_HTML.replace("__ERROR__", "<div class='error'>Pilot account not found.</div>"))

            session["logged_in"] = True
            session["full_name"] = account["full_name"]
            session["email"] = account["email"]
            session["user_role"] = account["user_role"]
            session["assigned_unit"] = "all" if account["user_role"] == "admin" else account["assigned_unit"]
            session["hospital_brand"] = account["hospital_brand"]
            return redirect("/command-center")

        return render_template_string(PILOT_ACCESS_HTML.replace("__ERROR__", ""))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    @app.get("/command-center")
    @_login_required
    def command_center():
        return render_template_string(COMMAND_CENTER_HTML)

    @app.get("/api/access-context")
    @_login_required
    def access_context():
        brand = _current_brand()
        return jsonify({
            "logged_in": _logged_in(),
            "role": _current_role(),
            "user_name": _current_user(),
            "assigned_unit": _current_unit_access(),
            "can_view_all_units": _current_role() == "admin" or _current_unit_access() == "all",
            "pilot_mode": True,
            **brand,
        })

    @app.get("/api/workflow")
    @_login_required
    def get_workflow():
        store = _load_workflow()
        return jsonify({
            "records": store.get("records", {}),
            "audit_log": store.get("audit_log", []),
        })

    @app.post("/api/workflow/action")
    @_login_required
    def workflow_action():
        data = request.get_json(silent=True) or {}
        patient_id = str(data.get("patient_id", "")).strip()
        action = str(data.get("action", "")).strip().lower()
        note = str(data.get("note", "")).strip()
        role = _current_role()

        if not patient_id:
            return jsonify({"ok": False, "error": "patient_id required"}), 400

        store = _load_workflow()
        record = _workflow_record(store, patient_id)

        if action == "ack":
            if not _has_permission("ack"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["ack"] = True
            record["state"] = "acknowledged"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ACK", role, note or "Alert acknowledged")

        elif action == "assign_nurse":
            if not _has_permission("assign"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["assigned"] = True
            record["assigned_label"] = note or "Assigned Nurse"
            record["state"] = "assigned"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ASSIGN", role, note or "Assigned Nurse")

        elif action == "escalate":
            if not _has_permission("escalate"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["escalated"] = True
            record["state"] = "escalated"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ESCALATE", role, note or "Escalated patient")

        elif action == "resolve":
            if not _has_permission("resolve"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["resolved"] = True
            record["state"] = "resolved"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "RESOLVE", role, note or "Resolved workflow")

        else:
            return jsonify({"ok": False, "error": "invalid action"}), 400

        _save_workflow(store)
        return jsonify({"ok": True, "record": record})

    @app.get("/api/system-health")
    @_login_required
    def api_system_health():
        snapshot = _simulated_snapshot()
        return jsonify({
            "status": "ok",
            "time": _utc_now_iso(),
            "patient_count": snapshot["summary"]["patient_count"],
            "open_alerts": snapshot["summary"]["open_alerts"],
            "critical_alerts": snapshot["summary"]["critical_alerts"],
        })

    @app.get("/api/v1/live-snapshot")
    @_login_required
    def live_snapshot():
        snapshot = _simulated_snapshot()
        patients = snapshot["patients"]
        alerts = snapshot["alerts"]

        allowed_unit = _current_unit_access()
        role = _current_role()

        if role != "admin" and allowed_unit != "all":
            patients = [
                p for p in patients
                if _normalize_room_to_unit(p.get("room", "")) == allowed_unit
            ]
            allowed_ids = {p["patient_id"] for p in patients}
            alerts = [a for a in alerts if a.get("patient_id") in allowed_ids]

        avg_risk = round(
            sum(_safe_float(p.get("risk", {}).get("risk_score", 0)) for p in patients) / len(patients),
            1
        ) if patients else 0.0

        return jsonify({
            "generated_at": snapshot["generated_at"],
            "patients": patients,
            "alerts": alerts,
            "summary": {
                "patient_count": len(patients),
                "open_alerts": len(alerts),
                "critical_alerts": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "critical"),
                "avg_risk_score": avg_risk,
            },
            "access": {
                "role": role,
                "assigned_unit": allowed_unit,
            }
        })

    @app.get("/api/command-center-stream")
    @_login_required
    def command_center_stream():
        def generate():
            while True:
                yield f"data: {json.dumps(_simulated_snapshot())}\\n\\n"
                time.sleep(5)
        return Response(generate(), mimetype="text/event-stream")

    @app.route("/hospital-demo", methods=["GET", "POST"])
    def hospital_demo():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "email": request.form.get("email", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
                "lead_score": 0,
            }
            payload["lead_score"] = _lead_score(payload, "hospital")
            _append_row(hospital_file, payload)
            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your hospital demo request has been received.")
                .replace("__DETAILS__", _detail_html(payload, ["full_name", "organization", "role", "email", "timeline"]))
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Role</label><input name="role" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field"><label>Message</label><textarea name="message"></textarea></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Request a Live Command Center Demo")
        out = out.replace("__HEADING__", "Request a Live Command Center Demo")
        out = out.replace("__COPY__", "See how Early Risk Alert AI identifies deterioration early, prioritizes risk, and enables faster clinical intervention through a real-time command center.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Request Demo Access")
        return render_template_string(out)

    @app.route("/executive-walkthrough", methods=["GET", "POST"])
    def executive_walkthrough():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "title": request.form.get("title", "").strip(),
                "email": request.form.get("email", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
                "lead_score": 0,
            }
            payload["lead_score"] = _lead_score(payload, "executive")
            _append_row(exec_file, payload)
            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your executive walkthrough request has been received.")
                .replace("__DETAILS__", _detail_html(payload, ["full_name", "organization", "title", "email", "timeline"]))
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Title</label><input name="title" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field"><label>Message</label><textarea name="message"></textarea></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Executive Walkthrough")
        out = out.replace("__HEADING__", "Executive Walkthrough")
        out = out.replace("__COPY__", "Request a leadership-level review of the platform, pilot readiness, and clinical operations use case.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Request Executive Walkthrough")
        return render_template_string(out)

    @app.route("/investor-intake", methods=["GET", "POST"])
    def investor_intake():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "email": request.form.get("email", "").strip(),
                "stage": request.form.get("stage", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
                "lead_score": 0,
            }
            payload["lead_score"] = _lead_score(payload, "investor")
            _append_row(investor_file, payload)
            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your investor access request has been received.")
                .replace("__DETAILS__", _detail_html(payload, ["full_name", "organization", "role", "email", "stage", "timeline"]))
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Role</label><input name="role" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Investor Stage</label>
          <select name="stage" required>
            <option value="Angel">Angel</option>
            <option value="Seed">Seed</option>
            <option value="Institutional">Institutional</option>
            <option value="Strategic">Strategic</option>
          </select>
        </div>
        <div class="field"><label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field"><label>Message</label><textarea name="message"></textarea></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Investor Access")
        out = out.replace("__HEADING__", "Request Investor Access")
        out = out.replace("__COPY__", "Request access to investor materials, platform overview, and pilot opportunities for Early Risk Alert AI.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Request Investor Access")
        return render_template_string(out)

    @app.get("/admin/review")
    @_admin_required
    def admin_review():
        hospitals = _read_json(hospital_file)
        executives = _read_json(exec_file)
        investors = _read_json(investor_file)

        def _table(rows: List[Dict[str, Any]]) -> str:
            if not rows:
                return "<p style='color:#9fb4d6;'>No submissions yet.</p>"
            headers = ["submitted_at", "full_name", "organization", "role", "email", "timeline", "lead_score", "status"]
            out = ["<table style='width:100%;border-collapse:collapse'><thead><tr>"]
            for h in headers:
                out.append(f"<th style='padding:12px;text-align:left;color:#9adfff;border-bottom:1px solid rgba(255,255,255,.08);'>{h.replace('_',' ')}</th>")
            out.append("</tr></thead><tbody>")
            for row in rows[::-1]:
                out.append("<tr>")
                for h in headers:
                    out.append(f"<td style='padding:12px;border-bottom:1px solid rgba(255,255,255,.08);'>{row.get(h,'')}</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
            return "".join(out)

        html = f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Admin Review — Early Risk Alert AI</title>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body{{margin:0;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}}
            .wrap{{max-width:1200px;margin:0 auto}}
            .card{{border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;margin-bottom:18px}}
            .btn{{display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:16px;font-weight:900;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none}}
            .top{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="card">
              <div class="top">
                <div>
                  <h1 style="margin:0 0 8px;font-size:42px;line-height:.95;">Admin Review</h1>
                  <div style="color:#9fb4d6;">Role: {_current_role()} · User: {_current_user()}</div>
                </div>
                <div style="display:flex;gap:12px;flex-wrap:wrap">
                  <a class="btn" href="/command-center">Back to Command Center</a>
                </div>
              </div>
            </div>

            <div class="card"><h2>Hospital Demo Requests</h2>{_table(hospitals)}</div>
            <div class="card"><h2>Executive Walkthrough Requests</h2>{_table(executives)}</div>
            <div class="card"><h2>Investor Access Requests</h2>{_table(investors)}</div>
          </div>
        </body>
        </html>
        """
        return render_template_string(html)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": "early-risk-alert-ai", "time": _utc_now_iso()})

    return app
