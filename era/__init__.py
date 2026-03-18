from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)

from era.web.command_center import COMMAND_CENTER_HTML

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except Exception:
    SendGridAPIClient = None
    Mail = None


INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder, Early Risk Alert AI"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"


LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Pilot Login</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --panel2:#13203a;
      --line:rgba(255,255,255,.08);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      min-height:100vh;
      display:grid;
      place-items:center;
      padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at 10% 10%, rgba(91,212,255,.12), transparent 20%),
        linear-gradient(180deg, #07101c, #0b1528);
    }
    .card{
      width:min(520px,100%);
      border:1px solid var(--line);
      border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;
      box-shadow:0 20px 60px rgba(0,0,0,.34);
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
    .note{margin-top:14px;font-size:12px;color:var(--muted);line-height:1.6}
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
    <div class="k">Pilot operating layer</div>
    <h1>Secure Pilot Login</h1>
    <p>Choose the role you want to simulate. This role controls what actions the command center allows during demos and pilots.</p>
    __ERROR__
    <form method="post" action="/login">
      <label>Full Name</label>
      <input name="full_name" placeholder="Your name" required>

      <label>Email</label>
      <input name="email" type="email" placeholder="you@example.com" required>

      <label>Role</label>
      <select name="user_role" required>
        <option value="viewer">Viewer</option>
        <option value="nurse">Nurse</option>
        <option value="operator">Operator</option>
        <option value="physician">Physician</option>
        <option value="admin">Admin</option>
      </select>

      <button type="submit">Enter Pilot Environment</button>
    </form>

    <div class="note">
      Pilot mode is enabled for evaluation workflows only. Demo and pilot usage should still be reviewed inside each hospital or partner’s own legal, privacy, compliance, and clinical process.
    </div>

    <div class="links">
      <a href="/terms">Terms</a>
      <a href="/privacy">Privacy</a>
      <a href="/pilot-disclaimer">Pilot Disclaimer</a>
    </div>
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
    h1{margin:0 0 10px;font-size:46px;line-height:.95;letter-spacing:-.05em}
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
        <a href="/">Command Center</a>
        <a href="/terms">Terms</a>
        <a href="/privacy">Privacy</a>
        <a href="/pilot-disclaimer">Pilot Disclaimer</a>
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
      margin:0;padding:24px;display:grid;place-items:center;min-height:100vh;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .card{
      width:min(760px,100%);
      border:1px solid var(--line);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 18px;color:var(--muted);line-height:1.6}
    .box{
      border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.03);
      padding:16px;line-height:1.8;color:#dce9ff;margin:16px 0;
    }
    a{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      background:linear-gradient(135deg,var(--blue),var(--blue2));color:#07101c;text-decoration:none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Thank You</h1>
    <p>__MESSAGE__</p>
    <div class="box">__DETAILS__</div>
    <a href="/">Return Home</a>
  </div>
</body>
</html>
"""


ADMIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin Review — Early Risk Alert AI</title>
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
    .wrap{max-width:1200px;margin:0 auto}
    .hero,.section{
      border:1px solid var(--line);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;margin-bottom:18px;
    }
    h1,h2{margin:0 0 12px}
    .muted{color:var(--muted)}
    table{width:100%;border-collapse:collapse}
    th,td{padding:12px;border-bottom:1px solid var(--line);text-align:left;font-size:14px;vertical-align:top}
    th{color:#9adfff;font-size:12px;text-transform:uppercase;letter-spacing:.12em}
    .top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      background:linear-gradient(135deg,var(--blue),var(--blue2));color:#07101c;text-decoration:none;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="top">
        <div>
          <h1>Admin Review</h1>
          <div class="muted">Role: __ROLE__ · User: __USER__ · Last Updated: __UPDATED__</div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap">
          <a class="btn" href="/admin/export">Export Requests CSV</a>
          <a class="btn" href="/">Back to Command Center</a>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Hospital Demo Requests</h2>
      __HOSPITAL_TABLE__
    </div>

    <div class="section">
      <h2>Executive Walkthrough Requests</h2>
      __EXEC_TABLE__
    </div>

    <div class="section">
      <h2>Investor Intake</h2>
      __INVESTOR_TABLE__
    </div>
  </div>
</body>
</html>
"""


LEGAL_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{
      margin:0;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .wrap{
      max-width:920px;margin:0 auto;border:1px solid rgba(255,255,255,.08);
      border-radius:24px;padding:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
    }
    h1{margin:0 0 14px;font-size:40px;line-height:.95;letter-spacing:-.05em}
    p, li{line-height:1.75;color:#dce9ff}
    .muted{color:#9fb4d6}
    a{color:#cfe7ff;text-decoration:none;font-weight:800}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>__TITLE__</h1>
    <p class="muted">Effective for demo and pilot environments.</p>
    __BODY__
    <p><a href="/">Return to Command Center</a></p>
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


def _severity_rank(severity: str) -> int:
    order = {"critical": 4, "high": 3, "moderate": 2, "stable": 1}
    return order.get((severity or "").lower(), 0)


def _detail_html(payload: Dict[str, Any], keys: List[str]) -> str:
    lines = []
    for key in keys:
        value = payload.get(key, "")
        label = key.replace("_", " ").title()
        lines.append(f"<strong>{label}:</strong> {value}")
    return "<br>".join(lines)


def _sendgrid_available() -> bool:
    return bool(SendGridAPIClient and Mail)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.secret_key = os.getenv("SECRET_KEY", "early-risk-alert-dev-secret")

    data_dir = Path(app.instance_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    hospital_file = data_dir / "hospital_demo_requests.json"
    exec_file = data_dir / "executive_walkthrough_requests.json"
    investor_file = data_dir / "investor_intake_requests.json"
    workflow_file = data_dir / "command_workflow.json"

    ROLE_PERMISSIONS = {
        "viewer": {"read"},
        "nurse": {"read", "ack"},
        "operator": {"read", "ack", "assign"},
        "physician": {"read", "ack", "assign", "escalate", "resolve"},
        "admin": {"read", "ack", "assign", "escalate", "resolve", "pilot_toggle", "admin"},
    }

    DEFAULT_THRESHOLDS = {
        "icu": {"spo2_low": 92, "hr_high": 120, "sbp_high": 160},
        "stepdown": {"spo2_low": 93, "hr_high": 115, "sbp_high": 155},
        "telemetry": {"spo2_low": 94, "hr_high": 110, "sbp_high": 150},
        "ward": {"spo2_low": 94, "hr_high": 110, "sbp_high": 150},
        "rpm": {"spo2_low": 93, "hr_high": 115, "sbp_high": 155},
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
            default = {
                "pilot_mode": True,
                "records": {},
                "audit_log": [],
            }
            with workflow_file.open("w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)
            return default
        try:
            with workflow_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("invalid workflow store")
            data.setdefault("pilot_mode", True)
            data.setdefault("records", {})
            data.setdefault("audit_log", [])
            return data
        except Exception:
            return {"pilot_mode": True, "records": {}, "audit_log": []}

    def _save_workflow(store: Dict[str, Any]) -> None:
        with workflow_file.open("w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)

    def _current_role() -> str:
        role = str(session.get("user_role") or session.get("role") or "viewer").strip().lower()
        if role not in ROLE_PERMISSIONS:
            return "viewer"
        return role

    def _current_user() -> str:
        return str(session.get("full_name") or session.get("user_name") or "Guest").strip()

    def _logged_in() -> bool:
        return bool(session.get("logged_in"))

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

    def _append_row(path: Path, row: Dict[str, Any]) -> None:
        rows = _read_json(path)
        rows.append(row)
        _write_json(path, rows)

    def _send_email(to_email: str, subject: str, html_body: str) -> bool:
        api_key = os.getenv("SENDGRID_API_KEY", "").strip()
        from_email = os.getenv("SENDGRID_FROM_EMAIL", "").strip()

        if not api_key or not from_email or not to_email or not _sendgrid_available():
            return False

        try:
            message = Mail(
                from_email=from_email,
                to_emails=to_email,
                subject=subject,
                html_content=html_body,
            )
            client = SendGridAPIClient(api_key)
            response = client.send(message)
            return 200 <= response.status_code < 300
        except Exception as e:
            print("SendGrid send failed:", str(e))
            return False

    def _send_confirmation_email(lead_type: str, payload: Dict[str, Any]) -> None:
        to_email = str(payload.get("email", "")).strip()
        full_name = payload.get("full_name", "there")
        org = payload.get("organization", "")
        timeline = payload.get("timeline", "")

        subject_map = {
            "hospital": "Early Risk Alert AI — Hospital Demo Request Received",
            "executive": "Early Risk Alert AI — Executive Walkthrough Request Received",
            "investor": "Early Risk Alert AI — Investor Intake Received",
        }

        heading_map = {
            "hospital": "Thank you for requesting a hospital demo.",
            "executive": "Thank you for requesting an executive walkthrough.",
            "investor": "Thank you for your investor interest.",
        }

        html_body = f"""
        <p>Hello {full_name},</p>
        <p>{heading_map.get(lead_type, "Thank you for reaching out to Early Risk Alert AI.")}</p>
        <p>We received your request and a follow-up will be sent shortly.</p>
        <p>
          <strong>Organization:</strong> {org}<br>
          <strong>Timeline:</strong> {timeline}
        </p>
        <p>— {FOUNDER_NAME}<br>{FOUNDER_ROLE}</p>
        """

        _send_email(
            to_email=to_email,
            subject=subject_map.get(lead_type, "Early Risk Alert AI — Request Received"),
            html_body=html_body,
        )

    def _send_admin_notification(lead_type: str, payload: Dict[str, Any]) -> None:
    admin_to = os.getenv(
        "SENDGRID_ADMIN_TO",
        "info@earlyriskalertai.com,milton@earlyriskalertai.com"
    )

    if not admin_to:
        return

    # 🔥 SPLIT MULTIPLE EMAILS
    admin_emails = [email.strip() for email in admin_to.split(",")]

    subject = f"New {lead_type.title()} Request — Early Risk Alert AI"

    rows = "".join(
        f"<tr><td style='padding:8px;border-bottom:1px solid #ddd'><strong>{k}</strong></td>"
        f"<td style='padding:8px;border-bottom:1px solid #ddd'>{v}</td></tr>"
        for k, v in payload.items()
    )

    html_body = f"""
    <h2>🚨 New {lead_type.title()} Request</h2>
    <table style="border-collapse:collapse;width:100%">
        {rows}
    </table>
    """

    # 🔥 SEND TO MULTIPLE RECIPIENTS
    for email in admin_emails:
        _send_email(email, subject, html_body)

    def _normalize_room_to_unit(room: str) -> str:
        r = str(room or "").lower()
        if "icu" in r:
            return "icu"
        if "stepdown" in r:
            return "stepdown"
        if "telemetry" in r:
            return "telemetry"
        if "ward" in r:
            return "ward"
        if "rpm" in r or "home" in r:
            return "rpm"
        return "telemetry"

    def _thresholds_for_unit(unit: str) -> Dict[str, float]:
        return DEFAULT_THRESHOLDS.get(unit, DEFAULT_THRESHOLDS["telemetry"])

    def _simulated_patients() -> List[Dict[str, Any]]:
        base = [
            {
                "patient_id": "p101",
                "patient_name": "Patient 1042",
                "room": "ICU-12",
                "program": "Cardiac",
                "vitals": {"heart_rate": 128, "systolic_bp": 165, "diastolic_bp": 102, "spo2": 89},
            },
            {
                "patient_id": "p202",
                "patient_name": "Patient 2188",
                "room": "Stepdown-04",
                "program": "Pulmonary",
                "vitals": {"heart_rate": 100, "systolic_bp": 150, "diastolic_bp": 90, "spo2": 93},
            },
            {
                "patient_id": "p303",
                "patient_name": "Patient 3045",
                "room": "Telemetry-09",
                "program": "Cardiac",
                "vitals": {"heart_rate": 111, "systolic_bp": 162, "diastolic_bp": 97, "spo2": 95},
            },
            {
                "patient_id": "p404",
                "patient_name": "Patient 4172",
                "room": "Ward-21",
                "program": "Recovery",
                "vitals": {"heart_rate": 84, "systolic_bp": 128, "diastolic_bp": 82, "spo2": 97},
            },
        ]

        patients: List[Dict[str, Any]] = []
        for item in base:
            unit = _normalize_room_to_unit(item["room"])
            thresholds = _thresholds_for_unit(unit)
            spo2 = _safe_float(item["vitals"]["spo2"])
            hr = _safe_float(item["vitals"]["heart_rate"])
            sbp = _safe_float(item["vitals"]["systolic_bp"])

            score = 0.0
            reasons: List[str] = []

            if spo2 < thresholds["spo2_low"]:
                score += 4.2
                reasons.append(f"SpO₂ below {thresholds['spo2_low']}")
            if hr > thresholds["hr_high"]:
                score += 2.3
                reasons.append(f"HR above {thresholds['hr_high']}")
            if sbp > thresholds["sbp_high"]:
                score += 2.6
                reasons.append(f"SBP above {thresholds['sbp_high']}")

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

            item["risk"] = {
                "risk_score": round(score, 1),
                "severity": severity,
                "alert_message": "Clinical alert surfaced" if severity != "stable" else "Vitals stable",
                "recommended_action": action,
                "reasons": reasons or ["Combined signal pattern within acceptable thresholds"],
            }
            patients.append(item)
        return patients

    def _simulated_snapshot() -> Dict[str, Any]:
        patients = _simulated_patients()
        alerts = []
        for patient in patients:
            risk = patient["risk"]
            if risk["severity"] != "stable":
                alerts.append(
                    {
                        "patient_id": patient["patient_id"],
                        "severity": risk["severity"],
                        "message": risk["alert_message"],
                        "room": patient["room"],
                    }
                )

        avg_risk = 0.0
        if patients:
            avg_risk = sum(_safe_float(p["risk"]["risk_score"]) for p in patients) / len(patients)

        critical_alerts = sum(1 for a in alerts if str(a["severity"]).lower() == "critical")

        return {
            "generated_at": _utc_now_iso(),
            "patients": patients,
            "alerts": alerts,
            "summary": {
                "patient_count": len(patients),
                "open_alerts": len(alerts),
                "critical_alerts": critical_alerts,
                "avg_risk_score": round(avg_risk, 1),
            },
        }

    def _workflow_record(store: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        records = store.setdefault("records", {})
        if patient_id not in records:
            records[patient_id] = {
                "ack": False,
                "assigned": False,
                "escalated": False,
                "resolved": False,
                "state": "new",
                "assigned_label": "",
                "updated_at": _utc_now_iso(),
                "role": "",
            }
        return records[patient_id]

    def _audit(store: Dict[str, Any], patient_id: str, action: str, role: str, note: str = "") -> None:
        log = store.setdefault("audit_log", [])
        log.append(
            {
                "time": _utc_now_iso(),
                "patient_id": patient_id,
                "action": action,
                "role": role,
                "note": note,
            }
        )
        store["audit_log"] = log[-200:]

    def _lead_score(payload: Dict[str, Any], lead_type: str) -> int:
        score = 0
        timeline = str(payload.get("timeline", "")).lower()
        if "immediate" in timeline:
            score += 3
        elif "30-60" in timeline:
            score += 2
        elif "quarter" in timeline:
            score += 1

        if lead_type == "investor":
            stage = str(payload.get("stage", "")).lower()
            if "institutional" in stage:
                score += 3
            elif "angel" in stage:
                score += 2
        return score

    def _table_html(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "<p class='muted'>No submissions yet.</p>"

        headers = [
            "submitted_at",
            "full_name",
            "organization",
            "role",
            "email",
            "phone",
            "timeline",
            "status",
            "lead_score",
        ]
        out = ["<table><thead><tr>"]
        for h in headers:
            out.append(f"<th>{h.replace('_', ' ')}</th>")
        out.append("</tr></thead><tbody>")
        for row in rows[::-1]:
            out.append("<tr>")
            for h in headers:
                out.append(f"<td>{row.get(h, '')}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    @app.get("/")
    def command_center():
        return render_template_string(COMMAND_CENTER_HTML)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            form = request.form or {}
            full_name = str(form.get("full_name", "")).strip()
            email = str(form.get("email", "")).strip()
            user_role = str(form.get("user_role", form.get("role", "viewer"))).strip().lower()

            if user_role not in ROLE_PERMISSIONS:
                user_role = "viewer"

            if not full_name or not email:
                error_html = "<div class='error'>Full name and email are required.</div>"
                return render_template_string(LOGIN_HTML.replace("__ERROR__", error_html))

            session["logged_in"] = True
            session["full_name"] = full_name
            session["email"] = email
            session["user_role"] = user_role
            session["role"] = user_role
            return redirect(url_for("command_center"))

        return render_template_string(LOGIN_HTML.replace("__ERROR__", ""))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/terms")
    def terms():
        body = """
        <p>Early Risk Alert AI is provided for product evaluation, demonstration, internal review, and pilot discussions.</p>
        <ul>
          <li>It is not a substitute for clinical judgment.</li>
          <li>Use of simulated or pilot data must be validated by participating organizations before operational use.</li>
          <li>Production deployment requires each organization’s own legal, compliance, privacy, clinical, and technical review.</li>
        </ul>
        """
        return render_template_string(LEGAL_PAGE_HTML.replace("__TITLE__", "Terms of Use").replace("__BODY__", body))

    @app.get("/privacy")
    def privacy():
        body = """
        <p>Early Risk Alert AI is designed to minimize unnecessary handling of personal data during demos and pilots.</p>
        <ul>
          <li>Demo and pilot environments should avoid real patient identifiers unless explicitly approved and properly governed.</li>
          <li>Organizations remain responsible for their own privacy, security, and compliance obligations.</li>
          <li>Contact submissions are used for follow-up, scheduling, and business communication.</li>
        </ul>
        """
        return render_template_string(LEGAL_PAGE_HTML.replace("__TITLE__", "Privacy").replace("__BODY__", body))

    @app.get("/pilot-disclaimer")
    def pilot_disclaimer():
        body = """
        <p>This environment may include simulated workflows, synthetic monitoring data, and configurable pilot-only controls.</p>
        <ul>
          <li>Pilot mode should be treated as an evaluation environment.</li>
          <li>Clinical actions shown in the interface are demonstrations of workflow behavior unless formally integrated and approved.</li>
          <li>Each pilot site must complete its own validation and governance review.</li>
        </ul>
        """
        return render_template_string(LEGAL_PAGE_HTML.replace("__TITLE__", "Pilot Disclaimer").replace("__BODY__", body))

    @app.route("/hospital-demo", methods=["GET", "POST"])
    def hospital_demo():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "last_updated": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "email": request.form.get("email", "").strip(),
                "phone": request.form.get("phone", "").strip(),
                "facility_type": request.form.get("facility_type", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            payload["lead_score"] = _lead_score(payload, "hospital")
            _append_row(hospital_file, payload)
            _send_confirmation_email("hospital", payload)
            _send_admin_notification("hospital", payload)

            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your hospital demo request was submitted successfully.")
                .replace("__DETAILS__", _detail_html(payload, ["full_name", "organization", "role", "email", "facility_type", "timeline"]))
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Role</label><input name="role" required></div>
        <div class="field"><label>Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Phone</label><input name="phone"></div>
        <div class="field"><label>Facility Type</label>
          <select name="facility_type" required>
            <option value="Hospital">Hospital</option>
            <option value="Clinic">Clinic</option>
            <option value="Health System">Health System</option>
            <option value="RPM Provider">RPM Provider</option>
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
        <div class="field"><label>What would you like to see?</label><textarea name="message"></textarea></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Hospital Demo — Early Risk Alert AI")
        out = out.replace("__HEADING__", "Hospital Demo Request")
        out = out.replace("__COPY__", "Request a hospital-focused walkthrough of the predictive monitoring wall, pilot operating layer, and workflow tools.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Submit Hospital Demo Request")
        return render_template_string(out)

    @app.route("/executive-walkthrough", methods=["GET", "POST"])
    def executive_walkthrough():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "last_updated": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "title": request.form.get("title", "").strip(),
                "email": request.form.get("email", "").strip(),
                "phone": request.form.get("phone", "").strip(),
                "priority": request.form.get("priority", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            payload["lead_score"] = _lead_score(payload, "executive")
            _append_row(exec_file, payload)
            _send_confirmation_email("executive", payload)
            _send_admin_notification("executive", payload)

            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your executive walkthrough request was submitted successfully.")
                .replace("__DETAILS__", _detail_html(payload, ["full_name", "organization", "title", "email", "priority", "timeline"]))
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Title</label><input name="title" required></div>
        <div class="field"><label>Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Phone</label><input name="phone"></div>
        <div class="field"><label>Priority</label>
          <select name="priority" required>
            <option value="Strategy">Strategy</option>
            <option value="Pilot Planning">Pilot Planning</option>
            <option value="Operations">Operations</option>
            <option value="Evaluation">Evaluation</option>
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
        <div class="field"><label>What would you like covered?</label><textarea name="message"></textarea></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Executive Walkthrough — Early Risk Alert AI")
        out = out.replace("__HEADING__", "Executive Walkthrough")
        out = out.replace("__COPY__", "Request a leadership-level review of the platform, pilot readiness, operating model, and commercial roadmap.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Submit Executive Walkthrough Request")
        return render_template_string(out)

    @app.route("/investor-intake", methods=["GET", "POST"])
    def investor_intake():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "last_updated": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "email": request.form.get("email", "").strip(),
                "phone": request.form.get("phone", "").strip(),
                "stage": request.form.get("stage", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            payload["lead_score"] = _lead_score(payload, "investor")
            _append_row(investor_file, payload)
            _send_confirmation_email("investor", payload)
            _send_admin_notification("investor", payload)

            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your investor intake was submitted successfully.")
                .replace("__DETAILS__", _detail_html(payload, ["full_name", "organization", "role", "email", "stage", "timeline"]))
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Role</label><input name="role" required></div>
        <div class="field"><label>Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Phone</label><input name="phone"></div>
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
        <div class="field"><label>What would you like to review?</label><textarea name="message"></textarea></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Investor Intake — Early Risk Alert AI")
        out = out.replace("__HEADING__", "Investor Intake")
        out = out.replace("__COPY__", "Request investor materials, platform review, and pipeline visibility for Early Risk Alert AI.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Submit Investor Intake")
        return render_template_string(out)

    @app.get("/admin/review")
    @_admin_required
    def admin_review():
        hospitals = _read_json(hospital_file)
        executives = _read_json(exec_file)
        investors = _read_json(investor_file)

        out = ADMIN_HTML.replace("__ROLE__", _current_role())
        out = out.replace("__USER__", _current_user())
        out = out.replace("__UPDATED__", _utc_now_iso())
        out = out.replace("__HOSPITAL_TABLE__", _table_html(hospitals))
        out = out.replace("__EXEC_TABLE__", _table_html(executives))
        out = out.replace("__INVESTOR_TABLE__", _table_html(investors))
        return render_template_string(out)

    @app.get("/admin/export")
    @_admin_required
    def admin_export():
        rows: List[Dict[str, Any]] = []
        for label, path in [
            ("hospital", hospital_file),
            ("executive", exec_file),
            ("investor", investor_file),
        ]:
            for row in _read_json(path):
                item = dict(row)
                item["lead_type"] = label
                rows.append(item)

        export_path = data_dir / "admin_export.json"
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return send_file(export_path, as_attachment=True, download_name="admin_export.json")

    @app.get("/api/pilot-mode")
    def get_pilot_mode():
        store = _load_workflow()
        return jsonify(
            {
                "enabled": bool(store.get("pilot_mode", True)),
                "role": _current_role(),
                "user": _current_user(),
                "logged_in": _logged_in(),
            }
        )

    @app.post("/api/pilot-mode/toggle")
    def toggle_pilot_mode():
        if not _has_permission("pilot_toggle"):
            return jsonify({"ok": False, "error": "permission denied"}), 403

        store = _load_workflow()
        store["pilot_mode"] = not bool(store.get("pilot_mode", True))
        _save_workflow(store)
        return jsonify({"ok": True, "enabled": store["pilot_mode"]})

    @app.get("/api/workflow")
    def get_workflow():
        store = _load_workflow()
        return jsonify(
            {
                "records": store.get("records", {}),
                "audit_log": store.get("audit_log", [])[-50:],
            }
        )

    @app.post("/api/workflow/action")
    def workflow_action():
        data = request.get_json(silent=True) or {}
        patient_id = str(data.get("patient_id", "")).strip()
        action = str(data.get("action", "")).strip().lower()
        note = str(data.get("note", "")).strip()
        role = _current_role()

        if not patient_id:
            return jsonify({"ok": False, "error": "patient_id required"}), 400

        store = _load_workflow()

        if not bool(store.get("pilot_mode", True)):
            return jsonify({"ok": False, "error": "Actions disabled outside pilot mode"}), 403

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
            record["assigned_label"] = note or "Nurse assigned"
            record["state"] = "assigned"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ASSIGN", role, note or "Nurse assigned")

        elif action == "escalate":
            if not _has_permission("escalate"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["escalated"] = True
            record["state"] = "escalated"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ESCALATE", role, note or "Patient escalated")

        elif action == "resolve":
            if not _has_permission("resolve"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["resolved"] = True
            record["state"] = "resolved"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "RESOLVE", role, note or "Patient resolved")

        else:
            return jsonify({"ok": False, "error": "invalid action"}), 400

        _save_workflow(store)
        return jsonify({"ok": True, "record": record, "audit_log": store.get("audit_log", [])[-50:]})

    @app.get("/api/system-health")
    def system_health():
        store = _load_workflow()
        snapshot = _simulated_snapshot()
        return jsonify(
            {
                "status": "ok",
                "pilot_mode": bool(store.get("pilot_mode", True)),
                "role": _current_role(),
                "user": _current_user(),
                "time": _utc_now_iso(),
                "patient_count": snapshot["summary"]["patient_count"],
                "open_alerts": snapshot["summary"]["open_alerts"],
                "critical_alerts": snapshot["summary"]["critical_alerts"],
            }
        )

    @app.get("/api/system/banner")
    def system_banner():
        store = _load_workflow()
        if bool(store.get("pilot_mode", True)):
            return jsonify(
                {
                    "message": "Pilot Mode Active — Simulation and pilot-safe workflow controls enabled",
                    "level": "info",
                }
            )
        return jsonify(
            {
                "message": "Pilot Mode Off — Workflow actions restricted",
                "level": "danger",
            }
        )

    @app.get("/api/v1/live-snapshot")
    def live_snapshot():
        snapshot = _simulated_snapshot()
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        focus = next((p for p in snapshot["patients"] if p["patient_id"] == patient_id), snapshot["patients"][0])

        return jsonify(
            {
                "tenant_id": tenant_id,
                "generated_at": snapshot["generated_at"],
                "alerts": snapshot["alerts"],
                "focus_patient": focus,
                "patients": snapshot["patients"],
                "summary": snapshot["summary"],
            }
        )

    @app.get("/api/command-center-stream")
    def command_center_stream():
        def generate():
            while True:
                snapshot = _simulated_snapshot()
                yield f"data: {json.dumps(snapshot)}\n\n"
                time.sleep(5)

        return Response(generate(), mimetype="text/event-stream")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": "early-risk-alert-ai", "time": _utc_now_iso()})

    return app
