from __future__ import annotations

import csv
import io
import json
import os
import random
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
)

# =========================================================
# Basic constants
# =========================================================

INFO_EMAIL = "info@earlyriskai.com"
FOUNDER_EMAIL = "Juneski93@gmail.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton MUNROE"
FOUNDER_ROLE = "Founder, Early Risk Alert AI"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"
PROD_BASE_URL = os.getenv("PROD_BASE_URL", "https://earlyriskalertai.com")

PILOT_MODE = True

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer": {"read"},
    "operator": {"read", "ack"},
    "clinician": {"read", "ack", "assign", "escalate"},
    "admin": {"read", "ack", "assign", "escalate", "resolve", "admin"},
}

DEFAULT_THRESHOLDS = {
    "icu": {"risk_alert": 7.0},
    "stepdown": {"risk_alert": 6.0},
    "telemetry": {"risk_alert": 5.0},
    "ward": {"risk_alert": 4.5},
    "rpm": {"risk_alert": 4.0},
}


# =========================================================
# HTML blocks
# =========================================================

HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Arial,sans-serif;background:#08111f;color:#eef4ff}
    .wrap{max-width:1100px;margin:0 auto;padding:32px 18px}
    .hero{padding:60px 0}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:24px;margin:18px 0}
    a.btn{display:inline-block;background:#7aa2ff;color:#08111f;text-decoration:none;padding:12px 18px;border-radius:12px;font-weight:700;margin:8px 10px 0 0}
    .muted{color:#9fb4d6}
    iframe{width:100%;aspect-ratio:16/9;border:0;border-radius:18px}
    h1,h2,h3{margin-top:0}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Early Risk Alert AI</h1>
      <p class="muted">AI-powered clinical monitoring, workflow visibility, and executive healthcare intelligence.</p>
      <a class="btn" href="/command-center">Open Command Center</a>
      <a class="btn" href="/hospital-demo">Hospital Demo Request</a>
      <a class="btn" href="/executive-walkthrough">Executive Walkthrough</a>
      <a class="btn" href="/investor-intake">Investor Intake</a>
    </section>

    <div class="card">
      <h2>Live Product Demo</h2>
      <iframe src="{{ youtube }}" allowfullscreen></iframe>
    </div>

    <div class="card">
      <h2>Contact</h2>
      <p>{{ founder }} · {{ role }}</p>
      <p>{{ email }} · {{ phone }}</p>
    </div>
  </div>
</body>
</html>
"""

COMMAND_CENTER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Arial,sans-serif;background:#07101c;color:#eef4ff}
    .wrap{max-width:1280px;margin:0 auto;padding:24px}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px}
    .wide{grid-column:span 2}
    .full{grid-column:1/-1}
    .muted{color:#9fb4d6}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left}
    button{background:#7aa2ff;color:#08111f;border:none;border-radius:10px;padding:8px 12px;font-weight:700;cursor:pointer}
    @media(max-width:900px){.grid{grid-template-columns:1fr}.wide,.full{grid-column:auto}}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Clinical Command Center</h1>
    <p class="muted">Pilot environment for simulated workflow visibility.</p>

    <div class="grid">
      <div class="card"><h3>Patient Count</h3><div id="patient-count">--</div></div>
      <div class="card"><h3>Open Alerts</h3><div id="open-alerts">--</div></div>
      <div class="card"><h3>Critical Alerts</h3><div id="critical-alerts">--</div></div>
      <div class="card"><h3>Avg Risk</h3><div id="avg-risk">--</div></div>

      <div class="card wide">
        <h3>Patients</h3>
        <table id="patients-table">
          <thead><tr><th>ID</th><th>Name</th><th>Unit</th><th>Risk</th><th>Status</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="card wide">
        <h3>Alerts</h3>
        <table id="alerts-table">
          <thead><tr><th>Patient</th><th>Severity</th><th>Text</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="card full">
        <h3>Focus Patient</h3>
        <pre id="focus-patient" style="white-space:pre-wrap"></pre>
      </div>
    </div>
  </div>

<script>
async function refreshDashboard(){
  const res = await fetch('/api/v1/live-snapshot');
  const data = await res.json();

  document.getElementById('patient-count').textContent = data.patients.length;
  document.getElementById('open-alerts').textContent = data.summary.open_alerts;
  document.getElementById('critical-alerts').textContent = data.summary.critical_alerts;
  document.getElementById('avg-risk').textContent = data.summary.avg_risk_score;

  const pbody = document.querySelector('#patients-table tbody');
  pbody.innerHTML = '';
  (data.patients || []).forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${p.patient_id}</td><td>${p.name}</td><td>${p.unit}</td><td>${p.risk.score}</td><td>${p.risk.severity}</td>`;
    pbody.appendChild(tr);
  });

  const abody = document.querySelector('#alerts-table tbody');
  abody.innerHTML = '';
  (data.alerts || []).forEach(a => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${a.patient_id}</td><td>${a.severity}</td><td>${a.text}</td>`;
    abody.appendChild(tr);
  });

  document.getElementById('focus-patient').textContent = JSON.stringify(data.focus_patient, null, 2);
}
refreshDashboard();
setInterval(refreshDashboard, 5000);
</script>
</body>
</html>
"""

ADMIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin Review</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Arial,sans-serif;background:#08111f;color:#eef4ff}
    .wrap{max-width:1200px;margin:0 auto;padding:24px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Admin Review</h1>
    <div class="card">
      <p>This page reads from <code>/admin/api/data</code>.</p>
      <p><a href="/admin/export.csv" style="color:#7aa2ff">Download CSV Export</a></p>
    </div>
  </div>
</body>
</html>
"""

LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Login</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Arial,sans-serif;background:#08111f;color:#eef4ff;display:grid;place-items:center;min-height:100vh}
    form{background:#101a2d;padding:28px;border-radius:18px;border:1px solid rgba(255,255,255,.08);width:min(420px,92vw)}
    input,select,button{width:100%;padding:12px;margin:8px 0;border-radius:10px;border:none}
    button{background:#7aa2ff;color:#08111f;font-weight:700}
  </style>
</head>
<body>
  <form method="post">
    <h2>Pilot Login</h2>
    <input name="user_name" placeholder="Name" required>
    <select name="user_role">
      <option value="viewer">viewer</option>
      <option value="operator">operator</option>
      <option value="clinician">clinician</option>
      <option value="admin">admin</option>
    </select>
    <button type="submit">Enter</button>
  </form>
</body>
</html>
"""

LEGAL_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Arial,sans-serif;background:#08111f;color:#eef4ff}
    .wrap{max-width:860px;margin:0 auto;padding:28px}
    .card{background:#101a2d;padding:24px;border-radius:18px;border:1px solid rgba(255,255,255,.08)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{{ title }}</h1>
      <div>{{ content|safe }}</div>
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
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Arial,sans-serif;background:#08111f;color:#eef4ff}
    .wrap{max-width:900px;margin:0 auto;padding:28px}
    .card{background:#101a2d;padding:24px;border-radius:18px;border:1px solid rgba(255,255,255,.08)}
    .field{margin:12px 0}
    label{display:block;margin-bottom:6px}
    input,select,textarea,button{width:100%;padding:12px;border-radius:10px;border:none}
    textarea{min-height:120px}
    button{background:#7aa2ff;color:#08111f;font-weight:700}
    .muted{color:#9fb4d6}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{{ heading }}</h1>
      <p class="muted">{{ copy }}</p>
      <form method="post">
        {{ fields|safe }}
        <button type="submit">{{ button_text }}</button>
      </form>
    </div>
  </div>
</body>
</html>
"""


# =========================================================
# Helpers
# =========================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir() -> Path:
    p = Path(__file__).resolve().parent.parent / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    rows = _read_jsonl(path)
    rows.append(payload)
    _write_jsonl(path, rows)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _current_role() -> str:
    role = str(session.get("user_role", "viewer")).strip().lower()
    return role if role in ROLE_PERMISSIONS else "viewer"


def _current_user() -> str:
    return str(session.get("user_name", "Guest")).strip() or "Guest"


def _has_permission(action: str) -> bool:
    return action in ROLE_PERMISSIONS.get(_current_role(), {"read"})


def _login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        return view_func(*args, **kwargs)
    return wrapped


def _role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect("/login")
            if _current_role() not in allowed_roles:
                return redirect("/login")
            return view_func(*args, **kwargs)
        return wrapped
    return decorator


def _severity_from_score(score: float) -> str:
    if score >= 8:
        return "critical"
    if score >= 6:
        return "high"
    if score >= 4.5:
        return "moderate"
    return "stable"


def _simulated_snapshot() -> dict[str, Any]:
    patients = []
    alerts = []

    units = ["icu", "stepdown", "telemetry", "ward"]
    names = ["Anderson", "Brooks", "Carter", "Davis", "Ellis", "Foster"]

    for i in range(6):
        score = round(random.uniform(2.5, 9.5), 1)
        sev = _severity_from_score(score)
        patient = {
            "patient_id": f"p10{i+1}",
            "name": f"Patient {names[i]}",
            "unit": random.choice(units),
            "risk": {"score": score, "severity": sev},
            "vitals": {
                "hr": random.randint(68, 128),
                "spo2": random.randint(90, 99),
                "rr": random.randint(14, 28),
                "bp": f"{random.randint(98,145)}/{random.randint(58,92)}",
            },
        }
        patients.append(patient)

        if sev != "stable":
            alerts.append(
                {
                    "patient_id": patient["patient_id"],
                    "severity": sev,
                    "text": f"{sev.title()} risk pattern detected",
                }
            )

    open_alerts = len(alerts)
    critical = len([a for a in alerts if a["severity"] == "critical"])
    avg_risk = round(sum(p["risk"]["score"] for p in patients) / len(patients), 1)
    focus_patient = sorted(patients, key=lambda x: x["risk"]["score"], reverse=True)[0]

    return {
        "tenant_id": request.args.get("tenant_id", "demo"),
        "generated_at": _utc_now_iso(),
        "patients": patients,
        "alerts": alerts,
        "focus_patient": focus_patient,
        "summary": {
            "open_alerts": open_alerts,
            "critical_alerts": critical,
            "avg_risk_score": avg_risk,
            "patients_with_alerts": open_alerts,
            "events_last_hour": random.randint(6, 24),
            "focus_patient_id": focus_patient["patient_id"],
        },
    }


def _workflow_file() -> Path:
    return _data_dir() / "command_center_workflow.json"


def _threshold_file() -> Path:
    return _data_dir() / "risk_thresholds.json"


def _load_workflow() -> dict[str, Any]:
    return _read_json(
        _workflow_file(),
        {"records": {}, "audit_log": []},
    )


def _save_workflow(store: dict[str, Any]) -> None:
    _write_json(_workflow_file(), store)


def _get_record(store: dict[str, Any], patient_id: str) -> dict[str, Any]:
    records = store.setdefault("records", {})
    if patient_id not in records:
        records[patient_id] = {
            "patient_id": patient_id,
            "ack": False,
            "assigned": False,
            "escalated": False,
            "state": "new",
            "updated_at": _utc_now_iso(),
            "role": _current_role(),
        }
    return records[patient_id]


def _audit(store: dict[str, Any], patient_id: str, action: str, role: str, note: str = "") -> None:
    log = store.setdefault("audit_log", [])
    log.insert(
        0,
        {
            "id": f"{patient_id}-{int(time.time() * 1000)}",
            "patient_id": patient_id,
            "action": action,
            "role": role,
            "note": note,
            "timestamp": _utc_now_iso(),
        },
    )
    del log[100:]


def _load_thresholds() -> dict[str, Any]:
    return _read_json(_threshold_file(), DEFAULT_THRESHOLDS)


def _save_thresholds(data: dict[str, Any]) -> None:
    _write_json(_threshold_file(), data)


def _lead_score(payload: dict[str, Any], lead_type: str) -> float:
    base = {
        "hospital": 8.4,
        "executive": 7.5,
        "investor": 7.9,
    }.get(lead_type, 6.0)
    return round(base + random.uniform(-1.0, 1.0), 1)


def _priority_tag(score: float, lead_type: str) -> str:
    if score >= 8.5:
        return f"{lead_type}-priority"
    if score >= 7:
        return f"{lead_type}-active"
    return f"{lead_type}-review"


def _status_norm(new_status: str, lead_type: str) -> str:
    value = (new_status or "").strip() or "New"
    return value


def _normalize_row(row: dict[str, Any], lead_type: str) -> dict[str, Any]:
    return {
        "lead_source": lead_type,
        "submitted_at": row.get("submitted_at", ""),
        "last_updated": row.get("last_updated", row.get("submitted_at", "")),
        "status": row.get("status", "New"),
        "lead_score": row.get("lead_score", ""),
        "priority_tag": row.get("priority_tag", ""),
        "full_name": row.get("full_name", ""),
        "organization": row.get("organization", ""),
        "role_or_title": row.get("role", row.get("title", "")),
        "email": row.get("email", ""),
        "phone": row.get("phone", ""),
        "category": row.get("facility_type", row.get("investor_type", "")),
        "timeline": row.get("timeline", ""),
        "message": row.get("message", ""),
    }


def _investor_stage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "New"))
        out[status] = out.get(status, 0) + 1
    return out


def _format_pretty_label(value: str) -> str:
    if not value:
        return "--"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%b %d %I:%M %p")
    except Exception:
        return value


def _send_admin_notification(_lead_type: str, _payload: dict[str, Any]) -> None:
    return


def _send_auto_reply(_lead_type: str, _payload: dict[str, Any]) -> None:
    return


def _detail_html(payload: dict[str, Any], keys: list[str]) -> str:
    items = []
    for key in keys:
        if key in payload and payload.get(key):
            items.append(f"<li><strong>{key}:</strong> {payload.get(key)}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _render_thank_you(message: str, detail_html: str) -> str:
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Submitted</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body{{margin:0;font-family:Arial,sans-serif;background:#08111f;color:#eef4ff}}
        .wrap{{max-width:860px;margin:0 auto;padding:28px}}
        .card{{background:#101a2d;padding:24px;border-radius:18px;border:1px solid rgba(255,255,255,.08)}}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="card">
          <h1>Submission received</h1>
          <p>{message}</p>
          {detail_html}
          <p><a href="/" style="color:#7aa2ff">Return home</a></p>
        </div>
      </div>
    </body>
    </html>
    """


# =========================================================
# App factory
# =========================================================

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "era-dev-secret")

    data_dir = _data_dir()
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"
    investor_file = data_dir / "investor_intake_requests.jsonl"

    def load_all_rows() -> dict[str, list[dict[str, Any]]]:
        return {
            "hospital": _read_jsonl(hospital_file),
            "executive": _read_jsonl(exec_file),
            "investor": _read_jsonl(investor_file),
        }

    def build_admin_rows() -> list[dict[str, Any]]:
        rows = load_all_rows()
        merged: list[dict[str, Any]] = []
        for lead_type, data in rows.items():
            for row in data:
                merged.append(_normalize_row(row, lead_type))
        merged.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)
        return merged

    def summary_payload() -> dict[str, Any]:
        raw = load_all_rows()
        merged = build_admin_rows()
        open_count = sum(1 for r in merged if r["status"] != "Closed")
        investor_stages = _investor_stage_summary(raw["investor"])
        last_updated = max([r["last_updated"] for r in merged], default="")
        return {
            "hospital_count": len(raw["hospital"]),
            "executive_count": len(raw["executive"]),
            "investor_count": len(raw["investor"]),
            "open_count": open_count,
            "investor_stages": investor_stages,
            "last_updated": _format_pretty_label(last_updated),
            "last_updated_label": _format_pretty_label(last_updated).split(" ")[1] if last_updated else "--",
        }

    @app.get("/")
    def home():
        return render_template_string(
            HOME_HTML,
            youtube=YOUTUBE_EMBED_URL,
            founder=FOUNDER_NAME,
            role=FOUNDER_ROLE,
            email=FOUNDER_EMAIL,
            phone=BUSINESS_PHONE,
        )

    @app.get("/command-center")
    @_login_required
    def command_center():
        return render_template_string(COMMAND_CENTER_HTML)

    @app.get("/admin/review")
    @_role_required("admin")
    def admin_review():
        return render_template_string(ADMIN_HTML)

    @app.get("/healthz")
    def healthz():
        summary = summary_payload()
        return jsonify({
            "ok": True,
            "service": "early-risk-alert-ai",
            "time": _utc_now_iso(),
            "hospital_requests": summary["hospital_count"],
            "executive_requests": summary["executive_count"],
            "investor_requests": summary["investor_count"],
            "open_requests": summary["open_count"],
        })

    @app.get("/api/v1/health")
    def api_health():
        return jsonify({"ok": True, "time": _utc_now_iso()})

    @app.get("/robots.txt")
    def robots_txt():
        return Response(
            "User-agent: *\nAllow: /\nSitemap: https://earlyriskalertai.com/sitemap.xml",
            mimetype="text/plain",
        )

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
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "hospital")
            _append_jsonl(hospital_file, payload)
            _send_admin_notification("hospital", payload)
            _send_auto_reply("hospital", payload)
            return _render_thank_you(
                "Your hospital demo request was submitted successfully.",
                _detail_html(payload, ["full_name", "organization", "role", "email", "facility_type", "timeline"]),
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
        <div class="field"><label>What would you like to see in the demo?</label><textarea name="message"></textarea></div>
        """
        return render_template_string(
            FORM_HTML,
            title="Hospital Demo – Early Risk Alert AI",
            heading="Request Hospital Demo",
            copy="Capture hospital interest, clinical use cases, and command-center demo requests.",
            fields=fields,
            button_text="Submit Hospital Demo Request",
        )

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
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "executive")
            _append_jsonl(exec_file, payload)
            _send_admin_notification("executive", payload)
            _send_auto_reply("executive", payload)
            return _render_thank_you(
                "Your executive walkthrough request was submitted successfully.",
                _detail_html(payload, ["full_name", "organization", "title", "email", "priority", "timeline"]),
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Executive Title</label><input name="title" required></div>
        <div class="field"><label>Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Phone</label><input name="phone"></div>
        <div class="field"><label>Priority</label>
          <select name="priority" required>
            <option value="Operational Review">Operational Review</option>
            <option value="Pilot Evaluation">Pilot Evaluation</option>
            <option value="Enterprise Discussion">Enterprise Discussion</option>
            <option value="Strategic Partnership">Strategic Partnership</option>
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
        <div class="field"><label>Walkthrough Focus</label><textarea name="message"></textarea></div>
        """
        return render_template_string(
            FORM_HTML,
            title="Executive Walkthrough – Early Risk Alert AI",
            heading="Schedule Executive Walkthrough",
            copy="Capture leadership-focused product reviews, pilot conversations, and strategic evaluation requests.",
            fields=fields,
            button_text="Submit Executive Walkthrough Request",
        )

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
                "investor_type": request.form.get("investor_type", "").strip(),
                "check_size": request.form.get("check_size", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            payload["lead_score"] = _lead_score(payload, "investor")
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "investor")
            _append_jsonl(investor_file, payload)
            _send_admin_notification("investor", payload)
            _send_auto_reply("investor", payload)
            return _render_thank_you(
                "Your investor intake was submitted successfully.",
                _detail_html(payload, ["full_name", "organization", "role", "email", "investor_type", "check_size", "timeline"]),
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" required></div>
        <div class="field"><label>Organization</label><input name="organization" required></div>
        <div class="field"><label>Role</label><input name="role" required></div>
        <div class="field"><label>Email</label><input type="email" name="email" required></div>
        <div class="field"><label>Phone</label><input name="phone"></div>
        <div class="field"><label>Investor Type</label>
          <select name="investor_type" required>
            <option value="Angel">Angel</option>
            <option value="Seed Fund">Seed Fund</option>
            <option value="Strategic">Strategic</option>
            <option value="Healthcare VC">Healthcare VC</option>
            <option value="Family Office">Family Office</option>
          </select>
        </div>
        <div class="field"><label>Check Size</label><input name="check_size" placeholder="$50K - $250K"></div>
        <div class="field"><label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field"><label>Interest / Notes</label><textarea name="message" placeholder="What are you interested in learning more about?"></textarea></div>
        """
        return render_template_string(
            FORM_HTML,
            title="Investor Intake – Early Risk Alert AI",
            heading="Investor Intake Form",
            copy="Capture investor interest, stage readiness, and commercial follow-up details directly from the platform.",
            fields=fields,
            button_text="Submit Investor Intake",
        )

    @app.get("/admin/api/data")
    @_role_required("admin")
    def admin_api_data():
        raw = load_all_rows()
        merged = build_admin_rows()
        investor_stages = _investor_stage_summary(raw["investor"])
        last_updated = max([r["last_updated"] for r in merged], default="")
        return jsonify({
            "rows": merged,
            "summary": {
                "hospital_count": len(raw["hospital"]),
                "executive_count": len(raw["executive"]),
                "investor_count": len(raw["investor"]),
                "open_count": sum(1 for r in merged if r["status"] != "Closed"),
                "investor_stages": investor_stages,
                "last_updated": _format_pretty_label(last_updated),
                "last_updated_label": _format_pretty_label(last_updated).split(" ")[1] if last_updated else "--",
            },
        })

    @app.post("/admin/api/status")
    @_role_required("admin")
    def admin_api_status():
        data = request.get_json(silent=True) or {}
        lead_type = str(data.get("lead_type", "")).strip()
        submitted_at = str(data.get("submitted_at", "")).strip()
        new_status = str(data.get("status", "")).strip()

        file_map = {
            "hospital": hospital_file,
            "executive": exec_file,
            "investor": investor_file,
        }

        path = file_map.get(lead_type)
        if not path or not submitted_at:
            return jsonify({"ok": False}), 400

        rows = _read_jsonl(path)
        for row in rows:
            if str(row.get("submitted_at", "")) == submitted_at:
                row["status"] = _status_norm(new_status, lead_type)
                row["last_updated"] = _utc_now_iso()
        _write_jsonl(path, rows)
        return jsonify({"ok": True})

    @app.get("/admin/export.csv")
    @_role_required("admin")
    def admin_export_csv():
        rows = []
        for lead_type, path in [("hospital", hospital_file), ("executive", exec_file), ("investor", investor_file)]:
            for row in _read_jsonl(path):
                rows.append(_normalize_row(row, lead_type))

        rows.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Lead Source", "Submitted At", "Last Updated", "Lead Status", "Lead Score",
            "Priority Tag", "Full Name", "Organization", "Role / Title", "Email",
            "Phone", "Category", "Timeline", "Message"
        ])
        for row in rows:
            writer.writerow([
                row["lead_source"], row["submitted_at"], row["last_updated"], row["status"],
                row["lead_score"], row["priority_tag"], row["full_name"], row["organization"],
                row["role_or_title"], row["email"], row["phone"], row["category"],
                row["timeline"], row["message"],
            ])

        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(
            mem,
            mimetype="text/csv",
            as_attachment=True,
            download_name="early_risk_alert_pipeline_export.csv",
        )

    @app.get("/api/v1/dashboard/overview")
    def dashboard_overview():
        snapshot = _simulated_snapshot()
        raw = load_all_rows()
        return jsonify({
            "tenant_id": request.args.get("tenant_id", "demo"),
            "patient_count": len(snapshot["patients"]),
            "open_alerts": snapshot["summary"]["open_alerts"],
            "critical_alerts": snapshot["summary"]["critical_alerts"],
            "events_last_hour": snapshot["summary"]["events_last_hour"],
            "avg_risk_score": snapshot["summary"]["avg_risk_score"],
            "patients_with_alerts": snapshot["summary"]["patients_with_alerts"],
            "focus_patient_id": snapshot["summary"]["focus_patient_id"],
            "hospital_requests": len(raw["hospital"]),
            "executive_requests": len(raw["executive"]),
            "investor_requests": len(raw["investor"]),
        })

    @app.get("/api/v1/live-snapshot")
    def live_snapshot():
        snapshot = _simulated_snapshot()
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        focus = next((r for r in snapshot["patients"] if r["patient_id"] == patient_id), snapshot["patients"][0])
        return jsonify({
            "tenant_id": tenant_id,
            "generated_at": snapshot["generated_at"],
            "alerts": snapshot["alerts"],
            "focus_patient": focus,
            "patients": snapshot["patients"],
            "summary": snapshot["summary"],
        })

    @app.get("/api/v1/command-workflow")
    @_login_required
    def command_workflow():
        store = _load_workflow()
        return jsonify({
            "records": store.get("records", {}),
            "audit_log": store.get("audit_log", []),
        })

    @app.post("/api/workflow/action")
    @_login_required
    def workflow_action():
        payload = request.get_json(silent=True) or {}
        patient_id = str(payload.get("patient_id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        role = _current_role()

        if not patient_id:
            return jsonify({"ok": False, "error": "patient_id required"}), 400

        store = _load_workflow()
        record = _get_record(store, patient_id)

        if action == "ack":
            if not _has_permission("ack"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["ack"] = True
            record["state"] = "acknowledged"
            _audit(store, patient_id, "ACK", role)

        elif action == "assign":
            if not _has_permission("assign"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["assigned"] = True
            record["state"] = "assigned"
            _audit(store, patient_id, "ASSIGN", role)

        elif action == "escalate":
            if not _has_permission("escalate"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["escalated"] = True
            record["state"] = "escalated"
            _audit(store, patient_id, "ESCALATE", role)

        elif action == "resolve":
            if not _has_permission("resolve") and not _has_permission("admin"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["state"] = "resolved"
            _audit(store, patient_id, "RESOLVE", role)

        else:
            return jsonify({"ok": False, "error": "invalid action"}), 400

        record["updated_at"] = _utc_now_iso()
        record["role"] = role
        _save_workflow(store)

        return jsonify({
            "ok": True,
            "record": record,
            "user_role": role,
            "user_name": _current_user(),
        })

    @app.get("/api/reporting")
    @_login_required
    def reporting():
        store = _load_workflow()
        records = list(store["records"].values())
        return jsonify({
            "total_patients": len(records),
            "acknowledged": len([r for r in records if r.get("ack")]),
            "assigned": len([r for r in records if r.get("assigned")]),
            "escalated": len([r for r in records if r.get("escalated")]),
            "resolved": len([r for r in records if r.get("state") == "resolved"]),
            "audit_events": len(store["audit_log"]),
            "pilot_mode": bool(session.get("pilot_mode", PILOT_MODE)),
            "user_role": _current_role(),
            "user_name": _current_user(),
        })

    @app.get("/api/audit/export")
    @_login_required
    def export_audit():
        if not _has_permission("admin"):
            return jsonify({"ok": False, "error": "permission denied"}), 403

        store = _load_workflow()
        return jsonify({
            "audit_log": store["audit_log"],
            "exported_by": _current_user(),
            "exported_role": _current_role(),
            "pilot_mode": bool(session.get("pilot_mode", PILOT_MODE)),
        })

    @app.get("/api/thresholds")
    @_login_required
    def get_thresholds():
        return jsonify(_load_thresholds())

    @app.post("/api/thresholds")
    @_login_required
    def update_thresholds():
        data = request.get_json(silent=True) or {}
        _save_thresholds(data)
        return jsonify({"ok": True})

    @app.get("/api/system-health")
    @_login_required
    def system_health():
        return jsonify({
            "ok": True,
            "service": "early-risk-alert-ai",
            "pilot_mode": bool(session.get("pilot_mode", PILOT_MODE)),
            "generated_at": _utc_now_iso(),
            "workflow_records": len(_load_workflow().get("records", {})),
            "audit_events": len(_load_workflow().get("audit_log", [])),
        })

    @app.get("/api/pilot-status")
    def pilot_status():
        return jsonify({
            "pilot_mode": bool(session.get("pilot_mode", PILOT_MODE)),
            "logged_in": bool(session.get("logged_in")),
            "user_name": _current_user(),
            "user_role": _current_role(),
            "permissions": sorted(list(ROLE_PERMISSIONS.get(_current_role(), {"read"}))),
        })

    @app.get("/api/v1/stream/channels")
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
            ],
        })

    @app.get("/api/command-center-stream")
    def command_center_stream():
        def generate():
            while True:
                snapshot = _simulated_snapshot()
                yield f"data: {json.dumps(snapshot)}\\n\\n"
                time.sleep(2)

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user_name = request.form.get("user_name", "").strip() or "Pilot User"
            user_role = request.form.get("user_role", "viewer").strip().lower()
            if user_role not in ROLE_PERMISSIONS:
                user_role = "viewer"

            session["logged_in"] = True
            session["user_name"] = user_name
            session["user_role"] = user_role
            session["pilot_mode"] = True
            return redirect("/command-center")

        return render_template_string(LOGIN_HTML)

    @app.get("/logout")
    def pilot_logout():
        session.clear()
        return redirect("/login")

    @app.get("/terms")
    def terms_page():
        content = """
        <h2>Evaluation Use</h2>
        <p>This platform environment is intended for demonstration, pilot review, and workflow evaluation.</p>
        <h2>No Clinical Replacement</h2>
        <p>Users must not rely on this pilot environment as a sole basis for diagnosis, treatment, or emergency action.</p>
        <h2>Organizational Review</h2>
        <p>Any hospital, clinic, insurer, or reviewer should evaluate the platform in accordance with internal policy and compliance requirements.</p>
        """
        return render_template_string(LEGAL_PAGE_HTML, title="Terms of Use", content=content)

    @app.get("/privacy")
    def privacy_page():
        content = """
        <h2>Privacy Notice</h2>
        <p>This pilot environment may display simulated data, demonstration workflows, evaluation analytics, and request submissions.</p>
        <h2>Non-Production Handling</h2>
        <p>Production patient data should only be used in appropriately secured and compliant deployments.</p>
        <h2>Pilot Records</h2>
        <p>Workflow actions, audit events, and pilot configuration changes may be stored for evaluation and review purposes.</p>
        """
        return render_template_string(LEGAL_PAGE_HTML, title="Privacy Notice", content=content)

    @app.get("/pilot-disclaimer")
    def pilot_disclaimer_page():
        content = """
        <h2>Pilot Environment</h2>
        <p>This environment is provided for workflow demonstration, pilot evaluation, user testing, and product review.</p>
        <h2>Not for Emergency Response</h2>
        <p>It must not be used as a substitute for established clinical escalation protocols, bedside care, or medical judgment.</p>
        <h2>Professional Review Required</h2>
        <p>All outputs, alerts, recommendations, and risk signals should be interpreted by qualified professionals.</p>
        """
        return render_template_string(LEGAL_PAGE_HTML, title="Pilot Disclaimer", content=content)

    return app
