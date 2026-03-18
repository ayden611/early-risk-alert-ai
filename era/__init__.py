from __future__ import annotations

import csv
import html
import io
import json
import os
import random
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, session

from era.web.command_center import COMMAND_CENTER_HTML


INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder, Early Risk Alert AI"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"
CANONICAL_HOST = os.getenv("CANONICAL_HOST", "").strip().lower()

PILOT_MODE = True

ROLE_PERMISSIONS = {
    "viewer": {"read"},
    "operator": {"read", "ack", "assign"},
    "nurse": {"read", "ack", "assign"},
    "physician": {"read", "ack", "assign", "escalate", "resolve"},
    "admin": {"read", "ack", "assign", "escalate", "resolve", "admin"},
}

DEFAULT_THRESHOLDS = {
    "icu": {"risk_alert": 7.0},
    "stepdown": {"risk_alert": 6.0},
    "telemetry": {"risk_alert": 5.0},
    "ward": {"risk_alert": 4.5},
    "rpm": {"risk_alert": 4.0},
}


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
    return {"critical": 4, "high": 3, "moderate": 2, "stable": 1}.get((severity or "").lower(), 0)


def _clamp(n: float, low: float, high: float) -> float:
    return max(low, min(high, n))


def _status_norm(value: str) -> str:
    v = (value or "New").strip().lower()
    if v == "contacted":
        return "Contacted"
    if v == "scheduled":
        return "Scheduled"
    return "New"


def _status_class(value: str) -> str:
    v = _status_norm(value).lower()
    if v == "contacted":
        return "status-pill status-contacted"
    if v == "scheduled":
        return "status-pill status-scheduled"
    return "status-pill status-new"


def _recommended_action(severity: str, vitals: Dict[str, Any]) -> str:
    spo2 = _safe_float(vitals.get("spo2"))
    hr = _safe_float(vitals.get("heart_rate"))
    sbp = _safe_float(vitals.get("systolic_bp"))

    if severity == "critical":
        if spo2 < 90:
            return "Escalate immediately to respiratory response and bedside reassessment."
        if sbp >= 180:
            return "Immediate physician escalation and hypertensive crisis review."
        if hr >= 130:
            return "Immediate cardiac assessment and rapid intervention workflow."
        return "Escalate immediately to rapid response workflow."
    if severity == "high":
        return "Notify assigned clinician, review vitals, and reassess within 15 minutes."
    if severity == "moderate":
        return "Continue close monitoring and validate trend progression."
    return "Continue routine monitoring."


def _clinical_priority(severity: str) -> str:
    return {
        "critical": "Priority 1",
        "high": "Priority 2",
        "moderate": "Priority 3",
        "stable": "Priority 4",
    }.get(severity, "Priority 4")


def _trend_direction(vitals: Dict[str, Any]) -> str:
    score = 0
    if _safe_float(vitals.get("spo2")) < 94:
        score += 1
    if _safe_float(vitals.get("heart_rate")) > 105:
        score += 1
    if _safe_float(vitals.get("systolic_bp")) > 150:
        score += 1
    if score >= 2:
        return "Worsening"
    if score == 1:
        return "Elevated"
    return "Stable"


def _trend_arrow(trend: str) -> str:
    t = (trend or "").lower()
    if t == "worsening":
        return "↑"
    if t == "elevated":
        return "→"
    return "↓"


def _detect_risk(vitals: Dict[str, Any]) -> Dict[str, Any]:
    heart_rate = _safe_float(vitals.get("heart_rate"))
    systolic_bp = _safe_float(vitals.get("systolic_bp"))
    diastolic_bp = _safe_float(vitals.get("diastolic_bp"))
    spo2 = _safe_float(vitals.get("spo2"))
    temperature = _safe_float(vitals.get("temperature"), 98.6)
    resp_rate = _safe_float(vitals.get("resp_rate"), 16)

    score = 8.0
    severity = "stable"
    alert_type = None
    alert_message = "Vitals stable"

    if spo2 < 90:
        score += 42
        severity = "critical"
        alert_type = "low_spo2"
        alert_message = "Oxygen saturation critical"
    elif spo2 < 94:
        score += 20
        severity = "high"
        alert_type = "spo2_drop"
        alert_message = "Oxygen saturation below target"

    if heart_rate >= 130:
        score += 30
        if _severity_rank("critical") > _severity_rank(severity):
            severity = "critical"
            alert_type = "tachycardia"
            alert_message = "Heart rate critically elevated"
    elif heart_rate >= 110:
        score += 18
        if _severity_rank("high") > _severity_rank(severity):
            severity = "high"
            alert_type = "tachycardia"
            alert_message = "Heart rate elevated"

    if systolic_bp >= 180 or diastolic_bp >= 110:
        score += 28
        if _severity_rank("critical") > _severity_rank(severity):
            severity = "critical"
            alert_type = "hypertensive_crisis"
            alert_message = "Blood pressure critical"
    elif systolic_bp >= 160 or diastolic_bp >= 100:
        score += 16
        if _severity_rank("high") > _severity_rank(severity):
            severity = "high"
            alert_type = "hypertension"
            alert_message = "Blood pressure elevated"

    if temperature >= 101.5:
        score += 10
        if _severity_rank("moderate") > _severity_rank(severity):
            severity = "moderate"
            alert_type = "fever"
            alert_message = "Temperature elevated"

    if resp_rate >= 24:
        score += 10
        if _severity_rank("moderate") > _severity_rank(severity):
            severity = "moderate"
            alert_type = "respiratory_rate"
            alert_message = "Respiratory rate elevated"

    score = int(_clamp(round(score), 1, 99))
    confidence = int(_clamp(round(58 + (score * 0.38)), 58, 98))
    trend = _trend_direction(vitals)

    return {
        "risk_score": score,
        "confidence": confidence,
        "severity": severity,
        "alert_type": alert_type,
        "alert_message": alert_message,
        "recommended_action": _recommended_action(severity, vitals),
        "clinical_priority": _clinical_priority(severity),
        "trend_direction": trend,
        "trend_arrow": _trend_arrow(trend),
    }


def _build_demo_patients() -> List[Dict[str, Any]]:
    seed = [
        {"patient_id": "p101", "patient_name": "Patient p101", "room": "ICU-12", "program": "Cardiac", "vitals": {"heart_rate": 126, "systolic_bp": 172, "diastolic_bp": 104, "spo2": 92, "temperature": 99.4, "resp_rate": 20}},
        {"patient_id": "p202", "patient_name": "Patient p202", "room": "Stepdown-04", "program": "Pulmonary", "vitals": {"heart_rate": 98, "systolic_bp": 146, "diastolic_bp": 92, "spo2": 88, "temperature": 100.2, "resp_rate": 24}},
        {"patient_id": "p303", "patient_name": "Patient p303", "room": "Telemetry-09", "program": "Cardiac", "vitals": {"heart_rate": 112, "systolic_bp": 164, "diastolic_bp": 98, "spo2": 95, "temperature": 98.9, "resp_rate": 18}},
        {"patient_id": "p404", "patient_name": "Patient p404", "room": "Ward-21", "program": "Recovery", "vitals": {"heart_rate": 84, "systolic_bp": 128, "diastolic_bp": 82, "spo2": 97, "temperature": 98.6, "resp_rate": 16}},
        {"patient_id": "p505", "patient_name": "Patient p505", "room": "ICU-05", "program": "Stroke", "vitals": {"heart_rate": 118, "systolic_bp": 156, "diastolic_bp": 96, "spo2": 93, "temperature": 99.8, "resp_rate": 22}},
        {"patient_id": "p606", "patient_name": "Patient p606", "room": "RPM-Home", "program": "Remote Monitoring", "vitals": {"heart_rate": 89, "systolic_bp": 138, "diastolic_bp": 86, "spo2": 96, "temperature": 98.4, "resp_rate": 17}},
    ]
    rows: List[Dict[str, Any]] = []
    for patient in seed:
        v = dict(patient["vitals"])
        v["heart_rate"] = max(55, int(v["heart_rate"] + random.randint(-5, 5)))
        v["systolic_bp"] = max(90, int(v["systolic_bp"] + random.randint(-5, 5)))
        v["diastolic_bp"] = max(50, int(v["diastolic_bp"] + random.randint(-4, 4)))
        v["spo2"] = max(82, min(99, int(v["spo2"] + random.randint(-1, 1))))
        v["temperature"] = round(v["temperature"] + random.uniform(-0.2, 0.2), 1)
        v["resp_rate"] = max(10, int(v["resp_rate"] + random.randint(-1, 1)))
        risk = _detect_risk(v)
        rows.append(
            {
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "room": patient["room"],
                "program": patient["program"],
                "event_ts": _utc_now_iso(),
                "vitals": v,
                "risk": risk,
            }
        )
    return rows


def _build_alerts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    for row in rows:
        risk = row["risk"]
        if risk.get("alert_type"):
            alerts.append(
                {
                    "patient_id": row["patient_id"],
                    "patient_name": row["patient_name"],
                    "room": row["room"],
                    "program": row["program"],
                    "alert_type": risk["alert_type"],
                    "severity": risk["severity"],
                    "message": risk["alert_message"],
                    "risk_score": risk["risk_score"],
                    "confidence": risk["confidence"],
                    "recommended_action": risk["recommended_action"],
                    "clinical_priority": risk["clinical_priority"],
                    "trend_direction": risk["trend_direction"],
                    "trend_arrow": risk["trend_arrow"],
                    "created_at": row["event_ts"],
                }
            )
    alerts.sort(key=lambda a: (_severity_rank(a["severity"]), a["risk_score"]), reverse=True)
    return alerts


def _build_rollups(row: Dict[str, Any]) -> Dict[str, Any]:
    vitals = row["vitals"]
    return {
        "avg_heart_rate": int(_safe_float(vitals.get("heart_rate"))),
        "avg_systolic_bp": int(_safe_float(vitals.get("systolic_bp"))),
        "avg_diastolic_bp": int(_safe_float(vitals.get("diastolic_bp"))),
        "avg_spo2": int(_safe_float(vitals.get("spo2"))),
    }


def _escape(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate_pitch_deck_pdf_bytes() -> bytes:
    lines = [
        "Early Risk Alert AI",
        "Predictive Clinical Intelligence Platform",
        "",
        "Executive Summary",
        "Early Risk Alert AI is a professional healthcare intelligence platform built for",
        "hospitals, clinics, command centers, executive buyers, and remote monitoring programs.",
        "The platform combines AI risk scoring, clinical visibility, hospital demo capture,",
        "executive walkthrough flow, investor intake, and branded presentation architecture.",
        "",
        "Hospital Value",
        "- Detect patient deterioration earlier",
        "- Prioritize alerts by severity and confidence",
        "- Present enterprise-grade command-center visibility",
        "- Support hospital demo and executive evaluation workflows",
        "",
        "Investor Value",
        "- Enterprise SaaS positioning",
        "- Hospital and RPM market exposure",
        "- Built-in investor intake and admin review flow",
        "- Real branded deck delivery and pipeline export",
        "",
        "Founder",
        f"- {FOUNDER_NAME}",
        f"- {FOUNDER_ROLE}",
        "",
        "Contact",
        f"- Email: {INFO_EMAIL}",
        f"- Phone: {BUSINESS_PHONE}",
    ]

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 790
    content = ["BT", "/F1 24 Tf", "50 810 Td", f"({esc('Early Risk Alert AI — Executive Pitch Deck')}) Tj"]
    for line in lines:
        y -= 24
        if not line:
            continue
        size = 12
        if line in {"Early Risk Alert AI", "Executive Summary", "Hospital Value", "Investor Value", "Founder", "Contact"}:
            size = 17
        content.extend([f"/F1 {size} Tf", f"50 {y} Td", f"({esc(line)}) Tj"])
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", errors="replace")

    objs = []
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objs.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objs.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n")
    objs.append(f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n")

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = bytearray(header)
    offsets = [0]
    for obj in objs:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    xref = [b"xref\n", f"0 {len(objs)+1}\n".encode("latin-1"), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode("latin-1"))
    trailer = f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    body.extend(b"".join(xref))
    body.extend(trailer)
    return bytes(body)


LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Login — Early Risk Alert AI Pilot Access</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;--bg2:#0b1528;--panel:#101a2d;--line:rgba(255,255,255,.08);
      --text:#eef4ff;--muted:#a7bddc;--blue:#7aa2ff;--blue2:#5bd4ff;--amber:#f4bd6a;
    }
    *{box-sizing:border-box}
    body{
      margin:0;min-height:100vh;display:grid;place-items:center;padding:20px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.14), transparent 24%),
        radial-gradient(circle at 88% 10%, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }
    .card{
      width:min(100%, 560px);
      border:1px solid var(--line);border-radius:28px;padding:28px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    .kicker{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff}
    h1{margin:10px 0 12px;font-size:42px;line-height:.94;letter-spacing:-.05em}
    p{color:var(--muted);line-height:1.7}
    form{display:grid;gap:14px;margin-top:18px}
    label{font-size:12px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb8dc}
    input,select{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.08);
      background:#0d1728;color:var(--text);font:inherit
    }
    .btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:14px 18px;border-radius:16px;border:0;cursor:pointer;
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#08111f;font-weight:1000;font-size:14px
    }
    .notice{
      margin-top:16px;padding:14px 16px;border-radius:16px;
      background:rgba(244,189,106,.12);border:1px solid rgba(244,189,106,.20);color:#ffe4b6;
      font-size:13px;line-height:1.6
    }
    .links{margin-top:16px;display:flex;gap:12px;flex-wrap:wrap}
    .links a{color:#dce9ff;text-decoration:none;font-weight:800;font-size:14px}
  </style>
</head>
<body>
  <div class="card">
    <div class="kicker">Pilot-safe access</div>
    <h1>Early Risk Alert AI Pilot Login</h1>
    <p>
      This environment is intended for platform evaluation, pilot demonstrations, and workflow review.
      It is not a production clinical system and does not replace clinician judgment, hospital policy,
      or emergency response procedures.
    </p>

    <form method="post">
      <div>
        <label>Name</label>
        <input name="user_name" placeholder="Enter your name" required>
      </div>
      <div>
        <label>Role</label>
        <select name="user_role" required>
          <option value="viewer">Viewer</option>
          <option value="operator">Operator</option>
          <option value="nurse">Nurse</option>
          <option value="physician">Physician</option>
          <option value="admin">Admin</option>
        </select>
      </div>
      <button class="btn" type="submit">Enter Pilot Environment</button>
    </form>

    <div class="notice">
      Pilot Mode is enabled. Demo and pilot workflows should be reviewed within your organization's
      own clinical, compliance, legal, privacy, and technical evaluation process.
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

LEGAL_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;--bg2:#0b1528;--panel:#101a2d;--line:rgba(255,255,255,.08);
      --text:#eef4ff;--muted:#a7bddc;--blue:#7aa2ff;--blue2:#5bd4ff;
    }
    *{box-sizing:border-box}
    body{
      margin:0;padding:20px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.14), transparent 24%),
        radial-gradient(circle at 88% 10%, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }
    .wrap{max-width:980px;margin:0 auto}
    .card{
      border:1px solid var(--line);border-radius:28px;padding:28px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    .kicker{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff}
    h1{margin:10px 0 16px;font-size:42px;line-height:.94;letter-spacing:-.05em}
    h2{margin:24px 0 8px;font-size:20px}
    p{color:var(--muted);line-height:1.75}
    .actions{margin-top:20px;display:flex;gap:12px;flex-wrap:wrap}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:16px;
      text-decoration:none;font-weight:1000;font-size:14px
    }
    .primary{background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f}
    .secondary{background:rgba(255,255,255,.04);border:1px solid var(--line);color:var(--text)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="kicker">Pilot safety</div>
      <h1>__TITLE__</h1>
      __CONTENT__
      <div class="actions">
        <a class="btn primary" href="/login">Back to Login</a>
        <a class="btn secondary" href="/command-center">Command Center</a>
      </div>
    </div>
  </div>
</body>
</html>
"""

FORM_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--bg:#08111f;--panel:#101a2d;--line:rgba(255,255,255,.08);--text:#edf4ff;--muted:#c6d7ef;--blue:#7aa2ff;--blue2:#5bd4ff;--shadow:0 18px 50px rgba(0,0,0,.24)}
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:var(--text)}
    .form-shell{max-width:1120px;margin:0 auto;padding:36px 18px 54px}
    .form-card{
      background:radial-gradient(circle at top left, rgba(122,162,255,.10), transparent 24%),linear-gradient(180deg, rgba(16,26,45,.98), rgba(12,18,32,.98));
      border:1px solid rgba(255,255,255,.08);border-radius:24px;padding:30px;box-shadow:var(--shadow)
    }
    .form-top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap;margin-bottom:22px}
    .form-title{font-size:clamp(38px,4vw,54px);font-weight:1000;line-height:.98;margin:0 0 10px;letter-spacing:-.045em}
    .form-copy{color:#c6d7ef;line-height:1.7;margin:0;max-width:760px}
    .form-badge{padding:10px 14px;border-radius:999px;background:rgba(56,211,159,.12);border:1px solid rgba(56,211,159,.22);color:#c8ffe5;font-weight:900;font-size:12px;letter-spacing:.1em;text-transform:uppercase}
    .form{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
    .full{grid-column:1/-1}
    .field{display:flex;flex-direction:column;gap:8px}
    .field label{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9eb4d6;font-weight:900}
    .field input,.field select,.field textarea{
      width:100%;background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:16px;color:#edf4ff;padding:15px 15px;font:inherit;
    }
    .field textarea{min-height:140px;resize:vertical}
    .btn{display:inline-flex;align-items:center;justify-content:center;padding:14px 18px;border-radius:16px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;font-weight:1000;text-decoration:none;border:none;cursor:pointer}
    @media (max-width:760px){.form{grid-template-columns:1fr}.form-shell{padding:14px 10px 48px}.form-card{padding:16px}.form-top{flex-direction:column}.form-title{font-size:36px}}
  </style>
</head>
<body>
  <div class="form-shell">
    <div class="form-card">
      <div class="form-top">
        <div>
          <h1 class="form-title">__HEADING__</h1>
          <p class="form-copy">__COPY__</p>
        </div>
        <div class="form-badge">Enterprise Intake</div>
      </div>
      <form method="post" class="form">
        __FIELDS__
        <div class="field full"><button class="btn" type="submit">__BUTTON__</button></div>
      </form>
    </div>
  </div>
</body>
</html>
"""

THANK_YOU_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Thank You — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;--text:#edf4ff;--muted:#c6d7ef;--blue:#7aa2ff;--blue2:#5bd4ff;--line:rgba(255,255,255,.08);
      --shadow:0 18px 50px rgba(0,0,0,.24)
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:var(--text)}
    .success-card{max-width:980px;margin:0 auto;padding:36px 18px 54px}
    .success-box{
      background:__BG__;
      border:1px solid var(--line);border-radius:24px;padding:32px;box-shadow:var(--shadow)
    }
    .ok{
      display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:999px;background:__BADGE_BG__;
      border:1px solid __BADGE_BORDER__;color:__BADGE_TEXT__;font-weight:900;font-size:12px;letter-spacing:.08em;text-transform:uppercase
    }
    h1{margin:12px 0 10px;font-size:clamp(38px,4vw,54px);letter-spacing:-.045em}
    p{color:var(--muted);line-height:1.75}
    .detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}
    .detail{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:16px;padding:14px}
    .dk{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:#9eb4d6;font-weight:900}
    .dv{margin-top:6px;color:#edf4ff;line-height:1.5}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;font-weight:900;text-decoration:none;margin-right:10px;margin-top:10px}
    @media (max-width:760px){.detail-grid{grid-template-columns:1fr}.success-card{padding:14px 10px 48px}.success-box{padding:16px}}
  </style>
</head>
<body>
  <div class="success-card">
    <div class="success-box">
      <div class="ok">__TYPE_LABEL__ received</div>
      <h1>Thank you</h1>
      <p>__THANK_YOU_COPY__</p>
      <div class="detail-grid">__DETAILS__</div>
      <p><a class="btn" href="/command-center">Return to Command Center</a><a class="btn" href="/admin/review">Open Admin Review</a></p>
    </div>
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
      --bg:#08111f;--panel:#101a2d;--line:rgba(255,255,255,.08);--text:#edf4ff;--muted:#bdd0ec;
      --blue:#7aa2ff;--blue2:#5bd4ff;--green:#38d39f;--amber:#f7be68;--red:#ff5c72;--shadow:0 18px 50px rgba(0,0,0,.24)
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:var(--text)}
    .wrap{max-width:1380px;margin:0 auto;padding:36px 18px 60px}
    .card{
      background:radial-gradient(circle at top right, rgba(91,212,255,.10), transparent 24%),linear-gradient(180deg, rgba(16,26,45,.98), rgba(12,18,32,.98));
      border:1px solid var(--line);border-radius:24px;padding:30px;box-shadow:var(--shadow)
    }
    h1{font-size:clamp(40px,4vw,56px);line-height:.98;margin:0 0 14px;letter-spacing:-.045em}
    h2{margin:0 0 10px;font-size:32px;letter-spacing:-.03em}
    p{color:var(--muted);line-height:1.7}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;font-weight:900;text-decoration:none;margin-right:10px;margin-bottom:10px}
    .admin-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0 24px}
    .admin-kpi{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.06);border-radius:20px;padding:24px;min-height:154px;display:flex;flex-direction:column;justify-content:space-between
    }
    .admin-kpi .k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .admin-kpi .v{font-size:44px;font-weight:1000;line-height:1;margin-top:14px}
    .admin-kpi .hint{font-size:13px;color:#c6d7ef;line-height:1.5}
    .section{margin-top:34px}
    .table-wrap{overflow:auto;border:1px solid rgba(255,255,255,.06);border-radius:18px;background:rgba(255,255,255,.02)}
    table{width:100%;border-collapse:collapse;font-size:14px;min-width:920px}
    th,td{padding:14px 12px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;vertical-align:top}
    th{color:#9eb4d6;text-transform:uppercase;font-size:12px;letter-spacing:.12em;background:rgba(255,255,255,.02);position:sticky;top:0}
    .empty{padding:18px;color:#c9daf0}
    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;min-width:108px;padding:9px 12px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.08em;text-transform:uppercase
    }
    .status-new{background:rgba(91,212,255,.16);border:1px solid rgba(91,212,255,.28);color:#d6f4ff}
    .status-contacted{background:rgba(247,190,104,.16);border:1px solid rgba(247,190,104,.28);color:#ffe3b2}
    .status-scheduled{background:rgba(56,211,159,.16);border:1px solid rgba(56,211,159,.28);color:#d3ffe8}
    .action-row{display:flex;gap:8px;flex-wrap:wrap}
    .mini-btn{
      display:inline-flex;align-items:center;justify-content:center;padding:9px 12px;border-radius:12px;background:#111b2f;border:1px solid rgba(255,255,255,.08);color:#edf4ff;font-weight:900;text-decoration:none;font-size:12px
    }
    @media (max-width:980px){.admin-grid{grid-template-columns:1fr}.wrap{padding:14px 10px 48px}.card{padding:16px}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Admin Review</h1>
      <p>Review hospital demo requests, executive walkthrough requests, and investor intake submissions. Update lead status and export CSV from one page.</p>
      <p><a class="btn" href="/admin/export.csv">Download CSV</a><a class="btn" href="/command-center">Command Center</a></p>

      <div class="admin-grid">
        <div class="admin-kpi"><div class="k">Hospital Demo Requests</div><div class="v">__HOSPITAL_COUNT__</div><div class="hint">Clinical buyer and operator pipeline</div></div>
        <div class="admin-kpi"><div class="k">Executive Walkthrough Requests</div><div class="v">__EXEC_COUNT__</div><div class="hint">Leadership and pilot evaluation requests</div></div>
        <div class="admin-kpi"><div class="k">Investor Intake Requests</div><div class="v">__INVESTOR_COUNT__</div><div class="hint">Commercial and investor follow-up flow</div></div>
      </div>

      <div class="section"><h2>Hospital Demo Requests</h2>__HOSPITAL_TABLE__</div>
      <div class="section"><h2>Executive Walkthrough Requests</h2>__EXEC_TABLE__</div>
      <div class="section"><h2>Investor Intake</h2>__INVESTOR_TABLE__</div>
    </div>
  </div>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.secret_key = os.getenv("SECRET_KEY", "early-risk-alert-dev-secret")

    data_dir = Path(app.instance_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    investor_file = data_dir / "investor_intake.jsonl"
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"
    workflow_file = data_dir / "command_center_workflow.json"
    threshold_file = data_dir / "risk_thresholds.json"

    def _save_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        rows.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
        return rows

    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _update_row_status(path: Path, submitted_at: str, new_status: str) -> bool:
        rows = _read_jsonl(path)
        updated = False
        for row in rows:
            if row.get("submitted_at") == submitted_at:
                row["status"] = _status_norm(new_status)
                updated = True
                break
        if updated:
            _write_jsonl(path, rows)
        return updated

    def _detail_html(payload: Dict[str, Any], keys: List[str]) -> str:
        blocks = []
        for key in keys:
            value = payload.get(key, "")
            if not value:
                continue
            blocks.append(
                "<div class='detail'>"
                f"<div class='dk'>{_escape(key.replace('_', ' ').title())}</div>"
                f"<div class='dv'>{_escape(value)}</div>"
                "</div>"
            )
        return "".join(blocks)

    def _thank_you_theme(kind: str) -> Tuple[str, str, str, str, str]:
        k = (kind or "").lower()
        if k == "hospital":
            return (
                "radial-gradient(circle at top left, rgba(91,212,255,.14), transparent 24%),linear-gradient(180deg, rgba(16,26,45,.98), rgba(12,18,32,.98))",
                "rgba(91,212,255,.12)",
                "rgba(91,212,255,.20)",
                "#d6f4ff",
                "Hospital Demo Request",
            )
        if k == "executive":
            return (
                "radial-gradient(circle at top left, rgba(247,190,104,.14), transparent 24%),linear-gradient(180deg, rgba(16,26,45,.98), rgba(12,18,32,.98))",
                "rgba(247,190,104,.14)",
                "rgba(247,190,104,.22)",
                "#ffe3b2",
                "Executive Walkthrough",
            )
        return (
            "radial-gradient(circle at top left, rgba(56,211,159,.14), transparent 24%),linear-gradient(180deg, rgba(16,26,45,.98), rgba(12,18,32,.98))",
            "rgba(56,211,159,.12)",
            "rgba(56,211,159,.20)",
            "#d3ffe8",
            "Investor Intake",
        )

    def _render_thank_you(kind: str, copy_text: str, details: str) -> str:
        bg, badge_bg, badge_border, badge_text, type_label = _thank_you_theme(kind)
        html_out = THANK_YOU_PAGE.replace("__BG__", bg)
        html_out = html_out.replace("__BADGE_BG__", badge_bg)
        html_out = html_out.replace("__BADGE_BORDER__", badge_border)
        html_out = html_out.replace("__BADGE_TEXT__", badge_text)
        html_out = html_out.replace("__TYPE_LABEL__", type_label)
        html_out = html_out.replace("__THANK_YOU_COPY__", copy_text)
        html_out = html_out.replace("__DETAILS__", details)
        return html_out

    def _table_html(rows: List[Dict[str, Any]], columns: List[str], labels: Dict[str, str], route_prefix: str) -> str:
        if not rows:
            return "<div class='table-wrap'><div class='empty'>No submissions yet.</div></div>"
        head = "".join(f"<th>{_escape(labels.get(c, c.replace('_', ' ').title()))}</th>" for c in columns + ["actions"])
        body_rows = []
        for row in rows:
            tds = []
            for c in columns:
                value = row.get(c, "")
                if c == "status":
                    tds.append(f"<td><span class='{_status_class(str(value))}'>{_escape(_status_norm(str(value)))}</span></td>")
                else:
                    tds.append(f"<td>{_escape(value)}</td>")
            submitted_at = _escape(row.get("submitted_at", ""))
            action_cell = (
                "<td>"
                "<div class='action-row'>"
                f"<a class='mini-btn' href='/admin/status/{route_prefix}?submitted_at={submitted_at}&status=New'>New</a>"
                f"<a class='mini-btn' href='/admin/status/{route_prefix}?submitted_at={submitted_at}&status=Contacted'>Contacted</a>"
                f"<a class='mini-btn' href='/admin/status/{route_prefix}?submitted_at={submitted_at}&status=Scheduled'>Scheduled</a>"
                "</div>"
                "</td>"
            )
            body_rows.append("<tr>" + "".join(tds) + action_cell + "</tr>")
        return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"

    def _load_thresholds() -> Dict[str, Any]:
        data = _read_json(threshold_file, DEFAULT_THRESHOLDS)
        if not isinstance(data, dict):
            data = DEFAULT_THRESHOLDS.copy()
        for k, v in DEFAULT_THRESHOLDS.items():
            data.setdefault(k, v)
        return data

    def _save_thresholds(data: Dict[str, Any]) -> None:
        _write_json(threshold_file, data)

    def _load_workflow() -> Dict[str, Any]:
        data = _read_json(workflow_file, {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("records", {})
        data.setdefault("audit_log", [])
        return data

    def _save_workflow(data: Dict[str, Any]) -> None:
        _write_json(workflow_file, data)

    def _get_record(store: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        records = store.setdefault("records", {})
        if patient_id not in records:
            records[patient_id] = {
                "state": "new",
                "ack": False,
                "assigned": False,
                "escalated": False,
                "role": "",
                "updated_at": "",
            }
        return records[patient_id]

    def _audit(store: Dict[str, Any], patient_id: str, action: str, role: str, note: str = "") -> None:
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

    def _current_role() -> str:
        return str(session.get("user_role", "viewer")).strip().lower()

    def _current_user() -> str:
        return str(session.get("user_name", "Guest")).strip() or "Guest"

    def _has_permission(action: str) -> bool:
        role = _current_role()
        allowed = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["viewer"])
        return action in allowed

    def _login_required(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect("/login")
            return view_func(*args, **kwargs)
        return wrapped

    @app.before_request
    def force_single_domain():
        if not CANONICAL_HOST:
            return None
        host = request.host.split(":")[0].lower()
        if host and host != CANONICAL_HOST:
            qs = request.query_string.decode("utf-8")
            target = f"{request.scheme}://{CANONICAL_HOST}{request.path}"
            if qs:
                target += f"?{qs}"
            return redirect(target, code=302)
        return None

    @app.before_request
    def protect_pilot_routes():
        protected_prefixes = (
            "/command-center",
            "/admin/review",
            "/admin/export.csv",
            "/admin/status/",
            "/api/workflow",
            "/api/reporting",
            "/api/audit/export",
            "/api/thresholds",
            "/api/system-health",
            "/api/pilot-status",
            "/api/v1/live-snapshot",
            "/deck",
        )

        open_prefixes = (
            "/login",
            "/logout",
            "/terms",
            "/privacy",
            "/pilot-disclaimer",
            "/healthz",
            "/api/v1/health",
        )

        path = request.path or "/"

        if path.startswith(open_prefixes):
            return None

        if path.startswith(protected_prefixes) and not session.get("logged_in"):
            return redirect("/login")

        return None

    @app.route("/login", methods=["GET", "POST"])
    def pilot_login():
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
        <p>This platform environment is intended for demonstration, pilot review, and workflow evaluation. It is not represented as a production medical device or as a replacement for licensed clinical judgment.</p>
        <h2>No Clinical Replacement</h2>
        <p>Users must not rely on this pilot environment as a sole basis for diagnosis, treatment, emergency escalation, or formal clinical decision-making.</p>
        <h2>Organizational Review</h2>
        <p>Any hospital, clinic, insurer, or reviewer should evaluate the platform in accordance with their own internal clinical, legal, compliance, procurement, and technical standards.</p>
        """
        return render_template_string(LEGAL_PAGE_HTML.replace("__TITLE__", "Terms of Use").replace("__CONTENT__", content))

    @app.get("/privacy")
    def privacy_page():
        content = """
        <h2>Privacy Notice</h2>
        <p>This pilot environment may display simulated data, demonstration workflows, evaluation alerts, and user-entered pilot activity records.</p>
        <h2>Non-Production Handling</h2>
        <p>Production patient data should only be used in appropriately secured and compliant deployment environments with proper access control, governance, and security review.</p>
        <h2>Pilot Records</h2>
        <p>Workflow actions, audit events, and pilot configuration changes may be stored for evaluation, reporting, and system testing purposes.</p>
        """
        return render_template_string(LEGAL_PAGE_HTML.replace("__TITLE__", "Privacy Notice").replace("__CONTENT__", content))

    @app.get("/pilot-disclaimer")
    def pilot_disclaimer_page():
        content = """
        <h2>Pilot Environment</h2>
        <p>This environment is provided for workflow demonstration, pilot evaluation, user testing, and stakeholder review.</p>
        <h2>Not for Emergency Response</h2>
        <p>It must not be used as a substitute for established clinical escalation protocols, bedside assessment, or emergency response systems.</p>
        <h2>Professional Review Required</h2>
        <p>All outputs, alerts, recommendations, and risk signals should be interpreted by qualified professionals within the context of clinical judgment and local policy.</p>
        """
        return render_template_string(LEGAL_PAGE_HTML.replace("__TITLE__", "Pilot Disclaimer").replace("__CONTENT__", content))

    @app.get("/")
    def home():
        if session.get("logged_in"):
            return redirect("/command-center")
        return redirect("/login")

    @app.get("/command-center")
    @_login_required
    def command_center():
        return render_template_string(COMMAND_CENTER_HTML)

    @app.get("/dashboard")
    @_login_required
    def dashboard():
        return redirect("/command-center")

    @app.route("/hospital-demo", methods=["GET", "POST"])
    @_login_required
    def hospital_demo():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
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
            _save_jsonl(hospital_file, payload)
            return render_template_string(
                _render_thank_you(
                    "hospital",
                    "Your hospital demo request was submitted successfully. The request is now visible in admin review and ready for follow-up.",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "facility_type", "timeline"]),
                )
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
        <div class="field full"><label>What would you like to see in the demo?</label><textarea name="message"></textarea></div>
        """
        html_out = FORM_PAGE.replace("__TITLE__", "Hospital Demo — Early Risk Alert AI")
        html_out = html_out.replace("__HEADING__", "Request Hospital Demo")
        html_out = html_out.replace("__COPY__", "Capture interest from hospital operations teams, clinical leaders, and remote monitoring stakeholders.")
        html_out = html_out.replace("__FIELDS__", fields)
        html_out = html_out.replace("__BUTTON__", "Submit Hospital Demo Request")
        return render_template_string(html_out)

    @app.route("/executive-walkthrough", methods=["GET", "POST"])
    @_login_required
    def executive_walkthrough():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
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
            _save_jsonl(exec_file, payload)
            return render_template_string(
                _render_thank_you(
                    "executive",
                    "Your executive walkthrough request was submitted successfully. The request is now visible in admin review and ready for scheduling follow-up.",
                    _detail_html(payload, ["full_name", "organization", "title", "email", "priority", "timeline"]),
                )
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
        <div class="field full"><label>Walkthrough Focus</label><textarea name="message"></textarea></div>
        """
        html_out = FORM_PAGE.replace("__TITLE__", "Executive Walkthrough — Early Risk Alert AI")
        html_out = html_out.replace("__HEADING__", "Schedule Executive Walkthrough")
        html_out = html_out.replace("__COPY__", "Capture executive-level product review requests for hospital leadership, system operations, and strategic evaluation.")
        html_out = html_out.replace("__FIELDS__", fields)
        html_out = html_out.replace("__BUTTON__", "Submit Executive Walkthrough Request")
        return render_template_string(html_out)

    @app.route("/investor-intake", methods=["GET", "POST"])
    @_login_required
    def investor_intake():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
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
            _save_jsonl(investor_file, payload)
            return render_template_string(
                _render_thank_you(
                    "investor",
                    "Your investor intake was submitted successfully. The request is now visible in admin review and ready for follow-up and export.",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "investor_type", "timeline"]),
                )
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
        <div class="field full"><label>Interest / Notes</label><textarea name="message" placeholder="What are you interested in learning more about?"></textarea></div>
        """
        html_out = FORM_PAGE.replace("__TITLE__", "Investor Intake — Early Risk Alert AI")
        html_out = html_out.replace("__HEADING__", "Investor Intake Form")
        html_out = html_out.replace("__COPY__", "Capture investor interest, timeline, and follow-up details directly from the platform.")
        html_out = html_out.replace("__FIELDS__", fields)
        html_out = html_out.replace("__BUTTON__", "Submit Investor Intake")
        return render_template_string(html_out)

    @app.get("/admin/status/hospital")
    @_login_required
    def admin_status_hospital():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(hospital_file, submitted_at, status)
        return redirect("/admin/review")

    @app.get("/admin/status/executive")
    @_login_required
    def admin_status_executive():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(exec_file, submitted_at, status)
        return redirect("/admin/review")

    @app.get("/admin/status/investor")
    @_login_required
    def admin_status_investor():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(investor_file, submitted_at, status)
        return redirect("/admin/review")

    @app.get("/admin/review")
    @_login_required
    def admin_review():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)

        labels_h = {
            "submitted_at": "Submitted",
            "status": "Status",
            "full_name": "Name",
            "organization": "Organization",
            "role": "Role",
            "email": "Email",
            "facility_type": "Facility Type",
            "timeline": "Timeline",
        }
        labels_e = {
            "submitted_at": "Submitted",
            "status": "Status",
            "full_name": "Name",
            "organization": "Organization",
            "title": "Executive Title",
            "email": "Email",
            "priority": "Priority",
            "timeline": "Timeline",
        }
        labels_i = {
            "submitted_at": "Submitted",
            "status": "Status",
            "full_name": "Name",
            "organization": "Organization",
            "role": "Role",
            "email": "Email",
            "investor_type": "Investor Type",
            "check_size": "Check Size",
            "timeline": "Timeline",
        }

        html_out = ADMIN_HTML
        html_out = html_out.replace("__HOSPITAL_COUNT__", str(len(hospital_rows)))
        html_out = html_out.replace("__EXEC_COUNT__", str(len(exec_rows)))
        html_out = html_out.replace("__INVESTOR_COUNT__", str(len(investor_rows)))
        html_out = html_out.replace("__HOSPITAL_TABLE__", _table_html(hospital_rows, ["submitted_at", "status", "full_name", "organization", "role", "email", "facility_type", "timeline"], labels_h, "hospital"))
        html_out = html_out.replace("__EXEC_TABLE__", _table_html(exec_rows, ["submitted_at", "status", "full_name", "organization", "title", "email", "priority", "timeline"], labels_e, "executive"))
        html_out = html_out.replace("__INVESTOR_TABLE__", _table_html(investor_rows, ["submitted_at", "status", "full_name", "organization", "role", "email", "investor_type", "check_size", "timeline"], labels_i, "investor"))
        return render_template_string(html_out)

    @app.get("/admin/export.csv")
    @_login_required
    def admin_export_csv():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Lead Source",
            "Submitted At",
            "Lead Status",
            "Full Name",
            "Organization",
            "Role Or Title",
            "Email Address",
            "Phone Number",
            "Lead Type Or Priority",
            "Timeline",
            "Notes",
        ])

        for row in hospital_rows:
            writer.writerow([
                "Hospital Demo",
                row.get("submitted_at", ""),
                _status_norm(row.get("status", "New")),
                row.get("full_name", ""),
                row.get("organization", ""),
                row.get("role", ""),
                row.get("email", ""),
                row.get("phone", ""),
                row.get("facility_type", ""),
                row.get("timeline", ""),
                row.get("message", ""),
            ])

        for row in exec_rows:
            writer.writerow([
                "Executive Walkthrough",
                row.get("submitted_at", ""),
                _status_norm(row.get("status", "New")),
                row.get("full_name", ""),
                row.get("organization", ""),
                row.get("title", ""),
                row.get("email", ""),
                row.get("phone", ""),
                row.get("priority", ""),
                row.get("timeline", ""),
                row.get("message", ""),
            ])

        for row in investor_rows:
            writer.writerow([
                "Investor Intake",
                row.get("submitted_at", ""),
                _status_norm(row.get("status", "New")),
                row.get("full_name", ""),
                row.get("organization", ""),
                row.get("role", ""),
                row.get("email", ""),
                row.get("phone", ""),
                row.get("investor_type", ""),
                row.get("timeline", ""),
                row.get("message", ""),
            ])

        mem = io.BytesIO()
        mem.write(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="early_risk_alert_pipeline_export.csv")

    @app.get("/deck")
    @_login_required
    def deck():
        pdf_bytes = _generate_pitch_deck_pdf_bytes()
        return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf")

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "service": "early-risk-alert-ai", "canonical_host": CANONICAL_HOST or None})

    @app.get("/api/v1/health")
    def api_health():
        return jsonify({"status": "ok", "service": "early-risk-alert-api", "canonical_host": CANONICAL_HOST or None})

    @app.get("/api/pilot-status")
    @_login_required
    def pilot_status():
        return jsonify({
            "pilot_mode": bool(session.get("pilot_mode", PILOT_MODE)),
            "logged_in": bool(session.get("logged_in")),
            "user_name": _current_user(),
            "user_role": _current_role(),
            "permissions": sorted(list(ROLE_PERMISSIONS.get(_current_role(), {"read"}))),
        })

    # -----------------------------
    # PILOT MODE STATE
    # -----------------------------
    _PILOT_STATE = {"enabled": True}

    @app.get("/api/pilot-mode")
    def get_pilot_mode():
        return jsonify(_PILOT_STATE)

    @app.post("/api/pilot-mode")
    def set_pilot_mode():
        data = request.get_json(silent=True) or {}
        enabled = bool(data.get("enabled", True))
        _PILOT_STATE["enabled"] = enabled
        return jsonify(_PILOT_STATE)

    @app.get("/api/workflow")
    @_login_required
    def get_workflow():
        store = _load_workflow()
        return jsonify({
            "records": store["records"],
            "audit_log": store["audit_log"],
        })

    @app.post("/api/workflow/action")
    @_login_required
    def workflow_action():
        payload = request.get_json() or {}

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
            if not _has_permission("resolve"):
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
        records = store["records"].values()

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
        if not _has_permission("admin"):
            return jsonify({"ok": False, "error": "permission denied"}), 403
        data = request.get_json() or {}
        _save_thresholds(data)
        return jsonify({"ok": True})

    @app.get("/api/system-health")
    @_login_required
    def system_health():
        return jsonify({
            "status": "operational",
            "pilot_mode": PILOT_MODE,
            "timestamp": _utc_now_iso(),
            "streams": "active",
            "workflow_storage": workflow_file.exists(),
            "thresholds_loaded": True,
        })

    @app.get("/api/v1/live-snapshot")
    @_login_required
    def live_snapshot():
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")

        patients = _build_demo_patients()
        alerts = _build_alerts(patients)

        focus = next(
            (p for p in patients if str(p.get("patient_id")) == str(patient_id)),
            patients[0] if patients else None,
        )
        if focus:
            focus["rollups"] = _build_rollups(focus)

        risk_values: List[float] = []
        critical_alerts = 0

        for p in patients:
            risk = p.get("risk", {}) or {}
            risk_score = risk.get("risk_score", 0)
            severity = str(risk.get("severity", "")).lower()

            try:
                risk_values.append(float(risk_score))
            except Exception:
                pass

            if severity == "critical":
                critical_alerts += 1

        summary = {
            "patient_count": len(patients),
            "open_alerts": len(alerts),
            "critical_alerts": critical_alerts,
            "avg_risk_score": round(sum(risk_values) / len(risk_values), 1) if risk_values else 0.0,
            "focus_patient_id": focus.get("patient_id") if focus else None,
        }

        return jsonify({
            "tenant_id": tenant_id,
            "generated_at": _utc_now_iso(),
            "alerts": alerts,
            "focus_patient": focus,
            "patients": patients,
            "summary": summary,
        })

    return app
