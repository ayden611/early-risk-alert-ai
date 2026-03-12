from __future__ import annotations
import csv
import html
import io
import json
import os
import random
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, redirect, render_template_string, request, send_file
from flask import Response



INFO_EMAIL = "info@earlyriskalertai.com"
FOUNDER_EMAIL = "MiltonMunroe@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder, Early Risk Alert AI"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"
CANONICAL_HOST = os.getenv("CANONICAL_HOST", "").strip().lower()


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

MAIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --bg2:#0b1424;
      --panel:#101a2d;
      --panel2:#13203a;
      --line:rgba(255,255,255,.08);
      --line2:rgba(255,255,255,.05);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --violet:#a68cff;
      --shadow:0 20px 60px rgba(0,0,0,.30);
      --radius:24px;
      --max:1360px;
    }

    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.16), transparent 22%),
        radial-gradient(circle at 85% 8%, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg,var(--bg),var(--bg2));
      overflow-x:hidden;
    }
    a{text-decoration:none;color:inherit}
    img{max-width:100%;display:block}

    .nav{
      position:sticky;top:0;z-index:100;
      background:rgba(7,16,28,.82);
      backdrop-filter:blur(16px);
      border-bottom:1px solid var(--line);
    }
    .nav-inner{
      max-width:var(--max);margin:0 auto;padding:14px 16px;
      display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#c8d7ef;
    }
    .brand-title{
      font-size:clamp(26px,3vw,40px);font-weight:1000;line-height:.94;letter-spacing:-.045em;
    }
    .brand-sub{
      font-size:14px;color:var(--muted);font-weight:800;
    }
    .nav-links{
      display:flex;align-items:center;gap:16px;flex-wrap:wrap;
    }
    .nav-links a{
      font-size:14px;font-weight:900;
    }

    .btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:13px 18px;border-radius:16px;
      font-size:14px;font-weight:900;cursor:pointer;
      border:1px solid transparent;
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, opacity .18s ease;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#07101c;
      box-shadow:0 12px 30px rgba(91,212,255,.18);
    }
    .btn.secondary{
      background:#111b2f;color:var(--text);border-color:var(--line);
    }
    .btn.ghost{
      background:transparent;color:var(--text);border-color:var(--line);
    }

    .shell{max-width:var(--max);margin:0 auto;padding:18px 16px 70px}

    .hero{
      position:relative;
      min-height:88vh;
      border-radius:30px;
      overflow:hidden;
      border:1px solid rgba(255,255,255,.08);
      box-shadow:var(--shadow);
      background:
        linear-gradient(180deg, rgba(5,12,22,.24), rgba(5,12,22,.70)),
        url('/static/images/ai-command-center.jpg') center center / cover no-repeat;
      isolation:isolate;
    }

    .hero::before{
      content:"";
      position:absolute;inset:0;
      background:
        radial-gradient(circle at center, rgba(91,212,255,.10), transparent 26%),
        linear-gradient(rgba(122,162,255,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(122,162,255,.045) 1px, transparent 1px);
      background-size:auto, 34px 34px, 34px 34px;
      opacity:.36;
      pointer-events:none;
    }

    .hero::after{
      content:"";
      position:absolute;left:-10%;right:-10%;bottom:-8%;
      height:34%;
      background:radial-gradient(circle at 50% 50%, rgba(91,212,255,.20), transparent 60%);
      filter:blur(22px);
      animation:heroGlow 8s ease-in-out infinite;
      pointer-events:none;
    }

    @keyframes heroGlow{
      0%,100%{transform:translateX(0) scale(1)}
      50%{transform:translateX(2%) scale(1.04)}
    }

    .sweep{
      position:absolute;inset:-22%;
      background:linear-gradient(105deg, transparent 38%, rgba(255,255,255,.06) 50%, transparent 62%);
      transform:translateX(-60%);
      animation:sweep 8s linear infinite;
      pointer-events:none;z-index:1;
    }
    @keyframes sweep{
      0%{transform:translateX(-60%)}
      100%{transform:translateX(60%)}
    }

    .hero-inner{
      position:relative;z-index:3;
      min-height:88vh;
      display:grid;
      align-content:space-between;
      gap:26px;
      padding:28px;
    }

    .top-badges{
      display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;
    }
    .badge-top{
      display:inline-flex;align-items:center;
      padding:10px 14px;border-radius:999px;
      background:rgba(7,16,28,.56);
      border:1px solid rgba(255,255,255,.10);
      backdrop-filter:blur(10px);
      font-size:12px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;
    }

    .hero-grid{
      display:grid;grid-template-columns:1.15fr .85fr;gap:18px;align-items:end;
    }

    .glass{
      background:linear-gradient(180deg, rgba(8,16,30,.58), rgba(8,16,30,.78));
      border:1px solid rgba(255,255,255,.09);
      border-radius:26px;
      padding:26px;
      backdrop-filter:blur(12px);
      box-shadow:0 16px 40px rgba(0,0,0,.20);
    }

    .hero-kicker{
      font-size:12px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#d2dff4;margin-bottom:12px;
    }
    h1{
      margin:0 0 16px;
      font-size:clamp(44px,6vw,84px);
      line-height:.92;
      font-weight:1000;
      letter-spacing:-.06em;
      text-shadow:0 0 22px rgba(0,0,0,.18);
      max-width:980px;
    }
    .lead{
      margin:0;
      color:#d7e4f8;
      font-size:clamp(16px,1.5vw,20px);
      line-height:1.62;
      max-width:920px;
    }

    .hero-actions{
      display:flex;gap:12px;flex-wrap:wrap;margin-top:22px;
    }

    .route-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px;
    }
    .route-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.018));
      border:1px solid rgba(255,255,255,.08);
      border-radius:20px;
      padding:18px;
      min-height:170px;
      transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .route-card:hover{
      transform:translateY(-3px);
      border-color:rgba(91,212,255,.24);
      box-shadow:0 12px 30px rgba(0,0,0,.16);
    }
    .route-label{
      font-size:11px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:#9fb7da;
    }
    .route-title{
      font-size:26px;font-weight:1000;line-height:1;margin-top:10px;
    }
    .route-copy{
      font-size:14px;line-height:1.56;color:#c7d7ef;margin-top:10px;
    }
    .route-arrow{
      margin-top:14px;font-size:14px;font-weight:900;color:#99e2ff;
    }

    .side-title{
      margin:0 0 10px;
      font-size:clamp(28px,3vw,42px);
      line-height:1;
      font-weight:1000;
      letter-spacing:-.04em;
    }
    .side-copy{
      margin:0;color:#c7d7ef;line-height:1.65;
    }
    .mini-grid{
      display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px;
    }
    .mini{
      background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      padding:16px;
    }
    .mini-k{
      font-size:11px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:#9fb7da;
    }
    .mini-v{
      font-size:34px;font-weight:1000;line-height:1;margin-top:10px;
    }
    .mini-s{
      font-size:13px;color:#c7d7ef;line-height:1.5;margin-top:8px;
    }

    .ticker-wrap{
      margin-top:18px;
      overflow:hidden;
      border-top:1px solid rgba(255,255,255,.08);
      border-bottom:1px solid rgba(255,255,255,.08);
    }
    .ticker{
      display:flex;gap:28px;white-space:nowrap;padding:12px 0;
      color:#d9e9ff;font-weight:800;
      animation:tickerMove 18s linear infinite;
    }
    .ticker span{display:inline-flex;align-items:center;gap:8px}
    .dot{width:8px;height:8px;border-radius:999px;background:#7aa2ff}
    @keyframes tickerMove{
      0%{transform:translateX(0)}
      100%{transform:translateX(-50%)}
    }

    .section{margin-top:34px}
    .section-title{
      margin:0 0 8px;
      font-size:clamp(30px,3vw,44px);
      line-height:1.04;
      font-weight:1000;
      letter-spacing:-.04em;
    }
    .section-sub{
      margin:0 0 18px;
      max-width:980px;
      font-size:16px;
      line-height:1.6;
      color:var(--muted);
    }

    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.016));
      border:1px solid var(--line);
      border-radius:22px;
      padding:22px;
      box-shadow:var(--shadow);
    }

    .metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
    .metric-label,.panel-title,.contact-label{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#9fb7da;
    }
    .metric-value{
      font-size:clamp(28px,2.6vw,54px);
      line-height:1;
      font-weight:1000;
      margin-top:10px;
    }
    .metric-note{
      margin-top:8px;font-size:14px;line-height:1.55;color:#bdd0eb;
    }

    .dashboard-grid{
      display:grid;grid-template-columns:1.18fr 1fr 1fr;gap:14px;
    }
    .dash-card{min-height:520px}
    .dash-card h3{margin:0 0 6px;font-size:22px}
    .dash-card p{margin:0 0 16px;color:var(--muted);line-height:1.6}
    .feed{display:flex;flex-direction:column;gap:12px}

    .alert{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.07);
      border-radius:20px;
      padding:16px;
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .alert:hover{transform:translateY(-2px)}
    .alert-critical{
      box-shadow:0 0 0 1px rgba(255,102,125,.28), 0 0 30px rgba(255,102,125,.18);
      border-color:rgba(255,102,125,.34);
      background:linear-gradient(180deg, rgba(255,102,125,.10), rgba(255,255,255,.02));
    }
    .alert-high{
      box-shadow:0 0 0 1px rgba(244,189,106,.18), 0 0 24px rgba(244,189,106,.10);
      border-color:rgba(244,189,106,.24);
      background:linear-gradient(180deg, rgba(244,189,106,.07), rgba(255,255,255,.02));
    }
    .alert-moderate{
      border-color:rgba(122,162,255,.18);
      background:linear-gradient(180deg, rgba(122,162,255,.06), rgba(255,255,255,.02));
    }

    .alert-top{display:flex;justify-content:space-between;gap:12px}
    .alert-name{font-size:18px;font-weight:900}
    .alert-patient{font-size:13px;color:#bdd0eb;font-weight:700}
    .alert-msg{font-size:18px;font-weight:800;line-height:1.35;margin-top:6px}
    .alert-time{font-size:13px;color:#9fb4d6;margin-top:6px}

    .meta-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .meta-pill{
      padding:7px 10px;border-radius:999px;
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.06);
      font-size:12px;font-weight:800;color:#dbe7fb;
    }

    .severity{
      display:inline-flex;align-items:center;justify-content:center;
      min-width:122px;padding:11px 16px;border-radius:999px;
      font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:1000;
    }
    .sev-critical{background:rgba(255,102,125,.20);border:1px solid rgba(255,102,125,.38);color:#ffd7de}
    .sev-high{background:rgba(244,189,106,.20);border:1px solid rgba(244,189,106,.38);color:#ffe1ad}
    .sev-moderate{background:rgba(122,162,255,.20);border:1px solid rgba(122,162,255,.38);color:#dce8ff}
    .sev-stable{background:rgba(56,211,159,.18);border:1px solid rgba(56,211,159,.32);color:#cdfbe8}

    .panel-block{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.018));
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      padding:16px;
      margin-top:12px;
    }

    .focus-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}
    .focus-id{font-size:34px;font-weight:1000;line-height:1}
    .kv{
      display:flex;justify-content:space-between;gap:12px;
      padding:10px 0;border-bottom:1px solid rgba(255,255,255,.05);
    }
    .kv:last-child{border-bottom:none}
    .k{color:#bdd0eb;font-weight:700}
    .v{font-weight:900}
    .trend-up{color:#ff98a5}
    .trend-mid{color:#ffd38f}
    .trend-down{color:#98f1c8}

    .confidence-wrap{margin-top:10px;display:grid;gap:7px}
    .confidence-label{
      display:flex;justify-content:space-between;gap:10px;
      font-size:12px;font-weight:900;color:#d3e2f9;text-transform:uppercase;letter-spacing:.08em;
    }
    .confidence-track{
      width:100%;height:10px;border-radius:999px;overflow:hidden;
      background:rgba(255,255,255,.07);
      border:1px solid rgba(255,255,255,.05);
    }
    .confidence-bar{
      height:100%;
      border-radius:999px;
      background:linear-gradient(90deg,var(--blue),var(--blue2));
      box-shadow:0 0 18px rgba(91,212,255,.22);
    }

    .sim-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
    .sim-title{font-size:16px;font-weight:900}
    .sim-room{font-size:12px;color:#a8bddf;margin-top:4px}
    .sim-risk{font-size:30px;font-weight:1000;margin:12px 0 6px}
    .sim-copy{font-size:14px;color:#c9d8ef}

    .trust-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
    .contact-box{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}
    .contact-row{
      background:linear-gradient(180deg, rgba(255,255,255,.032), rgba(255,255,255,.022));
      border:1px solid rgba(122,162,255,.12);
      border-radius:18px;
      padding:16px 18px;
    }
    .contact-value{
      font-size:18px;font-weight:900;margin-top:6px;color:#f1f7ff;
    }

    .footer{
      margin-top:30px;
      padding:24px 12px 8px;
      color:#96a8c6;
      font-size:14px;
      line-height:1.6;
      text-align:center;
    }

    .audio-toggle{
      display:inline-flex;align-items:center;gap:8px;
      padding:10px 14px;border-radius:14px;
      background:#111b2f;border:1px solid var(--line);
      color:var(--text);font-weight:900;cursor:pointer;
    }

    @media (max-width:1180px){
      .hero-grid{grid-template-columns:1fr}
      .route-grid,.metrics-grid,.sim-grid,.trust-grid,.contact-box,.mini-grid{grid-template-columns:repeat(2,1fr)}
      .dashboard-grid{grid-template-columns:1fr 1fr}
      .dashboard-grid .dash-card:first-child{grid-column:1/-1}
    }

    @media (max-width:760px){
      .route-grid,.metrics-grid,.dashboard-grid,.sim-grid,.trust-grid,.contact-box,.mini-grid{grid-template-columns:1fr}
      .shell{padding:14px 10px 54px}
      .hero-inner{padding:16px;min-height:auto}
      .glass,.card{padding:16px}
      .hero-actions{flex-direction:column;align-items:stretch}
      .btn,.audio-toggle{width:100%}
      .alert-top,.focus-top,.kv{flex-direction:column}
      h1{font-size:clamp(34px,9vw,48px)}
      .lead{font-size:15px}
    }
  </style>
</head>
<body>
  <div class="nav">
    <div class="nav-inner">
      <div>
        <div class="brand-kicker">AI-powered predictive clinical intelligence</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Hospitals · Clinics · Investors · Patients</div>
      </div>
      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#dashboard">Clinical Command Center</a>
        <a href="#simulator">Simulator</a>
        <a href="/investors">Investor View</a>
        <a href="/hospital-demo">Hospital Demo</a>
        <a class="btn primary" href="/deck">Download Pitch Deck</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview" style="position:relative;overflow:hidden;">
      <div class="sweep"></div>
      <div class="hero-inner">
        <div class="top-badges">
          <div class="badge-top">Single production domain experience</div>
          <div class="badge-top">Cinematic intro section with live platform paths</div>
        </div>
        

        <div class="hero-grid">
          <div class="glass">
            <div class="hero-kicker">Predictive clinical intelligence</div>
            <h1>Step into the platform the way your users will.</h1>
            <p class="lead">
              This cinematic entry experience guides hospital teams into demo capture, investors into the commercial portal,
              and product viewers into the live command-center walkthrough — all from one branded production homepage.
            </p>

            <div class="hero-actions">
              <a class="btn primary" href="/hospital-demo">Hospital Path</a>
              <a class="btn secondary" href="#dashboard">Live Demo Path</a>
              <a class="btn secondary" href="/investors">Investor Path</a>
              <button class="audio-toggle" id="audioToggle" type="button">🔔 Enable Alert Sound</button>
            </div>

            <div class="route-grid">
              <a class="route-card" href="/hospital-demo">
                <div class="route-label">Left path</div>
                <div class="route-title">Hospital Demo</div>
                <div class="route-copy">
                  Request a hospital demo, capture operational interest, and guide health-system stakeholders into evaluation.
                </div>
                <div class="route-arrow">Open hospital intake →</div>
              </a>

              <a class="route-card" href="#dashboard">
                <div class="route-label">Center path</div>
                <div class="route-title">Live Product Demo</div>
                <div class="route-copy">
                  Move directly into the clinical command center, AI risk scoring, confidence indicators, and recommended actions.
                </div>
                <div class="route-arrow">Jump to live command center →</div>
              </a>

              <a class="route-card" href="/investors">
                <div class="route-label">Right path</div>
                <div class="route-title">Investor Portal</div>
                <div class="route-copy">
                  Open the investor overview, download the pitch deck, and move into the investor intake workflow.
                </div>
                <div class="route-arrow">Open investor overview →</div>
              </a>
            </div>

            <div class="ticker-wrap">
              <div class="ticker" id="opsTicker"></div>
            </div>
          </div>

          <div class="glass">
            <div class="hero-kicker">Executive presentation layer</div>
            <div class="side-title">One homepage. Three clear journeys. No blank routes.</div>
            <p class="side-copy">
              This is a cinematic hero experience built from your command-center image. It is not a literal AI video,
              but it creates a premium production entry point while keeping every route pointed to the live platform.
            </p>

            <div class="mini-grid">
              <div class="mini">
                <div class="mini-k">Hospital</div>
                <div class="mini-v">Demo</div>
                <div class="mini-s">Request operational walkthroughs and demo interest directly from the homepage.</div>
              </div>
              <div class="mini">
                <div class="mini-k">Command Center</div>
                <div class="mini-v">Live</div>
                <div class="mini-s">Show severity, AI score, confidence, trend direction, and recommended action.</div>
              </div>
              <div class="mini">
                <div class="mini-k">Investor</div>
                <div class="mini-v">Portal</div>
                <div class="mini-s">Present credibility, product positioning, and branded pitch materials cleanly.</div>
              </div>
              <div class="mini">
                <div class="mini-k">Production</div>
                <div class="mini-v">Locked</div>
                <div class="mini-s">One public homepage flow with cleaner links and stronger route behavior.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Animated live metrics</h2>
      <p class="section-sub">Operational KPIs update automatically with stronger dashboard presentation.</p>
      <div class="metrics-grid">
        <div class="card">
          <div class="metric-label">Monitored Patients</div>
          <div class="metric-value" id="metricPatients">0</div>
          <div class="metric-note">Active patient population</div>
        </div>
        <div class="card">
          <div class="metric-label">Open Alerts</div>
          <div class="metric-value" id="metricAlerts">0</div>
          <div class="metric-note">Current command-center queue</div>
        </div>
        <div class="card">
          <div class="metric-label">Critical Alerts</div>
          <div class="metric-value" id="metricCritical">0</div>
          <div class="metric-note">Highest-priority interventions</div>
        </div>
        <div class="card">
          <div class="metric-label">Avg AI Risk Score</div>
          <div class="metric-value" id="metricRisk">0</div>
          <div class="metric-note">Enterprise scoring snapshot</div>
        </div>
      </div>
    </section>

    <section class="section" id="dashboard">
      <h2 class="section-title">Clinical command center</h2>
      <p class="section-sub">Larger severity badges, confidence indicators, AI recommendation panels, clinical priority, and trend direction.</p>

      <div class="dashboard-grid">
        <div class="card dash-card">
          <h3>Live Alerts Feed</h3>
          <p>Clinical alert prioritization with stronger severity emphasis and critical glow.</p>
          <div class="feed" id="alertsFeed"></div>
        </div>

        <div class="card dash-card">
          <h3>Patient Focus</h3>
          <p>AI score, confidence, recommendation, priority, and trend direction in one clinical view.</p>
          <div id="patientFocus"></div>
        </div>

        <div class="card dash-card">
          <h3>System Panels</h3>
          <p>Operational state, workflow readiness, stream channels, and live system activity.</p>
          <div class="panel-block" id="streamHealth"></div>
          <div class="panel-block">
            <div class="panel-title">Live System Activity</div>
            <div id="systemActivity"></div>
          </div>
          <div class="panel-block" id="streamChannels"></div>
        </div>
      </div>
    </section>

    <section class="section" id="simulator">
      <h2 class="section-title">Live patient simulator</h2>
      <p class="section-sub">Simulated patient activity with stronger severity colors and AI scoring clarity.</p>
      <div class="sim-grid" id="simulatorGrid"></div>
    </section>

    <section class="section">
      <h2 class="section-title">Business credibility layer</h2>
      <p class="section-sub">Professional positioning for hospitals, operators, and investors evaluating deployment readiness.</p>

      <div class="trust-grid">
        <div class="card"><h3>Hospital operations focus</h3><p>Supports command-center visibility, prioritization workflows, and faster intervention escalation.</p></div>
        <div class="card"><h3>Executive review flow</h3><p>Structured hospital demo and walkthrough capture for leadership and operational stakeholders.</p></div>
        <div class="card"><h3>Investor-ready presentation</h3><p>Unified platform, investor intake workflow, downloadable deck, and admin review layer.</p></div>
        <div class="card"><h3>Founder credibility</h3><p>Built and led by Milton Munroe to advance predictive healthcare intelligence for modern care operations.</p></div>
      </div>

      <div class="card" style="margin-top:14px;">
        <h3 style="margin-top:0;">Founder & Contact</h3>
        <div class="contact-box">
          <div class="contact-row">
            <div class="contact-label">Founder</div>
            <div class="contact-value">Milton Munroe</div>
          </div>
          <div class="contact-row">
            <div class="contact-label">Role</div>
            <div class="contact-value">Founder, Early Risk Alert AI</div>
          </div>
          <div class="contact-row">
            <div class="contact-label">Email</div>
            <div class="contact-value">info@earlyriskalertai.com</div>
          </div>
          <div class="contact-row">
            <div class="contact-label">Business Phone</div>
            <div class="contact-value">732-724-7267</div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC • Professional healthcare intelligence platform • info@earlyriskalertai.com • 732-724-7267
    </div>
  </div>

  <script>
    const tenantId = "demo";
    const focusPatientId = "p101";
    let soundEnabled = false;
    let prevOpenAlerts = 0;
    let prevCriticalAlerts = 0;

    document.getElementById("audioToggle").addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      document.getElementById("audioToggle").textContent = soundEnabled ? "🔔 Alert Sound On" : "🔔 Enable Alert Sound";
    });

    async function getJson(url) {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    }

    function severityClass(sev) {
      if (sev === "critical") return "severity sev-critical";
      if (sev === "high") return "severity sev-high";
      if (sev === "moderate") return "severity sev-moderate";
      return "severity sev-stable";
    }

    function alertClass(sev) {
      if (sev === "critical") return "alert alert-critical";
      if (sev === "high") return "alert alert-high";
      if (sev === "moderate") return "alert alert-moderate";
      return "alert";
    }

    function trendClass(trend) {
      const t = (trend || "").toLowerCase();
      if (t === "worsening") return "trend-up";
      if (t === "elevated") return "trend-mid";
      return "trend-down";
    }

    function animateNumber(id, endValue) {
      const el = document.getElementById(id);
      if (!el) return;
      const startValue = parseInt(el.textContent || "0", 10) || 0;
      const duration = 700;
      const start = performance.now();
      function frame(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(startValue + ((endValue - startValue) * eased));
        el.textContent = current;
        if (progress < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    }

    function playAlertBeep() {
      if (!soundEnabled) return;
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.value = 0.001;
      osc.connect(gain);
      gain.connect(ctx.destination);
      const now = ctx.currentTime;
      gain.gain.exponentialRampToValueAtTime(0.08, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
      osc.start(now);
      osc.stop(now + 0.23);
    }

    function renderAlerts(alerts) {
      const feed = document.getElementById("alertsFeed");
      feed.innerHTML = "";
      alerts.forEach((a) => {
        const div = document.createElement("div");
        div.className = alertClass(a.severity);
        div.innerHTML = `
          <div class="alert-top">
            <div>
              <div class="alert-name">${a.alert_type || "alert"}</div>
              <div class="alert-patient">${a.patient_name} • ${a.room}</div>
            </div>
            <div class="${severityClass(a.severity)}">${a.severity}</div>
          </div>
          <div class="alert-msg">${a.message || "Clinical event detected"}</div>
          <div class="alert-time">Program: ${a.program} • live</div>
          <div class="meta-row">
            <div class="meta-pill">AI score ${a.risk_score}</div>
            <div class="meta-pill">Confidence ${a.confidence}%</div>
            <div class="meta-pill">${a.clinical_priority}</div>
            <div class="meta-pill"><span class="${trendClass(a.trend_direction)}">${a.trend_arrow || ""}</span>&nbsp;${a.trend_direction}</div>
          </div>
          <div class="confidence-wrap">
            <div class="confidence-label"><span>Confidence Indicator</span><span>${a.confidence}%</span></div>
            <div class="confidence-track"><div class="confidence-bar" style="width:${a.confidence}%;"></div></div>
          </div>
          <div class="panel-block">
            <div class="panel-title">Recommended Action</div>
            <div style="color:#dce8fb;line-height:1.6;">${a.recommended_action}</div>
          </div>
        `;
        feed.appendChild(div);
      });
    }

    function renderFocus(focus) {
      const el = document.getElementById("patientFocus");
      const v = focus.vitals || {};
      const r = focus.risk || {};
      const roll = focus.rollups || {};
      el.innerHTML = `
        <div class="panel-block">
          <div class="focus-top">
            <div>
              <div class="alert-patient">${focus.patient_name || "Patient"} • ${focus.room || ""}</div>
              <div class="focus-id">${focus.patient_id || ""}</div>
              <div style="font-size:18px;font-weight:800;color:#dbe7fb;margin-top:8px;">${r.alert_message || "Vitals stable"}</div>
            </div>
            <div class="${severityClass(r.severity)}">${r.severity || "stable"}</div>
          </div>
        </div>

        <div class="panel-block">
          <div class="kv"><span class="k">AI Risk Score</span><span class="v">${r.risk_score ?? "--"}</span></div>
          <div class="kv"><span class="k">Clinical Priority</span><span class="v">${r.clinical_priority ?? "--"}</span></div>
          <div class="kv"><span class="k">Trend Direction</span><span class="v ${trendClass(r.trend_direction)}">${r.trend_arrow || ""} ${r.trend_direction ?? "--"}</span></div>
        </div>

        <div class="panel-block">
          <div class="panel-title">Confidence Indicator</div>
          <div class="confidence-wrap">
            <div class="confidence-label"><span>Confidence</span><span>${r.confidence ?? "--"}%</span></div>
            <div class="confidence-track"><div class="confidence-bar" style="width:${r.confidence || 0}%;"></div></div>
          </div>
        </div>

        <div class="panel-block">
          <div class="kv"><span class="k">Heart Rate</span><span class="v">${v.heart_rate ?? "--"}</span></div>
          <div class="kv"><span class="k">Systolic BP</span><span class="v">${v.systolic_bp ?? "--"}</span></div>
          <div class="kv"><span class="k">Diastolic BP</span><span class="v">${v.diastolic_bp ?? "--"}</span></div>
          <div class="kv"><span class="k">SpO2</span><span class="v">${v.spo2 ?? "--"}</span></div>
          <div class="kv"><span class="k">Avg SpO2</span><span class="v">${roll.avg_spo2 ?? "--"}</span></div>
        </div>

        <div class="panel-block">
          <div class="panel-title">Recommended Action</div>
          <div style="color:#dce8fb;line-height:1.65;">${r.recommended_action ?? "--"}</div>
        </div>
      `;
    }

    function renderStreamHealth(data) {
      document.getElementById("streamHealth").innerHTML = `
        <div class="panel-title">System Health</div>
        <div class="kv"><span class="k">Status</span><span class="v">${data.status || "running"}</span></div>
        <div class="kv"><span class="k">Mode</span><span class="v">${data.mode || "realtime"}</span></div>
        <div class="kv"><span class="k">Worker</span><span class="v">${data.worker_status || "simulated"}</span></div>
        <div class="kv"><span class="k">Patients with alerts</span><span class="v">${data.patients_with_alerts ?? 0}</span></div>
      `;
    }

    function renderSystemActivity(overview, live) {
      const focus = live.focus_patient || {};
      const risk = focus.risk || {};
      const items = [
        `Monitored patients ${overview.patient_count ?? 0}`,
        `Open alert queue ${overview.open_alerts ?? 0}`,
        `Critical queue ${overview.critical_alerts ?? 0}`,
        `Top focus ${focus.patient_id || "p101"}`,
        `AI risk ${risk.risk_score ?? 0}`,
        `Confidence ${risk.confidence ?? 0}%`
      ];
      document.getElementById("systemActivity").innerHTML =
        items.map(i => `<div class="kv"><span class="k">Live Activity</span><span class="v">${i}</span></div>`).join("");
    }

    function renderChannels(data) {
      const channels = data.channels || [];
      document.getElementById("streamChannels").innerHTML =
        `<div class="panel-title">Stream Channels</div>` +
        channels.map(c => `<div class="kv"><span class="k">Channel</span><span class="v">${c}</span></div>`).join("");
    }

    function renderSimulator(patients) {
      const grid = document.getElementById("simulatorGrid");
      grid.innerHTML = "";
      patients.forEach((p) => {
        const v = p.vitals || {};
        const r = p.risk || {};
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="sim-title">${p.patient_name}</div>
          <div class="sim-room">${p.room} • ${p.program}</div>
          <div class="sim-risk">${r.risk_score}</div>
          <div style="margin-bottom:10px;" class="${severityClass(r.severity)}">${r.severity}</div>
          <div class="sim-copy">HR ${v.heart_rate} • BP ${v.systolic_bp}/${v.diastolic_bp} • SpO2 ${v.spo2}</div>
          <div class="meta-row">
            <div class="meta-pill">${r.clinical_priority}</div>
            <div class="meta-pill">${r.confidence}% confidence</div>
            <div class="meta-pill"><span class="${trendClass(r.trend_direction)}">${r.trend_arrow || ""}</span>&nbsp;${r.trend_direction}</div>
          </div>
        `;
        grid.appendChild(div);
      });
    }

    function renderTicker(overview, live) {
      const parts = [
        `Hospital journey active`,
        `Live demo journey active`,
        `Investor journey active`,
        `Monitored patients ${overview.patient_count ?? 0}`,
        `Open alerts ${overview.open_alerts ?? 0}`,
        `Critical alerts ${overview.critical_alerts ?? 0}`,
        `Average risk ${overview.avg_risk_score ?? 0}`,
        `Top focus ${live.focus_patient?.patient_id || "p101"}`,
        `Confidence scoring visible`,
        `Recommended action engine active`
      ];
      const doubled = parts.concat(parts);
      document.getElementById("opsTicker").innerHTML =
        doubled.map(p => `<span><i class="dot"></i>${p}</span>`).join("");
    }

    async function refreshDashboard() {
      try {
        const overview = await getJson(`/api/v1/dashboard/overview?tenant_id=${tenantId}`);
        const live = await getJson(`/api/v1/live-snapshot?tenant_id=${tenantId}&patient_id=${focusPatientId}`);
        const health = await getJson(`/api/v1/stream/health?tenant_id=${tenantId}`);
        const channels = await getJson(`/api/v1/stream/channels?tenant_id=${tenantId}&patient_id=${focusPatientId}`);

        animateNumber("metricPatients", overview.patient_count ?? 0);
        animateNumber("metricAlerts", overview.open_alerts ?? 0);
        animateNumber("metricCritical", overview.critical_alerts ?? 0);
        animateNumber("metricRisk", overview.avg_risk_score ?? 0);

        if ((overview.open_alerts ?? 0) > prevOpenAlerts || (overview.critical_alerts ?? 0) > prevCriticalAlerts) {
          playAlertBeep();
        }

        prevOpenAlerts = overview.open_alerts ?? 0;
        prevCriticalAlerts = overview.critical_alerts ?? 0;

        renderAlerts(live.alerts || []);
        renderFocus(live.focus_patient || {});
        renderStreamHealth(health || {});
        renderSystemActivity(overview, live);
        renderChannels(channels || {});
        renderSimulator(live.patients || []);
        renderTicker(overview, live);
      } catch (err) {
        console.error(err);
      }
    }

    refreshDashboard();
    setInterval(refreshDashboard, 5000);
  </script>
  <section id="cinematic-pathways" style="max-width:1360px;margin:34px auto 0;padding:0 16px 40px;">
  <style>
    .cp-wrap{
      position:relative;
      border:1px solid rgba(255,255,255,.08);
      border-radius:30px;
      overflow:hidden;
      background:
        radial-gradient(circle at 14% 18%, rgba(91,212,255,.14), transparent 24%),
        radial-gradient(circle at 50% 10%, rgba(166,140,255,.08), transparent 20%),
        radial-gradient(circle at 84% 18%, rgba(122,162,255,.14), transparent 24%),
        linear-gradient(180deg, rgba(13,21,38,.98), rgba(7,16,28,.99));
      box-shadow:0 24px 80px rgba(0,0,0,.34);
      isolation:isolate;
    }

    .cp-wrap::before{
      content:"";
      position:absolute;
      inset:0;
      background:
        linear-gradient(rgba(122,162,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(122,162,255,.035) 1px, transparent 1px);
      background-size:34px 34px;
      opacity:.45;
      pointer-events:none;
    }

    .cp-wrap::after{
      content:"";
      position:absolute;
      inset:auto -10% -16% -10%;
      height:220px;
      background:radial-gradient(circle at center, rgba(91,212,255,.16), transparent 62%);
      filter:blur(18px);
      animation:cpAmbientGlow 7s ease-in-out infinite;
      pointer-events:none;
      z-index:0;
    }

    @keyframes cpAmbientGlow{
      0%,100%{transform:translateY(0) scale(1);opacity:.55}
      50%{transform:translateY(-8px) scale(1.05);opacity:.9}
    }

    .cp-top{
      position:relative;
      z-index:2;
      padding:30px 30px 10px;
    }

    .cp-kicker{
      font-size:11px;
      font-weight:900;
      letter-spacing:.16em;
      text-transform:uppercase;
      color:#8fd7ff;
      margin-bottom:10px;
    }

    .cp-title{
      margin:0;
      font-size:clamp(34px,5vw,64px);
      line-height:.95;
      letter-spacing:-.05em;
      font-weight:1000;
      max-width:980px;
      color:#eef4ff;
    }

    .cp-sub{
      max-width:960px;
      margin:14px 0 0;
      color:#a9bddc;
      font-size:18px;
      line-height:1.65;
    }

    .cp-stage{
      position:relative;
      z-index:2;
      padding:18px 30px 6px;
    }

    .cp-rail{
      position:relative;
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:18px;
      align-items:stretch;
    }

    .cp-card{
      position:relative;
      min-height:340px;
      border-radius:26px;
      border:1px solid rgba(255,255,255,.08);
      background:
        linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02)),
        linear-gradient(180deg, rgba(12,22,38,.74), rgba(9,16,30,.86));
      overflow:hidden;
      box-shadow:0 16px 44px rgba(0,0,0,.24);
      transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }

    .cp-card:hover{
      transform:translateY(-5px);
      border-color:rgba(91,212,255,.28);
      box-shadow:0 20px 54px rgba(0,0,0,.30), 0 0 26px rgba(91,212,255,.12);
    }

    .cp-card::before{
      content:"";
      position:absolute;
      inset:0;
      background:linear-gradient(135deg, rgba(91,212,255,.06), transparent 34%, transparent 65%, rgba(122,162,255,.05));
      pointer-events:none;
    }

    .cp-story-rail{
      position:absolute;
      inset:auto 0 76px 0;
      height:2px;
      background:linear-gradient(90deg, transparent, rgba(91,212,255,.35), rgba(122,162,255,.28), transparent);
      opacity:.65;
      pointer-events:none;
    }

    .cp-story-rail::before,
    .cp-story-rail::after{
      content:"";
      position:absolute;
      top:50%;
      width:16px;
      height:16px;
      margin-top:-8px;
      border-radius:50%;
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
      box-shadow:0 0 20px rgba(91,212,255,.75);
      animation:cpDoctorMove 5.4s linear infinite;
    }

    .cp-story-rail::after{
      animation-delay:2.2s;
      opacity:.8;
    }

    @keyframes cpDoctorMove{
      0%{left:4%;opacity:0}
      8%{opacity:1}
      50%{left:50%;opacity:.95}
      92%{opacity:1}
      100%{left:94%;opacity:0}
    }

    .cp-orb{
      position:absolute;
      width:16px;
      height:16px;
      border-radius:50%;
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
      box-shadow:0 0 18px rgba(91,212,255,.92);
      left:22px;
      top:22px;
      z-index:2;
    }

    .cp-orb::after{
      content:"";
      position:absolute;
      inset:-14px;
      border-radius:50%;
      border:1px solid rgba(91,212,255,.26);
      animation:cpRing 2.8s ease-out infinite;
    }

    @keyframes cpRing{
      0%{transform:scale(.55);opacity:1}
      100%{transform:scale(1.9);opacity:0}
    }

    .cp-path-tag{
      position:absolute;
      top:18px;
      right:18px;
      padding:8px 12px;
      border-radius:999px;
      background:rgba(7,16,28,.58);
      border:1px solid rgba(255,255,255,.08);
      color:#d9e8ff;
      font-size:11px;
      font-weight:900;
      letter-spacing:.12em;
      text-transform:uppercase;
      backdrop-filter:blur(10px);
      z-index:2;
    }

    .cp-body{
      position:relative;
      z-index:2;
      padding:56px 22px 22px;
      height:100%;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
      gap:16px;
    }

    .cp-label{
      font-size:11px;
      font-weight:900;
      letter-spacing:.15em;
      text-transform:uppercase;
      color:#9fb7da;
    }

    .cp-card-title{
      font-size:32px;
      line-height:1;
      font-weight:1000;
      color:#f2f7ff;
      margin-top:10px;
      letter-spacing:-.03em;
    }

    .cp-copy{
      color:#c8d8ef;
      font-size:15px;
      line-height:1.65;
      margin-top:10px;
      min-height:108px;
    }

    .cp-mini-flow{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      margin-top:2px;
    }

    .cp-step{
      padding:7px 10px;
      border-radius:999px;
      background:rgba(255,255,255,.05);
      border:1px solid rgba(255,255,255,.06);
      color:#dbe7fb;
      font-size:12px;
      font-weight:800;
    }

    .cp-actions{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-top:8px;
    }

    .cp-btn{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      gap:8px;
      padding:12px 15px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,.10);
      background:#111b2f;
      color:#eef4ff;
      font-size:14px;
      font-weight:900;
      cursor:pointer;
      text-decoration:none;
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, opacity .18s ease;
    }

    .cp-btn:hover{
      transform:translateY(-2px);
      border-color:rgba(91,212,255,.28);
    }

    .cp-btn.primary{
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
      color:#07101c;
      border-color:transparent;
      box-shadow:0 12px 28px rgba(91,212,255,.22);
    }

    .cp-btn.tour{
      background:rgba(255,255,255,.05);
    }

    .cp-band{
      position:relative;
      z-index:2;
      margin:10px 30px 30px;
      border-radius:28px;
      border:1px solid rgba(255,255,255,.07);
      background:
        radial-gradient(circle at 20% 50%, rgba(91,212,255,.08), transparent 26%),
        radial-gradient(circle at 80% 50%, rgba(122,162,255,.08), transparent 26%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.018));
      overflow:hidden;
      box-shadow:0 16px 40px rgba(0,0,0,.20);
    }

    .cp-band::before{
      content:"";
      position:absolute;
      inset:0;
      background:linear-gradient(100deg, transparent 35%, rgba(255,255,255,.05) 50%, transparent 65%);
      transform:translateX(-60%);
      animation:cpSweep 7.5s linear infinite;
      pointer-events:none;
    }

    @keyframes cpSweep{
      0%{transform:translateX(-60%)}
      100%{transform:translateX(60%)}
    }

    .cp-band-inner{
      display:grid;
      grid-template-columns:1.1fr .9fr;
      gap:20px;
      align-items:center;
      padding:26px;
      position:relative;
      z-index:2;
    }

    .cp-band-kicker{
      font-size:11px;
      font-weight:900;
      letter-spacing:.16em;
      text-transform:uppercase;
      color:#8fd7ff;
      margin-bottom:10px;
    }

    .cp-band-title{
      margin:0;
      font-size:clamp(30px,4vw,56px);
      line-height:.96;
      font-weight:1000;
      letter-spacing:-.05em;
      color:#eef4ff;
    }

    .cp-band-copy{
      margin:12px 0 0;
      color:#b8cbe6;
      font-size:17px;
      line-height:1.65;
      max-width:760px;
    }

    .cp-band-points{
      display:grid;
      gap:10px;
    }

    .cp-band-point{
      padding:13px 14px;
      border-radius:16px;
      border:1px solid rgba(255,255,255,.06);
      background:rgba(255,255,255,.035);
      color:#e5efff;
      font-size:14px;
      font-weight:800;
      line-height:1.5;
    }

    .tour-modal{
      position:fixed;
      inset:0;
      background:rgba(4,8,16,.74);
      backdrop-filter:blur(8px);
      display:none;
      align-items:center;
      justify-content:center;
      z-index:9999;
      padding:18px;
    }

    .tour-modal.active{ display:flex; }

    .tour-panel{
      width:min(760px, 100%);
      border-radius:26px;
      border:1px solid rgba(255,255,255,.08);
      background:linear-gradient(180deg, rgba(16,26,45,.98), rgba(9,16,30,.98));
      box-shadow:0 24px 80px rgba(0,0,0,.42);
      overflow:hidden;
    }

    .tour-head{
      padding:22px 22px 14px;
      border-bottom:1px solid rgba(255,255,255,.06);
    }

    .tour-kicker{
      font-size:11px;
      font-weight:900;
      letter-spacing:.15em;
      text-transform:uppercase;
      color:#8fd7ff;
      margin-bottom:8px;
    }

    .tour-title{
      font-size:32px;
      line-height:1;
      font-weight:1000;
      margin:0;
    }

    .tour-body{
      padding:22px;
      display:grid;
      gap:16px;
    }

    .tour-step-label{
      font-size:12px;
      font-weight:900;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#9fb7da;
    }

    .tour-step-title{
      font-size:26px;
      line-height:1.1;
      font-weight:1000;
      margin:6px 0 8px;
    }

    .tour-step-copy{
      color:#c8d8ef;
      font-size:16px;
      line-height:1.65;
    }

    .tour-progress{
      width:100%;
      height:10px;
      border-radius:999px;
      background:rgba(255,255,255,.07);
      overflow:hidden;
      border:1px solid rgba(255,255,255,.06);
    }

    .tour-progress-bar{
      height:100%;
      width:0;
      background:linear-gradient(90deg,#7aa2ff,#5bd4ff);
      box-shadow:0 0 18px rgba(91,212,255,.24);
      transition:width .22s ease;
    }

    .tour-meta{
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:12px;
    }

    .tour-meta-card{
      border-radius:18px;
      border:1px solid rgba(255,255,255,.06);
      background:rgba(255,255,255,.03);
      padding:14px;
    }

    .tour-meta-card .k{
      font-size:11px;
      font-weight:900;
      letter-spacing:.12em;
      text-transform:uppercase;
      color:#9fb7da;
    }

    .tour-meta-card .v{
      display:block;
      margin-top:8px;
      font-size:18px;
      line-height:1.25;
      font-weight:900;
      color:#eef4ff;
    }

    .tour-actions{
      display:flex;
      justify-content:space-between;
      gap:12px;
      flex-wrap:wrap;
      padding:0 22px 22px;
    }

    .tour-highlight{
      outline:2px solid rgba(91,212,255,.72);
      outline-offset:4px;
      border-radius:18px;
      box-shadow:0 0 0 9999px rgba(3,8,16,.18), 0 0 28px rgba(91,212,255,.24);
      transition:outline .18s ease, box-shadow .18s ease;
    }

    @media (max-width:1080px){
      .cp-rail{ grid-template-columns:1fr; }
      .cp-copy{ min-height:auto; }
      .cp-story-rail{ display:none; }
      .cp-band-inner{ grid-template-columns:1fr; }
      .tour-meta{ grid-template-columns:1fr; }
    }

    @media (max-width:760px){
      .cp-top,.cp-stage{ padding-left:16px; padding-right:16px; }
      .cp-top{ padding-top:20px; }
      .cp-band{ margin:10px 16px 18px; }
      .cp-band-inner{ padding:18px; }
      .cp-card-title{ font-size:28px; }
      .tour-title{ font-size:28px; }
      .tour-step-title{ font-size:24px; }
      .tour-actions{ flex-direction:column; }
      .tour-actions .cp-btn{ width:100%; }
      .cp-actions{ flex-direction:column; }
      .cp-actions .cp-btn{ width:100%; }
    }
  </style>

  <div class="cp-wrap">
    <div class="cp-top">
      <div class="cp-kicker">Cinematic Guided Entry</div>
      <h2 class="cp-title">Choose the right path. Then let the platform guide the experience.</h2>
      <p class="cp-sub">
        Hospital Demo becomes the left guided story. Live Platform becomes the center product walkthrough.
        Investor Portal becomes the right commercial path. The lower cinematic explainer band ties the whole
        experience together as one polished enterprise presentation.
      </p>
    </div>

    <div class="cp-stage">
      <div class="cp-rail">
        <div class="cp-story-rail"></div>

        <div class="cp-card" id="tour-hospital-card">
          <div class="cp-orb"></div>
          <div class="cp-path-tag">Left Story Path</div>
          <div class="cp-body">
            <div>
              <div class="cp-label">Hospital Guided Story</div>
              <div class="cp-card-title">Hospital Demo</div>
              <div class="cp-copy">
                This is the hospital buyer journey. Clinical operators move from demo request into workflow review,
                command-center visibility, and operational follow-up.
              </div>
              <div class="cp-mini-flow">
                <span class="cp-step">Request demo</span>
                <span class="cp-step">Review workflow</span>
                <span class="cp-step">Operational follow-up</span>
              </div>
            </div>
            <div class="cp-actions">
              <a class="cp-btn primary" href="/hospital-demo">Open Hospital Path</a>
              <button class="cp-btn tour" type="button" onclick="startGuidedTour('hospital')">Launch Hospital Tour</button>
            </div>
          </div>
        </div>

        <div class="cp-card" id="tour-live-card">
          <div class="cp-orb"></div>
          <div class="cp-path-tag">Center Demo Path</div>
          <div class="cp-body">
            <div>
              <div class="cp-label">Interactive Product Walkthrough</div>
              <div class="cp-card-title">Live Platform</div>
              <div class="cp-copy">
                This is the guided live tour. The platform walks viewers through metrics, alerts, patient focus,
                simulation behavior, AI scoring, and command-center logic in sequence.
              </div>
              <div class="cp-mini-flow">
                <span class="cp-step">Metrics</span>
                <span class="cp-step">Alerts</span>
                <span class="cp-step">AI scoring</span>
                <span class="cp-step">System panels</span>
              </div>
            </div>
            <div class="cp-actions">
              <a class="cp-btn primary" href="#overview">Open Live Platform</a>
              <button class="cp-btn tour" type="button" onclick="startGuidedTour('live')">Start Walkthrough</button>
            </div>
          </div>
        </div>

        <div class="cp-card" id="tour-investor-card">
          <div class="cp-orb"></div>
          <div class="cp-path-tag">Right Commercial Path</div>
          <div class="cp-body">
            <div>
              <div class="cp-label">Investor Commercial Story</div>
              <div class="cp-card-title">Investor Portal</div>
              <div class="cp-copy">
                This is the investor route. Viewers move through positioning, traction, founder credibility,
                deck access, intake flow, and commercial readiness from one premium portal.
              </div>
              <div class="cp-mini-flow">
                <span class="cp-step">Positioning</span>
                <span class="cp-step">Pitch deck</span>
                <span class="cp-step">Founder credibility</span>
                <span class="cp-step">Intake workflow</span>
              </div>
            </div>
            <div class="cp-actions">
              <a class="cp-btn primary" href="/investors">Open Investor Path</a>
              <button class="cp-btn tour" type="button" onclick="startGuidedTour('investor')">Launch Investor Tour</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="cp-band">
      <div class="cp-band-inner">
        <div>
          <div class="cp-band-kicker">Cinematic Explainer Band</div>
          <h3 class="cp-band-title">Three audience paths. One guided platform story.</h3>
          <p class="cp-band-copy">
            Hospitals can move into demo capture and command-center evaluation. Product viewers can launch a live walkthrough.
            Investors can enter the commercial portal with a guided summary of traction, positioning, founder credibility, and next steps.
          </p>
        </div>

        <div class="cp-band-points">
          <div class="cp-band-point">Hospital Demo is the left-side guided buyer story.</div>
          <div class="cp-band-point">Live Platform is the center interactive product walkthrough.</div>
          <div class="cp-band-point">Investor Portal is the right-side commercial and pitch path.</div>
          <div class="cp-band-point">Guided tours launch directly from the buttons for tighter flow.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<div class="tour-modal" id="tourModal">
  <div class="tour-panel">
    <div class="tour-head">
      <div class="tour-kicker">Guided Walkthrough</div>
      <h3 class="tour-title" id="tourModeTitle">Platform Tour</h3>
    </div>

    <div class="tour-body">
      <div>
        <div class="tour-step-label" id="tourStepLabel">Step 1 of 4</div>
        <div class="tour-step-title" id="tourStepTitle">Welcome</div>
        <div class="tour-step-copy" id="tourStepCopy">This tour will guide you through the platform.</div>
      </div>

      <div class="tour-progress">
        <div class="tour-progress-bar" id="tourProgressBar"></div>
      </div>

      <div class="tour-meta" id="tourMeta"></div>
    </div>

    <div class="tour-actions">
      <button class="cp-btn" type="button" onclick="closeGuidedTour()">Exit Tour</button>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <button class="cp-btn" type="button" onclick="prevTourStep()">Back</button>
        <button class="cp-btn primary" id="tourNextBtn" type="button" onclick="nextTourStep()">Next Step</button>
      </div>
    </div>
  </div>
</div>

<script>
  (function(){
    const tourData = {
      live: {
        title: "Live Platform Walkthrough",
        steps: [
          {
            target: "#overview",
            title: "Hero + Guided Entry",
            copy: "Start at the cinematic homepage layer. This section introduces the platform as a premium clinical intelligence experience and routes viewers into the right product path.",
            meta: [
              ["Area", "Homepage overview"],
              ["Focus", "Hospital, live, and investor routing"],
              ["Outcome", "Stronger first impression"]
            ]
          },
          {
            target: "#dashboard",
            title: "Clinical Command Center",
            copy: "This section presents the live operational experience: alerts, patient focus, AI scoring, confidence indicators, recommended action, and system activity.",
            meta: [
              ["Area", "Command center"],
              ["Focus", "Operational decision support"],
              ["Outcome", "Response visibility at scale"]
            ]
          },
          {
            target: "#simulator",
            title: "Patient Simulator",
            copy: "The simulator demonstrates how the platform behaves in motion: live patients, changing risk, and scenario flow for product storytelling.",
            meta: [
              ["Area", "Simulator"],
              ["Focus", "Scenario-based demo"],
              ["Outcome", "More believable product narrative"]
            ]
          },
          {
            target: "#investor-view",
            title: "Investor + Commercial Layer",
            copy: "The investor layer packages the same product into a commercial story with positioning, credibility, contact flow, and pitch delivery.",
            meta: [
              ["Area", "Investor view"],
              ["Focus", "Commercial readiness"],
              ["Outcome", "Investor-facing polish"]
            ]
          }
        ]
      },
      hospital: {
        title: "Hospital Buyer Tour",
        steps: [
          {
            target: "#tour-hospital-card",
            title: "Hospital Guided Story",
            copy: "The hospital route is designed for clinical buyers, RPM operators, and hospital decision-makers who want to see workflow value and real demo capture.",
            meta: [
              ["Entry", "Left hospital path"],
              ["Buyer", "Clinical operations"],
              ["Goal", "Move toward a demo request"]
            ]
          },
          {
            target: "#dashboard",
            title: "Review Live Workflow",
            copy: "Show the buyer the command center first. This is where alert prioritization, patient severity, AI scoring, and recommended action become operational value.",
            meta: [
              ["Area", "Live dashboard"],
              ["Focus", "Workflow visibility"],
              ["Value", "Operational review"]
            ]
          },
          {
            target: "#hospital-demo",
            title: "Capture Hospital Interest",
            copy: "The hospital demo workflow collects organization, facility type, timeline, and interest so follow-up can happen like a real enterprise demo process.",
            meta: [
              ["Area", "Hospital demo form"],
              ["Focus", "Lead capture"],
              ["Outcome", "Qualified request"]
            ]
          },
          {
            target: "#admin-review",
            title: "Admin Review + Status Management",
            copy: "Submitted leads can be reviewed, exported, and later moved through statuses like New, Contacted, and Scheduled.",
            meta: [
              ["Area", "Admin review"],
              ["Focus", "Lead workflow"],
              ["Outcome", "Structured follow-up"]
            ]
          }
        ]
      },
      investor: {
        title: "Investor Portal Tour",
        steps: [
          {
            target: "#tour-investor-card",
            title: "Investor Commercial Story",
            copy: "The investor route frames the platform as an enterprise-ready healthcare AI company with operational credibility, founder clarity, and commercial flow.",
            meta: [
              ["Entry", "Right investor path"],
              ["Buyer", "Investors and strategic partners"],
              ["Goal", "Commercial confidence"]
            ]
          },
          {
            target: "#investor-view",
            title: "Investor Summary + Positioning",
            copy: "This section explains the product in investor language: enterprise positioning, clinical relevance, founder-led execution, and traction narrative.",
            meta: [
              ["Area", "Investor summary"],
              ["Focus", "Positioning + credibility"],
              ["Outcome", "Stronger investor framing"]
            ]
          },
          {
            target: "#investor-intake",
            title: "Investor Intake Workflow",
            copy: "The intake flow captures interest, check size, timeline, and notes so the platform behaves like a real investor follow-up system instead of a static page.",
            meta: [
              ["Area", "Investor intake"],
              ["Focus", "Pipeline capture"],
              ["Outcome", "Commercial workflow"]
            ]
          },
          {
            target: "#admin-review",
            title: "Deck + Review Layer",
            copy: "Pitch delivery and admin review connect the investor flow to a real operating pipeline with exports and later follow-up control.",
            meta: [
              ["Area", "Admin review"],
              ["Focus", "Deck + lead workflow"],
              ["Outcome", "Investor process readiness"]
            ]
          }
        ]
      }
    };

    let tourMode = "live";
    let tourIndex = 0;

    const modal = document.getElementById("tourModal");
    const modeTitle = document.getElementById("tourModeTitle");
    const stepLabel = document.getElementById("tourStepLabel");
    const stepTitle = document.getElementById("tourStepTitle");
    const stepCopy = document.getElementById("tourStepCopy");
    const progressBar = document.getElementById("tourProgressBar");
    const meta = document.getElementById("tourMeta");
    const nextBtn = document.getElementById("tourNextBtn");

    window.startGuidedTour = function(mode){
      tourMode = mode;
      tourIndex = 0;
      modal.classList.add("active");
      renderTour();
    };

    window.closeGuidedTour = function(){
      modal.classList.remove("active");
      clearHighlights();
    };

    window.nextTourStep = function(){
      const steps = tourData[tourMode].steps;
      if (tourIndex < steps.length - 1){
        tourIndex += 1;
        renderTour();
      } else {
        closeGuidedTour();
      }
    };

    window.prevTourStep = function(){
      if (tourIndex > 0){
        tourIndex -= 1;
        renderTour();
      }
    };

    function clearHighlights(){
      document.querySelectorAll(".tour-highlight").forEach(el => el.classList.remove("tour-highlight"));
    }

    function renderTour(){
      const pack = tourData[tourMode];
      const step = pack.steps[tourIndex];
      modeTitle.textContent = pack.title;
      stepLabel.textContent = "Step " + (tourIndex + 1) + " of " + pack.steps.length;
      stepTitle.textContent = step.title;
      stepCopy.textContent = step.copy;
      progressBar.style.width = (((tourIndex + 1) / pack.steps.length) * 100) + "%";
      nextBtn.textContent = (tourIndex === pack.steps.length - 1) ? "Finish Tour" : "Next Step";

      meta.innerHTML = "";
      (step.meta || []).forEach(([k,v]) => {
        const card = document.createElement("div");
        card.className = "tour-meta-card";
        card.innerHTML = '<div class="k">' + k + '</div><span class="v">' + v + '</span>';
        meta.appendChild(card);
      });

      clearHighlights();
      const target = document.querySelector(step.target);
      if (target){
        target.classList.add("tour-highlight");
        target.scrollIntoView({behavior:"smooth", block:"center"});
      }
    }

    modal.addEventListener("click", function(e){
      if (e.target === modal) closeGuidedTour();
    });

    document.addEventListener("keydown", function(e){
      if (!modal.classList.contains("active")) return;
      if (e.key === "Escape") closeGuidedTour();
      if (e.key === "ArrowRight") nextTourStep();
      if (e.key === "ArrowLeft") prevTourStep();
    });
  })();
</script>
</body>
</html>
"""

INVESTOR_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Investor Overview — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101d;--panel:#101a2d;--line:rgba(255,255,255,.08);--text:#edf4ff;--muted:#bdd0ec;
      --blue:#7aa2ff;--blue2:#5bd4ff;--green:#38d39f;--amber:#f7be68;--shadow:0 18px 50px rgba(0,0,0,.24)
    }
    *{box-sizing:border-box}
    body{
      margin:0;font-family:Inter,Arial,sans-serif;color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.14), transparent 24%),
        radial-gradient(circle at 82% 12%, rgba(91,212,255,.10), transparent 22%),
        linear-gradient(180deg, #07101d, #0b1324)
    }
    .wrap{max-width:1280px;margin:0 auto;padding:36px 18px 60px}
    .hero{
      position:relative;overflow:hidden;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.16), transparent 28%),
        radial-gradient(circle at 82% 12%, rgba(91,212,255,.12), transparent 24%),
        linear-gradient(180deg, rgba(16,26,45,.98), rgba(10,16,28,.98));
      border:1px solid var(--line);border-radius:26px;padding:32px;box-shadow:var(--shadow)
    }
    .hero:before{
      content:"";position:absolute;inset:0;
      background:linear-gradient(rgba(122,162,255,.035) 1px, transparent 1px),linear-gradient(90deg, rgba(122,162,255,.035) 1px, transparent 1px);
      background-size:34px 34px;opacity:.28;pointer-events:none
    }
    .grid{display:grid;grid-template-columns:1.05fr .95fr;gap:16px}
    .card{position:relative;background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.07);border-radius:22px;padding:22px}
    .kicker{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#bfd0ea;font-weight:900}
    h1{font-size:clamp(40px,4vw,64px);line-height:.95;margin:10px 0 14px;letter-spacing:-.05em}
    p{color:var(--muted);line-height:1.72}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;font-weight:900;text-decoration:none;margin-right:10px;margin-bottom:10px;box-shadow:0 10px 28px rgba(0,0,0,.18)}
    .btn.secondary{background:#111b2f;color:#edf4ff;border:1px solid rgba(255,255,255,.08);box-shadow:none}
    .mini-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}
    .mini{background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:18px}
    .k{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .v{font-size:28px;font-weight:1000;margin-top:10px}
    .s{font-size:14px;color:#c4d6ef;margin-top:8px;line-height:1.55}
    .list{display:grid;gap:10px;margin-top:14px}
    .li{padding:12px 14px;border-radius:16px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);color:#dce8fb}
    .founder{
      margin-top:20px;border-top:1px solid rgba(255,255,255,.08);padding-top:18px;
      display:grid;grid-template-columns:repeat(2,1fr);gap:12px
    }
    .contact-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.022));
      border:1px solid rgba(122,162,255,.12);border-radius:18px;padding:16px 18px
    }
    .contact-label{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .contact-value{margin-top:6px;font-size:18px;font-weight:900;color:#f3f8ff}
    .deck-panel{
      margin-top:18px;padding:18px;border-radius:18px;
      background:linear-gradient(135deg, rgba(122,162,255,.14), rgba(91,212,255,.08));
      border:1px solid rgba(91,212,255,.16)
    }
    .deck-panel h3{margin:0 0 8px;font-size:22px}
    .deck-panel p{margin:0 0 14px}
    @media (max-width:980px){.grid,.mini-grid,.founder{grid-template-columns:1fr}.wrap{padding:16px 10px 54px}.hero,.card{padding:16px}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="grid">
        <div class="card">
          <div class="kicker">Investor view</div>
          <h1>Healthcare AI platform with real clinical command-center presentation.</h1>
          <p>
            Early Risk Alert AI is a predictive healthcare intelligence platform built for hospitals, clinics,
            command centers, investors, and modern remote monitoring operations.
          </p>
          <p>
            The platform combines AI risk scoring, a live hospital-facing dashboard, hospital demo capture,
            executive walkthrough requests, investor intake, downloadable pitch materials, and admin review architecture in one branded experience.
          </p>
          <p>
            <a class="btn" href="/investor-intake">Investor Intake Form</a>
            <a class="btn" href="/deck">Download Pitch Deck PDF</a>
            <a class="btn secondary" href="/admin/review">Admin Review</a>
          </p>

          <div class="founder">
            <div class="contact-card"><div class="contact-label">Founder</div><div class="contact-value">__FOUNDER_NAME__</div></div>
            <div class="contact-card"><div class="contact-label">Role</div><div class="contact-value">__FOUNDER_ROLE__</div></div>
            <div class="contact-card"><div class="contact-label">Email</div><div class="contact-value">__INFO_EMAIL__</div></div>
            <div class="contact-card"><div class="contact-label">Phone</div><div class="contact-value">__BUSINESS_PHONE__</div></div>
          </div>

          <div class="deck-panel">
            <h3>Investor materials ready</h3>
            <p>Download the branded pitch deck directly from the platform, review hospital traction positioning, and capture investor follow-up in one workflow.</p>
            <a class="btn" href="/deck">Download Branded Deck</a>
          </div>
        </div>

        <div class="card">
          <div class="k">Investor summary</div>
          <div class="v">Enterprise Ready</div>
          <div class="s">A hospital-facing clinical intelligence platform with executive review workflows, investor intake, and branded pitch delivery.</div>
          <div class="list">
            <div class="li">Predictive clinical intelligence for hospitals and care operations</div>
            <div class="li">Command-center style dashboard with AI risk scoring and confidence indicators</div>
            <div class="li">Executive walkthrough capture, hospital demo requests, and investor intake workflows</div>
            <div class="li">Admin review and CSV export for a real operating pipeline</div>
            <div class="li">Founder-led product with investor-facing commercial positioning</div>
          </div>
        </div>
      </div>

      <div class="mini-grid">
        <div class="mini"><div class="k">Traction Angle</div><div class="v">Product Live</div><div class="s">Core public platform, forms, admin review, and investor flow are operational now.</div></div>
        <div class="mini"><div class="k">Market</div><div class="v">Hospitals / RPM</div><div class="s">Targets command centers, clinics, remote monitoring, and enterprise healthcare workflows.</div></div>
        <div class="mini"><div class="k">Revenue Model</div><div class="v">Enterprise SaaS</div><div class="s">Structured for hospital deployments, command dashboards, and recurring software contracts.</div></div>
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
    :root{--bg:#08111f;--panel:#101a2d;--line:rgba(255,255,255,.08);--text:#edf4ff;--muted:#c6d7ef;--blue:#7aa2ff;--blue2:#5bd4ff;--green:#38d39f;--shadow:0 18px 50px rgba(0,0,0,.24)}
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
      transition:border-color .18s ease, box-shadow .18s ease, background .18s ease
    }
    .field input:focus,.field select:focus,.field textarea:focus{outline:none;border-color:rgba(122,162,255,.42);box-shadow:0 0 0 3px rgba(122,162,255,.10);background:#101b2f}
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
      --green:#38d39f;--amber:#f7be68;--violet:#a88bff;--shadow:0 18px 50px rgba(0,0,0,.24)
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
      <p><a class="btn" href="/">Return Home</a><a class="btn" href="/admin/review">Open Admin Review</a></p>
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
    .admin-kpi.hospital{box-shadow:0 0 0 1px rgba(91,212,255,.12)}
    .admin-kpi.exec{box-shadow:0 0 0 1px rgba(247,190,104,.12)}
    .admin-kpi.investor{box-shadow:0 0 0 1px rgba(56,211,159,.12)}
    .admin-kpi .hint{font-size:13px;color:#c6d7ef;line-height:1.5}
    .section{margin-top:34px}
    .lead-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px}
    .lead-type{flex:1 1 260px;background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.06);border-radius:20px;padding:22px;min-height:130px}
    .lead-type h3{margin:0 0 8px;font-size:24px}
    .lead-type p{margin:0;color:#c9daf0;line-height:1.65}
    .table-wrap{overflow:auto;border:1px solid rgba(255,255,255,.06);border-radius:18px;background:rgba(255,255,255,.02)}
    table{width:100%;border-collapse:collapse;font-size:14px;min-width:920px}
    th,td{padding:14px 12px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;vertical-align:top}
    th{color:#9eb4d6;text-transform:uppercase;font-size:12px;letter-spacing:.12em;background:rgba(255,255,255,.02);position:sticky;top:0}
    tr:hover td{background:rgba(255,255,255,.02)}
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
      <p>Review hospital demo requests, executive walkthrough requests, and investor intake submissions. Update lead status, export CSV, and manage follow-up pipeline from one page.</p>
      <p><a class="btn" href="/admin/export.csv">Download CSV</a><a class="btn" href="/">Return Home</a></p>

      <div class="admin-grid">
        <div class="admin-kpi hospital"><div class="k">Hospital Demo Requests</div><div class="v">__HOSPITAL_COUNT__</div><div class="hint">Clinical buyer and operator pipeline</div></div>
        <div class="admin-kpi exec"><div class="k">Executive Walkthrough Requests</div><div class="v">__EXEC_COUNT__</div><div class="hint">Leadership and pilot evaluation requests</div></div>
        <div class="admin-kpi investor"><div class="k">Investor Intake Requests</div><div class="v">__INVESTOR_COUNT__</div><div class="hint">Commercial and investor follow-up flow</div></div>
      </div>

      <div class="lead-row">
        <div class="lead-type"><h3>Hospital Leads</h3><p>Clinical buyers, hospital operators, RPM teams, and health systems.</p></div>
        <div class="lead-type"><h3>Executive Leads</h3><p>Leadership-facing walkthrough requests for evaluations, pilots, and strategic review.</p></div>
        <div class="lead-type"><h3>Investor Leads</h3><p>Investor pipeline capture with contact details, timing, and opportunity notes.</p></div>
      </div>

      <div class="section"><h2>Hospital Demo Requests</h2>__HOSPITAL_TABLE__</div>
      <div class="section"><h2>Executive Walkthrough Requests</h2>__EXEC_TABLE__</div>
      <div class="section"><h2>Investor Intake</h2>__INVESTOR_TABLE__</div>
    </div>
  </div>
</body>
</html>
"""


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
    
def send_notification_email(subject, message):
    sender = INFO_EMAIL
    recipients = [INFO_EMAIL, FOUNDER_EMAIL]

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        server = smtplib.SMTP("smtp.zoho.com", 587)
        server.starttls()
        server.login(sender, os.getenv("EMAIL_PASSWORD"))
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
    except Exception as e:
        print("Email send failed:", e)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    print("ENTER create_app", flush=True)
    app.secret_key = os.getenv("SECRET_KEY", "early-risk-alert-dev-secret")

    data_dir = Path(app.instance_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    investor_file = data_dir / "investor_intake.jsonl"
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"

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

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "service": "early-risk-alert-ai", "canonical_host": CANONICAL_HOST or None})

    @app.get("/")
    def home():
        html_out = MAIN_HTML.replace("__YOUTUBE_EMBED_URL__", YOUTUBE_EMBED_URL)
        html_out = html_out.replace("__INFO_EMAIL__", INFO_EMAIL).replace("__BUSINESS_PHONE__", BUSINESS_PHONE)
        html_out = html_out.replace("__FOUNDER_NAME__", FOUNDER_NAME).replace("__FOUNDER_ROLE__", FOUNDER_ROLE)
        return render_template_string(html_out)

    @app.get("/dashboard")
    def dashboard():
        html_out = MAIN_HTML.replace("__YOUTUBE_EMBED_URL__", YOUTUBE_EMBED_URL)
        html_out = html_out.replace("__INFO_EMAIL__", INFO_EMAIL).replace("__BUSINESS_PHONE__", BUSINESS_PHONE)
        html_out = html_out.replace("__FOUNDER_NAME__", FOUNDER_NAME).replace("__FOUNDER_ROLE__", FOUNDER_ROLE)
        return render_template_string(html_out)

    @app.get("/investors")
    def investors():
        html_out = INVESTOR_HTML.replace("__INFO_EMAIL__", INFO_EMAIL).replace("__BUSINESS_PHONE__", BUSINESS_PHONE)
        html_out = html_out.replace("__FOUNDER_NAME__", FOUNDER_NAME).replace("__FOUNDER_ROLE__", FOUNDER_ROLE)
        return render_template_string(html_out)

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
                "phone": request.form.get("phone", "").strip(),
                "facility_type": request.form.get("facility_type", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            _save_jsonl(hospital_file, payload)
            subject = "New Hospital Demo Request"

            message = f"""
            New Hospital Demo Request

            Name: {payload['full_name']}
            Organization: {payload['organization']}
            Role: {payload['role']}
            Email: {payload['email']}
            Phone: {payload['phone']}
            Facility Type: {payload['facility_type']}
            Timeline: {payload['timeline']}
            Message: {payload['message']}
            """

            send_notification_email(subject, message)

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
    def admin_status_hospital():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(hospital_file, submitted_at, status)
        return redirect("/admin/review")

    @app.get("/admin/status/executive")
    def admin_status_executive():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(exec_file, submitted_at, status)
        return redirect("/admin/review")

    @app.get("/admin/status/investor")
    def admin_status_investor():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(investor_file, submitted_at, status)
        return redirect("/admin/review")

    @app.get("/admin/review")
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
    def deck():
        pdf_bytes = _generate_pitch_deck_pdf_bytes()
        return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf")

    @app.get("/api/v1/health")
    def api_health():
        return jsonify({"status": "ok", "service": "early-risk-alert-api", "canonical_host": CANONICAL_HOST or None})

    @app.get("/api/v1/dashboard/overview")
    def dashboard_overview():
        tenant_id = request.args.get("tenant_id", "demo")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        avg_risk = round(sum(r["risk"]["risk_score"] for r in rows) / len(rows))
        return jsonify({
            "tenant_id": tenant_id,
            "patient_count": len(rows),
            "open_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a["severity"] == "critical"),
            "events_last_hour": len(rows) * 3,
            "avg_risk_score": avg_risk,
        })

    @app.get("/api/v1/live-snapshot")
    def live_snapshot():
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        focus = next((r for r in rows if r["patient_id"] == patient_id), rows[0])
        focus["rollups"] = _build_rollups(focus)
        return jsonify({"tenant_id": tenant_id, "generated_at": _utc_now_iso(), "alerts": alerts[:5], "focus_patient": focus, "patients": rows})

    @app.get("/api/v1/stream/health")
    def stream_health():
        tenant_id = request.args.get("tenant_id", "demo")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        return jsonify({
            "tenant_id": tenant_id,
            "status": "running",
            "redis_ok": True,
            "mode": "realtime",
            "worker_status": "simulated",
            "patients_with_alerts": len({a["patient_id"] for a in alerts}),
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
        
    @app.route("/robots.txt")
    def robots_txt():
        return Response(
            "User-agent: *\nAllow: /\nSitemap: https://earlyriskalertai.com/sitemap.xml",
            mimetype="text/plain"
        )
         print("ABOUT TO RETURN APP", app, type(app), flush=True)
         return app



