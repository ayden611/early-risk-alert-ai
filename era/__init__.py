from __future__ import annotations

import csv
import html
import io
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, redirect, render_template_string, request, send_file


INFO_EMAIL = "info@earlyriskalertai.com"
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
<script>
  (function () {
    var allowedHost = "early-risk-alert-ai-1.onrender.com";
    if (window.location.hostname !== allowedHost) {
      var target = "https://" + allowedHost + window.location.pathname + window.location.search + window.location.hash;
      window.location.replace(target);
    }
  })();
</script>

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
      radial-gradient(circle at 82% 12%, rgba(91,212,255,.10), transparent 20%),
      linear-gradient(180deg, var(--bg), var(--bg2));
    overflow-x:hidden;
  }
</style>

<!-- NAVIGATION -->
<div class="nav">
  <div class="nav-inner">
    <div>
      <div class="brand-kicker">AI-powered predictive clinical intelligence</div>
      <div class="brand-title">Early Risk Alert AI</div>
      <div class="brand-sub">Hospitals · Clinics · Investors · Patients</div>
    </div>

    <div class="nav-links">
      <a href="https://early-risk-alert-ai-1.onrender.com/#overview">Overview</a>
      <a href="https://early-risk-alert-ai-1.onrender.com/dashboard">Clinical Command Center</a>
      <a href="https://early-risk-alert-ai-1.onrender.com/#simulator">Simulator</a>
      <a href="https://early-risk-alert-ai-1.onrender.com/investors">Investor View</a>
      <a href="https://early-risk-alert-ai-1.onrender.com/hospital-demo">Hospital Demo</a>
      <a class="btn primary" href="https://early-risk-alert-ai-1.onrender.com/deck">Download Pitch Deck</a>
    </div>
  </div>
</div>

<!-- HERO SECTION -->
<div class="shell">
  <section class="hero" id="overview">
    <div class="hero-inner">
      <div class="hero-grid">

        <div class="glass hero-copy">
          <div class="hero-kicker">Single-domain production experience</div>
          <h1>See the platform. Open the demo. Move hospitals and investors into the right path.</h1>
          <p>
            Early Risk Alert AI is a professional predictive clinical intelligence platform for hospitals,
            care operations, and investors. This opening experience keeps everything on one clean production domain
            and uses your brand image as the polished entry to the live platform story.
          </p>

          <div class="hero-actions">
            <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank">▶ Play Demo</a>
            <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/hospital-demo">Open Hospital Demo</a>
            <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/investors">Open Investor View</a>
            <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/dashboard">Open Command Center</a>
          </div>
        </div>

        <div class="glass demo-card" id="demo">
          <div class="demo-stage">
            <div class="demo-badge">Platform Demo</div>
            <div class="demo-play">
              <a class="demo-play-btn" href="https://youtu.be/z4SbeYwwm7k" target="_blank">
                ▶
              </a>
            </div>
            <div class="demo-caption">
              <h3>Early Risk Alert AI Master Demo</h3>
              <p>Public-facing demo entry using your branded image as the thumbnail and walkthrough video.</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  </section>
</div>
</body>
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
 <section id="demo-center" style="max-width:1360px;margin:34px auto 0;padding:0 16px 48px;">
  <style>
    .demo-wrap{
      position:relative;
      overflow:hidden;
      border:1px solid rgba(255,255,255,.08);
      border-radius:32px;
      background:
        radial-gradient(circle at 15% 20%, rgba(91,212,255,.10), transparent 24%),
        radial-gradient(circle at 85% 20%, rgba(122,162,255,.10), transparent 24%),
        linear-gradient(180deg, rgba(18,28,48,.96), rgba(8,14,26,.98));
      box-shadow:0 28px 90px rgba(0,0,0,.36);
    }

    .demo-wrap::before{
      content:"";
      position:absolute;
      inset:0;
      background:
        linear-gradient(rgba(122,162,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(122,162,255,.03) 1px, transparent 1px);
      background-size:34px 34px;
      opacity:.32;
      pointer-events:none;
    }

    .demo-top{
      position:relative;
      z-index:2;
      padding:30px 30px 14px;
    }

    .demo-kicker{
      font-size:11px;
      font-weight:900;
      letter-spacing:.16em;
      text-transform:uppercase;
      color:#8fd7ff;
      margin-bottom:10px;
    }

    .demo-title{
      margin:0;
      font-size:clamp(34px,5vw,62px);
      line-height:.95;
      letter-spacing:-.05em;
      font-weight:1000;
      color:#eef4ff;
      max-width:980px;
    }

    .demo-sub{
      margin:14px 0 0;
      max-width:920px;
      color:#afc3df;
      font-size:18px;
      line-height:1.66;
    }

    .demo-grid{
      position:relative;
      z-index:2;
      display:grid;
      grid-template-columns:1.18fr .82fr;
      gap:18px;
      padding:18px 30px 30px;
      align-items:start;
    }

    .demo-player{
      border-radius:28px;
      overflow:hidden;
      border:1px solid rgba(255,255,255,.08);
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 18px 52px rgba(0,0,0,.30);
    }

    .demo-thumb{
      position:relative;
      aspect-ratio:16 / 9;
      min-height:320px;
      background:
        linear-gradient(180deg, rgba(0,0,0,.12), rgba(0,0,0,.42)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
      display:flex;
      align-items:flex-end;
      padding:26px;
      overflow:hidden;
    }

    .demo-thumb::before{
      content:"";
      position:absolute;
      inset:auto -20% -26% -20%;
      height:220px;
      background:radial-gradient(circle at center, rgba(91,212,255,.16), transparent 62%);
      filter:blur(18px);
      pointer-events:none;
    }

    .demo-thumb::after{
      content:"";
      position:absolute;
      inset:0;
      background:linear-gradient(110deg, transparent 35%, rgba(255,255,255,.08) 50%, transparent 65%);
      transform:translateX(-70%);
      animation:demoSweep 7s linear infinite;
      pointer-events:none;
    }

    @keyframes demoSweep{
      0%{transform:translateX(-70%)}
      100%{transform:translateX(70%)}
    }

    .demo-badge{
      position:absolute;
      top:18px;
      left:18px;
      z-index:3;
      padding:9px 13px;
      border-radius:999px;
      background:rgba(7,16,28,.58);
      border:1px solid rgba(255,255,255,.10);
      color:#eaf3ff;
      font-size:11px;
      font-weight:900;
      letter-spacing:.14em;
      text-transform:uppercase;
      backdrop-filter:blur(12px);
    }

    .demo-play{
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      z-index:3;
    }

    .demo-play-btn{
      width:96px;
      height:96px;
      border-radius:50%;
      border:1px solid rgba(255,255,255,.18);
      background:rgba(7,16,28,.58);
      backdrop-filter:blur(12px);
      box-shadow:0 20px 50px rgba(0,0,0,.32), 0 0 34px rgba(91,212,255,.18);
      display:flex;
      align-items:center;
      justify-content:center;
      cursor:pointer;
      transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease;
      position:relative;
      text-decoration:none;
    }

    .demo-play-btn:hover{
      transform:scale(1.05);
      border-color:rgba(91,212,255,.38);
      box-shadow:0 24px 60px rgba(0,0,0,.36), 0 0 40px rgba(91,212,255,.24);
    }

    .demo-play-btn::before{
      content:"";
      position:absolute;
      inset:-14px;
      border-radius:50%;
      border:1px solid rgba(91,212,255,.20);
      animation:demoRing 2.8s ease-out infinite;
    }

    @keyframes demoRing{
      0%{transform:scale(.7);opacity:1}
      100%{transform:scale(1.35);opacity:0}
    }

    .demo-play-btn svg{
      width:34px;
      height:34px;
      margin-left:5px;
      fill:#eef4ff;
    }

    .demo-caption{
      position:relative;
      z-index:2;
      width:100%;
      display:flex;
      justify-content:space-between;
      align-items:end;
      gap:16px;
      flex-wrap:wrap;
    }

    .demo-caption-text h3{
      margin:0 0 8px;
      font-size:30px;
      line-height:1;
      font-weight:1000;
      letter-spacing:-.04em;
      color:#eef4ff;
    }

    .demo-caption-text p{
      margin:0;
      color:#d4e2f3;
      font-size:15px;
      line-height:1.6;
      max-width:720px;
    }

    .demo-actions{
      padding:18px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      justify-content:space-between;
      align-items:center;
      border-top:1px solid rgba(255,255,255,.06);
    }

    .demo-note{
      color:#bdd0eb;
      font-size:14px;
      line-height:1.55;
      max-width:540px;
    }

    .demo-btns{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
    }

    .demo-btn{
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
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    .demo-btn:hover{
      transform:translateY(-2px);
      border-color:rgba(91,212,255,.28);
    }

    .demo-btn.primary{
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
      color:#07101c;
      border-color:transparent;
      box-shadow:0 12px 28px rgba(91,212,255,.22);
    }

    .demo-side{
      display:grid;
      gap:14px;
    }

    .demo-card{
      border-radius:24px;
      border:1px solid rgba(255,255,255,.08);
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:20px;
    }

    .demo-label{
      font-size:11px;
      font-weight:900;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#8fd7ff;
    }

    .demo-card h4{
      margin:10px 0 8px;
      font-size:28px;
      line-height:1;
      font-weight:1000;
      letter-spacing:-.03em;
      color:#eef4ff;
    }

    .demo-card p{
      margin:0;
      color:#c7d8ef;
      font-size:15px;
      line-height:1.64;
    }

    .demo-meta{
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:14px;
      padding:0 30px 30px;
      position:relative;
      z-index:2;
    }

    .demo-mini{
      border-radius:22px;
      border:1px solid rgba(255,255,255,.07);
      background:rgba(255,255,255,.03);
      padding:18px;
    }

    .demo-mini .k{
      font-size:11px;
      font-weight:900;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#8fd7ff;
    }

    .demo-mini .v{
      display:block;
      margin-top:10px;
      font-size:22px;
      line-height:1.1;
      font-weight:1000;
      color:#eef4ff;
      letter-spacing:-.03em;
    }

    .demo-mini p{
      margin:10px 0 0;
      color:#c7d8ef;
      font-size:14px;
      line-height:1.6;
    }

    @media (max-width:1100px){
      .demo-grid{grid-template-columns:1fr}
      .demo-meta{grid-template-columns:1fr}
    }

    @media (max-width:760px){
      .demo-top,.demo-grid,.demo-meta{padding-left:16px;padding-right:16px}
      .demo-top{padding-top:20px}
      .demo-grid{padding-bottom:18px}
      .demo-meta{padding-bottom:18px}
      .demo-actions{flex-direction:column;align-items:stretch}
      .demo-btns{flex-direction:column}
      .demo-btn{width:100%}
      .demo-thumb{min-height:260px;padding:18px}
      .demo-caption-text h3{font-size:24px}
      .demo-title{font-size:clamp(30px,9vw,46px)}
    }
  </style>

  <div class="demo-wrap">
    <div class="demo-top">
      <div class="demo-kicker">Platform Demo</div>
      <h2 class="demo-title">See Early Risk Alert AI in action.</h2>
      <p class="demo-sub">
        A guided walkthrough of the hospital journey, live platform experience, and investor-facing commercial story.
      </p>
    </div>

    <div class="demo-grid">
      <div class="demo-player">
        <div class="demo-thumb">
          <div class="demo-badge">Watch Demo</div>

          <div class="demo-play">
            <a class="demo-play-btn" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer" aria-label="Watch demo video">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 5v14l11-7z"></path>
              </svg>
            </a>
          </div>

          <div class="demo-caption">
            <div class="demo-caption-text">
              <h3>Early Risk Alert AI Master Demo</h3>
              <p>
                A single premium walkthrough covering hospitals, the live command center, and the investor story.
              </p>
            </div>
          </div>
        </div>

        <div class="demo-actions">
          <div class="demo-note">
            Use this thumbnail section as your clean public-facing demo entry. Clicking play opens the full video in YouTube.
          </div>
          <div class="demo-btns">
            <a class="demo-btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
            <a class="demo-btn" href="/hospital-demo">Hospital Demo</a>
            <a class="demo-btn" href="/investors">Investor View</a>
          </div>
        </div>
      </div>

      <div class="demo-side">
        <div class="demo-card">
          <div class="demo-label">Hospital Story</div>
          <h4>Clinical Buyer Path</h4>
          <p>Show workflows, command-center visibility, AI risk scoring, and operational value for hospitals and care teams.</p>
        </div>

        <div class="demo-card">
          <div class="demo-label">Live Platform</div>
          <h4>Product Walkthrough</h4>
          <p>Present alerts, patient focus, confidence indicators, recommended action, and the live platform experience.</p>
        </div>

        <div class="demo-card">
          <div class="demo-label">Investor Story</div>
          <h4>Commercial Path</h4>
          <p>Guide investors through product readiness, founder credibility, market relevance, and enterprise SaaS positioning.</p>
        </div>
      </div>
    </div>

    <div class="demo-meta">
      <div class="demo-mini">
        <div class="k">Audience One</div>
        <span class="v">Hospitals</span>
        <p>Share before meetings to give operators and buyers immediate workflow context.</p>
      </div>
      <div class="demo-mini">
        <div class="k">Audience Two</div>
        <span class="v">Investors</span>
        <p>Use the same walkthrough in pitch outreach and strategic conversations.</p>
      </div>
      <div class="demo-mini">
        <div class="k">Audience Three</div>
        <span class="v">Partners</span>
        <p>Present one polished enterprise story across RPM, advisors, and healthcare partnerships.</p>
      </div>
    </div>
  </div>
</section>
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


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
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

    return app
