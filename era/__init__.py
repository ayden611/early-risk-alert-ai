from __future__ import annotations

import io
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, make_response, render_template_string, request, send_file


INFO_EMAIL = "info@earlyriskai.com"
BUSINESS_PHONE = "732-724-7267"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"


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


def _clamp(n: float, low: float, high: float) -> float:
    return max(low, min(high, n))


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
    confidence = int(_clamp(round(55 + (score * 0.4)), 55, 98))

    return {
        "risk_score": score,
        "confidence": confidence,
        "severity": severity,
        "alert_type": alert_type,
        "alert_message": alert_message,
    }


def _build_demo_patients() -> List[Dict[str, Any]]:
    seed = [
        {
            "patient_id": "p101",
            "patient_name": "Patient p101",
            "room": "ICU-12",
            "program": "Cardiac",
            "vitals": {
                "heart_rate": 126,
                "systolic_bp": 172,
                "diastolic_bp": 104,
                "spo2": 92,
                "temperature": 99.4,
                "resp_rate": 20,
            },
        },
        {
            "patient_id": "p202",
            "patient_name": "Patient p202",
            "room": "Stepdown-04",
            "program": "Pulmonary",
            "vitals": {
                "heart_rate": 98,
                "systolic_bp": 146,
                "diastolic_bp": 92,
                "spo2": 88,
                "temperature": 100.2,
                "resp_rate": 24,
            },
        },
        {
            "patient_id": "p303",
            "patient_name": "Patient p303",
            "room": "Telemetry-09",
            "program": "Cardiac",
            "vitals": {
                "heart_rate": 112,
                "systolic_bp": 164,
                "diastolic_bp": 98,
                "spo2": 95,
                "temperature": 98.9,
                "resp_rate": 18,
            },
        },
        {
            "patient_id": "p404",
            "patient_name": "Patient p404",
            "room": "Ward-21",
            "program": "Recovery",
            "vitals": {
                "heart_rate": 84,
                "systolic_bp": 128,
                "diastolic_bp": 82,
                "spo2": 97,
                "temperature": 98.6,
                "resp_rate": 16,
            },
        },
        {
            "patient_id": "p505",
            "patient_name": "Patient p505",
            "room": "ICU-05",
            "program": "Stroke",
            "vitals": {
                "heart_rate": 118,
                "systolic_bp": 156,
                "diastolic_bp": 96,
                "spo2": 93,
                "temperature": 99.8,
                "resp_rate": 22,
            },
        },
        {
            "patient_id": "p606",
            "patient_name": "Patient p606",
            "room": "RPM-Home",
            "program": "Remote Monitoring",
            "vitals": {
                "heart_rate": 89,
                "systolic_bp": 138,
                "diastolic_bp": 86,
                "spo2": 96,
                "temperature": 98.4,
                "resp_rate": 17,
            },
        },
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


def _generate_pitch_deck_pdf_bytes() -> bytes:
    lines = [
        "Early Risk Alert AI",
        "Professional Healthcare Intelligence Platform",
        "",
        "Company Overview",
        "Early Risk Alert AI is a command-center style platform for hospitals, clinics,",
        "investors, and remote monitoring operations. The platform combines AI risk scoring,",
        "live dashboards, alert prioritization, investor demonstration flow, and enterprise presentation.",
        "",
        "Core Value",
        "- Detect patient deterioration earlier",
        "- Surface high-risk patients in real time",
        "- Enable command-center visibility",
        "- Support hospital and RPM operations",
        "",
        "Business Model",
        "- Enterprise SaaS",
        "- Hospital and health system deployments",
        "- Command-center and RPM workflows",
        "",
        "Contact",
        f"- Email: {INFO_EMAIL}",
        f"- Phone: {BUSINESS_PHONE}",
    ]

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 760
    content_parts = ["BT", "/F1 20 Tf", "50 800 Td", f"({esc('Early Risk Alert AI - Pitch Deck')}) Tj"]
    for line in lines:
        y -= 24
        font_size = 11 if line.startswith("-") or not line else 13
        if line in {"Early Risk Alert AI", "Company Overview", "Core Value", "Business Model", "Contact"}:
            font_size = 16
        if line == "":
            continue
        content_parts.extend([f"/F1 {font_size} Tf", f"50 {y} Td", f"({esc(line)}) Tj"])
    content_parts.append("ET")
    stream = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n")

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    xref = [b"xref\n", f"0 {len(objects)+1}\n".encode("latin-1"), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode("latin-1"))
    trailer = f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
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
      --bg:#07111e;
      --bg2:#0d1526;
      --panel:#101a2d;
      --panel2:#12203a;
      --line:rgba(255,255,255,.08);
      --text:#edf4ff;
      --muted:#98afd2;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --red:#ff7a7a;
      --amber:#f5c06a;
      --radius:22px;
      --max:1320px;
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.14), transparent 22%),
        radial-gradient(circle at top right, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }
    a{color:inherit;text-decoration:none}
    .nav{
      position:sticky;top:0;z-index:100;
      backdrop-filter:blur(14px);
      background:rgba(7,17,30,.82);
      border-bottom:1px solid var(--line);
    }
    .nav-inner{
      max-width:var(--max);margin:0 auto;padding:14px 16px;
      display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
    }
    .brand-kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#bfd0ea;font-weight:900}
    .brand-title{font-size:clamp(28px,3vw,42px);font-weight:1000;line-height:.95;letter-spacing:-.045em}
    .brand-sub{font-size:14px;color:var(--muted);font-weight:800}
    .nav-links{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
    .nav-links a{font-weight:900;font-size:14px}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:13px 18px;border-radius:16px;font-weight:900;font-size:14px;
      background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;
      border:1px solid transparent;cursor:pointer;
    }
    .btn.secondary{background:#111b2f;color:var(--text);border-color:var(--line)}
    .btn.ghost{background:transparent;color:var(--text);border-color:var(--line)}
    .shell{max-width:var(--max);margin:0 auto;padding:24px 16px 70px}
    .hero{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;padding-top:18px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--line);border-radius:var(--radius);padding:28px;
      box-shadow:0 16px 50px rgba(0,0,0,.18);
    }
    .hero-kicker,.small-k{
      font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#c8d7f1;font-weight:900;margin-bottom:12px
    }
    h1{margin:0 0 14px;font-size:clamp(40px,5vw,76px);line-height:.95;font-weight:1000;letter-spacing:-.055em}
    .lead{margin:0;color:#c8d7f1;font-size:clamp(16px,1.5vw,20px);line-height:1.55}
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}
    .live-pill{
      display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;
      background:rgba(56,211,159,.12);border:1px solid rgba(56,211,159,.25);
      font-weight:900;font-size:12px;color:#bff5df;text-transform:uppercase;letter-spacing:.08em
    }
    .dot{width:10px;height:10px;border-radius:999px;background:var(--green);box-shadow:0 0 18px rgba(56,211,159,.6)}
    .side-title{font-size:clamp(28px,2.8vw,38px);font-weight:950;line-height:1.02;margin:18px 0 8px;letter-spacing:-.04em}
    .side-copy{color:#b8cae8;line-height:1.65;margin:0 0 18px;font-size:15px}
    .mini-grid,.metrics-grid,.trust-grid{display:grid;gap:14px}
    .mini-grid{grid-template-columns:repeat(2,1fr)}
    .metrics-grid{grid-template-columns:repeat(4,1fr)}
    .trust-grid{grid-template-columns:repeat(4,1fr)}
    .mini,.metric,.trust-card{border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.025);border-radius:18px;padding:18px}
    .mini-k,.metric-label,.contact-label{
      font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#9fb7da;font-weight:900
    }
    .mini-v,.metric-value{font-size:clamp(28px,2.6vw,54px);font-weight:1000;line-height:1;margin-top:10px}
    .mini-s,.metric-note{font-size:14px;color:#b7c9e7;margin-top:8px;line-height:1.45}
    .metric-note.flash{color:#dff0ff}
    .section{margin-top:28px}
    .section-title{font-size:clamp(30px,3.1vw,42px);font-weight:1000;letter-spacing:-.04em;margin:0 0 8px;line-height:1.02}
    .section-sub{color:var(--muted);font-size:16px;line-height:1.65;margin:0 0 18px;max-width:980px}
    .dashboard-grid{display:grid;grid-template-columns:1.12fr 1fr 1fr;gap:14px}
    .dash-card{padding:22px;min-height:380px}
    .dash-card h3{margin:0 0 6px;font-size:22px}
    .dash-card p{margin:0 0 16px;color:var(--muted);line-height:1.6}
    .feed{display:flex;flex-direction:column;gap:12px}
    .alert{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px}
    .alert-top{display:flex;justify-content:space-between;gap:12px}
    .alert-name{font-size:18px;font-weight:900;text-transform:capitalize}
    .alert-patient{font-size:13px;color:#b6c8e8;font-weight:700}
    .alert-msg{font-size:18px;font-weight:800;margin-top:6px;line-height:1.35}
    .alert-time{font-size:13px;color:#9ab0d3;margin-top:6px}
    .badge{display:inline-flex;align-items:center;justify-content:center;min-width:78px;padding:7px 11px;border-radius:999px;font-size:12px;text-transform:uppercase;letter-spacing:.1em;font-weight:900}
    .critical{background:rgba(255,122,122,.12);border:1px solid rgba(255,122,122,.22);color:#ffc1c1}
    .high{background:rgba(245,192,106,.14);border:1px solid rgba(245,192,106,.24);color:#ffe0a8}
    .moderate{background:rgba(122,162,255,.14);border:1px solid rgba(122,162,255,.24);color:#d1e0ff}
    .stable{background:rgba(56,211,159,.14);border:1px solid rgba(56,211,159,.24);color:#c5ffe6}
    .kv{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.05)}
    .kv:last-child{border-bottom:none}
    .k{color:#b6c8e8;font-weight:700}
    .v{font-weight:900}
    .panel-block{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px;margin-top:12px}
    .channels{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#d7e5ff;font-size:13px;line-height:1.7}
    .sim-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
    .sim-card{padding:16px;border-radius:18px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05)}
    .sim-title{font-size:16px;font-weight:900}
    .sim-room{font-size:12px;color:#a7bddf;margin-top:4px}
    .sim-risk{font-size:28px;font-weight:1000;margin:12px 0 6px}
    .sim-copy{font-size:14px;color:#c6d7ef}
    .ticker{
      display:flex;gap:28px;overflow:hidden;white-space:nowrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);
      padding:12px 0;margin-top:18px;color:#cde0f9;font-weight:800
    }
    .ticker span{display:inline-flex;align-items:center;gap:8px}
    .bullet{width:8px;height:8px;border-radius:999px;background:#7aa2ff}
    .two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    iframe{width:100%;aspect-ratio:16/9;border:0;border-radius:18px;background:#0a0f18}
    .summary-list{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-top:16px}
    .summary-item .sk{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9db3d5;font-weight:900}
    .summary-item .sv{font-size:20px;font-weight:900}
    .contact-box{display:grid;gap:12px;margin-top:18px}
    .contact-row{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:16px;padding:14px 16px}
    .trust-card h3{margin:0 0 8px;font-size:18px}
    .trust-card p{margin:0;color:#bfd0ea;line-height:1.65}
    .footer{margin-top:24px;padding:24px 12px 8px;color:#94a7c6;font-size:14px;text-align:center;line-height:1.6}
    .audio-toggle{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:14px;background:#111b2f;border:1px solid var(--line);color:var(--text);font-weight:900;cursor:pointer}
    .investor-form{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
    .investor-form .full{grid-column:1/-1}
    .field{display:flex;flex-direction:column;gap:8px}
    .field label{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9eb4d6;font-weight:900}
    .field input,.field select,.field textarea{
      width:100%;background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:14px;color:var(--text);
      padding:14px 14px;font:inherit
    }
    .field textarea{min-height:120px;resize:vertical}
    .success{
      margin-top:12px;padding:14px 16px;border-radius:16px;border:1px solid rgba(56,211,159,.25);
      background:rgba(56,211,159,.12);color:#d0ffe8;font-weight:800
    }
    @media (max-width:1180px){
      .hero,.two-col{grid-template-columns:1fr}
      .dashboard-grid{grid-template-columns:1fr 1fr}
      .dashboard-grid .dash-card:first-child{grid-column:1/-1}
      .metrics-grid,.trust-grid,.sim-grid{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:760px){
      .metrics-grid,.dashboard-grid,.trust-grid,.mini-grid,.summary-list,.sim-grid,.investor-form{grid-template-columns:1fr}
      .nav-links{width:100%}
      h1{font-size:clamp(34px,10vw,46px)}
      .alert-top,.kv{flex-direction:column}
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
        <a href="#metrics">Metrics</a>
        <a href="#dashboard">Live Dashboard</a>
        <a href="#simulator">Simulator</a>
        <a href="#demo">Demo</a>
        <a href="/investors">Investor View</a>
        <a href="/investor-intake">Investor Intake</a>
        <a class="btn" href="/deck">Download Pitch Deck</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="card">
        <div class="hero-kicker">Clinical command platform</div>
        <h1>Detect patient deterioration earlier. Strengthen hospital response at scale.</h1>
        <p class="lead">
          Early Risk Alert AI is a professional healthcare intelligence platform built for hospitals,
          clinics, investors, care teams, and remote monitoring programs. The platform combines command-center
          visibility, simulated live patient activity, AI deterioration scoring, investor presentation flow,
          and enterprise-grade operational storytelling in one polished experience.
        </p>
        <div class="hero-actions">
          <a class="btn" href="#dashboard">Open Live Dashboard</a>
          <a class="btn secondary" href="#simulator">View Patient Simulator</a>
          <a class="btn secondary" href="/investors">Open Investor Portal</a>
          <button class="audio-toggle" id="audioToggle" type="button">🔔 Enable Alert Sound</button>
        </div>
        <div class="ticker" id="opsTicker"></div>
      </div>

      <div class="card">
        <div class="live-pill"><span class="dot"></span> Live system</div>
        <div class="side-title">Built for modern hospital operations</div>
        <p class="side-copy">
          Your public-facing platform now combines hospital product messaging, animated live metrics,
          AI risk scoring, patient simulation, a command-center style dashboard, investor intake, and a downloadable pitch deck.
        </p>
        <div class="mini-grid">
          <div class="mini">
            <div class="mini-k">Model</div>
            <div class="mini-v">AI</div>
            <div class="mini-s">Real-time deterioration scoring</div>
          </div>
          <div class="mini">
            <div class="mini-k">Deployment</div>
            <div class="mini-v">SaaS</div>
            <div class="mini-s">Cloud-based enterprise rollout</div>
          </div>
          <div class="mini">
            <div class="mini-k">Buyer</div>
            <div class="mini-v">Hospitals</div>
            <div class="mini-s">Health systems, clinics, RPM teams</div>
          </div>
          <div class="mini">
            <div class="mini-k">Investor Mode</div>
            <div class="mini-v">Ready</div>
            <div class="mini-s">Portal, intake, deck download</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="metrics">
      <h2 class="section-title">Animated live metrics</h2>
      <p class="section-sub">Enterprise-style metrics auto-refreshing with simulated operational changes.</p>
      <div class="metrics-grid">
        <div class="card metric">
          <div class="metric-label">Monitored Patients</div>
          <div class="metric-value" id="metricPatients">0</div>
          <div class="metric-note" id="metricPatientsNote">Active patient population</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Open Alerts</div>
          <div class="metric-value" id="metricAlerts">0</div>
          <div class="metric-note" id="metricAlertsNote">Current command-center queue</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Critical Alerts</div>
          <div class="metric-value" id="metricCritical">0</div>
          <div class="metric-note" id="metricCriticalNote">Highest-priority interventions</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Avg Risk Score</div>
          <div class="metric-value" id="metricRisk">0</div>
          <div class="metric-note" id="metricRiskNote">AI risk intensity snapshot</div>
        </div>
      </div>
    </section>

    <section class="section" id="dashboard">
      <h2 class="section-title">Live dashboard</h2>
      <p class="section-sub">Hospitals see live dashboards, investor flows work, buttons work, and the platform stays enterprise-grade.</p>
      <div class="dashboard-grid">
        <div class="card dash-card">
          <h3>Live Alerts Feed</h3>
          <p>Clinical alert prioritization and real-time visibility.</p>
          <div class="feed" id="alertsFeed"></div>
        </div>

        <div class="card dash-card">
          <h3>Patient Focus</h3>
          <p>Focused patient view for vitals, event timing, and rollup metrics.</p>
          <div id="patientFocus"></div>
        </div>

        <div class="card dash-card">
          <h3>System Panels</h3>
          <p>Operational stream health, monitoring state, and channels.</p>
          <div class="panel-block" id="streamHealth"></div>
          <div class="panel-block channels" id="streamChannels"></div>
        </div>
      </div>
    </section>

    <section class="section" id="simulator">
      <h2 class="section-title">Live patient simulator</h2>
      <p class="section-sub">Auto-refreshing patient tiles with simulated vitals, severity changes, and command-center readiness.</p>
      <div class="sim-grid" id="simulatorGrid"></div>
    </section>

    <section class="section" id="demo">
      <h2 class="section-title">Live product demo</h2>
      <p class="section-sub">Professional embedded demo for hospital and investor presentations.</p>
      <div class="two-col">
        <div class="card">
          <iframe src="__YOUTUBE_EMBED_URL__" allowfullscreen></iframe>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Platform summary</h3>
          <p class="side-copy">
            Early Risk Alert AI combines hospital-facing product messaging, AI risk scoring, live operational dashboarding,
            investor intake, and pitch deck delivery into one polished healthcare platform.
          </p>
          <div class="summary-list">
            <div class="summary-item"><div class="sk">Use case</div><div class="sv">Clinical command</div></div>
            <div class="summary-item"><div class="sk">Deployment</div><div class="sv">Enterprise SaaS</div></div>
            <div class="summary-item"><div class="sk">Audience</div><div class="sv">Hospitals / Clinics / Investors</div></div>
            <div class="summary-item"><div class="sk">Experience</div><div class="sv">Live dashboard + intake + deck</div></div>
          </div>
          <div class="hero-actions" style="margin-top:18px;">
            <a class="btn" href="/investors">Investor Portal</a>
            <a class="btn secondary" href="/investor-intake">Investor Intake Form</a>
            <a class="btn secondary" href="/deck">PDF Pitch Deck</a>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="credibility">
      <h2 class="section-title">Business credibility layer</h2>
      <p class="section-sub">Professional positioning for hospitals, operators, and investors evaluating deployment readiness.</p>
      <div class="trust-grid">
        <div class="card trust-card">
          <h3>Hospital operations focus</h3>
          <p>Built to support command-center visibility, prioritization workflows, and faster intervention escalation.</p>
        </div>
        <div class="card trust-card">
          <h3>Investor-ready presentation</h3>
          <p>Unified public platform, live metrics, demo section, investor intake, and downloadable pitch materials in one branded experience.</p>
        </div>
        <div class="card trust-card">
          <h3>Clinical intelligence positioning</h3>
          <p>AI-driven risk scoring and live deterioration awareness framed for modern healthcare system adoption.</p>
        </div>
        <div class="card trust-card">
          <h3>Professional contact path</h3>
          <p>Direct channel for hospital partnerships, investor inquiries, and product demonstrations.</p>
        </div>
      </div>

      <div class="card" style="margin-top:14px;">
        <h3 style="margin-top:0;">Contact</h3>
        <div class="contact-box">
          <div class="contact-row">
            <div class="contact-label">Email</div>
            <div class="contact-value">__INFO_EMAIL__</div>
          </div>
          <div class="contact-row">
            <div class="contact-label">Business Phone</div>
            <div class="contact-value">__BUSINESS_PHONE__</div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC • Professional healthcare intelligence platform • __INFO_EMAIL__ • __BUSINESS_PHONE__
    </div>
  </div>

  <script>
    const tenantId = "demo";
    const focusPatientId = "p101";
    let soundEnabled = false;
    let prevOpenAlerts = 0;
    let prevCriticalAlerts = 0;

    const audioToggle = document.getElementById("audioToggle");
    audioToggle.addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      audioToggle.textContent = soundEnabled ? "🔔 Alert Sound On" : "🔔 Enable Alert Sound";
    });

    async function getJson(url) {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    }

    function badgeClass(sev) {
      if (sev === "critical") return "badge critical";
      if (sev === "high") return "badge high";
      if (sev === "moderate") return "badge moderate";
      return "badge stable";
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

    function pulseNote(id, text) {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = text;
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 700);
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
        div.className = "alert";
        div.innerHTML = `
          <div class="alert-top">
            <div>
              <div class="alert-name">${a.alert_type || "alert"}</div>
              <div class="alert-patient">${a.patient_name} • ${a.room}</div>
            </div>
            <div class="${badgeClass(a.severity)}">${a.severity}</div>
          </div>
          <div class="alert-msg">${a.message || "Clinical event detected"}</div>
          <div class="alert-time">Program: ${a.program} • live</div>
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
          <div class="alert-patient">${focus.patient_name || "Patient"} • ${focus.room || ""}</div>
          <div style="font-size:32px;font-weight:1000;line-height:1.05;margin:6px 0;">${focus.patient_id || ""}</div>
          <div style="font-size:18px;font-weight:800;color:#dbe7fb;">${r.alert_message || "Vitals stable"}</div>
          <div style="margin-top:10px;" class="${badgeClass(r.severity)}">${r.severity || "stable"}</div>
        </div>
        <div class="panel-block">
          <div class="kv"><span class="k">Heart Rate</span><span class="v">${v.heart_rate ?? "--"}</span></div>
          <div class="kv"><span class="k">Systolic BP</span><span class="v">${v.systolic_bp ?? "--"}</span></div>
          <div class="kv"><span class="k">Diastolic BP</span><span class="v">${v.diastolic_bp ?? "--"}</span></div>
          <div class="kv"><span class="k">SpO2</span><span class="v">${v.spo2 ?? "--"}</span></div>
          <div class="kv"><span class="k">Risk Score</span><span class="v">${r.risk_score ?? "--"}</span></div>
          <div class="kv"><span class="k">Avg SpO2</span><span class="v">${roll.avg_spo2 ?? "--"}</span></div>
        </div>
      `;
    }

    function renderStreamHealth(data) {
      document.getElementById("streamHealth").innerHTML = `
        <div class="kv"><span class="k">Status</span><span class="v">${data.status || "running"}</span></div>
        <div class="kv"><span class="k">Mode</span><span class="v">${data.mode || "realtime"}</span></div>
        <div class="kv"><span class="k">Worker</span><span class="v">${data.worker_status || "simulated"}</span></div>
        <div class="kv"><span class="k">Patients with alerts</span><span class="v">${data.patients_with_alerts ?? 0}</span></div>
      `;
    }

    function renderChannels(data) {
      document.getElementById("streamChannels").innerHTML = (data.channels || []).map(x => `<div>${x}</div>`).join("");
    }

    function renderSimulator(patients) {
      const grid = document.getElementById("simulatorGrid");
      grid.innerHTML = "";
      patients.forEach((p) => {
        const v = p.vitals || {};
        const r = p.risk || {};
        const div = document.createElement("div");
        div.className = "sim-card";
        div.innerHTML = `
          <div class="sim-title">${p.patient_name}</div>
          <div class="sim-room">${p.room} • ${p.program}</div>
          <div class="sim-risk">${r.risk_score}</div>
          <div style="margin-bottom:10px;" class="${badgeClass(r.severity)}">${r.severity}</div>
          <div class="sim-copy">
            HR ${v.heart_rate} • BP ${v.systolic_bp}/${v.diastolic_bp} • SpO2 ${v.spo2}
          </div>
        `;
        grid.appendChild(div);
      });
    }

    function renderTicker(overview, live) {
      const parts = [
        `Monitored patients ${overview.patient_count ?? 0}`,
        `Open alerts ${overview.open_alerts ?? 0}`,
        `Critical alerts ${overview.critical_alerts ?? 0}`,
        `Average risk ${overview.avg_risk_score ?? 0}`,
        `Top focus ${live.focus_patient?.patient_id || "p101"}`,
        `Investor-ready demo platform active`,
        `Hospital-facing monitoring flow live`,
        `AI risk scoring enabled`
      ];
      document.getElementById("opsTicker").innerHTML = parts.map(p => `<span><i class="bullet"></i>${p}</span>`).join("");
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

        pulseNote("metricPatientsNote", "Auto-refreshing monitored census");
        pulseNote("metricAlertsNote", "Live alert queue updating");
        pulseNote("metricCriticalNote", "Critical interventions refreshing");
        pulseNote("metricRiskNote", "AI scoring recalculating");

        if ((overview.open_alerts ?? 0) > prevOpenAlerts || (overview.critical_alerts ?? 0) > prevCriticalAlerts) {
          playAlertBeep();
        }
        prevOpenAlerts = overview.open_alerts ?? 0;
        prevCriticalAlerts = overview.critical_alerts ?? 0;

        renderAlerts(live.alerts || []);
        renderFocus(live.focus_patient || {});
        renderStreamHealth(health || {});
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
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:#edf4ff}
    .wrap{max-width:1100px;margin:0 auto;padding:40px 20px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:28px}
    h1{font-size:48px;line-height:1;margin:0 0 14px}
    p{color:#bdd0ec;line-height:1.7}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:900;text-decoration:none;margin-right:10px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Investor Overview</h1>
      <p>
        Early Risk Alert AI is a professional predictive healthcare platform built for hospitals,
        clinics, command centers, investors, and modern remote monitoring operations.
      </p>
      <p>
        The platform combines AI risk scoring, a live hospital-facing dashboard, product demonstration flow,
        investor intake capture, and enterprise presentation architecture in one branded software experience.
      </p>
      <p>
        <a class="btn" href="/investor-intake">Investor Intake Form</a>
        <a class="btn" href="/deck">Download Pitch Deck PDF</a>
      </p>
    </div>
  </div>
</body>
</html>
"""

INVESTOR_INTAKE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Investor Intake — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:#edf4ff}
    .wrap{max-width:1100px;margin:0 auto;padding:40px 20px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:28px}
    h1{font-size:42px;line-height:1;margin:0 0 14px}
    p{color:#bdd0ec;line-height:1.7}
    .investor-form{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
    .full{grid-column:1/-1}
    .field{display:flex;flex-direction:column;gap:8px}
    .field label{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9eb4d6;font-weight:900}
    .field input,.field select,.field textarea{
      width:100%;background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:14px;color:#edf4ff;
      padding:14px 14px;font:inherit
    }
    .field textarea{min-height:120px;resize:vertical}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:900;text-decoration:none;border:none;cursor:pointer}
    .success{margin-top:12px;padding:14px 16px;border-radius:16px;border:1px solid rgba(56,211,159,.25);background:rgba(56,211,159,.12);color:#d0ffe8;font-weight:800}
    @media (max-width:760px){.investor-form{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Investor Intake Form</h1>
      <p>Use this intake form to capture investor interest, timelines, and follow-up details directly from your platform.</p>
      __SUCCESS_BLOCK__
      <form method="post" class="investor-form">
        <div class="field">
          <label>Full Name</label>
          <input name="full_name" required>
        </div>
        <div class="field">
          <label>Organization</label>
          <input name="organization" required>
        </div>
        <div class="field">
          <label>Role</label>
          <input name="role" required>
        </div>
        <div class="field">
          <label>Email</label>
          <input type="email" name="email" required>
        </div>
        <div class="field">
          <label>Phone</label>
          <input name="phone">
        </div>
        <div class="field">
          <label>Investor Type</label>
          <select name="investor_type" required>
            <option value="Angel">Angel</option>
            <option value="Seed Fund">Seed Fund</option>
            <option value="Strategic">Strategic</option>
            <option value="Healthcare VC">Healthcare VC</option>
            <option value="Family Office">Family Office</option>
          </select>
        </div>
        <div class="field">
          <label>Check Size</label>
          <input name="check_size" placeholder="$50K - $250K">
        </div>
        <div class="field">
          <label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field full">
          <label>Interest / Notes</label>
          <textarea name="message" placeholder="What are you interested in learning more about?"></textarea>
        </div>
        <div class="field full">
          <button class="btn" type="submit">Submit Investor Intake</button>
        </div>
      </form>
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

    def _save_investor_submission(payload: Dict[str, Any]) -> None:
        with investor_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "service": "early-risk-alert-ai"})

    @app.get("/")
    def home():
        html = MAIN_HTML.replace("__YOUTUBE_EMBED_URL__", YOUTUBE_EMBED_URL)
        html = html.replace("__INFO_EMAIL__", INFO_EMAIL).replace("__BUSINESS_PHONE__", BUSINESS_PHONE)
        return render_template_string(html)

    @app.get("/dashboard")
    def dashboard():
        html = MAIN_HTML.replace("__YOUTUBE_EMBED_URL__", YOUTUBE_EMBED_URL)
        html = html.replace("__INFO_EMAIL__", INFO_EMAIL).replace("__BUSINESS_PHONE__", BUSINESS_PHONE)
        return render_template_string(html)

    @app.get("/investors")
    def investors():
        return render_template_string(INVESTOR_HTML)

    @app.route("/investor-intake", methods=["GET", "POST"])
    def investor_intake():
        success_block = ""
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
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
            _save_investor_submission(payload)
            success_block = "<div class='success'>Investor intake submitted successfully.</div>"
        return render_template_string(INVESTOR_INTAKE_HTML.replace("__SUCCESS_BLOCK__", success_block))

    @app.get("/deck")
    def deck():
        pdf_bytes = _generate_pitch_deck_pdf_bytes()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf",
        )

    @app.get("/api/v1/health")
    def api_health():
        return jsonify({"status": "ok", "service": "early-risk-alert-api"})

    @app.get("/api/v1/dashboard/overview")
    def dashboard_overview():
        tenant_id = request.args.get("tenant_id", "demo")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        avg_risk = round(sum(r["risk"]["risk_score"] for r in rows) / len(rows))
        return jsonify(
            {
                "tenant_id": tenant_id,
                "patient_count": len(rows),
                "open_alerts": len(alerts),
                "critical_alerts": sum(1 for a in alerts if a["severity"] == "critical"),
                "events_last_hour": len(rows) * 3,
                "avg_risk_score": avg_risk,
            }
        )

    @app.get("/api/v1/live-snapshot")
    def live_snapshot():
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        focus = next((r for r in rows if r["patient_id"] == patient_id), rows[0])
        focus["rollups"] = _build_rollups(focus)
        return jsonify(
            {
                "tenant_id": tenant_id,
                "generated_at": _utc_now_iso(),
                "alerts": alerts[:5],
                "focus_patient": focus,
                "patients": rows,
            }
        )

    @app.get("/api/v1/stream/health")
    def stream_health():
        tenant_id = request.args.get("tenant_id", "demo")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        return jsonify(
            {
                "tenant_id": tenant_id,
                "status": "running",
                "redis_ok": True,
                "mode": "realtime",
                "worker_status": "simulated",
                "patients_with_alerts": len({a["patient_id"] for a in alerts}),
            }
        )

    @app.get("/api/v1/stream/channels")
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

    @app.get("/api/v1/investor-intake/summary")
    def investor_summary():
        count = 0
        if investor_file.exists():
            with investor_file.open("r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
        return jsonify({"submissions": count, "email": INFO_EMAIL, "phone": BUSINESS_PHONE})

    return app
