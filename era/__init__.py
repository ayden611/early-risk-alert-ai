from __future__ import annotations

import csv
import io
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request, send_file


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
    mapping = {
        "critical": "Priority 1",
        "high": "Priority 2",
        "moderate": "Priority 3",
        "stable": "Priority 4",
    }
    return mapping.get(severity, "Priority 4")


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
                    "confidence": risk["confidence"],
                    "recommended_action": risk["recommended_action"],
                    "clinical_priority": risk["clinical_priority"],
                    "trend_direction": risk["trend_direction"],
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
        if line in {"Early Risk Alert AI", "Company Overview", "Core Value", "Contact"}:
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
      --panel2:#13223d;
      --line:rgba(255,255,255,.08);
      --text:#edf4ff;
      --muted:#98afd2;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --red:#ff6d7f;
      --amber:#f5c06a;
      --violet:#b58cff;
      --radius:22px;
      --max:1320px;
      --shadow:0 18px 50px rgba(0,0,0,.24);
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
      border:1px solid transparent;cursor:pointer;box-shadow:var(--shadow);
    }
    .btn.secondary{background:#111b2f;color:var(--text);border-color:var(--line);box-shadow:none}
    .btn.ghost{background:transparent;color:var(--text);border-color:var(--line);border:1px solid var(--line)}
    .shell{max-width:var(--max);margin:0 auto;padding:20px 16px 70px}
    .hero{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;padding-top:12px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--line);border-radius:var(--radius);padding:28px;
      box-shadow:var(--shadow);
    }
    .hero-kicker,.small-k{
      font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#c8d7f1;font-weight:900;margin-bottom:12px
    }
    h1{margin:0 0 14px;font-size:clamp(38px,5vw,74px);line-height:.95;font-weight:1000;letter-spacing:-.055em}
    .lead{margin:0;color:#c8d7f1;font-size:clamp(16px,1.5vw,20px);line-height:1.55}
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}
    .cta-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}
    .cta-card{
      border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.028);border-radius:18px;padding:16px
    }
    .cta-card h4{margin:0 0 8px;font-size:18px}
    .cta-card p{margin:0;color:#bdd0ec;line-height:1.5;font-size:14px}
    .cta-card .mini-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
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
    .section{margin-top:28px}
    .section-title{font-size:clamp(30px,3.1vw,42px);font-weight:1000;letter-spacing:-.04em;margin:0 0 8px;line-height:1.02}
    .section-sub{color:var(--muted);font-size:16px;line-height:1.65;margin:0 0 18px;max-width:980px}
    .ticker-wrap{
      margin-top:18px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);overflow:hidden
    }
    .ticker{
      display:flex;gap:28px;white-space:nowrap;padding:12px 0;color:#d3e4ff;font-weight:800;
      animation:ticker 18s linear infinite;
    }
    .ticker span{display:inline-flex;align-items:center;gap:8px}
    .bullet{width:8px;height:8px;border-radius:999px;background:#7aa2ff}
    @keyframes ticker{
      0%{transform:translateX(0)}
      100%{transform:translateX(-50%)}
    }

    .dashboard-grid{display:grid;grid-template-columns:1.16fr 1fr 1fr;gap:14px}
    .dash-card{padding:22px;min-height:420px}
    .dash-card h3{margin:0 0 6px;font-size:22px}
    .dash-card p{margin:0 0 16px;color:var(--muted);line-height:1.6}
    .feed{display:flex;flex-direction:column;gap:12px}
    .alert{
      background:rgba(255,255,255,.028);
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      padding:16px;
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .alert:hover{transform:translateY(-2px)}
    .alert-critical{
      box-shadow:0 0 0 1px rgba(255,109,127,.25), 0 0 30px rgba(255,109,127,.16);
      border-color:rgba(255,109,127,.26);
    }
    .alert-high{
      box-shadow:0 0 0 1px rgba(245,192,106,.16), 0 0 22px rgba(245,192,106,.08);
    }
    .alert-top{display:flex;justify-content:space-between;gap:12px}
    .alert-name{font-size:18px;font-weight:900;text-transform:capitalize}
    .alert-patient{font-size:13px;color:#b6c8e8;font-weight:700}
    .alert-msg{font-size:18px;font-weight:800;margin-top:6px;line-height:1.35}
    .alert-time{font-size:13px;color:#9ab0d3;margin-top:6px}
    .meta-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .meta-pill{
      padding:7px 10px;border-radius:999px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
      font-size:12px;font-weight:800;color:#d7e6ff
    }
    .badge{
      display:inline-flex;align-items:center;justify-content:center;min-width:98px;padding:9px 13px;border-radius:999px;
      font-size:12px;text-transform:uppercase;letter-spacing:.1em;font-weight:1000
    }
    .critical{background:rgba(255,109,127,.16);border:1px solid rgba(255,109,127,.32);color:#ffd1d8}
    .high{background:rgba(245,192,106,.16);border:1px solid rgba(245,192,106,.32);color:#ffe3b2}
    .moderate{background:rgba(122,162,255,.16);border:1px solid rgba(122,162,255,.32);color:#d6e4ff}
    .stable{background:rgba(56,211,159,.16);border:1px solid rgba(56,211,159,.32);color:#c7ffe4}

    .focus-top{
      display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap
    }
    .focus-id{font-size:34px;font-weight:1000;line-height:1}
    .panel-block{
      background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px;margin-top:12px
    }
    .kv{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.05)}
    .kv:last-child{border-bottom:none}
    .k{color:#b6c8e8;font-weight:700}
    .v{font-weight:900}
    .channels{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#d7e5ff;font-size:13px;line-height:1.7}
    .sim-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
    .sim-card{padding:16px;border-radius:18px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05)}
    .sim-title{font-size:16px;font-weight:900}
    .sim-room{font-size:12px;color:#a7bddf;margin-top:4px}
    .sim-risk{font-size:28px;font-weight:1000;margin:12px 0 6px}
    .sim-copy{font-size:14px;color:#c6d7ef}
    .summary-list{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-top:16px}
    .summary-item .sk{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9db3d5;font-weight:900}
    .summary-item .sv{font-size:20px;font-weight:900}
    .contact-box{display:grid;gap:12px;margin-top:18px}
    .contact-row{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:16px;padding:14px 16px}
    .trust-card h3{margin:0 0 8px;font-size:18px}
    .trust-card p{margin:0;color:#bfd0ea;line-height:1.65}
    .footer{margin-top:24px;padding:24px 12px 8px;color:#94a7c6;font-size:14px;text-align:center;line-height:1.6}
    .audio-toggle{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:14px;background:#111b2f;border:1px solid var(--line);color:var(--text);font-weight:900;cursor:pointer}
    iframe{width:100%;aspect-ratio:16/9;border:0;border-radius:18px;background:#0a0f18}

    @media (max-width:1180px){
      .hero,.summary-list{grid-template-columns:1fr}
      .dashboard-grid{grid-template-columns:1fr 1fr}
      .dashboard-grid .dash-card:first-child{grid-column:1/-1}
      .metrics-grid,.trust-grid,.sim-grid,.cta-grid{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:760px){
      .metrics-grid,.dashboard-grid,.trust-grid,.sim-grid,.mini-grid,.summary-list,.cta-grid{grid-template-columns:1fr}
      .nav-links{width:100%}
      .hero-actions{flex-direction:column;align-items:stretch}
      .btn,.audio-toggle{width:100%}
      .alert-top,.kv,.focus-top{flex-direction:column}
      .shell{padding:16px 12px 56px}
      .card,.dash-card{padding:18px}
      h1{font-size:clamp(34px,9vw,48px)}
      .lead{font-size:16px}
      .section-title{font-size:32px}
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
        <a class="btn" href="/deck">Download Pitch Deck</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="card">
        <div class="hero-kicker">Clinical command platform</div>
        <h1>Detect patient deterioration earlier. Strengthen response at scale.</h1>
        <p class="lead">
          Early Risk Alert AI is a professional healthcare intelligence platform built for hospitals, clinics,
          investors, care teams, and remote monitoring programs. The platform combines live patient activity,
          AI risk scoring, executive-ready workflow visibility, and polished command-center presentation in one experience.
        </p>

        <div class="hero-actions">
          <a class="btn" href="/hospital-demo">Request Hospital Demo</a>
          <a class="btn secondary" href="/executive-walkthrough">Schedule Executive Walkthrough</a>
          <a class="btn secondary" href="#dashboard">View Clinical Command Center</a>
          <button class="audio-toggle" id="audioToggle" type="button">🔔 Enable Alert Sound</button>
        </div>

        <div class="cta-grid">
          <div class="cta-card">
            <h4>Request Hospital Demo</h4>
            <p>Capture interest from hospital operators, RPM teams, and care leadership.</p>
            <div class="mini-actions"><a class="btn secondary" href="/hospital-demo">Open Form</a></div>
          </div>
          <div class="cta-card">
            <h4>Schedule Executive Walkthrough</h4>
            <p>Collect executive outreach requests for product review, pilots, and command-center evaluation.</p>
            <div class="mini-actions"><a class="btn secondary" href="/executive-walkthrough">Open Form</a></div>
          </div>
          <div class="cta-card">
            <h4>Investor-Ready Flow</h4>
            <p>Present a clean investor portal, intake workflow, and downloadable pitch materials.</p>
            <div class="mini-actions"><a class="btn secondary" href="/investors">Open Investor View</a></div>
          </div>
        </div>

        <div class="ticker-wrap">
          <div class="ticker" id="opsTicker">
            <span><i class="bullet"></i>System activity rotating</span>
            <span><i class="bullet"></i>Clinical operations visible</span>
            <span><i class="bullet"></i>AI prioritization enabled</span>
            <span><i class="bullet"></i>Investor workflow active</span>
            <span><i class="bullet"></i>System activity rotating</span>
            <span><i class="bullet"></i>Clinical operations visible</span>
            <span><i class="bullet"></i>AI prioritization enabled</span>
            <span><i class="bullet"></i>Investor workflow active</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="live-pill"><span class="dot"></span> Live system</div>
        <div class="side-title">Built for hospital operations and executive review</div>
        <p class="side-copy">
          Your platform now combines hospital demo intake, executive walkthrough capture, animated metrics,
          AI recommendation logic, patient simulation, command-center visibility, and investor follow-up tools.
        </p>
        <div class="mini-grid">
          <div class="mini">
            <div class="mini-k">Model</div>
            <div class="mini-v">AI</div>
            <div class="mini-s">Risk detection with confidence scoring</div>
          </div>
          <div class="mini">
            <div class="mini-k">Deployment</div>
            <div class="mini-v">SaaS</div>
            <div class="mini-s">Enterprise-ready command-center presentation</div>
          </div>
          <div class="mini">
            <div class="mini-k">Buyer</div>
            <div class="mini-v">Hospitals</div>
            <div class="mini-s">Health systems, clinics, RPM, executive stakeholders</div>
          </div>
          <div class="mini">
            <div class="mini-k">Workflow</div>
            <div class="mini-v">Live</div>
            <div class="mini-s">Hospital demos, investor intake, CSV export, admin review</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Animated live metrics</h2>
      <p class="section-sub">Cleaner layout, stronger spacing, and executive-facing operational KPIs.</p>
      <div class="metrics-grid">
        <div class="card metric">
          <div class="metric-label">Monitored Patients</div>
          <div class="metric-value" id="metricPatients">0</div>
          <div class="metric-note">Active patient population</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Open Alerts</div>
          <div class="metric-value" id="metricAlerts">0</div>
          <div class="metric-note">Current command-center queue</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Critical Alerts</div>
          <div class="metric-value" id="metricCritical">0</div>
          <div class="metric-note">Highest-priority interventions</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Avg AI Risk Score</div>
          <div class="metric-value" id="metricRisk">0</div>
          <div class="metric-note">Enterprise scoring snapshot</div>
        </div>
      </div>
    </section>

    <section class="section" id="dashboard">
      <h2 class="section-title">Clinical command center</h2>
      <p class="section-sub">Better dashboard styling with stronger priority visuals, larger badges, and clearer AI guidance.</p>
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
          <p>Operational state, workflow readiness, and stream channels.</p>
          <div class="panel-block" id="streamHealth"></div>
          <div class="panel-block channels" id="streamChannels"></div>
        </div>
      </div>
    </section>

    <section class="section" id="simulator">
      <h2 class="section-title">Live patient simulator</h2>
      <p class="section-sub">Simulated patient activity with stronger severity colors and AI scoring clarity.</p>
      <div class="sim-grid" id="simulatorGrid"></div>
    </section>

    <section class="section">
      <h2 class="section-title">Live product demo</h2>
      <p class="section-sub">Polished product messaging for hospital demos, executive walkthroughs, and investor review.</p>
      <div class="dashboard-grid" style="grid-template-columns:1.1fr .9fr;">
        <div class="card">
          <iframe src="__YOUTUBE_EMBED_URL__" allowfullscreen></iframe>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Executive summary</h3>
          <p class="side-copy">
            Early Risk Alert AI combines hospital-facing command visibility, patient simulation, AI risk scoring,
            hospital demo capture, executive walkthrough requests, investor intake, and pitch materials into one branded platform.
          </p>
          <div class="summary-list">
            <div class="summary-item"><div class="sk">Primary use case</div><div class="sv">Clinical command</div></div>
            <div class="summary-item"><div class="sk">Secondary use case</div><div class="sv">Executive evaluation</div></div>
            <div class="summary-item"><div class="sk">Delivery model</div><div class="sv">Enterprise SaaS</div></div>
            <div class="summary-item"><div class="sk">Commercial layer</div><div class="sv">Investor-ready</div></div>
          </div>
          <div class="hero-actions" style="margin-top:18px;">
            <a class="btn" href="/hospital-demo">Hospital Demo Form</a>
            <a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a>
            <a class="btn secondary" href="/investors">Investor Portal</a>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Business credibility layer</h2>
      <p class="section-sub">Professional positioning for hospitals, operators, and investors evaluating deployment readiness.</p>
      <div class="trust-grid">
        <div class="card trust-card">
          <h3>Hospital operations focus</h3>
          <p>Supports command-center visibility, prioritization workflows, and faster intervention escalation.</p>
        </div>
        <div class="card trust-card">
          <h3>Executive review flow</h3>
          <p>Structured demo request and walkthrough capture for hospital leadership and operational stakeholders.</p>
        </div>
        <div class="card trust-card">
          <h3>Investor-ready presentation</h3>
          <p>Unified platform, intake workflow, downloadable deck, and admin review layer.</p>
        </div>
        <div class="card trust-card">
          <h3>Professional contact path</h3>
          <p>Direct channel for hospital partnerships, investor inquiries, and product demonstration requests.</p>
        </div>
      </div>

      <div class="card" style="margin-top:14px;">
        <h3 style="margin-top:0;">Contact</h3>
        <div class="contact-box">
          <div class="contact-row">
            <div class="contact-label">Email</div>
            <div>__INFO_EMAIL__</div>
          </div>
          <div class="contact-row">
            <div class="contact-label">Business Phone</div>
            <div>__BUSINESS_PHONE__</div>
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

    document.getElementById("audioToggle").addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      document.getElementById("audioToggle").textContent = soundEnabled ? "🔔 Alert Sound On" : "🔔 Enable Alert Sound";
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

    function alertClass(sev) {
      if (sev === "critical") return "alert alert-critical";
      if (sev === "high") return "alert alert-high";
      return "alert";
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
            <div class="${badgeClass(a.severity)}">${a.severity}</div>
          </div>
          <div class="alert-msg">${a.message || "Clinical event detected"}</div>
          <div class="alert-time">Program: ${a.program} • live</div>
          <div class="meta-row">
            <div class="meta-pill">AI score ${a.risk_score}</div>
            <div class="meta-pill">Confidence ${a.confidence}%</div>
            <div class="meta-pill">${a.clinical_priority}</div>
            <div class="meta-pill">Trend ${a.trend_direction}</div>
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
            <div class="${badgeClass(r.severity)}">${r.severity || "stable"}</div>
          </div>
        </div>
        <div class="panel-block">
          <div class="kv"><span class="k">AI Risk Score</span><span class="v">${r.risk_score ?? "--"}</span></div>
          <div class="kv"><span class="k">Confidence</span><span class="v">${r.confidence ?? "--"}%</span></div>
          <div class="kv"><span class="k">Clinical Priority</span><span class="v">${r.clinical_priority ?? "--"}</span></div>
          <div class="kv"><span class="k">Trend Direction</span><span class="v">${r.trend_direction ?? "--"}</span></div>
          <div class="kv"><span class="k">Recommended Action</span><span class="v" style="max-width:58%;text-align:right;">${r.recommended_action ?? "--"}</span></div>
        </div>
        <div class="panel-block">
          <div class="kv"><span class="k">Heart Rate</span><span class="v">${v.heart_rate ?? "--"}</span></div>
          <div class="kv"><span class="k">Systolic BP</span><span class="v">${v.systolic_bp ?? "--"}</span></div>
          <div class="kv"><span class="k">Diastolic BP</span><span class="v">${v.diastolic_bp ?? "--"}</span></div>
          <div class="kv"><span class="k">SpO2</span><span class="v">${v.spo2 ?? "--"}</span></div>
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
          <div class="meta-row">
            <div class="meta-pill">${r.clinical_priority}</div>
            <div class="meta-pill">${r.confidence}% confidence</div>
            <div class="meta-pill">${r.trend_direction}</div>
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
        `Hospital demo requests enabled`,
        `Executive walkthrough flow active`,
        `Investor intake workflow active`
      ];
      const doubled = parts.concat(parts);
      document.getElementById("opsTicker").innerHTML = doubled.map(p => `<span><i class="bullet"></i>${p}</span>`).join("");
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
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:900;text-decoration:none;margin-right:10px;margin-bottom:10px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Investor Overview</h1>
      <p>
        Early Risk Alert AI is a professional predictive healthcare platform built for hospitals, clinics,
        command centers, investors, and modern remote monitoring operations.
      </p>
      <p>
        The platform combines AI risk scoring, a live hospital-facing dashboard, hospital demo capture,
        executive walkthrough requests, investor intake, and presentation architecture in one branded software experience.
      </p>
      <p>
        <a class="btn" href="/investor-intake">Investor Intake Form</a>
        <a class="btn" href="/deck">Download Pitch Deck PDF</a>
        <a class="btn" href="/admin/review">Admin Review</a>
      </p>
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
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:#edf4ff}
    .wrap{max-width:1100px;margin:0 auto;padding:40px 20px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:28px}
    h1{font-size:42px;line-height:1;margin:0 0 14px}
    p{color:#bdd0ec;line-height:1.7}
    .form{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
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
    @media (max-width:760px){.form{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>__HEADING__</h1>
      <p>__COPY__</p>
      __SUCCESS_BLOCK__
      <form method="post" class="form">
        __FIELDS__
        <div class="field full">
          <button class="btn" type="submit">__BUTTON__</button>
        </div>
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
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:#edf4ff}
    .wrap{max-width:900px;margin:0 auto;padding:60px 20px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:22px;padding:32px}
    h1{margin:0 0 12px;font-size:44px}
    p{color:#c0d2ec;line-height:1.7}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:900;text-decoration:none;margin-right:10px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Thank you</h1>
      <p>Your request was submitted successfully. Our team can now follow up on your hospital demo, executive walkthrough, or investor inquiry.</p>
      <p>
        <a class="btn" href="/">Return Home</a>
        <a class="btn" href="/admin/review">Admin Review</a>
      </p>
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
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:#edf4ff}
    .wrap{max-width:1200px;margin:0 auto;padding:40px 20px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:28px}
    h1{font-size:42px;line-height:1;margin:0 0 14px}
    p{color:#bdd0ec;line-height:1.7}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:900;text-decoration:none;margin-right:10px;margin-bottom:10px}
    table{width:100%;border-collapse:collapse;margin-top:20px;font-size:14px}
    th,td{padding:12px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;vertical-align:top}
    th{color:#9eb4d6;text-transform:uppercase;font-size:12px;letter-spacing:.12em}
    .section{margin-top:24px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Admin Review</h1>
      <p>Review hospital demo requests, executive walkthrough requests, and investor intake submissions. Export CSV when needed.</p>
      <p>
        <a class="btn" href="/admin/export.csv">Download CSV</a>
        <a class="btn" href="/">Return Home</a>
      </p>

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
  </div>
</body>
</html>
"""


def _table_html(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "<p>No submissions yet.</p>"
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in columns)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.secret_key = os.getenv("SECRET_KEY", "early-risk-alert-dev-secret")

    data_dir = Path(app.instance_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    investor_file = data_dir / "investor_intake.jsonl"
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"

    def save_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\\n")

    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        return rows

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

    @app.route("/hospital-demo", methods=["GET", "POST"])
    def hospital_demo():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "email": request.form.get("email", "").strip(),
                "phone": request.form.get("phone", "").strip(),
                "facility_type": request.form.get("facility_type", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            save_jsonl(hospital_file, payload)
            return render_template_string(THANK_YOU_PAGE)

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
        html = FORM_PAGE.replace("__TITLE__", "Hospital Demo — Early Risk Alert AI")
        html = html.replace("__HEADING__", "Request Hospital Demo")
        html = html.replace("__COPY__", "Capture interest from hospital operations teams, clinical leaders, and remote monitoring stakeholders.")
        html = html.replace("__SUCCESS_BLOCK__", "")
        html = html.replace("__FIELDS__", fields)
        html = html.replace("__BUTTON__", "Submit Hospital Demo Request")
        return render_template_string(html)

    @app.route("/executive-walkthrough", methods=["GET", "POST"])
    def executive_walkthrough():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "title": request.form.get("title", "").strip(),
                "email": request.form.get("email", "").strip(),
                "phone": request.form.get("phone", "").strip(),
                "priority": request.form.get("priority", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
            }
            save_jsonl(exec_file, payload)
            return render_template_string(THANK_YOU_PAGE)

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
        html = FORM_PAGE.replace("__TITLE__", "Executive Walkthrough — Early Risk Alert AI")
        html = html.replace("__HEADING__", "Schedule Executive Walkthrough")
        html = html.replace("__COPY__", "Capture executive-level product review requests for hospital leadership, system operations, and strategic evaluation.")
        html = html.replace("__SUCCESS_BLOCK__", "")
        html = html.replace("__FIELDS__", fields)
        html = html.replace("__BUTTON__", "Submit Executive Walkthrough Request")
        return render_template_string(html)

    @app.route("/investor-intake", methods=["GET", "POST"])
    def investor_intake():
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
            save_jsonl(investor_file, payload)
            return render_template_string(THANK_YOU_PAGE)

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
        html = FORM_PAGE.replace("__TITLE__", "Investor Intake — Early Risk Alert AI")
        html = html.replace("__HEADING__", "Investor Intake Form")
        html = html.replace("__COPY__", "Capture investor interest, timeline, and follow-up details directly from the platform.")
        html = html.replace("__SUCCESS_BLOCK__", "")
        html = html.replace("__FIELDS__", fields)
        html = html.replace("__BUTTON__", "Submit Investor Intake")
        return render_template_string(html)

    @app.get("/admin/review")
    def admin_review():
        hospital_rows = read_jsonl(hospital_file)
        exec_rows = read_jsonl(exec_file)
        investor_rows = read_jsonl(investor_file)

        html = ADMIN_HTML
        html = html.replace("__HOSPITAL_TABLE__", _table_html(
            hospital_rows,
            ["submitted_at", "full_name", "organization", "role", "email", "facility_type", "timeline"]
        ))
        html = html.replace("__EXEC_TABLE__", _table_html(
            exec_rows,
            ["submitted_at", "full_name", "organization", "title", "email", "priority", "timeline"]
        ))
        html = html.replace("__INVESTOR_TABLE__", _table_html(
            investor_rows,
            ["submitted_at", "full_name", "organization", "role", "email", "investor_type", "check_size", "timeline"]
        ))
        return render_template_string(html)

    @app.get("/admin/export.csv")
    def admin_export_csv():
        hospital_rows = read_jsonl(hospital_file)
        exec_rows = read_jsonl(exec_file)
        investor_rows = read_jsonl(investor_file)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["source", "submitted_at", "full_name", "organization", "role_or_title", "email", "phone", "type_or_priority", "timeline", "message"])

        for row in hospital_rows:
            writer.writerow([
                "hospital_demo",
                row.get("submitted_at", ""),
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
                "executive_walkthrough",
                row.get("submitted_at", ""),
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
                "investor_intake",
                row.get("submitted_at", ""),
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
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="early_risk_alert_admin_export.csv")

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

    return app
