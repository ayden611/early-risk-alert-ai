from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import random
import smtplib
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request, send_file


INFO_EMAIL = "info@earlyriskalertai.com"
FOUNDER_EMAIL = "milton.munroe@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder & AI Systems Engineer"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/z4SbeYwwm7k"
PROD_BASE_URL = "https://early-risk-alert-ai-1.onrender.com"


HOME_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --bg2:#0b1528;
      --panel:#101a2d;
      --panel2:#13203a;
      --line:rgba(255,255,255,.08);
      --line2:rgba(255,255,255,.05);
      --text:#eef4ff;
      --muted:#a8bddc;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --purple:#ba8cff;
      --shadow:0 20px 60px rgba(0,0,0,.30);
      --radius:24px;
      --max:1380px;
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.15), transparent 22%),
        radial-gradient(circle at 84% 12%, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
      overflow-x:hidden;
    }
    a{text-decoration:none;color:inherit}
    .shell{max-width:var(--max);margin:0 auto;padding:22px 16px 60px}
    .nav{
      position:sticky;top:0;z-index:1000;
      background:rgba(7,16,28,.82);
      backdrop-filter:blur(14px);
      border-bottom:1px solid var(--line);
    }
    .nav-inner{
      max-width:var(--max);
      margin:0 auto;
      padding:14px 16px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:18px;
      flex-wrap:wrap;
    }
    .brand-kicker{font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff}
    .brand-title{font-size:clamp(26px,3vw,40px);font-weight:1000;line-height:.95;letter-spacing:-.05em}
    .brand-sub{font-size:14px;color:var(--muted);font-weight:800}
    .nav-links{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
    .nav-links a{font-size:14px;font-weight:900}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:13px 18px;border-radius:16px;font-size:14px;font-weight:900;cursor:pointer;
      border:1px solid transparent;transition:.18s ease;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;box-shadow:0 12px 30px rgba(91,212,255,.22)}
    .btn.secondary{background:rgba(255,255,255,.04);color:var(--text);border-color:var(--line)}
    .hero{
      position:relative;overflow:hidden;border:1px solid var(--line);border-radius:34px;box-shadow:var(--shadow);
      background:
        linear-gradient(180deg, rgba(10,18,31,.30), rgba(7,16,28,.82)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
      min-height:740px;
    }
    .hero::before{
      content:"";position:absolute;inset:0;
      background:
        radial-gradient(circle at 48% 10%, rgba(91,212,255,.26), transparent 18%),
        linear-gradient(110deg, transparent 35%, rgba(255,255,255,.06) 50%, transparent 65%);
      animation:heroSweep 9s linear infinite;
      pointer-events:none;
    }
    @keyframes heroSweep{
      0%{transform:translateX(-40px)}
      50%{transform:translateX(40px)}
      100%{transform:translateX(-40px)}
    }
    .hero-inner{position:relative;z-index:2;min-height:740px;display:flex;align-items:flex-end;padding:36px}
    .hero-grid{width:100%;display:grid;grid-template-columns:1.08fr .92fr;gap:18px;align-items:end}
    .glass{
      border:1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.90));
      border-radius:28px;box-shadow:0 16px 42px rgba(0,0,0,.24);backdrop-filter:blur(14px);
    }
    .hero-copy{padding:30px}
    .hero-kicker{font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px}
    .hero-copy h1{margin:0 0 14px;font-size:clamp(40px,6vw,84px);line-height:.92;letter-spacing:-.06em;font-weight:1000;max-width:800px}
    .hero-copy p{margin:0;color:#d0ddf0;font-size:18px;line-height:1.68;max-width:760px}
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}
    .hero-mini-grid{margin-top:20px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .hero-mini{border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03)}
    .hero-mini .k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:8px}
    .hero-mini .v{font-size:16px;font-weight:1000;line-height:1.1}
    .route-grid{margin-top:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
    .route-card{
      border:1px solid var(--line);
      border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:20px;
    }
    .route-label{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff}
    .route-card h3{margin:10px 0 10px;font-size:26px;line-height:1;font-weight:1000;letter-spacing:-.04em}
    .route-card p{margin:0;color:#d0ddf0;font-size:15px;line-height:1.64}
    .route-card .route-actions{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap}
    .section{
      margin-top:22px;border:1px solid var(--line);border-radius:28px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:26px;
    }
    .section-kicker{font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px}
    .section h2{margin:0 0 10px;font-size:clamp(34px,5vw,58px);line-height:.95;letter-spacing:-.05em;font-weight:1000}
    .section p{margin:0;color:#d0ddf0;font-size:16px;line-height:1.7;max-width:980px}
    .list{margin-top:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
    .mini{border:1px solid var(--line2);border-radius:18px;padding:16px;background:rgba(255,255,255,.03)}
    .mini .k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px}
    .mini .v{display:block;font-size:20px;font-weight:1000;line-height:1.1;letter-spacing:-.03em}
    .mini p{margin:10px 0 0;color:#c7d8ef;font-size:14px;line-height:1.6}
    .footer{margin-top:22px;padding:18px 0 8px;color:#9fb4d6;font-size:14px;line-height:1.6}
    @media (max-width:1100px){
      .hero-grid{grid-template-columns:1fr}
      .hero-mini-grid{grid-template-columns:repeat(2,1fr)}
      .route-grid,.list{grid-template-columns:1fr}
    }
    @media (max-width:760px){
      .nav-inner{padding:12px 14px}
      .nav-links{gap:12px}
      .hero{min-height:auto}
      .hero-inner{min-height:auto;padding:16px}
      .hero-copy{padding:18px}
      .hero-copy h1{font-size:clamp(34px,11vw,54px)}
      .hero-copy p{font-size:16px}
      .hero-actions,.route-card .route-actions{flex-direction:column}
      .btn{width:100%}
      .hero-mini-grid{grid-template-columns:1fr}
      .section{padding:18px}
    }
  </style>
</head>
<body>
  <div class="nav">
    <div class="nav-inner">
      <div>
        <div class="brand-kicker">AI-powered predictive clinical intelligence</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Hospitals · Clinics · Investors · Insurers · Patients</div>
      </div>
      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#hospital-story">Hospitals</a>
        <a href="#commercial-path">Investors</a>
        <a href="#insurer-path">Insurers</a>
        <a href="#patient-path">Patients</a>
        <a href="/command-center">Command Center</a>
        <a class="btn primary" href="/admin/review">Admin Review</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="hero-inner">
        <div class="hero-grid">
          <div class="glass hero-copy">
            <div class="hero-kicker">Production platform narrative</div>
            <h1>Predictive monitoring that feels like NASA mission control for hospitals.</h1>
            <p>
              Early Risk Alert AI combines real-time patient simulation, ICU-style waveform telemetry,
              escalation intelligence, live commercial intake, and hospital-grade operating visibility
              into one polished command-center experience.
            </p>
            <div class="hero-actions">
              <a class="btn primary" href="/command-center">Open Live Command Center</a>
              <a class="btn secondary" href="/hospital-demo">Hospital Demo</a>
              <a class="btn secondary" href="/investor-intake">Investor Intake</a>
              <a class="btn secondary" href="/admin/review">Admin Review</a>
            </div>
            <div class="hero-mini-grid">
              <div class="hero-mini"><div class="k">Hospitals</div><div class="v">ICU-style predictive monitoring</div></div>
              <div class="hero-mini"><div class="k">Investors</div><div class="v">Live platform and pipeline visibility</div></div>
              <div class="hero-mini"><div class="k">Insurers</div><div class="v">Earlier intervention and lower-cost escalation</div></div>
              <div class="hero-mini"><div class="k">Patients</div><div class="v">Safer, faster, more proactive care response</div></div>
            </div>
          </div>

          <div class="glass hero-copy">
            <div class="hero-kicker">Why it lands quickly</div>
            <h2 style="margin:0 0 12px;font-size:36px;line-height:1.02;letter-spacing:-.04em">One visual experience, four audience stories.</h2>
            <p>Hospitals see clinical operations. Investors see scale. Insurers see predictive cost control. Patients see safety and confidence.</p>
            <div class="list" style="grid-template-columns:repeat(2,1fr)">
              <div class="mini" id="hospital-story"><div class="k">Hospital Story</div><span class="v">Real-time monitoring wall</span><p>Prioritize deterioration, see alerts, and route intervention earlier.</p></div>
              <div class="mini" id="commercial-path"><div class="k">Investor Story</div><span class="v">Platform + pipeline</span><p>Product visuals and operating demand in one place.</p></div>
              <div class="mini" id="insurer-path"><div class="k">Insurer Story</div><span class="v">Predictive triage</span><p>Earlier intervention supports better outcomes and lower-cost escalations.</p></div>
              <div class="mini" id="patient-path"><div class="k">Patient Story</div><span class="v">Proactive protection</span><p>Signals are caught sooner, before crises become emergencies.</p></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="route-grid">
      <div class="route-card">
        <div class="route-label">Command Center</div>
        <h3>Live predictive monitoring wall</h3>
        <p>Waveform monitors, AI risk scoring, alert hierarchy, and mission-control telemetry from one flagship page.</p>
        <div class="route-actions"><a class="btn secondary" href="/command-center">Open Command Center</a></div>
      </div>
      <div class="route-card">
        <div class="route-label">Hospital Intake</div>
        <h3>Clinical buyer path</h3>
        <p>Capture hospital demo requests with priority scoring, auto-replies, and admin-ready follow-up flow.</p>
        <div class="route-actions"><a class="btn secondary" href="/hospital-demo">Request Demo</a></div>
      </div>
      <div class="route-card">
        <div class="route-label">Executive Intake</div>
        <h3>Leadership review path</h3>
        <p>Capture executive walkthrough requests for pilots, strategy, and operational evaluation.</p>
        <div class="route-actions"><a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a></div>
      </div>
      <div class="route-card">
        <div class="route-label">Investor Intake</div>
        <h3>Commercial pipeline path</h3>
        <p>Capture investor interest, stages, score tags, and export-ready follow-up data.</p>
        <div class="route-actions"><a class="btn secondary" href="/investor-intake">Investor Intake</a></div>
      </div>
    </div>

    <section class="section">
      <div class="section-kicker">Audience narrative</div>
      <h2>One platform story, tailored by audience.</h2>
      <p>
        The command center is the flagship visual. The admin dashboard is the operating layer.
        The intake flows are the commercial engine. Together, they turn the platform from a demo into a working product story.
      </p>
      <div class="list">
        <div class="mini"><div class="k">Hospitals</div><span class="v">Operational monitoring</span><p>Live deterioration visibility, alert prioritization, and command-center workflow.</p></div>
        <div class="mini"><div class="k">Investors</div><span class="v">Product + traction</span><p>Command-center visuals plus intake pipeline and follow-up stages.</p></div>
        <div class="mini"><div class="k">Insurers</div><span class="v">Predictive cost control</span><p>Earlier detection supports triage, better coordination, and avoidable escalation reduction.</p></div>
        <div class="mini"><div class="k">Patients</div><span class="v">Safer outcomes</span><p>The platform feels proactive, protective, and clinically reassuring.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="section-kicker">Founder & Contact</div>
      <h2>Milton Munroe</h2>
      <p>
        Founder, Early Risk Alert AI · <a href="mailto:info@earlyriskalertai.com">info@earlyriskalertai.com</a> ·
        <a href="tel:7327247267">732-724-7267</a>
      </p>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC · Predictive clinical intelligence platform · Hospitals · Clinics · Investors · Insurers · Patients
    </div>
  </div>
</body>
</html>
"""


FORM_HTML = r"""
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
      --text:#ecf4ff;
      --muted:#9eb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Inter,Arial,sans-serif;
      background:linear-gradient(180deg,#07101c,#0d1628);
      color:var(--text);
      padding:26px 14px 50px;
    }
    .wrap{max-width:860px;margin:0 auto}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)), var(--panel);
      border:1px solid var(--line);
      border-radius:28px;
      padding:28px;
      box-shadow:0 18px 50px rgba(0,0,0,.24);
    }
    h1{margin:0 0 10px;font-size:clamp(34px,5vw,56px);line-height:.96;letter-spacing:-.05em}
    p{margin:0;color:var(--muted);line-height:1.7}
    form{display:grid;gap:14px;margin-top:22px}
    .field{display:grid;gap:8px}
    label{font-size:13px;font-weight:900;letter-spacing:.04em;text-transform:uppercase;color:#d8e7ff}
    input,select,textarea{
      width:100%;border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.03);
      color:var(--text);padding:14px 14px;font:inherit;outline:none;
    }
    textarea{min-height:130px;resize:vertical}
    .btn{
      border:0;border-radius:16px;padding:14px 18px;font:inherit;font-weight:1000;cursor:pointer;
      color:#08111f;background:linear-gradient(135deg,var(--blue),var(--blue2));box-shadow:0 12px 28px rgba(91,212,255,.2);
    }
    .back{display:inline-flex;margin-top:16px;color:#cfe2ff;font-weight:900}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>__HEADING__</h1>
      <p>__COPY__</p>
      <form method="post">
        __FIELDS__
        <button class="btn" type="submit">__BUTTON__</button>
      </form>
      <a class="back" href="/">Return Home</a>
    </div>
  </div>
</body>
</html>
"""


THANK_YOU_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Thank You - Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#07101c;color:#eef4ff;padding:28px}
    .wrap{max-width:760px;margin:0 auto}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:24px;padding:28px}
    h1{margin:0 0 10px;font-size:42px;letter-spacing:-.04em}
    p{margin:0;color:#bdd0ea;line-height:1.7}
    .box{margin-top:18px;padding:18px;border-radius:18px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08)}
    a{display:inline-flex;margin-top:18px;color:#08111f;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);padding:12px 16px;border-radius:14px;font-weight:1000;text-decoration:none}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Thank You</h1>
      <p>__MESSAGE__</p>
      <div class="box">__DETAILS__</div>
      <a href="/">Return Home</a>
    </div>
  </div>
</body>
</html>
"""


COMMAND_CENTER_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Early Risk Alert AI – Hospital Operations Cockpit</title>

<style>

body{
background:#06101c;
color:white;
font-family:Arial;
margin:0;
padding:30px;
}

.title{
font-size:34px;
font-weight:700;
margin-bottom:10px;
}

.subtitle{
opacity:.7;
margin-bottom:30px;
}

.grid{
display:grid;
grid-template-columns:2fr 1fr;
gap:20px;
}

.wall{
display:grid;
grid-template-columns:repeat(2,1fr);
gap:20px;
}

.monitor{
background:#0c1a2c;
border-radius:16px;
padding:18px;
border:1px solid rgba(255,255,255,.1);
box-shadow:0 10px 30px rgba(0,0,0,.4);
}

.monitor h3{
margin:0 0 10px;
}

.metric{
font-size:22px;
margin:6px 0;
}

.ecg{
height:90px;
background:black;
border-radius:6px;
margin:12px 0;
position:relative;
overflow:hidden;
}

.wave{
position:absolute;
width:200%;
height:100%;
background:
linear-gradient(90deg, transparent 0%, #00ff9c 40%, transparent 60%);
animation:wave 2s linear infinite;
opacity:.6;
}

@keyframes wave{
0%{left:-100%}
100%{left:100%}
}

.status{
padding:6px 12px;
border-radius:20px;
display:inline-block;
font-size:12px;
margin-top:10px;
}

.stable{background:#2ecc71}
.high{background:#f1c40f}
.critical{background:#e74c3c}

.sidepanel{
display:grid;
gap:20px;
}

.card{
background:#0d1b30;
border-radius:16px;
padding:16px;
border:1px solid rgba(255,255,255,.1);
}

.card h4{
margin-top:0;
}

.spark{
height:40px;
background:black;
border-radius:6px;
margin-top:10px;
position:relative;
overflow:hidden;
}

.sparkline{
position:absolute;
width:200%;
height:100%;
background:linear-gradient(90deg, transparent 0%, #5bd4ff 40%, transparent 60%);
animation:wave 3s linear infinite;
opacity:.6;
}

.heatmap{
display:grid;
grid-template-columns:repeat(4,1fr);
gap:8px;
margin-top:10px;
}

.heat{
height:40px;
border-radius:6px;
}

.low{background:#2ecc71}
.medium{background:#f1c40f}
.highh{background:#e74c3c}

.queue{
margin-top:10px;
}

.queue div{
padding:6px 0;
border-bottom:1px solid rgba(255,255,255,.1);
}

.capacity{
font-size:28px;
font-weight:700;
}

</style>

</head>
<body>

<div class="title">
Early Risk Alert AI — Hospital Command Center
</div>

<div class="subtitle">
AI predictive monitoring · clinical alert intelligence · hospital operations cockpit
</div>

<div class="grid">

<div class="wall" id="wall"></div>

<div class="sidepanel">

<div class="card">
<h4>Oxygen Trend</h4>
<div class="spark"><div class="sparkline"></div></div>
</div>

<div class="card">
<h4>Risk Prediction Timeline</h4>
<div class="spark"><div class="sparkline"></div></div>
</div>

<div class="card">
<h4>Deterioration Forecast</h4>
15 min: Moderate<br>
1 hr: Elevated<br>
4 hr: Critical
</div>

<div class="card">
<h4>Predictive Alert Clusters</h4>
Respiratory: 3<br>
Cardiac: 2<br>
Hemodynamic: 1
</div>

<div class="card">
<h4>Hospital Capacity</h4>
<div class="capacity">78%</div>
Beds Occupied
</div>

<div class="card">
<h4>Staffing Load</h4>
ICU Nurses: 82% capacity<br>
Respiratory: 67%<br>
Physicians: 74%
</div>

<div class="card">
<h4>Unit Heatmap</h4>
<div class="heatmap">
<div class="heat low"></div>
<div class="heat medium"></div>
<div class="heat highh"></div>
<div class="heat medium"></div>
<div class="heat medium"></div>
<div class="heat low"></div>
<div class="heat highh"></div>
<div class="heat medium"></div>
</div>
</div>

<div class="card">
<h4>Rapid Response Queue</h4>
<div class="queue" id="queue"></div>
</div>

</div>

</div>

<script>

const wall=document.getElementById("wall")
const queue=document.getElementById("queue")

const evt=new EventSource("/api/command-center-stream")

evt.onmessage=function(e){

const data=JSON.parse(e.data)

const div=document.createElement("div")
div.className="monitor"

let statusClass="stable"
if(data.status==="HIGH") statusClass="high"
if(data.status==="CRITICAL") statusClass="critical"

div.innerHTML=`

<h3>${data.patient}</h3>

<div class="metric">HR: ${data.heart_rate}</div>
<div class="metric">SpO2: ${data.spo2}%</div>
<div class="metric">BP: ${data.bp}</div>
<div class="metric">AI Risk: ${data.risk}</div>

<div class="ecg"><div class="wave"></div></div>

<div class="status ${statusClass}">
${data.status}
</div>

`

wall.prepend(div)

if(wall.children.length>6){
wall.removeChild(wall.lastChild)
}

if(data.status==="CRITICAL" || data.status==="HIGH"){

const q=document.createElement("div")
q.innerText=data.patient+" – "+data.status

queue.prepend(q)

if(queue.children.length>6){
queue.removeChild(queue.lastChild)
}

}

}

</script>

</body>
</html>
"""

ADMIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin Review — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --bg2:#0b1528;
      --panel:#101a2d;
      --panel2:#13203a;
      --line:rgba(255,255,255,.08);
      --line2:rgba(255,255,255,.05);
      --text:#eef4ff;
      --muted:#a7bddc;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#3ad38f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --purple:#b58cff;
      --shadow:0 20px 60px rgba(0,0,0,.34);
      --radius:24px;
      --max:1440px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.14), transparent 22%),
        radial-gradient(circle at 88% 10%, rgba(91,212,255,.10), transparent 18%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }
    a{text-decoration:none;color:inherit}
    .shell{max-width:var(--max);margin:0 auto;padding:20px 16px 44px}
    .topbar{
      position:sticky;top:0;z-index:50;
      background:rgba(7,16,28,.82);
      backdrop-filter:blur(14px);
      border-bottom:1px solid var(--line);
    }
    .topbar-inner{
      max-width:var(--max);
      margin:0 auto;
      padding:14px 16px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:18px;
      flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fdcff;
    }
    .brand-title{
      font-size:clamp(26px,3vw,40px);font-weight:1000;line-height:.95;letter-spacing:-.05em;
    }
    .brand-sub{
      font-size:14px;color:var(--muted);font-weight:800;
    }
    .nav-links{
      display:flex;align-items:center;gap:14px;flex-wrap:wrap;
    }
    .nav-links a{font-size:14px;font-weight:900}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      border:1px solid transparent;cursor:pointer;
      transition:transform .18s ease, box-shadow .18s ease, opacity .18s ease;
      text-decoration:none;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#07101c;box-shadow:0 12px 30px rgba(91,212,255,.22);
    }
    .btn.secondary{
      background:rgba(255,255,255,.04);
      border-color:var(--line);
      color:var(--text);
    }

    .hero{
      margin-top:18px;
      border:1px solid var(--line);
      border-radius:30px;
      background:
        radial-gradient(circle at 20% 20%, rgba(91,212,255,.08), transparent 18%),
        radial-gradient(circle at 80% 0%, rgba(181,140,255,.08), transparent 18%),
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015)),
        linear-gradient(180deg, rgba(10,19,34,.86), rgba(7,14,26,.96));
      box-shadow:var(--shadow);
      padding:24px;
    }
    .hero-grid{
      display:grid;grid-template-columns:1.3fr .9fr;gap:18px;
    }
    .hero-copy{
      border:1px solid var(--line);
      border-radius:26px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:22px;
    }
    .hero-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#91ddff;margin-bottom:10px;
    }
    .hero-copy h1{
      margin:0 0 12px;font-size:clamp(36px,5vw,66px);line-height:.92;letter-spacing:-.06em;font-weight:1000;
    }
    .hero-copy p{
      margin:0;color:#d0def2;font-size:16px;line-height:1.68;max-width:820px;
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}
    .hero-mini-grid{
      margin-top:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
    }
    .hero-mini{
      border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
    }
    .hero-mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fdcff;margin-bottom:8px;
    }
    .hero-mini .v{
      font-size:18px;font-weight:1000;line-height:1.1;letter-spacing:-.03em;
    }

    .mission-panel{
      border:1px solid var(--line);
      border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;
      position:relative;
      overflow:hidden;
    }
    .mission-panel::before{
      content:"";
      position:absolute;inset:0;
      background:linear-gradient(120deg, transparent 30%, rgba(255,255,255,.04) 50%, transparent 70%);
      animation:panelSweep 9s linear infinite;
      pointer-events:none;
    }
    @keyframes panelSweep{
      0%{transform:translateX(-35%)}
      100%{transform:translateX(35%)}
    }
    .mission-header{
      display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px;
    }
    .mission-title{font-size:24px;font-weight:1000;letter-spacing:-.04em}
    .mission-sub{font-size:13px;color:var(--muted);line-height:1.55}
    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:9px 13px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.12);
      background:rgba(255,255,255,.05);
    }
    .status-new{color:#0b1528;background:linear-gradient(135deg,var(--blue),var(--blue2))}
    .status-contacted{color:#2a1b00;background:linear-gradient(135deg,var(--amber),#ffdba5)}
    .status-closed{color:#fff3f5;background:linear-gradient(135deg,#ff667d,#ff8e99)}
    .stage-dd{color:#120c25;background:linear-gradient(135deg,var(--purple),#d3b4ff)}

    .mission-stats{
      display:grid;grid-template-columns:repeat(4,1fr);gap:14px;
    }
    .stat-card{
      border:1px solid var(--line);
      border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(13,23,40,.90), rgba(8,15,28,.95));
      padding:16px;min-height:122px;position:relative;overflow:hidden;
    }
    .stat-card::after{
      content:"";
      position:absolute;right:-30px;top:-30px;width:120px;height:120px;border-radius:50%;
      background:radial-gradient(circle, rgba(91,212,255,.12), transparent 60%);
      pointer-events:none;
    }
    .stat-card .k{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8ddcff;margin-bottom:10px;
    }
    .stat-card .v{
      font-size:40px;font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .stat-card .hint{
      margin-top:8px;font-size:13px;line-height:1.5;color:#c8d8ef;
    }

    .filter-bar{
      margin-top:20px;
      display:grid;grid-template-columns:1.2fr .9fr .9fr .9fr auto;
      gap:12px;
    }
    .field{
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,255,255,.04);
      padding:12px 14px;
      color:var(--text);
      font-size:14px;
      font-weight:800;
      min-height:48px;
      width:100%;
      outline:none;
    }
    .field::placeholder{color:#9fb6d7}
    select.field{appearance:none}

    .pipeline-grid{
      margin-top:20px;
      display:grid;
      grid-template-columns:repeat(4,1fr);
      gap:14px;
    }
    .pipeline-card{
      border:1px solid var(--line);
      border-radius:20px;
      padding:16px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
    }
    .pipeline-card .k{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fdcff;margin-bottom:10px;
    }
    .pipeline-card .v{
      font-size:34px;font-weight:1000;line-height:.92;letter-spacing:-.04em;
    }

    .board{
      margin-top:20px;
      display:grid;
      grid-template-columns:1.35fr .95fr;
      gap:18px;
      align-items:start;
    }

    .table-card,
    .side-card{
      border:1px solid var(--line);
      border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;
      box-shadow:0 14px 36px rgba(0,0,0,.26);
    }
    .table-card h2,
    .side-card h2{
      margin:0 0 12px;font-size:22px;font-weight:1000;letter-spacing:-.03em;
    }
    .muted{
      color:var(--muted);font-size:13px;line-height:1.55;
    }

    .table-wrap{
      margin-top:12px;
      overflow:auto;
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      background:rgba(255,255,255,.02);
    }
    table{
      width:100%;
      border-collapse:collapse;
      min-width:1080px;
    }
    th, td{
      padding:14px 12px;
      border-bottom:1px solid rgba(255,255,255,.08);
      text-align:left;
      vertical-align:top;
      font-size:13px;
    }
    th{
      color:#9eb8d6;
      text-transform:uppercase;
      font-size:11px;
      letter-spacing:.12em;
      background:rgba(255,255,255,.02);
      position:sticky;top:0;
      z-index:1;
    }
    tr:hover td{background:rgba(255,255,255,.02)}
    .lead-type{
      display:inline-flex;align-items:center;justify-content:center;
      padding:8px 10px;border-radius:999px;
      font-size:11px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);
    }
    .type-hospital{color:#0b1528;background:linear-gradient(135deg,var(--green),#7ef5c0)}
    .type-executive{color:#07101c;background:linear-gradient(135deg,var(--blue),var(--blue2))}
    .type-investor{color:#120c25;background:linear-gradient(135deg,var(--purple),#d3b4ff)}
    .row-title{font-weight:1000;font-size:14px;line-height:1.4}
    .row-sub{margin-top:4px;color:#acc0dd;font-size:12px;line-height:1.45}
    .pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:8px 10px;border-radius:999px;
      font-size:11px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);
    }
    .actions{
      display:flex;gap:8px;flex-wrap:wrap;
    }
    .mini-btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:8px 10px;border-radius:12px;font-size:11px;font-weight:1000;
      border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);color:var(--text);
      cursor:pointer;
    }

    .side-grid{
      display:grid;gap:14px;
    }
    .summary-line{
      display:flex;justify-content:space-between;gap:10px;
      border:1px solid rgba(255,255,255,.06);
      border-radius:16px;padding:12px;background:rgba(255,255,255,.03);
      font-size:13px;font-weight:900;color:#e3efff;
    }
    .queue-item{
      border:1px solid rgba(255,255,255,.06);
      border-radius:16px;padding:12px;background:rgba(255,255,255,.03);
      margin-bottom:10px;
    }
    .queue-title{font-size:14px;font-weight:1000;line-height:1.45}
    .queue-sub{margin-top:4px;font-size:12px;color:#a9bedb;line-height:1.45}

    .empty{
      border:1px dashed rgba(255,255,255,.12);
      border-radius:18px;
      padding:18px;
      font-size:14px;
      line-height:1.6;
      color:#a7bddc;
      background:rgba(255,255,255,.02);
      text-align:center;
    }

    @media (max-width:1280px){
      .hero-grid{grid-template-columns:1fr}
      .board{grid-template-columns:1fr}
      .pipeline-grid{grid-template-columns:repeat(2,1fr)}
      .mission-stats{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:900px){
      .filter-bar{grid-template-columns:1fr}
      .hero-mini-grid{grid-template-columns:repeat(2,1fr)}
      .pipeline-grid{grid-template-columns:1fr}
      .mission-stats{grid-template-columns:1fr}
    }
    @media (max-width:640px){
      .shell{padding:16px 10px 36px}
      .hero{padding:16px}
      .hero-copy h1{font-size:clamp(30px,11vw,50px)}
      .hero-actions{flex-direction:column}
      .btn{width:100%}
      .hero-mini-grid{grid-template-columns:1fr}
      .topbar-inner{padding:12px 10px}
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="brand-kicker">Operating layer · live review dashboard</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Hospital requests · executive walkthroughs · investor pipeline</div>
      </div>
      <div class="nav-links">
        <a href="/">Overview</a>
        <a href="/command-center">Command Center</a>
        <a href="/hospital-demo">Hospitals</a>
        <a href="/investor-intake">Investors</a>
        <a class="btn primary" href="/command-center">Back to Command Center</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">Live operating layer</div>
          <h1>Admin Review Dashboard</h1>
          <p>
            Review hospital demo requests, executive walkthrough requests, and investor intake submissions in one
            production-ready operating layer with live refresh, status control, search, filters, and pipeline visibility.
          </p>
          <div class="hero-actions">
            <a class="btn primary" href="/command-center">Open Command Center</a>
            <a class="btn secondary" href="/hospital-demo">Hospital Demo Form</a>
            <a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a>
            <a class="btn secondary" href="/investor-intake">Investor Intake</a>
          </div>
          <div class="hero-mini-grid">
            <div class="hero-mini">
              <div class="k">Hospital Leads</div>
              <div class="v" id="hero-hospital-count">0</div>
            </div>
            <div class="hero-mini">
              <div class="k">Executive Leads</div>
              <div class="v" id="hero-exec-count">0</div>
            </div>
            <div class="hero-mini">
              <div class="k">Investor Leads</div>
              <div class="v" id="hero-investor-count">0</div>
            </div>
            <div class="hero-mini">
              <div class="k">Last Updated</div>
              <div class="v" id="last-updated-label">—</div>
            </div>
          </div>
        </div>

        <div class="mission-panel">
          <div class="mission-header">
            <div>
              <div class="mission-title">Review Status</div>
              <div class="mission-sub">Newest leads first · instant status updates · auto-refresh active</div>
            </div>
            <div class="status-pill status-new">Live</div>
          </div>

          <div class="mission-stats">
            <div class="stat-card">
              <div class="k">Total Leads</div>
              <div class="v" id="total-leads">0</div>
              <div class="hint">All inbound submissions across the platform.</div>
            </div>
            <div class="stat-card">
              <div class="k">Open Leads</div>
              <div class="v" id="open-leads">0</div>
              <div class="hint">New and active leads still needing follow-up.</div>
            </div>
            <div class="stat-card">
              <div class="k">Contacted</div>
              <div class="v" id="contacted-leads">0</div>
              <div class="hint">Requests already engaged or moved forward.</div>
            </div>
            <div class="stat-card">
              <div class="k">Closed</div>
              <div class="v" id="closed-leads">0</div>
              <div class="hint">Completed, resolved, or archived opportunities.</div>
            </div>
          </div>
        </div>
      </div>

      <div class="filter-bar">
        <input id="search-box" class="field" type="text" placeholder="Search by name, organization, email, phone, notes">
        <select id="type-filter" class="field">
          <option value="">All lead types</option>
          <option value="hospital">Hospital</option>
          <option value="executive">Executive</option>
          <option value="investor">Investor</option>
        </select>
        <select id="status-filter" class="field">
          <option value="">All statuses</option>
          <option value="New">New</option>
          <option value="Contacted">Contacted</option>
          <option value="Closed">Closed</option>
        </select>
        <select id="stage-filter" class="field">
          <option value="">All investor stages</option>
          <option value="New">New</option>
          <option value="Interested">Interested</option>
          <option value="Due Diligence">Due Diligence</option>
          <option value="Follow-Up">Follow-Up</option>
          <option value="Closed">Closed</option>
        </select>
        <button id="refresh-btn" class="btn secondary" type="button">Refresh</button>
      </div>

      <div class="pipeline-grid">
        <div class="pipeline-card">
          <div class="k">Interested</div>
          <div class="v" id="stage-interested">0</div>
        </div>
        <div class="pipeline-card">
          <div class="k">Due Diligence</div>
          <div class="v" id="stage-dd">0</div>
        </div>
        <div class="pipeline-card">
          <div class="k">Follow-Up</div>
          <div class="v" id="stage-followup">0</div>
        </div>
        <div class="pipeline-card">
          <div class="k">Closed Investors</div>
          <div class="v" id="stage-closed">0</div>
        </div>
      </div>

      <div class="board">
        <div class="table-card">
          <h2>Live Lead Board</h2>
          <div class="muted">Newest and hottest leads first. Use status buttons without reloading the page.</div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Lead Type</th>
                  <th>Name / Organization</th>
                  <th>Contact</th>
                  <th>Notes / Intent</th>
                  <th>Status</th>
                  <th>Investor Stage</th>
                  <th>Submitted</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="lead-table-body"></tbody>
            </table>
          </div>

          <div id="lead-empty" class="empty" style="display:none;margin-top:12px;">
            No leads match the current filters.
          </div>
        </div>

        <div class="side-grid">
          <div class="side-card">
            <h2>Lead Mix</h2>
            <div class="summary-line"><span>Hospital</span><span id="mix-hospital">0</span></div>
            <div class="summary-line"><span>Executive</span><span id="mix-executive">0</span></div>
            <div class="summary-line"><span>Investor</span><span id="mix-investor">0</span></div>
          </div>

          <div class="side-card">
            <h2>Newest Opportunities</h2>
            <div id="newest-queue"></div>
          </div>

          <div class="side-card">
            <h2>Hottest Investor Pipeline</h2>
            <div id="investor-queue"></div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    let ALL_ROWS = [];

    function safe(v, fallback) {
      if (v === undefined || v === null || v === "") return fallback || "—";
      return v;
    }

    function escapeHtml(s) {
      return String(s || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function statusClass(status) {
      const s = String(status || "").toLowerCase();
      if (s === "contacted") return "status-contacted";
      if (s === "closed") return "status-closed";
      return "status-new";
    }

    function typeClass(kind) {
      const s = String(kind || "").toLowerCase();
      if (s === "executive") return "type-executive";
      if (s === "investor") return "type-investor";
      return "type-hospital";
    }

    function stagePill(stage) {
      const s = String(stage || "New");
      if (s === "Due Diligence") return '<span class="pill stage-dd">Due Diligence</span>';
      if (s === "Interested") return '<span class="pill status-contacted">Interested</span>';
      if (s === "Follow-Up") return '<span class="pill status-new">Follow-Up</span>';
      if (s === "Closed") return '<span class="pill status-closed">Closed</span>';
      return '<span class="pill status-new">New</span>';
    }

    function statusPill(status) {
      const s = safe(status, "New");
      return '<span class="pill ' + statusClass(s) + '">' + escapeHtml(s) + '</span>';
    }

    function leadTypePill(kind) {
      return '<span class="lead-type ' + typeClass(kind) + '">' + escapeHtml(kind || "hospital") + '</span>';
    }

    function submittedText(row) {
      return escapeHtml(safe(row.submitted_at, "—"));
    }

    function noteText(row) {
      return escapeHtml(
        row.message ||
        row.notes ||
        row.intent ||
        row.facility_type ||
        row.priority ||
        row.investor_type ||
        "No notes"
      );
    }

    function titleText(row) {
      const org = row.organization || row.company || row.facility || "";
      const role = row.role || row.title || "";
      let sub = [];
      if (org) sub.push(org);
      if (role) sub.push(role);
      const subline = sub.length ? '<div class="row-sub">' + escapeHtml(sub.join(" · ")) + '</div>' : "";
      return '<div class="row-title">' + escapeHtml(row.full_name || row.name || "Unnamed lead") + '</div>' + subline;
    }

    function contactText(row) {
      const bits = [];
      if (row.email) bits.push(escapeHtml(row.email));
      if (row.phone) bits.push(escapeHtml(row.phone));
      return bits.join("<br>") || "—";
    }

    function buildActionButtons(row) {
      let html = '';
      html += '<button class="mini-btn" data-action="status" data-kind="' + escapeHtml(row.kind) + '" data-id="' + escapeHtml(row.submitted_at) + '" data-value="New">New</button>';
      html += '<button class="mini-btn" data-action="status" data-kind="' + escapeHtml(row.kind) + '" data-id="' + escapeHtml(row.submitted_at) + '" data-value="Contacted">Contacted</button>';
      html += '<button class="mini-btn" data-action="status" data-kind="' + escapeHtml(row.kind) + '" data-id="' + escapeHtml(row.submitted_at) + '" data-value="Closed">Closed</button>';
      if (row.kind === "investor") {
        html += '<button class="mini-btn" data-action="stage" data-kind="investor" data-id="' + escapeHtml(row.submitted_at) + '" data-value="Interested">Interested</button>';
        html += '<button class="mini-btn" data-action="stage" data-kind="investor" data-id="' + escapeHtml(row.submitted_at) + '" data-value="Due Diligence">Due Diligence</button>';
        html += '<button class="mini-btn" data-action="stage" data-kind="investor" data-id="' + escapeHtml(row.submitted_at) + '" data-value="Follow-Up">Follow-Up</button>';
      }
      return '<div class="actions">' + html + '</div>';
    }

    function renderTable(rows) {
      const body = document.getElementById("lead-table-body");
      const empty = document.getElementById("lead-empty");

      if (!rows.length) {
        body.innerHTML = "";
        empty.style.display = "block";
        return;
      }

      empty.style.display = "none";
      body.innerHTML = rows.map(function(row) {
        return ''
          + '<tr>'
          + '<td>' + leadTypePill(row.kind) + '</td>'
          + '<td>' + titleText(row) + '</td>'
          + '<td>' + contactText(row) + '</td>'
          + '<td>' + noteText(row) + '</td>'
          + '<td>' + statusPill(row.status || "New") + '</td>'
          + '<td>' + (row.kind === "investor" ? stagePill(row.stage || "New") : '<span class="muted">—</span>') + '</td>'
          + '<td>' + submittedText(row) + '</td>'
          + '<td>' + buildActionButtons(row) + '</td>'
          + '</tr>';
      }).join("");
    }

    function renderQueues(rows) {
      const newest = document.getElementById("newest-queue");
      const investor = document.getElementById("investor-queue");

      const newestRows = rows.slice(0, 5);
      newest.innerHTML = newestRows.length ? newestRows.map(function(row) {
        return ''
          + '<div class="queue-item">'
          +   '<div class="queue-title">' + escapeHtml(row.full_name || row.name || "Unnamed lead") + '</div>'
          +   '<div class="queue-sub">' + escapeHtml(row.kind) + ' · ' + escapeHtml(row.status || "New") + ' · ' + escapeHtml(row.organization || row.company || row.investor_type || "Lead") + '</div>'
          + '</div>';
      }).join("") : '<div class="empty">No recent leads yet.</div>';

      const investorRows = rows.filter(function(r){ return r.kind === "investor"; }).slice(0, 5);
      investor.innerHTML = investorRows.length ? investorRows.map(function(row) {
        return ''
          + '<div class="queue-item">'
          +   '<div class="queue-title">' + escapeHtml(row.full_name || "Investor lead") + '</div>'
          +   '<div class="queue-sub">' + escapeHtml(row.stage || "New") + ' · ' + escapeHtml(row.investor_type || "Investor") + ' · ' + escapeHtml(row.organization || "Pipeline") + '</div>'
          + '</div>';
      }).join("") : '<div class="empty">No investor leads yet.</div>';
    }

    function applyCounts(allRows) {
      const hospital = allRows.filter(function(r){ return r.kind === "hospital"; }).length;
      const executive = allRows.filter(function(r){ return r.kind === "executive"; }).length;
      const investor = allRows.filter(function(r){ return r.kind === "investor"; }).length;
      const open = allRows.filter(function(r){ return String(r.status || "New") !== "Closed"; }).length;
      const contacted = allRows.filter(function(r){ return String(r.status || "") === "Contacted"; }).length;
      const closed = allRows.filter(function(r){ return String(r.status || "") === "Closed"; }).length;

      document.getElementById("hero-hospital-count").textContent = String(hospital);
      document.getElementById("hero-exec-count").textContent = String(executive);
      document.getElementById("hero-investor-count").textContent = String(investor);
      document.getElementById("total-leads").textContent = String(allRows.length);
      document.getElementById("open-leads").textContent = String(open);
      document.getElementById("contacted-leads").textContent = String(contacted);
      document.getElementById("closed-leads").textContent = String(closed);
      document.getElementById("mix-hospital").textContent = String(hospital);
      document.getElementById("mix-executive").textContent = String(executive);
      document.getElementById("mix-investor").textContent = String(investor);

      const lastUpdated = allRows.length ? (allRows[0].last_updated || allRows[0].submitted_at || "—") : "—";
      document.getElementById("last-updated-label").textContent = lastUpdated;

      const investors = allRows.filter(function(r){ return r.kind === "investor"; });
      document.getElementById("stage-interested").textContent = String(investors.filter(function(r){ return (r.stage || "New") === "Interested"; }).length);
      document.getElementById("stage-dd").textContent = String(investors.filter(function(r){ return (r.stage || "New") === "Due Diligence"; }).length);
      document.getElementById("stage-followup").textContent = String(investors.filter(function(r){ return (r.stage || "New") === "Follow-Up"; }).length);
      document.getElementById("stage-closed").textContent = String(investors.filter(function(r){ return (r.stage || "New") === "Closed"; }).length);
    }

    function filteredRows() {
      const q = String(document.getElementById("search-box").value || "").trim().toLowerCase();
      const type = document.getElementById("type-filter").value;
      const status = document.getElementById("status-filter").value;
      const stage = document.getElementById("stage-filter").value;

      return ALL_ROWS.filter(function(row) {
        const hay = [
          row.kind, row.full_name, row.name, row.organization, row.company, row.email,
          row.phone, row.message, row.notes, row.intent, row.role, row.title,
          row.facility_type, row.priority, row.investor_type, row.stage
        ].join(" ").toLowerCase();

        if (q && hay.indexOf(q) === -1) return false;
        if (type && row.kind !== type) return false;
        if (status && (row.status || "New") !== status) return false;
        if (stage && row.kind === "investor" && (row.stage || "New") !== stage) return false;
        if (stage && row.kind !== "investor") return false;
        return true;
      });
    }

    function rerender() {
      const rows = filteredRows();
      renderTable(rows);
      renderQueues(rows);
    }

    async function loadData() {
      const res = await fetch("/admin/review/data?refresh=" + Date.now(), { cache: "no-store" });
      const payload = await res.json();
      ALL_ROWS = Array.isArray(payload.rows) ? payload.rows : [];
      applyCounts(ALL_ROWS);
      rerender();
    }

    async function updateLead(kind, submittedAt, field, value) {
      const res = await fetch("/admin/review/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: kind,
          submitted_at: submittedAt,
          field: field,
          value: value
        })
      });
      if (!res.ok) return;
      await loadData();
    }

    document.addEventListener("click", async function(e) {
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      const action = btn.getAttribute("data-action");
      const kind = btn.getAttribute("data-kind");
      const id = btn.getAttribute("data-id");
      const value = btn.getAttribute("data-value");
      if (action === "status") {
        await updateLead(kind, id, "status", value);
      } else if (action === "stage") {
        await updateLead(kind, id, "stage", value);
      }
    });

    document.getElementById("search-box").addEventListener("input", rerender);
    document.getElementById("type-filter").addEventListener("change", rerender);
    document.getElementById("status-filter").addEventListener("change", rerender);
    document.getElementById("stage-filter").addEventListener("change", rerender);
    document.getElementById("refresh-btn").addEventListener("click", loadData);

    loadData();
    setInterval(loadData, 4000);
  </script>
</body>
</html>
"""


SIM_STATE = {
    "seed": 1,
    "last_generated_at": "",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir() -> Path:
    path = Path(os.getenv("ERA_DATA_DIR", "data")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except Exception:
            continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _status_norm(status: str | None, lead_type: str) -> str:
    raw = (status or "").strip().lower()
    if lead_type == "investor":
        mapping = {
            "new": "New",
            "interested": "Interested",
            "due diligence": "Due Diligence",
            "follow-up": "Follow-Up",
            "contacted": "Contacted",
            "closed": "Closed",
        }
        return mapping.get(raw, "New")
    if raw == "contacted":
        return "Contacted"
    if raw == "closed":
        return "Closed"
    return "New"


def _available_statuses(lead_type: str) -> list[str]:
    if lead_type == "investor":
        return ["New", "Interested", "Due Diligence", "Follow-Up", "Closed"]
    return ["New", "Contacted", "Closed"]


def _score_class(score: int) -> str:
    if score >= 80:
        return "hot"
    if score >= 60:
        return "warm"
    return "cool"


def _priority_tag(score: int, lead_type: str) -> str:
    if lead_type == "investor":
        if score >= 85:
            return "High Priority"
        if score >= 70:
            return "Qualified"
        return "Early Stage"
    if lead_type == "executive":
        if score >= 80:
            return "Strategic"
        if score >= 65:
            return "Pilot Ready"
        return "Review"
    if score >= 80:
        return "Urgent"
    if score >= 65:
        return "High Value"
    return "Standard"


def _lead_score(payload: dict[str, Any], lead_type: str) -> int:
    score = 40
    timeline = (payload.get("timeline") or "").lower()
    message = (payload.get("message") or "").lower()
    org = (payload.get("organization") or "").lower()

    if "immediate" in timeline:
        score += 28
    elif "30-60" in timeline:
        score += 18
    elif "quarter" in timeline:
        score += 10
    else:
        score += 4

    if lead_type == "hospital":
        facility_type = (payload.get("facility_type") or "").lower()
        if "health system" in facility_type:
            score += 22
        elif "hospital" in facility_type:
            score += 18
        elif "rpm" in facility_type:
            score += 16
        elif "clinic" in facility_type:
            score += 10
        if any(x in message for x in ["pilot", "integration", "deployment", "command center", "live", "clinical", "alerts"]):
            score += 14

    elif lead_type == "executive":
        priority = (payload.get("priority") or "").lower()
        if "enterprise" in priority:
            score += 20
        elif "strategic" in priority:
            score += 18
        elif "pilot" in priority:
            score += 16
        elif "operational" in priority:
            score += 14
        if any(x in message for x in ["budget", "review", "rollout", "leadership", "system", "hospital", "partnership"]):
            score += 12

    elif lead_type == "investor":
        investor_type = (payload.get("investor_type") or "").lower()
        check_size = (payload.get("check_size") or "").lower()
        if "healthcare vc" in investor_type:
            score += 24
        elif "strategic" in investor_type:
            score += 22
        elif "seed fund" in investor_type:
            score += 18
        elif "angel" in investor_type:
            score += 12
        elif "family office" in investor_type:
            score += 16

        if "250" in check_size or "500" in check_size or "1m" in check_size:
            score += 22
        elif "100" in check_size:
            score += 14
        elif "50" in check_size:
            score += 10

        if any(x in message for x in ["deck", "traction", "pilot", "hospital", "funding", "partnership", "due diligence", "customers"]):
            score += 14

    if any(x in org for x in ["hospital", "health", "medical", "care", "system", "clinic", "capital", "ventures"]):
        score += 6

    return max(1, min(100, score))


def _detail_html(payload: dict[str, Any], fields: list[str]) -> str:
    rows = []
    for key in fields:
        value = html.escape(str(payload.get(key, "") or ""))
        label = html.escape(key.replace("_", " ").title())
        rows.append(f"<div style='margin:8px 0'><strong>{label}:</strong> {value}</div>")
    return "".join(rows)


def _render_thank_you(message: str, details_html: str) -> str:
    out = THANK_YOU_HTML.replace("__MESSAGE__", html.escape(message))
    out = out.replace("__DETAILS__", details_html)
    return out


def _format_pretty_label(dt: str) -> str:
    if not dt:
        return "--"
    return dt.replace("T", " ").replace("+00:00", " UTC")


def _send_email(subject: str, plain_text: str, html_body: str | None, recipients: list[str]) -> None:
    password = os.getenv("EMAIL_PASSWORD", "").strip()
    if not password:
        print("Email send skipped: EMAIL_PASSWORD is not set", flush=True)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = INFO_EMAIL
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(plain_text, "plain"))
    if html_body:
        msg.attach(MIMEText(html_body, "html"))

    try:
        server = smtplib.SMTP("smtp.zoho.com", 587)
        server.starttls()
        server.login(INFO_EMAIL, password)
        server.sendmail(INFO_EMAIL, recipients, msg.as_string())
        server.quit()
    except Exception as e:
        print("Email send failed:", e, flush=True)


def _notification_recipients(lead_type: str) -> list[str]:
    primary = os.getenv("PRIMARY_LEAD_EMAIL", INFO_EMAIL).strip() or INFO_EMAIL
    hospital = os.getenv("HOSPITAL_LEAD_EMAIL", primary).strip() or primary
    executive = os.getenv("EXECUTIVE_LEAD_EMAIL", primary).strip() or primary
    investor = os.getenv("INVESTOR_LEAD_EMAIL", primary).strip() or primary
    route_map = {"hospital": hospital, "executive": executive, "investor": investor}
    recipients = {primary, FOUNDER_EMAIL, route_map.get(lead_type, primary)}
    return [r for r in recipients if r]


def _admin_notification_subject(lead_type: str, payload: dict[str, Any]) -> str:
    if lead_type == "hospital":
        return f"[Hospital Demo] {payload.get('organization') or payload.get('full_name') or 'New Request'}"
    if lead_type == "executive":
        return f"[Executive Walkthrough] {payload.get('organization') or payload.get('full_name') or 'New Request'}"
    return f"[Investor Intake] {payload.get('organization') or payload.get('full_name') or 'New Request'}"


def _admin_notification_html(lead_type: str, payload: dict[str, Any]) -> str:
    accent = "#5bd4ff" if lead_type != "investor" else "#ba8cff"
    fields = {
        "Full Name": payload.get("full_name", ""),
        "Organization": payload.get("organization", ""),
        "Role / Title": payload.get("role") or payload.get("title") or "",
        "Email": payload.get("email", ""),
        "Phone": payload.get("phone", ""),
        "Category": payload.get("facility_type") or payload.get("priority") or payload.get("investor_type") or "",
        "Timeline": payload.get("timeline", ""),
        "Lead Score": payload.get("lead_score", ""),
        "Priority Tag": payload.get("priority_tag", ""),
        "Submitted At": payload.get("submitted_at", ""),
        "Message": payload.get("message", ""),
    }

    field_html = "".join(
        f"<tr><td style='padding:10px 12px;border-bottom:1px solid #eef3ff14;color:#a8bddc;font-weight:700'>{html.escape(k)}</td>"
        f"<td style='padding:10px 12px;border-bottom:1px solid #eef3ff14;color:#eef4ff'>{html.escape(str(v))}</td></tr>"
        for k, v in fields.items()
    )

    return f"""
    <div style="font-family:Inter,Arial,sans-serif;background:#08111f;padding:24px;color:#eef4ff">
      <div style="max-width:760px;margin:0 auto;background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;overflow:hidden">
        <div style="padding:18px 22px;background:linear-gradient(135deg,{accent},#7aa2ff);color:#08111f;font-weight:1000;font-size:20px">
          Early Risk Alert AI · New {html.escape(lead_type.title())} Lead
        </div>
        <div style="padding:22px">
          <p style="margin:0 0 14px;color:#c6d7ef;line-height:1.6">
            A new {html.escape(lead_type)} request was submitted and is now available in admin review.
          </p>
          <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,.02);border-radius:14px;overflow:hidden">
            {field_html}
          </table>
          <p style="margin:18px 0 0;color:#9eb4d6;font-size:13px">
            Admin review: <a href="{PROD_BASE_URL}/admin/review" style="color:#8fd7ff">Open Admin Dashboard</a>
          </p>
        </div>
      </div>
    </div>
    """


def _auto_reply_subject(lead_type: str) -> str:
    mapping = {
        "hospital": "Early Risk Alert AI · Hospital Demo Request Received",
        "executive": "Early Risk Alert AI · Executive Walkthrough Request Received",
        "investor": "Early Risk Alert AI · Investor Intake Submission Received",
    }
    return mapping[lead_type]


def _auto_reply_html(lead_type: str, payload: dict[str, Any]) -> str:
    headline_map = {
        "hospital": "Your hospital demo request has been received.",
        "executive": "Your executive walkthrough request has been received.",
        "investor": "Your investor intake submission has been received.",
    }
    body_map = {
        "hospital": "We will review your request and follow up regarding scheduling, clinical use case fit, and the most relevant command-center views for your team.",
        "executive": "We will review your request and follow up regarding leadership priorities, evaluation goals, and the most relevant platform walkthrough for your team.",
        "investor": "We will review your submission and follow up regarding platform materials, timing, and next-step commercial discussion.",
    }

    return f"""
    <div style="font-family:Inter,Arial,sans-serif;background:#08111f;padding:24px;color:#eef4ff">
      <div style="max-width:720px;margin:0 auto;background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;overflow:hidden">
        <div style="padding:18px 22px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:1000;font-size:20px">
          Early Risk Alert AI
        </div>
        <div style="padding:24px">
          <h2 style="margin:0 0 12px;font-size:30px;line-height:1;letter-spacing:-.03em">{headline_map[lead_type]}</h2>
          <p style="margin:0 0 14px;color:#c6d7ef;line-height:1.7">Hello {html.escape(payload.get("full_name") or "there")},</p>
          <p style="margin:0 0 14px;color:#c6d7ef;line-height:1.7">{body_map[lead_type]}</p>
          <div style="margin-top:18px;padding:16px;border-radius:16px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08)">
            <div style="font-size:12px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#9fdcff;margin-bottom:10px">Request Summary</div>
            <div style="color:#eef4ff;line-height:1.7">
              <div><strong>Organization:</strong> {html.escape(payload.get("organization") or "")}</div>
              <div><strong>Timeline:</strong> {html.escape(payload.get("timeline") or "")}</div>
              <div><strong>Submitted:</strong> {html.escape(payload.get("submitted_at") or "")}</div>
            </div>
          </div>
          <p style="margin:18px 0 0;color:#c6d7ef;line-height:1.7">
            Thank you,<br>
            {html.escape(FOUNDER_NAME)}<br>
            {html.escape(FOUNDER_ROLE)}<br>
            <a href="mailto:{INFO_EMAIL}" style="color:#8fd7ff">{INFO_EMAIL}</a> ·
            <a href="tel:{BUSINESS_PHONE}" style="color:#8fd7ff">{BUSINESS_PHONE}</a>
          </p>
        </div>
      </div>
    </div>
    """


def _send_admin_notification(lead_type: str, payload: dict[str, Any]) -> None:
    recipients = _notification_recipients(lead_type)
    subject = _admin_notification_subject(lead_type, payload)
    plain = f"""
New {lead_type.title()} Lead

Name: {payload.get('full_name', '')}
Organization: {payload.get('organization', '')}
Role / Title: {payload.get('role') or payload.get('title') or ''}
Email: {payload.get('email', '')}
Phone: {payload.get('phone', '')}
Category: {payload.get('facility_type') or payload.get('priority') or payload.get('investor_type') or ''}
Timeline: {payload.get('timeline', '')}
Lead Score: {payload.get('lead_score', '')}
Priority Tag: {payload.get('priority_tag', '')}
Submitted At: {payload.get('submitted_at', '')}
Message: {payload.get('message', '')}
"""
    _send_email(subject, plain, _admin_notification_html(lead_type, payload), recipients)


def _send_auto_reply(lead_type: str, payload: dict[str, Any]) -> None:
    email = (payload.get("email") or "").strip()
    if not email:
        return
    subject = _auto_reply_subject(lead_type)
    plain = f"""
Hello {payload.get('full_name') or 'there'},

Your {lead_type} request has been received by Early Risk Alert AI.

Organization: {payload.get('organization', '')}
Timeline: {payload.get('timeline', '')}
Submitted At: {payload.get('submitted_at', '')}

We will follow up soon.

{FOUNDER_NAME}
{FOUNDER_ROLE}
{INFO_EMAIL}
{BUSINESS_PHONE}
"""
    _send_email(subject, plain, _auto_reply_html(lead_type, payload), [email])


def _investor_stage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "New": 0,
        "Interested": 0,
        "Due Diligence": 0,
        "Follow-Up": 0,
        "Contacted": 0,
        "Closed": 0,
    }
    for row in rows:
        status = _status_norm(str(row.get("status")), "investor")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _normalize_row(row: dict[str, Any], lead_type: str) -> dict[str, Any]:
    status = _status_norm(row.get("status"), lead_type)
    score = int(row.get("lead_score", 0) or 0)
    category = row.get("facility_type") or row.get("priority") or row.get("investor_type") or ""
    role_or_title = row.get("role") or row.get("title") or ""
    last_updated = row.get("last_updated") or row.get("submitted_at") or ""
    return {
        "lead_type": {"hospital": "Hospital Demo", "executive": "Executive Walkthrough", "investor": "Investor Intake"}[lead_type],
        "lead_type_key": lead_type,
        "submitted_at": str(row.get("submitted_at", "")),
        "last_updated": str(last_updated),
        "status": status,
        "lead_score": score,
        "priority_tag": row.get("priority_tag") or _priority_tag(score, lead_type),
        "full_name": str(row.get("full_name", "")),
        "organization": str(row.get("organization", "")),
        "role_or_title": str(role_or_title),
        "email": str(row.get("email", "")),
        "phone": str(row.get("phone", "")),
        "category": str(category),
        "timeline": str(row.get("timeline", "")),
        "message": str(row.get("message", "")),
        "available_statuses": _available_statuses(lead_type),
    }


def _base_patient_state() -> list[dict[str, Any]]:
    return [
        {
            "patient_id": "p101",
            "name": "Patient 1042",
            "heart_rate": 126.0,
            "spo2": 89.0,
            "bp_systolic": 164.0,
            "bp_diastolic": 98.0,
            "resp_rate": 28.0,
            "temp": 100.7,
            "trend": "deteriorating",
            "risk_score": 9.1,
            "status": "Critical",
            "confidence": 0.94,
            "recommended_action": "Immediate Intervention",
        },
        {
            "patient_id": "p102",
            "name": "Patient 2188",
            "heart_rate": 112.0,
            "spo2": 93.0,
            "bp_systolic": 148.0,
            "bp_diastolic": 90.0,
            "resp_rate": 24.0,
            "temp": 99.4,
            "trend": "watch",
            "risk_score": 8.1,
            "status": "High",
            "confidence": 0.88,
            "recommended_action": "Escalation Watch",
        },
        {
            "patient_id": "p103",
            "name": "Patient 3045",
            "heart_rate": 82.0,
            "spo2": 98.0,
            "bp_systolic": 122.0,
            "bp_diastolic": 78.0,
            "resp_rate": 18.0,
            "temp": 98.4,
            "trend": "recovering",
            "risk_score": 3.4,
            "status": "Stable",
            "confidence": 0.81,
            "recommended_action": "Routine Monitoring",
        },
        {
            "patient_id": "p104",
            "name": "Patient 4172",
            "heart_rate": 106.0,
            "spo2": 94.0,
            "bp_systolic": 142.0,
            "bp_diastolic": 88.0,
            "resp_rate": 22.0,
            "temp": 99.1,
            "trend": "watch",
            "risk_score": 7.6,
            "status": "High",
            "confidence": 0.86,
            "recommended_action": "Escalation Watch",
        },
    ]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _risk_from_vitals(p: dict[str, Any]) -> tuple[float, str, float, str]:
    hr = float(p["heart_rate"])
    spo2 = float(p["spo2"])
    sys = float(p["bp_systolic"])
    dia = float(p["bp_diastolic"])
    rr = float(p["resp_rate"])
    temp = float(p["temp"])

    risk = 0.0
    risk += max(0, hr - 90) * 0.035
    risk += max(0, 94 - spo2) * 0.75
    risk += max(0, sys - 140) * 0.02
    risk += max(0, dia - 90) * 0.03
    risk += max(0, rr - 20) * 0.12
    risk += max(0, temp - 99.0) * 0.7

    if p["trend"] == "deteriorating":
        risk += 1.2
    elif p["trend"] == "watch":
        risk += 0.6
    elif p["trend"] == "recovering":
        risk -= 0.4

    risk = _clamp(round(risk, 1), 0.8, 9.9)

    if risk >= 8.5:
        status = "Critical"
        action = "Immediate Intervention"
    elif risk >= 6.2:
        status = "High"
        action = "Escalation Watch"
    else:
        status = "Stable"
        action = "Routine Monitoring"

    confidence = _clamp(round(0.72 + min(risk, 9.5) / 25 + random.uniform(-0.03, 0.03), 2), 0.74, 0.98)
    return risk, status, confidence, action


def _mutate_patient(p: dict[str, Any]) -> dict[str, Any]:
    trend = p["trend"]

    if trend == "deteriorating":
        p["heart_rate"] = _clamp(float(p["heart_rate"]) + random.uniform(-1, 5), 88, 142)
        p["spo2"] = _clamp(float(p["spo2"]) + random.uniform(-2.2, 0.4), 84, 98)
        p["bp_systolic"] = _clamp(float(p["bp_systolic"]) + random.uniform(-2, 6), 118, 176)
        p["bp_diastolic"] = _clamp(float(p["bp_diastolic"]) + random.uniform(-2, 4), 70, 108)
        p["resp_rate"] = _clamp(float(p["resp_rate"]) + random.uniform(-1, 2.5), 16, 34)
        p["temp"] = _clamp(float(p["temp"]) + random.uniform(-0.1, 0.2), 97.8, 102.2)
    elif trend == "watch":
        p["heart_rate"] = _clamp(float(p["heart_rate"]) + random.uniform(-3, 3), 76, 126)
        p["spo2"] = _clamp(float(p["spo2"]) + random.uniform(-1.0, 1.0), 89, 99)
        p["bp_systolic"] = _clamp(float(p["bp_systolic"]) + random.uniform(-3, 3), 108, 160)
        p["bp_diastolic"] = _clamp(float(p["bp_diastolic"]) + random.uniform(-2, 2), 68, 98)
        p["resp_rate"] = _clamp(float(p["resp_rate"]) + random.uniform(-1.2, 1.2), 15, 28)
        p["temp"] = _clamp(float(p["temp"]) + random.uniform(-0.08, 0.08), 97.8, 100.6)
    else:
        p["heart_rate"] = _clamp(float(p["heart_rate"]) + random.uniform(-2, 2), 62, 96)
        p["spo2"] = _clamp(float(p["spo2"]) + random.uniform(-0.5, 0.5), 95, 100)
        p["bp_systolic"] = _clamp(float(p["bp_systolic"]) + random.uniform(-2, 2), 108, 132)
        p["bp_diastolic"] = _clamp(float(p["bp_diastolic"]) + random.uniform(-2, 2), 66, 84)
        p["resp_rate"] = _clamp(float(p["resp_rate"]) + random.uniform(-1, 1), 14, 22)
        p["temp"] = _clamp(float(p["temp"]) + random.uniform(-0.05, 0.05), 97.5, 99.2)

    risk, status, confidence, action = _risk_from_vitals(p)
    p["risk_score"] = risk
    p["status"] = status
    p["confidence"] = confidence
    p["recommended_action"] = action
    return p


def _simulated_snapshot() -> dict[str, Any]:
    SIM_STATE["seed"] += 1
    random.seed(SIM_STATE["seed"] + int(time.time() // 2))
    patients = [_mutate_patient(dict(p)) for p in _base_patient_state()]

    alerts: list[dict[str, Any]] = []
    for p in patients:
        sev = p["status"].lower()
        if p["status"] == "Critical":
            title = "Critical deterioration signal"
        elif p["status"] == "High":
            title = "High-priority escalation watch"
        else:
            title = "Stable recovery trend"
        alerts.append({
            "patient_id": p["patient_id"],
            "title": title,
            "alert_type": title,
            "severity": sev,
            "risk_score": float(p["risk_score"]),
            "confidence": float(p["confidence"]),
            "recommended_action": p["recommended_action"],
            "heart_rate": int(round(p["heart_rate"])),
            "spo2": int(round(p["spo2"])),
        })

    alerts.sort(key=lambda a: float(a["risk_score"]), reverse=True)

    critical_count = sum(1 for a in alerts if a["severity"] == "critical")
    open_alerts = sum(1 for a in alerts if a["severity"] in {"critical", "high"})
    avg_risk = round(sum(float(a["risk_score"]) for a in alerts) / len(alerts), 1)
    patients_with_alerts = sum(1 for a in alerts if a["severity"] in {"critical", "high"})
    events_last_hour = open_alerts * 3 + critical_count * 2 + 4
    focus_patient_id = alerts[0]["patient_id"] if alerts else "p101"

    SIM_STATE["last_generated_at"] = _utc_now_iso()
    return {
        "generated_at": SIM_STATE["last_generated_at"],
        "patients": patients,
        "alerts": alerts[:8],
        "summary": {
            "open_alerts": open_alerts,
            "critical_alerts": critical_count,
            "avg_risk_score": avg_risk,
            "patients_with_alerts": patients_with_alerts,
            "events_last_hour": events_last_hour,
            "focus_patient_id": focus_patient_id,
        },
    }


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
    
    def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _lead_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for item in _read_jsonl(hospital_file):
            item = dict(item)
            item["kind"] = "hospital"
            item["status"] = _status_norm(item.get("status"))
            item["stage"] = ""
            rows.append(item)

        for item in _read_jsonl(exec_file):
            item = dict(item)
            item["kind"] = "executive"
            item["status"] = _status_norm(item.get("status"))
            item["stage"] = ""
            rows.append(item)

        for item in _read_jsonl(investor_file):
            item = dict(item)
            item["kind"] = "investor"
            item["status"] = _status_norm(item.get("status"))
            item["stage"] = item.get("stage", "New") or "New"
            rows.append(item)

        rows.sort(key=lambda r: str(r.get("submitted_at", "")), reverse=True)
        return rows

    @app.get("/admin/review/data")
    def admin_review_data():
        return jsonify({"ok": True, "rows": _lead_rows()})

    @app.post("/admin/review/update")
    def admin_review_update():
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind", "")).strip().lower()
        submitted_at = str(payload.get("submitted_at", "")).strip()
        field = str(payload.get("field", "")).strip()
        value = str(payload.get("value", "")).strip()

    if not kind or not submitted_at or field not in {"status", "stage"}:
        return jsonify({"ok": False, "error": "invalid payload"}), 400

    path_map = {
        "hospital": hospital_file,
        "executive": exec_file,
        "investor": investor_file,
    }
    path = path_map.get(kind)
    if not path:
        return jsonify({"ok": False, "error": "invalid lead type"}), 400

    rows = _read_jsonl(path)
    updated = False

    for row in rows:
        if str(row.get("submitted_at", "")).strip() == submitted_at:
            if field == "status":
                row["status"] = _status_norm(value)
                if kind == "investor" and row["status"] == "Closed":
                    row["stage"] = "Closed"
            elif field == "stage" and kind == "investor":
                row["stage"] = value or "New"
                if row["stage"] == "Closed":
                    row["status"] = "Closed"
            row["last_updated"] = _utc_now_iso()
            updated = True
            break

    if not updated:
        return jsonify({"ok": False, "error": "lead not found"}), 404

    _write_jsonl_rows(path, rows)
    return jsonify({"ok": True})

@app.get("/admin/review")
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

            return render_template_string(
                _render_thank_you(
                    "Your hospital demo request was submitted successfully. The request is now live in admin review and ready for follow-up.",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "facility_type", "timeline", "lead_score", "priority_tag"]),
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
<div class="field"><label>What would you like to see in the demo?</label><textarea name="message"></textarea></div>
"""
        out = FORM_HTML.replace("__TITLE__", "Hospital Demo - Early Risk Alert AI")
        out = out.replace("__HEADING__", "Request Hospital Demo")
        out = out.replace("__COPY__", "Capture hospital interest, clinical use cases, and command-center demo requests.")
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
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "executive")
            _append_jsonl(exec_file, payload)
            _send_admin_notification("executive", payload)
            _send_auto_reply("executive", payload)

            return render_template_string(
                _render_thank_you(
                    "Your executive walkthrough request was submitted successfully. The request is now live in admin review and ready for follow-up.",
                    _detail_html(payload, ["full_name", "organization", "title", "email", "priority", "timeline", "lead_score", "priority_tag"]),
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
<div class="field"><label>Walkthrough Focus</label><textarea name="message"></textarea></div>
"""
        out = FORM_HTML.replace("__TITLE__", "Executive Walkthrough - Early Risk Alert AI")
        out = out.replace("__HEADING__", "Schedule Executive Walkthrough")
        out = out.replace("__COPY__", "Capture leadership-focused product reviews, pilot conversations, and strategic evaluation requests.")
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

            return render_template_string(
                _render_thank_you(
                    "Your investor intake was submitted successfully. The submission is now live in the investor pipeline and ready for follow-up.",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "investor_type", "check_size", "timeline", "lead_score", "priority_tag"]),
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
<div class="field"><label>Interest / Notes</label><textarea name="message" placeholder="What are you interested in learning more about?"></textarea></div>
"""
        out = FORM_HTML.replace("__TITLE__", "Investor Intake - Early Risk Alert AI")
        out = out.replace("__HEADING__", "Investor Intake Form")
        out = out.replace("__COPY__", "Capture investor interest, stage readiness, and commercial follow-up details directly from the platform.")
        out = out.replace("__FIELDS__", fields)
        out = out.replace("__BUTTON__", "Submit Investor Intake")
        return render_template_string(out)

    @app.get("/admin/api/data")
    def admin_api_data():
        raw = {
            "hospital": _read_jsonl(hospital_file),
            "executive": _read_jsonl(exec_file),
            "investor": _read_jsonl(investor_file),
        }

        merged: list[dict[str, Any]] = []
        for lead_type, data in raw.items():
            for row in data:
                merged.append(_normalize_row(row, lead_type))
        merged.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)

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
    def admin_export_csv():
        rows = []
        for lead_type, path in [("hospital", hospital_file), ("executive", exec_file), ("investor", investor_file)]:
            for row in _read_jsonl(path):
                rows.append(_normalize_row(row, lead_type))
        rows.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Lead Source",
            "Submitted At",
            "Last Updated",
            "Lead Status",
            "Lead Score",
            "Priority Tag",
            "Full Name",
            "Organization",
            "Role / Title",
            "Email Address",
            "Phone Number",
            "Category",
            "Timeline",
            "Message",
        ])
        for row in rows:
            writer.writerow([
                row["lead_type"],
                row["submitted_at"],
                row["last_updated"],
                row["status"],
                row["lead_score"],
                row["priority_tag"],
                row["full_name"],
                row["organization"],
                row["role_or_title"],
                row["email"],
                row["phone"],
                row["category"],
                row["timeline"],
                row["message"],
            ])

        mem = io.BytesIO()
        mem.write(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="early_risk_alert_pipeline_export.csv")

    @app.get("/api/v1/dashboard/overview")
    def dashboard_overview():
        snapshot = _simulated_snapshot()
        raw = {
            "hospital": _read_jsonl(hospital_file),
            "executive": _read_jsonl(exec_file),
            "investor": _read_jsonl(investor_file),
        }
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
        focus = next((r for r in snapshot["patients"] if r["patient_id"] == patient_id), snapshot["patients"][0] if snapshot["patients"] else {})
        return jsonify({
            "tenant_id": tenant_id,
            "generated_at": snapshot["generated_at"],
            "alerts": snapshot["alerts"],
            "focus_patient": focus,
            "patients": snapshot["patients"],
            "summary": snapshot["summary"],
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
                yield f"data: {json.dumps(snapshot)}\n\n"
                time.sleep(2)
        return Response(generate(), mimetype="text/event-stream")

    return app
