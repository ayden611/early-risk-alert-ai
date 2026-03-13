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


Use this as your new `COMMAND_CENTER_HTML` block exactly as pasted.

It’s paste-ready and matches your current `/api/command-center-stream` and `/api/v1/live-snapshot` setup.

COMMAND_CENTER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI · Hospital Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#040913;
      --bg2:#07101c;
      --panel:#0d1728;
      --panel2:#101d33;
      --panel3:#07111e;
      --line:rgba(255,255,255,.08);
      --line2:rgba(255,255,255,.05);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --cyan:#5bd4ff;
      --green:#38d39f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --purple:#ba8cff;
      --shadow:0 18px 48px rgba(0,0,0,.34);
      --r:20px;
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;background:
      radial-gradient(circle at top left, rgba(91,212,255,.06), transparent 20%),
      radial-gradient(circle at 88% 10%, rgba(122,162,255,.05), transparent 18%),
      linear-gradient(180deg,var(--bg),var(--bg2));
      color:var(--text);
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
    }
    a{text-decoration:none;color:inherit}
    .wrap{max-width:1680px;margin:0 auto;padding:18px}
    .card{
      border:1px solid var(--line);
      border-radius:22px;
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.05), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        var(--panel);
      box-shadow:var(--shadow);
    }

    .topbar{
      display:grid;
      grid-template-columns:1.2fr auto;
      gap:16px;
      align-items:center;
      padding:18px 20px;
      margin-bottom:16px;
    }
    .eyebrow{font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#9fdcff}
    .title{margin-top:8px;font-size:44px;font-weight:1000;line-height:.94;letter-spacing:-.05em}
    .subtitle{margin-top:10px;color:#c6d7ef;line-height:1.65;max-width:960px}
    .actions{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:13px 16px;border-radius:14px;font-size:14px;font-weight:1000;
      border:1px solid transparent;
    }
    .btn.primary{background:linear-gradient(135deg,var(--blue),var(--cyan));color:#08111f}
    .btn.secondary{background:rgba(255,255,255,.03);border-color:var(--line);color:var(--text)}

    .alarm-banner{
      margin-bottom:16px;
      border:1px solid rgba(255,255,255,.08);
      border-radius:18px;
      overflow:hidden;
      background:
        linear-gradient(90deg, rgba(255,102,125,.10), rgba(244,189,106,.08), rgba(91,212,255,.06)),
        rgba(8,16,29,.96);
    }
    .alarm-head{
      display:flex;align-items:center;justify-content:space-between;gap:12px;
      padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06)
    }
    .alarm-title{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#d8ecff}
    .alarm-live{
      display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#eaf8ff
    }
    .alarm-live::before{
      content:"";width:8px;height:8px;border-radius:999px;background:var(--green);
      box-shadow:0 0 0 0 rgba(56,211,159,.55);animation:pulse 1.8s infinite
    }
    @keyframes pulse{
      0%{box-shadow:0 0 0 0 rgba(56,211,159,.55)}
      70%{box-shadow:0 0 0 12px rgba(56,211,159,0)}
      100%{box-shadow:0 0 0 0 rgba(56,211,159,0)}
    }
    .alarm-track{overflow:hidden;white-space:nowrap;padding:14px 0}
    .alarm-move{display:inline-flex;gap:14px;padding-left:100%;animation:marquee 26s linear infinite}
    @keyframes marquee{from{transform:translateX(0)}to{transform:translateX(-100%)}}
    .alarm-pill{
      display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:999px;
      border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);
      font-size:13px;font-weight:900;color:#eef4ff
    }
    .dot{width:9px;height:9px;border-radius:999px;display:inline-block}
    .dot.critical{background:var(--red);box-shadow:0 0 10px rgba(255,102,125,.45)}
    .dot.high{background:var(--amber);box-shadow:0 0 10px rgba(244,189,106,.35)}
    .dot.stable{background:var(--green);box-shadow:0 0 10px rgba(56,211,159,.35)}

    .kpi-grid{
      display:grid;grid-template-columns:repeat(8,1fr);gap:12px;margin-bottom:16px
    }
    .kpi{
      padding:16px;border-radius:18px;border:1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        var(--panel2);
      box-shadow:var(--shadow);
    }
    .kpi .k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9fdcff}
    .kpi .v{margin-top:10px;font-size:32px;font-weight:1000;line-height:1}
    .kpi .s{margin-top:8px;color:#c6d7ef;font-size:12px;line-height:1.45}

    .layout{
      display:grid;grid-template-columns:1.55fr .85fr;gap:16px
    }

    .left-stack,.right-stack{display:grid;gap:16px}

    .section-pad{padding:18px}
    .section-head{
      display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px
    }
    .section-title{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9fdcff
    }
    .section-copy{color:#c6d7ef;line-height:1.6}

    .monitor-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
    .monitor{
      border:1px solid rgba(255,255,255,.08);
      border-radius:20px;
      padding:14px;
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.05), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01)),
        var(--panel3);
      box-shadow:inset 0 0 0 1px rgba(255,255,255,.02);
    }
    .monitor.critical{border-color:rgba(255,102,125,.26)}
    .monitor.high{border-color:rgba(244,189,106,.24)}
    .monitor.stable{border-color:rgba(56,211,159,.24)}
    .monitor-top{
      display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px
    }
    .monitor-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6}
    .monitor-name{margin-top:6px;font-size:20px;font-weight:1000;line-height:1}
    .monitor-sub{margin-top:6px;font-size:12px;color:#bed1ea}
    .badge{
      display:inline-flex;align-items:center;justify-content:center;min-width:98px;padding:8px 10px;border-radius:999px;
      font-size:12px;font-weight:900;letter-spacing:.05em;text-transform:uppercase
    }
    .badge.critical{background:rgba(255,102,125,.16);border:1px solid rgba(255,102,125,.28);color:#ffd8e0}
    .badge.high{background:rgba(244,189,106,.16);border:1px solid rgba(244,189,106,.28);color:#ffe5bb}
    .badge.stable{background:rgba(56,211,159,.16);border:1px solid rgba(56,211,159,.28);color:#d9ffec}

    .screen{
      position:relative;height:230px;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.06);background:#050b14;
    }
    .screen-grid{
      position:absolute;inset:0;
      background:
        linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
      background-size:22px 22px,22px 22px,auto;
      pointer-events:none;
    }
    .scanline{
      position:absolute;inset:0;
      background:linear-gradient(180deg, transparent 0%, rgba(91,212,255,.03) 48%, transparent 100%);
      animation:scan 5s linear infinite;
      pointer-events:none
    }
    @keyframes scan{
      0%{transform:translateY(-100%)}
      100%{transform:translateY(100%)}
    }
    .wave-svg,.spark-svg{position:absolute;inset:0;width:100%;height:100%}
    .wave-line{
      fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round;
      stroke-dasharray:18 8;animation:ecgMove 1.8s linear infinite;filter:drop-shadow(0 0 10px currentColor)
    }
    .wave-line.critical{stroke:var(--red);color:var(--red)}
    .wave-line.high{stroke:var(--amber);color:var(--amber)}
    .wave-line.stable{stroke:var(--green);color:var(--green)}
    @keyframes ecgMove{from{stroke-dashoffset:0}to{stroke-dashoffset:-52}}

    .screen-readouts{
      position:absolute;left:12px;right:12px;bottom:12px;display:grid;grid-template-columns:repeat(5,1fr);gap:8px
    }
    .readout{
      border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:10px;background:rgba(255,255,255,.03);backdrop-filter:blur(4px)
    }
    .readout .rk{display:block;font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .readout .rv{display:block;margin-top:6px;font-size:18px;line-height:1;font-weight:1000}

    .rail-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
    .rail-card{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:16px;background:rgba(255,255,255,.03)
    }
    .rail-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9fdcff}
    .rail-v{margin-top:10px;font-size:30px;font-weight:1000;line-height:1}
    .rail-s{margin-top:8px;color:#c6d7ef;font-size:13px;line-height:1.5}

    .feed-list,.queue-list,.board-list,.zone-list,.patient-list{display:grid;gap:10px;margin-top:14px}
    .feed-item,.queue-item,.board-item,.zone-item,.patient-row{
      padding:12px 14px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.03)
    }
    .feed-item .n,.queue-item .n,.board-item .n,.zone-item .n,.patient-row .n{
      font-size:14px;font-weight:1000;color:#eef4ff;display:flex;align-items:center;gap:10px
    }
    .feed-item .m,.queue-item .m,.board-item .m,.zone-item .m,.patient-row .m{
      font-size:13px;color:#bdd0ea;margin-top:6px;line-height:1.5
    }

    .patient-table{
      display:grid;grid-template-columns:1.2fr .7fr .7fr .7fr .7fr .9fr .9fr;gap:10px;align-items:center
    }
    .patient-head{
      padding:0 2px 8px;color:#9eb4d6;font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase
    }
    .spark{
      width:100%;height:34px;border-radius:10px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);overflow:hidden
    }
    .spark-path{fill:none;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round}
    .spark-path.hr{stroke:var(--cyan)}
    .spark-path.risk{stroke:var(--purple)}

    .zone-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .zone-item .tag{
      display:inline-flex;align-items:center;justify-content:center;padding:7px 10px;border-radius:999px;
      font-size:11px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;margin-top:10px
    }
    .tag.good{background:rgba(56,211,159,.14);border:1px solid rgba(56,211,159,.26);color:#d9ffec}
    .tag.watch{background:rgba(244,189,106,.14);border:1px solid rgba(244,189,106,.26);color:#ffe6bd}
    .tag.hot{background:rgba(255,102,125,.14);border:1px solid rgba(255,102,125,.26);color:#ffd8e0}

    .status-bar{
      display:flex;gap:10px;flex-wrap:wrap;margin-top:12px
    }
    .status-chip{
      display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;font-size:12px;font-weight:900;
      letter-spacing:.06em;text-transform:uppercase;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:#eef4ff
    }
    .status-chip .s-dot{width:9px;height:9px;border-radius:999px}
    .status-chip .s-dot.red{background:var(--red)}
    .status-chip .s-dot.amber{background:var(--amber)}
    .status-chip .s-dot.green{background:var(--green)}
    .status-chip .s-dot.blue{background:var(--blue)}

    @media (max-width:1440px){
      .kpi-grid{grid-template-columns:repeat(4,1fr)}
      .layout{grid-template-columns:1fr}
    }
    @media (max-width:960px){
      .kpi-grid{grid-template-columns:repeat(2,1fr)}
      .monitor-grid{grid-template-columns:1fr}
      .zone-grid,.rail-grid{grid-template-columns:1fr}
      .proof-grid{grid-template-columns:1fr}
      .patient-table{grid-template-columns:1fr}
      .screen-readouts{grid-template-columns:repeat(2,1fr)}
      .topbar{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card topbar">
      <div>
        <div class="eyebrow">AI Hospital Mission Control</div>
        <div class="title">Early Risk Alert AI · Hospital Command Center</div>
        <div class="subtitle">
          ICU-grade waveform monitors, escalation queue, census tracking, rapid response view, clinical feed,
          care coordination board, unit status visibility, and real-time predictive telemetry in one flagship command center.
        </div>
      </div>
      <div class="actions">
        <a class="btn primary" href="/">Return Home</a>
        <a class="btn secondary" href="/admin/review">Admin Review</a>
      </div>
    </div>

    <div class="alarm-banner">
      <div class="alarm-head">
        <div class="alarm-title">Live Clinical Activity Ticker</div>
        <div class="alarm-live">Predictive Stream</div>
      </div>
      <div class="alarm-track">
        <div class="alarm-move" id="ticker">
          <div class="alarm-pill"><span class="dot critical"></span> Critical deterioration signal surfaced</div>
          <div class="alarm-pill"><span class="dot high"></span> High-risk escalation routed to rapid response</div>
          <div class="alarm-pill"><span class="dot stable"></span> Stable patient trend confirmed</div>
          <div class="alarm-pill"><span class="dot critical"></span> ICU watchlist refreshed</div>
          <div class="alarm-pill"><span class="dot high"></span> Mission control telemetry synced</div>
        </div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi"><div class="k">Open Alerts</div><div class="v" id="k-open">0</div><div class="s">Active AI-flagged alerts</div></div>
      <div class="kpi"><div class="k">Critical Alerts</div><div class="v" id="k-critical">0</div><div class="s">Immediate intervention cases</div></div>
      <div class="kpi"><div class="k">Avg Risk Score</div><div class="v" id="k-risk">0.0</div><div class="s">Average flagged-patient severity</div></div>
      <div class="kpi"><div class="k">Patients With Alerts</div><div class="v" id="k-patients">0</div><div class="s">Patients under active watch</div></div>
      <div class="kpi"><div class="k">Events Last Hour</div><div class="v" id="k-events">0</div><div class="s">Simulated clinical events</div></div>
      <div class="kpi"><div class="k">Rapid Response</div><div class="v" id="k-rrt">0</div><div class="s">Cases needing escalation</div></div>
      <div class="kpi"><div class="k">Focus Patient</div><div class="v" id="k-focus">p101</div><div class="s">Highest-priority patient ID</div></div>
      <div class="kpi"><div class="k">Engine State</div><div class="v">LIVE</div><div class="s">Predictive monitoring online</div></div>
    </div>

    <div class="layout">
      <div class="left-stack">
        <div class="card section-pad">
          <div class="section-head">
            <div>
              <div class="section-title">Waveform monitor wall</div>
              <div class="section-copy">Four active patient monitors with animated ECG waveforms, vitals, AI risk score, and escalation state.</div>
            </div>
          </div>

          <div class="monitor-grid">
            <div class="monitor critical" id="card-m1">
              <div class="monitor-top">
                <div>
                  <div class="monitor-k">ICU Bed A</div>
                  <div class="monitor-name" id="m1-name">p101 · Patient 1042</div>
                  <div class="monitor-sub" id="m1-sub">Trend: deteriorating · Confidence 0.94</div>
                </div>
                <div class="badge critical" id="m1-status">Critical</div>
              </div>
              <div class="screen">
                <div class="screen-grid"></div>
                <div class="scanline"></div>
                <svg class="wave-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                  <path id="m1-wave" class="wave-line critical"
                    d="M0 130 L40 130 L70 130 L90 130 L110 130 L130 130 L150 130 L170 130 L190 130 L210 130 L225 110 L240 145 L255 60 L270 185 L285 128 L300 130 L340 130 L380 130 L420 130 L435 112 L450 145 L465 65 L480 185 L495 128 L510 130 L550 130 L590 130 L630 130 L645 112 L660 145 L675 58 L690 190 L705 128 L720 130 L760 130 L800 130 L840 130 L855 114 L870 142 L885 63 L900 186 L915 128 L930 130 L970 130 L1010 130 L1050 130 L1065 112 L1080 144 L1095 60 L1110 184 L1125 128 L1140 130 L1200 130" />
                </svg>
                <div class="screen-readouts">
                  <div class="readout"><span class="rk">HR</span><span class="rv" id="m1-hr">128</span></div>
                  <div class="readout"><span class="rk">SpO₂</span><span class="rv" id="m1-spo2">89</span></div>
                  <div class="readout"><span class="rk">BP</span><span class="rv" id="m1-bp">164/98</span></div>
                  <div class="readout"><span class="rk">RR</span><span class="rv" id="m1-rr">28</span></div>
                  <div class="readout"><span class="rk">Risk</span><span class="rv" id="m1-risk">9.1</span></div>
                </div>
              </div>
            </div>

            <div class="monitor high" id="card-m2">
              <div class="monitor-top">
                <div>
                  <div class="monitor-k">ICU Bed B</div>
                  <div class="monitor-name" id="m2-name">p102 · Patient 2188</div>
                  <div class="monitor-sub" id="m2-sub">Trend: watch · Confidence 0.88</div>
                </div>
                <div class="badge high" id="m2-status">High</div>
              </div>
              <div class="screen">
                <div class="screen-grid"></div>
                <div class="scanline"></div>
                <svg class="wave-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                  <path id="m2-wave" class="wave-line high"
                    d="M0 132 L50 132 L90 132 L130 132 L170 132 L210 132 L225 120 L240 140 L255 92 L270 165 L285 132 L300 132 L350 132 L400 132 L450 132 L465 118 L480 140 L495 96 L510 162 L525 132 L540 132 L590 132 L640 132 L690 132 L705 120 L720 138 L735 98 L750 162 L765 132 L780 132 L830 132 L880 132 L930 132 L945 120 L960 140 L975 95 L990 164 L1005 132 L1020 132 L1080 132 L1140 132 L1200 132" />
                </svg>
                <div class="screen-readouts">
                  <div class="readout"><span class="rk">HR</span><span class="rv" id="m2-hr">112</span></div>
                  <div class="readout"><span class="rk">SpO₂</span><span class="rv" id="m2-spo2">93</span></div>
                  <div class="readout"><span class="rk">BP</span><span class="rv" id="m2-bp">148/90</span></div>
                  <div class="readout"><span class="rk">RR</span><span class="rv" id="m2-rr">24</span></div>
                  <div class="readout"><span class="rk">Risk</span><span class="rv" id="m2-risk">8.1</span></div>
                </div>
              </div>
            </div>

            <div class="monitor stable" id="card-m3">
              <div class="monitor-top">
                <div>
                  <div class="monitor-k">Stepdown Bed C</div>
                  <div class="monitor-name" id="m3-name">p103 · Patient 3045</div>
                  <div class="monitor-sub" id="m3-sub">Trend: recovering · Confidence 0.81</div>
                </div>
                <div class="badge stable" id="m3-status">Stable</div>
              </div>
              <div class="screen">
                <div class="screen-grid"></div>
                <div class="scanline"></div>
                <svg class="wave-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                  <path id="m3-wave" class="wave-line stable"
                    d="M0 140 L40 140 L80 140 L120 140 L160 140 L200 140 L218 134 L236 146 L254 112 L272 156 L290 140 L330 140 L370 140 L410 140 L450 140 L468 134 L486 146 L504 112 L522 156 L540 140 L580 140 L620 140 L660 140 L700 140 L718 134 L736 146 L754 112 L772 156 L790 140 L830 140 L870 140 L910 140 L950 140 L968 134 L986 146 L1004 112 L1022 156 L1040 140 L1080 140 L1120 140 L1200 140" />
                </svg>
                <div class="screen-readouts">
                  <div class="readout"><span class="rk">HR</span><span class="rv" id="m3-hr">82</span></div>
                  <div class="readout"><span class="rk">SpO₂</span><span class="rv" id="m3-spo2">98</span></div>
                  <div class="readout"><span class="rk">BP</span><span class="rv" id="m3-bp">122/78</span></div>
                  <div class="readout"><span class="rk">RR</span><span class="rv" id="m3-rr">18</span></div>
                  <div class="readout"><span class="rk">Risk</span><span class="rv" id="m3-risk">3.4</span></div>
                </div>
              </div>
            </div>

            <div class="monitor high" id="card-m4">
              <div class="monitor-top">
                <div>
                  <div class="monitor-k">Remote Watch D</div>
                  <div class="monitor-name" id="m4-name">p104 · Patient 4172</div>
                  <div class="monitor-sub" id="m4-sub">Trend: watch · Confidence 0.86</div>
                </div>
                <div class="badge high" id="m4-status">High</div>
              </div>
              <div class="screen">
                <div class="screen-grid"></div>
                <div class="scanline"></div>
                <svg class="wave-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                  <path id="m4-wave" class="wave-line high"
                    d="M0 136 L40 136 L80 136 L120 136 L160 136 L200 136 L220 126 L240 142 L260 100 L280 162 L300 136 L340 136 L380 136 L420 136 L460 136 L480 126 L500 142 L520 100 L540 162 L560 136 L600 136 L640 136 L680 136 L720 136 L740 126 L760 142 L780 100 L800 162 L820 136 L860 136 L900 136 L940 136 L980 136 L1000 126 L1020 142 L1040 100 L1060 162 L1080 136 L1120 136 L1160 136 L1200 136" />
                </svg>
                <div class="screen-readouts">
                  <div class="readout"><span class="rk">HR</span><span class="rv" id="m4-hr">105</span></div>
                  <div class="readout"><span class="rk">SpO₂</span><span class="rv" id="m4-spo2">94</span></div>
                  <div class="readout"><span class="rk">BP</span><span class="rv" id="m4-bp">142/88</span></div>
                  <div class="readout"><span class="rk">RR</span><span class="rv" id="m4-rr">22</span></div>
                  <div class="readout"><span class="rk">Risk</span><span class="rv" id="m4-risk">7.6</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="card section-pad">
          <div class="section-head">
            <div>
              <div class="section-title">Care coordination board</div>
              <div class="section-copy">Patient census, trend lines, and care-team view for command-center coordination.</div>
            </div>
          </div>

          <div class="patient-table patient-head">
            <div>Patient</div>
            <div>Status</div>
            <div>Risk</div>
            <div>HR</div>
            <div>SpO₂</div>
            <div>HR Trend</div>
            <div>Risk Trend</div>
          </div>
          <div class="patient-list" id="patient-list">
            <div class="patient-row">
              <div class="patient-table">
                <div class="n">Waiting for patient stream</div>
                <div class="m">--</div>
                <div class="m">--</div>
                <div class="m">--</div>
                <div class="m">--</div>
                <div class="spark"></div>
                <div class="spark"></div>
              </div>
            </div>
          </div>

          <div class="status-bar">
            <div class="status-chip"><span class="s-dot red"></span>Rapid Response Queue</div>
            <div class="status-chip"><span class="s-dot amber"></span>Escalation Watch</div>
            <div class="status-chip"><span class="s-dot green"></span>Routine Monitoring</div>
            <div class="status-chip"><span class="s-dot blue"></span>Command Center Online</div>
          </div>
        </div>

        <div class="card section-pad">
          <div class="section-head">
            <div>
              <div class="section-title">Unit operations board</div>
              <div class="section-copy">A hospital-style unit overview showing which areas are stable, on watch, or escalated.</div>
            </div>
          </div>
          <div class="zone-grid" id="zone-grid">
            <div class="zone-item"><div class="n">ICU North</div><div class="m">Monitoring bed occupancy, deterioration watch, and escalation queue.</div><div class="tag watch">Watch</div></div>
            <div class="zone-item"><div class="n">Stepdown</div><div class="m">Post-acute stabilization and recovery oversight.</div><div class="tag good">Stable</div></div>
            <div class="zone-item"><div class="n">ED Overflow</div><div class="m">Rapid triage visibility and escalation backup.</div><div class="tag hot">Elevated</div></div>
            <div class="zone-item"><div class="n">Remote Watch</div><div class="m">Cross-site predictive monitoring and triage.</div><div class="tag watch">Watch</div></div>
          </div>
        </div>
      </div>

      <div class="right-stack">
        <div class="card section-pad">
          <div class="section-head">
            <div>
              <div class="section-title">Mission control metrics</div>
              <div class="section-copy">House-wide monitoring metrics and operational state.</div>
            </div>
          </div>
          <div class="rail-grid">
            <div class="rail-card"><div class="rail-k">Engine State</div><div class="rail-v">LIVE</div><div class="rail-s">Predictive engine online</div></div>
            <div class="rail-card"><div class="rail-k">Shift</div><div class="rail-v" id="rail-shift">Day</div><div class="rail-s">Live command-center coverage window</div></div>
            <div class="rail-card"><div class="rail-k">Clock</div><div class="rail-v" id="rail-clock">--:--</div><div class="rail-s">Local command-center time</div></div>
            <div class="rail-card"><div class="rail-k">Focus Patient</div><div class="rail-v" id="rail-focus">p101</div><div class="rail-s">Highest-priority watch target</div></div>
            <div class="rail-card"><div class="rail-k">RRT Queue</div><div class="rail-v" id="rail-rrt">0</div><div class="rail-s">Rapid response activations</div></div>
            <div class="rail-card"><div class="rail-k">Census</div><div class="rail-v" id="rail-census">4</div><div class="rail-s">Visible monitored patients</div></div>
          </div>
        </div>

        <div class="card section-pad">
          <div class="section-title">Live alert feed</div>
          <div class="section-copy">Newest alerts appear first with severity, patient ID, recommended action, and AI confidence.</div>
          <div class="feed-list" id="feed-list">
            <div class="feed-item">
              <div class="n"><span class="dot stable"></span>Waiting for predictive stream</div>
              <div class="m">The command center will populate as simulated patient signals arrive.</div>
            </div>
          </div>
        </div>

        <div class="card section-pad">
          <div class="section-title">Rapid response queue</div>
          <div class="section-copy">Operational queue for critical and high-risk escalations.</div>
          <div class="queue-list" id="queue-list">
            <div class="queue-item">
              <div class="n"><span class="dot high"></span>No current escalations</div>
              <div class="m">The rapid response queue will populate from the predictive engine.</div>
            </div>
          </div>
        </div>

        <div class="card section-pad">
          <div class="section-title">Disposition and transport board</div>
          <div class="section-copy">Hospital-style workflow summary for transfer, observation, and intervention planning.</div>
          <div class="board-list" id="board-list">
            <div class="board-item">
              <div class="n"><span class="dot stable"></span>Awaiting workflow updates</div>
              <div class="m">Transfer, stepdown, observation, and intervention states will appear here.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const historyStore = {};

    function setClock() {
      const d = new Date();
      const hh = d.getHours().toString().padStart(2, "0");
      const mm = d.getMinutes().toString().padStart(2, "0");
      document.getElementById("rail-clock").textContent = hh + ":" + mm;
      document.getElementById("rail-shift").textContent = d.getHours() >= 19 || d.getHours() < 7 ? "Night" : "Day";
    }
    setClock();
    setInterval(setClock, 1000);

    function wavePath(status) {
      if (status === "critical") {
        return "M0 130 L40 130 L70 130 L90 130 L110 130 L130 130 L150 130 L170 130 L190 130 L210 130 L225 110 L240 145 L255 60 L270 185 L285 128 L300 130 L340 130 L380 130 L420 130 L435 112 L450 145 L465 65 L480 185 L495 128 L510 130 L550 130 L590 130 L630 130 L645 112 L660 145 L675 58 L690 190 L705 128 L720 130 L760 130 L800 130 L840 130 L855 114 L870 142 L885 63 L900 186 L915 128 L930 130 L970 130 L1010 130 L1050 130 L1065 112 L1080 144 L1095 60 L1110 184 L1125 128 L1140 130 L1200 130";
      }
      if (status === "high") {
        return "M0 132 L50 132 L90 132 L130 132 L170 132 L210 132 L225 120 L240 140 L255 92 L270 165 L285 132 L300 132 L350 132 L400 132 L450 132 L465 118 L480 140 L495 96 L510 162 L525 132 L540 132 L590 132 L640 132 L690 132 L705 120 L720 138 L735 98 L750 162 L765 132 L780 132 L830 132 L880 132 L930 132 L945 120 L960 140 L975 95 L990 164 L1005 132 L1020 132 L1080 132 L1140 132 L1200 132";
      }
      return "M0 140 L40 140 L80 140 L120 140 L160 140 L200 140 L218 134 L236 146 L254 112 L272 156 L290 140 L330 140 L370 140 L410 140 L450 140 L468 134 L486 146 L504 112 L522 156 L540 140 L580 140 L620 140 L660 140 L700 140 L718 134 L736 146 L754 112 L772 156 L790 140 L830 140 L870 140 L910 140 L950 140 L968 134 L986 146 L1004 112 L1022 156 L1040 140 L1080 140 L1120 140 L1200 140";
    }

    function clsFromStatus(status) {
      const s = String(status || "").toLowerCase();
      return s === "critical" ? "critical" : (s === "high" ? "high" : "stable");
    }

    function keepHistory(patient) {
      if (!historyStore[patient.patient_id]) {
        historyStore[patient.patient_id] = { hr: [], risk: [] };
      }
      const h = historyStore[patient.patient_id];
      h.hr.push(Number(patient.heart_rate || 0));
      h.risk.push(Number(patient.risk_score || 0));
      if (h.hr.length > 24) h.hr.shift();
      if (h.risk.length > 24) h.risk.shift();
    }

    function sparkPath(values, width, height, low, high) {
      if (!values.length) return "";
      const range = Math.max(1, high - low);
      return values.map((v, i) => {
        const x = (i / Math.max(1, values.length - 1)) * width;
        const y = height - ((v - low) / range) * height;
        return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ");
    }

    function sparkSvg(values, cls, low, high) {
      const path = sparkPath(values, 180, 34, low, high);
      return `
        <svg class="spark-svg" viewBox="0 0 180 34" preserveAspectRatio="none">
          <path class="spark-path ${cls}" d="${path}"></path>
        </svg>
      `;
    }

    function applyMonitor(slot, patient, label) {
      keepHistory(patient);
      const cls = clsFromStatus(patient.status);
      document.getElementById(slot + "-name").textContent = patient.patient_id + " · " + patient.name;
      document.getElementById(slot + "-sub").textContent =
        "Trend: " + patient.trend + " · Confidence " + Number(patient.confidence || 0).toFixed(2) +
        " · " + patient.recommended_action;
      const badge = document.getElementById(slot + "-status");
      badge.textContent = patient.status;
      badge.className = "badge " + cls;
      document.getElementById(slot + "-hr").textContent = Math.round(patient.heart_rate);
      document.getElementById(slot + "-spo2").textContent = Math.round(patient.spo2);
      document.getElementById(slot + "-bp").textContent = Math.round(patient.bp_systolic) + "/" + Math.round(patient.bp_diastolic);
      document.getElementById(slot + "-rr").textContent = Math.round(patient.resp_rate);
      document.getElementById(slot + "-risk").textContent = Number(patient.risk_score).toFixed(1);
      document.getElementById(slot + "-wave").setAttribute("d", wavePath(cls));
      document.getElementById(slot + "-wave").setAttribute("class", "wave-line " + cls);
      document.getElementById("card-" + slot).className = "monitor " + cls;
    }

    function renderFeed(alerts) {
      const feed = document.getElementById("feed-list");
      if (!alerts.length) {
        feed.innerHTML = '<div class="feed-item"><div class="n"><span class="dot stable"></span>No active alerts</div><div class="m">The predictive engine is online and waiting for new events.</div></div>';
        return;
      }
      feed.innerHTML = alerts.slice(0, 6).map(a => {
        const sev = clsFromStatus(a.severity);
        return `
          <div class="feed-item">
            <div class="n"><span class="dot ${sev}"></span>${a.title} · ${a.patient_id}</div>
            <div class="m">Severity: ${a.severity} · Risk ${Number(a.risk_score).toFixed(1)} · ${a.recommended_action} · Confidence ${Number(a.confidence).toFixed(2)}</div>
          </div>
        `;
      }).join("");
    }

    function renderQueue(alerts) {
      const queue = document.getElementById("queue-list");
      const urgent = alerts.filter(a => ["critical","high"].includes(clsFromStatus(a.severity)));
      if (!urgent.length) {
        queue.innerHTML = '<div class="queue-item"><div class="n"><span class="dot stable"></span>No current escalations</div><div class="m">The rapid response queue will populate from the predictive engine.</div></div>';
        return;
      }
      queue.innerHTML = urgent.slice(0, 5).map((a, idx) => {
        const sev = clsFromStatus(a.severity);
        const team = sev === "critical" ? "Rapid Response Team" : "Charge Nurse Review";
        return `
          <div class="queue-item">
            <div class="n"><span class="dot ${sev}"></span>Queue ${idx + 1} · ${a.patient_id} · ${a.severity}</div>
            <div class="m">${team} · ${a.recommended_action} · AI Confidence ${Number(a.confidence).toFixed(2)}</div>
          </div>
        `;
      }).join("");
    }

    function renderBoard(patients) {
      const board = document.getElementById("board-list");
      board.innerHTML = patients.slice(0, 4).map((p, idx) => {
        const cls = clsFromStatus(p.status);
        let disp = "Continue Observation";
        if (cls === "critical") disp = "Immediate Intervention / ICU Review";
        else if (cls === "high") disp = "Escalation Watch / Bedside Reassessment";
        else if (p.trend === "recovering") disp = "Stepdown Readiness Review";
        return `
          <div class="board-item">
            <div class="n"><span class="dot ${cls}"></span>${p.patient_id} · ${p.name}</div>
            <div class="m">Disposition: ${disp} · Trend: ${p.trend} · Temp ${Number(p.temp).toFixed(1)}°F</div>
          </div>
        `;
      }).join("");
    }

    function renderTicker(alerts) {
      const ticker = document.getElementById("ticker");
      if (!alerts.length) return;
      ticker.innerHTML = alerts.slice(0, 6).map(a => {
        const sev = clsFromStatus(a.severity);
        return `<div class="alarm-pill"><span class="dot ${sev}"></span>${a.title} · ${a.patient_id} · Risk ${Number(a.risk_score).toFixed(1)} · ${a.recommended_action}</div>`;
      }).join("");
    }

    function renderZones(patients) {
      const grid = document.getElementById("zone-grid");
      const critical = patients.filter(p => clsFromStatus(p.status) === "critical").length;
      const high = patients.filter(p => clsFromStatus(p.status) === "high").length;

      const zones = [
        { name: "ICU North", desc: critical ? critical + " critical case(s) under intervention watch." : "No critical cases currently assigned.", tag: critical ? "Elevated" : "Stable", cls: critical ? "hot" : "good" },
        { name: "Stepdown", desc: patients.filter(p => p.trend === "recovering").length + " recovering patient(s) under stabilization review.", tag: "Stable", cls: "good" },
        { name: "ED Overflow", desc: high ? high + " high-risk case(s) flagged for reassessment." : "No high-risk overflow cases right now.", tag: high ? "Watch" : "Stable", cls: high ? "watch" : "good" },
        { name: "Remote Watch", desc: patients.length + " monitored patient(s) visible to mission control.", tag: "Watch", cls: "watch" }
      ];

      grid.innerHTML = zones.map(z => `
        <div class="zone-item">
          <div class="n">${z.name}</div>
          <div class="m">${z.desc}</div>
          <div class="tag ${z.cls}">${z.tag}</div>
        </div>
      `).join("");
    }

    function renderPatientList(patients) {
      const wrap = document.getElementById("patient-list");
      wrap.innerHTML = patients.map(p => {
        keepHistory(p);
        const cls = clsFromStatus(p.status);
        const h = historyStore[p.patient_id];
        return `
          <div class="patient-row">
            <div class="patient-table">
              <div>
                <div class="n"><span class="dot ${cls}"></span>${p.patient_id} · ${p.name}</div>
                <div class="m">${p.recommended_action}</div>
              </div>
              <div class="m">${p.status}</div>
              <div class="m">${Number(p.risk_score).toFixed(1)}</div>
              <div class="m">${Math.round(p.heart_rate)}</div>
              <div class="m">${Math.round(p.spo2)}%</div>
              <div class="spark">${sparkSvg(h.hr, "hr", 60, 145)}</div>
              <div class="spark">${sparkSvg(h.risk, "risk", 0, 10)}</div>
            </div>
          </div>
        `;
      }).join("");
    }

    function refreshOverviewFromSummary(summary) {
      document.getElementById("k-open").textContent = summary.open_alerts ?? 0;
      document.getElementById("k-critical").textContent = summary.critical_alerts ?? 0;
      document.getElementById("k-risk").textContent = Number(summary.avg_risk_score ?? 0).toFixed(1);
      document.getElementById("k-patients").textContent = summary.patients_with_alerts ?? 0;
      document.getElementById("k-events").textContent = summary.events_last_hour ?? 0;
      document.getElementById("k-focus").textContent = summary.focus_patient_id || "p101";
      document.getElementById("k-rrt").textContent = summary.critical_alerts ?? 0;
      document.getElementById("rail-focus").textContent = summary.focus_patient_id || "p101";
      document.getElementById("rail-rrt").textContent = summary.critical_alerts ?? 0;
    }

    const es = new EventSource("/api/command-center-stream");
    es.onmessage = function(evt) {
      const payload = JSON.parse(evt.data || "{}");
      const patients = payload.patients || [];
      const alerts = payload.alerts || [];
      const summary = payload.summary || {};

      if (patients[0]) applyMonitor("m1", patients[0], "ICU Bed A");
      if (patients[1]) applyMonitor("m2", patients[1], "ICU Bed B");
      if (patients[2]) applyMonitor("m3", patients[2], "Stepdown Bed C");
      if (patients[3]) applyMonitor("m4", patients[3], "Remote Watch D");

      renderFeed(alerts);
      renderQueue(alerts);
      renderBoard(patients);
      renderTicker(alerts);
      renderZones(patients);
      renderPatientList(patients);
      refreshOverviewFromSummary(summary);
      document.getElementById("rail-census").textContent = patients.length;
    };

    async function refreshFallback() {
      try {
        const r = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), { cache: "no-store" });
        if (!r.ok) return;
        const payload = await r.json();
        renderFeed(payload.alerts || []);
        renderQueue(payload.alerts || []);
        renderBoard(payload.patients || []);
        renderTicker(payload.alerts || []);
        renderZones(payload.patients || []);
        renderPatientList(payload.patients || []);
        refreshOverviewFromSummary(payload.summary || {});
      } catch (e) {
        console.error(e);
      }
    }
    refreshFallback();
    setInterval(refreshFallback, 5000);
  </script>
</body>
</html>
"""


ADMIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin Review - Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --line:rgba(255,255,255,.08);
      --text:#edf4ff;
      --muted:#bdd0ec;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --amber:#f7be68;
      --red:#ff667d;
      --purple:#ba8cff;
      --shadow:0 18px 50px rgba(0,0,0,.24);
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:var(--text)}
    .wrap{max-width:1480px;margin:0 auto;padding:36px 18px 60px}
    .card{
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.10), transparent 24%),
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(16,26,45,.96), rgba(11,18,31,.98));
      border:1px solid var(--line);border-radius:24px;padding:30px;box-shadow:var(--shadow)
    }
    h1{font-size:clamp(40px,4vw,56px);line-height:.98;margin:0 0 14px;letter-spacing:-.045em}
    h2{margin:0 0 10px;font-size:32px;letter-spacing:-.03em}
    p{color:var(--muted);line-height:1.7}
    .btn{display:inline-flex;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;font-weight:1000}
    .toolbar{margin-top:18px;display:grid;grid-template-columns:1.2fr .8fr .8fr .8fr auto;gap:12px;align-items:end}
    .field{display:grid;gap:8px}
    .field label{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6}
    .field input,.field select{width:100%;padding:13px 14px;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:#eef4ff;font:inherit}
    .admin-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:20px 0 24px}
    .admin-kpi{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.06);
      border-radius:20px;padding:24px;min-height:154px;display:flex;flex-direction:column;justify-content:space-between;
    }
    .admin-kpi .k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .admin-kpi .v{font-size:44px;font-weight:1000;line-height:1;margin-top:14px}
    .admin-kpi .hint{font-size:13px;color:#c6d7ef;line-height:1.5}
    .status-legend{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
    .status-chip{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:#eef4ff}
    .status-chip .s-dot{width:9px;height:9px;border-radius:999px}
    .status-chip.new .s-dot{background:#7aa2ff}
    .status-chip.contacted .s-dot{background:#f4bd6a}
    .status-chip.closed .s-dot{background:#38d39f}
    .status-chip.interested .s-dot{background:#ba8cff}
    .status-chip.dd .s-dot{background:#ff667d}
    .last-updated{font-size:12px;color:#9eb4d6;font-weight:900;letter-spacing:.08em;text-transform:uppercase}
    .table-wrap{overflow:auto;border:1px solid rgba(255,255,255,.06);border-radius:18px;background:rgba(255,255,255,.02)}
    table{width:100%;border-collapse:collapse;font-size:14px;min-width:1280px}
    th,td{padding:14px 12px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;vertical-align:top}
    th{color:#9eb4d6;text-transform:uppercase;font-size:12px;letter-spacing:.12em;background:rgba(255,255,255,.02);position:sticky;top:0}
    tr:hover td{background:rgba(255,255,255,.02)}
    .empty{padding:18px;color:#c9daf0}
    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;min-width:108px;padding:9px 12px;border-radius:999px;font-size:12px;font-weight:1000
    }
    .status-new{background:rgba(91,212,255,.16);border:1px solid rgba(91,212,255,.28);color:#d6f4ff}
    .status-contacted{background:rgba(247,190,104,.16);border:1px solid rgba(247,190,104,.28);color:#ffe3b2}
    .status-closed{background:rgba(56,211,159,.16);border:1px solid rgba(56,211,159,.28);color:#d3ffe8}
    .status-interested{background:rgba(186,140,255,.16);border:1px solid rgba(186,140,255,.28);color:#f0e4ff}
    .status-due-diligence{background:rgba(255,102,125,.16);border:1px solid rgba(255,102,125,.28);color:#ffd8e0}
    .status-follow-up{background:rgba(122,162,255,.16);border:1px solid rgba(122,162,255,.28);color:#d6e5ff}
    .score{display:inline-flex;align-items:center;justify-content:center;min-width:64px;padding:8px 10px;border-radius:999px;background:rgba(122,162,255,.14);border:1px solid rgba(122,162,255,.22);font-weight:1000}
    .score.hot{background:rgba(255,102,125,.16);border-color:rgba(255,102,125,.28);color:#ffd8df}
    .score.warm{background:rgba(244,189,106,.16);border-color:rgba(244,189,106,.28);color:#ffe5bb}
    .score.cool{background:rgba(56,211,159,.16);border-color:rgba(56,211,159,.26);color:#d9ffec}
    .action-row{display:flex;gap:8px;flex-wrap:wrap}
    .mini-btn{display:inline-flex;align-items:center;justify-content:center;padding:9px 12px;border-radius:12px;background:#111b2f;border:1px solid rgba(255,255,255,.08);color:#eef4ff;font-size:12px;font-weight:1000;cursor:pointer}
    .meta-stack{display:grid;gap:6px}
    .meta-item{font-size:13px;color:#c6d7ef;line-height:1.5}
    .priority-tag{display:inline-flex;align-items:center;justify-content:center;padding:8px 10px;border-radius:999px;background:rgba(186,140,255,.14);border:1px solid rgba(186,140,255,.22);font-size:12px;font-weight:1000;color:#f0e4ff}
    @media (max-width:1200px){
      .toolbar{grid-template-columns:1fr 1fr;align-items:stretch}
      .admin-grid{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:760px){
      .admin-grid{grid-template-columns:1fr}
      .toolbar{grid-template-columns:1fr}
      .wrap{padding:14px 10px 48px}
      .card{padding:16px}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Admin Review</h1>
      <p>Live request operations, investor pipeline visibility, instant status updates, search, filtering, lead scoring, and export-ready follow-up flow.</p>
      <p><a class="btn" href="/admin/export.csv">Download Full Export</a> <a class="btn" href="/">Return Home</a></p>

      <div class="toolbar">
        <div class="field">
          <label>Search</label>
          <input id="admin-search" placeholder="Search by name, organization, email">
        </div>
        <div class="field">
          <label>Lead Type</label>
          <select id="admin-type">
            <option value="all">All</option>
            <option value="hospital">Hospital</option>
            <option value="executive">Executive</option>
            <option value="investor">Investor</option>
          </select>
        </div>
        <div class="field">
          <label>Status</label>
          <select id="admin-status">
            <option value="all">All</option>
            <option value="New">New</option>
            <option value="Contacted">Contacted</option>
            <option value="Closed">Closed</option>
            <option value="Interested">Interested</option>
            <option value="Due Diligence">Due Diligence</option>
            <option value="Follow-Up">Follow-Up</option>
          </select>
        </div>
        <div class="field">
          <label>Sort</label>
          <select id="admin-sort">
            <option value="newest">Newest First</option>
            <option value="score">Hottest Leads</option>
          </select>
        </div>
        <div class="field">
          <label>&nbsp;</label>
          <button class="mini-btn" id="admin-refresh-btn" type="button">Refresh</button>
        </div>
      </div>

      <div class="admin-grid">
        <div class="admin-kpi"><div class="k">Hospital Demo Requests</div><div class="v" id="kpi-hospital">0</div><div class="hint">Clinical buyer and operator pipeline</div></div>
        <div class="admin-kpi"><div class="k">Executive Walkthroughs</div><div class="v" id="kpi-exec">0</div><div class="hint">Leadership-facing product review requests</div></div>
        <div class="admin-kpi"><div class="k">Investor Intake</div><div class="v" id="kpi-investor">0</div><div class="hint">Commercial and financing pipeline</div></div>
        <div class="admin-kpi"><div class="k">Open Leads</div><div class="v" id="kpi-open">0</div><div class="hint">All active requests not yet closed</div></div>
        <div class="admin-kpi"><div class="k">Interested / DD</div><div class="v" id="kpi-dd">0</div><div class="hint">Investor pipeline progression</div></div>
        <div class="admin-kpi"><div class="k">Last Updated</div><div class="v" style="font-size:24px" id="kpi-updated">--</div><div class="hint">Live admin refresh timestamp</div></div>
      </div>

      <div class="status-legend">
        <div class="status-chip new"><span class="s-dot"></span>New</div>
        <div class="status-chip contacted"><span class="s-dot"></span>Contacted</div>
        <div class="status-chip closed"><span class="s-dot"></span>Closed</div>
        <div class="status-chip interested"><span class="s-dot"></span>Interested</div>
        <div class="status-chip dd"><span class="s-dot"></span>Due Diligence</div>
      </div>

      <div class="last-updated" id="admin-last-updated">Last Updated --</div>

      <div class="table-wrap" style="margin-top:14px">
        <table id="admin-table">
          <thead>
            <tr>
              <th>Lead Type</th>
              <th>Submitted</th>
              <th>Status</th>
              <th>Lead Score</th>
              <th>Priority</th>
              <th>Name</th>
              <th>Organization</th>
              <th>Role / Title</th>
              <th>Email</th>
              <th>Phone</th>
              <th>Category</th>
              <th>Timeline</th>
              <th>Message</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="admin-table-body">
            <tr><td colspan="14" class="empty">Loading live admin data...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let adminRows = [];

    function statusClass(value) {
      if (value === "Contacted") return "status-contacted";
      if (value === "Closed") return "status-closed";
      if (value === "Interested") return "status-interested";
      if (value === "Due Diligence") return "status-due-diligence";
      if (value === "Follow-Up") return "status-follow-up";
      return "status-new";
    }

    function scoreClass(score) {
      if (score >= 80) return "hot";
      if (score >= 60) return "warm";
      return "cool";
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function applyFilters() {
      const query = document.getElementById("admin-search").value.toLowerCase().trim();
      const type = document.getElementById("admin-type").value;
      const status = document.getElementById("admin-status").value;
      const sort = document.getElementById("admin-sort").value;

      let rows = adminRows.filter(function(row) {
        const haystack = [
          row.full_name, row.organization, row.email, row.role_or_title, row.message, row.lead_type
        ].join(" ").toLowerCase();
        const matchesQuery = !query || haystack.includes(query);
        const matchesType = type === "all" || row.lead_type_key === type;
        const matchesStatus = status === "all" || row.status === status;
        return matchesQuery && matchesType && matchesStatus;
      });

      if (sort === "score") {
        rows.sort(function(a, b) { return (b.lead_score || 0) - (a.lead_score || 0); });
      } else {
        rows.sort(function(a, b) { return String(b.submitted_at).localeCompare(String(a.submitted_at)); });
      }

      const body = document.getElementById("admin-table-body");
      if (!rows.length) {
        body.innerHTML = '<tr><td colspan="14" class="empty">No matching results.</td></tr>';
        return;
      }

      body.innerHTML = rows.map(function(row) {
        return `
          <tr>
            <td>${escapeHtml(row.lead_type)}</td>
            <td>
              <div class="meta-stack">
                <div class="meta-item">${escapeHtml(row.submitted_at)}</div>
                <div class="meta-item">Updated: ${escapeHtml(row.last_updated || row.submitted_at)}</div>
              </div>
            </td>
            <td><span class="status-pill ${statusClass(row.status)}">${escapeHtml(row.status)}</span></td>
            <td><span class="score ${scoreClass(row.lead_score || 0)}">${escapeHtml(row.lead_score || 0)}</span></td>
            <td><span class="priority-tag">${escapeHtml(row.priority_tag || "Standard")}</span></td>
            <td>${escapeHtml(row.full_name)}</td>
            <td>${escapeHtml(row.organization)}</td>
            <td>${escapeHtml(row.role_or_title)}</td>
            <td>${escapeHtml(row.email)}</td>
            <td>${escapeHtml(row.phone)}</td>
            <td>${escapeHtml(row.category)}</td>
            <td>${escapeHtml(row.timeline)}</td>
            <td>${escapeHtml(row.message)}</td>
            <td>
              <div class="action-row">
                ${row.available_statuses.map(function(s) {
                  return `<button class="mini-btn" type="button" onclick="updateLeadStatus('${escapeHtml(row.lead_type_key)}','${escapeHtml(row.submitted_at)}','${escapeHtml(s)}')">${escapeHtml(s)}</button>`;
                }).join("")}
              </div>
            </td>
          </tr>
        `;
      }).join("");
    }

    async function updateLeadStatus(type, submittedAt, status) {
      try {
        const res = await fetch("/admin/api/status", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({lead_type: type, submitted_at: submittedAt, status: status})
        });
        if (res.ok) {
          await refreshAdminData();
        }
      } catch (e) {
        console.error("status update failed", e);
      }
    }

    async function refreshAdminData() {
      try {
        const res = await fetch("/admin/api/data?refresh=" + Date.now(), { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          adminRows = data.rows || [];
          document.getElementById("kpi-hospital").textContent = data.summary.hospital_count || 0;
          document.getElementById("kpi-exec").textContent = data.summary.executive_count || 0;
          document.getElementById("kpi-investor").textContent = data.summary.investor_count || 0;
          document.getElementById("kpi-open").textContent = data.summary.open_count || 0;
          document.getElementById("kpi-dd").textContent = (data.summary.investor_stages?.Interested || 0) + (data.summary.investor_stages?.["Due Diligence"] || 0);
          document.getElementById("kpi-updated").textContent = data.summary.last_updated_label || "--";
          document.getElementById("admin-last-updated").textContent = "Last Updated " + (data.summary.last_updated || "--");
          applyFilters();
        }
      } catch (e) {
        console.error("admin data refresh failed", e);
      }
    }

    document.getElementById("admin-search").addEventListener("input", applyFilters);
    document.getElementById("admin-type").addEventListener("change", applyFilters);
    document.getElementById("admin-status").addEventListener("change", applyFilters);
    document.getElementById("admin-sort").addEventListener("change", applyFilters);
    document.getElementById("admin-refresh-btn").addEventListener("click", refreshAdminData);

    refreshAdminData();
    setInterval(refreshAdminData, 5000);
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

    @app.get("/")
    def home():
        return render_template_string(HOME_HTML)

    @app.get("/command-center")
    def command_center():
        return render_template_string(COMMAND_CENTER_HTML)

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
