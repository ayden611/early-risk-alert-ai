from __future__ import annotations

import html
import json
import os
import random
import re
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template_string, request


INFO_EMAIL = "info@earlyriskalertai.com"
FOUNDER_EMAIL = os.getenv("FOUNDER_EMAIL", "milton.munroe@earlyriskalertai.com")
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder & AI Systems Engineer"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/z4SbeYwwm7k"


MAIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --bg2:#0c1425;
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
      --shadow:0 20px 60px rgba(0,0,0,.30);
      --radius:24px;
      --max:1360px;
    }
    
.monitor-grid{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:14px;
  margin-top:18px;
  margin-bottom:18px;
}
.monitor-panel{
  border:1px solid rgba(255,255,255,.10);
  border-radius:20px;
  padding:16px;
  background:
    radial-gradient(circle at top right, rgba(91,212,255,.08), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
    #08101d;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.02), 0 12px 30px rgba(0,0,0,.22);
}
.monitor-panel.critical{border-color:rgba(255,107,107,.26)}
.monitor-panel.warning{border-color:rgba(247,190,104,.24)}
.monitor-panel.stable{border-color:rgba(91,211,141,.24)}
.monitor-top{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:12px;
}
.monitor-label{
  font-size:11px;
  font-weight:900;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:#9eb4d6;
}
.monitor-name{
  margin-top:6px;
  font-size:20px;
  font-weight:1000;
  line-height:1;
  color:#eef4ff;
}
.monitor-status{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  min-width:84px;
  padding:8px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:900;
  letter-spacing:.05em;
  text-transform:uppercase;
}
.status-critical{
  color:#ffd8d8;
  background:rgba(255,107,107,.16);
  border:1px solid rgba(255,107,107,.26);
}
.status-warning{
  color:#ffe7bf;
  background:rgba(247,190,104,.16);
  border:1px solid rgba(247,190,104,.24);
}
.status-stable{
  color:#dfffea;
  background:rgba(91,211,141,.14);
  border:1px solid rgba(91,211,141,.24);
}
.wave-wrap{
  height:122px;
  border-radius:16px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,.06);
  background:
    linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px),
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
  background-size:20px 20px, 20px 20px, auto;
}
.wave-svg{
  width:100%;
  height:100%;
}
.wave-line{
  fill:none;
  stroke-width:3.5;
  stroke-linecap:round;
  stroke-linejoin:round;
  filter:drop-shadow(0 0 8px currentColor);
}
.critical-line{
  stroke:#ff6b6b;
  color:#ff6b6b;
}
.warning-line{
  stroke:#f7be68;
  color:#f7be68;
}
.stable-line{
  stroke:#5bd38d;
  color:#5bd38d;
}
.monitor-metrics{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px;
  margin-top:14px;
}
.metric-box{
  border:1px solid rgba(255,255,255,.08);
  border-radius:14px;
  padding:12px 10px;
  background:rgba(255,255,255,.03);
}
.metric-k{
  display:block;
  font-size:11px;
  letter-spacing:.10em;
  text-transform:uppercase;
  color:#9eb4d6;
  font-weight:900;
}
.metric-v{
  display:block;
  margin-top:8px;
  font-size:22px;
  line-height:1;
  font-weight:1000;
  color:#eef4ff;
}
@media (max-width:1100px){
  .monitor-grid{grid-template-columns:1fr}
}
@media (max-width:760px){
  .monitor-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
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
    a{text-decoration:none;color:inherit}
    img{max-width:100%;display:block}
    .shell{max-width:var(--max);margin:0 auto;padding:22px 16px 56px}
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
    .brand-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8cd7ef;
    }
    .brand-title{
      font-size:clamp(26px,3vw,40px);font-weight:1000;line-height:.95;letter-spacing:-.05em;
    }
    .brand-sub{
      font-size:14px;color:var(--muted);font-weight:800;
    }
    .nav-links{
      display:flex;align-items:center;gap:16px;flex-wrap:wrap;
    }
    .nav-links a{font-size:14px;font-weight:900}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:13px 18px;border-radius:16px;font-size:14px;font-weight:900;cursor:pointer;
      border:1px solid transparent;
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, opacity .18s ease;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#07101c;box-shadow:0 12px 30px rgba(91,212,255,.22);
    }
    .btn.secondary{
      background:rgba(255,255,255,.04);
      color:var(--text);border-color:var(--line);
    }
    .hero{
      position:relative;overflow:hidden;border:1px solid var(--line);border-radius:32px;
      box-shadow:var(--shadow);
      background:
        linear-gradient(180deg, rgba(10,18,31,.26), rgba(7,16,28,.78)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
      min-height:720px;
    }
    .hero::before{
      content:"";position:absolute;inset:0;
      background:
        radial-gradient(circle at 50% 10%, rgba(91,212,255,.24), transparent 18%),
        linear-gradient(110deg, transparent 35%, rgba(255,255,255,.06) 50%, transparent 65%);
      animation:heroSweep 9s linear infinite;
      pointer-events:none;
    }
    @keyframes heroSweep{
      0%{transform:translateX(-40px)}
      50%{transform:translateX(40px)}
      100%{transform:translateX(-40px)}
    }
    .hero-inner{
      position:relative;z-index:2;min-height:720px;display:flex;align-items:flex-end;padding:36px;
    }
    .hero-grid{
      width:100%;display:grid;grid-template-columns:1.08fr .92fr;gap:18px;align-items:end;
    }
    .glass{
      border:1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      border-radius:28px;box-shadow:0 16px 42px rgba(0,0,0,.24);backdrop-filter:blur(14px);
    }
    .hero-copy{padding:28px}
    .hero-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px;
    }
    .hero-copy h1{
      margin:0 0 14px;font-size:clamp(40px,6vw,82px);line-height:.92;letter-spacing:-.06em;font-weight:1000;max-width:760px;text-wrap:balance;
    }
    .hero-copy p{
      margin:0;color:#d0ddf0;font-size:18px;line-height:1.68;max-width:760px;
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}
    .hero-mini-grid{margin-top:20px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .hero-mini{
      border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
    }
    .hero-mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:8px;
    }
    .hero-mini .v{font-size:16px;font-weight:1000;line-height:1.1}
    .demo-card{overflow:hidden}
    .demo-stage{
      position:relative;aspect-ratio:16/9;min-height:360px;
      background:
        linear-gradient(180deg, rgba(0,0,0,.12), rgba(0,0,0,.40)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
    }
    .demo-badge{
      position:absolute;top:18px;left:18px;z-index:3;padding:9px 13px;border-radius:999px;
      background:rgba(7,16,28,.62);border:1px solid var(--line);color:#eaf3ff;
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;backdrop-filter:blur(12px);
    }
    .demo-play{
      position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:3;
    }
    .demo-play-btn{
      width:96px;height:96px;border-radius:50%;border:1px solid rgba(255,255,255,.18);
      background:rgba(7,16,28,.58);backdrop-filter:blur(12px);
      box-shadow:0 20px 50px rgba(0,0,0,.32), 0 0 34px rgba(91,212,255,.18);
      display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative;
    }
    .demo-play-btn::before{
      content:"";position:absolute;inset:-14px;border-radius:50%;
      border:1px solid rgba(91,212,255,.20);animation:ring 2.8s ease-out infinite;
    }
    @keyframes ring{
      0%{transform:scale(.7);opacity:1}
      100%{transform:scale(1.35);opacity:0}
    }
    .demo-play-btn svg{width:34px;height:34px;margin-left:5px;fill:#eef4ff}
    .demo-caption{
      position:absolute;left:0;right:0;bottom:0;z-index:2;padding:22px;
      background:linear-gradient(180deg, transparent, rgba(7,16,28,.82));
    }
    .demo-caption h3{
      margin:0 0 8px;font-size:30px;line-height:1;font-weight:1000;letter-spacing:-.04em;
    }
    .demo-caption p{margin:0;color:#d4e2f3;font-size:15px;line-height:1.6}
    .demo-bottom{padding:18px;border-top:1px solid var(--line)}
    .demo-note{color:#bdd0eb;font-size:14px;line-height:1.55;margin-bottom:14px}
    .demo-btns{display:flex;gap:10px;flex-wrap:wrap}
    .route-grid{
      margin-top:18px;
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:14px;
    }
    .route-card{
      border:1px solid var(--line);
      border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:20px;
    }
    .route-label{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;
    }
    .route-card h3{
      margin:10px 0 10px;font-size:30px;line-height:1;font-weight:1000;letter-spacing:-.04em;
    }
    .route-card p{
      margin:0;color:#d0ddf0;font-size:15px;line-height:1.64;
    }
    .route-card .route-actions{
      margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;
    }
    .section{
      margin-top:22px;
      border:1px solid var(--line);
      border-radius:28px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:26px;
    }
    .section-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px;
    }
    .section h2{
      margin:0 0 10px;font-size:clamp(34px,5vw,58px);line-height:.95;letter-spacing:-.05em;font-weight:1000;
    }
    .section p{
      margin:0;color:#d0ddf0;font-size:16px;line-height:1.7;max-width:980px;
    }
    .list{
      margin-top:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
    }
    .mini{
      border:1px solid var(--line2);border-radius:18px;padding:16px;background:rgba(255,255,255,.03);
    }
    .mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px;
    }
    .mini .v{
      display:block;font-size:20px;font-weight:1000;line-height:1.1;letter-spacing:-.03em;
    }
    .mini p{
      margin:10px 0 0;color:#c7d8ef;font-size:14px;line-height:1.6;
    }
    .live-band{
      display:grid;grid-template-columns:1.15fr .85fr;gap:18px;margin-top:22px;
    }
    .command-card,.pipeline-card,.form-card{
      border:1px solid var(--line);
      border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:22px;
    }
    .cc-top{
      display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px;
    }
    .cc-mini{
      border:1px solid var(--line2);border-radius:16px;padding:14px;background:rgba(255,255,255,.03);
    }
    .cc-mini .k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff}
    .cc-mini .v{font-size:28px;font-weight:1000;line-height:1;margin-top:10px}
    .stream-list{display:grid;gap:10px;margin-top:16px}
    .stream-item{padding:12px 14px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.03)}
    .stream-item .name{font-size:14px;font-weight:900;color:#ecf4ff}
    .stream-item .meta{font-size:13px;color:#c6d7ef;margin-top:4px}
    .alert-dot{display:inline-block;width:10px;height:10px;border-radius:999px;background:#5bd38d;margin-right:8px}
    .alert-dot.warn{background:#f7be68}
    .alert-dot.danger{background:#ff6b6b}
    .forms-grid{
      display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:22px;
    }
    .form-card h3{margin:0 0 6px;font-size:26px}
    .form-card p{margin:0 0 16px;color:#c7d8ef;line-height:1.6}
    .field{display:grid;gap:8px;margin-bottom:12px}
    .field label{font-size:12px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:#a8bddc}
    .field input,.field textarea,.field select{
      width:100%;padding:12px 14px;border-radius:14px;border:1px solid rgba(255,255,255,.10);
      background:#0d1526;color:#eef4ff;outline:none;
    }
    .field textarea{min-height:96px;resize:vertical}
    .result{
      margin-top:12px;padding:12px 14px;border-radius:14px;background:rgba(91,212,255,.10);border:1px solid rgba(91,212,255,.18);display:none;
    }
    .result.show{display:block}
    .result.error{background:rgba(255,107,107,.12);border-color:rgba(255,107,107,.18)}
    .footer{
      margin-top:22px;
      padding:18px 0 8px;
      color:#9fb4d6;
      font-size:14px;
      line-height:1.6;
    }
    @media (max-width:1100px){
      .hero-grid,.live-band,.forms-grid{grid-template-columns:1fr}
      .hero-mini-grid,.cc-top{grid-template-columns:repeat(2,1fr)}
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
      .hero-actions,.demo-btns,.route-card .route-actions{flex-direction:column}
      .btn{width:100%}
      .demo-stage{min-height:250px}
      .hero-mini-grid,.cc-top{grid-template-columns:1fr}
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
        <div class="brand-sub">Hospitals · Clinics · Investors · Patients</div>
      </div>

      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#hospital-story">Hospital Story</a>
        <a href="#product-walkthrough">Live Platform</a>
        <a href="#commercial-path">Investor Story</a>
        <a href="#dashboard">Command Center</a>
        <a href="/admin/review">Admin Review</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="hero-inner">
        <div class="hero-grid">
          <div class="glass hero-copy">
            <div class="hero-kicker">Single-domain production experience</div>
            <h1>See the platform. Open the demo. Move hospitals and investors into the right path.</h1>
            <p>
              Early Risk Alert AI is a predictive clinical intelligence platform built to make risk visible earlier, support clinician action sooner,
              and present a live command-center story that investors and health systems understand immediately.
            </p>

            <div class="hero-actions">
              <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
              <a class="btn secondary" href="#hospital-form">Book Hospital Demo</a>
              <a class="btn secondary" href="#investor-form">Investor Intake</a>
              <a class="btn secondary" href="#dashboard">Open Command Center</a>
            </div>

            <div class="hero-mini-grid">
              <div class="hero-mini">
                <div class="k">Hospital Path</div>
                <div class="v">Clinical workflow review</div>
              </div>
              <div class="hero-mini">
                <div class="k">Live Platform</div>
                <div class="v">Alerts, scoring, confidence</div>
              </div>
              <div class="hero-mini">
                <div class="k">Investor Path</div>
                <div class="v">Commercial positioning</div>
              </div>
              <div class="hero-mini">
                <div class="k">Production Lock</div>
                <div class="v">Single trusted domain</div>
              </div>
            </div>
          </div>

          <div class="glass demo-card" id="demo">
            <div class="demo-stage">
              <div class="demo-badge">Platform Demo</div>

              <div class="demo-play">
                <a class="demo-play-btn" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer" aria-label="Watch demo video">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M8 5v14l11-7z"></path>
                  </svg>
                </a>
              </div>

              <div class="demo-caption">
                <h3>Early Risk Alert AI Master Demo</h3>
                <p>A clean public-facing demo entry using your branded image as the thumbnail and your walkthrough as the video destination.</p>
              </div>
            </div>

            <div class="demo-bottom">
              <div class="demo-note">
                Clicking play opens the full walkthrough. All platform paths below stay on the working homepage and do not break.
              </div>
              <div class="demo-btns">
                <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
                <a class="btn secondary" href="#hospital-story">Hospital Demo</a>
                <a class="btn secondary" href="#commercial-path">Investor View</a>
                <a class="btn secondary" href="#dashboard">Command Center</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="route-grid">
      <div class="route-card" id="hospital-story">
        <div class="route-label">Hospital Story</div>
        <h3>Clinical Buyer Path</h3>
        <p>Show workflows, command-center visibility, AI risk scoring, and operational value for hospitals and care teams.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#hospital-form">Hospital Intake</a>
          <a class="btn secondary" href="#dashboard">Open Command Center</a>
        </div>
      </div>

      <div class="route-card" id="product-walkthrough">
        <div class="route-label">Live Platform</div>
        <h3>Product Walkthrough</h3>
        <p>Present alerts, patient focus, confidence indicators, recommended action, and the live platform experience.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#dashboard">View Live Stream</a>
          <a class="btn secondary" href="/admin/review">Live Admin Feed</a>
        </div>
      </div>

      <div class="route-card" id="commercial-path">
        <div class="route-label">Investor Story</div>
        <h3>Commercial Path</h3>
        <p>Guide investors through product readiness, founder credibility, market relevance, and enterprise SaaS positioning.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#investor-form">Investor Intake</a>
          <a class="btn secondary" href="#contact">Founder Contact</a>
        </div>
      </div>
    </div>

<div class="live-band">
  <div class="command-card" id="dashboard">
    <div class="section-kicker">Live Clinical Command Center</div>
    <h2 style="margin:0 0 8px">Hospital-grade visual monitoring with real command-center panels.</h2>
    <p>Live alerts, risk scoring, patient condition status, and bedside-style monitors update automatically so hospitals and investors instantly understand the platform.</p>

    <div class="cc-top">
      <div class="cc-mini">
        <div class="k">Open Alerts</div>
        <div class="v" id="cc-open-alerts">0</div>
      </div>
      <div class="cc-mini">
        <div class="k">Critical Alerts</div>
        <div class="v" id="cc-critical-alerts">0</div>
      </div>
      <div class="cc-mini">
        <div class="k">Avg Risk Score</div>
        <div class="v" id="cc-avg-risk">0.0</div>
      </div>
      <div class="cc-mini">
        <div class="k">Events Last Hour</div>
        <div class="v" id="cc-events-hour">0</div>
      </div>
    </div>

    <div class="monitor-grid">
      <div class="monitor-panel critical">
        <div class="monitor-top">
          <div>
            <div class="monitor-label">Patient Monitor A</div>
            <div class="monitor-name">Critical Watch</div>
          </div>
          <div class="monitor-status status-critical" id="monitor-a-status">Critical</div>
        </div>

        <div class="wave-wrap">
          <svg class="wave-svg" viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true">
            <path id="wave-a" class="wave-line critical-line"
              d="M0,72 L18,72 L30,72 L42,72 L54,72 L66,72 L78,72 L90,72 L102,72 L114,72 L126,72
                 L138,72 L150,30 L162,108 L174,72 L186,72 L198,72 L210,72 L222,72 L234,72 L246,72
                 L258,72 L270,72 L282,72 L294,72 L306,34 L318,106 L330,72 L342,72 L354,72 L366,72
                 L378,72 L390,72 L402,72 L414,72 L426,72 L438,72 L450,72 L462,28 L474,108 L486,72
                 L498,72 L510,72 L522,72 L534,72 L546,72 L558,72 L570,72 L582,72 L600,72" />
          </svg>
        </div>

        <div class="monitor-metrics">
          <div class="metric-box">
            <span class="metric-k">HR</span>
            <span class="metric-v" id="monitor-a-hr">128</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">SpO₂</span>
            <span class="metric-v" id="monitor-a-spo2">89</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">BP</span>
            <span class="metric-v" id="monitor-a-bp">164/98</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">Risk</span>
            <span class="metric-v" id="monitor-a-risk">9.1</span>
          </div>
        </div>
      </div>

      <div class="monitor-panel warning">
        <div class="monitor-top">
          <div>
            <div class="monitor-label">Patient Monitor B</div>
            <div class="monitor-name">Escalation Watch</div>
          </div>
          <div class="monitor-status status-warning" id="monitor-b-status">High</div>
        </div>

        <div class="wave-wrap">
          <svg class="wave-svg" viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true">
            <path id="wave-b" class="wave-line warning-line"
              d="M0,72 L20,72 L40,72 L60,72 L80,72 L100,72 L120,72 L140,72 L160,58 L180,86 L200,72
                 L220,72 L240,72 L260,72 L280,72 L300,60 L320,84 L340,72 L360,72 L380,72 L400,72
                 L420,72 L440,56 L460,88 L480,72 L500,72 L520,72 L540,72 L560,72 L580,72 L600,72" />
          </svg>
        </div>

        <div class="monitor-metrics">
          <div class="metric-box">
            <span class="metric-k">HR</span>
            <span class="metric-v" id="monitor-b-hr">112</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">SpO₂</span>
            <span class="metric-v" id="monitor-b-spo2">93</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">BP</span>
            <span class="metric-v" id="monitor-b-bp">148/90</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">Risk</span>
            <span class="metric-v" id="monitor-b-risk">8.2</span>
          </div>
        </div>
      </div>

      <div class="monitor-panel stable">
        <div class="monitor-top">
          <div>
            <div class="monitor-label">Patient Monitor C</div>
            <div class="monitor-name">Stable Watch</div>
          </div>
          <div class="monitor-status status-stable" id="monitor-c-status">Stable</div>
        </div>

        <div class="wave-wrap">
          <svg class="wave-svg" viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true">
            <path id="wave-c" class="wave-line stable-line"
              d="M0,72 L18,72 L36,72 L54,72 L72,72 L90,72 L108,68 L126,76 L144,72 L162,72 L180,72
                 L198,72 L216,69 L234,75 L252,72 L270,72 L288,72 L306,68 L324,76 L342,72 L360,72
                 L378,72 L396,70 L414,74 L432,72 L450,72 L468,72 L486,69 L504,75 L522,72 L540,72
                 L558,72 L576,70 L594,74 L600,72" />
          </svg>
        </div>

        <div class="monitor-metrics">
          <div class="metric-box">
            <span class="metric-k">HR</span>
            <span class="metric-v" id="monitor-c-hr">84</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">SpO₂</span>
            <span class="metric-v" id="monitor-c-spo2">98</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">BP</span>
            <span class="metric-v" id="monitor-c-bp">122/78</span>
          </div>
          <div class="metric-box">
            <span class="metric-k">Risk</span>
            <span class="metric-v" id="monitor-c-risk">3.4</span>
          </div>
        </div>
      </div>
    </div>

    <div class="stream-list" id="cc-alert-stream">
      <div class="stream-item">
        <div class="name">Waiting for live stream data</div>
        <div class="meta">The command center updates automatically from the platform APIs.</div>
      </div>
    </div>
  </div>

  <div class="pipeline-card">
    <div class="section-kicker">Investor Pipeline Dashboard</div>
    <h2 style="margin:0 0 8px">Built to show product value and inbound momentum.</h2>
    <p style="margin-bottom:14px">Hospital demos, executive walkthroughs, and investor intake all flow into one operating dashboard.</p>
    <div class="list" style="margin-top:0">
      <div class="mini">
        <div class="k">Patients With Alerts</div>
        <span class="v" id="cc-patients-alerts">0</span>
        <p>Live demo signal that the platform is actively watching risk.</p>
      </div>
      <div class="mini">
        <div class="k">Command Center</div>
        <span class="v">Live</span>
        <p>Refreshes automatically to simulate hospital operations visibility.</p>
      </div>
      <div class="mini">
        <div class="k">Lead Capture</div>
        <span class="v">Live</span>
        <p>Hospital and investor requests feed directly into admin review.</p>
      </div>
    </div>
  </div>
</div>

    <div class="forms-grid">
      <div class="form-card" id="hospital-form">
        <div class="section-kicker">Hospital Demo Request</div>
        <h3>Book a hospital demo</h3>
        <p>Capture hospital interest, ops review, and clinical buyer conversations.</p>
        <form id="hospitalLeadForm">
          <div class="field"><label>Full Name</label><input name="full_name" required></div>
          <div class="field"><label>Organization</label><input name="organization" required></div>
          <div class="field"><label>Role</label><input name="role" required></div>
          <div class="field"><label>Email</label><input name="email" type="email" required></div>
          <div class="field"><label>Phone</label><input name="phone"></div>
          <div class="field"><label>Facility Type</label><input name="facility_type" placeholder="Hospital, health system, clinic"></div>
          <div class="field"><label>Timeline</label><input name="timeline" placeholder="Immediate, 30 days, this quarter"></div>
          <div class="field"><label>Message</label><textarea name="message"></textarea></div>
          <button class="btn primary" type="submit">Submit Hospital Request</button>
        </form>
        <div class="result" id="hospitalLeadResult"></div>
      </div>

      <div class="form-card" id="executive-form">
        <div class="section-kicker">Executive Walkthrough</div>
        <h3>Request leadership review</h3>
        <p>Capture executive, strategic, and pilot evaluation conversations.</p>
        <form id="executiveLeadForm">
          <div class="field"><label>Full Name</label><input name="full_name" required></div>
          <div class="field"><label>Organization</label><input name="organization" required></div>
          <div class="field"><label>Executive Title</label><input name="title" required></div>
          <div class="field"><label>Email</label><input name="email" type="email" required></div>
          <div class="field"><label>Priority</label><select name="priority"><option>High</option><option>Medium</option><option>Low</option></select></div>
          <div class="field"><label>Timeline</label><input name="timeline" placeholder="This week, this month, this quarter"></div>
          <div class="field"><label>Message</label><textarea name="message"></textarea></div>
          <button class="btn primary" type="submit">Submit Executive Request</button>
        </form>
        <div class="result" id="executiveLeadResult"></div>
      </div>

      <div class="form-card" id="investor-form">
        <div class="section-kicker">Investor Intake</div>
        <h3>Start investor conversation</h3>
        <p>Capture investor interest, check size, and commercial follow-up.</p>
        <form id="investorLeadForm">
          <div class="field"><label>Full Name</label><input name="full_name" required></div>
          <div class="field"><label>Organization</label><input name="organization" required></div>
          <div class="field"><label>Role</label><input name="role" required></div>
          <div class="field"><label>Email</label><input name="email" type="email" required></div>
          <div class="field"><label>Investor Type</label><input name="investor_type" placeholder="Angel, VC, strategic, family office"></div>
          <div class="field"><label>Check Size</label><input name="check_size" placeholder="$100K, $500K, $2M"></div>
          <div class="field"><label>Timeline</label><input name="timeline" placeholder="Now, this month, this quarter"></div>
          <div class="field"><label>Message</label><textarea name="message"></textarea></div>
          <button class="btn primary" type="submit">Submit Investor Request</button>
        </form>
        <div class="result" id="investorLeadResult"></div>
      </div>
    </div>

    <section class="section" id="contact">
      <div class="section-kicker">Founder & Contact</div>
      <h2>Milton Munroe</h2>
      <p>
        {{ founder_role }} · <a href="mailto:{{ info_email }}">{{ info_email }}</a> ·
        <a href="tel:{{ business_phone|replace('-', '') }}">{{ business_phone }}</a>
      </p>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC · Predictive clinical intelligence platform · Hospitals · Clinics · Investors · Patients
    </div>
  </div>

  <script>
    async function submitLeadForm(formId, endpoint, resultId, successText) {
      const form = document.getElementById(formId);
      const result = document.getElementById(resultId);

      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(form).entries());

        try {
          const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
          });

          const payload = await res.json();

          if (!res.ok || !payload.ok) {
            throw new Error(payload.error || "Submission failed");
          }

          result.className = "result show";
          result.textContent = successText;
          form.reset();
        } catch (err) {
          result.className = "result show error";
          result.textContent = "Submission failed. Please try again.";
        }
      });
    }

    submitLeadForm("hospitalLeadForm", "/submit/hospital", "hospitalLeadResult", "Hospital demo request submitted successfully.");
    submitLeadForm("executiveLeadForm", "/submit/executive", "executiveLeadResult", "Executive walkthrough request submitted successfully.");
    submitLeadForm("investorLeadForm", "/submit/investor", "investorLeadResult", "Investor request submitted successfully.");

    async function refreshCommandCenter() {
      try {
    const res = await fetch("/api/v1/dashboard/overview?tenant_id=demo&refresh=" + Date.now(), {
      headers: { "Accept": "application/json" },
      cache: "no-store"
    });

    if (res.ok) {
      const data = await res.json();
      document.getElementById("cc-open-alerts").textContent = data.open_alerts ?? 0;
      document.getElementById("cc-critical-alerts").textContent = data.critical_alerts ?? 0;
      document.getElementById("cc-avg-risk").textContent = Number(data.avg_risk_score ?? 0).toFixed(1);
      document.getElementById("cc-events-hour").textContent = data.events_last_hour ?? 0;
      document.getElementById("cc-patients-alerts").textContent = data.patients_with_alerts ?? 0;
    }
  } catch (err) {
    console.error("Overview refresh failed", err);
  }

  try {
    const res = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), {
      headers: { "Accept": "application/json" },
      cache: "no-store"
    });

    if (res.ok) {
      const data = await res.json();
      const alerts = Array.isArray(data.alerts) ? data.alerts.slice(0, 5) : [];
      const wrap = document.getElementById("cc-alert-stream");

      if (!alerts.length) {
        wrap.innerHTML = '<div class="stream-item"><div class="name">No active alerts right now</div><div class="meta">The system is running and waiting for new events.</div></div>';
      } else {
        wrap.innerHTML = alerts.map(function (a) {
          const sev = (a.severity || "").toLowerCase();
          const dot = sev === "critical" ? "danger" : (sev === "high" ? "warn" : "");
          return `
            <div class="stream-item">
              <div class="name"><span class="alert-dot ${dot}"></span>${a.title || a.alert_type || "Risk alert"}</div>
              <div class="meta">Patient: ${a.patient_id || "Unknown"} · Severity: ${sev || "n/a"} · Risk: ${a.risk_score ?? ""}</div>
            </div>
          `;
        }).join("");
      }

      const monitorA = alerts[0] || {};
      const monitorB = alerts[1] || {};
      const monitorC = alerts[2] || {};

      function setMonitor(prefix, alert, defaults) {
        const severity = (alert.severity || defaults.severity || "stable").toLowerCase();
        const statusEl = document.getElementById(prefix + "-status");
        const hrEl = document.getElementById(prefix + "-hr");
        const spo2El = document.getElementById(prefix + "-spo2");
        const bpEl = document.getElementById(prefix + "-bp");
        const riskEl = document.getElementById(prefix + "-risk");

        const statusText =
          severity === "critical" ? "Critical" :
          (severity === "high" ? "High" : "Stable");

        statusEl.textContent = statusText;
        statusEl.className =
          "monitor-status " +
          (severity === "critical"
            ? "status-critical"
            : (severity === "high" ? "status-warning" : "status-stable"));

        const baseRisk = Number(alert.risk_score ?? defaults.risk ?? 3.4);

        const hr =
          severity === "critical"
            ? 124 + Math.floor(Math.random() * 10)
            : (severity === "high"
                ? 106 + Math.floor(Math.random() * 10)
                : 78 + Math.floor(Math.random() * 10));

        const spo2 =
          severity === "critical"
            ? 87 + Math.floor(Math.random() * 3)
            : (severity === "high"
                ? 92 + Math.floor(Math.random() * 3)
                : 97 + Math.floor(Math.random() * 2));

        const sys =
          severity === "critical"
            ? 160 + Math.floor(Math.random() * 8)
            : (severity === "high"
                ? 146 + Math.floor(Math.random() * 8)
                : 120 + Math.floor(Math.random() * 6));

        const dia =
          severity === "critical"
            ? 96 + Math.floor(Math.random() * 6)
            : (severity === "high"
                ? 88 + Math.floor(Math.random() * 5)
                : 76 + Math.floor(Math.random() * 4));

        hrEl.textContent = hr;
        spo2El.textContent = spo2;
        bpEl.textContent = sys + "/" + dia;
        riskEl.textContent = baseRisk.toFixed(1);
      }

      setMonitor("monitor-a", monitorA, { severity: "critical", risk: 9.1 });
      setMonitor("monitor-b", monitorB, { severity: "high", risk: 8.2 });
      setMonitor("monitor-c", monitorC, { severity: "stable", risk: 3.4 });
    }
  } catch (err) {
    console.error("Snapshot refresh failed", err);
  }
}

    refreshCommandCenter();
    setInterval(refreshCommandCenter, 5000);
  </script>
</body>
</html>
"""

ADMIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Admin Review</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --panel:#101a2d;
      --panel2:#13203a;
      --line:rgba(255,255,255,.08);
      --text:#eef4ff;
      --muted:#9eb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --amber:#f7be68;
      --red:#ff6b6b;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      background:linear-gradient(180deg, #07101c, #0c1425);
      color:var(--text);
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      padding:24px;
    }
    .shell{max-width:1440px;margin:0 auto}
    h1{margin:0 0 10px;font-size:44px;line-height:1;letter-spacing:-.04em}
    .sub{color:#b7cae5;font-size:16px;line-height:1.6;margin-bottom:18px}
    .live-status-bar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:18px 0 24px;padding:12px 16px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(91,212,255,.08);color:#dff6ff;font-size:14px}
    .ticker-wrap{margin:18px 0 26px;padding:16px;border:1px solid rgba(255,255,255,.08);border-radius:18px;background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02))}
    .ticker-title{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#9eb4d6;font-weight:900;margin-bottom:10px}
    .ticker-stream{display:flex;gap:10px;overflow:auto;padding-bottom:6px}
    .ticker-chip{min-width:280px;display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03)}
    .ticker-chip strong{display:block;font-size:13px;color:#ecf4ff}
    .ticker-chip span{display:block;font-size:13px;color:#b8c7df}
    .ticker-empty{color:#9eb4d6;font-size:14px}
    .pipeline-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 24px}
    .pipeline-card,.admin-kpi,.command-card,.section,.lead-type{
      border:1px solid rgba(255,255,255,.08);
      border-radius:18px;
      padding:18px;
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
    }
    .pipeline-card .k,.admin-kpi .k,.command-title{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .pipeline-card .v,.admin-kpi .v{font-size:30px;font-weight:1000;line-height:1;margin-top:10px;color:#ecf4ff}
    .pipeline-card .hint,.admin-kpi .hint,.command-sub{margin-top:8px;font-size:13px;color:#c6d7ef;line-height:1.5}
    .admin-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 24px}
    .command-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;margin:0 0 24px}
    .command-big{font-size:34px;font-weight:1000;line-height:1;color:#ecf4ff}
    .stream-list{display:grid;gap:10px;margin-top:14px}
    .stream-item{padding:12px 14px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.03)}
    .stream-item .name{font-size:14px;font-weight:900;color:#ecf4ff}
    .stream-item .meta{font-size:13px;color:#c6d7ef;margin-top:4px}
    .alert-dot{display:inline-block;width:10px;height:10px;border-radius:999px;background:#5bd38d;margin-right:8px}
    .alert-dot.warn{background:#f7be68}
    .alert-dot.danger{background:#ff6b6b}
    .lead-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:24px}
    .lead-type h3{margin:0 0 8px;font-size:20px}
    .lead-type p{margin:0;color:#c6d7ef;line-height:1.6}
    .section{margin:0 0 18px}
    .section h2{margin:0 0 14px;font-size:26px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:12px 10px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;font-size:14px;vertical-align:top}
    th{color:#9eb4d6;text-transform:uppercase;font-size:12px;letter-spacing:.08em}
    td{color:#ecf4ff}
    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:7px 10px;border-radius:999px;font-size:12px;font-weight:900;
      letter-spacing:.03em;border:1px solid transparent
    }
    .status-new{background:rgba(91,212,255,.12);border-color:rgba(91,212,255,.22);color:#dff6ff}
    .status-contacted{background:rgba(91,211,141,.12);border-color:rgba(91,211,141,.24);color:#dfffea}
    .status-closed{background:rgba(255,107,107,.14);border:1px solid rgba(255,107,107,.24);color:#ffd8d8}
    .status-actions{display:flex;gap:8px;flex-wrap:wrap}
    .status-btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:8px 10px;border-radius:12px;border:1px solid rgba(255,255,255,.10);
      background:rgba(255,255,255,.03);color:#eef4ff;font-size:12px;font-weight:900;
      cursor:pointer;text-decoration:none
    }
    .status-btn:hover{border-color:rgba(91,212,255,.24)}
    .back-link{display:inline-flex;margin-bottom:16px;color:#b8c7df;font-weight:900}
    @media (max-width:980px){
      .pipeline-grid,.admin-grid,.command-grid,.lead-row{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
  <div class="shell">
    <a href="/" class="back-link">← Back to platform</a>
    <h1>Admin Review</h1>
    <div class="sub">Live activity ticker, live clinical command center stream, investor pipeline analytics, and status updates without page refresh.</div>

    <div class="live-status-bar">
      <span>Live admin refresh active</span>
      <span>Last refresh: <strong id="last-refresh-time">Just now</strong></span>
    </div>

    <div class="ticker-wrap">
      <div class="ticker-title">Live Activity Ticker</div>
      <div class="ticker-stream" id="recent-activity-wrap">__RECENT_ACTIVITY__</div>
    </div>

    <div class="pipeline-grid">
      <div class="pipeline-card">
        <div class="k">Total Leads</div>
        <div class="v" id="total-count">__TOTAL_COUNT__</div>
        <div class="hint">Live combined intake across hospitals, executives, and investors.</div>
      </div>
      <div class="pipeline-card">
        <div class="k">New</div>
        <div class="v" id="new-count">__NEW_COUNT__</div>
        <div class="hint">Fresh requests waiting for first response.</div>
      </div>
      <div class="pipeline-card">
        <div class="k">Contacted</div>
        <div class="v" id="contacted-count">__CONTACTED_COUNT__</div>
        <div class="hint">Active follow-up pipeline in progress.</div>
      </div>
      <div class="pipeline-card">
        <div class="k">Closed</div>
        <div class="v" id="closed-count">__CLOSED_COUNT__</div>
        <div class="hint">Requests fully worked through and archived as complete.</div>
      </div>
    </div>

    <div class="admin-grid">
      <div class="admin-kpi">
        <div class="k">Hospital Demo Requests</div>
        <div class="v" id="hospital-count">__HOSPITAL_COUNT__</div>
        <div class="hint">Clinical buyer and operator pipeline</div>
      </div>
      <div class="admin-kpi">
        <div class="k">Executive Walkthrough Requests</div>
        <div class="v" id="exec-count">__EXEC_COUNT__</div>
        <div class="hint">Leadership and pilot evaluation requests</div>
      </div>
      <div class="admin-kpi">
        <div class="k">Investor Intake Requests</div>
        <div class="v" id="investor-count">__INVESTOR_COUNT__</div>
        <div class="hint">Commercial and investor follow-up flow</div>
      </div>
    </div>

    <div class="command-grid">
      <div class="command-card">
        <div class="command-title">Live Clinical Command Center</div>
        <div class="command-big" id="cc-open-alerts">0</div>
        <div class="command-sub">Open alerts in the live demo environment. This gives investors and health systems instant proof that the platform is running as an active monitoring system.</div>
        <div class="stream-list" id="cc-alert-stream">
          <div class="stream-item">
            <div class="name">Waiting for live stream data</div>
            <div class="meta">The command center updates automatically from the platform APIs.</div>
          </div>
        </div>
      </div>
      <div class="command-card">
        <div class="command-title">Investor Pipeline Dashboard</div>
        <div class="command-big" id="cc-avg-risk">0.0</div>
        <div class="command-sub">Average live risk score from the demo command center. This helps investors understand the product in seconds: it continuously watches, scores, and escalates patient risk in real time.</div>
        <div class="stream-list">
          <div class="stream-item">
            <div class="name"><span class="alert-dot danger"></span>Critical alerts</div>
            <div class="meta" id="cc-critical-alerts">0</div>
          </div>
          <div class="stream-item">
            <div class="name"><span class="alert-dot warn"></span>Events last hour</div>
            <div class="meta" id="cc-events-hour">0</div>
          </div>
          <div class="stream-item">
            <div class="name"><span class="alert-dot"></span>Patients with alerts</div>
            <div class="meta" id="cc-patients-alerts">0</div>
          </div>
        </div>
      </div>
    </div>

    <div class="lead-row">
      <div class="lead-type"><h3>Hospital Leads</h3><p>Clinical buyers, hospital operators, RPM teams, and health systems.</p></div>
      <div class="lead-type"><h3>Executive Leads</h3><p>Leadership-facing walkthrough requests for evaluations, pilots, and strategic review.</p></div>
      <div class="lead-type"><h3>Investor Leads</h3><p>Investor pipeline capture with contact details, timing, and opportunity notes.</p></div>
    </div>

    <div class="section">
      <h2>Hospital Demo Requests</h2>
      <div id="hospital-table-wrap">__HOSPITAL_TABLE__</div>
    </div>

    <div class="section">
      <h2>Executive Walkthrough Requests</h2>
      <div id="exec-table-wrap">__EXEC_TABLE__</div>
    </div>

    <div class="section">
      <h2>Investor Intake</h2>
      <div id="investor-table-wrap">__INVESTOR_TABLE__</div>
    </div>
  </div>

  <script>
    (function () {
      const POLL_MS = 7000;
      let busy = false;

      function setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
      }

      function setHTML(id, value) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = value;
      }

      async function refreshAdminReview() {
        if (busy) return;
        busy = true;
        try {
          const res = await fetch("/admin/review/data?refresh=" + Date.now(), {
            headers: { "Accept": "application/json" },
            cache: "no-store"
          });
          if (!res.ok) return;
          const data = await res.json();

          setText("hospital-count", data.hospital_count);
          setText("exec-count", data.exec_count);
          setText("investor-count", data.investor_count);
          setText("total-count", data.total_count);
          setText("new-count", data.new_count);
          setText("contacted-count", data.contacted_count);
          setText("closed-count", data.closed_count);

          setHTML("recent-activity-wrap", data.recent_activity_html);
          setHTML("hospital-table-wrap", data.hospital_table);
          setHTML("exec-table-wrap", data.exec_table);
          setHTML("investor-table-wrap", data.investor_table);

          setText("last-refresh-time", new Date().toLocaleTimeString());
        } catch (err) {
          console.error("Admin refresh failed:", err);
        } finally {
          busy = false;
        }
      }

      async function refreshCommandCenter() {
        try {
          const res = await fetch("/api/v1/dashboard/overview?tenant_id=demo&refresh=" + Date.now(), {
            headers: { "Accept": "application/json" },
            cache: "no-store"
          });
          if (!res.ok) return;
          const data = await res.json();

          setText("cc-open-alerts", data.open_alerts ?? 0);
          setText("cc-critical-alerts", data.critical_alerts ?? 0);
          setText("cc-events-hour", data.events_last_hour ?? 0);
          setText("cc-patients-alerts", data.patients_with_alerts ?? 0);
          setText("cc-avg-risk", Number(data.avg_risk_score ?? 0).toFixed(1));
        } catch (err) {
          console.error("Command center overview refresh failed:", err);
        }

        try {
          const res = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), {
            headers: { "Accept": "application/json" },
            cache: "no-store"
          });
          if (!res.ok) return;
          const data = await res.json();

          const alerts = Array.isArray(data.alerts) ? data.alerts.slice(0, 5) : [];
          const wrap = document.getElementById("cc-alert-stream");
          if (!wrap) return;

          if (!alerts.length) {
            wrap.innerHTML = `
              <div class="stream-item">
                <div class="name">No active alerts right now</div>
                <div class="meta">The system is running and waiting for new events.</div>
              </div>
            `;
            return;
          }

          wrap.innerHTML = alerts.map(function (a) {
            const sev = (a.severity || "").toLowerCase();
            const dot = sev === "critical" ? "danger" : (sev === "high" ? "warn" : "");
            const patient = a.patient_id || "Unknown patient";
            const title = a.title || a.alert_type || "Risk alert";
            const score = a.risk_score ?? "";
            return `
              <div class="stream-item">
                <div class="name"><span class="alert-dot ${dot}"></span>${title}</div>
                <div class="meta">Patient: ${patient} · Severity: ${sev || "n/a"} · Risk: ${score}</div>
              </div>
            `;
          }).join("");
        } catch (err) {
          console.error("Command center alert refresh failed:", err);
        }
      }

      async function handleStatusClick(e) {
        const link = e.target.closest("a[data-status-link='1']");
        if (!link) return;
        e.preventDefault();

        try {
          const res = await fetch(link.href, {
            headers: { "X-Requested-With": "fetch" },
            cache: "no-store"
          });
          if (!res.ok) return;
          await refreshAdminReview();
        } catch (err) {
          console.error("Status update failed:", err);
        }
      }

      document.body.addEventListener("click", handleStatusClick);
      refreshAdminReview();
      refreshCommandCenter();

      setInterval(refreshAdminReview, POLL_MS);
      setInterval(refreshCommandCenter, 5000);

      window.addEventListener("focus", function () {
        refreshAdminReview();
        refreshCommandCenter();
      });

      document.addEventListener("visibilitychange", function () {
        if (!document.hidden) {
          refreshAdminReview();
          refreshCommandCenter();
        }
      });
    })();
  </script>
</body>
</html>
"""

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_norm(status: str | None) -> str:
    raw = (status or "").strip().lower()
    if raw == "contacted":
        return "Contacted"
    if raw == "closed":
        return "Closed"
    return "New"


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


def _save_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_money(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return 0.0
    m = re.search(r"(-?\$?\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0
    num = float(m.group(1).replace("$", ""))
    text_upper = text.upper()
    if "M" in text_upper:
        num *= 1_000_000
    elif "K" in text_upper:
        num *= 1_000
    return num


def send_notification_email(subject: str, message: str) -> None:
    sender = INFO_EMAIL
    recipients = [INFO_EMAIL, FOUNDER_EMAIL]

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    password = os.getenv("EMAIL_PASSWORD", "").strip()
    if not password:
        print("Email send skipped: EMAIL_PASSWORD is not set", flush=True)
        return

    try:
        server = smtplib.SMTP("smtp.zoho.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        print("Email sent successfully to:", recipients, flush=True)
    except Exception as e:
        print("Email send failed:", e, flush=True)


def _status_badge(status: str) -> str:
    s = _status_norm(status)
    cls = {
        "New": "status-pill status-new",
        "Contacted": "status-pill status-contacted",
        "Closed": "status-pill status-closed",
    }.get(s, "status-pill status-new")
    return f"<span class='{cls}'>{html.escape(s)}</span>"


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r.get("submitted_at", ""), reverse=True)


def _update_row_status(path: Path, submitted_at: str, new_status: str) -> None:
    rows = _read_jsonl(path)
    updated = []
    norm = _status_norm(new_status)
    for row in rows:
        if row.get("submitted_at", "") == submitted_at:
            row["status"] = norm
        updated.append(row)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in updated), encoding="utf-8")


def _table_html(rows: list[dict[str, Any]], columns: list[str], labels: dict[str, str], lead_type: str) -> str:
    rows = _sort_rows(rows)
    if not rows:
        return "<div class='stream-item'><div class='name'>No entries yet</div><div class='meta'>New requests will appear here automatically.</div></div>"

    head = "".join(f"<th>{html.escape(labels.get(c, c.replace('_',' ').title()))}</th>" for c in columns)
    head += "<th>Status Actions</th>"

    body_parts = []
    for row in rows:
        cells = []
        for c in columns:
            value = row.get(c, "")
            if c == "status":
                cells.append(f"<td>{_status_badge(str(value))}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")

        submitted_at = html.escape(row.get("submitted_at", ""))
        actions = f"""
        <div class="status-actions">
          <a class="status-btn" data-status-link="1" href="/admin/status/{lead_type}?submitted_at={submitted_at}&status=New">New</a>
          <a class="status-btn" data-status-link="1" href="/admin/status/{lead_type}?submitted_at={submitted_at}&status=Contacted">Contacted</a>
          <a class="status-btn" data-status-link="1" href="/admin/status/{lead_type}?submitted_at={submitted_at}&status=Closed">Closed</a>
        </div>
        """
        cells.append(f"<td>{actions}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")

    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def _recent_activity_html(
    hospital_rows: list[dict[str, Any]],
    exec_rows: list[dict[str, Any]],
    investor_rows: list[dict[str, Any]],
    limit: int = 8,
) -> str:
    items: list[dict[str, Any]] = []

    for row in hospital_rows:
        items.append({
            "submitted_at": row.get("submitted_at", ""),
            "label": "Hospital Demo Request",
            "org": row.get("organization", ""),
            "name": row.get("full_name", ""),
            "status": row.get("status", "New"),
        })
    for row in exec_rows:
        items.append({
            "submitted_at": row.get("submitted_at", ""),
            "label": "Executive Walkthrough",
            "org": row.get("organization", ""),
            "name": row.get("full_name", ""),
            "status": row.get("status", "New"),
        })
    for row in investor_rows:
        items.append({
            "submitted_at": row.get("submitted_at", ""),
            "label": "Investor Intake",
            "org": row.get("organization", ""),
            "name": row.get("full_name", ""),
            "status": row.get("status", "New"),
        })

    items = sorted(items, key=lambda r: r.get("submitted_at", ""), reverse=True)[:limit]

    if not items:
        return "<div class='ticker-empty'>No activity yet.</div>"

    chips = []
    for item in items:
        title = html.escape(item["label"])
        org = html.escape(item["org"] or item["name"] or "New lead")
        badge = _status_badge(item["status"])
        chips.append(f"<div class='ticker-chip'><div><strong>{title}</strong><span>{org}</span></div>{badge}</div>")
    return "".join(chips)


def _admin_snapshot_payload(
    hospital_rows: list[dict[str, Any]],
    exec_rows: list[dict[str, Any]],
    investor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    hospital_rows = _sort_rows(hospital_rows)
    exec_rows = _sort_rows(exec_rows)
    investor_rows = _sort_rows(investor_rows)

    all_rows = [("hospital", r) for r in hospital_rows] + [("executive", r) for r in exec_rows] + [("investor", r) for r in investor_rows]
    all_rows = sorted(all_rows, key=lambda x: x[1].get("submitted_at", ""), reverse=True)

    new_count = sum(1 for _, r in all_rows if _status_norm(r.get("status", "New")) == "New")
    contacted_count = sum(1 for _, r in all_rows if _status_norm(r.get("status", "")) == "Contacted")
    closed_count = sum(1 for _, r in all_rows if _status_norm(r.get("status", "")) == "Closed")

    investor_amounts = [_parse_money(r.get("check_size", "")) for r in investor_rows if _parse_money(r.get("check_size", "")) > 0]
    avg_investor_check = round(sum(investor_amounts) / len(investor_amounts), 2) if investor_amounts else 0.0

    return {
        "hospital_count": len(hospital_rows),
        "exec_count": len(exec_rows),
        "investor_count": len(investor_rows),
        "total_count": len(hospital_rows) + len(exec_rows) + len(investor_rows),
        "new_count": new_count,
        "contacted_count": contacted_count,
        "closed_count": closed_count,
        "avg_investor_check": avg_investor_check,
        "recent_activity_html": _recent_activity_html(hospital_rows, exec_rows, investor_rows),
        "hospital_table": _table_html(
            hospital_rows,
            ["submitted_at", "status", "full_name", "organization", "role", "email", "facility_type", "timeline"],
            {
                "submitted_at": "Submitted",
                "status": "Status",
                "full_name": "Name",
                "organization": "Organization",
                "role": "Role",
                "email": "Email",
                "facility_type": "Facility Type",
                "timeline": "Timeline",
            },
            "hospital",
        ),
        "exec_table": _table_html(
            exec_rows,
            ["submitted_at", "status", "full_name", "organization", "title", "email", "priority", "timeline"],
            {
                "submitted_at": "Submitted",
                "status": "Status",
                "full_name": "Name",
                "organization": "Organization",
                "title": "Executive Title",
                "email": "Email",
                "priority": "Priority",
                "timeline": "Timeline",
            },
            "executive",
        ),
        "investor_table": _table_html(
            investor_rows,
            ["submitted_at", "status", "full_name", "organization", "role", "email", "investor_type", "check_size", "timeline"],
            {
                "submitted_at": "Submitted",
                "status": "Status",
                "full_name": "Name",
                "organization": "Organization",
                "role": "Role",
                "email": "Email",
                "investor_type": "Investor Type",
                "check_size": "Check Size",
                "timeline": "Timeline",
            },
            "investor",
        ),
    }


def _simulated_overview(
    hospital_rows: list[dict[str, Any]],
    exec_rows: list[dict[str, Any]],
    investor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(hospital_rows) + len(exec_rows) + len(investor_rows)
    seed = datetime.now(timezone.utc).second
    open_alerts = max(3, total + 2 + (seed % 3))
    critical_alerts = max(1, min(open_alerts, (total // 2) + 1))
    patients_with_alerts = max(2, open_alerts + 3)
    events_last_hour = patients_with_alerts + 8 + (seed % 5)
    avg_risk_score = round(6.2 + ((seed % 11) / 10) + min(total, 6) * 0.18, 1)
    return {
        "open_alerts": open_alerts,
        "critical_alerts": critical_alerts,
        "patients_with_alerts": patients_with_alerts,
        "events_last_hour": events_last_hour,
        "avg_risk_score": avg_risk_score,
        "worker_status": "live",
    }


def _simulated_snapshot(
    hospital_rows: list[dict[str, Any]],
    exec_rows: list[dict[str, Any]],
    investor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    names = [r.get("full_name") or "Lead" for r in (hospital_rows + exec_rows + investor_rows)]
    orgs = [r.get("organization") or "Health System" for r in (hospital_rows + exec_rows + investor_rows)]
    base_alerts = [
        {
            "patient_id": "P-1042",
            "title": "Cardiovascular risk escalation",
            "severity": "critical",
            "risk_score": 9.1,
        },
        {
            "patient_id": "P-2021",
            "title": "Hemodynamic instability trend",
            "severity": "high",
            "risk_score": 8.2,
        },
        {
            "patient_id": "P-3055",
            "title": "Deterioration watchlist update",
            "severity": "moderate",
            "risk_score": 7.0,
        },
    ]

    if names:
        base_alerts.append(
            {
                "patient_id": "P-4108",
                "title": f"New inbound activity linked to {orgs[0]}",
                "severity": "high",
                "risk_score": 8.0,
            }
        )

    return {
        "alerts": base_alerts[:5],
        "lead_context": names[:5],
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "era-dev-secret")

    data_dir = _data_dir()
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"
    investor_file = data_dir / "investor_intake_requests.jsonl"

    @app.get("/")
    def home():
        return render_template_string(
            MAIN_HTML,
            info_email=INFO_EMAIL,
            founder_role=FOUNDER_ROLE,
            business_phone=BUSINESS_PHONE,
        )

    @app.get("/overview")
    def overview():
        return redirect("/#overview")

    @app.get("/investors")
    @app.get("/investor")
    @app.get("/investor-view")
    def investors():
        return redirect("/#commercial-path")

    @app.get("/hospital-demo")
    @app.get("/hospital")
    @app.get("/hospital-intake")
    def hospital():
        return redirect("/#hospital-form")

    @app.get("/dashboard")
    @app.get("/command-center")
    def dashboard():
        return redirect("/#dashboard")

    @app.get("/demo")
    @app.get("/deck")
    def demo():
        return redirect("/#demo")

    @app.get("/healthz")
    def healthz():
        return jsonify(
            {
                "ok": True,
                "service": "early-risk-alert-ai",
                "time": _utc_now_iso(),
                "hospital_requests": len(_read_jsonl(hospital_file)),
                "executive_requests": len(_read_jsonl(exec_file)),
                "investor_requests": len(_read_jsonl(investor_file)),
            }
        )

    @app.get("/robots.txt")
    def robots_txt():
        return Response(
            "User-agent: *\nAllow: /\nSitemap: https://earlyriskalertai.com/sitemap.xml",
            mimetype="text/plain",
        )

    @app.post("/submit/hospital")
    def submit_hospital():
        payload = request.get_json(silent=True) or {}
        payload = {
            "submitted_at": _utc_now_iso(),
            "status": "New",
            "full_name": str(payload.get("full_name", "")).strip(),
            "organization": str(payload.get("organization", "")).strip(),
            "role": str(payload.get("role", "")).strip(),
            "email": str(payload.get("email", "")).strip(),
            "phone": str(payload.get("phone", "")).strip(),
            "facility_type": str(payload.get("facility_type", "")).strip(),
            "timeline": str(payload.get("timeline", "")).strip(),
            "message": str(payload.get("message", "")).strip(),
        }
        _save_jsonl(hospital_file, payload)
        subject = "New Hospital Demo Request"
        message = f"""New Hospital Demo Request

Name: {payload['full_name']}
Organization: {payload['organization']}
Role: {payload['role']}
Email: {payload['email']}
Phone: {payload['phone']}
Facility Type: {payload['facility_type']}
Timeline: {payload['timeline']}
Message: {payload['message']}
Submitted At: {payload['submitted_at']}
"""
        send_notification_email(subject, message)
        return jsonify({"ok": True})

    @app.post("/submit/executive")
    def submit_executive():
        payload = request.get_json(silent=True) or {}
        payload = {
            "submitted_at": _utc_now_iso(),
            "status": "New",
            "full_name": str(payload.get("full_name", "")).strip(),
            "organization": str(payload.get("organization", "")).strip(),
            "title": str(payload.get("title", "")).strip(),
            "email": str(payload.get("email", "")).strip(),
            "priority": str(payload.get("priority", "")).strip(),
            "timeline": str(payload.get("timeline", "")).strip(),
            "message": str(payload.get("message", "")).strip(),
        }
        _save_jsonl(exec_file, payload)
        subject = "New Executive Walkthrough Request"
        message = f"""New Executive Walkthrough Request

Name: {payload['full_name']}
Organization: {payload['organization']}
Title: {payload['title']}
Email: {payload['email']}
Priority: {payload['priority']}
Timeline: {payload['timeline']}
Message: {payload['message']}
Submitted At: {payload['submitted_at']}
"""
        send_notification_email(subject, message)
        return jsonify({"ok": True})

    @app.post("/submit/investor")
    def submit_investor():
        payload = request.get_json(silent=True) or {}
        payload = {
            "submitted_at": _utc_now_iso(),
            "status": "New",
            "full_name": str(payload.get("full_name", "")).strip(),
            "organization": str(payload.get("organization", "")).strip(),
            "role": str(payload.get("role", "")).strip(),
            "email": str(payload.get("email", "")).strip(),
            "investor_type": str(payload.get("investor_type", "")).strip(),
            "check_size": str(payload.get("check_size", "")).strip(),
            "timeline": str(payload.get("timeline", "")).strip(),
            "message": str(payload.get("message", "")).strip(),
        }
        _save_jsonl(investor_file, payload)
        subject = "New Investor Intake Request"
        message = f"""New Investor Intake Request

Name: {payload['full_name']}
Organization: {payload['organization']}
Role: {payload['role']}
Email: {payload['email']}
Investor Type: {payload['investor_type']}
Check Size: {payload['check_size']}
Timeline: {payload['timeline']}
Message: {payload['message']}
Submitted At: {payload['submitted_at']}
"""
        send_notification_email(subject, message)
        return jsonify({"ok": True})

    @app.get("/admin/status/hospital")
    def admin_status_hospital():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(hospital_file, submitted_at, status)
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify({"ok": True, "status": _status_norm(status)})
        return redirect("/admin/review")

    @app.get("/admin/status/executive")
    def admin_status_executive():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(exec_file, submitted_at, status)
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify({"ok": True, "status": _status_norm(status)})
        return redirect("/admin/review")

    @app.get("/admin/status/investor")
    def admin_status_investor():
        submitted_at = request.args.get("submitted_at", "")
        status = request.args.get("status", "New")
        _update_row_status(investor_file, submitted_at, status)
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify({"ok": True, "status": _status_norm(status)})
        return redirect("/admin/review")

    @app.get("/admin/review/data")
    def admin_review_data():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)
        snapshot = _admin_snapshot_payload(hospital_rows, exec_rows, investor_rows)
        snapshot["overview"] = _simulated_overview(hospital_rows, exec_rows, investor_rows)
        snapshot["live_snapshot"] = _simulated_snapshot(hospital_rows, exec_rows, investor_rows)
        return jsonify(snapshot)

    @app.get("/admin/review")
    def admin_review():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)
        snapshot = _admin_snapshot_payload(hospital_rows, exec_rows, investor_rows)

        html_out = ADMIN_HTML
        html_out = html_out.replace("__HOSPITAL_COUNT__", str(snapshot["hospital_count"]))
        html_out = html_out.replace("__EXEC_COUNT__", str(snapshot["exec_count"]))
        html_out = html_out.replace("__INVESTOR_COUNT__", str(snapshot["investor_count"]))
        html_out = html_out.replace("__TOTAL_COUNT__", str(snapshot["total_count"]))
        html_out = html_out.replace("__NEW_COUNT__", str(snapshot["new_count"]))
        html_out = html_out.replace("__CONTACTED_COUNT__", str(snapshot["contacted_count"]))
        html_out = html_out.replace("__CLOSED_COUNT__", str(snapshot["closed_count"]))
        html_out = html_out.replace("__RECENT_ACTIVITY__", snapshot["recent_activity_html"])
        html_out = html_out.replace("__HOSPITAL_TABLE__", snapshot["hospital_table"])
        html_out = html_out.replace("__EXEC_TABLE__", snapshot["exec_table"])
        html_out = html_out.replace("__INVESTOR_TABLE__", snapshot["investor_table"])
        return render_template_string(html_out)

    @app.get("/api/v1/dashboard/overview")
    def api_dashboard_overview():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)
        return jsonify(_simulated_overview(hospital_rows, exec_rows, investor_rows))

    @app.get("/api/v1/live-snapshot")
    def api_live_snapshot():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)
        return jsonify(_simulated_snapshot(hospital_rows, exec_rows, investor_rows))

    @app.get("/api/v1/stream/channels")
    def api_stream_channels():
        return jsonify(
            {
                "ok": True,
                "channels": [
                    "risk-alerts",
                    "patient-stream",
                    "ops-events",
                    "lead-activity",
                ],
            }
        )

    return app
