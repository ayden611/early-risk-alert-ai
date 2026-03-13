from __future__ import annotations

import csv
import html
import io
import json
import os
import random
import smtplib
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
    .hero-copy h1{
      margin:0 0 14px;font-size:clamp(40px,6vw,84px);line-height:.92;letter-spacing:-.06em;font-weight:1000;max-width:800px
    }
    .hero-copy p{margin:0;color:#d0ddf0;font-size:18px;line-height:1.68;max-width:760px}
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}
    .hero-mini-grid{margin-top:20px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .hero-mini{border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03)}
    .hero-mini .k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:8px}
    .hero-mini .v{font-size:16px;font-weight:1000;line-height:1.1}
    .demo-card{overflow:hidden}
    .demo-stage{
      position:relative;aspect-ratio:16/9;min-height:360px;
      background:
        linear-gradient(180deg, rgba(0,0,0,.12), rgba(0,0,0,.40)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
    }
    .demo-badge{
      position:absolute;top:18px;left:18px;z-index:3;padding:9px 13px;border-radius:999px;background:rgba(7,16,28,.62);
      border:1px solid var(--line);color:#eaf3ff;font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;backdrop-filter:blur(12px);
    }
    .demo-play{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:3}
    .demo-play-btn{
      width:96px;height:96px;border-radius:50%;border:1px solid rgba(255,255,255,.18);
      background:rgba(7,16,28,.58);backdrop-filter:blur(12px);
      box-shadow:0 20px 50px rgba(0,0,0,.32),0 0 34px rgba(91,212,255,.18);
      display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative;
    }
    .demo-play-btn::before{
      content:"";position:absolute;inset:-14px;border-radius:50%;border:1px solid rgba(91,212,255,.20);animation:ring 2.8s ease-out infinite;
    }
    @keyframes ring{
      0%{transform:scale(.7);opacity:1}
      100%{transform:scale(1.35);opacity:0}
    }
    .demo-play-btn svg{width:34px;height:34px;margin-left:5px;fill:#eef4ff}
    .demo-caption{position:absolute;left:0;right:0;bottom:0;z-index:2;padding:22px;background:linear-gradient(180deg, transparent, rgba(7,16,28,.82))}
    .demo-caption h3{margin:0 0 8px;font-size:30px;line-height:1;font-weight:1000;letter-spacing:-.04em}
    .demo-caption p{margin:0;color:#d4e2f3;font-size:15px;line-height:1.6}
    .demo-bottom{padding:18px;border-top:1px solid var(--line)}
    .demo-note{color:#bdd0eb;font-size:14px;line-height:1.55;margin-bottom:14px}
    .demo-btns{display:flex;gap:10px;flex-wrap:wrap}
    .route-grid{margin-top:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
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
    .route-card h3{margin:10px 0 10px;font-size:30px;line-height:1;font-weight:1000;letter-spacing:-.04em}
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
    .ticker-wrap{
      margin-top:18px;margin-bottom:18px;border:1px solid rgba(255,255,255,.08);border-radius:18px;overflow:hidden;
      background:
        linear-gradient(90deg, rgba(91,212,255,.08), rgba(122,162,255,.04), rgba(255,255,255,.02)),
        rgba(8,16,29,.92);
      box-shadow:0 12px 28px rgba(0,0,0,.18);
    }
    .ticker-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06)}
    .ticker-title{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9fdcff}
    .ticker-live{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#dff7ff}
    .ticker-live::before{
      content:"";width:8px;height:8px;border-radius:999px;background:#38d39f;box-shadow:0 0 0 0 rgba(56,211,159,.55);animation:livePulse 1.8s infinite;
    }
    @keyframes livePulse{
      0%{box-shadow:0 0 0 0 rgba(56,211,159,.55)}
      70%{box-shadow:0 0 0 12px rgba(56,211,159,0)}
      100%{box-shadow:0 0 0 0 rgba(56,211,159,0)}
    }
    .ticker-track{position:relative;overflow:hidden;white-space:nowrap;padding:14px 0}
    .ticker-move{display:inline-flex;gap:14px;padding-left:100%;animation:tickerMove 26s linear infinite}
    @keyframes tickerMove{
      from{transform:translateX(0)}
      to{transform:translateX(-100%)}
    }
    .ticker-pill{
      display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.04);color:#edf4ff;font-size:13px;font-weight:900;
    }
    .ticker-pill .dot{width:9px;height:9px;border-radius:999px;display:inline-block}
    .ticker-pill .dot.critical{background:#ff667d;box-shadow:0 0 10px rgba(255,102,125,.45)}
    .ticker-pill .dot.high{background:#f4bd6a;box-shadow:0 0 10px rgba(244,189,106,.35)}
    .ticker-pill .dot.stable{background:#38d39f;box-shadow:0 0 10px rgba(56,211,159,.35)}
    .icu-wall{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin-top:18px}
    .icu-main{display:grid;gap:14px}
    .icu-monitor{
      border:1px solid rgba(255,255,255,.10);border-radius:22px;padding:16px;
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.08), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        #08101d;
      box-shadow:inset 0 0 0 1px rgba(255,255,255,.02),0 12px 30px rgba(0,0,0,.22),0 0 50px rgba(91,212,255,.05);
    }
    .critical-monitor{border-color:rgba(255,107,107,.26)}
    .watch-monitor{border-color:rgba(247,190,104,.24)}
    .stable-monitor{border-color:rgba(56,211,159,.24)}
    .icu-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}
    .icu-kicker{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6}
    .icu-title{margin-top:6px;font-size:22px;font-weight:1000;line-height:1;color:#eef4ff}
    .icu-state{
      display:inline-flex;align-items:center;justify-content:center;min-width:92px;padding:8px 10px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.05em;text-transform:uppercase;
    }
    .icu-state.critical{color:#ffd8d8;background:rgba(255,107,107,.16);border:1px solid rgba(255,107,107,.26)}
    .icu-state.warning{color:#ffe7bf;background:rgba(247,190,104,.16);border:1px solid rgba(247,190,104,.24)}
    .icu-state.stable{color:#dfffea;background:rgba(91,211,141,.14);border:1px solid rgba(91,211,141,.24)}
    .icu-screen{position:relative;height:220px;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.06);background:#050b14}
    .screen-grid{
      position:absolute;inset:0;background:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
      background-size:22px 22px,22px 22px,auto;pointer-events:none;
    }
    .ecg-svg{position:absolute;inset:0;width:100%;height:100%}
    .ecg-path{
      fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:18 8;animation:ecgMove 1.8s linear infinite;filter:drop-shadow(0 0 10px currentColor);
    }
    .critical-path{stroke:#ff6b6b;color:#ff6b6b}
    .warning-path{stroke:#f7be68;color:#f7be68}
    .stable-path{stroke:#38d39f;color:#38d39f}
    @keyframes ecgMove{
      from{stroke-dashoffset:0}
      to{stroke-dashoffset:-52}
    }
    .screen-readouts{position:absolute;left:14px;right:14px;bottom:14px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
    .readout{border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:12px 10px;background:rgba(255,255,255,.03);backdrop-filter:blur(4px)}
    .r-k{display:block;font-size:11px;letter-spacing:.10em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .r-v{display:block;margin-top:8px;font-size:22px;line-height:1;font-weight:1000;color:#eef4ff}
    .icu-side-rail{display:grid;gap:12px}
    .rail-card{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
    }
    .rail-k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .rail-v{font-size:34px;font-weight:1000;line-height:1;margin-top:10px;color:#ecf4ff}
    .rail-sub{margin-top:8px;font-size:13px;color:#c6d7ef;line-height:1.5}
    .proof-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}
    .proof-card{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:16px;
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),rgba(8,16,29,.82);
      box-shadow:0 12px 28px rgba(0,0,0,.16);
    }
    .proof-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9fdcff}
    .proof-v{margin-top:10px;font-size:28px;line-height:1;font-weight:1000;color:#eef4ff}
    .proof-sub{margin-top:8px;color:#c6d7ef;font-size:13px;line-height:1.5}
    .list{margin-top:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
    .mini{border:1px solid var(--line2);border-radius:18px;padding:16px;background:rgba(255,255,255,.03)}
    .mini .k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px}
    .mini .v{display:block;font-size:20px;font-weight:1000;line-height:1.1;letter-spacing:-.03em}
    .mini p{margin:10px 0 0;color:#c7d8ef;font-size:14px;line-height:1.6}
    .footer{margin-top:22px;padding:18px 0 8px;color:#9fb4d6;font-size:14px;line-height:1.6}
    @media (max-width:1100px){
      .hero-grid{grid-template-columns:1fr}
      .hero-mini-grid{grid-template-columns:repeat(2,1fr)}
      .route-grid{grid-template-columns:1fr}
      .list{grid-template-columns:1fr}
      .icu-wall{grid-template-columns:1fr}
      .proof-strip{grid-template-columns:repeat(2,minmax(0,1fr))}
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
      .hero-mini-grid{grid-template-columns:1fr}
      .section{padding:18px}
      .screen-readouts{grid-template-columns:repeat(2,minmax(0,1fr))}
      .proof-strip{grid-template-columns:1fr}
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
        <a href="#product-walkthrough">Live Platform</a>
        <a href="#commercial-path">Investors</a>
        <a href="#insurer-path">Insurers</a>
        <a href="#patient-path">Patients</a>
        <a href="#dashboard">Command Center</a>
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
            <h1>Predictive clinical intelligence that looks hospital-grade and sells in seconds.</h1>
            <p>
              Early Risk Alert AI is a real-time monitoring and escalation platform for hospitals, investors, insurers,
              and patients. The experience below shows live clinical visibility, commercial traction, operating flow,
              and the product story from one polished domain.
            </p>

            <div class="hero-actions">
              <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
              <a class="btn secondary" href="#hospital-story">Hospital Story</a>
              <a class="btn secondary" href="#commercial-path">Investor Story</a>
              <a class="btn secondary" href="#dashboard">Open Command Center</a>
            </div>

            <div class="hero-mini-grid">
              <div class="hero-mini">
                <div class="k">Hospitals</div>
                <div class="v">Operational command-center visibility</div>
              </div>
              <div class="hero-mini">
                <div class="k">Investors</div>
                <div class="v">Scalable platform with pipeline motion</div>
              </div>
              <div class="hero-mini">
                <div class="k">Insurers</div>
                <div class="v">Earlier intervention and lower-cost escalation</div>
              </div>
              <div class="hero-mini">
                <div class="k">Patients</div>
                <div class="v">Safer, faster, more proactive care response</div>
              </div>
            </div>
          </div>

          <div class="glass demo-card" id="demo">
            <div class="demo-stage">
              <div class="demo-badge">Flagship Demo</div>
              <div class="demo-play">
                <a class="demo-play-btn" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer" aria-label="Watch demo video">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M8 5v14l11-7z"></path>
                  </svg>
                </a>
              </div>
              <div class="demo-caption">
                <h3>Early Risk Alert AI Master Demo</h3>
                <p>See the product story, operating flow, live command-center visuals, and commercial readiness from one polished experience.</p>
              </div>
            </div>

            <div class="demo-bottom">
              <div class="demo-note">
                The platform below combines hospital-grade monitoring visuals, investor pipeline credibility, and live admin operations into one working product experience.
              </div>
              <div class="demo-btns">
                <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
                <a class="btn secondary" href="/hospital-demo">Hospital Demo Request</a>
                <a class="btn secondary" href="/investor-intake">Investor Intake</a>
                <a class="btn secondary" href="/admin/review">Admin Review</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="route-grid">
      <div class="route-card" id="hospital-story">
        <div class="route-label">Hospital Story</div>
        <h3>Clinical buyer path</h3>
        <p>Show operational monitoring, patient prioritization, escalation hierarchy, and a hospital-grade control environment that feels deployment-ready.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#dashboard">View Command Center</a>
          <a class="btn secondary" href="/hospital-demo">Request Hospital Demo</a>
        </div>
      </div>

      <div class="route-card" id="commercial-path">
        <div class="route-label">Investor Story</div>
        <h3>Commercial path</h3>
        <p>Investors see a scalable AI platform with live operating flow, request capture, analytics, and a clear product narrative tied to real buyer demand.</p>
        <div class="route-actions">
          <a class="btn secondary" href="/investor-intake">Open Investor Intake</a>
          <a class="btn secondary" href="/admin/review">View Pipeline</a>
        </div>
      </div>

      <div class="route-card" id="product-walkthrough">
        <div class="route-label">Live Platform</div>
        <h3>Product walkthrough</h3>
        <p>Display command-center visuals, telemetry-style monitors, risk scoring, alert state, and operating metrics that make the product understandable in under 10 seconds.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#dashboard">Open Live Platform</a>
          <a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a>
        </div>
      </div>
    </div>

    <section class="section" id="dashboard">
      <div class="section-kicker">Flagship AI Hospital Command Center</div>
      <h2>Hospital-grade ICU control-wall visuals bundled with live platform metrics.</h2>
      <p>
        The command center updates automatically and gives hospitals, investors, insurers, and patients an immediate sense
        of urgency, visibility, and real clinical operating value.
      </p>

      <div class="ticker-wrap">
        <div class="ticker-head">
          <div class="ticker-title">Live Clinical Activity Ticker</div>
          <div class="ticker-live">Live Stream</div>
        </div>
        <div class="ticker-track">
          <div class="ticker-move" id="home-ticker">
            <div class="ticker-pill"><span class="dot critical"></span> Critical deterioration signal surfaced</div>
            <div class="ticker-pill"><span class="dot high"></span> High-risk escalation routed to care team</div>
            <div class="ticker-pill"><span class="dot stable"></span> Stable patient trend confirmed</div>
            <div class="ticker-pill"><span class="dot critical"></span> ICU watchlist refreshed</div>
            <div class="ticker-pill"><span class="dot high"></span> Clinical operations dashboard synced</div>
          </div>
        </div>
      </div>

      <div class="icu-wall">
        <div class="icu-main">
          <div class="icu-monitor critical-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed A</div>
                <div class="icu-title">Critical Deterioration Monitor</div>
              </div>
              <div class="icu-state critical" id="home-monitor-a-status">Critical</div>
            </div>
            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                <path class="ecg-path critical-path"
                  d="M0 130 L40 130 L70 130 L90 130 L110 130 L130 130 L150 130 L170 130 L190 130
                     L210 130 L225 110 L240 145 L255 60 L270 185 L285 128
                     L300 130 L340 130 L380 130 L420 130 L435 112 L450 145 L465 65 L480 185 L495 128
                     L510 130 L550 130 L590 130 L630 130 L645 112 L660 145 L675 58 L690 190 L705 128
                     L720 130 L760 130 L800 130 L840 130 L855 114 L870 142 L885 63 L900 186 L915 128
                     L930 130 L970 130 L1010 130 L1050 130 L1065 112 L1080 144 L1095 60 L1110 184 L1125 128
                     L1140 130 L1200 130" />
              </svg>
              <div class="screen-readouts">
                <div class="readout"><span class="r-k">HR</span><span class="r-v" id="home-monitor-a-hr">128</span></div>
                <div class="readout"><span class="r-k">SpO₂</span><span class="r-v" id="home-monitor-a-spo2">89</span></div>
                <div class="readout"><span class="r-k">BP</span><span class="r-v" id="home-monitor-a-bp">164/98</span></div>
                <div class="readout"><span class="r-k">Risk</span><span class="r-v" id="home-monitor-a-risk">9.1</span></div>
              </div>
            </div>
          </div>

          <div class="icu-monitor watch-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed B</div>
                <div class="icu-title">Escalation Watch Monitor</div>
              </div>
              <div class="icu-state warning" id="home-monitor-b-status">High</div>
            </div>
            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                <path class="ecg-path warning-path"
                  d="M0 132 L50 132 L90 132 L130 132 L170 132 L210 132 L225 120 L240 140 L255 92 L270 165 L285 132
                     L300 132 L350 132 L400 132 L450 132 L465 118 L480 140 L495 96 L510 162 L525 132
                     L540 132 L590 132 L640 132 L690 132 L705 120 L720 138 L735 98 L750 162 L765 132
                     L780 132 L830 132 L880 132 L930 132 L945 120 L960 140 L975 95 L990 164 L1005 132
                     L1020 132 L1080 132 L1140 132 L1200 132" />
              </svg>
              <div class="screen-readouts">
                <div class="readout"><span class="r-k">HR</span><span class="r-v" id="home-monitor-b-hr">112</span></div>
                <div class="readout"><span class="r-k">SpO₂</span><span class="r-v" id="home-monitor-b-spo2">93</span></div>
                <div class="readout"><span class="r-k">BP</span><span class="r-v" id="home-monitor-b-bp">148/90</span></div>
                <div class="readout"><span class="r-k">Risk</span><span class="r-v" id="home-monitor-b-risk">8.2</span></div>
              </div>
            </div>
          </div>

          <div class="icu-monitor stable-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed C</div>
                <div class="icu-title">Stable Recovery Monitor</div>
              </div>
              <div class="icu-state stable" id="home-monitor-c-status">Stable</div>
            </div>
            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                <path class="ecg-path stable-path"
                  d="M0 140 L40 140 L80 140 L120 140 L160 140
                     L200 140 L218 134 L236 146 L254 112 L272 156 L290 140
                     L330 140 L370 140 L410 140
                     L450 140 L468 134 L486 146 L504 112 L522 156 L540 140
                     L580 140 L620 140 L660 140
                     L700 140 L718 134 L736 146 L754 112 L772 156 L790 140
                     L830 140 L870 140 L910 140
                     L950 140 L968 134 L986 146 L1004 112 L1022 156 L1040 140
                     L1080 140 L1120 140 L1200 140" />
              </svg>
              <div class="screen-readouts">
                <div class="readout"><span class="r-k">HR</span><span class="r-v" id="home-monitor-c-hr">82</span></div>
                <div class="readout"><span class="r-k">SpO₂</span><span class="r-v" id="home-monitor-c-spo2">98</span></div>
                <div class="readout"><span class="r-k">BP</span><span class="r-v" id="home-monitor-c-bp">122/78</span></div>
                <div class="readout"><span class="r-k">Risk</span><span class="r-v" id="home-monitor-c-risk">3.4</span></div>
              </div>
            </div>
          </div>
        </div>

        <div class="icu-side-rail">
          <div class="rail-card"><div class="rail-k">Open Alerts</div><div class="rail-v" id="home-open-alerts">0</div><div class="rail-sub">Active alerts surfaced by the command center.</div></div>
          <div class="rail-card"><div class="rail-k">Critical Alerts</div><div class="rail-v" id="home-critical-alerts">0</div><div class="rail-sub">Highest urgency patients requiring immediate intervention.</div></div>
          <div class="rail-card"><div class="rail-k">Avg Risk Score</div><div class="rail-v" id="home-avg-risk">0.0</div><div class="rail-sub">Average intelligence-layer risk level across flagged patients.</div></div>
          <div class="rail-card"><div class="rail-k">Patients With Alerts</div><div class="rail-v" id="home-patients-alerts">0</div><div class="rail-sub">Patients actively surfaced by AI monitoring.</div></div>
        </div>
      </div>

      <div class="proof-strip">
        <div class="proof-card"><div class="proof-k">Hospitals</div><div class="proof-v">Operational</div><div class="proof-sub">Looks like a real command center, not a generic dashboard.</div></div>
        <div class="proof-card" id="insurer-path"><div class="proof-k">Insurers</div><div class="proof-v">Predictive</div><div class="proof-sub">Shows earlier intervention, triage value, and scalable monitoring.</div></div>
        <div class="proof-card" id="patient-path"><div class="proof-k">Patients</div><div class="proof-v">Protective</div><div class="proof-sub">Feels safer, more proactive, and clinically reassuring.</div></div>
        <div class="proof-card"><div class="proof-k">Investors</div><div class="proof-v">Credible</div><div class="proof-sub">Communicates product maturity, urgency, and platform value immediately.</div></div>
      </div>
    </section>

    <section class="section">
      <div class="section-kicker">Audience Narrative</div>
      <h2>One platform story, tailored by audience.</h2>
      <p>
        Hospitals see operating value. Investors see scale and product readiness. Insurers see predictive cost control.
        Patients see confidence, safety, and faster escalation when risk rises.
      </p>

      <div class="list">
        <div class="mini">
          <div class="k">Hospitals</div>
          <span class="v">Real-time visibility for care operations</span>
          <p>Monitor high-risk patients, prioritize intervention, and give teams a command-center view of deterioration signals.</p>
        </div>
        <div class="mini">
          <div class="k">Investors</div>
          <span class="v">Scalable platform with live pipeline flow</span>
          <p>See a product that combines market relevance, buyer intake, operating analytics, and compelling visual differentiation.</p>
        </div>
        <div class="mini">
          <div class="k">Insurers & Patients</div>
          <span class="v">Earlier intervention, safer outcomes</span>
          <p>Predictive monitoring supports lower-cost escalations, better coordination, and stronger confidence in ongoing patient care.</p>
        </div>
      </div>
    </section>

    <section class="section" id="contact">
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

  <script>
    function setMonitor(prefix, alert, stableFallback) {
      const severity = (alert.severity || stableFallback || "stable").toLowerCase();
      const statusText = severity === "critical" ? "Critical" : (severity === "high" ? "High" : "Stable");

      const statusEl = document.getElementById(prefix + "-status");
      const hrEl = document.getElementById(prefix + "-hr");
      const spo2El = document.getElementById(prefix + "-spo2");
      const bpEl = document.getElementById(prefix + "-bp");
      const riskEl = document.getElementById(prefix + "-risk");

      statusEl.textContent = statusText;
      statusEl.className = "icu-state " + (severity === "critical" ? "critical" : (severity === "high" ? "warning" : "stable"));

      const baseRisk = Number(alert.risk_score || (severity === "critical" ? 9.1 : (severity === "high" ? 7.9 : 3.4)));
      const hr = severity === "critical" ? 124 + Math.floor(Math.random() * 10) : (severity === "high" ? 108 + Math.floor(Math.random() * 10) : 78 + Math.floor(Math.random() * 8));
      const spo2 = severity === "critical" ? 87 + Math.floor(Math.random() * 3) : (severity === "high" ? 92 + Math.floor(Math.random() * 3) : 97 + Math.floor(Math.random() * 2));
      const sys = severity === "critical" ? 160 + Math.floor(Math.random() * 10) : (severity === "high" ? 145 + Math.floor(Math.random() * 8) : 120 + Math.floor(Math.random() * 6));
      const dia = severity === "critical" ? 96 + Math.floor(Math.random() * 6) : (severity === "high" ? 88 + Math.floor(Math.random() * 4) : 76 + Math.floor(Math.random() * 4));

      hrEl.textContent = hr;
      spo2El.textContent = spo2;
      bpEl.textContent = sys + "/" + dia;
      riskEl.textContent = baseRisk.toFixed(1);
    }

    async function refreshHomeDashboard() {
      try {
        const overview = await fetch("/api/v1/dashboard/overview?tenant_id=demo&refresh=" + Date.now(), { cache: "no-store" });
        if (overview.ok) {
          const data = await overview.json();
          document.getElementById("home-open-alerts").textContent = data.open_alerts ?? 0;
          document.getElementById("home-critical-alerts").textContent = data.critical_alerts ?? 0;
          document.getElementById("home-avg-risk").textContent = Number(data.avg_risk_score ?? 0).toFixed(1);
          document.getElementById("home-patients-alerts").textContent = data.patients_with_alerts ?? 0;
        }
      } catch (e) {
        console.error("overview refresh failed", e);
      }

      try {
        const snapshot = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), { cache: "no-store" });
        if (snapshot.ok) {
          const data = await snapshot.json();
          const alerts = Array.isArray(data.alerts) ? data.alerts : [];
          const ticker = document.getElementById("home-ticker");

          if (alerts.length) {
            ticker.innerHTML = alerts.slice(0, 6).map(function(a) {
              const sev = (a.severity || "").toLowerCase();
              const dot = sev === "critical" ? "critical" : (sev === "high" ? "high" : "stable");
              return '<div class="ticker-pill"><span class="dot ' + dot + '"></span>' + (a.title || "Clinical alert") + ' · ' + (a.patient_id || "patient") + ' · Risk ' + Number(a.risk_score || 0).toFixed(1) + '</div>';
            }).join("");
          }

          setMonitor("home-monitor-a", alerts[0] || {severity:"critical", risk_score:9.2}, "critical");
          setMonitor("home-monitor-b", alerts[1] || {severity:"high", risk_score:8.1}, "high");
          setMonitor("home-monitor-c", alerts[2] || {severity:"stable", risk_score:3.4}, "stable");
        }
      } catch (e) {
        console.error("snapshot refresh failed", e);
      }
    }

    refreshHomeDashboard();
    setInterval(refreshHomeDashboard, 5000);
  </script>
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
    .toolbar{
      margin-top:18px;display:grid;grid-template-columns:1.2fr .8fr .8fr .8fr auto;gap:12px;align-items:end
    }
    .field{display:grid;gap:8px}
    .field label{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6}
    .field input,.field select{
      width:100%;padding:13px 14px;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:#eef4ff;font:inherit
    }
    .admin-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0 24px}
    .admin-kpi{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.06);
      border-radius:20px;padding:24px;min-height:154px;display:flex;flex-direction:column;justify-content:space-between;
    }
    .admin-kpi .k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .admin-kpi .v{font-size:44px;font-weight:1000;line-height:1;margin-top:14px}
    .admin-kpi .hint{font-size:13px;color:#c6d7ef;line-height:1.5}
    .ticker-wrap{
      margin:18px 0;border:1px solid rgba(255,255,255,.08);border-radius:18px;overflow:hidden;
      background:linear-gradient(90deg, rgba(91,212,255,.08), rgba(122,162,255,.04), rgba(255,255,255,.02)),rgba(8,16,29,.92);
    }
    .ticker-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06)}
    .ticker-title{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9fdcff}
    .ticker-live{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#dff7ff}
    .ticker-live::before{content:"";width:8px;height:8px;border-radius:999px;background:#38d39f;box-shadow:0 0 0 0 rgba(56,211,159,.55);animation:livePulse 1.8s infinite}
    @keyframes livePulse{0%{box-shadow:0 0 0 0 rgba(56,211,159,.55)}70%{box-shadow:0 0 0 12px rgba(56,211,159,0)}100%{box-shadow:0 0 0 0 rgba(56,211,159,0)}}
    .ticker-track{position:relative;overflow:hidden;white-space:nowrap;padding:14px 0}
    .ticker-move{display:inline-flex;gap:14px;padding-left:100%;animation:tickerMove 26s linear infinite}
    @keyframes tickerMove{from{transform:translateX(0)}to{transform:translateX(-100%)}}
    .ticker-pill{
      display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.04);color:#edf4ff;font-size:13px;font-weight:900;
    }
    .ticker-pill .dot{width:9px;height:9px;border-radius:999px;display:inline-block}
    .ticker-pill .dot.critical{background:#ff667d;box-shadow:0 0 10px rgba(255,102,125,.45)}
    .ticker-pill .dot.high{background:#f4bd6a;box-shadow:0 0 10px rgba(244,189,106,.35)}
    .ticker-pill .dot.stable{background:#38d39f;box-shadow:0 0 10px rgba(56,211,159,.35)}
    .icu-wall{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin:18px 0}
    .icu-main{display:grid;gap:14px}
    .icu-monitor{
      border:1px solid rgba(255,255,255,.10);border-radius:22px;padding:16px;
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.08), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        #08101d;
      box-shadow:inset 0 0 0 1px rgba(255,255,255,.02),0 12px 30px rgba(0,0,0,.22),0 0 50px rgba(91,212,255,.05)
    }
    .critical-monitor{border-color:rgba(255,107,107,.26)}
    .watch-monitor{border-color:rgba(247,190,104,.24)}
    .stable-monitor{border-color:rgba(56,211,159,.24)}
    .icu-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}
    .icu-kicker{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6}
    .icu-title{margin-top:6px;font-size:22px;font-weight:1000;line-height:1;color:#eef4ff}
    .icu-state{display:inline-flex;align-items:center;justify-content:center;min-width:92px;padding:8px 10px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.05em;text-transform:uppercase}
    .icu-state.critical{color:#ffd8d8;background:rgba(255,107,107,.16);border:1px solid rgba(255,107,107,.26)}
    .icu-state.warning{color:#ffe7bf;background:rgba(247,190,104,.16);border:1px solid rgba(247,190,104,.24)}
    .icu-state.stable{color:#dfffea;background:rgba(91,211,141,.14);border:1px solid rgba(91,211,141,.24)}
    .icu-screen{position:relative;height:220px;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.06);background:#050b14}
    .screen-grid{
      position:absolute;inset:0;background:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
      background-size:22px 22px,22px 22px,auto;pointer-events:none
    }
    .ecg-svg{position:absolute;inset:0;width:100%;height:100%}
    .ecg-path{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:18 8;animation:ecgMove 1.8s linear infinite;filter:drop-shadow(0 0 10px currentColor)}
    .critical-path{stroke:#ff6b6b;color:#ff6b6b}
    .warning-path{stroke:#f7be68;color:#f7be68}
    .stable-path{stroke:#38d39f;color:#38d39f}
    @keyframes ecgMove{from{stroke-dashoffset:0}to{stroke-dashoffset:-52}}
    .screen-readouts{position:absolute;left:14px;right:14px;bottom:14px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
    .readout{border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:12px 10px;background:rgba(255,255,255,.03);backdrop-filter:blur(4px)}
    .r-k{display:block;font-size:11px;letter-spacing:.10em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .r-v{display:block;margin-top:8px;font-size:22px;line-height:1;font-weight:1000;color:#eef4ff}
    .icu-side-rail{display:grid;gap:12px}
    .rail-card{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px;
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02)),linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
    }
    .rail-k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .rail-v{font-size:34px;font-weight:1000;line-height:1;margin-top:10px;color:#ecf4ff}
    .rail-sub{margin-top:8px;font-size:13px;color:#c6d7ef;line-height:1.5}
    .status-legend{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
    .status-chip{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:#eef4ff}
    .status-chip .s-dot{width:9px;height:9px;border-radius:999px}
    .status-chip.new .s-dot{background:#7aa2ff;box-shadow:0 0 10px rgba(122,162,255,.45)}
    .status-chip.contacted .s-dot{background:#f4bd6a;box-shadow:0 0 10px rgba(244,189,106,.35)}
    .status-chip.closed .s-dot{background:#38d39f;box-shadow:0 0 10px rgba(56,211,159,.35)}
    .status-chip.interested .s-dot{background:#ba8cff;box-shadow:0 0 10px rgba(186,140,255,.35)}
    .status-chip.dd .s-dot{background:#ff667d;box-shadow:0 0 10px rgba(255,102,125,.35)}
    .lead-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px}
    .lead-type{
      flex:1 1 220px;background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:16px;
    }
    .lead-type h3{margin:0 0 8px;font-size:24px}
    .lead-type p{margin:0;color:#c9daf0;line-height:1.65}
    .section-block{margin-top:22px}
    .section-head{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:12px}
    .small-note{font-size:13px;color:#9eb4d6}
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
    .score{
      display:inline-flex;align-items:center;justify-content:center;min-width:64px;padding:8px 10px;border-radius:999px;background:rgba(122,162,255,.14);border:1px solid rgba(122,162,255,.22);font-weight:1000
    }
    .score.hot{background:rgba(255,102,125,.16);border-color:rgba(255,102,125,.28);color:#ffd8df}
    .score.warm{background:rgba(244,189,106,.16);border-color:rgba(244,189,106,.28);color:#ffe5bb}
    .score.cool{background:rgba(56,211,159,.16);border-color:rgba(56,211,159,.26);color:#d9ffec}
    .action-row{display:flex;gap:8px;flex-wrap:wrap}
    .mini-btn{
      display:inline-flex;align-items:center;justify-content:center;padding:9px 12px;border-radius:12px;background:#111b2f;border:1px solid rgba(255,255,255,.08);color:#eef4ff;font-size:12px;font-weight:1000;cursor:pointer
    }
    .meta-stack{display:grid;gap:6px}
    .meta-item{font-size:13px;color:#c6d7ef;line-height:1.5}
    .priority-tag{
      display:inline-flex;align-items:center;justify-content:center;padding:8px 10px;border-radius:999px;background:rgba(186,140,255,.14);border:1px solid rgba(186,140,255,.22);font-size:12px;font-weight:1000;color:#f0e4ff
    }
    .last-updated{font-size:12px;color:#9eb4d6;font-weight:900;letter-spacing:.08em;text-transform:uppercase}
    @media (max-width:1200px){
      .toolbar{grid-template-columns:1fr 1fr;align-items:stretch}
      .admin-grid{grid-template-columns:repeat(2,1fr)}
      .icu-wall{grid-template-columns:1fr}
    }
    @media (max-width:760px){
      .admin-grid{grid-template-columns:1fr}
      .toolbar{grid-template-columns:1fr}
      .screen-readouts{grid-template-columns:repeat(2,minmax(0,1fr))}
      .wrap{padding:14px 10px 48px}
      .card{padding:16px}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Admin Review</h1>
      <p>Live request operations, investor pipeline visibility, instant status updates, search, filtering, lead scoring, and hospital-grade command-center visuals.</p>
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
        <div class="admin-kpi"><div class="k">Last Updated</div><div class="v" style="font-size:24px" id="kpi-updated">--</div><div class="hint">Live admin refresh timestamp</div></div>
      </div>

      <div class="ticker-wrap">
        <div class="ticker-head">
          <div class="ticker-title">Live Clinical Activity Ticker</div>
          <div class="ticker-live">Live Stream</div>
        </div>
        <div class="ticker-track">
          <div class="ticker-move" id="admin-ticker">
            <div class="ticker-pill"><span class="dot critical"></span> Critical deterioration signal surfaced</div>
            <div class="ticker-pill"><span class="dot high"></span> High-risk escalation routed</div>
            <div class="ticker-pill"><span class="dot stable"></span> Stable patient trend confirmed</div>
            <div class="ticker-pill"><span class="dot critical"></span> ICU watchlist refreshed</div>
            <div class="ticker-pill"><span class="dot high"></span> Command center synced</div>
          </div>
        </div>
      </div>

      <div class="icu-wall">
        <div class="icu-main">
          <div class="icu-monitor critical-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed A</div>
                <div class="icu-title">Critical Deterioration Monitor</div>
              </div>
              <div class="icu-state critical" id="admin-monitor-a-status">Critical</div>
            </div>
            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                <path class="ecg-path critical-path"
                  d="M0 130 L40 130 L70 130 L90 130 L110 130 L130 130 L150 130 L170 130 L190 130
                     L210 130 L225 110 L240 145 L255 60 L270 185 L285 128
                     L300 130 L340 130 L380 130 L420 130 L435 112 L450 145 L465 65 L480 185 L495 128
                     L510 130 L550 130 L590 130 L630 130 L645 112 L660 145 L675 58 L690 190 L705 128
                     L720 130 L760 130 L800 130 L840 130 L855 114 L870 142 L885 63 L900 186 L915 128
                     L930 130 L970 130 L1010 130 L1050 130 L1065 112 L1080 144 L1095 60 L1110 184 L1125 128
                     L1140 130 L1200 130" />
              </svg>
              <div class="screen-readouts">
                <div class="readout"><span class="r-k">HR</span><span class="r-v" id="admin-monitor-a-hr">128</span></div>
                <div class="readout"><span class="r-k">SpO₂</span><span class="r-v" id="admin-monitor-a-spo2">89</span></div>
                <div class="readout"><span class="r-k">BP</span><span class="r-v" id="admin-monitor-a-bp">164/98</span></div>
                <div class="readout"><span class="r-k">Risk</span><span class="r-v" id="admin-monitor-a-risk">9.1</span></div>
              </div>
            </div>
          </div>

          <div class="icu-monitor watch-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed B</div>
                <div class="icu-title">Escalation Watch Monitor</div>
              </div>
              <div class="icu-state warning" id="admin-monitor-b-status">High</div>
            </div>
            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                <path class="ecg-path warning-path"
                  d="M0 132 L50 132 L90 132 L130 132 L170 132 L210 132 L225 120 L240 140 L255 92 L270 165 L285 132
                     L300 132 L350 132 L400 132 L450 132 L465 118 L480 140 L495 96 L510 162 L525 132
                     L540 132 L590 132 L640 132 L690 132 L705 120 L720 138 L735 98 L750 162 L765 132
                     L780 132 L830 132 L880 132 L930 132 L945 120 L960 140 L975 95 L990 164 L1005 132
                     L1020 132 L1080 132 L1140 132 L1200 132" />
              </svg>
              <div class="screen-readouts">
                <div class="readout"><span class="r-k">HR</span><span class="r-v" id="admin-monitor-b-hr">112</span></div>
                <div class="readout"><span class="r-k">SpO₂</span><span class="r-v" id="admin-monitor-b-spo2">93</span></div>
                <div class="readout"><span class="r-k">BP</span><span class="r-v" id="admin-monitor-b-bp">148/90</span></div>
                <div class="readout"><span class="r-k">Risk</span><span class="r-v" id="admin-monitor-b-risk">8.2</span></div>
              </div>
            </div>
          </div>

          <div class="icu-monitor stable-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed C</div>
                <div class="icu-title">Stable Recovery Monitor</div>
              </div>
              <div class="icu-state stable" id="admin-monitor-c-status">Stable</div>
            </div>
            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none">
                <path class="ecg-path stable-path"
                  d="M0 140 L40 140 L80 140 L120 140 L160 140
                     L200 140 L218 134 L236 146 L254 112 L272 156 L290 140
                     L330 140 L370 140 L410 140
                     L450 140 L468 134 L486 146 L504 112 L522 156 L540 140
                     L580 140 L620 140 L660 140
                     L700 140 L718 134 L736 146 L754 112 L772 156 L790 140
                     L830 140 L870 140 L910 140
                     L950 140 L968 134 L986 146 L1004 112 L1022 156 L1040 140
                     L1080 140 L1120 140 L1200 140" />
              </svg>
              <div class="screen-readouts">
                <div class="readout"><span class="r-k">HR</span><span class="r-v" id="admin-monitor-c-hr">82</span></div>
                <div class="readout"><span class="r-k">SpO₂</span><span class="r-v" id="admin-monitor-c-spo2">98</span></div>
                <div class="readout"><span class="r-k">BP</span><span class="r-v" id="admin-monitor-c-bp">122/78</span></div>
                <div class="readout"><span class="r-k">Risk</span><span class="r-v" id="admin-monitor-c-risk">3.4</span></div>
              </div>
            </div>
          </div>
        </div>

        <div class="icu-side-rail">
          <div class="rail-card"><div class="rail-k">Open Alerts</div><div class="rail-v" id="admin-open-alerts">0</div><div class="rail-sub">Active command-center alerts</div></div>
          <div class="rail-card"><div class="rail-k">Critical Alerts</div><div class="rail-v" id="admin-critical-alerts">0</div><div class="rail-sub">Immediate intervention cases</div></div>
          <div class="rail-card"><div class="rail-k">Avg Risk Score</div><div class="rail-v" id="admin-avg-risk">0.0</div><div class="rail-sub">Live intelligence-layer risk average</div></div>
          <div class="rail-card"><div class="rail-k">Patients With Alerts</div><div class="rail-v" id="admin-patients-alerts">0</div><div class="rail-sub">Flagged patients under monitoring</div></div>
        </div>
      </div>

      <div class="status-legend">
        <div class="status-chip new"><span class="s-dot"></span>New</div>
        <div class="status-chip contacted"><span class="s-dot"></span>Contacted</div>
        <div class="status-chip closed"><span class="s-dot"></span>Closed</div>
        <div class="status-chip interested"><span class="s-dot"></span>Interested</div>
        <div class="status-chip dd"><span class="s-dot"></span>Due Diligence</div>
      </div>

      <div class="lead-row">
        <div class="lead-type"><h3>Hospital Leads</h3><p>Clinical buyers, hospital operators, RPM teams, and health systems.</p></div>
        <div class="lead-type"><h3>Executive Leads</h3><p>Leadership-facing walkthroughs for pilots, reviews, and strategic evaluation.</p></div>
        <div class="lead-type"><h3>Investor Leads</h3><p>Investor pipeline stages, score tags, and commercial follow-up flow.</p></div>
      </div>

      <div class="section-block">
        <div class="section-head">
          <h2>All Requests</h2>
          <div class="last-updated" id="admin-last-updated">Last Updated --</div>
        </div>
        <div class="table-wrap">
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

    function setMonitor(prefix, alert, stableFallback) {
      const severity = (alert.severity || stableFallback || "stable").toLowerCase();
      const statusText = severity === "critical" ? "Critical" : (severity === "high" ? "High" : "Stable");

      const statusEl = document.getElementById(prefix + "-status");
      const hrEl = document.getElementById(prefix + "-hr");
      const spo2El = document.getElementById(prefix + "-spo2");
      const bpEl = document.getElementById(prefix + "-bp");
      const riskEl = document.getElementById(prefix + "-risk");

      statusEl.textContent = statusText;
      statusEl.className = "icu-state " + (severity === "critical" ? "critical" : (severity === "high" ? "warning" : "stable"));

      const baseRisk = Number(alert.risk_score || (severity === "critical" ? 9.1 : (severity === "high" ? 7.9 : 3.4)));
      const hr = severity === "critical" ? 124 + Math.floor(Math.random() * 10) : (severity === "high" ? 108 + Math.floor(Math.random() * 10) : 78 + Math.floor(Math.random() * 8));
      const spo2 = severity === "critical" ? 87 + Math.floor(Math.random() * 3) : (severity === "high" ? 92 + Math.floor(Math.random() * 3) : 97 + Math.floor(Math.random() * 2));
      const sys = severity === "critical" ? 160 + Math.floor(Math.random() * 10) : (severity === "high" ? 145 + Math.floor(Math.random() * 8) : 120 + Math.floor(Math.random() * 6));
      const dia = severity === "critical" ? 96 + Math.floor(Math.random() * 6) : (severity === "high" ? 88 + Math.floor(Math.random() * 4) : 76 + Math.floor(Math.random() * 4));

      hrEl.textContent = hr;
      spo2El.textContent = spo2;
      bpEl.textContent = sys + "/" + dia;
      riskEl.textContent = baseRisk.toFixed(1);
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
          document.getElementById("kpi-updated").textContent = data.summary.last_updated_label || "--";
          document.getElementById("admin-last-updated").textContent = "Last Updated " + (data.summary.last_updated || "--");

          applyFilters();
        }
      } catch (e) {
        console.error("admin data refresh failed", e);
      }

      try {
        const overview = await fetch("/api/v1/dashboard/overview?tenant_id=demo&refresh=" + Date.now(), { cache: "no-store" });
        if (overview.ok) {
          const data = await overview.json();
          document.getElementById("admin-open-alerts").textContent = data.open_alerts ?? 0;
          document.getElementById("admin-critical-alerts").textContent = data.critical_alerts ?? 0;
          document.getElementById("admin-avg-risk").textContent = Number(data.avg_risk_score ?? 0).toFixed(1);
          document.getElementById("admin-patients-alerts").textContent = data.patients_with_alerts ?? 0;
        }
      } catch (e) {
        console.error("admin overview refresh failed", e);
      }

      try {
        const snapshot = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), { cache: "no-store" });
        if (snapshot.ok) {
          const data = await snapshot.json();
          const alerts = Array.isArray(data.alerts) ? data.alerts : [];
          const ticker = document.getElementById("admin-ticker");

          if (alerts.length) {
            ticker.innerHTML = alerts.slice(0, 6).map(function(a) {
              const sev = (a.severity || "").toLowerCase();
              const dot = sev === "critical" ? "critical" : (sev === "high" ? "high" : "stable");
              return '<div class="ticker-pill"><span class="dot ' + dot + '"></span>' + (a.title || "Clinical alert") + ' · ' + (a.patient_id || "patient") + ' · Risk ' + Number(a.risk_score || 0).toFixed(1) + '</div>';
            }).join("");
          }

          setMonitor("admin-monitor-a", alerts[0] || {severity:"critical", risk_score:9.2}, "critical");
          setMonitor("admin-monitor-b", alerts[1] || {severity:"high", risk_score:8.1}, "high");
          setMonitor("admin-monitor-c", alerts[2] || {severity:"stable", risk_score:3.4}, "stable");
        }
      } catch (e) {
        console.error("admin snapshot refresh failed", e);
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
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _status_norm(status: str | None, lead_type: str) -> str:
    raw = (status or "").strip().lower()
    if lead_type == "investor":
        if raw in {"interested", "due diligence", "follow-up", "contacted", "closed"}:
            mapping = {
                "interested": "Interested",
                "due diligence": "Due Diligence",
                "follow-up": "Follow-Up",
                "contacted": "Contacted",
                "closed": "Closed",
            }
            return mapping[raw]
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

    route_map = {
        "hospital": hospital,
        "executive": executive,
        "investor": investor,
    }
    routed = route_map.get(lead_type, primary)
    recipients = {primary, routed, FOUNDER_EMAIL}
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


def _follow_up_templates() -> dict[str, str]:
    return {
        "hospital": (
            "Hospital Follow-Up Template:\n"
            "Thank you for your interest in Early Risk Alert AI. We would like to schedule a product demo focused on your clinical workflow, alert priorities, and operational goals."
        ),
        "executive": (
            "Executive Follow-Up Template:\n"
            "Thank you for requesting an executive walkthrough. We would like to align the review around your leadership goals, pilot readiness, and platform evaluation criteria."
        ),
        "investor": (
            "Investor Follow-Up Template:\n"
            "Thank you for your interest in Early Risk Alert AI. We would be glad to share product materials, platform visuals, and discuss fit, timing, and next-step diligence."
        ),
    }


def _build_demo_patients() -> list[dict[str, Any]]:
    return [
        {"patient_id": "p101", "name": "Patient A", "status": "Critical", "risk_score": round(8.7 + random.random() * 1.2, 1)},
        {"patient_id": "p102", "name": "Patient B", "status": "High", "risk_score": round(7.2 + random.random() * 1.1, 1)},
        {"patient_id": "p103", "name": "Patient C", "status": "Stable", "risk_score": round(3.1 + random.random() * 1.0, 1)},
        {"patient_id": "p104", "name": "Patient D", "status": "High", "risk_score": round(6.8 + random.random() * 1.0, 1)},
        {"patient_id": "p105", "name": "Patient E", "status": "Stable", "risk_score": round(3.3 + random.random() * 1.0, 1)},
    ]


def _build_alerts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in rows:
        risk = float(row.get("risk_score", 0))
        if risk >= 8.5:
            severity = "critical"
            title = "Critical deterioration signal"
        elif risk >= 6.5:
            severity = "high"
            title = "High-priority risk escalation"
        else:
            severity = "stable"
            title = "Stable patient trend"
        alerts.append({
            "patient_id": row["patient_id"],
            "title": title,
            "alert_type": title,
            "severity": severity,
            "risk_score": risk,
        })
    alerts.sort(key=lambda a: float(a.get("risk_score", 0)), reverse=True)
    return alerts


def _investor_stage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "New": 0,
        "Interested": 0,
        "Due Diligence": 0,
        "Follow-Up": 0,
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
            "follow_up_templates": _follow_up_templates(),
        }

    @app.get("/")
    def home():
        return render_template_string(HOME_HTML)

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

    @app.get("/admin/review")
    def admin_review():
        return render_template_string(ADMIN_HTML)

    @app.get("/admin/api/data")
    def admin_api_data():
        rows = build_admin_rows()
        summary = summary_payload()
        return jsonify({
            "rows": rows,
            "summary": summary,
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
        merged = build_admin_rows()
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

        for row in merged:
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
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        summary = summary_payload()

        avg_risk = 0.0
        if alerts:
            avg_risk = sum(float(a.get("risk_score", 0)) for a in alerts) / len(alerts)

        return jsonify({
            "tenant_id": request.args.get("tenant_id", "demo"),
            "patient_count": len(rows),
            "open_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "critical"),
            "events_last_hour": len(alerts) * 3,
            "avg_risk_score": round(avg_risk, 1),
            "patients_with_alerts": len({a.get("patient_id") for a in alerts}),
            "hospital_requests": summary["hospital_count"],
            "executive_requests": summary["executive_count"],
            "investor_requests": summary["investor_count"],
            "open_requests": summary["open_count"],
        })

    @app.get("/api/v1/live-snapshot")
    def live_snapshot():
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        rows = _build_demo_patients()
        alerts = _build_alerts(rows)
        focus = next((r for r in rows if r["patient_id"] == patient_id), rows[0] if rows else {})
        return jsonify({
            "tenant_id": tenant_id,
            "generated_at": _utc_now_iso(),
            "alerts": alerts[:6],
            "focus_patient": focus,
            "patients": rows,
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
