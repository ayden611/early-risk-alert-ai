from __future__ import annotations

import csv
import html
import io
import json
import os
import random
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file


INFO_EMAIL = "info@earlyriskalertai.com"
FOUNDER_EMAIL = "milton.munroe@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder & AI Systems Engineer"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/z4SbeYwwm7k"
PROD_BASE_URL = "https://early-risk-alert-ai-1.onrender.com"


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

    .ticker-wrap{
      margin-top:18px;
      margin-bottom:18px;
      border:1px solid rgba(255,255,255,.08);
      border-radius:18px;
      overflow:hidden;
      background:
        linear-gradient(90deg, rgba(91,212,255,.08), rgba(122,162,255,.04), rgba(255,255,255,.02)),
        rgba(8,16,29,.92);
      box-shadow:0 12px 28px rgba(0,0,0,.18);
    }
    .ticker-head{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      padding:12px 16px;
      border-bottom:1px solid rgba(255,255,255,.06);
    }
    .ticker-title{
      font-size:11px;
      font-weight:1000;
      letter-spacing:.16em;
      text-transform:uppercase;
      color:#9fdcff;
    }
    .ticker-live{
      display:inline-flex;
      align-items:center;
      gap:8px;
      font-size:11px;
      font-weight:900;
      letter-spacing:.12em;
      text-transform:uppercase;
      color:#dff7ff;
    }
    .ticker-live::before{
      content:"";
      width:8px;
      height:8px;
      border-radius:999px;
      background:#38d39f;
      box-shadow:0 0 0 0 rgba(56,211,159,.55);
      animation:livePulse 1.8s infinite;
    }
    @keyframes livePulse{
      0%{box-shadow:0 0 0 0 rgba(56,211,159,.55)}
      70%{box-shadow:0 0 0 12px rgba(56,211,159,0)}
      100%{box-shadow:0 0 0 0 rgba(56,211,159,0)}
    }
    .ticker-track{
      position:relative;
      overflow:hidden;
      white-space:nowrap;
      padding:14px 0;
    }
    .ticker-move{
      display:inline-flex;
      gap:14px;
      padding-left:100%;
      animation:tickerMove 24s linear infinite;
    }
    @keyframes tickerMove{
      from{transform:translateX(0)}
      to{transform:translateX(-100%)}
    }
    .ticker-pill{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding:10px 14px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.04);
      color:#edf4ff;
      font-size:13px;
      font-weight:900;
    }
    .ticker-pill .dot{
      width:9px;
      height:9px;
      border-radius:999px;
      display:inline-block;
    }
    .ticker-pill .dot.critical{background:#ff667d;box-shadow:0 0 10px rgba(255,102,125,.45)}
    .ticker-pill .dot.high{background:#f4bd6a;box-shadow:0 0 10px rgba(244,189,106,.35)}
    .ticker-pill .dot.stable{background:#38d39f;box-shadow:0 0 10px rgba(56,211,159,.35)}

    .icu-wall{
      display:grid;
      grid-template-columns:1.15fr .85fr;
      gap:16px;
      margin-top:18px;
      margin-bottom:18px;
    }
    .icu-main{
      display:grid;
      gap:14px;
    }
    .icu-monitor{
      border:1px solid rgba(255,255,255,.10);
      border-radius:22px;
      padding:16px;
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.08), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        #08101d;
      box-shadow:
        inset 0 0 0 1px rgba(255,255,255,.02),
        0 12px 30px rgba(0,0,0,.22),
        0 0 50px rgba(91,212,255,.05);
    }
    .critical-monitor{border-color:rgba(255,107,107,.26)}
    .watch-monitor{border-color:rgba(247,190,104,.24)}
    .icu-head{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      margin-bottom:12px;
    }
    .icu-kicker{
      font-size:11px;
      font-weight:900;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#9eb4d6;
    }
    .icu-title{
      margin-top:6px;
      font-size:22px;
      font-weight:1000;
      line-height:1;
      color:#eef4ff;
    }
    .icu-state{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:92px;
      padding:8px 10px;
      border-radius:999px;
      font-size:12px;
      font-weight:900;
      letter-spacing:.05em;
      text-transform:uppercase;
    }
    .icu-state.critical{
      color:#ffd8d8;
      background:rgba(255,107,107,.16);
      border:1px solid rgba(255,107,107,.26);
    }
    .icu-state.warning{
      color:#ffe7bf;
      background:rgba(247,190,104,.16);
      border:1px solid rgba(247,190,104,.24);
    }
    .icu-state.stable{
      color:#dfffea;
      background:rgba(91,211,141,.14);
      border:1px solid rgba(91,211,141,.24);
    }
    .icu-screen{
      position:relative;
      height:240px;
      border-radius:18px;
      overflow:hidden;
      border:1px solid rgba(255,255,255,.06);
      background:#050b14;
    }
    .screen-grid{
      position:absolute;
      inset:0;
      background:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
      background-size:22px 22px, 22px 22px, auto;
      pointer-events:none;
    }
    .ecg-svg{
      position:absolute;
      inset:0;
      width:100%;
      height:100%;
    }
    .ecg-path{
      fill:none;
      stroke-width:4;
      stroke-linecap:round;
      stroke-linejoin:round;
      stroke-dasharray:18 8;
      animation:ecgMove 1.8s linear infinite;
      filter:drop-shadow(0 0 10px currentColor);
    }
    .critical-path{
      stroke:#ff6b6b;
      color:#ff6b6b;
    }
    .warning-path{
      stroke:#f7be68;
      color:#f7be68;
    }
    @keyframes ecgMove{
      from{stroke-dashoffset:0}
      to{stroke-dashoffset:-52}
    }
    .screen-readouts{
      position:absolute;
      left:14px;
      right:14px;
      bottom:14px;
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:10px;
    }
    .readout{
      border:1px solid rgba(255,255,255,.08);
      border-radius:14px;
      padding:12px 10px;
      background:rgba(255,255,255,.03);
      backdrop-filter:blur(4px);
    }
    .r-k{
      display:block;
      font-size:11px;
      letter-spacing:.10em;
      text-transform:uppercase;
      color:#9eb4d6;
      font-weight:900;
    }
    .r-v{
      display:block;
      margin-top:8px;
      font-size:22px;
      line-height:1;
      font-weight:1000;
      color:#eef4ff;
    }
    .icu-side-rail{
      display:grid;
      gap:12px;
    }
    .rail-card{
      border:1px solid rgba(255,255,255,.08);
      border-radius:18px;
      padding:18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
    }
    .rail-k{
      font-size:12px;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#9eb4d6;
      font-weight:900;
    }
    .rail-v{
      font-size:34px;
      font-weight:1000;
      line-height:1;
      margin-top:10px;
      color:#ecf4ff;
    }
    .rail-sub{
      margin-top:8px;
      font-size:13px;
      color:#c6d7ef;
      line-height:1.5;
    }

    .proof-strip{
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:12px;
      margin-top:18px;
    }
    .proof-card{
      border:1px solid rgba(255,255,255,.08);
      border-radius:18px;
      padding:16px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        rgba(8,16,29,.82);
      box-shadow:0 12px 28px rgba(0,0,0,.16);
    }
    .proof-k{
      font-size:11px;
      font-weight:900;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#9fdcff;
    }
    .proof-v{
      margin-top:10px;
      font-size:28px;
      line-height:1;
      font-weight:1000;
      color:#eef4ff;
    }
    .proof-sub{
      margin-top:8px;
      color:#c6d7ef;
      font-size:13px;
      line-height:1.5;
    }

    .status-legend{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-top:16px;
    }
    .status-chip{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:10px 14px;
      border-radius:999px;
      font-size:12px;
      font-weight:900;
      letter-spacing:.06em;
      text-transform:uppercase;
      border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.03);
      color:#eef4ff;
    }
    .status-chip .s-dot{
      width:9px;
      height:9px;
      border-radius:999px;
    }
    .status-chip.new .s-dot{background:#7aa2ff;box-shadow:0 0 10px rgba(122,162,255,.45)}
    .status-chip.contacted .s-dot{background:#f4bd6a;box-shadow:0 0 10px rgba(244,189,106,.35)}
    .status-chip.closed .s-dot{background:#38d39f;box-shadow:0 0 10px rgba(56,211,159,.35)}

    .stream-list{
      display:grid;
      gap:10px;
      margin-top:18px;
    }
    .stream-item{
      border:1px solid rgba(255,255,255,.08);
      border-radius:16px;
      padding:14px 16px;
      background:rgba(255,255,255,.03);
    }
    .stream-item .name{
      font-size:15px;
      font-weight:1000;
      display:flex;
      align-items:center;
      gap:10px;
      color:#eef4ff;
    }
    .stream-item .meta{
      margin-top:6px;
      font-size:13px;
      color:#bdd0ea;
      line-height:1.5;
    }
    .alert-dot{
      width:9px;
      height:9px;
      border-radius:999px;
      background:#7aa2ff;
      display:inline-block;
      box-shadow:0 0 8px rgba(122,162,255,.45);
    }
    .alert-dot.warn{
      background:#f4bd6a;
      box-shadow:0 0 8px rgba(244,189,106,.35);
    }
    .alert-dot.danger{
      background:#ff667d;
      box-shadow:0 0 8px rgba(255,102,125,.45);
    }

    .footer{
      margin-top:22px;
      padding:18px 0 8px;
      color:#9fb4d6;
      font-size:14px;
      line-height:1.6;
    }

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
            <h1>Predictive clinical intelligence built for hospitals, investors, insurers, and patients.</h1>
            <p>
              Early Risk Alert AI is a professional predictive clinical intelligence platform with live command-center visuals,
              alert intelligence, investor pipeline tracking, and enterprise-style operating visibility from one branded experience.
            </p>

            <div class="hero-actions">
              <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
              <a class="btn secondary" href="#hospital-story">Hospital Story</a>
              <a class="btn secondary" href="#commercial-path">Investor Story</a>
              <a class="btn secondary" href="#dashboard">Command Center</a>
            </div>

            <div class="hero-mini-grid">
              <div class="hero-mini">
                <div class="k">Clinical Visibility</div>
                <div class="v">Hospital command-center intelligence</div>
              </div>
              <div class="hero-mini">
                <div class="k">Alerting</div>
                <div class="v">Severity, confidence, escalation flow</div>
              </div>
              <div class="hero-mini">
                <div class="k">Commercial</div>
                <div class="v">Investor and partnership pipeline</div>
              </div>
              <div class="hero-mini">
                <div class="k">Operations</div>
                <div class="v">Live admin review and request flow</div>
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
                <p>Open the platform story, command center, admin review, and commercial path from one clean production experience.</p>
              </div>
            </div>

            <div class="demo-bottom">
              <div class="demo-note">
                This demo opening gives hospitals and investors a polished, high-trust first impression and directs them into the strongest platform views.
              </div>
              <div class="demo-btns">
                <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
                <a class="btn secondary" href="#dashboard">Live Command Center</a>
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
        <h3>Clinical Buyer Path</h3>
        <p>Show live risk visibility, telemetry-style monitoring, prioritization logic, and operational value for hospitals and care teams.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#dashboard">Open Command Center</a>
          <a class="btn secondary" href="/hospital-demo">Request Hospital Demo</a>
        </div>
      </div>

      <div class="route-card" id="product-walkthrough">
        <div class="route-label">Live Platform</div>
        <h3>Product Walkthrough</h3>
        <p>Present ICU-style monitoring, alert flow, patient-level focus, and live admin visibility from one command wall.</p>
        <div class="route-actions">
          <a class="btn secondary" href="#dashboard">View Live Platform</a>
          <a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a>
        </div>
      </div>

      <div class="route-card" id="commercial-path">
        <div class="route-label">Investor Story</div>
        <h3>Commercial Path</h3>
        <p>Guide investors through traction, product visuals, pipeline activity, operating readiness, and enterprise SaaS positioning.</p>
        <div class="route-actions">
          <a class="btn secondary" href="/investor-intake">Investor Intake</a>
          <a class="btn secondary" href="/admin/review">View Pipeline</a>
        </div>
      </div>
    </div>

    <section class="section" id="dashboard">
      <div class="section-kicker">Live Clinical Command Center</div>
      <h2>Hospital-grade visual monitoring with real command-center presence.</h2>
      <p>
        Live alerts, patient status signals, animated ICU telemetry visuals, and operating metrics update automatically so hospitals,
        investors, and insurers understand the platform in seconds.
      </p>

      <div class="ticker-wrap">
        <div class="ticker-head">
          <div class="ticker-title">Live Clinical Activity Ticker</div>
          <div class="ticker-live">Live Stream</div>
        </div>
        <div class="ticker-track">
          <div class="ticker-move" id="cc-ticker">
            <div class="ticker-pill"><span class="dot critical"></span> Critical risk escalation detected</div>
            <div class="ticker-pill"><span class="dot high"></span> Care team routing updated</div>
            <div class="ticker-pill"><span class="dot stable"></span> Stable patient trend confirmed</div>
            <div class="ticker-pill"><span class="dot critical"></span> ICU watchlist refreshed</div>
            <div class="ticker-pill"><span class="dot high"></span> Executive dashboard synced</div>
          </div>
        </div>
      </div>

      <div class="icu-wall">
        <div class="icu-main">
          <div class="icu-monitor critical-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed A</div>
                <div class="icu-title">Critical Patient Monitor</div>
              </div>
              <div class="icu-state critical" id="monitor-a-status">Critical</div>
            </div>

            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none" aria-hidden="true">
                <path
                  id="ecg-a"
                  class="ecg-path critical-path"
                  d="M0 130
                     L40 130 L70 130 L90 130
                     L110 130 L130 130 L150 130
                     L170 130 L190 130
                     L210 130 L225 110 L240 145 L255 60 L270 185 L285 128
                     L300 130 L340 130 L380 130
                     L420 130 L435 112 L450 145 L465 65 L480 185 L495 128
                     L510 130 L550 130 L590 130
                     L630 130 L645 112 L660 145 L675 58 L690 190 L705 128
                     L720 130 L760 130 L800 130
                     L840 130 L855 114 L870 142 L885 63 L900 186 L915 128
                     L930 130 L970 130 L1010 130
                     L1050 130 L1065 112 L1080 144 L1095 60 L1110 184 L1125 128
                     L1140 130 L1200 130" />
              </svg>

              <div class="screen-readouts">
                <div class="readout">
                  <span class="r-k">HR</span>
                  <span class="r-v" id="monitor-a-hr">128</span>
                </div>
                <div class="readout">
                  <span class="r-k">SpO₂</span>
                  <span class="r-v" id="monitor-a-spo2">89</span>
                </div>
                <div class="readout">
                  <span class="r-k">BP</span>
                  <span class="r-v" id="monitor-a-bp">164/98</span>
                </div>
                <div class="readout">
                  <span class="r-k">Risk</span>
                  <span class="r-v" id="monitor-a-risk">9.1</span>
                </div>
              </div>
            </div>
          </div>

          <div class="icu-monitor watch-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed B</div>
                <div class="icu-title">Escalation Watch Monitor</div>
              </div>
              <div class="icu-state warning" id="monitor-b-status">High</div>
            </div>

            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none" aria-hidden="true">
                <path
                  id="ecg-b"
                  class="ecg-path warning-path"
                  d="M0 132
                     L50 132 L90 132 L130 132 L170 132
                     L210 132 L225 120 L240 140 L255 92 L270 165 L285 132
                     L300 132 L350 132 L400 132
                     L450 132 L465 118 L480 140 L495 96 L510 162 L525 132
                     L540 132 L590 132 L640 132
                     L690 132 L705 120 L720 138 L735 98 L750 162 L765 132
                     L780 132 L830 132 L880 132
                     L930 132 L945 120 L960 140 L975 95 L990 164 L1005 132
                     L1020 132 L1080 132 L1140 132 L1200 132" />
              </svg>

              <div class="screen-readouts">
                <div class="readout">
                  <span class="r-k">HR</span>
                  <span class="r-v" id="monitor-b-hr">112</span>
                </div>
                <div class="readout">
                  <span class="r-k">SpO₂</span>
                  <span class="r-v" id="monitor-b-spo2">93</span>
                </div>
                <div class="readout">
                  <span class="r-k">BP</span>
                  <span class="r-v" id="monitor-b-bp">148/90</span>
                </div>
                <div class="readout">
                  <span class="r-k">Risk</span>
                  <span class="r-v" id="monitor-b-risk">8.2</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="icu-side-rail">
          <div class="rail-card">
            <div class="rail-k">Open Alerts</div>
            <div class="rail-v" id="cc-open-alerts">0</div>
            <div class="rail-sub">Live platform alerts across the command center.</div>
          </div>

          <div class="rail-card">
            <div class="rail-k">Critical Alerts</div>
            <div class="rail-v" id="cc-critical-alerts">0</div>
            <div class="rail-sub">Highest urgency cases requiring immediate response.</div>
          </div>

          <div class="rail-card">
            <div class="rail-k">Avg Risk Score</div>
            <div class="rail-v" id="cc-avg-risk">0.0</div>
            <div class="rail-sub">Average risk severity from the intelligence layer.</div>
          </div>

          <div class="rail-card">
            <div class="rail-k">Patients With Alerts</div>
            <div class="rail-v" id="cc-patients-alerts">0</div>
            <div class="rail-sub">Patients currently surfaced by AI monitoring.</div>
          </div>
        </div>
      </div>

      <div class="proof-strip">
        <div class="proof-card">
          <div class="proof-k">Hospital Story</div>
          <div class="proof-v">Operational</div>
          <div class="proof-sub">Shows a real command-center environment for hospitals and RPM teams.</div>
        </div>
        <div class="proof-card">
          <div class="proof-k">Investor Story</div>
          <div class="proof-v">Visual</div>
          <div class="proof-sub">Communicates product value in seconds with immediate platform credibility.</div>
        </div>
        <div class="proof-card">
          <div class="proof-k">Insurer Story</div>
          <div class="proof-v">Predictive</div>
          <div class="proof-sub">Supports scalable monitoring, escalation, and lower-cost intervention.</div>
        </div>
        <div class="proof-card">
          <div class="proof-k">Patient Story</div>
          <div class="proof-v">Protective</div>
          <div class="proof-sub">Feels proactive, responsive, and clinically reassuring.</div>
        </div>
      </div>

      <div class="stream-list" id="cc-alert-stream">
        <div class="stream-item">
          <div class="name">Waiting for live stream data</div>
          <div class="meta">The command center updates automatically from the platform APIs.</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-kicker">Lead Pipeline Visibility</div>
      <h2>Request flow and commercial momentum from one operating layer.</h2>
      <p>
        Live lead visibility, admin review, investor pipeline analytics, and status tracking turn the platform from a demo into a polished operating system.
      </p>

      <div class="list">
        <div class="mini">
          <div class="k">Hospital Demo Requests</div>
          <span class="v" id="mini-hospital-count">0</span>
          <p>Clinical buyer interest flowing through hospital demo intake.</p>
        </div>
        <div class="mini">
          <div class="k">Executive Walkthroughs</div>
          <span class="v" id="mini-exec-count">0</span>
          <p>Leadership-facing product reviews, pilots, and strategic evaluations.</p>
        </div>
        <div class="mini">
          <div class="k">Investor Intake</div>
          <span class="v" id="mini-investor-count">0</span>
          <p>Commercial conversations, follow-up interest, and financing pipeline flow.</p>
        </div>
      </div>

      <div class="status-legend">
        <div class="status-chip new"><span class="s-dot"></span>New</div>
        <div class="status-chip contacted"><span class="s-dot"></span>Contacted</div>
        <div class="status-chip closed"><span class="s-dot"></span>Closed</div>
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
      Early Risk Alert AI LLC · Predictive clinical intelligence platform · Hospitals · Clinics · Investors · Patients
    </div>
  </div>

  <script>
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
          document.getElementById("cc-patients-alerts").textContent = data.patients_with_alerts ?? 0;
          document.getElementById("mini-hospital-count").textContent = data.hospital_requests ?? 0;
          document.getElementById("mini-exec-count").textContent = data.executive_requests ?? 0;
          document.getElementById("mini-investor-count").textContent = data.investor_requests ?? 0;
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
          const alerts = Array.isArray(data.alerts) ? data.alerts.slice(0, 6) : [];
          const wrap = document.getElementById("cc-alert-stream");
          const ticker = document.getElementById("cc-ticker");

          if (!alerts.length) {
            wrap.innerHTML = '<div class="stream-item"><div class="name">No active alerts right now</div><div class="meta">The system is running and waiting for new events.</div></div>';
            if (ticker) {
              ticker.innerHTML = `
                <div class="ticker-pill"><span class="dot stable"></span> Stable patient monitoring active</div>
                <div class="ticker-pill"><span class="dot high"></span> Command center awaiting next escalation</div>
                <div class="ticker-pill"><span class="dot stable"></span> AI surveillance layer online</div>
              `;
            }
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

            if (ticker) {
              ticker.innerHTML = alerts.map(function (a) {
                const sev = (a.severity || "").toLowerCase();
                const dot = sev === "critical" ? "critical" : (sev === "high" ? "high" : "stable");
                return `<div class="ticker-pill"><span class="dot ${dot}"></span>${a.title || a.alert_type || "Risk alert"} · ${a.patient_id || "Patient"} · Risk ${Number(a.risk_score ?? 0).toFixed(1)}</div>`;
              }).join("");
            }
          }

          const monitorA = alerts[0] || { severity: "critical", risk_score: 9.1 };
          const monitorB = alerts[1] || { severity: "high", risk_score: 8.2 };

          function setMonitor(prefix, alert) {
            const severity = (alert.severity || "stable").toLowerCase();
            const statusEl = document.getElementById(prefix + "-status");
            const hrEl = document.getElementById(prefix + "-hr");
            const spo2El = document.getElementById(prefix + "-spo2");
            const bpEl = document.getElementById(prefix + "-bp");
            const riskEl = document.getElementById(prefix + "-risk");

            const statusText = severity === "critical" ? "Critical" : (severity === "high" ? "High" : "Stable");
            statusEl.textContent = statusText;
            statusEl.className = "icu-state " + (severity === "critical" ? "critical" : (severity === "high" ? "warning" : "stable"));

            const baseRisk = Number(alert.risk_score ?? 3.4);
            const hr = severity === "critical" ? 124 + Math.floor(Math.random() * 10) : (severity === "high" ? 106 + Math.floor(Math.random() * 10) : 78 + Math.floor(Math.random() * 10));
            const spo2 = severity === "critical" ? 87 + Math.floor(Math.random() * 3) : (severity === "high" ? 92 + Math.floor(Math.random() * 3) : 97 + Math.floor(Math.random() * 2));
            const sys = severity === "critical" ? 160 + Math.floor(Math.random() * 8) : (severity === "high" ? 146 + Math.floor(Math.random() * 8) : 120 + Math.floor(Math.random() * 6));
            const dia = severity === "critical" ? 96 + Math.floor(Math.random() * 6) : (severity === "high" ? 88 + Math.floor(Math.random() * 5) : 76 + Math.floor(Math.random() * 4));

            hrEl.textContent = hr;
            spo2El.textContent = spo2;
            bpEl.textContent = sys + "/" + dia;
            riskEl.textContent = baseRisk.toFixed(1);
          }

          setMonitor("monitor-a", monitorA);
          setMonitor("monitor-b", monitorB);
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

INVESTOR_HTML = MAIN_HTML
HOSPITAL_HTML = MAIN_HTML
COMMAND_CENTER_HTML = MAIN_HTML


FORM_PAGE = r"""
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
    .wrap{max-width:840px;margin:0 auto}
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
      width:100%;
      border:1px solid rgba(255,255,255,.08);
      border-radius:16px;
      background:rgba(255,255,255,.03);
      color:var(--text);
      padding:14px 14px;
      font:inherit;
      outline:none;
    }
    textarea{min-height:130px;resize:vertical}
    .btn{
      border:0;
      border-radius:16px;
      padding:14px 18px;
      font:inherit;
      font-weight:1000;
      cursor:pointer;
      color:#08111f;
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      box-shadow:0 12px 28px rgba(91,212,255,.2);
    }
    .back{
      display:inline-flex;
      margin-top:16px;
      color:#cfe2ff;
      font-weight:900;
    }
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
      --shadow:0 18px 50px rgba(0,0,0,.24);
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:var(--text)}
    .wrap{max-width:1380px;margin:0 auto;padding:36px 18px 60px}
    .card{
      background:
        radial-gradient(circle at top right, rgba(91,212,255,.10), transparent 24%),
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(16,26,45,.96), rgba(11,18,31,.98));
      border:1px solid var(--line);
      border-radius:24px;
      padding:30px;
      box-shadow:var(--shadow)
    }
    h1{font-size:clamp(40px,4vw,56px);line-height:.98;margin:0 0 14px;letter-spacing:-.045em}
    h2{margin:0 0 10px;font-size:32px;letter-spacing:-.03em}
    p{color:var(--muted);line-height:1.7}
    .btn{display:inline-flex;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;font-weight:1000}
    .admin-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0 24px}
    .admin-kpi{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.06);
      border-radius:20px;
      padding:24px;
      min-height:154px;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
    }
    .admin-kpi .k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .admin-kpi .v{font-size:44px;font-weight:1000;line-height:1;margin-top:14px}
    .admin-kpi .hint{font-size:13px;color:#c6d7ef;line-height:1.5}
    .ticker-wrap{
      margin:18px 0;
      border:1px solid rgba(255,255,255,.08);
      border-radius:18px;
      overflow:hidden;
      background:
        linear-gradient(90deg, rgba(91,212,255,.08), rgba(122,162,255,.04), rgba(255,255,255,.02)),
        rgba(8,16,29,.92);
    }
    .ticker-head{
      display:flex;align-items:center;justify-content:space-between;gap:12px;
      padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.06)
    }
    .ticker-title{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9fdcff}
    .ticker-live{
      display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#dff7ff
    }
    .ticker-live::before{
      content:"";width:8px;height:8px;border-radius:999px;background:#38d39f;box-shadow:0 0 0 0 rgba(56,211,159,.55);animation:livePulse 1.8s infinite
    }
    @keyframes livePulse{
      0%{box-shadow:0 0 0 0 rgba(56,211,159,.55)}
      70%{box-shadow:0 0 0 12px rgba(56,211,159,0)}
      100%{box-shadow:0 0 0 0 rgba(56,211,159,0)}
    }
    .ticker-track{position:relative;overflow:hidden;white-space:nowrap;padding:14px 0}
    .ticker-move{display:inline-flex;gap:14px;padding-left:100%;animation:tickerMove 24s linear infinite}
    @keyframes tickerMove{
      from{transform:translateX(0)}
      to{transform:translateX(-100%)}
    }
    .ticker-pill{
      display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);color:#edf4ff;font-size:13px;font-weight:900
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
    .icu-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}
    .icu-kicker{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6}
    .icu-title{margin-top:6px;font-size:22px;font-weight:1000;line-height:1;color:#eef4ff}
    .icu-state{
      display:inline-flex;align-items:center;justify-content:center;min-width:92px;padding:8px 10px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.05em;text-transform:uppercase
    }
    .icu-state.critical{color:#ffd8d8;background:rgba(255,107,107,.16);border:1px solid rgba(255,107,107,.26)}
    .icu-state.warning{color:#ffe7bf;background:rgba(247,190,104,.16);border:1px solid rgba(247,190,104,.24)}
    .icu-state.stable{color:#dfffea;background:rgba(91,211,141,.14);border:1px solid rgba(91,211,141,.24)}
    .icu-screen{position:relative;height:240px;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.06);background:#050b14}
    .screen-grid{
      position:absolute;inset:0;background:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
      background-size:22px 22px,22px 22px,auto;pointer-events:none
    }
    .ecg-svg{position:absolute;inset:0;width:100%;height:100%}
    .ecg-path{
      fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:18 8;animation:ecgMove 1.8s linear infinite;filter:drop-shadow(0 0 10px currentColor)
    }
    .critical-path{stroke:#ff6b6b;color:#ff6b6b}
    .warning-path{stroke:#f7be68;color:#f7be68}
    @keyframes ecgMove{
      from{stroke-dashoffset:0}
      to{stroke-dashoffset:-52}
    }
    .screen-readouts{position:absolute;left:14px;right:14px;bottom:14px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
    .readout{
      border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:12px 10px;background:rgba(255,255,255,.03);backdrop-filter:blur(4px)
    }
    .r-k{display:block;font-size:11px;letter-spacing:.10em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .r-v{display:block;margin-top:8px;font-size:22px;line-height:1;font-weight:1000;color:#eef4ff}
    .icu-side-rail{display:grid;gap:12px}
    .rail-card{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88))
    }
    .rail-k{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9eb4d6;font-weight:900}
    .rail-v{font-size:34px;font-weight:1000;line-height:1;margin-top:10px;color:#ecf4ff}
    .rail-sub{margin-top:8px;font-size:13px;color:#c6d7ef;line-height:1.5}
    .proof-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}
    .proof-card{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:16px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
        rgba(8,16,29,.82);
      box-shadow:0 12px 28px rgba(0,0,0,.16)
    }
    .proof-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9fdcff}
    .proof-v{margin-top:10px;font-size:28px;line-height:1;font-weight:1000;color:#eef4ff}
    .proof-sub{margin-top:8px;color:#c6d7ef;font-size:13px;line-height:1.5}
    .status-legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
    .status-chip{
      display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;font-size:12px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:#eef4ff
    }
    .status-chip .s-dot{width:9px;height:9px;border-radius:999px}
    .status-chip.new .s-dot{background:#7aa2ff;box-shadow:0 0 10px rgba(122,162,255,.45)}
    .status-chip.contacted .s-dot{background:#f4bd6a;box-shadow:0 0 10px rgba(244,189,106,.35)}
    .status-chip.closed .s-dot{background:#38d39f;box-shadow:0 0 10px rgba(56,211,159,.35)}
    .lead-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px}
    .lead-type{
      flex:1 1 260px;
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      padding:16px;
    }
    .lead-type h3{margin:0 0 8px;font-size:24px}
    .lead-type p{margin:0;color:#c9daf0;line-height:1.65}
    .table-wrap{overflow:auto;border:1px solid rgba(255,255,255,.06);border-radius:18px;background:rgba(255,255,255,.02)}
    table{width:100%;border-collapse:collapse;font-size:14px;min-width:920px}
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
    .action-row{display:flex;gap:8px;flex-wrap:wrap}
    .mini-btn{
      display:inline-flex;align-items:center;justify-content:center;padding:9px 12px;border-radius:12px;background:#111b2f;border:1px solid rgba(255,255,255,.08);color:#eef4ff;font-size:12px;font-weight:1000
    }
    .score{
      display:inline-flex;align-items:center;justify-content:center;min-width:64px;padding:8px 10px;border-radius:999px;background:rgba(122,162,255,.14);border:1px solid rgba(122,162,255,.22);font-weight:1000
    }
    .score.hot{background:rgba(255,102,125,.16);border-color:rgba(255,102,125,.28);color:#ffd8df}
    .score.warm{background:rgba(244,189,106,.16);border-color:rgba(244,189,106,.28);color:#ffe5bb}
    .score.cool{background:rgba(56,211,159,.16);border-color:rgba(56,211,159,.26);color:#d9ffec}
    @media (max-width:1100px){
      .admin-grid{grid-template-columns:repeat(2,1fr)}
      .icu-wall{grid-template-columns:1fr}
      .proof-strip{grid-template-columns:repeat(2,minmax(0,1fr))}
    }
    @media (max-width:760px){
      .admin-grid{grid-template-columns:1fr}
      .proof-strip{grid-template-columns:1fr}
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
      <p>Review hospital demo requests, executive walkthrough requests, and investor intake submissions. Update lead status, export CSV, and manage follow-up pipeline.</p>
      <p><a class="btn" href="/admin/export.csv">Download CSV</a> <a class="btn" href="/">Return Home</a></p>

      <div class="ticker-wrap">
        <div class="ticker-head">
          <div class="ticker-title">Live Clinical Activity Ticker</div>
          <div class="ticker-live">Live Stream</div>
        </div>
        <div class="ticker-track">
          <div class="ticker-move" id="admin-ticker">
            <div class="ticker-pill"><span class="dot critical"></span> Critical risk escalation detected</div>
            <div class="ticker-pill"><span class="dot high"></span> Care team routing updated</div>
            <div class="ticker-pill"><span class="dot stable"></span> Stable patient trend confirmed</div>
            <div class="ticker-pill"><span class="dot critical"></span> ICU watchlist refreshed</div>
            <div class="ticker-pill"><span class="dot high"></span> Executive dashboard synced</div>
          </div>
        </div>
      </div>

      <div class="icu-wall">
        <div class="icu-main">
          <div class="icu-monitor critical-monitor">
            <div class="icu-head">
              <div>
                <div class="icu-kicker">ICU Bed A</div>
                <div class="icu-title">Critical Patient Monitor</div>
              </div>
              <div class="icu-state critical" id="admin-monitor-a-status">Critical</div>
            </div>

            <div class="icu-screen">
              <div class="screen-grid"></div>
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none" aria-hidden="true">
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
              <svg class="ecg-svg" viewBox="0 0 1200 220" preserveAspectRatio="none" aria-hidden="true">
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
        </div>

        <div class="icu-side-rail">
          <div class="rail-card"><div class="rail-k">Hospital Demo Requests</div><div class="rail-v">__HOSPITAL_COUNT__</div><div class="rail-sub">Clinical buyer and operator pipeline</div></div>
          <div class="rail-card"><div class="rail-k">Executive Walkthrough Requests</div><div class="rail-v">__EXEC_COUNT__</div><div class="rail-sub">Leadership and pilot evaluation requests</div></div>
          <div class="rail-card"><div class="rail-k">Investor Intake Requests</div><div class="rail-v">__INVESTOR_COUNT__</div><div class="rail-sub">Commercial and investor follow-up flow</div></div>
          <div class="rail-card"><div class="rail-k">Open Admin Leads</div><div class="rail-v">__OPEN_LEADS__</div><div class="rail-sub">Requests not yet marked closed</div></div>
        </div>
      </div>

      <div class="proof-strip">
        <div class="proof-card"><div class="proof-k">Hospital Story</div><div class="proof-v">Operational</div><div class="proof-sub">Clinical buyers, hospital operators, RPM teams, and health systems.</div></div>
        <div class="proof-card"><div class="proof-k">Executive Story</div><div class="proof-v">Strategic</div><div class="proof-sub">Leadership-facing walkthrough requests for evaluations, pilots, and strategic review.</div></div>
        <div class="proof-card"><div class="proof-k">Investor Story</div><div class="proof-v">Commercial</div><div class="proof-sub">Investor pipeline capture with contact details, timing, and opportunity notes.</div></div>
        <div class="proof-card"><div class="proof-k">Status Flow</div><div class="proof-v">Actionable</div><div class="proof-sub">New, Contacted, and Closed status flow supports real follow-up operations.</div></div>
      </div>

      <div class="status-legend">
        <div class="status-chip new"><span class="s-dot"></span>New</div>
        <div class="status-chip contacted"><span class="s-dot"></span>Contacted</div>
        <div class="status-chip closed"><span class="s-dot"></span>Closed</div>
      </div>

      <div class="lead-row">
        <div class="lead-type">
          <h3>Hospital Leads</h3>
          <p>Clinical buyers, hospital operators, RPM teams, and health systems.</p>
        </div>
        <div class="lead-type">
          <h3>Executive Leads</h3>
          <p>Leadership-facing walkthrough requests for evaluations, pilots, and strategic review.</p>
        </div>
        <div class="lead-type">
          <h3>Investor Leads</h3>
          <p>Investor pipeline capture with contact details, timing, and opportunity notes.</p>
        </div>
      </div>

      <div class="section"><h2>Hospital Demo Requests</h2>__HOSPITAL_TABLE__</div>
      <div class="section"><h2>Executive Walkthrough Requests</h2>__EXEC_TABLE__</div>
      <div class="section"><h2>Investor Intake</h2>__INVESTOR_TABLE__</div>
    </div>
  </div>

  <script>
    async function refreshAdminReview() {
      try {
        const res = await fetch("/admin/review?refresh=" + Date.now(), { cache: "no-store" });
        const html = await res.text();

        const parser = new DOMParser();
        const doc = parser.parseFromString(html, "text/html");

        const newHospital = doc.querySelector("#hospital-table");
        const newExec = doc.querySelector("#exec-table");
        const newInvestor = doc.querySelector("#investor-table");

        if (newHospital && document.querySelector("#hospital-table")) {
          document.querySelector("#hospital-table").innerHTML = newHospital.innerHTML;
        }
        if (newExec && document.querySelector("#exec-table")) {
          document.querySelector("#exec-table").innerHTML = newExec.innerHTML;
        }
        if (newInvestor && document.querySelector("#investor-table")) {
          document.querySelector("#investor-table").innerHTML = newInvestor.innerHTML;
        }
      } catch (err) {
        console.error("Admin auto-refresh failed", err);
      }

      try {
        const res = await fetch("/api/v1/dashboard/overview?tenant_id=demo&refresh=" + Date.now(), {
          headers: { "Accept": "application/json" },
          cache: "no-store"
        });
        if (res.ok) {
          const data = await res.json();
          const ticker = document.getElementById("admin-ticker");
          if (ticker) {
            ticker.innerHTML = `
              <div class="ticker-pill"><span class="dot critical"></span> Critical alerts ${data.critical_alerts ?? 0}</div>
              <div class="ticker-pill"><span class="dot high"></span> Open alerts ${data.open_alerts ?? 0}</div>
              <div class="ticker-pill"><span class="dot stable"></span> Avg risk ${Number(data.avg_risk_score ?? 0).toFixed(1)}</div>
              <div class="ticker-pill"><span class="dot high"></span> Hospital requests ${data.hospital_requests ?? 0}</div>
              <div class="ticker-pill"><span class="dot stable"></span> Investor requests ${data.investor_requests ?? 0}</div>
            `;
          }
        }
      } catch (err) {
        console.error("Admin KPI refresh failed", err);
      }

      try {
        const res = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), {
          headers: { "Accept": "application/json" },
          cache: "no-store"
        });
        if (res.ok) {
          const data = await res.json();
          const alerts = Array.isArray(data.alerts) ? data.alerts.slice(0, 2) : [];
          const a = alerts[0] || { severity: "critical", risk_score: 9.1 };
          const b = alerts[1] || { severity: "high", risk_score: 8.2 };

          function setMonitor(prefix, alert) {
            const severity = (alert.severity || "stable").toLowerCase();
            const statusEl = document.getElementById(prefix + "-status");
            const hrEl = document.getElementById(prefix + "-hr");
            const spo2El = document.getElementById(prefix + "-spo2");
            const bpEl = document.getElementById(prefix + "-bp");
            const riskEl = document.getElementById(prefix + "-risk");

            const statusText = severity === "critical" ? "Critical" : (severity === "high" ? "High" : "Stable");
            statusEl.textContent = statusText;
            statusEl.className = "icu-state " + (severity === "critical" ? "critical" : (severity === "high" ? "warning" : "stable"));

            const baseRisk = Number(alert.risk_score ?? 3.4);
            const hr = severity === "critical" ? 124 + Math.floor(Math.random() * 10) : (severity === "high" ? 106 + Math.floor(Math.random() * 10) : 78 + Math.floor(Math.random() * 10));
            const spo2 = severity === "critical" ? 87 + Math.floor(Math.random() * 3) : (severity === "high" ? 92 + Math.floor(Math.random() * 3) : 97 + Math.floor(Math.random() * 2));
            const sys = severity === "critical" ? 160 + Math.floor(Math.random() * 8) : (severity === "high" ? 146 + Math.floor(Math.random() * 8) : 120 + Math.floor(Math.random() * 6));
            const dia = severity === "critical" ? 96 + Math.floor(Math.random() * 6) : (severity === "high" ? 88 + Math.floor(Math.random() * 5) : 76 + Math.floor(Math.random() * 4));

            hrEl.textContent = hr;
            spo2El.textContent = spo2;
            bpEl.textContent = sys + "/" + dia;
            riskEl.textContent = baseRisk.toFixed(1);
          }

          setMonitor("admin-monitor-a", a);
          setMonitor("admin-monitor-b", b);
        }
      } catch (err) {
        console.error("Admin monitor refresh failed", err);
      }
    }

    setInterval(refreshAdminReview, 6000);
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


def _lead_score(payload: dict[str, Any], lead_type: str) -> int:
    score = 40

    timeline = (payload.get("timeline") or "").lower()
    message = (payload.get("message") or "").lower()
    facility_type = (payload.get("facility_type") or "").lower()
    priority = (payload.get("priority") or "").lower()
    investor_type = (payload.get("investor_type") or "").lower()
    check_size = (payload.get("check_size") or "").lower()

    if "immediate" in timeline:
        score += 28
    elif "30-60" in timeline:
        score += 18
    elif "quarter" in timeline:
        score += 10
    else:
        score += 5

    if lead_type == "hospital":
        if "hospital" in facility_type:
            score += 18
        elif "health system" in facility_type:
            score += 20
        elif "rpm" in facility_type:
            score += 15
        if any(x in message for x in ["pilot", "integration", "deployment", "command center", "live", "evaluation"]):
            score += 12

    if lead_type == "executive":
        if "operational" in priority:
            score += 18
        elif "pilot" in priority:
            score += 16
        elif "enterprise" in priority:
            score += 20
        elif "strategic" in priority:
            score += 18
        if any(x in message for x in ["budget", "review", "rollout", "system", "leadership", "hospital"]):
            score += 12

    if lead_type == "investor":
        if "vc" in investor_type:
            score += 18
        elif "angel" in investor_type:
            score += 12
        elif "strategic" in investor_type:
            score += 20
        elif "healthcare" in investor_type:
            score += 22
        elif "family office" in investor_type:
            score += 16
        if "$250k" in check_size or "250k" in check_size:
            score += 20
        elif "$50k" in check_size or "50k" in check_size:
            score += 10
        if any(x in message for x in ["deck", "traction", "pilot", "hospital", "round", "funding", "partnership"]):
            score += 12

    return max(1, min(100, score))


def _score_class(score: int) -> str:
    if score >= 80:
        return "hot"
    if score >= 60:
        return "warm"
    return "cool"


def send_notification_email(subject: str, message: str, recipients: list[str] | None = None) -> None:
    sender = INFO_EMAIL
    recipients = recipients or [INFO_EMAIL, FOUNDER_EMAIL]

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
    except Exception as e:
        print("Email send failed:", e, flush=True)


def _send_auto_reply(name: str, email: str, lead_type: str) -> None:
    if not email:
        return

    subject_map = {
        "hospital": "Your Hospital Demo Request - Early Risk Alert AI",
        "executive": "Your Executive Walkthrough Request - Early Risk Alert AI",
        "investor": "Your Investor Intake Submission - Early Risk Alert AI",
    }
    body_map = {
        "hospital": f"""Hello {name or "there"},

Thank you for requesting a hospital demo with Early Risk Alert AI.

Your request has been received successfully and is now in review.
Our team will follow up regarding next steps, scheduling, and platform walkthrough details.

Early Risk Alert AI
{INFO_EMAIL}
{BUSINESS_PHONE}
""",
        "executive": f"""Hello {name or "there"},

Thank you for requesting an executive walkthrough with Early Risk Alert AI.

Your request has been received successfully and is now in review.
We will follow up regarding scheduling, evaluation goals, and the most relevant platform views for your team.

Early Risk Alert AI
{INFO_EMAIL}
{BUSINESS_PHONE}
""",
        "investor": f"""Hello {name or "there"},

Thank you for your investor intake submission to Early Risk Alert AI.

Your request has been received successfully and is now in review.
We will follow up regarding materials, timing, and the next conversation.

Early Risk Alert AI
{INFO_EMAIL}
{BUSINESS_PHONE}
""",
    }
    send_notification_email(subject_map[lead_type], body_map[lead_type], recipients=[email])


def _detail_html(payload: dict[str, Any], fields: list[str]) -> str:
    rows = []
    for key in fields:
        value = html.escape(str(payload.get(key, "") or ""))
        label = html.escape(key.replace("_", " ").title())
        rows.append(f"<div style='margin:8px 0'><strong>{label}:</strong> {value}</div>")
    return "".join(rows)


def _render_thank_you(kind: str, message: str, detail_html: str) -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Thank You - Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{{margin:0;font-family:Inter,Arial,sans-serif;background:#07101c;color:#eef4ff;padding:28px}}
    .wrap{{max-width:760px;margin:0 auto}}
    .card{{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:24px;padding:28px}}
    h1{{margin:0 0 10px;font-size:42px;letter-spacing:-.04em}}
    p{{margin:0;color:#bdd0ea;line-height:1.7}}
    .box{{margin-top:18px;padding:18px;border-radius:18px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08)}}
    a{{display:inline-flex;margin-top:18px;color:#08111f;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);padding:12px 16px;border-radius:14px;font-weight:1000;text-decoration:none}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Thank You</h1>
      <p>{html.escape(message)}</p>
      <div class="box">{detail_html}</div>
      <a href="/">Return Home</a>
    </div>
  </div>
</body>
</html>
"""


def _update_row_status(path: Path, submitted_at: str, status: str) -> None:
    rows = _read_jsonl(path)
    new_rows: list[dict[str, Any]] = []
    normalized = _status_norm(status)

    for row in rows:
        if str(row.get("submitted_at", "")) == str(submitted_at):
            row["status"] = normalized
        new_rows.append(row)

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + ("\n" if new_rows else ""),
        encoding="utf-8",
    )


def _table_html(
    rows: list[dict[str, Any]],
    columns: list[str],
    labels: dict[str, str],
    route_prefix: str,
) -> str:
    if not rows:
        return "<div class='empty'>No submissions yet.</div>"

    head = "".join(f"<th>{html.escape(labels.get(col, col.title()))}</th>" for col in columns) + "<th>Actions</th>"
    body_rows: list[str] = []

    for row in rows:
        submitted_at = html.escape(str(row.get("submitted_at", "")))
        score = int(row.get("lead_score", 0) or 0)
        score_class = _score_class(score)
        tds: list[str] = []

        for col in columns:
            value = row.get(col, "")
            if col == "status":
                normalized = _status_norm(str(value))
                cls = "status-new"
                if normalized == "Contacted":
                    cls = "status-contacted"
                elif normalized == "Closed":
                    cls = "status-closed"
                cell = f"<span class='status-pill {cls}'>{html.escape(normalized)}</span>"
            elif col == "lead_score":
                cell = f"<span class='score {score_class}'>{score}</span>"
            else:
                cell = html.escape(str(value or ""))
            tds.append(f"<td>{cell}</td>")

        action_cell = (
            "<td>"
            "<div class='action-row'>"
            f"<a class='mini-btn' href='/admin/status/{route_prefix}?submitted_at={submitted_at}&status=New'>New</a>"
            f"<a class='mini-btn' href='/admin/status/{route_prefix}?submitted_at={submitted_at}&status=Contacted'>Contacted</a>"
            f"<a class='mini-btn' href='/admin/status/{route_prefix}?submitted_at={submitted_at}&status=Closed'>Closed</a>"
            "</div>"
            "</td>"
        )

        body_rows.append("<tr>" + "".join(tds) + action_cell + "</tr>")

    return f"<div class='table-wrap' id='{route_prefix}-table'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "era-dev-secret")

    data_dir = _data_dir()
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"
    investor_file = data_dir / "investor_intake_requests.jsonl"

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

    @app.route("/robots.txt")
    def robots_txt():
        return Response(
            "User-agent: *\nAllow: /\nSitemap: https://earlyriskalertai.com/sitemap.xml",
            mimetype="text/plain",
        )

    @app.get("/")
    def home():
        return render_template_string(MAIN_HTML)

    @app.get("/dashboard")
    def dashboard():
        return render_template_string(COMMAND_CENTER_HTML)

    @app.get("/investors")
    def investors():
        return render_template_string(INVESTOR_HTML)

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
            payload["lead_score"] = _lead_score(payload, "hospital")
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
Lead Score: {payload['lead_score']}
Message: {payload['message']}
"""
            send_notification_email(subject, message)
            _send_auto_reply(payload["full_name"], payload["email"], "hospital")

            return render_template_string(
                _render_thank_you(
                    "hospital",
                    "Your hospital demo request was submitted successfully. The request is now visible in admin review and ready for follow-up.",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "facility_type", "timeline", "lead_score"]),
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
        html_out = FORM_PAGE.replace("__TITLE__", "Hospital Demo - Early Risk Alert AI")
        html_out = html_out.replace("__HEADING__", "Request Hospital Demo")
        html_out = html_out.replace("__COPY__", "Submit interest from hospital operations teams, clinical leaders, and remote monitoring stakeholders.")
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
            payload["lead_score"] = _lead_score(payload, "executive")
            _save_jsonl(exec_file, payload)

            subject = "New Executive Walkthrough Request"
            message = f"""
New Executive Walkthrough Request

Name: {payload['full_name']}
Organization: {payload['organization']}
Title: {payload['title']}
Email: {payload['email']}
Phone: {payload['phone']}
Priority: {payload['priority']}
Timeline: {payload['timeline']}
Lead Score: {payload['lead_score']}
Message: {payload['message']}
"""
            send_notification_email(subject, message)
            _send_auto_reply(payload["full_name"], payload["email"], "executive")

            return render_template_string(
                _render_thank_you(
                    "executive",
                    "Your executive walkthrough request was submitted successfully. The request is now visible in admin review and ready for scheduling follow-up.",
                    _detail_html(payload, ["full_name", "organization", "title", "email", "priority", "timeline", "lead_score"]),
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
        html_out = FORM_PAGE.replace("__TITLE__", "Executive Walkthrough - Early Risk Alert AI")
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
            payload["lead_score"] = _lead_score(payload, "investor")
            _save_jsonl(investor_file, payload)

            subject = "New Investor Intake Request"
            message = f"""
New Investor Intake Request

Name: {payload['full_name']}
Organization: {payload['organization']}
Role: {payload['role']}
Email: {payload['email']}
Phone: {payload['phone']}
Investor Type: {payload['investor_type']}
Check Size: {payload['check_size']}
Timeline: {payload['timeline']}
Lead Score: {payload['lead_score']}
Message: {payload['message']}
"""
            send_notification_email(subject, message)
            _send_auto_reply(payload["full_name"], payload["email"], "investor")

            return render_template_string(
                _render_thank_you(
                    "investor",
                    "Your investor intake was submitted successfully. The request is now visible in admin review and ready for follow-up and export.",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "investor_type", "timeline", "check_size", "lead_score"]),
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
        html_out = FORM_PAGE.replace("__TITLE__", "Investor Intake - Early Risk Alert AI")
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
            "lead_score": "Score",
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
            "lead_score": "Score",
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
            "lead_score": "Score",
            "full_name": "Name",
            "organization": "Organization",
            "role": "Role",
            "email": "Email",
            "investor_type": "Investor Type",
            "check_size": "Check Size",
            "timeline": "Timeline",
        }

        open_leads = sum(
            1
            for row in hospital_rows + exec_rows + investor_rows
            if _status_norm(row.get("status")) != "Closed"
        )

        html_out = ADMIN_HTML
        html_out = html_out.replace("__HOSPITAL_COUNT__", str(len(hospital_rows)))
        html_out = html_out.replace("__EXEC_COUNT__", str(len(exec_rows)))
        html_out = html_out.replace("__INVESTOR_COUNT__", str(len(investor_rows)))
        html_out = html_out.replace("__OPEN_LEADS__", str(open_leads))
        html_out = html_out.replace(
            "__HOSPITAL_TABLE__",
            _table_html(hospital_rows, ["submitted_at", "status", "lead_score", "full_name", "organization", "role", "email", "facility_type", "timeline"], labels_h, "hospital"),
        )
        html_out = html_out.replace(
            "__EXEC_TABLE__",
            _table_html(exec_rows, ["submitted_at", "status", "lead_score", "full_name", "organization", "title", "email", "priority", "timeline"], labels_e, "executive"),
        )
        html_out = html_out.replace(
            "__INVESTOR_TABLE__",
            _table_html(investor_rows, ["submitted_at", "status", "lead_score", "full_name", "organization", "role", "email", "investor_type", "check_size", "timeline"], labels_i, "investor"),
        )
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
            "Lead Score",
            "Full Name",
            "Organization",
            "Role / Title",
            "Email Address",
            "Phone Number",
            "Lead Type or Priority",
            "Timeline",
            "Notes",
        ])

        for row in hospital_rows:
            writer.writerow([
                "Hospital Demo",
                row.get("submitted_at", ""),
                _status_norm(row.get("status", "New")),
                row.get("lead_score", ""),
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
                row.get("lead_score", ""),
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
                row.get("lead_score", ""),
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

    @app.get("/api/v1/dashboard/overview")
    def dashboard_overview():
        hospital_rows = _read_jsonl(hospital_file)
        exec_rows = _read_jsonl(exec_file)
        investor_rows = _read_jsonl(investor_file)

        alerts = _build_alerts(_build_demo_patients())

        avg_risk = 0.0
        if alerts:
            avg_risk = sum(float(a.get("risk_score", 0)) for a in alerts) / len(alerts)

        return jsonify({
            "tenant_id": request.args.get("tenant_id", "demo"),
            "patient_count": len(_build_demo_patients()),
            "open_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "critical"),
            "events_last_hour": len(alerts) * 3,
            "avg_risk_score": round(avg_risk, 1),
            "patients_with_alerts": len({a.get("patient_id") for a in alerts}),
            "hospital_requests": len(hospital_rows),
            "executive_requests": len(exec_rows),
            "investor_requests": len(investor_rows),
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


def _build_demo_patients() -> list[dict[str, Any]]:
    base = [
        {"patient_id": "p101", "name": "Patient A", "status": "Critical", "risk_score": round(8.7 + random.random() * 1.2, 1)},
        {"patient_id": "p102", "name": "Patient B", "status": "High", "risk_score": round(7.1 + random.random() * 1.1, 1)},
        {"patient_id": "p103", "name": "Patient C", "status": "Stable", "risk_score": round(2.8 + random.random() * 1.2, 1)},
        {"patient_id": "p104", "name": "Patient D", "status": "High", "risk_score": round(6.8 + random.random() * 1.0, 1)},
        {"patient_id": "p105", "name": "Patient E", "status": "Stable", "risk_score": round(3.1 + random.random() * 1.1, 1)},
    ]
    return base


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
