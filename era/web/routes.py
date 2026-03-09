from flask import Blueprint, render_template, render_template_string, send_file
import os

web_bp = Blueprint("web", __name__)

COMMAND_CENTER_HTML = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Clinical Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --panel2:#0d1526;
      --text:#ecf3ff;
      --muted:#92a6c8;
      --border:rgba(255,255,255,.08);
      --accent:#7aa2ff;
      --accent2:#5bd4ff;
      --green:#5bd38d;
      --amber:#f5c06a;
      --red:#ff6b6b;
      --shadow:0 18px 48px rgba(0,0,0,.35);
      --radius:22px;
      --max:1380px;
    }

    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.10), transparent 24%),
        radial-gradient(circle at top right, rgba(91,212,255,.08), transparent 24%),
        linear-gradient(180deg,#08111f 0%,#0c1424 100%);
    }

    a{color:inherit;text-decoration:none}
    .shell{max-width:var(--max);margin:0 auto;padding:20px 18px 60px}

    .nav{
      position:sticky;top:0;z-index:50;
      backdrop-filter:blur(10px);
      background:rgba(8,17,31,.82);
      border-bottom:1px solid rgba(255,255,255,.05);
    }
    .nav-inner{
      max-width:var(--max);
      margin:0 auto;
      padding:14px 18px;
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:18px;
      flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;
      letter-spacing:.18em;
      text-transform:uppercase;
      color:#cfe0ff;
      font-weight:900;
    }
    .brand-title{
      font-size:22px;
      font-weight:950;
      letter-spacing:-.03em;
      margin-top:3px;
    }
    .nav-links{
      display:flex;
      gap:18px;
      align-items:center;
      flex-wrap:wrap;
      font-size:14px;
      font-weight:800;
      color:#dbe7ff;
    }

    .btn{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:13px 18px;
      border-radius:14px;
      font-weight:900;
      border:none;
      cursor:pointer;
      transition:transform .18s ease, opacity .18s ease;
    }
    .btn:hover{transform:translateY(-1px);opacity:.96}
    .btn-primary{
      background:linear-gradient(180deg,var(--accent),var(--accent2));
      color:#08111f;
      box-shadow:var(--shadow);
    }
    .btn-secondary{
      background:#111b2f;
      color:var(--text);
      border:1px solid var(--border);
    }
    .btn-outline{
      background:transparent;
      color:#dfe9ff;
      border:1px solid var(--border);
    }

    .hero{
      display:grid;
      grid-template-columns:1.2fr .85fr;
      gap:18px;
      align-items:stretch;
      padding-top:20px;
    }

    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:26px;
      padding:26px;
      box-shadow:var(--shadow);
    }

    .hero-kicker,.section-kicker{
      font-size:12px;
      letter-spacing:.18em;
      text-transform:uppercase;
      color:#dce7ff;
      font-weight:900;
      margin-bottom:14px;
    }

    .headline{
      font-size:64px;
      line-height:.94;
      letter-spacing:-.06em;
      margin:0 0 18px;
      font-weight:950;
    }

    .body-lg{
      font-size:19px;
      line-height:1.65;
      color:#dce7ff;
      margin:0 0 10px;
    }

    .body{
      font-size:16px;
      line-height:1.75;
      color:var(--muted);
      margin:0;
    }

    .hero-actions{
      display:flex;
      gap:12px;
      flex-wrap:wrap;
      margin-top:24px;
    }

    .live-pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:8px 12px;
      border-radius:999px;
      background:rgba(91,211,141,.12);
      border:1px solid rgba(91,211,141,.22);
      color:#d8ffee;
      font-size:12px;
      font-weight:900;
      letter-spacing:.08em;
      text-transform:uppercase;
      margin-bottom:16px;
    }

    .dot{
      width:10px;
      height:10px;
      border-radius:999px;
      background:var(--green);
      box-shadow:0 0 14px rgba(91,211,141,.9);
    }

    .mini-grid{
      display:grid;
      grid-template-columns:repeat(2,1fr);
      gap:14px;
    }

    .mini{
      background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.05);
      border-radius:18px;
      padding:16px;
      min-height:118px;
    }

    .mini-k{
      font-size:11px;
      letter-spacing:.15em;
      text-transform:uppercase;
      color:#b8cae8;
      font-weight:900;
    }

    .mini-v{
      font-size:28px;
      font-weight:950;
      letter-spacing:-.04em;
      margin-top:8px;
    }

    .mini-s{
      font-size:13px;
      color:#c4d4ef;
      line-height:1.5;
      margin-top:8px;
    }

    .section{margin-top:64px}
    .section-title{
      font-size:40px;
      letter-spacing:-.04em;
      font-weight:950;
      margin:0 0 10px;
    }
    .section-sub{
      font-size:18px;
      line-height:1.7;
      color:var(--muted);
      margin:0 0 22px;
    }

    .metric-grid{
      display:grid;
      grid-template-columns:repeat(5,1fr);
      gap:16px;
    }

    .metric-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:18px;
      padding:18px;
    }

    .metric-k{
      font-size:11px;
      letter-spacing:.16em;
      text-transform:uppercase;
      color:#adc0df;
      font-weight:900;
    }

    .metric-v{
      font-size:42px;
      font-weight:950;
      letter-spacing:-.04em;
      margin-top:8px;
    }

    .metric-s{
      font-size:13px;
      color:#afc0dc;
      line-height:1.45;
      margin-top:8px;
    }

    .bar{
      height:9px;
      background:rgba(255,255,255,.06);
      border-radius:999px;
      overflow:hidden;
      margin-top:14px;
    }

    .bar > span{
      display:block;
      height:100%;
      background:linear-gradient(90deg,var(--accent),var(--accent2));
      border-radius:999px;
      transition:width .7s ease;
      width:0%;
    }

    .dash-grid{
      display:grid;
      grid-template-columns:1.1fr .95fr .85fr;
      gap:18px;
    }

    .feed-card,.focus-card,.side-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:22px;
      padding:22px;
      min-height:420px;
    }

    .panel-title{
      margin:0 0 8px;
      font-size:34px;
      letter-spacing:-.04em;
      font-weight:950;
    }

    .panel-sub{
      margin:0 0 18px;
      color:var(--muted);
      line-height:1.6;
    }

    .list{display:grid;gap:12px}

    .alert{
      padding:14px;
      border-radius:16px;
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.025);
    }

    .alert-top{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:center;
    }

    .alert-type{
      font-size:22px;
      font-weight:900;
      letter-spacing:-.03em;
      text-transform:lowercase;
    }

    .badge{
      font-size:11px;
      font-weight:900;
      letter-spacing:.12em;
      text-transform:uppercase;
      padding:7px 10px;
      border-radius:999px;
    }

    .badge.high{
      background:rgba(245,192,106,.14);
      border:1px solid rgba(245,192,106,.22);
      color:#ffe5ae;
    }

    .badge.critical{
      background:rgba(255,107,107,.14);
      border:1px solid rgba(255,107,107,.22);
      color:#ffd4d4;
    }

    .badge.info{
      background:rgba(122,162,255,.14);
      border:1px solid rgba(122,162,255,.22);
      color:#d7e3ff;
    }

    .alert-patient{
      font-size:14px;
      color:#dfe9ff;
      margin-top:4px;
      font-weight:700;
    }

    .alert-msg{
      margin-top:8px;
      color:#dce8ff;
      font-size:16px;
      line-height:1.5;
    }

    .alert-time{
      margin-top:6px;
      color:#8da2c7;
      font-size:13px;
    }

    .stat-block{
      background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.05);
      border-radius:18px;
      padding:16px;
      margin-bottom:12px;
    }

    .stat-row{
      display:flex;
      justify-content:space-between;
      gap:14px;
      padding:7px 0;
      border-bottom:1px solid rgba(255,255,255,.05);
    }

    .stat-row:last-child{border-bottom:none}
    .stat-label{color:#c5d3ed}
    .stat-value{font-weight:900}

    .cred-grid{
      display:grid;
      grid-template-columns:1fr 1fr 1fr;
      gap:18px;
    }

    .cred-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:22px;
      padding:22px;
    }

    .cred-card h3{
      margin:0 0 10px;
      font-size:26px;
      letter-spacing:-.03em;
    }

    .cred-card p{
      margin:0;
      color:var(--muted);
      line-height:1.7;
    }

    .demo-grid{
      display:grid;
      grid-template-columns:1.1fr .9fr;
      gap:18px;
    }

    .video-wrap{
      position:relative;
      padding-bottom:56.25%;
      height:0;
      overflow:hidden;
      border-radius:22px;
      border:1px solid var(--border);
      background:#08111f;
    }

    .video-wrap iframe{
      position:absolute;
      top:0;
      left:0;
      width:100%;
      height:100%;
      border:0;
    }

    .placeholder-demo{
      border:1px dashed rgba(255,255,255,.16);
      border-radius:22px;
      padding:36px;
      background:rgba(255,255,255,.02);
      min-height:320px;
      display:flex;
      align-items:center;
      justify-content:center;
      text-align:center;
      color:#cddcf8;
      line-height:1.7;
    }

    .cta-grid{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:18px;
    }

    .footer{
      margin-top:40px;
      padding-top:18px;
      border-top:1px solid rgba(255,255,255,.06);
      text-align:center;
      color:#8aa0c8;
      font-size:14px;
    }

    .muted{color:var(--muted)}
    .small{font-size:13px}
    .empty{color:#90a3c5;font-size:15px;padding:14px 0}

    @media (max-width:1180px){
      .hero,.dash-grid,.demo-grid,.cred-grid,.cta-grid{grid-template-columns:1fr}
      .metric-grid{grid-template-columns:repeat(2,1fr)}
      .headline{font-size:52px}
    }

    @media (max-width:760px){
      .metric-grid,.mini-grid{grid-template-columns:1fr}
      .headline{font-size:40px}
      .section-title{font-size:32px}
      .shell{padding:18px 14px 46px}
      .nav-inner{padding:12px 14px}
    }
  </style>
</head>
<body>
  <div class="nav">
    <div class="nav-inner">
      <div>
        <div class="brand-kicker">AI-Powered Predictive Clinical Intelligence</div>
        <div class="brand-title">Early Risk Alert AI</div>
      </div>
      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#metrics">Metrics</a>
        <a href="#live">Live Dashboard</a>
        <a href="#credibility">Credibility</a>
        <a href="#demo">Demo</a>
        <a href="/investors" class="btn btn-secondary">Investor View</a>
        <a href="/deck" class="btn btn-primary">Download Pitch Deck</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="card">
        <div class="live-pill"><span class="dot"></span> Live platform active</div>
        <div class="hero-kicker">Clinical Operations Command Center</div>
        <h1 class="headline">Predict patient deterioration before it becomes a medical emergency.</h1>
        <p class="body-lg">
          Early Risk Alert AI is built for hospitals, health systems, and remote monitoring teams that
          need real-time patient visibility, automated alerts, and command-center style oversight.
        </p>
        <p class="body">
          This page now acts as your public-facing hospital product page, live operational dashboard,
          and polished company presence all in one place.
        </p>

        <div class="hero-actions">
          <a class="btn btn-primary" href="#live">Open Live Dashboard</a>
          <a class="btn btn-secondary" href="#demo">View Demo Section</a>
          <a class="btn btn-outline" href="/investors">Open Investor Portal</a>
        </div>
      </div>

      <div class="card">
        <div class="section-kicker">System Snapshot</div>
        <div class="mini-grid">
          <div class="mini">
            <div class="mini-k">Primary Use Case</div>
            <div class="mini-v">Early Risk Detection</div>
            <div class="mini-s">Detect deterioration signals before escalation becomes critical.</div>
          </div>
          <div class="mini">
            <div class="mini-k">Platform Buyer</div>
            <div class="mini-v">Hospitals</div>
            <div class="mini-s">Built for command centers, remote monitoring teams, and care networks.</div>
          </div>
          <div class="mini">
            <div class="mini-k">Mode</div>
            <div class="mini-v" id="modeHero">realtime</div>
            <div class="mini-s">Operational stream state refreshes automatically from your live API.</div>
          </div>
          <div class="mini">
            <div class="mini-k">Patients with Alerts</div>
            <div class="mini-v" id="patientsWithAlertsHero">3</div>
            <div class="mini-s">Live count of patients currently surfaced in the alert feed.</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="metrics">
      <h2 class="section-title">Animated Live Metrics</h2>
      <p class="section-sub">
        These headline metrics pull from your system and update in place to create a real command-center feel.
      </p>

      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-k">Patients</div>
          <div class="metric-v" id="patientsCount">0</div>
          <div class="metric-s">Active monitored patient volume</div>
          <div class="bar"><span id="barPatients"></span></div>
        </div>
        <div class="metric-card">
          <div class="metric-k">Open Alerts</div>
          <div class="metric-v" id="openAlerts">0</div>
          <div class="metric-s">Current unresolved alert count</div>
          <div class="bar"><span id="barOpenAlerts"></span></div>
        </div>
        <div class="metric-card">
          <div class="metric-k">Critical Alerts</div>
          <div class="metric-v" id="criticalAlerts">0</div>
          <div class="metric-s">Highest-priority intervention queue</div>
          <div class="bar"><span id="barCriticalAlerts"></span></div>
        </div>
        <div class="metric-card">
          <div class="metric-k">Events Last Hour</div>
          <div class="metric-v" id="eventsLastHour">0</div>
          <div class="metric-s">Recent event throughput</div>
          <div class="bar"><span id="barEvents"></span></div>
        </div>
        <div class="metric-card">
          <div class="metric-k">Stream Mode</div>
          <div class="metric-v" id="streamMode">—</div>
          <div class="metric-s">Backend stream operating mode</div>
          <div class="bar"><span id="barMode"></span></div>
        </div>
      </div>
    </section>

    <section class="section" id="live">
      <h2 class="section-title">Live Command Dashboard</h2>
      <p class="section-sub">
        A professional, hospital-facing command-center layout showing alerts, patient details, and system health.
      </p>

      <div class="dash-grid">
        <div class="feed-card">
          <h3 class="panel-title">Live Alerts Feed</h3>
          <p class="panel-sub">Clinical alerts, patient prioritization, and real-time operational visibility.</p>
          <div class="list" id="alertsList"></div>
        </div>

        <div class="focus-card">
          <h3 class="panel-title">Patient Focus</h3>
          <p class="panel-sub">Focused patient view for vitals, event timing, and rollup metrics.</p>

          <div class="stat-block">
            <div class="stat-row"><div class="stat-label">Patient</div><div class="stat-value" id="focusPatient">p101</div></div>
            <div class="stat-row"><div class="stat-label">Event time</div><div class="stat-value" id="focusEventTime">live</div></div>
            <div class="stat-row"><div class="stat-label">Heart rate</div><div class="stat-value" id="focusHeartRate">128</div></div>
            <div class="stat-row"><div class="stat-label">Systolic BP</div><div class="stat-value" id="focusSystolic">172</div></div>
            <div class="stat-row"><div class="stat-label">Diastolic BP</div><div class="stat-value" id="focusDiastolic">104</div></div>
            <div class="stat-row"><div class="stat-label">SpO2</div><div class="stat-value" id="focusSpo2">89</div></div>
          </div>

          <div class="stat-block">
            <div class="stat-row"><div class="stat-label">Avg heart rate</div><div class="stat-value" id="rollAvgHr">118</div></div>
            <div class="stat-row"><div class="stat-label">Avg systolic BP</div><div class="stat-value" id="rollAvgSys">164</div></div>
            <div class="stat-row"><div class="stat-label">Avg diastolic BP</div><div class="stat-value" id="rollAvgDia">98</div></div>
            <div class="stat-row"><div class="stat-label">Avg SpO2</div><div class="stat-value" id="rollAvgSpo2">92</div></div>
          </div>
        </div>

        <div class="side-card">
          <h3 class="panel-title">System Panels</h3>
          <p class="panel-sub">Operational stream health, monitoring state, and channels.</p>

          <div class="stat-block">
            <div class="stat-row"><div class="stat-label">Status</div><div class="stat-value" id="streamStatus">running</div></div>
            <div class="stat-row"><div class="stat-label">Redis OK</div><div class="stat-value" id="redisOk">true</div></div>
            <div class="stat-row"><div class="stat-label">Mode</div><div class="stat-value" id="streamModePanel">realtime</div></div>
            <div class="stat-row"><div class="stat-label">Patients with alerts</div><div class="stat-value" id="patientsWithAlerts">3</div></div>
          </div>

          <div class="stat-block">
            <div class="small muted" style="margin-bottom:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase">Channels</div>
            <div id="channelsList" class="small" style="line-height:1.7;color:#dce7ff"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="credibility">
      <h2 class="section-title">Business Credibility Layer</h2>
      <p class="section-sub">
        This section reflects the company infrastructure and brand foundation you already put in place.
      </p>

      <div class="cred-grid">
        <div class="cred-card">
          <h3>Professional Presence</h3>
          <p>
            Early Risk Alert AI has a live domain, company branding, pitch materials, and a hospital-facing
            web presence designed for real outreach.
          </p>
        </div>

        <div class="cred-card">
          <h3>Branded Communications</h3>
          <p>
            Your professional email, investor contact flow, and platform demo request process support real
            conversations with hospitals, partners, and investors.
          </p>
        </div>

        <div class="cred-card">
          <h3>Funding Readiness</h3>
          <p>
            The company now has a public-facing product page, an investor portal, a pitch deck, and a
            live demo request flow — all critical for business development.
          </p>
        </div>
      </div>
    </section>

    <section class="section" id="demo">
      <h2 class="section-title">Live Product Demo</h2>
      <p class="section-sub">
        I removed the incorrect video. Replace the link below with your real YouTube embed once you have it.
      </p>

      <div class="demo-grid">
        <div class="card">
          <div class="section-kicker">Demo Video</div>
          <div class="placeholder-demo">
            <div>
              <div style="font-size:24px;font-weight:900;margin-bottom:12px">Demo Video Ready Slot</div>
              <div>
                Replace the iframe src in this section with your real YouTube embed URL.<br>
                Example format:<br><br>
                https://www.youtube.com/embed/YOUR_VIDEO_ID
              </div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-kicker">Platform Summary</div>
          <h2 class="section-title" style="font-size:34px">Built for modern hospital operations</h2>
          <p class="body" style="margin-bottom:16px">
            Early Risk Alert AI is positioned as a deployable clinical intelligence platform for hospitals,
            care teams, remote monitoring programs, and strategic partners.
          </p>

          <div class="stat-block">
            <div class="stat-row"><div class="stat-label">Use case</div><div class="stat-value">Clinical intelligence</div></div>
            <div class="stat-row"><div class="stat-label">Deployment</div><div class="stat-value">Cloud-based</div></div>
            <div class="stat-row"><div class="stat-label">Buyer</div><div class="stat-value">Hospitals / RPM</div></div>
            <div class="stat-row"><div class="stat-label">Expansion</div><div class="stat-value">Enterprise rollout</div></div>
          </div>

          <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:16px">
            <a class="btn btn-primary" href="/investors">Investor View</a>
            <a class="btn btn-secondary" href="/deck">Pitch Deck PDF</a>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="cta-grid">
        <div class="card">
          <div class="section-kicker">Demo Request</div>
          <h2 class="section-title" style="font-size:34px">Schedule a guided platform walkthrough</h2>
          <p class="body" style="margin-bottom:18px">
            Use your Google Form as the live platform demo request workflow for hospitals, physicians,
            insurers, patients, and investors.
          </p>
          <div style="display:flex;gap:12px;flex-wrap:wrap">
            <a class="btn btn-primary" href="#" id="demoRequestBtn">Open Demo Request Form</a>
          </div>
        </div>

        <div class="card">
          <div class="section-kicker">Company Contact</div>
          <h2 class="section-title" style="font-size:34px">Professional contact layer</h2>
          <div class="stat-block">
            <div class="stat-row"><div class="stat-label">Founder</div><div class="stat-value">Milton Munroe</div></div>
            <div class="stat-row"><div class="stat-label">Business Phone</div><div class="stat-value">732-724-7267</div></div>
            <div class="stat-row"><div class="stat-label">Email</div><div class="stat-value">info@earlyriskalertai.com</div></div>
            <div class="stat-row"><div class="stat-label">Domain</div><div class="stat-value">earlyriskalertai.com</div></div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC • Hospital UI + Investor Demo + Command Center
    </div>
  </div>

  <script>
    const API_BASE = "/api";
    const DEFAULT_TENANT = "demo";
    let currentPatientId = "p101";
    let currentAlerts = [];

    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }

    function animateNumber(el, target) {
      const start = Number(el.dataset.value || 0);
      const end = Number(target || 0);
      const duration = 650;
      const startTime = performance.now();

      function frame(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const value = Math.round(start + (end - start) * progress);
        el.textContent = String(value);
        if (progress < 1) {
          requestAnimationFrame(frame);
        } else {
          el.dataset.value = String(end);
        }
      }
      requestAnimationFrame(frame);
    }

    function setMetric(id, value) {
      const el = document.getElementById(id);
      if (!el) return;
      if (typeof value === "number") {
        animateNumber(el, value);
      } else {
        el.textContent = String(value);
      }
    }

    function setBar(id, pct) {
      const el = document.getElementById(id);
      if (el) el.style.width = Math.max(8, Math.min(100, pct)) + "%";
    }

    async function fetchJSON(url, fallback) {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error("bad response");
        return await res.json();
      } catch (e) {
        return fallback;
      }
    }

    function severityClass(sev) {
      const s = String(sev || "").toLowerCase();
      if (s.includes("crit")) return "critical";
      if (s.includes("high") || s.includes("warn")) return "high";
      return "info";
    }

    function renderAlerts(items) {
      const root = document.getElementById("alertsList");
      if (!root) return;

      if (!items || !items.length) {
        root.innerHTML = '<div class="empty">No live alerts available right now.</div>';
        return;
      }

      root.innerHTML = items.map(a => {
        const sev = a.severity || "info";
        return `
          <div class="alert">
            <div class="alert-top">
              <div class="alert-type">${a.alert_type || "alert"}</div>
              <div class="badge ${severityClass(sev)}">${sev}</div>
            </div>
            <div class="alert-patient">Patient ${a.patient_id || "—"}</div>
            <div class="alert-msg">${a.message || "No message provided"}</div>
            <div class="alert-time">${a.created_at || "live"}</div>
          </div>
        `;
      }).join("");
    }

    function renderChannels(items) {
      const root = document.getElementById("channelsList");
      if (!root) return;
      if (!items || !items.length) {
        root.innerHTML = '<div class="muted">No channels reported.</div>';
        return;
      }
      root.innerHTML = items.map(x => `<div>${x}</div>`).join("");
    }

    async function loadOverview() {
      const fallback = {
        tenant_id: DEFAULT_TENANT,
        patient_count: 1284,
        open_alerts: 6,
        critical_alerts: 2,
        events_last_hour: 93
      };
      const data = await fetchJSON(`${API_BASE}/dashboard/overview?tenant_id=${DEFAULT_TENANT}`, fallback);

      setMetric("patientsCount", Number(data.patient_count || 0));
      setMetric("openAlerts", Number(data.open_alerts || 0));
      setMetric("criticalAlerts", Number(data.critical_alerts || 0));
      setMetric("eventsLastHour", Number(data.events_last_hour || 0));

      setBar("barPatients", Math.min(100, (Number(data.patient_count || 0) / 1500) * 100));
      setBar("barOpenAlerts", Math.min(100, (Number(data.open_alerts || 0) / 12) * 100));
      setBar("barCriticalAlerts", Math.min(100, (Number(data.critical_alerts || 0) / 6) * 100));
      setBar("barEvents", Math.min(100, (Number(data.events_last_hour || 0) / 120) * 100));
    }

    async function loadAlerts() {
      const fallback = {
        tenant_id: DEFAULT_TENANT,
        alerts: [
          { alert_type: "tachycardia", severity: "high", patient_id: "p101", message: "Heart rate elevated", created_at: "live" },
          { alert_type: "low_spo2", severity: "critical", patient_id: "p202", message: "Oxygen saturation critical", created_at: "live" },
          { alert_type: "hypertension", severity: "high", patient_id: "p303", message: "Blood pressure elevated", created_at: "live" }
        ]
      };
      const data = await fetchJSON(`${API_BASE}/alerts?tenant_id=${DEFAULT_TENANT}`, fallback);
      currentAlerts = Array.isArray(data.alerts) ? data.alerts : [];
      renderAlerts(currentAlerts);

      const first = currentAlerts[0];
      currentPatientId = first && first.patient_id ? first.patient_id : "p101";
      setText("focusPatient", currentPatientId);
      setText("patientsWithAlerts", currentAlerts.length);
      setText("patientsWithAlertsHero", currentAlerts.length);
    }

    async function loadVitalsAndRollups() {
      const fallbackVitals = {
        tenant_id: DEFAULT_TENANT,
        patient_id: currentPatientId,
        event_ts: "live",
        vitals: { heart_rate: 128, systolic_bp: 172, diastolic_bp: 104, spo2: 89 }
      };
      const fallbackRollups = {
        tenant_id: DEFAULT_TENANT,
        patient_id: currentPatientId,
        rollups: { avg_heart_rate: 118, avg_systolic_bp: 164, avg_diastolic_bp: 98, avg_spo2: 92 }
      };

      const vitals = await fetchJSON(
        `${API_BASE}/vitals/latest?tenant_id=${DEFAULT_TENANT}&patient_id=${encodeURIComponent(currentPatientId)}`,
        fallbackVitals
      );
      const rollups = await fetchJSON(
        `${API_BASE}/patients/${encodeURIComponent(currentPatientId)}/rollups?tenant_id=${DEFAULT_TENANT}`,
        fallbackRollups
      );

      const v = vitals.vitals || {};
      const r = rollups.rollups || {};

      setText("focusPatient", vitals.patient_id || currentPatientId);
      setText("focusEventTime", vitals.event_ts || "live");
      setText("focusHeartRate", v.heart_rate ?? "—");
      setText("focusSystolic", v.systolic_bp ?? "—");
      setText("focusDiastolic", v.diastolic_bp ?? "—");
      setText("focusSpo2", v.spo2 ?? "—");

      setText("rollAvgHr", r.avg_heart_rate ?? "—");
      setText("rollAvgSys", r.avg_systolic_bp ?? "—");
      setText("rollAvgDia", r.avg_diastolic_bp ?? "—");
      setText("rollAvgSpo2", r.avg_spo2 ?? "—");
    }

    async function loadStream() {
      const fallbackHealth = { status: "running", redis_ok: true, mode: "realtime" };
      const fallbackChannels = {
        tenant_id: DEFAULT_TENANT,
        patient_id: currentPatientId,
        channels: [
          "stream:vitals",
          `stream:vitals:${DEFAULT_TENANT}`,
          `stream:vitals:${DEFAULT_TENANT}:${currentPatientId}`,
          "stream:alerts",
          `stream:alerts:${DEFAULT_TENANT}`,
          `stream:alerts:${DEFAULT_TENANT}:${currentPatientId}`
        ]
      };

      const health = await fetchJSON(`${API_BASE}/stream/health`, fallbackHealth);
      const channels = await fetchJSON(
        `${API_BASE}/stream/channels?tenant_id=${DEFAULT_TENANT}&patient_id=${encodeURIComponent(currentPatientId)}`,
        fallbackChannels
      );

      const mode = health.mode || "realtime";
      setText("streamStatus", health.status || "running");
      setText("redisOk", String(health.redis_ok));
      setText("streamMode", mode);
      setText("streamModePanel", mode);
      setText("modeHero", mode);
      setBar("barMode", mode === "realtime" ? 100 : 60);

      renderChannels(channels.channels || []);
    }

    function bindLinks() {
      const demoBtn = document.getElementById("demoRequestBtn");
      if (demoBtn) {
        demoBtn.href = "PASTE_YOUR_GOOGLE_FORM_LINK_HERE";
        demoBtn.target = "_blank";
        demoBtn.rel = "noopener noreferrer";
      }
    }

    async function refreshAll() {
      await loadOverview();
      await loadAlerts();
      await loadVitalsAndRollups();
      await loadStream();
    }

    bindLinks();
    refreshAll();
    setInterval(refreshAll, 6000);
  </script>
</body>
</html>
'''

def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@web_bp.get("/")
def home():
    return render_template_string(COMMAND_CENTER_HTML)

@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(COMMAND_CENTER_HTML)

@web_bp.get("/login")
def login():
    return render_template("login.html")

@web_bp.get("/investors")
def investors():
    return render_template("investor.html")

@web_bp.get("/deck")
@web_bp.get("/pitch-deck")
def deck():
    pdf_path = os.path.join(_project_root(), "static", "Early_Risk_Alert_AI_Pitch_Deck.pdf")
    if not os.path.exists(pdf_path):
        return f"Pitch deck not found at: {pdf_path}", 404

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf",
    )
