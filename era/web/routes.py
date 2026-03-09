from flask import Blueprint, render_template_string, send_file
import os

web_bp = Blueprint("web", __name__)

YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"
BUSINESS_EMAIL = "info@earlyriskai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"

MAIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Clinical Command Platform</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --bg2:#0d1526;
      --panel:#101a2d;
      --panel2:#0f1728;
      --line:rgba(255,255,255,.08);
      --text:#edf4ff;
      --muted:#94a7c6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --amber:#f5c06a;
      --red:#ff7a7a;
      --shadow:0 18px 50px rgba(0,0,0,.35);
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
        radial-gradient(circle at top left, rgba(122,162,255,.12), transparent 22%),
        radial-gradient(circle at top right, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }

    a{color:inherit;text-decoration:none}
    .shell{max-width:var(--max);margin:0 auto;padding:20px 16px 70px}

    .nav{
      position:sticky;top:0;z-index:50;
      backdrop-filter:blur(14px);
      background:rgba(8,17,31,.74);
      border-bottom:1px solid var(--line);
    }
    .nav-inner{
      max-width:var(--max);margin:0 auto;padding:14px 16px;
      display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;
    }
    .brand{display:flex;flex-direction:column;gap:4px}
    .brand-kicker{
      font-size:11px;letter-spacing:.18em;text-transform:uppercase;
      color:#b9c9e7;font-weight:800;
    }
    .brand-title{font-size:38px;font-weight:1000;line-height:.95}
    .brand-sub{font-size:14px;color:var(--muted);font-weight:700}

    .nav-links{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
    .nav-links a{font-weight:800;font-size:14px;color:#d7e4ff}

    .btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:13px 18px;border-radius:16px;font-weight:900;font-size:14px;
      border:1px solid transparent;transition:.18s transform ease,.18s opacity ease,.18s border-color ease;
      cursor:pointer;
      background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;
      box-shadow:0 10px 28px rgba(122,162,255,.22);
    }
    .btn:hover{transform:translateY(-1px);opacity:.96}
    .btn.secondary{
      background:#111b2f;color:var(--text);border-color:var(--line);box-shadow:none;
    }
    .btn.ghost{
      background:transparent;color:var(--text);border-color:var(--line);box-shadow:none;
    }

    .hero{
      display:grid;grid-template-columns:1.35fr .95fr;gap:18px;align-items:stretch;
      padding-top:26px;
    }
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--line);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
    }
    .hero-main{padding:28px}
    .hero-kicker{
      font-size:12px;letter-spacing:.18em;text-transform:uppercase;
      color:#c8d7f1;font-weight:900;margin-bottom:12px;
    }
    .hero h1{
      margin:0 0 14px;font-size:68px;line-height:.92;font-weight:1000;letter-spacing:-.05em;
    }
    .hero p{
      margin:0;color:#c8d7f1;font-size:20px;line-height:1.55;max-width:900px;
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}
    .hero-side{padding:24px}

    .live-pill{
      display:inline-flex;align-items:center;gap:8px;
      padding:8px 12px;border-radius:999px;background:rgba(56,211,159,.12);
      border:1px solid rgba(56,211,159,.25);font-weight:900;font-size:12px;color:#bff5df;
      text-transform:uppercase;letter-spacing:.08em;
    }
    .dot{
      width:10px;height:10px;border-radius:999px;background:var(--green);
      box-shadow:0 0 18px rgba(56,211,159,.6);
      animation:pulseDot 1.8s infinite;
    }
    @keyframes pulseDot{
      0%{transform:scale(1);opacity:1}
      50%{transform:scale(1.35);opacity:.65}
      100%{transform:scale(1);opacity:1}
    }

    .side-title{font-size:34px;font-weight:950;line-height:1;margin:18px 0 8px}
    .side-copy{color:#b8cae8;line-height:1.65;margin:0 0 18px}
    .mini-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
    .mini{
      border:1px solid rgba(255,255,255,.06);
      background:rgba(255,255,255,.025);
      border-radius:18px;padding:16px;min-height:110px;
    }
    .mini-k{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#9fb7da;font-weight:900}
    .mini-v{font-size:34px;font-weight:950;line-height:1;margin-top:10px}
    .mini-s{font-size:14px;color:#b7c9e7;margin-top:8px}

    .ticker-wrap{
      margin-top:18px;
      border:1px solid rgba(91,212,255,.16);
      background:linear-gradient(180deg, rgba(91,212,255,.06), rgba(91,212,255,.02));
      border-radius:18px;padding:12px 14px;overflow:hidden;
    }
    .ticker-track{
      display:flex;gap:26px;white-space:nowrap;min-width:max-content;
      animation:tickerMove 24s linear infinite;
      color:#d8e7ff;font-weight:800;
    }
    @keyframes tickerMove{
      0%{transform:translateX(0)}
      100%{transform:translateX(-50%)}
    }
    .ticker-item{display:inline-flex;align-items:center;gap:8px;font-size:14px}
    .pulse-bar{
      width:18px;height:10px;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--blue2));
      animation:pulseBar 1.2s ease-in-out infinite alternate;
    }
    @keyframes pulseBar{
      from{transform:scaleX(.65);opacity:.65}
      to{transform:scaleX(1.1);opacity:1}
    }

    .section{margin-top:22px}
    .section-title{
      font-size:40px;font-weight:1000;letter-spacing:-.04em;margin:0 0 8px;
    }
    .section-sub{
      color:var(--muted);font-size:16px;line-height:1.65;margin:0 0 18px;
    }

    .metrics-grid{
      display:grid;grid-template-columns:repeat(4,1fr);gap:14px;
    }
    .metric{padding:22px;position:relative;overflow:hidden}
    .metric::after{
      content:"";position:absolute;inset:auto -20% -40% auto;width:180px;height:180px;
      background:radial-gradient(circle, rgba(122,162,255,.12), transparent 60%);
      pointer-events:none;
    }
    .metric-label{
      color:#a7bad9;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.14em;
    }
    .metric-value{
      font-size:54px;font-weight:1000;line-height:1;margin-top:10px;
    }
    .metric-note{
      color:#bfd0ea;font-size:14px;margin-top:8px;
    }
    .progress{
      width:100%;height:8px;border-radius:999px;background:rgba(255,255,255,.06);margin-top:16px;overflow:hidden;
    }
    .progress > span{
      display:block;height:100%;width:0%;
      background:linear-gradient(90deg,var(--blue),var(--blue2));transition:width .8s ease;
    }

    .dashboard-grid{
      display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:14px;
    }
    .dash-card{padding:22px;min-height:390px}
    .dash-card h3{margin:0 0 6px;font-size:22px}
    .dash-card p{margin:0 0 16px;color:var(--muted);line-height:1.6}
    .feed{display:flex;flex-direction:column;gap:12px}
    .alert{
      background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);
      border-radius:18px;padding:16px;position:relative;overflow:hidden;
    }
    .alert::before{
      content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,var(--blue),var(--blue2));
      opacity:.75;
    }
    .alert-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
    .alert-name{font-size:18px;font-weight:900}
    .alert-patient{font-size:13px;color:#b6c8e8;font-weight:700}
    .alert-msg{font-size:18px;font-weight:800;margin-top:6px}
    .alert-time{font-size:13px;color:#9ab0d3;margin-top:6px}
    .badge{
      display:inline-flex;align-items:center;justify-content:center;
      min-width:78px;padding:7px 11px;border-radius:999px;font-size:12px;
      text-transform:uppercase;letter-spacing:.1em;font-weight:900;border:1px solid transparent;
    }
    .critical{background:rgba(255,122,122,.12);border-color:rgba(255,122,122,.22);color:#ffc1c1}
    .high{background:rgba(245,192,106,.14);border-color:rgba(245,192,106,.24);color:#ffe0a8}
    .moderate{background:rgba(122,162,255,.14);border-color:rgba(122,162,255,.24);color:#d1e0ff}
    .stable{background:rgba(56,211,159,.12);border-color:rgba(56,211,159,.22);color:#c7f6e3}

    .focus-wrap{display:grid;gap:12px}
    .kv{
      display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.05)
    }
    .kv:last-child{border-bottom:none}
    .k{color:#b6c8e8;font-weight:700}
    .v{font-weight:900}
    .focus-top{
      background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);
      border-radius:18px;padding:16px;
    }
    .focus-name{font-size:16px;color:#a7bad9;font-weight:800}
    .focus-id{font-size:32px;font-weight:1000;line-height:1.05;margin:6px 0}
    .focus-risk{font-size:18px;color:#dbe7fb;font-weight:800}
    .panel-block{
      background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);
      border-radius:18px;padding:16px;margin-top:12px;
    }
    .channels{
      font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      color:#d7e5ff;font-size:13px;line-height:1.7;word-break:break-word
    }

    .two-col{
      display:grid;grid-template-columns:1fr 1fr;gap:14px;
    }
    .demo-card,.summary-card{padding:24px}
    iframe{
      width:100%;aspect-ratio:16/9;border:0;border-radius:18px;background:#0a0f18;
    }
    .summary-list{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-top:16px}
    .summary-item{display:flex;flex-direction:column;gap:3px}
    .summary-item .sk{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9db3d5;font-weight:900}
    .summary-item .sv{font-size:20px;font-weight:900}

    .cred-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
    }
    .cred-card{padding:22px}
    .cred-card h3{margin:0 0 8px;font-size:22px}
    .cred-card p{margin:0;color:#bfd0ea;line-height:1.65}

    .trust-grid{
      display:grid;grid-template-columns:repeat(4,1fr);gap:14px;
    }
    .trust-card{padding:22px}
    .trust-card h3{margin:0 0 8px;font-size:18px}
    .trust-card p{margin:0;color:#bfd0ea;line-height:1.65}

    .founder-grid{
      display:grid;grid-template-columns:1.1fr .9fr;gap:14px;
    }
    .founder-card{padding:24px}
    .founder-name{font-size:34px;font-weight:1000;line-height:1;margin:10px 0}
    .founder-role{font-size:16px;color:#bcd0ec;font-weight:800;margin-bottom:16px}
    .founder-copy{color:#d3e1f8;line-height:1.7}
    .contact-box{
      display:grid;gap:12px;margin-top:18px;
    }
    .contact-row{
      background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);
      border-radius:16px;padding:14px 16px;
    }
    .contact-label{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9eb4d6;font-weight:900}
    .contact-value{font-size:18px;font-weight:900;margin-top:6px;word-break:break-word}

    .investor-banner{
      padding:24px;
      background:linear-gradient(180deg, rgba(122,162,255,.12), rgba(122,162,255,.04));
      border:1px solid rgba(122,162,255,.18);
    }
    .investor-banner h3{margin:0 0 10px;font-size:28px}
    .investor-banner p{margin:0;color:#d5e4ff;line-height:1.65}
    .banner-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}

    .footer{
      margin-top:24px;padding:24px 12px 8px;color:#94a7c6;font-size:14px;text-align:center
    }

    @media (max-width:1100px){
      .hero,.dashboard-grid,.two-col,.cred-grid,.metrics-grid,.trust-grid,.founder-grid{grid-template-columns:1fr 1fr}
      .dashboard-grid .dash-card:first-child{grid-column:1/-1}
      .hero h1{font-size:54px}
    }
    @media (max-width:760px){
      .hero,.dashboard-grid,.two-col,.cred-grid,.metrics-grid,.trust-grid,.mini-grid,.summary-list,.founder-grid{grid-template-columns:1fr}
      .brand-title{font-size:30px}
      .hero h1{font-size:42px}
      .section-title{font-size:32px}
      .metric-value{font-size:42px}
    }
  </style>
</head>
<body>
  <div class="nav">
    <div class="nav-inner">
      <div class="brand">
        <div class="brand-kicker">AI-Powered Predictive Clinical Intelligence</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Hospitals · Clinics · Investors · Patients</div>
      </div>
      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#metrics">Metrics</a>
        <a href="#dashboard">Live Dashboard</a>
        <a href="#credibility">Credibility</a>
        <a href="#demo">Demo</a>
        <a href="/investors">Investor View</a>
        <a class="btn" href="/deck">Download Pitch Deck</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="card hero-main">
        <div class="hero-kicker">Clinical Command Platform</div>
        <h1>Detect patient deterioration earlier. Strengthen hospital response at scale.</h1>
        <p>
          Early Risk Alert AI is a professional healthcare intelligence platform built to help hospitals,
          care teams, clinics, and remote monitoring programs surface high-risk patients in real time,
          prioritize action faster, and present enterprise-grade command-center visibility.
        </p>
        <div class="hero-actions">
          <a class="btn" href="#dashboard">Open Live Dashboard</a>
          <a class="btn secondary" href="#demo">View Demo Section</a>
          <a class="btn ghost" href="/investors">Open Investor Portal</a>
        </div>

        <div class="ticker-wrap">
          <div class="ticker-track">
            <div class="ticker-item"><span class="pulse-bar"></span> Live alert routing active</div>
            <div class="ticker-item"><span class="pulse-bar"></span> AI risk scoring connected</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Command-center visibility enabled</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Hospital-facing monitoring flow live</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Investor-ready demo platform active</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Live alert routing active</div>
            <div class="ticker-item"><span class="pulse-bar"></span> AI risk scoring connected</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Command-center visibility enabled</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Hospital-facing monitoring flow live</div>
            <div class="ticker-item"><span class="pulse-bar"></span> Investor-ready demo platform active</div>
          </div>
        </div>
      </div>

      <div class="card hero-side">
        <div class="live-pill"><span class="dot"></span> Live system</div>
        <div class="side-title">Built for modern hospital operations</div>
        <p class="side-copy">
          Your public-facing platform now combines hospital product messaging, animated live metrics,
          AI risk scoring, command-center style monitoring, investor-ready presentation, and enterprise trust positioning in one polished experience.
        </p>
        <div class="mini-grid">
          <div class="mini">
            <div class="mini-k">Model</div>
            <div class="mini-v">AI</div>
            <div class="mini-s">Live deterioration scoring</div>
          </div>
          <div class="mini">
            <div class="mini-k">Deployment</div>
            <div class="mini-v">SaaS</div>
            <div class="mini-s">Cloud-based enterprise rollout</div>
          </div>
          <div class="mini">
            <div class="mini-k">Buyer</div>
            <div class="mini-v">Hospitals</div>
            <div class="mini-s">Health systems & RPM teams</div>
          </div>
          <div class="mini">
            <div class="mini-k">Use Case</div>
            <div class="mini-v">Command</div>
            <div class="mini-s">Real-time visibility & alerts</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="metrics">
      <h2 class="section-title">Animated live metrics</h2>
      <p class="section-sub">
        These metrics refresh from your API layer to create a more professional enterprise experience for hospitals, clinics, and investors.
      </p>
      <div class="metrics-grid">
        <div class="card metric">
          <div class="metric-label">Patients monitored</div>
          <div class="metric-value" id="patientCount">0</div>
          <div class="metric-note">Active monitored population</div>
          <div class="progress"><span id="barPatients"></span></div>
        </div>
        <div class="card metric">
          <div class="metric-label">Open alerts</div>
          <div class="metric-value" id="openAlerts">0</div>
          <div class="metric-note">Detected elevated-risk cases</div>
          <div class="progress"><span id="barAlerts"></span></div>
        </div>
        <div class="card metric">
          <div class="metric-label">Critical alerts</div>
          <div class="metric-value" id="criticalAlerts">0</div>
          <div class="metric-note">Highest-priority intervention queue</div>
          <div class="progress"><span id="barCritical"></span></div>
        </div>
        <div class="card metric">
          <div class="metric-label">Avg risk score</div>
          <div class="metric-value" id="avgRisk">0</div>
          <div class="metric-note">AI-generated risk trend</div>
          <div class="progress"><span id="barRisk"></span></div>
        </div>
      </div>
    </section>

    <section class="section" id="dashboard">
      <h2 class="section-title">Hospital command center dashboard</h2>
      <p class="section-sub">
        A professional command-center layout showing alerts, focused patient detail, system health, and monitoring state in one place.
      </p>
      <div class="dashboard-grid">
        <div class="card dash-card">
          <h3>Live Alerts Feed</h3>
          <p>Clinical alerts, prioritization, and real-time visibility.</p>
          <div class="feed" id="alertsFeed"></div>
        </div>

        <div class="card dash-card">
          <h3>Patient Focus</h3>
          <p>Focused patient view for vitals, event timing, and rollup metrics.</p>
          <div class="focus-wrap">
            <div class="focus-top">
              <div class="focus-name">Patient focus</div>
              <div class="focus-id" id="focusPatientId">p101</div>
              <div class="focus-risk" id="focusPatientMessage">Loading live data...</div>
              <div style="margin-top:12px">
                <span class="badge moderate" id="focusSeverity">moderate</span>
              </div>
            </div>

            <div class="panel-block" id="focusVitals"></div>
            <div class="panel-block" id="focusRollups"></div>
          </div>
        </div>

        <div class="card dash-card">
          <h3>System Panels</h3>
          <p>Operational stream health, monitoring state, and channels.</p>
          <div class="panel-block" id="streamHealth"></div>
          <div class="panel-block">
            <div class="k" style="margin-bottom:10px">Channels</div>
            <div class="channels" id="channelList">Loading...</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="demo">
      <div class="two-col">
        <div class="card demo-card">
          <h2 class="section-title" style="margin-top:0">Live Product Demo</h2>
          <p class="section-sub" style="margin-bottom:16px">
            Your live YouTube demo is embedded below for platform walkthroughs and investor presentations.
          </p>
          <iframe src="{{ youtube_embed_url }}" title="Early Risk Alert AI Demo" allowfullscreen></iframe>
        </div>

        <div class="card summary-card">
          <h2 class="section-title" style="margin-top:0">Platform summary</h2>
          <p class="section-sub" style="margin-bottom:0">
            This main page now combines your hospital-facing product pitch, live metrics, command dashboard,
            AI platform narrative, and professional business presentation into one unified platform experience.
          </p>

          <div class="summary-list">
            <div class="summary-item"><span class="sk">Use case</span><span class="sv">Clinical intelligence</span></div>
            <div class="summary-item"><span class="sk">Deployment</span><span class="sv">Cloud-based SaaS</span></div>
            <div class="summary-item"><span class="sk">Buyer</span><span class="sv">Hospitals / RPM</span></div>
            <div class="summary-item"><span class="sk">Expansion</span><span class="sv">Enterprise rollout</span></div>
          </div>

          <div class="hero-actions" style="margin-top:22px">
            <a class="btn" href="/investors">Investor View</a>
            <a class="btn secondary" href="/deck">Pitch Deck PDF</a>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="credibility">
      <h2 class="section-title">Business credibility layer</h2>
      <p class="section-sub">
        Early Risk Alert AI now presents as a broader multi-audience platform for hospitals, clinics, investors, and patients.
      </p>
      <div class="cred-grid">
        <div class="card cred-card">
          <h3>Professional presence</h3>
          <p>Live domain, company branding, demo materials, and a polished hospital-facing product experience strengthen institutional confidence.</p>
        </div>
        <div class="card cred-card">
          <h3>Command-center positioning</h3>
          <p>Hospitals can see alerts, vitals, patient focus, AI risk scoring, and operational visibility in an enterprise-style interface.</p>
        </div>
        <div class="card cred-card">
          <h3>Investor readiness</h3>
          <p>Your investor portal, live demo, and command dashboard create a stronger story for outreach, demos, and future funding discussions.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Enterprise readiness</h2>
      <p class="section-sub">
        These trust signals help hospitals and healthcare buyers understand the platform as a serious operational solution.
      </p>
      <div class="trust-grid">
        <div class="card trust-card">
          <h3>HIPAA-ready architecture</h3>
          <p>Structured for secure healthcare deployment workflows and enterprise-focused operational design.</p>
        </div>
        <div class="card trust-card">
          <h3>Secure cloud infrastructure</h3>
          <p>Cloud-based delivery model designed for live access, operational resilience, and scalable rollout.</p>
        </div>
        <div class="card trust-card">
          <h3>Real-time clinical monitoring</h3>
          <p>Built to support streaming vitals, alert prioritization, patient focus workflows, and command visibility.</p>
        </div>
        <div class="card trust-card">
          <h3>Hospital command center fit</h3>
          <p>Purpose-built presentation for hospital command teams, remote monitoring programs, and clinical operations visibility.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Founder credibility</h2>
      <p class="section-sub">
        Hospitals, partners, and investors trust the platform more when they can clearly see who is building it and why.
      </p>
      <div class="founder-grid">
        <div class="card founder-card">
          <div class="hero-kicker">Founder & Platform Architect</div>
          <div class="founder-name">{{ founder_name }}</div>
          <div class="founder-role">AI Systems Engineer • Founder • Clinical Intelligence Platform Builder</div>
          <div class="founder-copy">
            Early Risk Alert AI began as an app and evolved into a broader professional platform serving hospitals,
            clinics, investors, and patients. The focus is now on platform quality, AI advancement, dashboard polish,
            real-world usability, and enterprise presentation before moving deeper into funding.
          </div>
        </div>

        <div class="card founder-card">
          <div class="hero-kicker">Business Contact</div>
          <div class="contact-box">
            <div class="contact-row">
              <div class="contact-label">Email</div>
              <div class="contact-value">{{ business_email }}</div>
            </div>
            <div class="contact-row">
              <div class="contact-label">Business Phone</div>
              <div class="contact-value">{{ business_phone }}</div>
            </div>
            <div class="contact-row">
              <div class="contact-label">Platform Access</div>
              <div class="contact-value"><a href="/investors">Investor View</a> • <a href="/deck">Pitch Deck</a></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card investor-banner">
        <h3>Investor and partnership mode</h3>
        <p>
          Early Risk Alert AI now presents as a scalable predictive clinical intelligence platform for hospitals,
          health systems, remote care programs, strategic partners, and investors seeking exposure to modern healthcare infrastructure.
        </p>
        <div class="banner-actions">
          <a class="btn" href="/investors">Open Investor View</a>
          <a class="btn secondary" href="#demo">See Demo</a>
          <a class="btn ghost" href="/deck">Download Deck</a>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC • Main Platform • {{ founder_name }} • {{ business_phone }}
    </div>
  </div>

  <script>
    const tenantId = "demo";
    const patientId = "p101";

    function animateNumber(el, target, duration=900) {
      const start = Number(el.dataset.value || 0);
      const end = Number(target || 0);
      const startTime = performance.now();

      function frame(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const current = Math.round(start + (end - start) * progress);
        el.textContent = current;
        if (progress < 1) requestAnimationFrame(frame);
        else el.dataset.value = end;
      }
      requestAnimationFrame(frame);
    }

    function badgeClass(severity) {
      const s = (severity || "").toLowerCase();
      if (s === "critical") return "badge critical";
      if (s === "high") return "badge high";
      if (s === "moderate") return "badge moderate";
      return "badge stable";
    }

    function severityText(severity) {
      return (severity || "stable").toLowerCase();
    }

    async function getJson(url) {
      const res = await fetch(url, {cache: "no-store"});
      if (!res.ok) throw new Error("Request failed");
      return res.json();
    }

    function setProgress(id, value, max) {
      const pct = Math.max(4, Math.min(100, Math.round((value / max) * 100)));
      document.getElementById(id).style.width = pct + "%";
    }

    function renderAlerts(alerts) {
      const root = document.getElementById("alertsFeed");
      if (!alerts || !alerts.length) {
        root.innerHTML = '<div class="alert"><div class="alert-name">No active alerts</div><div class="alert-msg">System stable</div></div>';
        return;
      }
      root.innerHTML = alerts.slice(0, 5).map(a => `
        <div class="alert">
          <div class="alert-top">
            <div>
              <div class="alert-name">${a.alert_type}</div>
              <div class="alert-patient">${a.patient_name || a.patient_id}</div>
            </div>
            <span class="${badgeClass(a.severity)}">${severityText(a.severity)}</span>
          </div>
          <div class="alert-msg">${a.message}</div>
          <div class="alert-time">${a.created_at || "live"}</div>
        </div>
      `).join("");
    }

    function renderFocus(snapshot) {
      const fp = snapshot.focus_patient || {};
      const vitals = fp.vitals || {};
      const risk = fp.risk || {};
      const roll = ((fp || {}).rollups) || {};

      document.getElementById("focusPatientId").textContent = fp.patient_id || "p101";
      document.getElementById("focusPatientMessage").textContent = risk.alert_message || "Monitoring active";
      const sev = document.getElementById("focusSeverity");
      sev.className = badgeClass(risk.severity);
      sev.textContent = severityText(risk.severity);

      document.getElementById("focusVitals").innerHTML = `
        <div class="k" style="margin-bottom:10px">Latest vitals</div>
        <div class="kv"><span class="k">Heart rate</span><span class="v">${vitals.heart_rate ?? "--"}</span></div>
        <div class="kv"><span class="k">Systolic BP</span><span class="v">${vitals.systolic_bp ?? "--"}</span></div>
        <div class="kv"><span class="k">Diastolic BP</span><span class="v">${vitals.diastolic_bp ?? "--"}</span></div>
        <div class="kv"><span class="k">SpO2</span><span class="v">${vitals.spo2 ?? "--"}</span></div>
        <div class="kv"><span class="k">Temperature</span><span class="v">${vitals.temperature ?? "--"}</span></div>
        <div class="kv"><span class="k">Resp. rate</span><span class="v">${vitals.resp_rate ?? "--"}</span></div>
      `;

      document.getElementById("focusRollups").innerHTML = `
        <div class="k" style="margin-bottom:10px">Patient rollups</div>
        <div class="kv"><span class="k">AI risk score</span><span class="v">${risk.risk_score ?? "--"}</span></div>
        <div class="kv"><span class="k">Confidence</span><span class="v">${risk.confidence ?? "--"}</span></div>
        <div class="kv"><span class="k">Avg heart rate</span><span class="v">${roll.avg_heart_rate ?? "--"}</span></div>
        <div class="kv"><span class="k">Avg systolic BP</span><span class="v">${roll.avg_systolic_bp ?? "--"}</span></div>
        <div class="kv"><span class="k">Avg diastolic BP</span><span class="v">${roll.avg_diastolic_bp ?? "--"}</span></div>
        <div class="kv"><span class="k">Avg SpO2</span><span class="v">${roll.avg_spo2 ?? "--"}</span></div>
      `;
    }

    function renderStreamHealth(health) {
      document.getElementById("streamHealth").innerHTML = `
        <div class="kv"><span class="k">Status</span><span class="v">${health.status || "running"}</span></div>
        <div class="kv"><span class="k">Redis OK</span><span class="v">${String(health.redis_ok)}</span></div>
        <div class="kv"><span class="k">Mode</span><span class="v">${health.mode || "realtime"}</span></div>
        <div class="kv"><span class="k">Worker</span><span class="v">${health.worker_status || "active"}</span></div>
      `;
    }

    function renderChannels(data) {
      document.getElementById("channelList").innerHTML = (data.channels || []).join("<br>");
    }

    async function loadDashboard() {
      try {
        const [overview, snapshot, streamHealth, channels] = await Promise.all([
          getJson(`/api/dashboard/overview?tenant_id=${tenantId}`),
          getJson(`/api/live-snapshot?tenant_id=${tenantId}&patient_id=${patientId}`),
          getJson(`/api/stream/health?tenant_id=${tenantId}`),
          getJson(`/api/stream/channels?tenant_id=${tenantId}&patient_id=${patientId}`)
        ]);

        animateNumber(document.getElementById("patientCount"), overview.patient_count || 0);
        animateNumber(document.getElementById("openAlerts"), overview.open_alerts || 0);
        animateNumber(document.getElementById("criticalAlerts"), overview.critical_alerts || 0);
        animateNumber(document.getElementById("avgRisk"), overview.avg_risk_score || 0);

        setProgress("barPatients", overview.patient_count || 0, 2000);
        setProgress("barAlerts", overview.open_alerts || 0, 20);
        setProgress("barCritical", overview.critical_alerts || 0, 10);
        setProgress("barRisk", overview.avg_risk_score || 0, 100);

        renderAlerts(snapshot.alerts || []);
        renderFocus(snapshot);
        renderStreamHealth(streamHealth);
        renderChannels(channels);
      } catch (e) {
        document.getElementById("alertsFeed").innerHTML =
          '<div class="alert"><div class="alert-name">Connection issue</div><div class="alert-msg">Dashboard API not responding yet</div></div>';
        document.getElementById("streamHealth").innerHTML =
          '<div class="kv"><span class="k">Status</span><span class="v">degraded</span></div>';
        document.getElementById("channelList").textContent = "Waiting for API routes...";
      }
    }

    loadDashboard();
    setInterval(loadDashboard, 6000);
  </script>
</body>
</html>
"""

INVESTOR_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Investor Overview — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;--panel:#101a2d;--line:rgba(255,255,255,.08);
      --text:#edf4ff;--muted:#9eb0cc;--blue:#7aa2ff;--blue2:#5bd4ff;--radius:22px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.12), transparent 22%),
        linear-gradient(180deg,#08111f,#0d1526);
    }
    .shell{max-width:1200px;margin:0 auto;padding:24px 16px 60px}
    .nav{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
    .brand-k{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#bccce8;font-weight:900}
    .brand-t{font-size:38px;font-weight:1000;line-height:.96}
    .brand-s{font-size:14px;color:var(--muted);font-weight:700}
    .actions{display:flex;gap:10px;flex-wrap:wrap}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;padding:13px 18px;border-radius:16px;
      font-weight:900;font-size:14px;text-decoration:none
    }
    .btn.primary{background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f}
    .btn.secondary{background:#111b2f;color:var(--text);border:1px solid var(--line)}
    .hero,.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin-top:18px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--line);border-radius:var(--radius);padding:26px;
    }
    h1{margin:0 0 14px;font-size:60px;line-height:.94;letter-spacing:-.05em}
    p{margin:0;color:#c8d7f1;line-height:1.65}
    .kicker{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#bfd0ea;font-weight:900;margin-bottom:12px}
    .mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}
    .mini{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px}
    .mini-k{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#9bb3d5;font-weight:900}
    .mini-v{font-size:32px;font-weight:1000;margin-top:10px}
    .title{font-size:34px;font-weight:1000;letter-spacing:-.04em;margin:20px 0 8px}
    .cards3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
    .box{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:18px}
    .box h3{margin:0 0 8px;font-size:20px}
    .quote{font-size:24px;line-height:1.45;font-weight:850;color:#dce8fb}
    .contact-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:16px}
    .contact-card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px}
    .contact-label{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#9db3d5;font-weight:900}
    .contact-value{font-size:18px;font-weight:900;margin-top:8px;word-break:break-word}
    .footer{text-align:center;color:#8fa4c5;margin-top:24px}
    @media (max-width:900px){
      .hero,.grid,.cards3,.contact-grid,.mini-grid{grid-template-columns:1fr}
      h1{font-size:42px}
      .brand-t{font-size:32px}
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="nav">
      <div>
        <div class="brand-k">Investor Overview</div>
        <div class="brand-t">Early Risk Alert AI</div>
        <div class="brand-s">Predictive clinical intelligence platform</div>
      </div>
      <div class="actions">
        <a class="btn secondary" href="/">Main Platform</a>
        <a class="btn primary" href="/deck">Download Pitch Deck</a>
      </div>
    </div>

    <div class="hero">
      <div class="card">
        <div class="kicker">AI-Powered Predictive Clinical Intelligence</div>
        <h1>Built for hospital response, command-center visibility, and scalable enterprise deployment.</h1>
        <p>
          Early Risk Alert AI is positioned as a modern healthcare intelligence platform that helps hospitals,
          health systems, clinics, and care networks identify deterioration risk earlier and prioritize action at scale.
        </p>
        <div class="actions" style="margin-top:20px">
          <a class="btn primary" href="/">Open Main Platform</a>
          <a class="btn secondary" href="/deck">Pitch Deck PDF</a>
        </div>
      </div>

      <div class="card">
        <div class="kicker">Investment Snapshot</div>
        <div class="mini-grid">
          <div class="mini"><div class="mini-k">Model</div><div class="mini-v">SaaS</div></div>
          <div class="mini"><div class="mini-k">Buyer</div><div class="mini-v">Hospitals</div></div>
          <div class="mini"><div class="mini-k">Use Case</div><div class="mini-v">Early Risk</div></div>
          <div class="mini"><div class="mini-k">Expansion</div><div class="mini-v">Multi-Site</div></div>
        </div>
      </div>
    </div>

    <div class="title">Why this matters now</div>
    <div class="cards3">
      <div class="box"><h3>Reactive care model</h3><p>Hospitals still depend on delayed intervention after deterioration has already escalated.</p></div>
      <div class="box"><h3>Operational pressure</h3><p>Staffing strain and fragmented monitoring tools reduce intervention speed and visibility.</p></div>
      <div class="box"><h3>Scalable platform timing</h3><p>Cloud delivery, RPM growth, and AI-assisted monitoring create strong expansion potential.</p></div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="title" style="margin-top:0">Positioning</div>
        <p class="quote">
          “Early Risk Alert AI brings live patient visibility, AI-assisted risk scoring,
          and command-center style operations into one professional healthcare platform.”
        </p>
      </div>

      <div class="card">
        <div class="title" style="margin-top:0">Founder</div>
        <p>
          {{ founder_name }} is building Early Risk Alert AI as a broader clinical intelligence platform for hospitals,
          investors, clinics, and patients, with platform quality, operational polish, and AI advancement prioritized first.
        </p>
      </div>
    </div>

    <div class="title">Investor contact</div>
    <div class="contact-grid">
      <div class="contact-card"><div class="contact-label">Founder</div><div class="contact-value">{{ founder_name }}</div></div>
      <div class="contact-card"><div class="contact-label">Email</div><div class="contact-value">{{ business_email }}</div></div>
      <div class="contact-card"><div class="contact-label">Business Phone</div><div class="contact-value">{{ business_phone }}</div></div>
      <div class="contact-card"><div class="contact-label">Live Demo Access</div><div class="contact-value"><a href="/">Open Hospital Command Center</a></div></div>
    </div>

    <div class="footer">Early Risk Alert AI LLC • Investor Overview • {{ business_phone }}</div>
  </div>
</body>
</html>
"""

@web_bp.get("/")
def home():
    return render_template_string(
        MAIN_HTML,
        youtube_embed_url=YOUTUBE_EMBED_URL,
        business_email=BUSINESS_EMAIL,
        business_phone=BUSINESS_PHONE,
        founder_name=FOUNDER_NAME,
    )

@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(
        MAIN_HTML,
        youtube_embed_url=YOUTUBE_EMBED_URL,
        business_email=BUSINESS_EMAIL,
        business_phone=BUSINESS_PHONE,
        founder_name=FOUNDER_NAME,
    )

@web_bp.get("/investors")
def investors():
    return render_template_string(
        INVESTOR_HTML,
        business_email=BUSINESS_EMAIL,
        business_phone=BUSINESS_PHONE,
        founder_name=FOUNDER_NAME,
    )

@web_bp.get("/deck")
def deck():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_path = os.path.join(root_dir, "static", "Early_Risk_Alert_AI_Pitch_Deck.pdf")
    if not os.path.exists(pdf_path):
        return "Pitch deck not found.", 404
    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf",
    )
