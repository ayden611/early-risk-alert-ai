from flask import Blueprint, render_template, render_template_string, send_from_directory
import os

web_bp = Blueprint("web", __name__)

COMMAND_CENTER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert Command Center</title>
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
      --accent2:#5b84f7;
      --red:#ff6b6b;
      --amber:#f5c06a;
      --green:#5bd38d;
      --glow:0 12px 40px rgba(0,0,0,.28);
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.10), transparent 22%),
        radial-gradient(circle at top right, rgba(91,211,141,.08), transparent 20%),
        var(--bg);
      color:var(--text);
    }
    a{color:inherit}
    .shell{max-width:1520px;margin:0 auto;padding:20px 18px 48px}
    .topbar{
      display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
      margin-bottom:18px;flex-wrap:wrap
    }
    .brand-kicker{
      color:#d8e4ff;font-size:11px;letter-spacing:.18em;font-weight:900;
      text-transform:uppercase;opacity:.9;margin-bottom:6px
    }
    .title{
      font-size:48px;font-weight:900;line-height:1;
      letter-spacing:-.03em;margin:0 0 8px
    }
    .subtitle{
      color:var(--muted);font-size:15px;font-weight:600
    }
    .live-row{
      display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:14px
    }
    .live-system{
      display:inline-flex;align-items:center;gap:10px;
      background:rgba(91,211,141,.11);
      border:1px solid rgba(91,211,141,.24);
      color:#cbffe0;padding:8px 12px;border-radius:999px;
      font-size:12px;font-weight:900;letter-spacing:.08em;text-transform:uppercase
    }
    .dot{
      width:10px;height:10px;border-radius:999px;background:var(--green);
      box-shadow:0 0 14px rgba(91,211,141,.9);
      animation:pulse 1.5s infinite
    }
    @keyframes pulse{
      0%{transform:scale(1);opacity:1}
      50%{transform:scale(1.28);opacity:.7}
      100%{transform:scale(1);opacity:1}
    }

    .hero-controls{
      display:flex;gap:10px;align-items:center;flex-wrap:wrap;
      background:rgba(255,255,255,.03);
      border:1px solid var(--border);
      border-radius:18px;padding:12px
    }
    input{
      border:1px solid var(--border);
      background:var(--panel2);
      color:var(--text);
      padding:12px 14px;
      border-radius:12px;
      font-size:14px;
      min-width:120px;
      outline:none;
    }
    input:focus{border-color:rgba(122,162,255,.45)}
    button,.navbtn,.linkbtn{
      background:linear-gradient(180deg,var(--accent),var(--accent2));
      color:#08111f;font-weight:900;border:none;cursor:pointer;
      padding:12px 16px;border-radius:12px;text-decoration:none;
      display:inline-flex;align-items:center;justify-content:center;
      box-shadow:0 8px 24px rgba(91,132,247,.28)
    }
    .navbtn.secondary,.linkbtn.secondary{
      background:#151f34;color:var(--text);box-shadow:none;border:1px solid var(--border)
    }

    .top-links{
      display:flex;gap:10px;flex-wrap:wrap
    }

    .tabs{
      display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 16px
    }
    .tab{
      padding:11px 16px;border-radius:999px;
      background:#10192c;border:1px solid var(--border);
      color:#cfe0ff;font-size:13px;font-weight:800;cursor:pointer
    }
    .tab.active{
      background:linear-gradient(180deg,var(--accent),var(--accent2));
      color:#09111f;border-color:transparent
    }

    .ticker{
      width:100%;overflow:hidden;
      background:linear-gradient(90deg, rgba(255,255,255,.03), rgba(255,255,255,.05), rgba(255,255,255,.03));
      border:1px solid var(--border);
      border-radius:16px;
      margin:0 0 18px 0;
      box-shadow:var(--glow)
    }
    .ticker-track{
      white-space:nowrap;
      padding:11px 0;
      font-weight:800;
      color:#ffd3d3;
      animation:tickerMove 34s linear infinite
    }
    @keyframes tickerMove{
      0%{transform:translateX(100%)}
      100%{transform:translateX(-100%)}
    }

    .kpis{
      display:grid;
      grid-template-columns:repeat(6,1fr);
      gap:14px;
      margin-bottom:18px
    }
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:22px;
      padding:18px;
      box-shadow:var(--glow);
      backdrop-filter: blur(10px);
    }
    .label{color:var(--muted);font-size:13px;font-weight:700}
    .value{font-size:44px;font-weight:900;line-height:1;margin-top:10px;letter-spacing:-.03em}
    .value.small{font-size:28px}

    .layout{
      display:grid;
      grid-template-columns:1.15fr .95fr .95fr;
      gap:16px
    }
    .section{
      font-size:24px;font-weight:900;letter-spacing:-.02em;margin-bottom:14px
    }
    .list{display:flex;flex-direction:column;gap:12px}
    .item{
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.02);
      border-radius:16px;
      padding:14px
    }
    .row{
      display:flex;justify-content:space-between;gap:10px;align-items:flex-start
    }
    .muted{color:var(--muted)}
    .small{font-size:12px}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .badge{
      display:inline-flex;align-items:center;justify-content:center;
      min-width:86px;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:900;
      text-transform:uppercase;letter-spacing:.04em
    }
    .critical,.risk-high{background:rgba(255,107,107,.18);color:#ffd0d0}
    .high,.risk-moderate{background:rgba(245,192,106,.18);color:#ffe3af}
    .info,.risk-low{background:rgba(91,211,141,.18);color:#d5ffe3}
    .kv{
      display:grid;grid-template-columns:1fr auto;gap:8px 12px;font-size:14px
    }
    .footer-note{
      margin-top:18px;color:#6e82a7;font-size:13px;text-align:right
    }

    .premium-note{
      margin-top:12px;
      padding:12px 14px;border-radius:14px;
      background:rgba(122,162,255,.08);
      border:1px solid rgba(122,162,255,.18);
      color:#dbe8ff;font-size:13px;font-weight:700
    }

    /* Investor page */
    .investor-shell{max-width:1400px;margin:0 auto;padding:22px 18px 60px}
    .investor-nav{
      position:sticky;top:0;z-index:50;
      display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
      padding:16px 0 18px;
      background:linear-gradient(180deg, rgba(8,17,31,.96), rgba(8,17,31,.88));
      backdrop-filter: blur(10px);
      border-bottom:1px solid rgba(255,255,255,.05);
      margin-bottom:24px
    }
    .investor-logo-wrap{display:flex;flex-direction:column;gap:4px}
    .investor-logo-kicker{
      font-size:11px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#d8e4ff
    }
    .investor-logo{
      font-size:32px;font-weight:900;letter-spacing:-.03em
    }
    .investor-nav-links{
      display:flex;gap:18px;align-items:center;flex-wrap:wrap
    }
    .investor-nav-links a{
      text-decoration:none;color:#dfe9ff;font-weight:800;font-size:14px
    }

    .hero-grid{
      display:grid;grid-template-columns:1.15fr .85fr;gap:22px;align-items:stretch
    }
    .hero-copy,.hero-side{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:28px;padding:28px;box-shadow:var(--glow)
    }
    .hero-kicker{
      color:#dce7ff;font-size:12px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;margin-bottom:16px
    }
    .hero-headline{
      font-size:78px;line-height:.95;font-weight:950;letter-spacing:-.06em;margin:0 0 18px
    }
    .hero-body{
      font-size:22px;line-height:1.45;color:#d3def2;max-width:900px
    }
    .hero-sub{
      margin-top:16px;font-size:17px;line-height:1.65;color:#a9bad8
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}

    .invest-highlight{
      font-size:32px;font-weight:900;letter-spacing:-.03em;margin:0 0 14px
    }
    .invest-side-text{
      color:#d5e1f5;font-size:18px;line-height:1.6;margin-bottom:18px
    }
    .highlight-grid{
      display:grid;grid-template-columns:1fr 1fr;gap:12px
    }
    .highlight-box{
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.02);
      border-radius:18px;padding:16px;min-height:110px
    }
    .highlight-box .hl-k{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#9cb1d6;font-weight:900}
    .highlight-box .hl-v{font-size:32px;font-weight:950;letter-spacing:-.04em;margin-top:10px}
    .highlight-box .hl-s{margin-top:6px;color:#b7c7e4;font-size:13px;font-weight:700;line-height:1.45}

    .micro-grid{
      display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px
    }
    .micro{
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.02);
      border-radius:16px;padding:15px
    }
    .micro .n{font-size:22px;font-weight:950}
    .micro .t{color:#bccce8;font-size:13px;font-weight:700;margin-top:6px;line-height:1.4}

    .invest-section{margin-top:34px}
    .invest-title{font-size:40px;font-weight:950;letter-spacing:-.04em;margin:0 0 18px}
    .invest-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .invest-grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
    .invest-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);border-radius:22px;padding:22px;box-shadow:var(--glow)
    }
    .invest-card h3{
      margin:0 0 10px;font-size:24px;font-weight:900;letter-spacing:-.02em
    }
    .invest-card p,.invest-card li{
      color:#c9d7ef;font-size:16px;line-height:1.6
    }
    .invest-card ul{margin:12px 0 0 18px;padding:0}
    .band{
      background:linear-gradient(180deg, rgba(122,162,255,.08), rgba(122,162,255,.03));
      border:1px solid rgba(122,162,255,.16);
      border-radius:26px;padding:28px;box-shadow:var(--glow)
    }
    .quote{
      font-size:28px;line-height:1.45;color:#eaf2ff;font-weight:800;letter-spacing:-.02em
    }

    .video-wrap{
      position:relative;
      width:100%;
      padding-top:56.25%;
      border-radius:22px;
      overflow:hidden;
      border:1px solid rgba(255,255,255,.08);
      box-shadow:var(--glow)
    }
    .video-wrap iframe{
      position:absolute;inset:0;width:100%;height:100%;border:0
    }

    .contact-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:16px
    }
    .contact-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:20px;padding:20px;box-shadow:var(--glow)
    }
    .contact-label{
      font-size:11px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:#9fb3d8;margin-bottom:10px
    }
    .contact-value{
      font-size:28px;font-weight:900;letter-spacing:-.03em
    }
    .contact-value.small{
      font-size:20px;line-height:1.4;word-break:break-word
    }
    .center{text-align:center}
    .footer{
      margin-top:42px;padding-top:18px;border-top:1px solid rgba(255,255,255,.06);
      color:#88a0c8;font-size:14px;text-align:center
    }

    @media (max-width:1280px){
      .hero-headline{font-size:64px}
      .kpis{grid-template-columns:repeat(3,1fr)}
      .hero-grid{grid-template-columns:1fr}
      .invest-grid-3{grid-template-columns:1fr 1fr}
      .contact-grid{grid-template-columns:1fr 1fr}
      .micro-grid{grid-template-columns:1fr 1fr}
    }
    @media (max-width:900px){
      .layout{grid-template-columns:1fr}
      .invest-grid-3,.invest-grid-2,.contact-grid,.highlight-grid,.micro-grid{grid-template-columns:1fr}
      .title{font-size:38px}
      .hero-headline{font-size:48px}
      .investor-logo{font-size:26px}
      .value{font-size:34px}
      .shell,.investor-shell{padding-left:14px;padding-right:14px}
    }
    @media (max-width:640px){
      .kpis{grid-template-columns:repeat(2,1fr)}
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="brand-kicker">Predictive Clinical Intelligence</div>
        <div class="title">Early Risk Alert</div>
        <div class="subtitle">Hospital UI + Investor Demo + Command Center</div>
        <div class="live-row">
          <div class="live-system"><span class="dot"></span> Live System</div>
        </div>
      </div>

      <div class="top-links">
        <a class="navbtn secondary" href="/investors">Investor View</a>
        <a class="navbtn secondary" href="/pitch-deck">Download Deck</a>
      </div>
    </div>

    <div class="hero-controls">
      <input id="tenantId" value="demo" placeholder="tenant_id">
      <input id="patientId" value="p101" placeholder="patient_id">
      <button onclick="refreshAll()">Refresh</button>
    </div>

    <div class="tabs">
      <button class="tab active" data-view="hospital">Hospital View</button>
      <button class="tab" data-view="executive">Executive View</button>
      <button class="tab" data-view="investor">Investor View</button>
    </div>

    <div class="ticker">
      <div class="ticker-track" id="alertTicker">🚨 LIVE ALERTS — System monitoring patients in real time</div>
    </div>

    <div class="kpis">
      <div class="card"><div class="label">Patients</div><div class="value" id="kpiPatients">—</div></div>
      <div class="card"><div class="label">Open Alerts</div><div class="value" id="kpiOpenAlerts">—</div></div>
      <div class="card"><div class="label">Critical Alerts</div><div class="value" id="kpiCriticalAlerts">—</div></div>
      <div class="card"><div class="label">Events Last Hour</div><div class="value" id="kpiEventsHour">—</div></div>
      <div class="card"><div class="label">Stream Mode</div><div class="value small" id="kpiStreamMode">—</div></div>
      <div class="card"><div class="label">Commercial Model</div><div class="value small" id="kpiCommercialModel">Enterprise SaaS</div></div>
    </div>

    <div class="layout">
      <div class="card">
        <div class="section">Live Alerts Feed</div>
        <div id="alertsList" class="list"></div>
      </div>

      <div class="card">
        <div class="section">Patient Focus</div>
        <div class="list">
          <div class="item">
            <div class="row">
              <div>
                <div class="muted">Patient</div>
                <div id="patientIdLabel" class="mono" style="font-size:30px;font-weight:900">—</div>
              </div>
              <div id="riskBadge" class="badge risk-low">unknown</div>
            </div>
            <div id="riskSummary" class="muted" style="margin-top:10px">No data yet.</div>
          </div>

          <div class="item">
            <div class="muted">Latest vitals</div>
            <div class="kv" style="margin-top:10px">
              <div>Heart rate</div><div id="vHr">—</div>
              <div>Systolic BP</div><div id="vSys">—</div>
              <div>Diastolic BP</div><div id="vDia">—</div>
              <div>SpO2</div><div id="vSpo2">—</div>
              <div>Event time</div><div id="vTs" class="small">—</div>
            </div>
          </div>

          <div class="item">
            <div class="muted">Patient rollups</div>
            <div id="rollupsBox" style="margin-top:10px"></div>
          </div>

          <div class="item">
            <div class="muted">Operational Note</div>
            <div class="premium-note" id="opsNote">
              Designed to demonstrate centralized monitoring, early intervention support, and clinical intelligence visibility across high-volume care environments.
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="section">System Panels</div>
        <div class="list">
          <div class="item">
            <div class="muted">Stream health</div>
            <div id="streamBox" style="margin-top:10px"></div>
          </div>

          <div class="item">
            <div class="muted">Channels</div>
            <div id="channelsBox" class="small mono" style="margin-top:10px"></div>
          </div>

          <div class="item">
            <div class="muted">Summary</div>
            <div id="summaryBox" style="margin-top:10px"></div>
          </div>

          <div class="item">
            <div class="muted">AI Risk Engine Status</div>
            <div id="engineBox" style="margin-top:10px"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="footer-note">Early Risk Alert AI™ • Premium Live Command Center Demo</div>
  </div>

  <script>
    let currentView = "hospital";

    function tenant(){ return document.getElementById("tenantId").value || "demo"; }
    function patient(){ return document.getElementById("patientId").value || "p101"; }

    function esc(v){
      return (v ?? "")
        .toString()
        .replaceAll("&","&amp;")
        .replaceAll("<","&lt;")
        .replaceAll(">","&gt;");
    }

    function riskClass(v){
      v = (v || "").toLowerCase();
      if (v === "high") return "risk-high";
      if (v === "moderate") return "risk-moderate";
      return "risk-low";
    }

    function sevClass(v){
      v = (v || "").toLowerCase();
      if (v.includes("critical")) return "critical";
      if (v.includes("high")) return "high";
      return "info";
    }

    async function getJson(url){
      try{
        const r = await fetch(url);
        if (!r.ok) return null;
        return await r.json();
      }catch(e){
        return null;
      }
    }

    function applyViewMode(){
      document.querySelectorAll(".tab").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.view === currentView);
      });

      const model = document.getElementById("kpiCommercialModel");
      const opsNote = document.getElementById("opsNote");

      if (currentView === "executive"){
        model.textContent = "Executive Ops";
        opsNote.textContent = "Executive mode emphasizes network oversight, alert burden, patient visibility, and rapid-response coordination across enterprise operations.";
      } else if (currentView === "investor"){
        model.textContent = "Enterprise SaaS";
        opsNote.textContent = "Investor mode emphasizes recurring revenue infrastructure, scalable hospital deployment, command-center differentiation, and platform expansion potential.";
      } else {
        model.textContent = "Enterprise SaaS";
        opsNote.textContent = "Designed to demonstrate centralized monitoring, early intervention support, and clinical intelligence visibility across high-volume care environments.";
      }
    }

    async function refreshOverview(){
      const ov = await getJson(`/api/v1/dashboard/overview?tenant_id=${encodeURIComponent(tenant())}`);
      if (!ov) return null;
      document.getElementById("kpiPatients").textContent = ov.patient_count ?? 0;
      document.getElementById("kpiOpenAlerts").textContent = ov.open_alerts ?? 0;
      document.getElementById("kpiCriticalAlerts").textContent = ov.critical_alerts ?? 0;
      document.getElementById("kpiEventsHour").textContent = ov.events_last_hour ?? 0;
      return ov;
    }

    async function refreshStream(){
      const stream = await getJson(`/api/v1/stream/health`);
      document.getElementById("kpiStreamMode").textContent = stream ? (stream.mode || "-") : "-";

      const streamBox = document.getElementById("streamBox");
      if (!stream){
        streamBox.innerHTML = '<div class="muted">Unavailable</div>';
      } else {
        streamBox.innerHTML = `
          <div class="kv">
            <div>Status</div><div>${esc(stream.status)}</div>
            <div>Redis OK</div><div>${esc(stream.redis_ok)}</div>
            <div>Mode</div><div>${esc(stream.mode)}</div>
          </div>
        `;
      }

      const channels = await getJson(`/api/v1/stream/channels?tenant_id=${encodeURIComponent(tenant())}&patient_id=${encodeURIComponent(patient())}`);
      document.getElementById("channelsBox").innerHTML =
        channels && channels.channels
          ? channels.channels.map(c => esc(c)).join("<br>")
          : '<div class="muted">No channels</div>';

      document.getElementById("engineBox").innerHTML = `
        <div class="kv">
          <div>Model status</div><div>Active</div>
          <div>Risk engine</div><div>Processing live streams</div>
          <div>Prediction latency</div><div>&lt; 1.2s</div>
          <div>Data sources</div><div>RPM / vitals / event streams</div>
          <div>Deployment class</div><div>Enterprise-ready</div>
          <div>Compliance architecture</div><div>HIPAA-ready design</div>
        </div>
      `;
    }

    async function refreshAlerts(){
      const data = await getJson(`/api/v1/alerts?tenant_id=${encodeURIComponent(tenant())}&limit=24`);
      const el = document.getElementById("alertsList");
      const ticker = document.getElementById("alertTicker");

      if (ticker && data && data.alerts && data.alerts.length){
        ticker.innerText = data.alerts.map(a =>
          `🚨 ${String(a.severity || "info").toUpperCase()} — Patient ${a.patient_id} — ${a.message}`
        ).join("     ✦     ");
      }

      if (!data || !data.alerts || data.alerts.length === 0){
        el.innerHTML = '<div class="muted">No alerts yet.</div>';
        return [];
      }

      el.innerHTML = data.alerts.map(a => `
        <div class="item">
          <div class="row">
            <div>
              <div><strong>${esc(a.alert_type)}</strong></div>
              <div class="muted">Patient ${esc(a.patient_id)}</div>
            </div>
            <div class="badge ${sevClass(a.severity)}">${esc(a.severity || "info")}</div>
          </div>
          <div style="margin-top:8px">${esc(a.message || "")}</div>
          <div class="muted small" style="margin-top:8px">${esc(a.created_at || "")}</div>
        </div>
      `).join("");

      return data.alerts;
    }

    async function refreshPatient(){
      const pid = patient();
      document.getElementById("patientIdLabel").textContent = pid;

      const vitals = await getJson(`/api/v1/vitals/latest?tenant_id=${encodeURIComponent(tenant())}&patient_id=${encodeURIComponent(pid)}`);
      if (vitals && vitals.vitals){
        document.getElementById("vHr").textContent = vitals.vitals.heart_rate ?? "-";
        document.getElementById("vSys").textContent = vitals.vitals.systolic_bp ?? "-";
        document.getElementById("vDia").textContent = vitals.vitals.diastolic_bp ?? "-";
        document.getElementById("vSpo2").textContent = vitals.vitals.spo2 ?? "-";
        document.getElementById("vTs").textContent = vitals.event_ts ?? "-";
      }

      const roll = await getJson(`/api/v1/patients/${encodeURIComponent(pid)}/rollups?tenant_id=${encodeURIComponent(tenant())}`);
      const rollBox = document.getElementById("rollupsBox");
      if (!roll || !roll.rollups){
        rollBox.innerHTML = '<div class="muted">No rollups yet.</div>';
      } else {
        const r = roll.rollups;
        rollBox.innerHTML = `
          <div class="kv">
            <div>Avg heart rate</div><div>${esc(r.avg_heart_rate ?? "-")}</div>
            <div>Avg systolic BP</div><div>${esc(r.avg_systolic_bp ?? "-")}</div>
            <div>Avg diastolic BP</div><div>${esc(r.avg_diastolic_bp ?? "-")}</div>
            <div>Avg SpO2</div><div>${esc(r.avg_spo2 ?? "-")}</div>
          </div>
        `;
      }

      const alerts = await getJson(`/api/v1/alerts?tenant_id=${encodeURIComponent(tenant())}&patient_id=${encodeURIComponent(pid)}&limit=10`);
      const badge = document.getElementById("riskBadge");
      if (alerts && alerts.alerts && alerts.alerts.length){
        const critical = alerts.alerts.some(a => (a.severity || "").toLowerCase() === "critical");
        const high = alerts.alerts.some(a => (a.severity || "").toLowerCase() === "high");
        const level = critical ? "high" : high ? "moderate" : "low";
        badge.className = "badge " + riskClass(level);
        badge.textContent = level;
        document.getElementById("riskSummary").textContent = alerts.alerts[0].message || "Recent alert found.";
      } else {
        badge.className = "badge risk-low";
        badge.textContent = "low";
        document.getElementById("riskSummary").textContent = "No elevated alert found for selected patient.";
      }
    }

    async function refreshSummary(alerts){
      const box = document.getElementById("summaryBox");
      if (!alerts || !alerts.length){
        box.innerHTML = '<div class="muted">No summary yet.</div>';
        return;
      }
      const counts = {};
      const patients = new Set();
      alerts.forEach(a => {
        const sev = (a.severity || "info").toLowerCase();
        counts[sev] = (counts[sev] || 0) + 1;
        patients.add(a.patient_id);
      });

      box.innerHTML =
        Object.keys(counts).sort().map(k =>
          `<div class="kv"><div>${esc(k)}</div><div>${esc(counts[k])}</div></div>`
        ).join("") +
        `<div class="kv" style="margin-top:10px"><div>Patients with alerts</div><div>${patients.size}</div></div>`;
    }

    async function refreshAll(){
      applyViewMode();
      await refreshOverview();
      await refreshStream();
      const alerts = await refreshAlerts();
      await refreshPatient();
      await refreshSummary(alerts);
    }

    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => {
        currentView = btn.dataset.view;
        refreshAll();
      });
    });

    refreshAll();
    setInterval(refreshAll, 10000);
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
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --panel2:#0d1526;
      --text:#ecf3ff;
      --muted:#92a6c8;
      --border:rgba(255,255,255,.08);
      --accent:#7aa2ff;
      --accent2:#5b84f7;
      --red:#ff6b6b;
      --amber:#f5c06a;
      --green:#5bd38d;
      --glow:0 12px 40px rgba(0,0,0,.28);
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.10), transparent 22%),
        radial-gradient(circle at top right, rgba(91,211,141,.08), transparent 20%),
        var(--bg);
      color:var(--text);
    }
    a{color:inherit}
    .investor-shell{max-width:1440px;margin:0 auto;padding:22px 18px 60px}
    .nav{
      position:sticky;top:0;z-index:60;
      display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
      padding:16px 0 18px;
      background:linear-gradient(180deg, rgba(8,17,31,.96), rgba(8,17,31,.88));
      backdrop-filter: blur(10px);
      border-bottom:1px solid rgba(255,255,255,.05);
      margin-bottom:26px
    }
    .logo-wrap{display:flex;flex-direction:column;gap:4px}
    .logo-kicker{
      font-size:11px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#d8e4ff
    }
    .logo{
      font-size:32px;font-weight:950;letter-spacing:-.03em
    }
    .nav-links{
      display:flex;gap:18px;align-items:center;flex-wrap:wrap
    }
    .nav-links a{
      text-decoration:none;color:#dfe9ff;font-weight:800;font-size:14px
    }
    .btn{
      background:linear-gradient(180deg,var(--accent),var(--accent2));
      color:#08111f;font-weight:900;border:none;cursor:pointer;
      padding:13px 18px;border-radius:12px;text-decoration:none;
      display:inline-flex;align-items:center;justify-content:center;
      box-shadow:0 8px 24px rgba(91,132,247,.28)
    }
    .btn.secondary{
      background:#151f34;color:var(--text);box-shadow:none;border:1px solid var(--border)
    }
    .btn.outline{
      background:transparent;color:#dfe9ff;border:1px solid var(--border);box-shadow:none
    }

    .hero-grid{
      display:grid;grid-template-columns:1.12fr .88fr;gap:22px;align-items:stretch
    }
    .hero-copy,.hero-side{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:28px;padding:30px;box-shadow:var(--glow)
    }
    .hero-kicker{
      color:#dce7ff;font-size:12px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;margin-bottom:16px
    }
    .hero-headline{
      font-size:80px;line-height:.94;font-weight:950;letter-spacing:-.065em;margin:0 0 18px
    }
    .hero-body{
      font-size:22px;line-height:1.45;color:#d3def2;max-width:900px
    }
    .hero-sub{
      margin-top:16px;font-size:17px;line-height:1.65;color:#a9bad8
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}

    .invest-highlight{
      font-size:34px;font-weight:950;letter-spacing:-.03em;margin:0 0 14px
    }
    .invest-side-text{
      color:#d5e1f5;font-size:18px;line-height:1.6;margin-bottom:18px
    }
    .highlight-grid{
      display:grid;grid-template-columns:1fr 1fr;gap:12px
    }
    .highlight-box{
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.02);
      border-radius:18px;padding:16px;min-height:116px
    }
    .hl-k{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#9cb1d6;font-weight:900}
    .hl-v{font-size:30px;font-weight:950;letter-spacing:-.04em;margin-top:10px}
    .hl-s{margin-top:6px;color:#b7c7e4;font-size:13px;font-weight:700;line-height:1.45}

    .micro-grid{
      display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px
    }
    .micro{
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.02);
      border-radius:16px;padding:15px
    }
    .micro .n{font-size:22px;font-weight:950}
    .micro .t{color:#bccce8;font-size:13px;font-weight:700;margin-top:6px;line-height:1.4}

    .section{margin-top:34px}
    .section-title{font-size:40px;font-weight:950;letter-spacing:-.04em;margin:0 0 18px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);border-radius:22px;padding:22px;box-shadow:var(--glow)
    }
    .card h3{margin:0 0 10px;font-size:24px;font-weight:900;letter-spacing:-.02em}
    .card p,.card li{color:#c9d7ef;font-size:16px;line-height:1.6}
    .card ul{margin:12px 0 0 18px;padding:0}
    .band{
      background:linear-gradient(180deg, rgba(122,162,255,.08), rgba(122,162,255,.03));
      border:1px solid rgba(122,162,255,.16);
      border-radius:26px;padding:28px;box-shadow:var(--glow)
    }
    .quote{
      font-size:28px;line-height:1.45;color:#eaf2ff;font-weight:800;letter-spacing:-.02em
    }
    .video-wrap{
      position:relative;width:100%;padding-top:56.25%;
      border-radius:22px;overflow:hidden;border:1px solid rgba(255,255,255,.08);box-shadow:var(--glow)
    }
    .video-wrap iframe{
      position:absolute;inset:0;width:100%;height:100%;border:0
    }

    .contact-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:16px
    }
    .contact-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);border-radius:20px;padding:20px;box-shadow:var(--glow)
    }
    .contact-label{
      font-size:11px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:#9fb3d8;margin-bottom:10px
    }
    .contact-value{
      font-size:28px;font-weight:900;letter-spacing:-.03em
    }
    .contact-value.small{
      font-size:20px;line-height:1.4;word-break:break-word
    }
    .contact-link{
      color:#9bc0ff;text-decoration:none;font-weight:900
    }

    .footer{
      margin-top:42px;padding-top:18px;border-top:1px solid rgba(255,255,255,.06);
      color:#88a0c8;font-size:14px;text-align:center
    }

    @media (max-width:1280px){
      .hero-headline{font-size:64px}
      .hero-grid{grid-template-columns:1fr}
      .grid-3{grid-template-columns:1fr 1fr}
      .contact-grid{grid-template-columns:1fr 1fr}
      .micro-grid{grid-template-columns:1fr 1fr}
    }
    @media (max-width:900px){
      .grid-3,.grid-2,.contact-grid,.highlight-grid,.micro-grid{grid-template-columns:1fr}
      .hero-headline{font-size:48px}
      .logo{font-size:26px}
      .investor-shell{padding-left:14px;padding-right:14px}
    }
  </style>
</head>
<body>
  <div class="investor-shell">

    <div class="nav">
      <div class="logo-wrap">
        <div class="logo-kicker">Investor Overview</div>
        <div class="logo">Early Risk Alert AI</div>
      </div>

      <div class="nav-links">
        <a href="#problem">Problem</a>
        <a href="#solution">Solution</a>
        <a href="#demo">Demo</a>
        <a href="#revenue">Revenue Model</a>
        <a href="#compliance">Compliance</a>
        <a href="#partnership">Partnership</a>
        <a href="#founder">Founder</a>
        <a href="/" class="btn secondary">Hospital Command Center</a>
        <a href="#contact" class="btn">Contact Founder</a>
      </div>
    </div>

    <div class="hero-grid">
      <div class="hero-copy">
        <div class="hero-kicker">AI-Powered Predictive Clinical Intelligence</div>
        <h1 class="hero-headline">Detect patient deterioration earlier. Strengthen hospital response at scale.</h1>
        <div class="hero-body">
          Early Risk Alert AI is a clinical intelligence platform designed to help hospitals,
          health systems, and remote monitoring programs identify elevated-risk patients in real time.
        </div>
        <div class="hero-sub">
          Our platform transforms fragmented monitoring into a unified command-center model—
          bringing together live patient visibility, AI-assisted risk prioritization, and
          enterprise-grade operational oversight.
        </div>
        <div class="hero-actions">
          <a class="btn" href="#demo">View Live Product Demo</a>
          <a class="btn secondary" href="/">Open Hospital Command Center</a>
          <a class="btn outline" href="/pitch-deck">Download Pitch Deck PDF</a>
        </div>
      </div>

      <div class="hero-side">
        <div class="invest-highlight">Investment Highlights</div>
        <div class="invest-side-text">
          Positioned at the intersection of hospital efficiency, predictive monitoring,
          and scalable healthcare software infrastructure.
        </div>

        <div class="highlight-grid">
          <div class="highlight-box">
            <div class="hl-k">Model</div>
            <div class="hl-v">Enterprise SaaS</div>
            <div class="hl-s">Recurring platform revenue with expansion across facilities and care networks.</div>
          </div>
          <div class="highlight-box">
            <div class="hl-k">Primary Buyer</div>
            <div class="hl-v">Hospitals</div>
            <div class="hl-s">Built for hospital systems, command centers, and remote monitoring teams.</div>
          </div>
          <div class="highlight-box">
            <div class="hl-k">Use Case</div>
            <div class="hl-v">Early Risk Detection</div>
            <div class="hl-s">Identify deterioration signals before emergencies escalate.</div>
          </div>
          <div class="highlight-box">
            <div class="hl-k">Expansion</div>
            <div class="hl-v">Multi-Site Rollout</div>
            <div class="hl-s">Designed for scale across departments, facilities, and enterprise networks.</div>
          </div>
        </div>

        <div class="micro-grid">
          <div class="micro"><div class="n">24/7</div><div class="t">Continuous monitoring framework</div></div>
          <div class="micro"><div class="n">AI</div><div class="t">Risk detection and prioritization</div></div>
          <div class="micro"><div class="n">RPM</div><div class="t">Remote monitoring integration path</div></div>
          <div class="micro"><div class="n">SaaS</div><div class="t">Recurring enterprise revenue model</div></div>
        </div>
      </div>
    </div>

    <div id="problem" class="section">
      <div class="section-title">The Problem</div>
      <div class="grid-3">
        <div class="card">
          <h3>Reactive Care Models</h3>
          <p>Many healthcare environments still depend on delayed intervention after patient risk has already escalated.</p>
        </div>
        <div class="card">
          <h3>Operational Fragmentation</h3>
          <p>Clinical teams often work across disconnected monitoring tools, scattered dashboards, and inconsistent alert workflows.</p>
        </div>
        <div class="card">
          <h3>Scaling Constraints</h3>
          <p>Hospitals face increasing patient loads, staffing shortages, and rising pressure to improve intervention speed.</p>
        </div>
      </div>
    </div>

    <div id="solution" class="section">
      <div class="section-title">The Solution</div>
      <div class="band">
        <div class="quote">
          Early Risk Alert AI centralizes live patient monitoring, AI-assisted risk detection,
          and command-center visibility into one premium clinical intelligence platform.
        </div>
      </div>

      <div class="grid-3" style="margin-top:16px">
        <div class="card">
          <h3>Live Monitoring Infrastructure</h3>
          <p>Real-time patient visibility supports earlier recognition of meaningful risk signals.</p>
        </div>
        <div class="card">
          <h3>AI-Driven Prioritization</h3>
          <p>Risk scoring and alert ranking help teams focus on high-priority clinical events first.</p>
        </div>
        <div class="card">
          <h3>Command Center Oversight</h3>
          <p>Operational dashboards provide a centralized view for enterprise monitoring and intervention support.</p>
        </div>
      </div>
    </div>

    <div id="demo" class="section">
      <div class="section-title">Live Product Demo</div>
      <div class="video-wrap">
        <iframe src="https://www.youtube.com/embed/HiidXiXifY4" title="Early Risk Alert AI Demo" allowfullscreen></iframe>
      </div>
    </div>

    <div id="revenue" class="section">
      <div class="section-title">Revenue Model</div>
      <div class="grid-3">
        <div class="card">
          <h3>Enterprise Platform Licensing</h3>
          <p>Hospitals and health systems subscribe to command-center access, monitoring dashboards, and platform infrastructure.</p>
        </div>
        <div class="card">
          <h3>Per-Patient Program Revenue</h3>
          <p>Remote monitoring and distributed care programs create scalable usage-based commercial opportunities.</p>
        </div>
        <div class="card">
          <h3>Integration & Deployment Services</h3>
          <p>Additional revenue can be generated through custom integrations, implementation support, and enterprise onboarding.</p>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Why Now</div>
      <div class="grid-2">
        <div class="card">
          <h3>Healthcare Systems Need Earlier Intervention</h3>
          <p>Rising preventable emergencies, clinician burnout, and demand for operational efficiency create strong urgency for predictive monitoring infrastructure.</p>
        </div>
        <div class="card">
          <h3>Digital Health Infrastructure Is Expanding</h3>
          <p>Hospitals are actively modernizing care delivery, remote monitoring, and enterprise software environments—creating a timely window for adoption.</p>
        </div>
      </div>
    </div>

    <div id="compliance" class="section">
      <div class="section-title">Compliance Positioning</div>
      <div class="grid-3">
        <div class="card">
          <h3>HIPAA-Ready Architecture</h3>
          <p>Designed with secure healthcare deployment principles suitable for regulated clinical environments.</p>
        </div>
        <div class="card">
          <h3>Enterprise Deployment Readiness</h3>
          <p>Built for cloud-based delivery, operational resilience, and future hospital system integration pathways.</p>
        </div>
        <div class="card">
          <h3>Trust & Credibility</h3>
          <p>Premium product presentation, live demos, and clear operating model improve institutional confidence during outreach.</p>
        </div>
      </div>
    </div>

    <div id="partnership" class="section">
      <div class="section-title">Hospital Partnership Opportunity</div>
      <div class="band">
        <div class="quote">
          Early Risk Alert AI is positioned for strategic partnerships with hospitals, care networks,
          pilot programs, and investors seeking exposure to modern healthcare intelligence infrastructure.
        </div>
      </div>

      <div class="grid-3" style="margin-top:16px">
        <div class="card">
          <h3>Pilot Programs</h3>
          <p>Opportunity to validate workflows, alert routing, and real-time monitoring value in live care environments.</p>
        </div>
        <div class="card">
          <h3>Strategic Investors</h3>
          <p>Supports product acceleration, infrastructure expansion, and stronger institutional go-to-market readiness.</p>
        </div>
        <div class="card">
          <h3>Health System Expansion</h3>
          <p>Built to grow from early pilots into broader system-wide monitoring deployments.</p>
        </div>
      </div>
    </div>

    <div id="founder" class="section">
      <div class="section-title">Founder</div>
      <div class="grid-2">
        <div class="card">
          <h3>Milton Munroe</h3>
          <p>
            Milton Munroe founded Early Risk Alert AI to advance predictive healthcare through intelligent monitoring infrastructure.
            The company vision centers on earlier detection, stronger operational visibility, and scalable clinical intelligence systems.
          </p>
          <p>
            Built from a mission to improve how elevated-risk patients are recognized and managed,
            the platform reflects both product ambition and founder-led execution.
          </p>
        </div>
        <div class="card">
          <h3>Founder Mission</h3>
          <p>
            Early Risk Alert AI represents a broader vision for proactive healthcare: identify patient risk sooner,
            support clinicians with better operational tools, and create more scalable systems for hospital response.
          </p>
          <p>
            This mission is reflected in product demonstrations, live system interfaces, investor materials, and hospital outreach readiness.
          </p>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="band">
        <div class="section-title" style="margin-top:0">Partner with Early Risk Alert AI</div>
        <div class="quote" style="font-size:22px">
          Early Risk Alert AI is building a premium predictive clinical intelligence platform for hospitals,
          health systems, and remote monitoring programs. The company is open to investor discussions,
          strategic partnerships, and product demonstration conversations.
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:22px">
          <a class="btn" href="#contact">Contact Founder</a>
          <a class="btn secondary" href="/">Open Hospital Command Center</a>
          <a class="btn outline" href="/pitch-deck">Watch Deck Download</a>
        </div>
      </div>
    </div>

    <div id="contact" class="section">
      <div class="section-title">Investor Contact</div>
      <div class="contact-grid">
        <div class="contact-card">
          <div class="contact-label">Founder</div>
          <div class="contact-value">Milton Munroe</div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Email</div>
          <div class="contact-value small">info@earlyriskalertai.com</div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Business Phone</div>
          <div class="contact-value">732-724-7267</div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Request Investor Deck</div>
          <div class="contact-value small">
            <a class="contact-link" href="mailto:info@earlyriskalertai.com?subject=Investor%20Inquiry%20-%20Early%20Risk%20Alert%20AI">info@earlyriskalertai.com</a>
          </div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Pitch Deck Download</div>
          <div class="contact-value small">
            <a class="contact-link" href="/pitch-deck">Download Pitch Deck PDF</a>
          </div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Live Demo Access</div>
          <div class="contact-value small">
            <a class="contact-link" href="/">Open Hospital Command Center</a>
          </div>
        </div>
      </div>
    </div>

    <div class="footer">
      Early Risk Alert AI LLC • Investor Overview • Milton Munroe • 732-724-7267
    </div>

  </div>
</body>
</html>
"""

@web_bp.get("/")
def home():
    return render_template_string(COMMAND_CENTER_HTML)

@web_bp.get("/login")
def login():
    return render_template("login.html")

@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(COMMAND_CENTER_HTML)

@web_bp.get("/investors")
def investors():
    return render_template_string(INVESTOR_HTML)

@web_bp.get("/pitch-deck")
def pitch_deck():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    static_dir = os.path.join(root_dir, "static")
    return send_from_directory(
        static_dir,
        "Early_Risk_Alert_AI_Pitch_Deck.pdf",
        as_attachment=True
    )
```
