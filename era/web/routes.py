from flask import Blueprint, render_template, render_template_string

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
      --bg:#07111f;
      --bg2:#0b1627;
      --panel:#0f1b2d;
      --panel2:#13233a;
      --text:#ecf3ff;
      --muted:#92a6c8;
      --border:rgba(255,255,255,.08);
      --accent:#6ea8ff;
      --red:#ff6b6b;
      --amber:#f7c95c;
      --green:#31d17c;
      --shadow:0 18px 60px rgba(0,0,0,.35);
      --glow:0 0 0 1px rgba(255,255,255,.03), 0 20px 60px rgba(0,0,0,.30);
    }

    *{box-sizing:border-box}
    html,body{margin:0;padding:0}
    body{
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at top left, rgba(110,168,255,.10), transparent 28%),
        radial-gradient(circle at top right, rgba(49,209,124,.06), transparent 22%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      color:var(--text);
      min-height:100vh;
    }

    .shell{
      max-width:1500px;
      margin:0 auto;
      padding:20px 18px 28px;
    }

    .hero{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:18px;
      flex-wrap:wrap;
      margin-bottom:14px;
    }

    .brand-wrap{display:flex;flex-direction:column;gap:10px}
    .eyebrow{
      color:#b9c8e8;
      font-size:12px;
      letter-spacing:.18em;
      text-transform:uppercase;
      font-weight:800;
    }
    .title{
      font-size:40px;
      line-height:1.02;
      font-weight:900;
      letter-spacing:-.03em;
      margin:0;
    }
    .subtitle{
      color:var(--muted);
      font-size:15px;
      max-width:760px;
      line-height:1.5;
      margin-top:4px;
    }

    .hero-actions{
      display:flex;
      gap:10px;
      align-items:center;
      flex-wrap:wrap;
      justify-content:flex-end;
    }

    .toolbar{
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      align-items:center;
      margin:10px 0 14px;
    }

    input, button{
      border-radius:12px;
      border:1px solid var(--border);
      background:rgba(19,35,58,.88);
      color:var(--text);
      padding:11px 13px;
      font-size:14px;
      outline:none;
    }

    input{min-width:130px}

    button{
      cursor:pointer;
      font-weight:800;
      background:linear-gradient(180deg, #78aefc 0%, #5f95f0 100%);
      color:#08111f;
      border:none;
    }

    button.secondary{
      background:rgba(255,255,255,.04);
      color:var(--text);
      border:1px solid var(--border);
    }

    button.ghost{
      background:transparent;
      border:1px solid var(--border);
      color:var(--text);
    }

    .live-pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:8px 14px;
      border-radius:999px;
      background:rgba(49,209,124,.10);
      border:1px solid rgba(49,209,124,.28);
      color:#d7ffe8;
      width:max-content;
      font-size:12px;
      font-weight:900;
      letter-spacing:.12em;
      text-transform:uppercase;
    }

    .live-dot{
      width:10px;
      height:10px;
      border-radius:999px;
      background:var(--green);
      box-shadow:0 0 14px rgba(49,209,124,.9);
      animation:pulse 1.4s infinite;
    }

    @keyframes pulse{
      0%{transform:scale(.92);opacity:.85}
      50%{transform:scale(1.1);opacity:1}
      100%{transform:scale(.92);opacity:.85}
    }

    .ticker{
      width:100%;
      overflow:hidden;
      border:1px solid var(--border);
      border-radius:14px;
      background:linear-gradient(90deg, rgba(255,107,107,.10), rgba(247,201,92,.08), rgba(110,168,255,.08));
      box-shadow:var(--glow);
      margin:10px 0 18px;
    }

    .ticker-track{
      white-space:nowrap;
      display:inline-block;
      min-width:100%;
      padding:11px 0;
      color:#ffd0d0;
      font-size:13px;
      font-weight:800;
      letter-spacing:.03em;
      animation:tickerMove 30s linear infinite;
    }

    @keyframes tickerMove{
      0%{transform:translateX(100%)}
      100%{transform:translateX(-100%)}
    }

    .kpis{
      display:grid;
      grid-template-columns:repeat(6, minmax(0,1fr));
      gap:14px;
      margin-bottom:16px;
    }

    .card{
      background:linear-gradient(180deg, rgba(20,34,56,.95) 0%, rgba(13,24,40,.96) 100%);
      border:1px solid var(--border);
      border-radius:20px;
      padding:16px;
      box-shadow:var(--shadow);
      position:relative;
      overflow:hidden;
    }

    .card:before{
      content:"";
      position:absolute;
      inset:0 auto auto 0;
      width:100%;
      height:1px;
      background:linear-gradient(90deg, rgba(255,255,255,.18), transparent);
      pointer-events:none;
    }

    .label{
      color:var(--muted);
      font-size:13px;
      font-weight:700;
    }

    .value{
      font-size:42px;
      font-weight:900;
      letter-spacing:-.03em;
      margin-top:6px;
      line-height:1;
    }

    .section{
      font-size:16px;
      font-weight:900;
      margin-bottom:12px;
      letter-spacing:.01em;
    }

    .layout{
      display:grid;
      grid-template-columns:1.15fr 1fr .95fr;
      gap:14px;
      align-items:start;
    }

    .list{
      display:flex;
      flex-direction:column;
      gap:10px;
      max-height:720px;
      overflow:auto;
    }

    .item{
      border:1px solid rgba(255,255,255,.06);
      background:rgba(255,255,255,.025);
      border-radius:16px;
      padding:12px;
      transition:transform .15s ease;
    }

    .item:hover{transform:translateY(-1px)}

    .row{
      display:flex;
      justify-content:space-between;
      gap:12px;
      align-items:flex-start;
    }

    .muted{color:var(--muted)}
    .small{font-size:12px}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

    .badge{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:84px;
      padding:5px 10px;
      border-radius:999px;
      font-size:12px;
      font-weight:900;
      text-transform:uppercase;
      letter-spacing:.05em;
      border:1px solid transparent;
    }

    .critical,.risk-high{
      background:rgba(255,107,107,.16);
      color:#ffc3c3;
      border-color:rgba(255,107,107,.26);
    }

    .high,.risk-moderate{
      background:rgba(247,201,92,.18);
      color:#ffe59a;
      border-color:rgba(247,201,92,.25);
    }

    .info,.low,.risk-low{
      background:rgba(110,168,255,.14);
      color:#bed6ff;
      border-color:rgba(110,168,255,.22);
    }

    .critical-glow{
      box-shadow:0 0 0 1px rgba(255,107,107,.18), 0 0 24px rgba(255,107,107,.10);
      animation:criticalPulse 1.8s infinite;
    }

    @keyframes criticalPulse{
      0%{box-shadow:0 0 0 1px rgba(255,107,107,.12), 0 0 12px rgba(255,107,107,.06)}
      50%{box-shadow:0 0 0 1px rgba(255,107,107,.25), 0 0 28px rgba(255,107,107,.16)}
      100%{box-shadow:0 0 0 1px rgba(255,107,107,.12), 0 0 12px rgba(255,107,107,.06)}
    }

    .kv{
      display:grid;
      grid-template-columns:1fr auto;
      gap:8px 12px;
      font-size:14px;
    }

    .hero-stat{
      display:flex;
      gap:8px;
      align-items:center;
      padding:10px 12px;
      border-radius:14px;
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
      color:var(--muted);
      font-size:13px;
      font-weight:700;
    }

    .mode-group{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
    }

    .mode-btn.active{
      background:linear-gradient(180deg, rgba(110,168,255,.22), rgba(110,168,255,.12));
      border:1px solid rgba(110,168,255,.32);
      color:#dbe9ff;
    }

    .mode-hospital .investor-only{display:none}
    .mode-investor .clinical-only{display:none}
    .mode-executive .detail-heavy{opacity:.9}

    .footer-note{
      color:#7f94b7;
      font-size:12px;
      text-align:right;
      margin-top:10px;
    }

    body.fullscreen{overflow:hidden}

    body.fullscreen .shell{
      max-width:none;
      padding:14px;
    }

    @media (max-width:1250px){
      .kpis{grid-template-columns:repeat(3, minmax(0,1fr))}
      .layout{grid-template-columns:1fr}
    }

    @media (max-width:760px){
      .kpis{grid-template-columns:repeat(2, minmax(0,1fr))}
      .title{font-size:30px}
      .value{font-size:34px}
      .hero{flex-direction:column}
      .hero-actions{justify-content:flex-start}
    }
  </style>
</head>
<body class="mode-hospital">
  <div class="shell">
    <div class="hero">
      <div class="brand-wrap">
        <div class="eyebrow">Predictive Clinical Intelligence</div>
        <h1 class="title">Early Risk Alert</h1>
        <div class="subtitle">Hospital UI + Investor Demo + Command Center</div>
        <div class="live-pill"><span class="live-dot"></span>Live System</div>
      </div>

      <div class="hero-actions">
        <div class="hero-stat">Tenant: <strong id="tenantHero">demo</strong></div>
        <div class="hero-stat">Focus Patient: <strong id="patientHero">p101</strong></div>
        <button id="fullscreenBtn" class="secondary" onclick="toggleFullscreen()">Fullscreen</button>
      </div>
    </div>

    <div class="toolbar">
      <input id="tenantId" value="demo" placeholder="tenant_id">
      <input id="patientId" value="p101" placeholder="patient_id">
      <button onclick="refreshAll()">Refresh</button>

      <div class="mode-group">
        <button class="ghost mode-btn active" id="modeHospital" onclick="setMode('hospital')">Hospital View</button>
        <button class="ghost mode-btn" id="modeExecutive" onclick="setMode('executive')">Executive View</button>
        <button class="ghost mode-btn" id="modeInvestor" onclick="setMode('investor')">Investor View</button>
      </div>
    </div>

    <div class="ticker">
      <div class="ticker-track" id="alertTicker">
        🚨 LIVE ALERTS — System monitoring patients in real time
      </div>
    </div>

    <div class="kpis">
      <div class="card"><div class="label">Patients</div><div class="value" id="kpiPatients">—</div></div>
      <div class="card"><div class="label">Open Alerts</div><div class="value" id="kpiOpenAlerts">—</div></div>
      <div class="card"><div class="label">Critical Alerts</div><div class="value" id="kpiCriticalAlerts">—</div></div>
      <div class="card"><div class="label">Events Last Hour</div><div class="value" id="kpiEventsHour">—</div></div>
      <div class="card"><div class="label">Stream Mode</div><div class="value" id="kpiStreamMode" style="font-size:28px">—</div></div>
      <div class="card investor-only"><div class="label">Commercial Model</div><div class="value" style="font-size:22px">SaaS</div></div>
    </div>

    <div class="layout">
      <div class="card">
        <div class="section">Live Alerts Feed</div>
        <div id="alertsList" class="list"></div>
      </div>

      <div class="card">
        <div class="section">Patient Focus</div>

        <div class="list">
          <div class="item detail-heavy">
            <div class="row">
              <div>
                <div class="muted">Patient</div>
                <div id="patientIdLabel" class="mono" style="font-size:28px;font-weight:900">—</div>
              </div>
              <div id="riskBadge" class="badge risk-low">unknown</div>
            </div>
            <div id="riskSummary" class="muted" style="margin-top:10px">No data yet.</div>
          </div>

          <div class="item clinical-only">
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

          <div class="item investor-only">
            <div class="muted">Clinical value proposition</div>
            <div style="margin-top:10px;line-height:1.6">
              Real-time risk detection, prioritized alerts, and centralized operational visibility
              for hospitals, care networks, and remote monitoring programs.
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

          <div class="item investor-only">
            <div class="muted">Commercial Snapshot</div>
            <div class="kv" style="margin-top:10px">
              <div>Revenue model</div><div>SaaS + RPM</div>
              <div>Primary buyers</div><div>Hospitals / Health Systems</div>
              <div>Deployment</div><div>Enterprise / Multi-site</div>
              <div>Expansion path</div><div>Network-wide rollout</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="footer-note">Early Risk Alert AI™ • Premium Live Command Center Demo</div>
  </div>

  <script>
    function tenant(){ return document.getElementById("tenantId").value || "demo"; }
    function patient(){ return document.getElementById("patientId").value || "p101"; }

    function esc(v){
      return (v ?? "").toString()
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

    function setMode(mode){
      document.body.classList.remove("mode-hospital","mode-executive","mode-investor");
      document.body.classList.add("mode-" + mode);

      document.getElementById("modeHospital").classList.remove("active");
      document.getElementById("modeExecutive").classList.remove("active");
      document.getElementById("modeInvestor").classList.remove("active");

      if (mode === "hospital") document.getElementById("modeHospital").classList.add("active");
      if (mode === "executive") document.getElementById("modeExecutive").classList.add("active");
      if (mode === "investor") document.getElementById("modeInvestor").classList.add("active");
    }

    async function toggleFullscreen(){
      try{
        if (!document.fullscreenElement){
          await document.documentElement.requestFullscreen();
          document.body.classList.add("fullscreen");
          document.getElementById("fullscreenBtn").innerText = "Exit Fullscreen";
        }else{
          await document.exitFullscreen();
          document.body.classList.remove("fullscreen");
          document.getElementById("fullscreenBtn").innerText = "Fullscreen";
        }
      }catch(e){}
    }

    document.addEventListener("fullscreenchange", () => {
      if (!document.fullscreenElement){
        document.body.classList.remove("fullscreen");
        document.getElementById("fullscreenBtn").innerText = "Fullscreen";
      }
    });

    async function refreshOverview(){
      const ov = await getJson(`/api/v1/dashboard/overview?tenant_id=${encodeURIComponent(tenant())}`);
      if (!ov) return null;

      document.getElementById("tenantHero").textContent = tenant();
      document.getElementById("patientHero").textContent = patient();
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

      const engineBox = document.getElementById("engineBox");
      engineBox.innerHTML = `
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
        <div class="item ${String(a.severity || "").toLowerCase().includes("critical") ? "critical-glow" : ""}">
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
        const critical = alerts.alerts.some(a => (a.severity || "").toLowerCase() == "critical");
        const high = alerts.alerts.some(a => (a.severity || "").toLowerCase() == "high");
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
      await refreshOverview();
      await refreshStream();
      const alerts = await refreshAlerts();
      await refreshPatient();
      await refreshSummary(alerts);
    }

    refreshAll();
    setInterval(refreshAll, 10000);
  </script>
</body>
</html>
"""

INVESTOR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Investor Overview — Early Risk Alert AI</title>

<style>
body{
    margin:0;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    background:#0b1220;
    color:#e6edf3;
}
.container{max-width:1200px;margin:auto;padding:40px 20px}
.nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px;flex-wrap:wrap;gap:14px}
.nav a{color:#9fb3c8;text-decoration:none;margin-left:20px;font-size:14px}
.logo{font-weight:700;font-size:18px;color:#fff}
.hero{display:grid;grid-template-columns:1.2fr 1fr;gap:40px;align-items:center;margin-bottom:60px}
.hero h1{font-size:48px;line-height:1.1;margin-bottom:20px}
.hero p{color:#9fb3c8;font-size:18px}
.btn{display:inline-block;padding:14px 22px;border-radius:10px;margin-top:20px;text-decoration:none;font-weight:600}
.btn-primary{background:#2563eb;color:white}
.btn-secondary{background:#1e293b;color:#cbd5e1}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;margin-top:40px}
.card{background:#111827;border-radius:16px;padding:22px}
.card h3{margin:0 0 10px 0}
.section{margin:80px 0}
.section h2{font-size:32px;margin-bottom:20px}
.footer{margin-top:80px;color:#64748b;font-size:14px;text-align:center}
.highlight{color:#60a5fa;font-weight:600}
.band{background:#0f172a;padding:50px;border-radius:20px;margin-top:40px}
.video{position:relative;padding-top:56.25%;border-radius:16px;overflow:hidden}
.video iframe{position:absolute;top:0;left:0;width:100%;height:100%}
.credibility{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-top:30px}
.cred-box{background:#111827;border-radius:14px;padding:16px;text-align:center;font-weight:600;color:#cbd5e1}
.contact-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;margin-top:30px}
.contact-card{background:#111827;border-radius:16px;padding:22px}
.small-label{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
@media(max-width:900px){
    .hero{grid-template-columns:1fr}
    .hero h1{font-size:38px}
}
</style>
</head>

<body>
<div class="container">

<div class="nav">
    <div class="logo">Early Risk Alert AI</div>
    <div>
        <a href="#model">Investment Model</a>
        <a href="#market">Market</a>
        <a href="#funds">Use of Funds</a>
        <a href="#partnership">Partnership</a>
        <a href="#founder">Founder</a>
        <a href="/" class="btn btn-primary">Live Hospital Demo</a>
    </div>
</div>

<div class="hero">
    <div>
        <h1>AI-Powered Early Risk Detection for Modern Healthcare Systems</h1>
        <p>
            Early Risk Alert AI delivers real-time predictive monitoring infrastructure
            that enables hospitals to detect patient deterioration earlier, improve
            intervention speed, and strengthen operational efficiency.
        </p>
        <a href="#demo" class="btn btn-primary">Watch Product Demo</a>
        <a href="#contact" class="btn btn-secondary">Request Investor Deck</a>
    </div>
    <div class="card">
        <h3>Investment Highlights</h3>
        <p><span class="highlight">Category:</span> Healthcare AI SaaS</p>
        <p><span class="highlight">Customers:</span> Hospitals & Health Systems</p>
        <p><span class="highlight">Revenue:</span> Recurring Enterprise Contracts</p>
        <p><span class="highlight">Expansion:</span> Multi-Facility Rollouts</p>
        <p><span class="highlight">Positioning:</span> Clinical Intelligence Infrastructure</p>
    </div>
</div>

<div class="credibility">
    <div class="cred-box">Enterprise SaaS</div>
    <div class="cred-box">Predictive Monitoring</div>
    <div class="cred-box">Hospital Command Center</div>
    <div class="cred-box">HIPAA-Ready Architecture</div>
    <div class="cred-box">Remote Monitoring Ready</div>
</div>

<div id="demo" class="section">
    <h2>Live Product Demonstration</h2>
    <div class="video">
        <iframe src="https://www.youtube.com/embed/HiidXiXifY4" allowfullscreen></iframe>
    </div>
</div>

<div id="model" class="section">
    <h2>Investment Model</h2>
    <div class="card-grid">
        <div class="card">
            <h3>Recurring SaaS Revenue</h3>
            <p>Enterprise hospital subscriptions create predictable recurring revenue streams.</p>
        </div>
        <div class="card">
            <h3>Per-Patient Programs</h3>
            <p>Remote monitoring contracts scale with patient volumes and care networks.</p>
        </div>
        <div class="card">
            <h3>Enterprise Expansion</h3>
            <p>Platform grows across hospital departments, facilities, and regional networks.</p>
        </div>
        <div class="card">
            <h3>Long-Term Value</h3>
            <p>High-value infrastructure positioning strengthens retention and platform dependency.</p>
        </div>
    </div>
</div>

<div id="market" class="section">
    <h2>Market Opportunity</h2>
    <div class="band">
        Hospitals face rising patient complexity, staffing shortages, and demand for earlier
        clinical intervention. Predictive monitoring platforms represent a major growth segment
        in digital health infrastructure and enterprise care modernization.
    </div>
</div>

<div id="funds" class="section">
    <h2>Use of Investment Capital</h2>
    <div class="card-grid">
        <div class="card"><h3>Platform Expansion</h3><p>Enhancing AI capabilities and infrastructure scalability.</p></div>
        <div class="card"><h3>Hospital Pilots</h3><p>Launching deployments and validating real-world clinical workflows.</p></div>
        <div class="card"><h3>Engineering Growth</h3><p>Expanding development team and accelerating product roadmap.</p></div>
        <div class="card"><h3>Enterprise Readiness</h3><p>Compliance positioning and large-scale deployment preparation.</p></div>
    </div>
</div>

<div id="partnership" class="section">
    <h2>Hospital Partnership Opportunity</h2>
    <div class="band">
        Early Risk Alert AI is actively seeking hospital partners, pilot sites, and strategic
        investors to support deployment, product validation, and enterprise expansion.
    </div>
</div>

<div id="founder" class="section">
    <h2>Founder</h2>
    <div class="card">
        <h3>Milton Munroe</h3>
        <p>
            Founder of Early Risk Alert AI, focused on advancing predictive healthcare technology
            that helps clinicians act earlier, reduce risk, and modernize patient monitoring.
        </p>
        <p><strong>Email:</strong> info@earlyriskalertai.com</p>
        <p><strong>Phone:</strong> 732-724-7267</p>
    </div>
</div>

<div class="section">
    <div class="band">
        <h2>Partner With Early Risk Alert AI</h2>
        <p>
            We are seeking strategic investors and healthcare partners to accelerate
            platform deployment, hospital integration, and enterprise expansion.
        </p>
        <a href="#contact" class="btn btn-primary">Contact Founder</a>
    </div>
</div>

<div id="contact" class="section">
    <h2>Investor Contact</h2>
    <div class="contact-grid">
        <div class="contact-card">
            <div class="small-label">Founder</div>
            <div><strong>Milton Munroe</strong></div>
        </div>
        <div class="contact-card">
            <div class="small-label">Email</div>
            <div><strong>info@earlyriskalertai.com</strong></div>
        </div>
        <div class="contact-card">
            <div class="small-label">Business Phone</div>
            <div><strong>732-724-7267</strong></div>
        </div>
        <div class="contact-card">
            <div class="small-label">Investor Materials</div>
            <div><a href="mailto:info@earlyriskalertai.com?subject=Request%20Investor%20Deck" style="color:#60a5fa;font-weight:700">Request Investor Deck</a></div>
        </div>
        <div class="contact-card">
            <div class="small-label">Pitch Deck</div>
            <div><a href="#" style="color:#60a5fa;font-weight:700">Download Pitch Deck PDF</a></div>
        </div>
        <div class="contact-card">
            <div class="small-label">Demo Access</div>
            <div><a href="/" style="color:#60a5fa;font-weight:700">Open Live Hospital Demo</a></div>
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
