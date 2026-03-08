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
      --panel3:#182b45;
      --text:#ecf3ff;
      --muted:#92a6c8;
      --border:rgba(255,255,255,.08);
      --border-strong:rgba(255,255,255,.16);
      --accent:#6ea8ff;
      --accent-2:#86efac;
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

    input, button, select{
      border-radius:12px;
      border:1px solid var(--border);
      background:rgba(19,35,58,.88);
      color:var(--text);
      padding:11px 13px;
      font-size:14px;
      outline:none;
      box-shadow:none;
    }

    input{
      min-width:130px;
    }

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

    body.fullscreen{
      overflow:hidden;
    }

    body.fullscreen .shell{
      max-width:none;
      padding:14px;
    }

    body.fullscreen .title{font-size:46px}
    body.fullscreen .value{font-size:52px}
    body.fullscreen .card{border-radius:22px}

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
        <div class="subtitle">
          Hospital UI + Investor Demo + Command Center
        </div>
        <div class="live-pill">
          <span class="live-dot"></span>
          Live System
        </div>
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
      <div class="card">
        <div class="label">Patients</div>
        <div class="value" id="kpiPatients">—</div>
      </div>
      <div class="card">
        <div class="label">Open Alerts</div>
        <div class="value" id="kpiOpenAlerts">—</div>
      </div>
      <div class="card">
        <div class="label">Critical Alerts</div>
        <div class="value" id="kpiCriticalAlerts">—</div>
      </div>
      <div class="card">
        <div class="label">Events Last Hour</div>
        <div class="value" id="kpiEventsHour">—</div>
      </div>
      <div class="card">
        <div class="label">Stream Mode</div>
        <div class="value" id="kpiStreamMode" style="font-size:28px">—</div>
      </div>
      <div class="card investor-only">
        <div class="label">Commercial Model</div>
        <div class="value" style="font-size:22px">SaaS</div>
      </div>
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

    <div class="footer-note">
      Early Risk Alert AI™ • Premium Live Command Center Demo
    </div>
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
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI | Investor Overview</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07111f;
      --bg2:#0d1728;
      --panel:#101b2d;
      --panel2:#14233a;
      --text:#edf4ff;
      --muted:#9db0cf;
      --line:rgba(255,255,255,.09);
      --accent:#7db2ff;
      --accent2:#86efac;
      --gold:#f4d38a;
      --shadow:0 20px 60px rgba(0,0,0,.35);
      --max:1220px;
    }

    *{box-sizing:border-box}
    html,body{margin:0;padding:0}
    body{
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(125,178,255,.14), transparent 28%),
        radial-gradient(circle at top right, rgba(134,239,172,.07), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      line-height:1.6;
    }

    a{color:inherit;text-decoration:none}

    .wrap{
      max-width:var(--max);
      margin:0 auto;
      padding:0 22px;
    }

    .nav{
      position:sticky;
      top:0;
      z-index:20;
      backdrop-filter:blur(10px);
      background:rgba(7,17,31,.72);
      border-bottom:1px solid rgba(255,255,255,.06);
    }

    .nav-inner{
      max-width:var(--max);
      margin:0 auto;
      padding:16px 22px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:16px;
      flex-wrap:wrap;
    }

    .brand{
      display:flex;
      flex-direction:column;
      gap:2px;
    }

    .brand small{
      color:var(--muted);
      text-transform:uppercase;
      letter-spacing:.16em;
      font-size:11px;
      font-weight:800;
    }

    .brand strong{
      font-size:20px;
      font-weight:900;
      letter-spacing:-.02em;
    }

    .nav-links{
      display:flex;
      gap:16px;
      flex-wrap:wrap;
      color:var(--muted);
      font-size:14px;
      font-weight:700;
    }

    .nav-cta{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
    }

    .btn{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:12px 18px;
      border-radius:999px;
      font-size:14px;
      font-weight:900;
      letter-spacing:.02em;
      border:1px solid transparent;
      transition:transform .15s ease, opacity .15s ease, border-color .15s ease;
    }

    .btn:hover{transform:translateY(-1px);opacity:.98}

    .btn-primary{
      background:linear-gradient(180deg, #88b7ff 0%, #6399f1 100%);
      color:#08111f;
      box-shadow:0 10px 30px rgba(99,153,241,.25);
    }

    .btn-secondary{
      background:rgba(255,255,255,.04);
      color:var(--text);
      border-color:var(--line);
    }

    .hero{
      padding:76px 0 52px;
    }

    .hero-grid{
      display:grid;
      grid-template-columns:1.15fr .85fr;
      gap:26px;
      align-items:center;
    }

    .eyebrow{
      color:#cddaf2;
      text-transform:uppercase;
      letter-spacing:.18em;
      font-size:12px;
      font-weight:900;
      margin-bottom:12px;
    }

    h1{
      margin:0;
      font-size:64px;
      line-height:.96;
      letter-spacing:-.05em;
    }

    .hero-sub{
      margin-top:18px;
      font-size:20px;
      color:#d7e4fb;
      max-width:760px;
    }

    .hero-copy{
      margin-top:18px;
      color:var(--muted);
      font-size:16px;
      max-width:760px;
    }

    .hero-actions{
      margin-top:28px;
      display:flex;
      gap:12px;
      flex-wrap:wrap;
    }

    .hero-card{
      background:linear-gradient(180deg, rgba(20,35,58,.88), rgba(13,24,40,.95));
      border:1px solid var(--line);
      border-radius:28px;
      padding:22px;
      box-shadow:var(--shadow);
      position:relative;
      overflow:hidden;
    }

    .hero-card:before{
      content:"";
      position:absolute;
      inset:0;
      background:linear-gradient(135deg, rgba(255,255,255,.06), transparent 45%);
      pointer-events:none;
    }

    .hero-card h3{
      margin:0 0 10px 0;
      font-size:18px;
      letter-spacing:.01em;
    }

    .mini-grid{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:12px;
      margin-top:16px;
    }

    .mini{
      border:1px solid rgba(255,255,255,.07);
      background:rgba(255,255,255,.03);
      border-radius:18px;
      padding:14px;
    }

    .mini .k{
      color:var(--muted);
      font-size:12px;
      font-weight:800;
      text-transform:uppercase;
      letter-spacing:.08em;
    }

    .mini .v{
      margin-top:6px;
      font-size:24px;
      font-weight:900;
      letter-spacing:-.03em;
    }

    .section{
      padding:28px 0 34px;
    }

    .section-title{
      font-size:38px;
      line-height:1.04;
      letter-spacing:-.04em;
      margin:0 0 12px 0;
    }

    .section-sub{
      color:var(--muted);
      font-size:17px;
      max-width:860px;
      margin-bottom:22px;
    }

    .grid-2{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:18px;
    }

    .grid-3{
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:18px;
    }

    .panel{
      background:linear-gradient(180deg, rgba(19,34,56,.88), rgba(14,24,40,.96));
      border:1px solid var(--line);
      border-radius:24px;
      padding:22px;
      box-shadow:var(--shadow);
    }

    .panel h3{
      margin:0 0 10px 0;
      font-size:22px;
      letter-spacing:-.02em;
    }

    .panel p{
      margin:0;
      color:var(--muted);
      font-size:15px;
    }

    .bullet-list{
      margin:12px 0 0 0;
      padding-left:18px;
      color:var(--text);
    }

    .bullet-list li{
      margin-bottom:8px;
    }

    .stat-band{
      margin:12px 0 0;
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:14px;
    }

    .stat{
      background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      padding:16px;
    }

    .stat .num{
      font-size:28px;
      font-weight:900;
      letter-spacing:-.03em;
    }

    .stat .txt{
      color:var(--muted);
      font-size:13px;
      margin-top:6px;
      font-weight:700;
    }

    .demo-wrap{
      display:grid;
      grid-template-columns:1.05fr .95fr;
      gap:18px;
      align-items:stretch;
    }

    .video-frame{
      position:relative;
      width:100%;
      padding-top:56.25%;
      border-radius:24px;
      overflow:hidden;
      border:1px solid var(--line);
      box-shadow:var(--shadow);
      background:#000;
    }

    .video-frame iframe{
      position:absolute;
      inset:0;
      width:100%;
      height:100%;
      border:0;
    }

    .callout{
      display:flex;
      gap:12px;
      align-items:flex-start;
      padding:14px 0;
      border-top:1px solid rgba(255,255,255,.06);
    }

    .callout:first-child{border-top:none;padding-top:0}

    .icon{
      width:36px;
      height:36px;
      border-radius:12px;
      display:flex;
      align-items:center;
      justify-content:center;
      background:rgba(125,178,255,.14);
      border:1px solid rgba(125,178,255,.22);
      flex:0 0 36px;
      font-weight:900;
    }

    .highlight{
      color:var(--gold);
      font-weight:800;
    }

    .quote{
      font-size:20px;
      line-height:1.5;
      color:#dbe8ff;
      margin:0;
    }

    .founder{
      display:grid;
      grid-template-columns:.8fr 1.2fr;
      gap:18px;
    }

    .founder-card{
      background:linear-gradient(180deg, rgba(19,34,56,.88), rgba(14,24,40,.96));
      border:1px solid var(--line);
      border-radius:26px;
      padding:24px;
      box-shadow:var(--shadow);
    }

    .founder-name{
      font-size:30px;
      font-weight:900;
      letter-spacing:-.03em;
      margin:0 0 6px 0;
    }

    .founder-role{
      color:var(--gold);
      font-size:14px;
      font-weight:800;
      text-transform:uppercase;
      letter-spacing:.12em;
      margin-bottom:18px;
    }

    .contact-block{
      margin-top:18px;
      color:var(--muted);
      font-size:15px;
    }

    .cta-band{
      margin:20px 0 70px;
      padding:28px;
      border-radius:28px;
      background:
        radial-gradient(circle at left top, rgba(125,178,255,.16), transparent 32%),
        linear-gradient(180deg, rgba(17,30,49,.96), rgba(13,23,40,.96));
      border:1px solid var(--line);
      box-shadow:var(--shadow);
    }

    .cta-band h2{
      margin:0 0 10px 0;
      font-size:38px;
      line-height:1.04;
      letter-spacing:-.04em;
    }

    .cta-band p{
      margin:0;
      color:var(--muted);
      max-width:820px;
      font-size:17px;
    }

    .cta-actions{
      margin-top:22px;
      display:flex;
      gap:12px;
      flex-wrap:wrap;
    }

    .footer{
      border-top:1px solid rgba(255,255,255,.06);
      padding:24px 0 40px;
      color:var(--muted);
      font-size:14px;
    }

    @media (max-width:1024px){
      .hero-grid,.demo-wrap,.founder,.grid-2,.grid-3{
        grid-template-columns:1fr;
      }
      .stat-band{
        grid-template-columns:repeat(2,minmax(0,1fr));
      }
      h1{font-size:50px}
    }

    @media (max-width:640px){
      h1{font-size:40px}
      .section-title,.cta-band h2{font-size:30px}
      .stat-band{
        grid-template-columns:1fr;
      }
      .nav-inner{padding:14px 16px}
      .wrap{padding:0 16px}
      .hero{padding:56px 0 34px}
    }
  </style>
</head>
<body>
  <div class="nav">
    <div class="nav-inner">
      <div class="brand">
        <small>Investor Overview</small>
        <strong>Early Risk Alert AI</strong>
      </div>

      <div class="nav-links">
        <a href="#problem">Problem</a>
        <a href="#solution">Solution</a>
        <a href="#demo">Demo</a>
        <a href="#model">Revenue Model</a>
        <a href="#compliance">Compliance</a>
        <a href="#founder">Founder</a>
      </div>

      <div class="nav-cta">
        <a class="btn btn-secondary" href="/">Hospital Command Center</a>
        <a class="btn btn-primary" href="mailto:info@earlyriskalertai.com?subject=Investor%20Inquiry%20-%20Early%20Risk%20Alert%20AI">Contact Founder</a>
      </div>
    </div>
  </div>

  <section class="hero">
    <div class="wrap">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">AI-Powered Predictive Clinical Intelligence</div>
          <h1>Detect patient deterioration earlier. Strengthen hospital response at scale.</h1>
          <div class="hero-sub">
            Early Risk Alert AI is a clinical intelligence platform designed to help hospitals, health systems, and remote monitoring programs identify elevated-risk patients in real time.
          </div>
          <div class="hero-copy">
            Our platform transforms fragmented monitoring into a unified command-center model—bringing together live patient visibility, AI-assisted risk prioritization, and enterprise-grade operational oversight.
          </div>

          <div class="hero-actions">
            <a class="btn btn-primary" href="#demo">View Live Product Demo</a>
            <a class="btn btn-secondary" href="/">Open Hospital Command Center</a>
          </div>
        </div>

        <div class="hero-card">
          <h3>Investment Highlights</h3>
          <p>
            Positioned at the intersection of hospital efficiency, predictive monitoring, and scalable healthcare software infrastructure.
          </p>

          <div class="mini-grid">
            <div class="mini">
              <div class="k">Model</div>
              <div class="v">Enterprise SaaS</div>
            </div>
            <div class="mini">
              <div class="k">Primary Buyer</div>
              <div class="v">Hospitals</div>
            </div>
            <div class="mini">
              <div class="k">Use Case</div>
              <div class="v">Early Risk Detection</div>
            </div>
            <div class="mini">
              <div class="k">Expansion</div>
              <div class="v">Multi-Site Rollout</div>
            </div>
          </div>

          <div class="stat-band">
            <div class="stat">
              <div class="num">24/7</div>
              <div class="txt">Continuous monitoring framework</div>
            </div>
            <div class="stat">
              <div class="num">AI</div>
              <div class="txt">Risk detection and prioritization</div>
            </div>
            <div class="stat">
              <div class="num">RPM</div>
              <div class="txt">Remote monitoring integration path</div>
            </div>
            <div class="stat">
              <div class="num">SaaS</div>
              <div class="txt">Recurring enterprise revenue model</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="problem">
    <div class="wrap">
      <h2 class="section-title">The problem: healthcare systems are still forced to react too late.</h2>
      <div class="section-sub">
        Hospitals face growing pressure to manage rising patient complexity, staffing shortages, and preventable escalations with systems that are often fragmented, manual, and difficult to scale.
      </div>

      <div class="grid-3">
        <div class="panel">
          <h3>Reactive Monitoring</h3>
          <p>
            Traditional workflows often identify deterioration after patient risk has already escalated, limiting the time available for early intervention.
          </p>
        </div>
        <div class="panel">
          <h3>Operational Strain</h3>
          <p>
            Care teams are expected to monitor more patients with fewer resources, creating delays, alert fatigue, and inconsistent prioritization.
          </p>
        </div>
        <div class="panel">
          <h3>Missed Visibility</h3>
          <p>
            Vital trends, warning signs, and risk signals are too often buried across disconnected systems rather than surfaced in a unified real-time view.
          </p>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="solution">
    <div class="wrap">
      <h2 class="section-title">The solution: a centralized intelligence layer for modern clinical operations.</h2>
      <div class="section-sub">
        Early Risk Alert AI provides a smarter operating model for care delivery by converting real-time health signals into prioritized clinical intelligence.
      </div>

      <div class="grid-2">
        <div class="panel">
          <h3>Core Platform Capabilities</h3>
          <ul class="bullet-list">
            <li>Continuous patient monitoring visibility across care environments</li>
            <li>AI-assisted early warning detection for elevated-risk cases</li>
            <li>Live alert prioritization by severity and operational importance</li>
            <li>Command-center style oversight for hospitals and care networks</li>
            <li>Scalable architecture designed for enterprise deployment</li>
          </ul>
        </div>

        <div class="panel">
          <h3>Clinical and Operational Value</h3>
          <ul class="bullet-list">
            <li>Earlier intervention opportunities before emergencies escalate</li>
            <li>More efficient allocation of clinician attention and response capacity</li>
            <li>Improved visibility across departments, facilities, and remote programs</li>
            <li>Stronger infrastructure for proactive, data-informed care delivery</li>
            <li>Premium software positioning for hospital modernization initiatives</li>
          </ul>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="demo">
    <div class="wrap">
      <h2 class="section-title">Live product demo</h2>
      <div class="section-sub">
        A short live demonstration showing Early Risk Alert AI in action through a hospital command-center interface, including patient monitoring, alert prioritization, and centralized clinical visibility.
      </div>

      <div class="demo-wrap">
        <div class="video-frame">
          <iframe src="https://www.youtube.com/embed/HiidXiXifY4" title="Early Risk Alert AI Live Demo" allowfullscreen></iframe>
        </div>

        <div class="panel">
          <h3>What the demo shows</h3>

          <div class="callout">
            <div class="icon">1</div>
            <div>
              <strong>Real-time patient monitoring</strong>
              <div class="muted">Live overview of patient status and critical metrics.</div>
            </div>
          </div>

          <div class="callout">
            <div class="icon">2</div>
            <div>
              <strong>AI-prioritized alerts</strong>
              <div class="muted">Risk signals surfaced clearly by severity and urgency.</div>
            </div>
          </div>

          <div class="callout">
            <div class="icon">3</div>
            <div>
              <strong>Command-center visibility</strong>
              <div class="muted">A unified operational interface for modern hospital workflows.</div>
            </div>
          </div>

          <div class="callout">
            <div class="icon">4</div>
            <div>
              <strong>Enterprise presentation quality</strong>
              <div class="muted">Built to communicate credibility with hospitals, partners, and investors.</div>
            </div>
          </div>

          <div style="margin-top:18px">
            <a class="btn btn-primary" href="/">Open Command Center</a>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="market">
    <div class="wrap">
      <h2 class="section-title">Why now</h2>
      <div class="section-sub">
        Hospitals and health systems are under increasing pressure to improve outcomes, manage labor constraints, and modernize digital care infrastructure. That shift creates strong demand for predictive intelligence platforms that can support earlier intervention and scalable operational oversight.
      </div>

      <div class="grid-3">
        <div class="panel">
          <h3>Staffing and Burnout Pressure</h3>
          <p>
            Health systems need better tools to help limited care teams identify the highest-priority patients faster and with greater clarity.
          </p>
        </div>
        <div class="panel">
          <h3>Growth of Remote Monitoring</h3>
          <p>
            As distributed care models expand, providers need software that can unify patient oversight beyond facility walls.
          </p>
        </div>
        <div class="panel">
          <h3>Value-Based Care Momentum</h3>
          <p>
            Early intervention, reduced escalations, and stronger operational efficiency increasingly align with how modern healthcare systems are measured and rewarded.
          </p>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="opportunity">
    <div class="wrap">
      <h2 class="section-title">Market opportunity</h2>
      <div class="section-sub">
        Early Risk Alert AI is positioned for deployment across multiple institutional healthcare segments where patient visibility, early risk detection, and enterprise coordination matter most.
      </div>

      <div class="grid-2">
        <div class="panel">
          <h3>Primary Market Segments</h3>
          <ul class="bullet-list">
            <li>Hospitals and integrated health systems</li>
            <li>Remote patient monitoring providers</li>
            <li>Care management networks</li>
            <li>Clinical command centers and centralized operations teams</li>
            <li>Enterprise digital health and care coordination programs</li>
          </ul>
        </div>

        <div class="panel">
          <h3>Expansion Logic</h3>
          <ul class="bullet-list">
            <li>Initial hospital and provider deployments</li>
            <li>Department-level adoption expanding to multi-site systems</li>
            <li>Remote monitoring and care network integrations</li>
            <li>Analytics and enterprise workflow extensions</li>
            <li>Long-term platform positioning inside larger health infrastructures</li>
          </ul>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="model">
    <div class="wrap">
      <h2 class="section-title">Revenue model</h2>
      <div class="section-sub">
        Early Risk Alert AI is structured as a scalable enterprise software business designed for recurring revenue and network-level growth.
      </div>

      <div class="grid-3">
        <div class="panel">
          <h3>Enterprise Platform Subscriptions</h3>
          <p>
            Recurring software subscriptions for hospitals, care networks, and monitoring organizations adopting the platform operationally.
          </p>
        </div>
        <div class="panel">
          <h3>Per-Patient Monitoring Programs</h3>
          <p>
            Usage-aligned pricing for remote monitoring and distributed care initiatives where scale increases recurring platform value.
          </p>
        </div>
        <div class="panel">
          <h3>Analytics and Integration Packages</h3>
          <p>
            Additional revenue through premium analytics, deployment services, enterprise configuration, and system integrations.
          </p>
        </div>
      </div>

      <div class="panel" style="margin-top:18px">
        <h3>Commercial positioning</h3>
        <p>
          The business model is designed to support land-and-expand growth: initial adoption within focused hospital programs, followed by broader deployment across departments, facilities, and enterprise care environments.
        </p>
      </div>
    </div>
  </section>

  <section class="section" id="compliance">
    <div class="wrap">
      <h2 class="section-title">Compliance and enterprise readiness</h2>
      <div class="section-sub">
        Early Risk Alert AI is being positioned with the language, architecture, and operational framing expected of enterprise healthcare technology.
      </div>

      <div class="grid-2">
        <div class="panel">
          <h3>Platform Readiness</h3>
          <ul class="bullet-list">
            <li>HIPAA-ready architectural positioning</li>
            <li>Secure cloud deployment model</li>
            <li>Audit-friendly workflow structure</li>
            <li>Scalable multi-site software framework</li>
            <li>Command-center interface aligned with clinical operations</li>
          </ul>
        </div>

        <div class="panel">
          <h3>Enterprise Confidence</h3>
          <p class="quote">
            “Built to support hospitals, providers, and care networks seeking a premium, modern platform for proactive patient monitoring and early risk detection.”
          </p>
          <div style="margin-top:18px;color:var(--muted)">
            The platform is presented as infrastructure for clinical modernization, not simply a dashboard. That distinction strengthens both enterprise and investor credibility.
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="section" id="founder">
    <div class="wrap">
      <h2 class="section-title">About the founder</h2>
      <div class="section-sub">
        Early Risk Alert AI was founded to advance predictive healthcare through intelligent monitoring systems that help care teams act earlier, prioritize faster, and operate with greater confidence.
      </div>

      <div class="founder">
        <div class="founder-card">
          <div class="founder-name">Milton MUNROE</div>
          <div class="founder-role">Founder, Early Risk Alert AI</div>

          <div>
            Building AI-powered preventive health technology focused on earlier detection, smarter monitoring, and better patient outcomes.
          </div>

          <div class="contact-block">
            <div><strong>Email:</strong> info@earlyriskalertai.com</div>
            <div><strong>Phone:</strong> 732-724-7267</div>
          </div>
        </div>

        <div class="founder-card">
          <h3>Founder mission</h3>
          <p>
            Milton MUNROE founded Early Risk Alert AI to help move healthcare from reactive response toward proactive, intelligence-driven prevention. The platform is built around a clear vision: identify patient deterioration sooner, strengthen clinical decision support, and create more scalable systems for modern care delivery.
          </p>
          <p style="margin-top:14px">
            The company’s positioning combines technical credibility, hospital relevance, and premium software presentation—creating a foundation for both institutional adoption and investor interest.
          </p>
        </div>
      </div>
    </div>
  </section>

  <section class="wrap">
    <div class="cta-band">
      <h2>Partner with Early Risk Alert AI</h2>
      <p>
        For investors, strategic partners, and healthcare organizations interested in product demonstrations, partnership discussions, or founder conversations, Early Risk Alert AI is available for direct outreach.
      </p>

      <div class="cta-actions">
        <a class="btn btn-primary" href="mailto:info@earlyriskalertai.com?subject=Investor%20Inquiry%20-%20Early%20Risk%20Alert%20AI">Contact Founder</a>
        <a class="btn btn-secondary" href="/">Open Hospital Command Center</a>
        <a class="btn btn-secondary" href="https://youtu.be/HiidXiXifY4" target="_blank">Watch Demo Video</a>
      </div>
    </div>
  </section>

  <div class="footer">
    <div class="wrap">
      Early Risk Alert AI LLC • Investor Overview • Milton MUNROE • 732-724-7267
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
