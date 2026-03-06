from flask import Blueprint, render_template_string

web_bp = Blueprint("web", __name__)

HTML = """
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
      --accent:#60a5fa;
      --red:#ef4444;
      --amber:#f59e0b;
      --green:#22c55e;
      --purple:#8b5cf6;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:var(--bg);
      color:var(--text);
    }
    .shell{max-width:1500px;margin:20px auto;padding:0 16px}
    .topbar{
      display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;
      margin-bottom:16px;
    }
    .title{font-size:30px;font-weight:800;line-height:1.1}
    .subtitle{margin-top:6px;color:var(--muted);font-size:14px}
    .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    input,button,select{
      border:1px solid var(--border);
      background:var(--panel2);
      color:var(--text);
      padding:10px 12px;
      border-radius:10px;
      font-size:14px;
    }
    button{
      background:var(--accent);
      color:#07111f;
      font-weight:800;
      border:none;
      cursor:pointer;
    }
    .modes{
      display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;
    }
    .mode-btn{
      background:var(--panel);
      color:var(--text);
      border:1px solid var(--border);
    }
    .mode-btn.active{
      background:var(--accent);
      color:#07111f;
    }
    .kpis{
      display:grid;
      grid-template-columns:repeat(5,1fr);
      gap:14px;
      margin-bottom:14px;
    }
    .card{
      background:var(--panel);
      border:1px solid var(--border);
      border-radius:18px;
      padding:16px;
      box-shadow:0 8px 24px rgba(0,0,0,.18);
    }
    .label{color:var(--muted);font-size:13px}
    .value{font-size:30px;font-weight:800;margin-top:8px}
    .section{font-size:16px;font-weight:800;margin-bottom:12px}
    .layout{
      display:grid;
      grid-template-columns:1.1fr 1fr .9fr;
      gap:14px;
    }
    .list{
      display:flex;
      flex-direction:column;
      gap:10px;
      max-height:620px;
      overflow:auto;
    }
    .item{
      border:1px solid var(--border);
      background:rgba(255,255,255,.02);
      border-radius:14px;
      padding:12px;
    }
    .row{
      display:flex;justify-content:space-between;gap:10px;align-items:flex-start;
    }
    .muted{color:var(--muted);font-size:13px}
    .small{font-size:12px}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .badge{
      display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:800;
    }
    .critical,.risk-high{background:rgba(239,68,68,.18);color:#fecaca}
    .high,.risk-moderate{background:rgba(245,158,11,.18);color:#fde68a}
    .info,.low,.risk-low{background:rgba(34,197,94,.18);color:#bbf7d0}
    .kv{
      display:grid;grid-template-columns:1fr auto;gap:8px;font-size:14px;
    }
    .hidden{display:none}
    .grid-wall{
      display:grid;
      grid-template-columns:repeat(4,1fr);
      gap:12px;
    }
    .patient-tile{
      background:var(--panel);
      border:1px solid var(--border);
      border-radius:18px;
      padding:14px;
      min-height:150px;
    }
    .tile-top{
      display:flex;justify-content:space-between;align-items:flex-start;gap:8px;
    }
    .spark{
      margin-top:10px;
      height:8px;
      border-radius:999px;
      background:linear-gradient(90deg,var(--green),var(--amber),var(--red));
      opacity:.9;
    }
    .investor-grid{
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:14px;
    }
    .hero{
      background:linear-gradient(135deg, rgba(96,165,250,.15), rgba(139,92,246,.12));
      border:1px solid var(--border);
      border-radius:18px;
      padding:18px;
      margin-bottom:14px;
    }
    .hero-title{font-size:24px;font-weight:800}
    .hero-sub{margin-top:8px;color:var(--muted)}
    .full{grid-column:1/-1}
    @media (max-width:1200px){
      .kpis{grid-template-columns:repeat(2,1fr)}
      .layout{grid-template-columns:1fr}
      .investor-grid{grid-template-columns:1fr}
      .grid-wall{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:700px){
      .kpis{grid-template-columns:1fr}
      .grid-wall{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div>
      <div class="title">Early Risk Alert</div>
      <div class="subtitle">Hospital UI + Investor Demo + Million-Patient Command Wall</div>
    </div>
    <div class="toolbar">
      <input id="tenantId" value="demo" placeholder="tenant_id">
      <input id="patientId" value="p101" placeholder="patient_id">
      <button onclick="refreshAll()">Refresh</button>
    </div>
  </div>

  <div class="modes">
    <button id="mode-hospital" class="mode-btn active" onclick="setMode('hospital')">Hospital UI</button>
    <button id="mode-investor" class="mode-btn" onclick="setMode('investor')">Investor Demo</button>
    <button id="mode-wall" class="mode-btn" onclick="setMode('wall')">Command Wall</button>
  </div>

  <div class="kpis">
    <div class="card"><div class="label">Patients</div><div class="value" id="kpiPatients">-</div></div>
    <div class="card"><div class="label">Open Alerts</div><div class="value" id="kpiOpenAlerts">-</div></div>
    <div class="card"><div class="label">Critical Alerts</div><div class="value" id="kpiCriticalAlerts">-</div></div>
    <div class="card"><div class="label">Events Last Hour</div><div class="value" id="kpiEventsHour">-</div></div>
    <div class="card"><div class="label">Stream Mode</div><div class="value" id="kpiStreamMode" style="font-size:20px">-</div></div>
  </div>

  <div id="hospital-view">
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
                <div id="patientIdLabel" class="mono">-</div>
              </div>
              <div id="riskBadge" class="badge risk-low">unknown</div>
            </div>
            <div id="riskSummary" class="muted" style="margin-top:10px">No data yet.</div>
          </div>

          <div class="item">
            <div class="muted">Latest vitals</div>
            <div class="kv" style="margin-top:10px">
              <div>Heart rate</div><div id="vHr">-</div>
              <div>Systolic BP</div><div id="vSys">-</div>
              <div>Diastolic BP</div><div id="vDia">-</div>
              <div>SpO2</div><div id="vSpo2">-</div>
              <div>Event time</div><div id="vTs" class="small">-</div>
            </div>
          </div>

          <div class="item">
            <div class="muted">Patient rollups</div>
            <div id="rollupsBox" style="margin-top:10px"></div>
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
        </div>
      </div>
    </div>
  </div>

  <div id="investor-view" class="hidden">
    <div class="hero">
      <div class="hero-title">Remote Patient Monitoring AI Platform</div>
      <div class="hero-sub">Live cloud infrastructure for vitals ingestion, anomaly detection, patient scoring, and command-center operations.</div>
    </div>

    <div class="investor-grid">
      <div class="card">
        <div class="section">Business Value</div>
        <div class="list">
          <div class="item">Early deterioration detection</div>
          <div class="item">Command-center visibility across tenants</div>
          <div class="item">Redis and worker architecture ready for streaming</div>
          <div class="item">Scalable to high patient volume</div>
        </div>
      </div>

      <div class="card">
        <div class="section">Current KPI Snapshot</div>
        <div id="investorKpis" class="list"></div>
      </div>

      <div class="card">
        <div class="section">Risk Distribution</div>
        <div id="investorRiskMix" class="list"></div>
      </div>

      <div class="card full">
        <div class="section">Platform Readiness</div>
        <div id="scaleBox" class="list"></div>
      </div>
    </div>
  </div>

  <div id="wall-view" class="hidden">
    <div class="card">
      <div class="section">Million-Patient Command Wall</div>
      <div id="wallGrid" class="grid-wall"></div>
    </div>
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

let currentMode = "hospital";

function setMode(mode){
  currentMode = mode;
  document.getElementById("hospital-view").classList.toggle("hidden", mode !== "hospital");
  document.getElementById("investor-view").classList.toggle("hidden", mode !== "investor");
  document.getElementById("wall-view").classList.toggle("hidden", mode !== "wall");

  document.getElementById("mode-hospital").classList.toggle("active", mode === "hospital");
  document.getElementById("mode-investor").classList.toggle("active", mode === "investor");
  document.getElementById("mode-wall").classList.toggle("active", mode === "wall");
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

async function refreshOverview(){
  const ov = await getJson(`/api/v1/dashboard/overview?tenant_id=${encodeURIComponent(tenant())}`);
  if (!ov){
    document.getElementById("kpiPatients").textContent = "-";
    document.getElementById("kpiOpenAlerts").textContent = "-";
    document.getElementById("kpiCriticalAlerts").textContent = "-";
    document.getElementById("kpiEventsHour").textContent = "-";
    return null;
  }
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
  document.getElementById("channelsBox").innerHTML = channels && channels.channels
    ? channels.channels.map(c => esc(c)).join("<br>")
    : '<div class="muted">No channels</div>';

  return stream;
}

async function refreshAlerts(){
  const data = await getJson(`/api/v1/alerts?tenant_id=${encodeURIComponent(tenant())}&limit=24`);
  const el = document.getElementById("alertsList");

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
  } else {
    document.getElementById("vHr").textContent = "-";
    document.getElementById("vSys").textContent = "-";
    document.getElementById("vDia").textContent = "-";
    document.getElementById("vSpo2").textContent = "-";
    document.getElementById("vTs").textContent = "-";
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
    badge.className = `badge ${riskClass(level)}`;
    badge.textContent = level;
    document.getElementById("riskSummary").textContent = alerts.alerts[0].message || "Recent alert found.";
  } else {
    badge.className = "badge risk-low";
    badge.textContent = "low";
    document.getElementById("riskSummary").textContent = "No recent alerts for this patient.";
  }
}

async function refreshSummary(alerts){
  const summaryBox = document.getElementById("summaryBox");
  if (!alerts || !alerts.length){
    summaryBox.innerHTML = '<div class="muted">No summary yet.</div>';
    return;
  }
  const counts = {};
  const patients = new Set();
  alerts.forEach(a => {
    const sev = (a.severity || "info").toLowerCase();
    counts[sev] = (counts[sev] || 0) + 1;
    patients.add(a.patient_id);
  });

  summaryBox.innerHTML =
    Object.keys(counts).sort().map(k => `<div class="kv"><div>${esc(k)}</div><div>${esc(counts[k])}</div></div>`).join("") +
    `<div class="kv" style="margin-top:10px"><div>Patients with alerts</div><div>${patients.size}</div></div>`;
}

async function refreshInvestor(overview, alerts){
  const kpis = document.getElementById("investorKpis");
  kpis.innerHTML = `
    <div class="item"><div class="row"><div>Patients monitored</div><div><strong>${esc(overview?.patient_count ?? 0)}</strong></div></div></div>
    <div class="item"><div class="row"><div>Open alerts</div><div><strong>${esc(overview?.open_alerts ?? 0)}</strong></div></div></div>
    <div class="item"><div class="row"><div>Critical alerts</div><div><strong>${esc(overview?.critical_alerts ?? 0)}</strong></div></div></div>
    <div class="item"><div class="row"><div>Events last hour</div><div><strong>${esc(overview?.events_last_hour ?? 0)}</strong></div></div></div>
  `;

  const mix = {};
  (alerts || []).forEach(a => {
    const sev = (a.severity || "info").toLowerCase();
    if (sev === "critical") mix["high"] = (mix["high"] || 0) + 1;
    else if (sev === "high") mix["moderate"] = (mix["moderate"] || 0) + 1;
    else mix["low"] = (mix["low"] || 0) + 1;
  });

  const investorRiskMix = document.getElementById("investorRiskMix");
  const keys = Object.keys(mix);
  investorRiskMix.innerHTML = keys.length
    ? keys.map(k => `<div class="item"><div class="row"><div>${esc(k)} risk signals</div><div><strong>${esc(mix[k])}</strong></div></div></div>`).join("")
    : '<div class="muted">No risk data yet.</div>';

  const scale = await getJson(`/api/v1/scale/readiness`);
  const scaleBox = document.getElementById("scaleBox");
  if (!scale){
    scaleBox.innerHTML = '<div class="muted">Unavailable</div>';
  } else {
    scaleBox.innerHTML = `
      <div class="item"><strong>${esc(scale.scaling_mode || scale.status)}</strong></div>
      ${(scale.next_steps || []).map(s => `<div class="item">${esc(s)}</div>`).join("")}
    `;
  }
}

async function refreshWall(alerts){
  const wall = document.getElementById("wallGrid");
  if (!alerts || !alerts.length){
    wall.innerHTML = '<div class="muted">No wall data yet.</div>';
    return;
  }

  const grouped = {};
  alerts.forEach(a => {
    if (!grouped[a.patient_id]) grouped[a.patient_id] = [];
    grouped[a.patient_id].push(a);
  });

  const patients = Object.keys(grouped).slice(0, 12);
  wall.innerHTML = patients.map(pid => {
    const items = grouped[pid];
    const critical = items.some(a => (a.severity || "").toLowerCase() === "critical");
    const high = items.some(a => (a.severity || "").toLowerCase() === "high");
    const level = critical ? "high" : high ? "moderate" : "low";
    const latest = items[0];

    return `
      <div class="patient-tile">
        <div class="tile-top">
          <div>
            <div class="muted">Patient</div>
            <div class="mono"><strong>${esc(pid)}</strong></div>
          </div>
          <div class="badge ${riskClass(level)}">${esc(level)}</div>
        </div>
        <div style="margin-top:10px">${esc(latest.alert_type || "no alert")}</div>
        <div class="muted small" style="margin-top:6px">${esc(latest.message || "")}</div>
        <div class="spark"></div>
        <div class="muted small" style="margin-top:8px">${items.length} active signal(s)</div>
      </div>
    `;
  }).join("");
}

async function refreshAll(){
  const overview = await refreshOverview();
  await refreshStream();
  const alerts = await refreshAlerts();
  await refreshPatient();
  await refreshSummary(alerts);
  await refreshInvestor(overview, alerts);
  await refreshWall(alerts);
}

setMode("hospital");
refreshAll();
setInterval(refreshAll, 10000);
</script>
</body>
</html>
"""

@web_bp.get("/")
def home():
    return render_template_string(HTML)
