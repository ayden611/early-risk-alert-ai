from flask import Blueprint, render_template, render_template_string

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
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--text)}
    .shell{max-width:1500px;margin:20px auto;padding:0 16px}
    .topbar{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:16px}
    .title{font-size:30px;font-weight:800;line-height:1.1}
    .subtitle{margin-top:6px;color:var(--muted);font-size:14px}
    .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    input,button{
      border:1px solid var(--border);
      background:var(--panel2);
      color:var(--text);
      padding:10px 12px;
      border-radius:10px;
      font-size:14px;
    }
    button{background:var(--accent);color:#07111f;font-weight:800;border:none;cursor:pointer}
    .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:14px}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
    .label{color:var(--muted);font-size:13px}
    .value{font-size:30px;font-weight:800;margin-top:8px}
    .section{font-size:16px;font-weight:800;margin-bottom:12px}
    .layout{display:grid;grid-template-columns:1.1fr 1fr .9fr;gap:14px}
    .list{display:flex;flex-direction:column;gap:10px;max-height:620px;overflow:auto}
    .item{border:1px solid var(--border);background:rgba(255,255,255,.02);border-radius:14px;padding:12px}
    .row{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
    .muted{color:var(--muted);font-size:13px}
    .small{font-size:12px}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:800}
    .critical,.risk-high{background:rgba(239,68,68,.18);color:#fecaca}
    .high,.risk-moderate{background:rgba(245,158,11,.18);color:#fde68a}
    .info,.low,.risk-low{background:rgba(34,197,94,.18);color:#bbf7d0}
    .kv{display:grid;grid-template-columns:1fr auto;gap:8px;font-size:14px}
    @media (max-width:1200px){.kpis{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}}
    @media (max-width:700px){.kpis{grid-template-columns:1fr}}
    .live-pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  margin-top:10px;
  padding:8px 14px;
  border-radius:999px;
  background:rgba(34,197,94,.12);
  border:1px solid rgba(34,197,94,.35);
  color:#d1fae5;
  font-size:12px;
  font-weight:700;
  letter-spacing:.08em;
}

.live-dot{
  width:10px;
  height:10px;
  border-radius:999px;
  background:#22c55e;
  box-shadow:0 0 10px rgba(34,197,94,.8);
  animation
  </style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div>
      <div class="title">Early Risk Alert</div>
      <div class="subtitle">Hospital UI + Investor Demo + Command Center</div>
    </div>
    <div class="topbar">
    <div>
        <div class="title">Early Risk Alert</div>
        <div class="subtitle">Hospital UI + Investor Demo + Command Center</div>
    </div>
    <div class="toolbar">
      <input id="tenantId" value="demo" placeholder="tenant_id">
      <input id="patientId" value="p101" placeholder="patient_id">
      <button onclick="refreshAll()">Refresh</button>
    </div>
  </div>

  <div class="kpis">
    <div class="card"><div class="label">Patients</div><div class="value" id="kpiPatients">-</div></div>
    <div class="card"><div class="label">Open Alerts</div><div class="value" id="kpiOpenAlerts">-</div></div>
    <div class="card"><div class="label">Critical Alerts</div><div class="value" id="kpiCriticalAlerts">-</div></div>
    <div class="card"><div class="label">Events Last Hour</div><div class="value" id="kpiEventsHour">-</div></div>
    <div class="card"><div class="label">Stream Mode</div><div class="value" id="kpiStreamMode" style="font-size:20px">-</div></div>
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
  document.getElementById("channelsBox").innerHTML = channels && channels.channels
    ? channels.channels.map(c => esc(c)).join("<br>")
    : '<div class="muted">No channels</div>';
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

async function refreshAll(){
  const alerts = await refreshAlerts();
  await refreshOverview();
  await refreshStream();
  await refreshPatient();
  await refreshSummary(alerts);
}

refreshAll();
setInterval(refreshAll, 10000);
</script>
</body>
</html>
"""
@web_bp.get("/")
def home():
    return render_template_string(HTML)

@web_bp.get("/login")
def login():
    return render_template("login.html")

@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(HTML)
