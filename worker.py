from flask import Blueprint, render_template_string

web_bp = Blueprint("web", __name__)

DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#0b1220;
      --card:#121a2b;
      --muted:#93a4c3;
      --text:#eaf0ff;
      --border:rgba(255,255,255,.08);
      --accent:#60a5fa;
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--text)}
    .shell{max-width:1400px;margin:24px auto;padding:0 16px}
    .topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:18px}
    .title{font-size:28px;font-weight:800}
    .subtitle{color:var(--muted);font-size:14px;margin-top:4px}
    .toolbar{display:flex;gap:10px;flex-wrap:wrap}
    input,button{border:1px solid var(--border);background:#0f1729;color:var(--text);padding:10px 12px;border-radius:10px;font-size:14px}
    button{cursor:pointer;background:var(--accent);color:#081120;font-weight:700;border:none}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px}
    .card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:16px}
    .kpi-label{color:var(--muted);font-size:13px}
    .kpi-value{font-size:32px;font-weight:800;margin-top:8px}
    .layout{display:grid;grid-template-columns:1.1fr 1.1fr .8fr;gap:14px}
    .section-title{font-size:16px;font-weight:800;margin-bottom:12px}
    .list{display:flex;flex-direction:column;gap:10px;max-height:520px;overflow:auto}
    .item{border:1px solid var(--border);border-radius:14px;padding:12px;background:rgba(255,255,255,.02)}
    .row{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
    .muted{color:var(--muted);font-size:13px}
    .badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:700}
    .sev-high,.risk-high{background:rgba(239,68,68,.18);color:#fecaca}
    .sev-warn,.risk-moderate{background:rgba(245,158,11,.18);color:#fde68a}
    .sev-good,.risk-low{background:rgba(31,157,85,.18);color:#bbf7d0}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .small{font-size:12px}
    .kv{display:grid;grid-template-columns:1fr auto;gap:8px;font-size:14px}
    .panel-note{color:var(--muted);font-size:13px;line-height:1.5}
    @media (max-width:1100px){.grid{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}}
    @media (max-width:700px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="title">Early Risk Alert Command Center</div>
        <div class="subtitle">Real-time patient monitoring dashboard</div>
      </div>
      <div class="toolbar">
        <input id="tenantId" value="demo" placeholder="tenant_id">
        <input id="patientId" value="p100" placeholder="patient_id">
        <button onclick="refreshAll()">Refresh</button>
      </div>
    </div>

    <div class="grid">
      <div class="card"><div class="kpi-label">Patients</div><div class="kpi-value" id="kpiPatients">-</div></div>
      <div class="card"><div class="kpi-label">Open Alerts</div><div class="kpi-value" id="kpiOpenAlerts">-</div></div>
      <div class="card"><div class="kpi-label">Critical Alerts</div><div class="kpi-value" id="kpiCriticalAlerts">-</div></div>
      <div class="card"><div class="kpi-label">Events Last Hour</div><div class="kpi-value" id="kpiEventsHour">-</div></div>
    </div>

    <div class="layout">
      <div class="card">
        <div class="section-title">Live Alerts Feed</div>
        <div class="list" id="alertsList"></div>
      </div>

      <div class="card">
        <div class="section-title">Patient Risk + Latest Vitals</div>
        <div class="list">
          <div class="item">
            <div class="row">
              <div>
                <div class="muted">Patient</div>
                <div id="patientRiskId" class="mono">-</div>
              </div>
              <div id="patientRiskBadge" class="badge risk-low">unknown</div>
            </div>
            <div style="margin-top:10px" id="patientRiskSummary" class="panel-note">No data yet.</div>
          </div>

          <div class="item">
            <div class="muted">Latest vitals</div>
            <div class="kv" style="margin-top:10px">
              <div>Heart rate</div><div id="vHeartRate">-</div>
              <div>Systolic BP</div><div id="vSys">-</div>
              <div>Diastolic BP</div><div id="vDia">-</div>
              <div>SpO2</div><div id="vSpo2">-</div>
              <div>Event time</div><div id="vEventTs" class="small">-</div>
            </div>
          </div>

          <div class="item">
            <div class="muted">15-minute rollups</div>
            <div id="rollupsList" style="margin-top:10px" class="list"></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="section-title">Command Center</div>
        <div class="list">
          <div class="item">
            <div class="muted">Open alerts by severity</div>
            <div id="severitySummary" style="margin-top:10px"></div>
          </div>
          <div class="item">
            <div class="muted">Patient risk mix</div>
            <div id="riskMixSummary" style="margin-top:10px"></div>
          </div>
          <div class="item">
            <div class="muted">Scaling readiness</div>
            <div id="scaleMode" style="margin-top:8px;font-weight:700">-</div>
            <div id="scaleSteps" class="panel-note" style="margin-top:8px"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
function tenant(){ return document.getElementById("tenantId").value || "demo"; }
function patient(){ return document.getElementById("patientId").value || "p100"; }

function esc(v){
  return (v ?? "").toString()
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;");
}

function sevClass(v){
  v = (v || "").toLowerCase();
  if (v.includes("critical") || v.includes("high")) return "sev-high";
  if (v.includes("moderate") || v.includes("warn")) return "sev-warn";
  return "sev-good";
}

function riskClass(v){
  v = (v || "").toLowerCase();
  if (v === "high") return "risk-high";
  if (v === "moderate") return "risk-moderate";
  return "risk-low";
}

async function getJson(url){
  const r = await fetch(url);
  if (!r.ok) return null;
  return await r.json();
}

async function refreshOverview(){
  const data = await getJson(`/api/v1/dashboard/overview?tenant_id=${encodeURIComponent(tenant())}`);
  if (!data) return;
  document.getElementById("kpiPatients").textContent = data.patient_count ?? 0;
  document.getElementById("kpiOpenAlerts").textContent = data.open_alerts ?? 0;
  document.getElementById("kpiCriticalAlerts").textContent = data.critical_alerts ?? 0;
  document.getElementById("kpiEventsHour").textContent = data.events_last_hour ?? 0;
}

async function refreshAlerts(){
  const data = await getJson(`/api/v1/alerts/live?tenant_id=${encodeURIComponent(tenant())}`);
  const el = document.getElementById("alertsList");
  if (!data || !data.alerts) {
    el.innerHTML = '<div class="item">No alerts available.</div>';
    return;
  }
  if (data.alerts.length === 0) {
    el.innerHTML = '<div class="item">No live alerts.</div>';
    return;
  }
  el.innerHTML = data.alerts.map(a => `
    <div class="item">
      <div class="row">
        <div>
          <div><strong>${esc(a.alert_type)}</strong></div>
          <div class="muted">Patient ${esc(a.patient_id)}</div>
        </div>
        <div class="badge ${sevClass(a.severity)}">${esc(a.severity)}</div>
      </div>
      <div style="margin-top:8px">${esc(a.message)}</div>
      <div class="muted small" style="margin-top:8px">${esc(a.created_at || "")}</div>
    </div>
  `).join("");
}

async function refreshRisk(){
  const pid = patient();
  document.getElementById("patientRiskId").textContent = pid;

  const risk = await getJson(`/api/v1/patients/${encodeURIComponent(pid)}/risk?tenant_id=${encodeURIComponent(tenant())}`);
  const badge = document.getElementById("patientRiskBadge");
  if (risk) {
    badge.className = `badge ${riskClass(risk.risk_level)}`;
    badge.textContent = `${risk.risk_level} (${risk.risk_score})`;
    document.getElementById("patientRiskSummary").textContent = risk.summary || "No summary.";
  } else {
    badge.className = "badge risk-low";
    badge.textContent = "no data";
    document.getElementById("patientRiskSummary").textContent = "No risk data yet.";
  }

  const vitals = await getJson(`/api/v1/vitals/latest?tenant_id=${encodeURIComponent(tenant())}&patient_id=${encodeURIComponent(pid)}`);
  if (vitals && vitals.vitals) {
    document.getElementById("vHeartRate").textContent = vitals.vitals.heart_rate ?? "-";
    document.getElementById("vSys").textContent = vitals.vitals.systolic_bp ?? "-";
    document.getElementById("vDia").textContent = vitals.vitals.diastolic_bp ?? "-";
    document.getElementById("vSpo2").textContent = vitals.vitals.spo2 ?? "-";
    document.getElementById("vEventTs").textContent = vitals.event_ts ?? "-";
  } else {
    document.getElementById("vHeartRate").textContent = "-";
    document.getElementById("vSys").textContent = "-";
    document.getElementById("vDia").textContent = "-";
    document.getElementById("vSpo2").textContent = "-";
    document.getElementById("vEventTs").textContent = "-";
  }

  const rollups = await getJson(`/api/v1/rollups/15m?tenant_id=${encodeURIComponent(tenant())}&patient_id=${encodeURIComponent(pid)}&since_minutes=120`);
  const rollupsEl = document.getElementById("rollupsList");
  if (!rollups || !rollups.rollups || rollups.rollups.length === 0) {
    rollupsEl.innerHTML = '<div class="item small">No rollup data yet.</div>';
    return;
  }
  rollupsEl.innerHTML = rollups.rollups.slice(0, 6).map(r => `
    <div class="item small">
      <div><strong>${esc(r.bucket_start || "")}</strong></div>
      <div class="muted">Samples: ${esc(r.samples)}</div>
      <div>HR ${esc(r.avg_heart_rate)} | BP ${esc(r.avg_systolic_bp)}/${esc(r.avg_diastolic_bp)} | SpO2 ${esc(r.avg_spo2)}</div>
    </div>
  `).join("");
}

async function refreshCommandCenter(){
  const cc = await getJson(`/api/v1/command-center/summary?tenant_id=${encodeURIComponent(tenant())}`);
  if (cc) {
    document.getElementById("severitySummary").innerHTML =
      (cc.open_alerts_by_severity || []).map(x => `<div class="kv"><div>${esc(x.severity)}</div><div>${esc(x.count)}</div></div>`).join("") || '<div class="muted">No open alerts.</div>';

    document.getElementById("riskMixSummary").innerHTML =
      (cc.patient_risk_mix || []).map(x => `<div class="kv"><div>${esc(x.risk_level)}</div><div>${esc(x.count)}</div></div>`).join("") || '<div class="muted">No patients scored yet.</div>';
  }

  const scale = await getJson(`/api/v1/scale/readiness`);
  if (scale) {
    document.getElementById("scaleMode").textContent = scale.scaling_mode || scale.status || "-";
    document.getElementById("scaleSteps").innerHTML = (scale.next_steps || []).map(s => `• ${esc(s)}`).join("<br>");
  }
}

async function refreshAll(){
  await Promise.all([
    refreshOverview(),
    refreshAlerts(),
    refreshRisk(),
    refreshCommandCenter()
  ]);
}

refreshAll();
setInterval(refreshAll, 10000);
</script>
</body>
</html>
"""

@web_bp.get("/")
def home():
    return render_template_string(DASHBOARD_HTML)
