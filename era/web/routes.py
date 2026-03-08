from flask import Blueprint, render_template, render_template_string, send_from_directory
import os

web_bp = Blueprint("web", __name__)

COMMAND_CENTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Early Risk Alert Command Center</title>
<style>
body{
    margin:0;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    background:#0b1220;
    color:#e6edf3;
}
.container{max-width:1400px;margin:auto;padding:30px 20px}
.hero{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:20px}
.title{font-size:38px;font-weight:800}
.subtitle{color:#94a3b8;margin-top:6px}
.live-pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    margin-top:14px;
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
    animation:pulseLive 1.4s infinite;
}
@keyframes pulseLive{
    0%{transform:scale(1);opacity:1;}
    50%{transform:scale(1.25);opacity:.55;}
    100%{transform:scale(1);opacity:1;}
}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}
input,button{
    padding:10px 14px;
    border-radius:10px;
    border:1px solid #334155;
    background:#111827;
    color:white;
}
button{cursor:pointer;background:#2563eb;border:none}
button.secondary{background:#1e293b}
.ticker{
  width:100%;
  overflow:hidden;
  background:linear-gradient(90deg,#0f1b2d,#13233a);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:12px;
  margin:12px 0 20px 0;
}
.ticker-track{
  white-space:nowrap;
  padding:10px 0;
  font-weight:600;
  color:#ff6767;
  animation:tickerMove 25s linear infinite;
}
@keyframes tickerMove{
  0% { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:20px}
.card{
    background:#111827;
    border-radius:16px;
    padding:18px;
    box-shadow:0 4px 24px rgba(0,0,0,.25);
}
.label{font-size:13px;color:#94a3b8}
.value{font-size:36px;font-weight:800;margin-top:8px}
.layout{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:18px}
.section{font-size:18px;font-weight:700;margin-bottom:14px}
.list{display:flex;flex-direction:column;gap:12px}
.item{
    background:#0f172a;
    border-radius:14px;
    padding:14px;
    border:1px solid rgba(255,255,255,.05);
}
.row{display:flex;justify-content:space-between;align-items:center;gap:12px}
.badge{
    padding:5px 10px;
    border-radius:999px;
    font-size:12px;
    font-weight:700;
}
.critical{background:rgba(239,68,68,.18);color:#fecaca}
.high{background:rgba(245,158,11,.18);color:#fde68a}
.info{background:rgba(59,130,246,.18);color:#bfdbfe}
.risk-high{background:rgba(239,68,68,.18);color:#fecaca}
.risk-moderate{background:rgba(245,158,11,.18);color:#fde68a}
.risk-low{background:rgba(34,197,94,.18);color:#bbf7d0}
.kv{display:grid;grid-template-columns:1fr auto;gap:8px 14px}
.muted{color:#94a3b8}
.small{font-size:12px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.footer-note{margin-top:18px;color:#64748b;font-size:13px;text-align:right}
@media(max-width:1200px){
    .kpis{grid-template-columns:repeat(3,1fr)}
    .layout{grid-template-columns:1fr}
}
@media(max-width:700px){
    .kpis{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>
<div class="container">
    <div class="hero">
        <div>
            <div class="title">Early Risk Alert</div>
            <div class="subtitle">Hospital UI + Investor Demo + Command Center</div>
            <div class="live-pill">
                <span class="live-dot"></span>
                LIVE SYSTEM
            </div>
        </div>
        <div>
            <a href="/investors" style="color:#93c5fd;text-decoration:none;font-weight:700;">Investor Site</a>
        </div>
    </div>

    <div class="toolbar">
        <input id="tenantId" value="demo" placeholder="tenant_id">
        <input id="patientId" value="p101" placeholder="patient_id">
        <button onclick="refreshAll()">Refresh</button>
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
        <div class="card"><div class="label">Stream Mode</div><div class="value" id="kpiStreamMode" style="font-size:24px">—</div></div>
        <div class="card"><div class="label">Commercial Model</div><div class="value" style="font-size:24px">SaaS</div></div>
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
                            <div id="patientIdLabel" class="mono" style="font-size:26px;font-weight:800">—</div>
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
function tenant(){ return document.getElementById("tenantId").value || "demo"; }
function patient(){ return document.getElementById("patientId").value || "p101"; }
function esc(v){
  return (v ?? "").toString().replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
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
.nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px;gap:20px;flex-wrap:wrap}
.nav a{color:#9fb3c8;text-decoration:none;margin-left:20px;font-size:14px}
.logo{font-weight:700;font-size:18px;color:#fff}
.hero{display:grid;grid-template-columns:1.2fr 1fr;gap:40px;align-items:center;margin-bottom:60px}
.hero h1{font-size:48px;line-height:1.1;margin-bottom:20px}
.hero p{color:#9fb3c8;font-size:18px}
.btn{display:inline-block;padding:14px 22px;border-radius:10px;margin-top:20px;text-decoration:none;font-weight:600}
.btn-primary{background:#2563eb;color:white}
.btn-secondary{background:#1e293b;color:#cbd5e1}
.btn-outline{background:transparent;color:#cbd5e1;border:1px solid #334155}
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
.cta-row{display:flex;gap:14px;flex-wrap:wrap;margin-top:20px}
.contact-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;margin-top:30px}
.contact-card{background:#111827;border-radius:16px;padding:22px}
.contact-label{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.contact-value{font-size:20px;font-weight:700;color:#fff}
.quote{font-size:22px;line-height:1.4;color:#dbeafe}
.credibility{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-top:30px}
.cred-box{background:#111827;border-radius:14px;padding:16px;text-align:center;font-weight:600;color:#cbd5e1}
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
        <a href="/pitch-deck" class="btn btn-outline">Download Pitch Deck PDF</a>
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
        <div class="cta-row">
            <a href="#contact" class="btn btn-primary">Contact Founder</a>
            <a href="/pitch-deck" class="btn btn-secondary">Download Pitch Deck</a>
            <a href="/" class="btn btn-outline">Open Live Demo</a>
        </div>
    </div>
</div>

<div id="contact" class="section">
    <h2>Investor Contact</h2>
    <div class="contact-grid">
        <div class="contact-card">
            <div class="contact-label">Founder</div>
            <div class="contact-value">Milton Munroe</div>
        </div>
        <div class="contact-card">
            <div class="contact-label">Email</div>
            <div class="contact-value">info@earlyriskalertai.com</div>
        </div>
        <div class="contact-card">
            <div class="contact-label">Business Phone</div>
            <div class="contact-value">732-724-7267</div>
        </div>
        <div class="contact-card">
            <div class="contact-label">Investor Materials</div>
            <div class="contact-value"><a href="mailto:info@earlyriskalertai.com?subject=Request%20Investor%20Deck" style="color:#60a5fa;font-weight:700">Request Investor Deck</a></div>
        </div>
        <div class="contact-card">
            <div class="contact-label">Pitch Deck</div>
            <div class="contact-value"><a href="/pitch-deck" style="color:#60a5fa;font-weight:700">Download Pitch Deck PDF</a></div>
        </div>
        <div class="contact-card">
            <div class="contact-label">Demo Access</div>
            <div class="contact-value"><a href="/" style="color:#60a5fa;font-weight:700">Open Live Hospital Demo</a></div>
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
