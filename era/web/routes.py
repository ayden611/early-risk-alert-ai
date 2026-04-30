from flask import Blueprint, render_template_string, send_file
import os

web_bp = Blueprint("web", __name__)

YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/HiidXiXifY4"

MAIN_HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{{
      --bg:#08111f;
      --bg2:#0d1526;
      --panel:#101a2d;
      --line:rgba(255,255,255,.08);
      --text:#edf4ff;
      --muted:#94a7c6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --red:#ff7a7a;
      --amber:#f5c06a;
      --radius:22px;
      --max:1320px;
    }}
    *{{box-sizing:border-box}}
    html{{scroll-behavior:smooth}}
    body{{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.12), transparent 22%),
        radial-gradient(circle at top right, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }}
    a{{color:inherit;text-decoration:none}}
    .nav{{
      position:sticky;top:0;z-index:50;
      backdrop-filter:blur(14px);
      background:rgba(8,17,31,.76);
      border-bottom:1px solid var(--line);
    }}
    .nav-inner{{
      max-width:var(--max);margin:0 auto;padding:14px 16px;
      display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
    }}
    .brand-kicker{{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#b9c9e7;font-weight:800}}
    .brand-title{{font-size:clamp(26px,3vw,40px);font-weight:1000;line-height:.95;letter-spacing:-.04em}}
    .brand-sub{{font-size:14px;color:var(--muted);font-weight:700}}
    .nav-links{{display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
    .nav-links a{{font-weight:800;font-size:14px}}
    .btn{{
      display:inline-flex;align-items:center;justify-content:center;
      padding:13px 18px;border-radius:16px;font-weight:900;font-size:14px;
      background:linear-gradient(135deg,var(--blue),var(--blue2));color:#08111f;
      border:1px solid transparent;
    }}
    .btn.secondary{{background:#111b2f;color:var(--text);border-color:var(--line)}}
    .shell{{max-width:var(--max);margin:0 auto;padding:24px 16px 70px}}
    .hero{{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;padding-top:18px}}
    .card{{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--line);border-radius:var(--radius);padding:28px;
    }}
    .hero-kicker{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#c8d7f1;font-weight:900;margin-bottom:12px}}
    h1{{margin:0 0 14px;font-size:clamp(40px,5vw,76px);line-height:.95;font-weight:1000;letter-spacing:-.055em}}
    .lead{{margin:0;color:#c8d7f1;font-size:clamp(16px,1.5vw,20px);line-height:1.55}}
    .hero-actions{{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}}
    .live-pill{{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;background:rgba(56,211,159,.12);border:1px solid rgba(56,211,159,.25);font-weight:900;font-size:12px;color:#bff5df;text-transform:uppercase;letter-spacing:.08em}}
    .dot{{width:10px;height:10px;border-radius:999px;background:var(--green);box-shadow:0 0 18px rgba(56,211,159,.6)}}
    .side-title{{font-size:clamp(28px,2.8vw,38px);font-weight:950;line-height:1.02;margin:18px 0 8px;letter-spacing:-.04em}}
    .side-copy{{color:#b8cae8;line-height:1.65;margin:0 0 18px;font-size:15px}}
    .mini-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
    .mini{{border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.025);border-radius:18px;padding:16px}}
    .mini-k{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#9fb7da;font-weight:900}}
    .mini-v{{font-size:clamp(28px,2.4vw,34px);font-weight:950;line-height:1;margin-top:10px}}
    .mini-s{{font-size:14px;color:#b7c9e7;margin-top:8px;line-height:1.45}}
    .section{{margin-top:26px}}
    .section-title{{font-size:clamp(30px,3.1vw,42px);font-weight:1000;letter-spacing:-.04em;margin:0 0 8px;line-height:1.02}}
    .section-sub{{color:var(--muted);font-size:16px;line-height:1.65;margin:0 0 18px;max-width:980px}}
    .metrics-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
    .metric{{padding:22px}}
    .metric-label{{color:#a7bad9;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.14em}}
    .metric-value{{font-size:clamp(38px,4vw,54px);font-weight:1000;line-height:1;margin-top:10px}}
    .metric-note{{color:#bfd0ea;font-size:14px;margin-top:8px;line-height:1.45}}
    .dashboard-grid{{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:14px}}
    .dash-card{{padding:22px;min-height:360px}}
    .dash-card h3{{margin:0 0 6px;font-size:22px}}
    .dash-card p{{margin:0 0 16px;color:var(--muted);line-height:1.6}}
    .feed{{display:flex;flex-direction:column;gap:12px}}
    .alert{{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px}}
    .alert-top{{display:flex;justify-content:space-between;gap:12px}}
    .alert-name{{font-size:18px;font-weight:900}}
    .alert-patient{{font-size:13px;color:#b6c8e8;font-weight:700}}
    .alert-msg{{font-size:18px;font-weight:800;margin-top:6px;line-height:1.35}}
    .alert-time{{font-size:13px;color:#9ab0d3;margin-top:6px}}
    .badge{{display:inline-flex;align-items:center;justify-content:center;min-width:78px;padding:7px 11px;border-radius:999px;font-size:12px;text-transform:uppercase;letter-spacing:.1em;font-weight:900}}
    .critical{{background:rgba(255,122,122,.12);border:1px solid rgba(255,122,122,.22);color:#ffc1c1}}
    .high{{background:rgba(245,192,106,.14);border:1px solid rgba(245,192,106,.24);color:#ffe0a8}}
    .moderate{{background:rgba(122,162,255,.14);border:1px solid rgba(122,162,255,.24);color:#d1e0ff}}
    .kv{{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.05)}}
    .kv:last-child{{border-bottom:none}}
    .k{{color:#b6c8e8;font-weight:700}}
    .v{{font-weight:900}}
    .panel-block{{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:18px;padding:16px;margin-top:12px}}
    .channels{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#d7e5ff;font-size:13px;line-height:1.7}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
    iframe{{width:100%;aspect-ratio:16/9;border:0;border-radius:18px;background:#0a0f18}}
    .summary-list{{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-top:16px}}
    .summary-item .sk{{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9db3d5;font-weight:900}}
    .summary-item .sv{{font-size:20px;font-weight:900}}
    .trust-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
    .trust-card{{padding:22px}}
    .trust-card h3{{margin:0 0 8px;font-size:18px}}
    .trust-card p{{margin:0;color:#bfd0ea;line-height:1.65}}
    .contact-box{{display:grid;gap:12px;margin-top:18px}}
    .contact-row{{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);border-radius:16px;padding:14px 16px}}
    .contact-label{{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9eb4d6;font-weight:900}}
    .contact-value{{font-size:18px;font-weight:900;margin-top:6px;line-height:1.4}}
    .footer{{margin-top:24px;padding:24px 12px 8px;color:#94a7c6;font-size:14px;text-align:center;line-height:1.6}}
    @media (max-width:1180px){{
      .hero,.two-col{{grid-template-columns:1fr}}
      .dashboard-grid{{grid-template-columns:1fr 1fr}}
      .dashboard-grid .dash-card:first-child{{grid-column:1/-1}}
      .metrics-grid,.trust-grid{{grid-template-columns:repeat(2,1fr)}}
    }}
    @media (max-width:760px){{
      .metrics-grid,.dashboard-grid,.trust-grid,.mini-grid,.summary-list{{grid-template-columns:1fr}}
      .nav-links{{width:100%}}
      h1{{font-size:clamp(34px,10vw,46px)}}
      .alert-top,.kv{{flex-direction:column}}
    }}
  </style>
<!-- ERA_METRIC_GUARDRAIL_CLARITY_HEAD_START -->
<link rel="stylesheet" href="/era-static/era_metric_guardrail_clarity.css?v=guardrail1">
<!-- ERA_METRIC_GUARDRAIL_CLARITY_HEAD_END -->
<!-- ERA_COMMAND_QUEUE_CONSISTENCY_HEAD_START -->
<link rel="stylesheet" href="/era-static/era_command_queue_consistency.css?v=queue-consistency1">
<!-- ERA_COMMAND_QUEUE_CONSISTENCY_HEAD_END -->
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
      <div class="card">
        <div class="hero-kicker">Clinical command platform</div>
        <h1>Detect patient deterioration earlier. Strengthen hospital response at scale.</h1>
        <p class="lead">
          Early Risk Alert AI is a professional healthcare intelligence platform built to help hospitals,
          care teams, clinics, and remote monitoring programs surface high-risk patients in real time,
          prioritize action faster, and present enterprise-grade command-center visibility.
        </p>
        <div class="hero-actions">
          <a class="btn" href="#dashboard">Open Live Dashboard</a>
          <a class="btn secondary" href="#demo">View Demo Section</a>
          <a class="btn secondary" href="/investors">Open Investor Portal</a>
        </div>
      </div>

      <div class="card">
        <div class="live-pill"><span class="dot"></span> Live system</div>
        <div class="side-title">Built for modern hospital operations</div>
        <p class="side-copy">
          Your public-facing platform now combines hospital product messaging, animated live metrics,
          AI risk scoring, a command-center style dashboard, and investor-ready presentation in one polished experience.
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
      <p class="section-sub">A professional hospital-facing metrics layer showing alert volume, monitored patients, critical cases, and clinical activity.</p>
      <div class="metrics-grid">
        <div class="card metric">
          <div class="metric-label">Monitored Patients</div>
          <div class="metric-value" id="metricPatients">0</div>
          <div class="metric-note">Active patient population</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Open Alerts</div>
          <div class="metric-value" id="metricAlerts">0</div>
          <div class="metric-note">Current command-center queue</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Critical Alerts</div>
          <div class="metric-value" id="metricCritical">0</div>
          <div class="metric-note">Highest-priority interventions</div>
        </div>
        <div class="card metric">
          <div class="metric-label">Avg Risk Score</div>
          <div class="metric-value" id="metricRisk">0</div>
          <div class="metric-note">AI risk intensity snapshot</div>
        </div>
      </div>
    </section>

    <section class="section" id="dashboard">
      <h2 class="section-title">Live dashboard</h2>
      <p class="section-sub">A professional, hospital-facing command-center layout showing alerts, patient details, and system health.</p>

      <div class="dashboard-grid">
        <div class="card dash-card">
          <h3>Live Alerts Feed</h3>
          <p>Clinical alert prioritization and real-time visibility.</p>
          <div class="feed" id="alertsFeed"></div>
        </div>

        <div class="card dash-card">
          <h3>Patient Focus</h3>
          <p>Focused patient view for vitals, event timing, and rollup metrics.</p>
          <div id="patientFocus"></div>
        </div>

        <div class="card dash-card">
          <h3>System Panels</h3>
          <p>Operational stream health, monitoring state, and channels.</p>
          <div class="panel-block" id="streamHealth"></div>
          <div class="panel-block channels" id="streamChannels"></div>
        </div>
      </div>
    </section>

    <section class="section" id="credibility">
      <h2 class="section-title">Business credibility layer</h2>
      <p class="section-sub">Professional positioning for hospitals, operators, and investors evaluating deployment readiness.</p>
      <div class="trust-grid">
        <div class="card trust-card">
          <h3>Hospital operations focus</h3>
          <p>Built to support command-center visibility, prioritization workflows, and faster intervention escalation.</p>
        </div>
        <div class="card trust-card">
          <h3>Investor-ready presentation</h3>
          <p>Unified public platform, live metrics, demo section, and downloadable pitch materials in one branded experience.</p>
        </div>
        <div class="card trust-card">
          <h3>Clinical intelligence positioning</h3>
          <p>AI-driven risk scoring and live deterioration awareness framed for modern healthcare system adoption.</p>
        </div>
        <div class="card trust-card">
          <h3>Professional contact path</h3>
          <p>Direct communication channel for hospital partnerships, investor inquiries, and product demonstrations.</p>
        </div>
      </div>

      <div class="card" style="margin-top:14px;">
        <h3 style="margin-top:0;">Contact</h3>
        <div class="contact-box">
          <div class="contact-row">
            <div class="contact-label">Email</div>
            <div class="contact-value">info@earlyriskai.com</div>
          </div>
          <div class="contact-row">
            <div class="contact-label">Business Phone</div>
            <div class="contact-value">732-724-7267</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="demo">
      <h2 class="section-title">Live product demo</h2>
      <p class="section-sub">Professional embedded demo for hospital and investor presentations.</p>

      <div class="two-col">
        <div class="card">
          <iframe src="{YOUTUBE_EMBED_URL}" allowfullscreen></iframe>
        </div>

        <div class="card">
          <h3 style="margin-top:0;">Platform summary</h3>
          <p class="side-copy">
            Early Risk Alert AI combines hospital-facing product messaging, AI risk scoring, live operational dashboarding,
            and investor presentation flow into one polished healthcare platform.
          </p>

          <div class="summary-list">
            <div class="summary-item">
              <div class="sk">Use case</div>
              <div class="sv">Clinical command</div>
            </div>
            <div class="summary-item">
              <div class="sk">Deployment</div>
              <div class="sv">Enterprise SaaS</div>
            </div>
            <div class="summary-item">
              <div class="sk">Audience</div>
              <div class="sv">Hospitals / Clinics / Investors</div>
            </div>
            <div class="summary-item">
              <div class="sk">Experience</div>
              <div class="sv">Live dashboard + demo</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC • Professional healthcare intelligence platform • info@earlyriskai.com • 732-724-7267
    </div>
  </div>

  <script>
    const tenantId = "demo";
    const patientId = "p101";

    async function getJson(url) {{
      const res = await fetch(url, {{ cache: "no-store" }});
      if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
      return await res.json();
    }}

    function setText(id, value) {{
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }}

    function badgeClass(sev) {{
      if (sev === "critical") return "badge critical";
      if (sev === "high") return "badge high";
      if (sev === "moderate") return "badge moderate";
      return "badge";
    }}

    function renderAlerts(alerts) {{
      const feed = document.getElementById("alertsFeed");
      if (!feed) return;
      feed.innerHTML = "";

      alerts.forEach(a => {{
        const div = document.createElement("div");
        div.className = "alert";
        div.innerHTML = `
          <div class="alert-top">
            <div>
              <div class="alert-name">${{a.alert_type || "alert"}}</div>
              <div class="alert-patient">Patient ${{a.patient_id}}</div>
            </div>
            <div class="${{badgeClass(a.severity)}}">${{a.severity}}</div>
          </div>
          <div class="alert-msg">${{a.message || "Clinical event detected"}}</div>
          <div class="alert-time">live</div>
        `;
        feed.appendChild(div);
      }});
    }}

    function renderFocus(focus) {{
      const el = document.getElementById("patientFocus");
      if (!el) return;

      const v = focus.vitals || {{}};
      const r = focus.risk || {{}};
      const roll = focus.rollups || {{}};

      el.innerHTML = `
        <div class="panel-block">
          <div class="alert-patient">${{focus.patient_name || "Patient"}}</div>
          <div style="font-size:32px;font-weight:1000;line-height:1.05;margin:6px 0;">${{focus.patient_id || ""}}</div>
          <div style="font-size:18px;font-weight:800;color:#dbe7fb;">${{r.alert_message || "Vitals stable"}}</div>
          <div style="margin-top:10px;" class="${{badgeClass(r.severity)}}">${{r.severity || "stable"}}</div>
        </div>

        <div class="panel-block">
          <div class="kv"><span class="k">Heart Rate</span><span class="v">${{v.heart_rate ?? "--"}}</span></div>
          <div class="kv"><span class="k">Systolic BP</span><span class="v">${{v.systolic_bp ?? "--"}}</span></div>
          <div class="kv"><span class="k">Diastolic BP</span><span class="v">${{v.diastolic_bp ?? "--"}}</span></div>
          <div class="kv"><span class="k">SpO2</span><span class="v">${{v.spo2 ?? "--"}}</span></div>
          <div class="kv"><span class="k">Risk Score</span><span class="v">${{r.risk_score ?? "--"}}</span></div>
          <div class="kv"><span class="k">Avg SpO2</span><span class="v">${{roll.avg_spo2 ?? "--"}}</span></div>
        </div>
      `;
    }}

    function renderStreamHealth(data) {{
      const el = document.getElementById("streamHealth");
      if (!el) return;
      el.innerHTML = `
        <div class="kv"><span class="k">Status</span><span class="v">${{data.status}}</span></div>
        <div class="kv"><span class="k">Mode</span><span class="v">${{data.mode}}</span></div>
        <div class="kv"><span class="k">Worker</span><span class="v">${{data.worker_status}}</span></div>
        <div class="kv"><span class="k">Patients with alerts</span><span class="v">${{data.patients_with_alerts}}</span></div>
      `;
    }}

    function renderChannels(data) {{
      const el = document.getElementById("streamChannels");
      if (!el) return;
      el.innerHTML = (data.channels || []).map(x => `<div>${{x}}</div>`).join("");
    }}

    async function refreshDashboard() {{
      try {{
        const overview = await getJson(`/api/v1/dashboard/overview?tenant_id=${{tenantId}}`);
        const live = await getJson(`/api/v1/live-snapshot?tenant_id=${{tenantId}}&patient_id=${{patientId}}`);
        const health = await getJson(`/api/v1/stream/health?tenant_id=${{tenantId}}`);
        const channels = await getJson(`/api/v1/stream/channels?tenant_id=${{tenantId}}&patient_id=${{patientId}}`);

        setText("metricPatients", overview.patient_count ?? 0);
        setText("metricAlerts", overview.open_alerts ?? 0);
        setText("metricCritical", overview.critical_alerts ?? 0);
        setText("metricRisk", overview.avg_risk_score ?? 0);

        renderAlerts(live.alerts || []);
        renderFocus(live.focus_patient || {{}});
        renderStreamHealth(health || {{}});
        renderChannels(channels || {{}});
      }} catch (err) {{
        console.error(err);
      }}
    }}

    refreshDashboard();
    setInterval(refreshDashboard, 5000);
  </script>
<!-- ERA_METRIC_GUARDRAIL_CLARITY_BODY_START -->
<script src="/era-static/era_metric_guardrail_clarity.js?v=guardrail1"></script>
<!-- ERA_METRIC_GUARDRAIL_CLARITY_BODY_END -->
<!-- ERA_COMMAND_QUEUE_CONSISTENCY_BODY_START -->
<script src="/era-static/era_command_queue_consistency.js?v=queue-consistency1"></script>
<!-- ERA_COMMAND_QUEUE_CONSISTENCY_BODY_END -->
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
    body{margin:0;font-family:Inter,Arial,sans-serif;background:#08111f;color:#edf4ff}
    .wrap{max-width:1100px;margin:0 auto;padding:40px 20px}
    .card{background:#101a2d;border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:28px}
    h1{font-size:48px;line-height:1;margin:0 0 14px}
    p{color:#bdd0ec;line-height:1.7}
    .btn{display:inline-block;padding:13px 18px;border-radius:14px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#08111f;font-weight:900;text-decoration:none}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Investor Overview</h1>
      <p>
        Early Risk Alert AI is a professional predictive healthcare platform built for hospitals,
        clinics, command centers, and modern remote monitoring operations.
      </p>
      <p>
        The platform combines AI risk scoring, a live hospital-facing dashboard, product demonstration flow,
        and investor-ready presentation architecture in one branded software experience.
      </p>
      <p><a class="btn" href="/deck">Download Pitch Deck</a></p>
    </div>
  </div>
</body>
</html>
"""


@web_bp.get("/")
def home():
    return render_template_string(MAIN_HTML)


@web_bp.get("/investors")
def investors():
    return render_template_string(INVESTOR_HTML)


@web_bp.get("/login")
def login():
    return render_template_string("<h1 style='font-family:Arial;padding:40px'>Login page placeholder</h1>")


@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(MAIN_HTML)


@web_bp.get("/deck")
def deck():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_path = os.path.join(root_dir, "static", "Early_Risk_Alert_AI_Pitch_Deck.pdf")
    if not os.path.exists(pdf_path):
        return "Pitch deck not found", 404
    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf",
    )
