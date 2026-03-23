COMMAND_CENTER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Clinical Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --bg2:#0b1528;
      --panel:#101a2d;
      --panel2:#13203a;
      --panel3:#0d1728;
      --line:rgba(255,255,255,.08);
      --line2:rgba(255,255,255,.05);
      --text:#eef4ff;
      --muted:#a7bddc;
      --blue:#7aa2ff;
      --cyan:#5bd4ff;
      --green:#3ad38f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --purple:#b58cff;
      --shadow:0 20px 60px rgba(0,0,0,.34);
      --radius:22px;
      --brand1:#7aa2ff;
      --brand2:#5bd4ff;
      --brandText:#07101c;
    }

    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.14), transparent 22%),
        radial-gradient(circle at 88% 10%, rgba(91,212,255,.10), transparent 18%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }
    a{text-decoration:none;color:inherit}
    .shell{max-width:1480px;margin:0 auto;padding:18px 16px 56px}

    .topbar{
      position:sticky;top:0;z-index:60;
      border-bottom:1px solid var(--line);
      background:rgba(7,16,28,.88);
      backdrop-filter:blur(16px);
    }
    .topbar-inner{
      max-width:1480px;margin:0 auto;padding:14px 16px;
      display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;
    }
    .brand-title{
      font-size:clamp(28px,3vw,42px);font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .brand-sub{
      font-size:14px;font-weight:800;color:var(--muted);
    }
    .nav-links{
      display:flex;gap:14px;flex-wrap:wrap;align-items:center;
    }
    .nav-links a{
      font-size:14px;font-weight:900;color:#dce9ff;
    }

    .hero{
      margin-top:18px;
      border:1px solid var(--line);
      border-radius:28px;
      background:
        radial-gradient(circle at 20% 20%, rgba(91,212,255,.08), transparent 18%),
        radial-gradient(circle at 80% 0%, rgba(181,140,255,.08), transparent 18%),
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015)),
        linear-gradient(180deg, rgba(10,19,34,.86), rgba(7,14,26,.96));
      box-shadow:var(--shadow);
      padding:20px;
    }
    .hero-grid{
      display:grid;grid-template-columns:1.2fr .8fr;gap:18px;align-items:stretch;
    }
    .hero-copy,.hero-side{
      border:1px solid var(--line);
      border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:20px;
    }
    .hero-kicker{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#91ddff;margin-bottom:10px;
    }
    .hero-copy h1{
      margin:0 0 12px;font-size:clamp(34px,4vw,62px);line-height:.92;letter-spacing:-.06em;font-weight:1000;
    }
    .hero-copy p,.hero-side p{
      margin:0;color:#d0def2;font-size:16px;line-height:1.68;
    }
    .hero-actions{
      display:flex;gap:12px;flex-wrap:wrap;margin-top:18px;
    }

    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      border:1px solid transparent;cursor:pointer;
      transition:transform .18s ease, box-shadow .18s ease, opacity .18s ease, border-color .18s ease;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--brand1),var(--brand2));
      color:var(--brandText);
      box-shadow:0 12px 30px rgba(91,212,255,.22);
    }
    .btn.secondary{
      background:rgba(255,255,255,.04);border-color:var(--line);color:var(--text);
    }
    .btn.small{padding:9px 12px;border-radius:12px;font-size:12px}
    .btn.live-btn{background:linear-gradient(135deg,#3ad38f,#8ff3c1);color:#08111f}
    .btn.warn-btn{background:linear-gradient(135deg,#f4bd6a,#ffe09b);color:#2a1b00}
    .btn.critical-btn{background:linear-gradient(135deg,#ff667d,#ff97aa);color:#08111f}

    .pill-row{
      display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:18px;
    }
    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:10px 14px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.05);
    }
    .status-pill.live{color:#0b1528;background:linear-gradient(135deg,var(--green),#7ef5c0)}
    .status-pill.watch{color:#2a1b00;background:linear-gradient(135deg,var(--amber),#ffdba5)}
    .status-pill.critical{color:#fff3f5;background:linear-gradient(135deg,#ff667d,#ff8e99)}
    .status-pill.info{color:#07101c;background:linear-gradient(135deg,var(--brand1),var(--brand2))}
    .status-pill.muted{color:#dce9ff;background:rgba(255,255,255,.06)}

    .toolbar{
      display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:18px;
    }
    .toolbar select{
      background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:14px;color:var(--text);
      padding:12px 14px;font:inherit;font-weight:800;
    }

    .section{
      margin-top:18px;
    }
    .section-card{
      border:1px solid var(--line);
      border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 14px 36px rgba(0,0,0,.26);
      padding:18px;
    }
    .section-head{
      display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:14px;
    }
    .section-title{
      font-size:28px;font-weight:1000;letter-spacing:-.04em;margin:0;
    }
    .section-sub{
      font-size:14px;color:var(--muted);line-height:1.55;margin:6px 0 0;
    }

    .telemetry-top{
      display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px;
    }
    .stat-card{
      border:1px solid var(--line);border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(13,23,40,.90), rgba(8,15,28,.95));
      padding:16px;min-height:122px;position:relative;overflow:hidden;
    }
    .stat-card .k{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8ddcff;margin-bottom:10px;
    }
    .stat-card .v{
      font-size:42px;font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .stat-card .hint{
      margin-top:8px;font-size:13px;line-height:1.5;color:#c8d8ef;
    }

    .command-grid{
      display:grid;grid-template-columns:1.5fr .95fr;gap:18px;align-items:start;
    }
    .monitor-grid{
      display:grid;grid-template-columns:1fr 1fr;gap:16px;
    }
    .monitor{
      border:1px solid var(--line);border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.012)),
        linear-gradient(180deg, rgba(11,20,35,.96), rgba(6,12,22,.98));
      box-shadow:0 16px 44px rgba(0,0,0,.32);
      padding:18px;position:relative;overflow:hidden;min-height:370px;
    }
    .monitor::before{
      content:"";position:absolute;inset:0;
      background:
        repeating-linear-gradient(to right, rgba(255,255,255,.03) 0, rgba(255,255,255,.03) 1px, transparent 1px, transparent 58px),
        repeating-linear-gradient(to bottom, rgba(255,255,255,.025) 0, rgba(255,255,255,.025) 1px, transparent 1px, transparent 40px);
      opacity:.12;pointer-events:none;
    }
    .monitor-top{
      position:relative;z-index:1;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px;
    }
    .monitor-bed{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#95ddff;margin-bottom:6px;
    }
    .monitor-title{
      font-size:28px;font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .monitor-wave{
      position:relative;z-index:1;border:1px solid rgba(255,255,255,.06);border-radius:20px;min-height:138px;
      background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01)), rgba(0,0,0,.18);
      overflow:hidden;display:flex;align-items:center;padding:8px 0;
      cursor:pointer;
    }
    .monitor-wave svg{width:100%;height:118px;display:block}
    .ecg-path{
      fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:920;stroke-dashoffset:0;
      animation:ecgMove 2.8s linear infinite;filter:drop-shadow(0 0 10px currentColor);
    }
    @keyframes ecgMove{0%{stroke-dashoffset:920}100%{stroke-dashoffset:0}}
    .ecg-green{stroke:#77ffb4;color:#77ffb4}
    .ecg-amber{stroke:#ffd96c;color:#ffd96c}
    .ecg-red{stroke:#ff7c8d;color:#ff7c8d}

    .monitor-metrics{
      position:relative;z-index:1;margin-top:14px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
    }
    .metric-box{
      border:1px solid rgba(255,255,255,.06);border-radius:16px;background:rgba(255,255,255,.03);
      padding:12px;min-height:78px;display:flex;flex-direction:column;justify-content:space-between;
    }
    .metric-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb8dc}
    .metric-v{font-size:18px;font-weight:1000;line-height:1;letter-spacing:-.03em}

    .monitor-story{
      position:relative;z-index:1;margin-top:14px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
    }
    .story-text{
      font-size:13px;color:#cfe0f4;line-height:1.55;max-width:74%;
    }
    .monitor-actions{
      position:relative;z-index:1;margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;
    }

    .side-stack{display:grid;gap:16px}
    .intel-card{
      border:1px solid var(--line);border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;box-shadow:0 14px 36px rgba(0,0,0,.26);
    }
    .intel-card h3{
      margin:0 0 12px;font-size:20px;font-weight:1000;letter-spacing:-.03em;
    }

    .alert-feed,.queue-list,.timeline-panel,.audit-list,.why-now{
      display:grid;gap:10px;
    }
    .alert-item,.queue-item,.timeline-item,.audit-item,.why-line{
      border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:12px;background:rgba(255,255,255,.03);
    }
    .alert-item,.queue-item,.audit-item{
      display:flex;align-items:flex-start;justify-content:space-between;gap:10px;
    }
    .alert-copy,.queue-copy,.audit-copy{
      font-size:14px;font-weight:900;line-height:1.45;color:#e4efff;
    }
    .alert-sub,.audit-sub{
      font-size:12px;color:#acc0dd;font-weight:700;margin-top:4px;
    }

    .timeline-item{
      display:grid;grid-template-columns:78px 1fr;gap:12px;align-items:start;
    }
    .timeline-time{
      font-size:12px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;color:#8fdcff;
    }
    .timeline-copy{
      font-size:13px;line-height:1.55;color:#e4efff;
    }

    .detail-drawer{
      position:fixed;top:0;right:-460px;width:440px;max-width:100%;height:100vh;z-index:120;
      background:linear-gradient(180deg, rgba(12,22,38,.98), rgba(7,14,26,.99));
      border-left:1px solid var(--line);box-shadow:-20px 0 60px rgba(0,0,0,.35);
      transition:right .25s ease;padding:18px;overflow:auto;
    }
    .detail-drawer.open{right:0}
    .drawer-top{
      display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:16px;
    }
    .drawer-title{
      font-size:28px;font-weight:1000;letter-spacing:-.04em;margin:0;
    }
    .drawer-sub{
      font-size:13px;color:var(--muted);line-height:1.5;margin-top:6px;
    }
    .drawer-block{
      border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);margin-bottom:12px;
    }
    .drawer-k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb8dc;margin-bottom:8px;
    }
    .drawer-v{
      font-size:18px;font-weight:1000;line-height:1.4;color:#eef4ff;
    }
    .drawer-grid{
      display:grid;grid-template-columns:1fr 1fr;gap:10px;
    }
    .overlay{
      position:fixed;inset:0;background:rgba(0,0,0,.42);z-index:110;display:none;
    }
    .overlay.show{display:block}

    .footer{
      margin-top:18px;border:1px solid var(--line);border-radius:22px;padding:18px;
      background:rgba(255,255,255,.03);color:#a7bddc;font-size:13px;line-height:1.6;text-align:center;
    }

    @media (max-width:1280px){
      .hero-grid,.command-grid{grid-template-columns:1fr}
    }
    @media (max-width:980px){
      .telemetry-top{grid-template-columns:repeat(2,1fr)}
      .monitor-grid{grid-template-columns:1fr}
      .drawer-grid{grid-template-columns:1fr}
    }
    @media (max-width:700px){
      .shell{padding:14px 10px 40px}
      .hero{padding:16px}
      .hero-copy h1{font-size:clamp(32px,11vw,52px)}
      .btn{width:100%}
      .telemetry-top,.monitor-metrics{grid-template-columns:1fr}
      .story-text{max-width:100%}
      .timeline-item{grid-template-columns:1fr}
      .detail-drawer{width:100%}
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="brand-kicker" id="brandKicker">AI-powered predictive clinical intelligence</div>
        <div class="brand-title" id="brandTitle">Early Risk Alert AI</div>
        <div class="brand-sub" id="brandSub">Clinical Command Center</div>
      </div>
      <div class="nav-links">
        <a href="/command-center">Command Center</a>
        <a href="/hospital-demo">Hospital Demo</a>
        <a href="/investor-intake">Investor Access</a>
        <a href="/executive-walkthrough">Executive Walkthrough</a>
        <a href="/admin/review">Admin Review</a>
        <a href="/logout">Logout</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">Pilot operating layer</div>
          <h1 id="heroTitle">Real-time clinical command center</h1>
          <p id="heroCopy">
            Early Risk Alert AI surfaces high-risk patients earlier, prioritizes deterioration patterns in real time,
            and gives care teams a structured workflow to acknowledge, assign, escalate, and resolve clinical alerts.
          </p>

          <div class="hero-actions">
            <a class="btn primary" href="/hospital-demo">Request Live Demo</a>
            <a class="btn secondary" href="/investor-intake">Investor Access</a>
            <a class="btn secondary" href="/admin/review">Open Admin Review</a>
          </div>

          <div class="pill-row">
            <div class="status-pill watch" id="pilotModePill">Pilot Mode</div>
            <div class="status-pill info" id="currentRolePill">Role</div>
            <div class="status-pill muted" id="currentUserPill">User</div>
            <div class="status-pill info" id="unitAccessPill">Unit Access</div>
            <div class="status-pill live" id="feedHealthPill">Feed Live</div>
            <div class="status-pill muted" id="lastUpdatedPill">Last Updated</div>
          </div>
        </div>

        <div class="hero-side">
          <div class="hero-kicker">Access + scope</div>
          <p>
            Hospital pilot users can be restricted to a single unit such as ICU, telemetry, stepdown, ward, or RPM.
            Admin users retain full hospital-wide visibility across all units and all pilot accounts.
          </p>

          <div class="toolbar">
            <select id="unitFilter">
              <option value="all">All Units</option>
              <option value="icu">ICU</option>
              <option value="telemetry">Telemetry</option>
              <option value="stepdown">Stepdown</option>
              <option value="ward">Ward</option>
              <option value="rpm">RPM / Home</option>
            </select>
            <div class="status-pill info" id="selectedUnitPill">All Units</div>
            <div class="status-pill muted" id="hospitalBrandPill">Pilot Account</div>
          </div>

          <div class="pill-row">
            <div class="status-pill critical" id="systemBannerPill">Restricted</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Hospital Command Wall</h2>
            <div class="section-sub">Real-time telemetry, AI risk scoring, explainable alerts, workflow actions, audit trail, and pilot unit access control.</div>
          </div>
          <div class="status-pill live" id="wallStatus">Live</div>
        </div>

        <div class="telemetry-top">
          <div class="stat-card">
            <div class="k">Open Alerts</div>
            <div class="v" id="open-alerts">0</div>
            <div class="hint">Active alert count within your current access scope.</div>
          </div>
          <div class="stat-card">
            <div class="k">Critical Alerts</div>
            <div class="v" id="critical-alerts">0</div>
            <div class="hint">Highest urgency patients needing immediate attention.</div>
          </div>
          <div class="stat-card">
            <div class="k">Avg Risk Score</div>
            <div class="v" id="avg-risk">0.0</div>
            <div class="hint">Average risk across visible patients.</div>
          </div>
          <div class="stat-card">
            <div class="k">Visible Unit</div>
            <div class="v" id="unit-count">All</div>
            <div class="hint">Current pilot or admin command center scope.</div>
          </div>
        </div>

        <div class="command-grid">
          <div class="monitor-grid" id="wall"></div>

          <div class="side-stack">
            <div class="intel-card">
              <h3>Live Alerts Feed</h3>
              <div class="alert-feed" id="queue"></div>
            </div>

            <div class="intel-card">
              <h3>Top Risk Queue</h3>
              <div class="queue-list" id="top-risk-list"></div>
            </div>

            <div class="intel-card">
              <h3>AI Clinical Reasoning</h3>
              <div class="why-now" id="ai-reasoning-panel">
                <div class="why-line">Waiting for patient intelligence stream...</div>
              </div>
            </div>

            <div class="intel-card">
              <h3>Workflow Snapshot</h3>
              <div class="queue-list">
                <div class="queue-item">
                  <div class="queue-copy">New Alerts <strong id="wf-new">0</strong></div>
                  <div class="status-pill watch">Open</div>
                </div>
                <div class="queue-item">
                  <div class="queue-copy">Acknowledged <strong id="wf-ack">0</strong></div>
                  <div class="status-pill live">Tracked</div>
                </div>
                <div class="queue-item">
                  <div class="queue-copy">Assigned <strong id="wf-assigned">0</strong></div>
                  <div class="status-pill info">Assigned</div>
                </div>
                <div class="queue-item">
                  <div class="queue-copy">Escalated <strong id="wf-escalated">0</strong></div>
                  <div class="status-pill critical">Priority</div>
                </div>
                <div class="queue-item">
                  <div class="queue-copy">Resolved <strong id="wf-resolved">0</strong></div>
                  <div class="status-pill live">Closed</div>
                </div>
              </div>
            </div>

            <div class="intel-card">
              <h3>System Health</h3>
              <div class="queue-list" id="system-health-list"></div>
            </div>

            <div class="intel-card">
              <h3>Patient Timeline</h3>
              <div class="timeline-panel" id="patient-timeline"></div>
            </div>

            <div class="intel-card">
              <h3>Audit Trail</h3>
              <div class="audit-list" id="audit-log"></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI · Real-Time Clinical Command Center · Unit-Based Pilot Access · Hospital Branded Accounts · Explainable Alert Workflow
    </div>
  </div>

  <div class="overlay" id="drawerOverlay" onclick="closePatientDrawer()"></div>

  <aside class="detail-drawer" id="patientDrawer">
    <div class="drawer-top">
      <div>
        <h3 class="drawer-title" id="drawerPatientName">Patient</h3>
        <div class="drawer-sub" id="drawerPatientSub">Patient details</div>
      </div>
      <button class="btn secondary small" onclick="closePatientDrawer()">Close</button>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Clinical Summary</div>
      <div class="drawer-v" id="drawerSummary">Waiting for patient selection.</div>
    </div>

    <div class="drawer-grid">
      <div class="drawer-block">
        <div class="drawer-k">Heart Rate</div>
        <div class="drawer-v" id="drawerHr">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">SpO₂</div>
        <div class="drawer-v" id="drawerSpo2">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Blood Pressure</div>
        <div class="drawer-v" id="drawerBp">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Risk Score</div>
        <div class="drawer-v" id="drawerRisk">--</div>
      </div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Recommended Action</div>
      <div class="drawer-v" id="drawerAction">--</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Explainable Reasoning</div>
      <div class="drawer-v" id="drawerExplainability">--</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Workflow Status</div>
      <div class="drawer-v" id="drawerWorkflow">No workflow action saved yet.</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Timeline Snapshot</div>
      <div class="timeline-panel" id="drawerTimeline"></div>
    </div>
  </aside>

  <script>
    let activePatients = [];
    let activeAlerts = [];
    let workflowState = {};
    let auditLog = [];
    let currentUnitFilter = "all";
    let selectedPatientId = null;

    let accessRole = "viewer";
    let accessAssignedUnit = "all";
    let canViewAllUnits = true;
    let accessHospital = "Pilot";
    let brandName = "Early Risk Alert AI";
    let brandTagline = "Clinical Command Center";
    let brand1 = "#7aa2ff";
    let brand2 = "#5bd4ff";
    let pilotModeEnabled = true;
    let lastSystemHealth = {};
    let lastUpdatedAt = null;

    function safe(v, fallback="--"){
      return v === undefined || v === null || v === "" ? fallback : v;
    }

    function unitLabel(unit){
      unit = String(unit || "all").toLowerCase();
      if (unit === "icu") return "ICU";
      if (unit === "telemetry") return "Telemetry";
      if (unit === "stepdown") return "Stepdown";
      if (unit === "ward") return "Ward";
      if (unit === "rpm") return "RPM / Home";
      return "All Units";
    }

    function statusClass(status){
      const s = String(status || "").toLowerCase();
      if (s === "critical" || s === "escalated") return "critical";
      if (s === "high" || s === "moderate" || s === "acknowledged" || s === "assigned") return "watch";
      return "live";
    }

    function pulseLabel(status){
      const s = String(status || "").toLowerCase();
      if (s === "critical") return "Critical";
      if (s === "high") return "High";
      if (s === "moderate") return "Moderate";
      if (s === "acknowledged") return "Acknowledged";
      if (s === "assigned") return "Assigned";
      if (s === "escalated") return "Escalated";
      if (s === "resolved") return "Resolved";
      return "Stable";
    }

    function ecgClass(status){
      const s = String(status || "").toLowerCase();
      if (s === "critical" || s === "escalated") return "ecg-red";
      if (s === "high" || s === "moderate" || s === "acknowledged" || s === "assigned") return "ecg-amber";
      return "ecg-green";
    }

    function roomToUnit(room){
      const r = String(room || "").toLowerCase();
      if (r.includes("icu")) return "icu";
      if (r.includes("stepdown")) return "stepdown";
      if (r.includes("telemetry")) return "telemetry";
      if (r.includes("ward")) return "ward";
      if (r.includes("rpm") || r.includes("home")) return "rpm";
      return "telemetry";
    }

    function buildPath(points){
      return points.map((p, i) => (i === 0 ? "M" : "L") + p[0] + "," + p[1]).join(" ");
    }

    function waveformPoints(mode){
      if (mode === "critical") {
        return [[0,72],[30,72],[46,72],[58,70],[72,72],[86,72],[98,50],[106,100],[116,24],[126,108],[136,72],[162,72],[186,72],[204,70],[220,72],[236,72],[250,54],[258,96],[268,22],[278,106],[290,72],[318,72],[336,72],[352,70],[368,72],[382,72],[398,46],[406,104],[416,20],[428,110],[440,72],[466,72],[486,72],[504,70],[520,72],[536,72],[550,52],[558,98],[568,24],[578,106],[592,72],[620,72]];
      }
      if (mode === "high" || mode === "moderate") {
        return [[0,76],[36,76],[60,76],[74,74],[88,76],[104,76],[118,56],[126,90],[136,38],[146,96],[158,76],[190,76],[214,76],[228,74],[242,76],[258,76],[272,54],[280,88],[290,40],[300,94],[314,76],[348,76],[372,76],[388,74],[402,76],[418,76],[432,58],[440,92],[452,42],[464,96],[478,76],[514,76],[538,76],[552,74],[566,76],[582,76],[596,60],[604,94],[614,44],[624,98],[640,76]];
      }
      return [[0,78],[48,78],[82,78],[96,76],[110,78],[126,78],[138,66],[146,84],[156,52],[166,88],[178,78],[220,78],[254,78],[268,76],[282,78],[298,78],[310,68],[318,84],[328,56],[338,88],[350,78],[392,78],[426,78],[440,76],[454,78],[470,78],[482,68],[490,84],[500,56],[510,88],[522,78],[566,78],[600,78]];
    }

    function explainabilityForPatient(patient){
      const reasons = [];
      const spo2 = Number(patient.spo2 || 0);
      const hr = Number(patient.heart_rate || 0);
      const sbp = Number(patient.bp_systolic || 0);

      if (spo2 && spo2 < 90) reasons.push("SpO₂ below critical threshold");
      else if (spo2 && spo2 < 94) reasons.push("SpO₂ trending below target");
      if (hr && hr >= 120) reasons.push("heart rate elevated");
      if (sbp && sbp >= 160) reasons.push("blood pressure elevated");
      if (!reasons.length) reasons.push("combined signal pattern indicates monitored risk");
      return reasons.join(", ") + ".";
    }

    function normalizePatient(raw){
      if (!raw) return null;
      const vitals = raw.vitals || {};
      const risk = raw.risk || {};
      const room = raw.room || raw.bed || "Unit";
      return {
        patient_id: raw.patient_id || "p---",
        name: raw.patient_name || raw.name || "Patient",
        bed: room,
        unit: roomToUnit(room),
        program: raw.program || "Clinical Monitoring",
        title: risk.alert_message || raw.title || "Predictive Risk Monitor",
        heart_rate: raw.heart_rate ?? vitals.heart_rate ?? "--",
        spo2: raw.spo2 ?? vitals.spo2 ?? "--",
        bp_systolic: raw.bp_systolic ?? vitals.systolic_bp ?? "--",
        bp_diastolic: raw.bp_diastolic ?? vitals.diastolic_bp ?? "--",
        risk_score: raw.risk_score ?? risk.risk_score ?? "--",
        status: raw.status || risk.severity || "Stable",
        story: raw.story || risk.recommended_action || "Predictive monitoring active.",
        alert_message: risk.alert_message || "Vitals stable",
        recommended_action: risk.recommended_action || "Continue routine monitoring."
      };
    }

    function normalizeAlert(alert){
      if (!alert) return null;
      return {
        patient_id: alert.patient_id || "Patient",
        severity: alert.severity || "Stable",
        text: alert.message || alert.text || "Clinical alert surfaced",
        unit: alert.room || alert.unit || "Unit"
      };
    }

    function filterPatientsByUnit(patients){
      const source = (patients || []).map(normalizePatient).filter(Boolean);
      if (currentUnitFilter === "all") return source;
      return source.filter(p => p.unit === currentUnitFilter);
    }

    function findPatient(patientId){
      return activePatients.find(p => String(p.patient_id) === String(patientId)) || null;
    }

    function getWorkflowRecord(patientId){
      return workflowState[String(patientId)] || {
        ack: false,
        assigned: false,
        escalated: false,
        resolved: false,
        assigned_label: "",
        state: "new",
        updated_at: "",
        role: ""
      };
    }

    function workflowText(patientId){
      const record = getWorkflowRecord(patientId);
      const parts = [];
      if (record.state) parts.push(record.state);
      if (record.role) parts.push("by " + record.role);
      if (record.updated_at) parts.push(new Date(record.updated_at).toLocaleTimeString());
      return parts.join(" · ");
    }

    function applyBranding(data){
      brandName = safe(data.brand_name, "Early Risk Alert AI");
      brandTagline = safe(data.brand_tagline, "Clinical Command Center");
      brand1 = safe(data.brand_primary, "#7aa2ff");
      brand2 = safe(data.brand_secondary, "#5bd4ff");

      document.documentElement.style.setProperty("--brand1", brand1);
      document.documentElement.style.setProperty("--brand2", brand2);

      const brandTitle = document.getElementById("brandTitle");
      const brandSub = document.getElementById("brandSub");
      const heroTitle = document.getElementById("heroTitle");
      const hospitalBrandPill = document.getElementById("hospitalBrandPill");

      if (brandTitle) brandTitle.textContent = brandName;
      if (brandSub) brandSub.textContent = brandTagline;
      if (heroTitle) heroTitle.textContent = brandName + " Command Center";
      if (hospitalBrandPill) hospitalBrandPill.textContent = accessHospital;
    }

    async function loadAccessContext(){
      try{
        const res = await fetch("/api/access-context", {cache:"no-store"});
        const data = await res.json();

        accessRole = String(data.role || "viewer").toLowerCase();
        accessAssignedUnit = String(data.assigned_unit || "all").toLowerCase();
        canViewAllUnits = !!data.can_view_all_units;
        accessHospital = safe(data.hospital_name, "Pilot");
        pilotModeEnabled = !!data.pilot_mode;

        applyBranding(data);

        const unitFilter = document.getElementById("unitFilter");
        const selectedUnitPill = document.getElementById("selectedUnitPill");
        const unitAccessPill = document.getElementById("unitAccessPill");
        const currentRolePill = document.getElementById("currentRolePill");
        const currentUserPill = document.getElementById("currentUserPill");
        const pilotModePill = document.getElementById("pilotModePill");
        const systemBannerPill = document.getElementById("systemBannerPill");

        if (unitFilter) {
          if (!canViewAllUnits) {
            unitFilter.value = accessAssignedUnit;
            unitFilter.disabled = true;
            currentUnitFilter = accessAssignedUnit;
          } else {
            currentUnitFilter = unitFilter.value || "all";
          }
        }

        if (selectedUnitPill) selectedUnitPill.textContent = unitLabel(currentUnitFilter);
        if (unitAccessPill) unitAccessPill.textContent = "Unit Access: " + unitLabel(accessAssignedUnit);
        if (currentRolePill) currentRolePill.textContent = "Role: " + accessRole;
        if (currentUserPill) currentUserPill.textContent = safe(data.user_name, "Pilot User");
        if (pilotModePill) pilotModePill.textContent = pilotModeEnabled ? "Pilot Mode" : "Restricted";
        if (systemBannerPill) systemBannerPill.textContent = pilotModeEnabled ? "Pilot Active" : "Pilot Off";
      }catch(err){
        console.error("access context failed", err);
      }
    }

    async function loadWorkflowState(){
      try{
        const res = await fetch("/api/workflow", {cache:"no-store"});
        const payload = await res.json();
        workflowState = payload.records || {};
        auditLog = payload.audit_log || [];
      }catch(err){
        console.error("workflow load failed", err);
      }
    }

    async function loadSystemHealth(){
      try{
        const res = await fetch("/api/system-health", {cache:"no-store"});
        lastSystemHealth = await res.json();
      }catch(err){
        console.error("system health failed", err);
      }
    }

    async function refreshSnapshot(){
      try{
        const res = await fetch("/api/v1/live-snapshot?refresh=" + Date.now(), {cache:"no-store"});
        const payload = await res.json();
        activePatients = (payload.patients || []).map(normalizePatient).filter(Boolean);
        activeAlerts = (payload.alerts || []).map(normalizeAlert).filter(Boolean);
        lastUpdatedAt = payload.generated_at || new Date().toISOString();
        rerenderAll();
      }catch(err){
        console.error("snapshot failed", err);
      }
    }

    async function postWorkflowAction(patientId, action, note=""){
      try{
        const res = await fetch("/api/workflow/action", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({
            patient_id: patientId,
            action: action,
            note: note
          })
        });
        const payload = await res.json();
        if (!res.ok || !payload.ok){
          alert(payload.error || "Action not allowed");
          return false;
        }
        await loadWorkflowState();
        return true;
      }catch(err){
        console.error("workflow action failed", err);
        alert("Unable to save workflow action.");
        return false;
      }
    }

    async function ackPatient(patientId){
      const ok = await postWorkflowAction(patientId, "ack");
      if (!ok) return;
      rerenderAll();
      if (selectedPatientId === patientId) openPatientDrawer(patientId);
    }

    async function assignPatient(patientId){
      const ok = await postWorkflowAction(patientId, "assign_nurse", "Assigned Nurse");
      if (!ok) return;
      rerenderAll();
      if (selectedPatientId === patientId) openPatientDrawer(patientId);
    }

    async function escalatePatient(patientId){
      const ok = await postWorkflowAction(patientId, "escalate");
      if (!ok) return;
      rerenderAll();
      if (selectedPatientId === patientId) openPatientDrawer(patientId);
    }

    async function resolvePatient(patientId){
      const ok = await postWorkflowAction(patientId, "resolve");
      if (!ok) return;
      rerenderAll();
      if (selectedPatientId === patientId) openPatientDrawer(patientId);
    }

    function openPatientDrawer(patientId){
      const patient = findPatient(patientId);
      if (!patient) return;
      selectedPatientId = patientId;

      document.getElementById("drawerPatientName").textContent = safe(patient.name);
      document.getElementById("drawerPatientSub").textContent = `${safe(patient.patient_id)} · ${safe(patient.bed)} · ${safe(patient.program)}`;
      document.getElementById("drawerSummary").textContent = safe(patient.alert_message);
      document.getElementById("drawerHr").textContent = safe(patient.heart_rate);
      document.getElementById("drawerSpo2").textContent = safe(patient.spo2);
      document.getElementById("drawerBp").textContent = `${safe(patient.bp_systolic)}/${safe(patient.bp_diastolic)}`;
      document.getElementById("drawerRisk").textContent = typeof patient.risk_score === "number" ? patient.risk_score.toFixed(1) : safe(patient.risk_score);
      document.getElementById("drawerAction").textContent = safe(patient.recommended_action);
      document.getElementById("drawerExplainability").textContent = explainabilityForPatient(patient);
      document.getElementById("drawerWorkflow").textContent = workflowText(patientId);

      document.getElementById("drawerTimeline").innerHTML = `
        <div class="timeline-item">
          <div class="timeline-time">Now</div>
          <div class="timeline-copy">${safe(patient.name)} is currently ${safe(patient.status)} with risk ${typeof patient.risk_score === "number" ? patient.risk_score.toFixed(1) : safe(patient.risk_score)}.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">-15 min</div>
          <div class="timeline-copy">Trend drift detected across oxygen saturation and heart rate signals.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">-30 min</div>
          <div class="timeline-copy">Predictive model increased deterioration probability before standard threshold alarm.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">Action</div>
          <div class="timeline-copy">${safe(patient.recommended_action)}</div>
        </div>
      `;

      document.getElementById("patientDrawer").classList.add("open");
      document.getElementById("drawerOverlay").classList.add("show");
    }

    function closePatientDrawer(){
      document.getElementById("patientDrawer").classList.remove("open");
      document.getElementById("drawerOverlay").classList.remove("show");
    }

    function renderActionButtons(record, pid){
      const viewOnly = accessRole === "viewer";
      const ackDisabled = viewOnly ? "disabled" : "";
      const assignDisabled = (accessRole !== "admin" && accessRole !== "operator" && accessRole !== "physician") ? "disabled" : "";
      const escalateDisabled = (accessRole !== "admin" && accessRole !== "physician") ? "disabled" : "";
      const resolveDisabled = (accessRole !== "admin" && accessRole !== "physician") ? "disabled" : "";

      return `
        <button class="btn ${record.ack ? 'live-btn' : 'secondary'} small" onclick="ackPatient('${pid}')" ${ackDisabled}>${record.ack ? 'ACK Saved' : 'ACK'}</button>
        <button class="btn ${record.assigned ? 'warn-btn' : 'secondary'} small" onclick="assignPatient('${pid}')" ${assignDisabled}>${record.assigned ? 'Assigned' : 'Assign'}</button>
        <button class="btn ${record.escalated ? 'critical-btn' : 'secondary'} small" onclick="escalatePatient('${pid}')" ${escalateDisabled}>${record.escalated ? 'Escalated' : 'Escalate'}</button>
        <button class="btn ${record.resolved ? 'live-btn' : 'secondary'} small" onclick="resolvePatient('${pid}')" ${resolveDisabled}>${record.resolved ? 'Resolved' : 'Resolve'}</button>
        <button class="btn secondary small" onclick="openPatientDrawer('${pid}')">Details</button>
      `;
    }

    function renderMonitor(patient){
      const status = String(patient.status || "").toLowerCase();
      const ecg = ecgClass(status);
      const path = buildPath(waveformPoints(status === "critical" ? "critical" : (status === "high" || status === "moderate") ? "high" : "stable"));
      const riskText = typeof patient.risk_score === "number" ? patient.risk_score.toFixed(1) : safe(patient.risk_score);
      const pid = safe(patient.patient_id);
      const record = getWorkflowRecord(pid);

      return `
        <div class="monitor">
          <div class="monitor-top">
            <div>
              <div class="monitor-bed">${safe(patient.bed)}</div>
              <div class="monitor-title">${safe(patient.title)}</div>
            </div>
            <div class="status-pill ${statusClass(record.state !== 'new' ? record.state : patient.status)}">${pulseLabel(record.state !== 'new' ? record.state : patient.status)}</div>
          </div>

          <div class="monitor-wave" onclick="openPatientDrawer('${pid}')">
            <svg viewBox="0 0 640 120" preserveAspectRatio="none" aria-hidden="true">
              <path class="ecg-path ${ecg}" d="${path}"></path>
            </svg>
          </div>

          <div class="monitor-metrics">
            <div class="metric-box"><span class="metric-k">HR</span><span class="metric-v">${safe(patient.heart_rate)}</span></div>
            <div class="metric-box"><span class="metric-k">SpO₂</span><span class="metric-v">${safe(patient.spo2)}</span></div>
            <div class="metric-box"><span class="metric-k">BP</span><span class="metric-v">${safe(patient.bp_systolic)}/${safe(patient.bp_diastolic)}</span></div>
            <div class="metric-box"><span class="metric-k">Risk</span><span class="metric-v">${riskText}</span></div>
          </div>

          <div class="monitor-story">
            <div class="story-text">${safe(patient.story)}</div>
            <div class="status-pill ${statusClass(record.state !== 'new' ? record.state : patient.status)}">${pid}</div>
          </div>

          <div class="monitor-actions">
            ${renderActionButtons(record, pid)}
          </div>
        </div>
      `;
    }

    function renderAlert(alert){
      const sev = String(alert.severity || "").toLowerCase();
      const pill = sev === "critical" ? "critical" : (sev === "high" || sev === "moderate") ? "watch" : "live";
      return `
        <div class="alert-item">
          <div>
            <div class="alert-copy">${safe(alert.text)}</div>
            <div class="alert-sub">${safe(alert.patient_id)} · ${safe(alert.unit)} · ${safe(alert.severity)}</div>
          </div>
          <div class="status-pill ${pill}">${safe(alert.severity)}</div>
        </div>
      `;
    }

    function renderPatients(){
      const source = filterPatientsByUnit(activePatients)
        .sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))
        .slice(0,4);
      document.getElementById("wall").innerHTML = source.map(renderMonitor).join("") || `
        <div class="monitor">
          <div class="monitor-title">No patients visible in current access scope.</div>
        </div>
      `;
    }

    function renderAlertsList(){
      const source = activeAlerts.slice(0,6);
      document.getElementById("queue").innerHTML = source.map(renderAlert).join("") || `
        <div class="alert-item"><div><div class="alert-copy">No active alerts in current scope.</div></div><div class="status-pill live">Clear</div></div>
      `;
    }

    function renderTopRiskPatients(){
      const source = filterPatientsByUnit(activePatients)
        .sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))
        .slice(0,4);

      document.getElementById("top-risk-list").innerHTML = source.map(p => `
        <div class="queue-item">
          <div class="queue-copy">${safe(p.name)} · ${safe(p.patient_id)} · Risk ${typeof p.risk_score === "number" ? p.risk_score.toFixed(1) : safe(p.risk_score)}</div>
          <div class="status-pill ${statusClass(p.status)}">${pulseLabel(p.status)}</div>
        </div>
      `).join("") || `<div class="queue-item"><div class="queue-copy">No risk queue visible.</div><div class="status-pill muted">Scope</div></div>`;
    }

    function renderAIReasoning(){
      const source = filterPatientsByUnit(activePatients).sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));
      const top = source[0];
      const panel = document.getElementById("ai-reasoning-panel");

      if (!top){
        panel.innerHTML = `<div class="why-line">Waiting for patient intelligence stream...</div>`;
        return;
      }

      panel.innerHTML = `
        <div class="why-line">Highest-risk patient: ${safe(top.name)} (${safe(top.patient_id)})</div>
        <div class="why-line">Current severity: ${safe(top.status)} · Risk ${typeof top.risk_score === "number" ? top.risk_score.toFixed(1) : safe(top.risk_score)}</div>
        <div class="why-line">${safe(top.story)}</div>
        <div class="why-line">Explainable reason: ${explainabilityForPatient(top)}</div>
      `;
    }

    function renderWorkflow(){
      const records = Object.values(workflowState || {});
      document.getElementById("wf-new").textContent = String(activeAlerts.length);
      document.getElementById("wf-ack").textContent = String(records.filter(r => r.ack).length);
      document.getElementById("wf-assigned").textContent = String(records.filter(r => r.assigned).length);
      document.getElementById("wf-escalated").textContent = String(records.filter(r => r.escalated).length);
      document.getElementById("wf-resolved").textContent = String(records.filter(r => r.resolved).length);
    }

    function renderSystemHealth(){
      const list = document.getElementById("system-health-list");
      list.innerHTML = `
        <div class="queue-item"><div class="queue-copy">Hospital</div><div class="status-pill info">${safe(accessHospital)}</div></div>
        <div class="queue-item"><div class="queue-copy">Role</div><div class="status-pill info">${safe(accessRole)}</div></div>
        <div class="queue-item"><div class="queue-copy">Assigned Unit</div><div class="status-pill watch">${unitLabel(accessAssignedUnit)}</div></div>
        <div class="queue-item"><div class="queue-copy">System Status</div><div class="status-pill ${safe(lastSystemHealth.status) === 'ok' ? 'live' : 'critical'}">${safe(lastSystemHealth.status, 'unknown')}</div></div>
      `;
    }

    function renderPatientTimeline(){
      const source = filterPatientsByUnit(activePatients).sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));
      const top = source[0];
      const panel = document.getElementById("patient-timeline");

      if (!top){
        panel.innerHTML = `
          <div class="timeline-item">
            <div class="timeline-time">Now</div>
            <div class="timeline-copy">No patient timeline visible in current scope.</div>
          </div>
        `;
        return;
      }

      panel.innerHTML = `
        <div class="timeline-item">
          <div class="timeline-time">Now</div>
          <div class="timeline-copy">${safe(top.name)} flagged with ${safe(top.status)} severity. Risk ${typeof top.risk_score === "number" ? top.risk_score.toFixed(1) : safe(top.risk_score)}.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">-15 min</div>
          <div class="timeline-copy">Trend drift detected across oxygen saturation and heart rate signals.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">-30 min</div>
          <div class="timeline-copy">Predictive model increased deterioration probability before standard threshold alarm.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">Action</div>
          <div class="timeline-copy">${safe(top.recommended_action)}</div>
        </div>
      `;
    }

    function renderAuditLog(){
      const target = document.getElementById("audit-log");
      if (!auditLog.length){
        target.innerHTML = `
          <div class="audit-item">
            <div>
              <div class="audit-copy">System initialized</div>
              <div class="audit-sub">Awaiting user actions</div>
            </div>
            <div class="status-pill info">Log</div>
          </div>
        `;
        return;
      }

      target.innerHTML = auditLog.slice(-8).reverse().map(entry => `
        <div class="audit-item">
          <div>
            <div class="audit-copy">${safe(entry.action)} · ${safe(entry.patient_id)}</div>
            <div class="audit-sub">${safe(entry.role)} · ${safe(entry.note)} · ${new Date(entry.time).toLocaleTimeString()}</div>
          </div>
          <div class="status-pill info">Log</div>
        </div>
      `).join("");
    }

    function updateSummary(){
      const sourcePatients = filterPatientsByUnit(activePatients);
      const sourceAlerts = activeAlerts;
      const openAlerts = sourceAlerts.length;
      const criticalAlerts = sourceAlerts.filter(a => String(a.severity || "").toLowerCase() === "critical").length;
      const avgRisk = sourcePatients.length
        ? (sourcePatients.reduce((n, p) => n + Number(p.risk_score || 0), 0) / sourcePatients.length)
        : 0;

      document.getElementById("open-alerts").textContent = String(openAlerts);
      document.getElementById("critical-alerts").textContent = String(criticalAlerts);
      document.getElementById("avg-risk").textContent = avgRisk.toFixed(1);
      document.getElementById("unit-count").textContent = unitLabel(currentUnitFilter);

      const lastUpdatedPill = document.getElementById("lastUpdatedPill");
      if (lastUpdatedPill) {
        lastUpdatedPill.textContent = "Last Updated " + (lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleTimeString() : "--");
      }
    }

    function rerenderAll(){
      renderPatients();
      renderAlertsList();
      renderTopRiskPatients();
      renderAIReasoning();
      renderWorkflow();
      renderSystemHealth();
      renderPatientTimeline();
      renderAuditLog();
      updateSummary();
    }

    async function boot(){
      await loadAccessContext();
      await loadWorkflowState();
      await loadSystemHealth();
      await refreshSnapshot();

      const unitFilter = document.getElementById("unitFilter");
      if (unitFilter){
        unitFilter.addEventListener("change", function(){
          if (!canViewAllUnits){
            this.value = accessAssignedUnit;
            currentUnitFilter = accessAssignedUnit;
          } else {
            currentUnitFilter = this.value;
          }
          document.getElementById("selectedUnitPill").textContent = unitLabel(currentUnitFilter);
          rerenderAll();
        });
      }

      setInterval(async () => {
        await loadWorkflowState();
        await loadSystemHealth();
        await refreshSnapshot();
      }, 5000);
    }

    boot();
  </script>
</body>
</html>
"""
