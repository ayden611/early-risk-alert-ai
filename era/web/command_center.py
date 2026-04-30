from __future__ import annotations

from flask import Blueprint, Response

command_center_bp = Blueprint("command_center", __name__)

COMMAND_CENTER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Early Risk Alert AI — Command Center</title>
  <style>
    :root{
      --bg:#07111f;
      --panel:#101b2d;
      --panel2:#142238;
      --line:rgba(181,211,255,.18);
      --text:#f7fbff;
      --muted:#a9b8cd;
      --green:#a7ff9a;
      --cyan:#9bd7ff;
      --amber:#ffd36d;
      --red:#ff8fa3;
      --purple:#d8b4fe;
      --shadow:0 18px 70px rgba(0,0,0,.35);
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at 12% 5%,rgba(88,166,255,.18),transparent 30%),
        radial-gradient(circle at 80% 0%,rgba(167,255,154,.12),transparent 28%),
        linear-gradient(180deg,#06101d 0%,#07111f 45%,#050912 100%);
      color:var(--text);
    }
    a{color:inherit;text-decoration:none}
    .wrap{max-width:1540px;margin:0 auto;padding:18px 18px 80px}
    .topbar{
      position:sticky;top:0;z-index:50;
      background:rgba(7,17,31,.88);
      backdrop-filter:blur(14px);
      border-bottom:1px solid var(--line);
    }
    .topbar-inner{
      max-width:1540px;margin:0 auto;padding:12px 18px;
      display:flex;gap:18px;align-items:center;justify-content:space-between;flex-wrap:wrap;
    }
    .brand-kicker{
      color:#b8ffd0;
      font-size:11px;
      letter-spacing:.18em;
      text-transform:uppercase;
      font-weight:900;
    }
    .brand-title{
      margin-top:2px;
      font-size:clamp(24px,3vw,40px);
      line-height:.88;
      font-weight:1000;
      letter-spacing:-.06em;
    }
    .brand-sub{color:var(--muted);font-size:13px;font-weight:800;margin-top:5px}
    .nav{display:flex;gap:7px;flex-wrap:wrap}
    .nav a,.btn,.chip{
      border:1px solid var(--line);
      background:rgba(255,255,255,.06);
      color:var(--text);
      border-radius:999px;
      padding:9px 12px;
      font-weight:900;
      font-size:12px;
      cursor:pointer;
      transition:.15s ease;
      white-space:nowrap;
    }
    .nav a:hover,.btn:hover{transform:translateY(-1px);background:rgba(255,255,255,.1)}
    .hero{
      display:grid;
      grid-template-columns:1.1fr .9fr;
      gap:16px;
      margin-top:18px;
    }
    .panel{
      background:linear-gradient(180deg,rgba(20,34,56,.94),rgba(11,20,35,.94));
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:var(--shadow);
    }
    .intro{padding:20px}
    .badge{
      display:inline-flex;align-items:center;gap:8px;
      border:1px solid rgba(167,255,154,.35);
      background:rgba(167,255,154,.09);
      color:#caffbf;
      border-radius:999px;
      padding:7px 10px;
      font-size:11px;
      letter-spacing:.14em;
      text-transform:uppercase;
      font-weight:1000;
    }
    h1,h2,h3{margin:0;letter-spacing:-.045em}
    .intro h1{
      font-size:clamp(31px,4vw,58px);
      line-height:.92;
      margin:12px 0 10px;
      max-width:850px;
    }
    .intro p{color:#d4dfef;font-size:15px;line-height:1.5;margin:0;max-width:900px;font-weight:700}
    .quick{
      padding:20px;
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:10px;
    }
    .quick .metric{
      background:rgba(255,255,255,.055);
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px;
    }
    .metric label{
      display:block;
      color:#c2d0e3;
      font-size:11px;
      letter-spacing:.14em;
      text-transform:uppercase;
      font-weight:1000;
      margin-bottom:6px;
    }
    .metric strong{
      font-size:clamp(26px,3vw,40px);
      color:var(--green);
      line-height:1;
      letter-spacing:-.05em;
    }
    .metric span{display:block;color:var(--muted);font-size:12px;font-weight:800;margin-top:7px;line-height:1.35}
    .workboard{
      margin-top:16px;
      padding:16px;
      border-color:rgba(167,255,154,.25);
    }
    .work-head{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:12px;
      flex-wrap:wrap;
      margin-bottom:12px;
    }
    .work-head h2{
      font-size:clamp(27px,3vw,44px);
      line-height:.95;
    }
    .work-head p{
      margin:6px 0 0;
      color:#d6e1f0;
      font-weight:750;
      line-height:1.35;
      max-width:850px;
      font-size:13px;
    }
    .controls{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
    .btn.active{background:rgba(167,255,154,.18);border-color:rgba(167,255,154,.5);color:#d9ffd2}
    .btn.purple{background:rgba(216,180,254,.14);border-color:rgba(216,180,254,.35)}
    .stats-row{
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:10px;
      margin:12px 0;
    }
    .stat{
      background:rgba(255,255,255,.055);
      border:1px solid var(--line);
      border-radius:16px;
      padding:13px;
      min-height:88px;
    }
    .stat small{
      color:#b8c7da;
      text-transform:uppercase;
      letter-spacing:.13em;
      font-size:10px;
      font-weight:1000;
    }
    .stat b{
      display:block;
      margin-top:6px;
      font-size:28px;
      color:var(--green);
      letter-spacing:-.04em;
    }
    .stat span{
      display:block;
      color:var(--muted);
      font-size:12px;
      font-weight:800;
      line-height:1.25;
      margin-top:3px;
    }
    .command-grid{
      display:grid;
      grid-template-columns:1.15fr .85fr;
      gap:12px;
      align-items:start;
    }
    .queue-zone{
      background:rgba(255,255,255,.035);
      border:1px solid var(--line);
      border-radius:18px;
      padding:12px;
    }
    .snapshot-meta{
      display:flex;align-items:center;justify-content:space-between;gap:10px;
      color:var(--muted);font-size:12px;font-weight:900;margin-bottom:10px;
    }
    .cards{
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:9px;
      margin-bottom:10px;
    }
    .patient-card{
      border:1px solid var(--line);
      background:rgba(255,255,255,.045);
      border-radius:16px;
      padding:12px;
      cursor:pointer;
      min-height:150px;
      transition:.15s ease;
    }
    .patient-card:hover,.patient-card.selected{
      transform:translateY(-1px);
      border-color:rgba(167,255,154,.55);
      box-shadow:0 0 0 1px rgba(167,255,154,.16),0 16px 45px rgba(0,0,0,.25);
    }
    .patient-top{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:start;
    }
    .rank-title{
      font-size:20px;
      font-weight:1000;
      letter-spacing:-.04em;
      line-height:1.02;
    }
    .unit{color:#c6d6e8;font-size:12px;font-weight:900;margin-top:2px}
    .score{
      color:var(--green);
      font-size:31px;
      font-weight:1000;
      letter-spacing:-.06em;
      line-height:1;
      margin-top:14px;
    }
    .score small{font-size:14px;color:#dfffd9;letter-spacing:0}
    .mini{margin-top:8px;color:#dbe7f5;font-size:12px;font-weight:850;line-height:1.35}
    .pill{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      border-radius:999px;
      padding:5px 8px;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.09em;
      font-weight:1000;
      border:1px solid var(--line);
    }
    .tier-critical{background:rgba(255,143,163,.14);border-color:rgba(255,143,163,.55);color:#ffd2da}
    .tier-elevated{background:rgba(255,211,109,.14);border-color:rgba(255,211,109,.55);color:#ffe6a2}
    .tier-watch{background:rgba(155,215,255,.12);border-color:rgba(155,215,255,.45);color:#cceeff}
    .tier-low{background:rgba(167,255,154,.1);border-color:rgba(167,255,154,.42);color:#d7ffd1}
    table{
      width:100%;
      border-collapse:separate;
      border-spacing:0;
      overflow:hidden;
      border-radius:14px;
      border:1px solid var(--line);
      font-size:12px;
    }
    th{
      background:rgba(154,190,255,.19);
      color:#f3f8ff;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.12em;
      padding:10px 9px;
      text-align:left;
      font-weight:1000;
    }
    td{
      border-top:1px solid rgba(181,211,255,.12);
      padding:10px 9px;
      font-weight:850;
      color:#f5f9ff;
      vertical-align:middle;
    }
    tr{cursor:pointer}
    tr:hover td{background:rgba(255,255,255,.04)}
    .detail{
      padding:14px;
      background:rgba(255,255,255,.045);
      border:1px solid var(--line);
      border-radius:18px;
      min-height:100%;
    }
    .detail-title{
      display:flex;
      align-items:start;
      justify-content:space-between;
      gap:12px;
      margin-bottom:12px;
    }
    .detail-title h3{font-size:26px;line-height:1}
    .detail-title span{color:var(--muted);font-size:12px;font-weight:900}
    .vitals{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:8px;
      margin:10px 0;
    }
    .vital{
      background:rgba(255,255,255,.055);
      border:1px solid rgba(181,211,255,.14);
      border-radius:12px;
      padding:10px;
    }
    .vital small{
      display:block;
      color:#bdcadb;
      font-size:10px;
      letter-spacing:.11em;
      text-transform:uppercase;
      font-weight:1000;
    }
    .vital b{display:block;font-size:20px;margin-top:4px}
    .reason{
      border-left:4px solid var(--green);
      background:rgba(167,255,154,.07);
      border-radius:12px;
      padding:11px;
      margin-top:10px;
      color:#e9fff0;
      font-size:13px;
      line-height:1.35;
      font-weight:800;
    }
    .detail ul{
      margin:10px 0 0 18px;
      padding:0;
      color:#dce7f6;
      font-size:12px;
      line-height:1.55;
      font-weight:800;
    }
    .actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
    .action{
      border:1px solid var(--line);
      background:rgba(255,255,255,.07);
      border-radius:10px;
      color:#fff;
      padding:8px 10px;
      font-size:12px;
      font-weight:900;
      cursor:pointer;
    }
    .guardrail{
      margin-top:10px;
      border:1px solid rgba(255,211,109,.5);
      background:rgba(255,211,109,.08);
      color:#ffe6a2;
      border-radius:14px;
      padding:10px 12px;
      font-size:12px;
      line-height:1.35;
      font-weight:900;
    }
    .lower{
      margin-top:16px;
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:12px;
    }
    .lower .panel{padding:16px}
    .lower h3{font-size:20px;margin-bottom:8px}
    .lower p,.lower li{
      color:#d7e3f2;
      font-size:13px;
      line-height:1.45;
      font-weight:750;
    }
    .lower ul{margin:0;padding-left:18px}
    .mini-table{
      display:grid;
      gap:7px;
      margin-top:10px;
    }
    .mini-row{
      display:grid;
      grid-template-columns:1fr 1fr 1fr;
      gap:7px;
      padding:9px;
      border:1px solid rgba(181,211,255,.13);
      border-radius:12px;
      background:rgba(255,255,255,.04);
      color:#eef6ff;
      font-size:12px;
      font-weight:850;
    }
    .footer{
      color:#aab8cb;
      font-size:12px;
      font-weight:800;
      margin-top:20px;
      padding:14px 0;
      border-top:1px solid var(--line);
    }
    .hidden{display:none!important}
    @media(max-width:1100px){
      .hero,.command-grid,.lower{grid-template-columns:1fr}
      .cards,.stats-row{grid-template-columns:repeat(2,minmax(0,1fr))}
    }
    @media(max-width:720px){
      .cards,.stats-row,.quick{grid-template-columns:1fr}
      .wrap{padding:12px}
      table{font-size:11px}
      th,td{padding:8px 6px}
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="brand-kicker">HCP-facing decision support • rules-based • explainable • CITI certified</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Explainable Rules-Based Command-Center Platform</div>
      </div>
      <nav class="nav" aria-label="Platform navigation">
        <a href="/command-center">Command Center</a>
        <a href="/hospital-demo">Hospital Demo</a>
        <a href="/investor-intake">Investor Access</a>
        <a href="/executive-walkthrough">Executive Walkthrough</a>
        <a href="/admin/review">Admin Review</a>
        <a href="/pilot-docs">Pilot Docs</a>
        <a href="/validation-evidence">Evidence Packet</a>
        <a href="/validation-intelligence">Validation Intelligence</a>
        <a href="/model-card">Model Card</a>
        <a href="/pilot-success-guide">Pilot Success Guide</a>
      </nav>
    </div>
  </header>

  <main class="wrap">
    <section class="hero">
      <div class="panel intro">
        <span class="badge">Pilot mode • controlled demonstration environment</span>
        <h1>Live review workboard for explainable prioritization.</h1>
        <p>
          A compact command-center view showing queue rank, priority tier, primary driver, trend,
          lead-time context, and workflow state. Review Score is a 0–10 simulated prioritization
          score for queue ordering, not a patient-risk percentage.
        </p>
      </div>
      <div class="panel quick">
        <div class="metric">
          <label>Evidence status</label>
          <strong>Locked</strong>
          <span>MIMIC-IV and eICU aggregate retrospective evidence. Raw data remains local-only.</span>
        </div>
        <div class="metric">
          <label>Safety posture</label>
          <strong>CDS</strong>
          <span>Decision support only. No diagnosis, treatment direction, or independent escalation.</span>
        </div>
        <div class="metric">
          <label>Operating point</label>
          <strong>t=6.0</strong>
          <span>Conservative review-burden setting used for public aggregate framing.</span>
        </div>
        <div class="metric">
          <label>Lead-time context</label>
          <strong>3.4–4.8h</strong>
          <span>Retrospective timing context across locked aggregate runs.</span>
        </div>
      </div>
    </section>

    <section class="panel workboard" id="review-workboard">
      <div class="work-head">
        <div>
          <span class="badge">Live review queue • simulated pilot view</span>
          <h2>Prioritized patient review queue</h2>
          <p>
            The top 3–4 review items are visible without scrolling. Tiers, drivers, lead-time context,
            and workflow state are shown visually so the reviewer can understand the queue quickly.
          </p>
        </div>
        <div class="controls">
          <button class="btn active" data-filter="all">All</button>
          <button class="btn" data-filter="Critical">Critical</button>
          <button class="btn" data-filter="Elevated">Elevated</button>
          <button class="btn" data-filter="Watch">Watch</button>
          <button class="btn purple" id="rotateBtn">Rotate snapshot</button>
          <button class="btn" id="pauseBtn">Pause</button>
        </div>
      </div>

      <div class="stats-row">
        <div class="stat">
          <small>Open review items</small>
          <b id="statOpen">4</b>
          <span>Within current demo scope</span>
        </div>
        <div class="stat">
          <small>Highest tier</small>
          <b id="statTier">Critical</b>
          <span>Tier shown visually, not buried in text</span>
        </div>
        <div class="stat">
          <small>Top review driver</small>
          <b id="statDriver">SpO₂ decline</b>
          <span id="statDriverSub">ICU-12 • Worsening</span>
        </div>
        <div class="stat">
          <small>Median lead-time context</small>
          <b id="statLead">3.8h</b>
          <span>Across visible queue</span>
        </div>
      </div>

      <div class="command-grid">
        <div class="queue-zone">
          <div class="snapshot-meta">
            <span id="snapshotLabel">Snapshot 1/4</span>
            <span id="snapshotTime">Demo time</span>
          </div>
          <div class="cards" id="cards"></div>
          <table aria-label="Prioritized review queue">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Patient</th>
                <th>Unit</th>
                <th>Tier</th>
                <th>Review Score</th>
                <th>Primary Driver</th>
                <th>Trend</th>
                <th>Lead Time</th>
                <th>Workflow</th>
              </tr>
            </thead>
            <tbody id="queueBody"></tbody>
          </table>
          <div class="guardrail">
            Guardrail: simulated de-identified demonstration queue. Review Score is a 0–10 prioritization score, not a patient-risk percentage. Workflow actions are review-state examples only.
          </div>
        </div>

        <aside class="detail" id="detail">
          <div class="detail-title">
            <div>
              <h3 id="detailPatient">ICU-12</h3>
              <span id="detailSub">ICU • Queue rank #1 • First threshold context ~4.6h</span>
            </div>
            <span class="pill tier-critical" id="detailTier">Critical</span>
          </div>

          <div class="score" id="detailScore">8.6<small>/10</small></div>

          <div class="vitals" id="vitals"></div>

          <div class="reason" id="reason">
            Oxygenation is the primary driver, with worsening trend and lead-time context.
          </div>

          <ul id="explainBullets"></ul>

          <div class="actions">
            <button class="action">Acknowledge</button>
            <button class="action">Assign</button>
            <button class="action">Escalate review</button>
            <button class="action">Resolve</button>
          </div>

          <div class="guardrail">
            Decision support only. This display does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
          </div>
        </aside>
      </div>
    </section>

    <section class="lower">
      <div class="panel">
        <h3>Validation alignment</h3>
        <p>
          Current public framing should stay conservative: retrospectively evaluated across MIMIC-IV
          and eICU using a harmonized threshold framework.
        </p>
        <div class="mini-table">
          <div class="mini-row"><span>MIMIC-IV t=6.0</span><span>94.3% alert reduction</span><span>4.2% FPR</span></div>
          <div class="mini-row"><span>eICU harmonized t=6.0</span><span>94.25% alert reduction</span><span>0.98% FPR</span></div>
          <div class="mini-row"><span>eICU B/C t=6.0</span><span>95.26%–95.64%</span><span>1.78%–2.02% FPR</span></div>
        </div>
      </div>

      <div class="panel">
        <h3>Explainability layer</h3>
        <ul>
          <li>Priority Tier: Critical / Elevated / Watch / Low.</li>
          <li>Queue Rank: relative review ordering.</li>
          <li>Primary Driver: SpO₂, HR, BP, RR, or trend context.</li>
          <li>Lead Time: retrospective first-flag timing context.</li>
          <li>Workflow State: needs review, acknowledged, assigned, monitoring.</li>
        </ul>
      </div>

      <div class="panel">
        <h3>Pilot readiness links</h3>
        <ul>
          <li><a href="/model-card">Model Card</a> — intended use, evidence status, limitations.</li>
          <li><a href="/pilot-success-guide">Pilot Success Guide</a> — pilot success metrics and workflow guardrails.</li>
          <li><a href="/validation-evidence">Evidence Packet</a> — DUA-safe aggregate evidence.</li>
          <li><a href="/validation-runs">Validation Runs</a> — locked aggregate run registry.</li>
        </ul>
      </div>
    </section>

    <section class="lower">
      <div class="panel">
        <h3>Governance guardrails</h3>
        <ul>
          <li>Authorized healthcare professional review required.</li>
          <li>Aggregate retrospective evidence only.</li>
          <li>No raw MIMIC/eICU/HiRID row-level data in public pages or commits.</li>
          <li>No autonomous diagnosis, treatment direction, or escalation.</li>
        </ul>
      </div>

      <div class="panel">
        <h3>Operational controls</h3>
        <ul>
          <li>Role and unit scope visible in pilot view.</li>
          <li>Workflow states support audit-aware review coordination.</li>
          <li>MFA/security posture remains part of pilot readiness.</li>
          <li>Data upload is positioned as retrospective pilot evaluation.</li>
        </ul>
      </div>

      <div class="panel">
        <h3>What changed</h3>
        <p>
          The old fixed high-percentage ICU-12 display has been removed. The queue now rotates controlled
          snapshots with realistic 0–10 Review Scores, visible tiers, drivers, trends, and lead-time context.
        </p>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC • Decision support only • Retrospective aggregate analysis only • Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
    </div>
  </main>

  <script>
    const snapshots = [
      {
        label:"Snapshot 1/4",
        time:"03:43 PM",
        rows:[
          {patient:"ICU-12",unit:"ICU",tier:"Critical",score:8.6,driver:"SpO₂ decline",trend:"Worsening",lead:"~4.6 hrs",workflow:"Needs review",vitals:{HR:"127 bpm",SpO2:"88%",BP:"168/101",RR:"29/min",Temp:"100.6°F"},why:["SpO₂ decline is the leading driver.","Trend is worsening across the visible window.","Lead-time context places this item at the top of the review queue."]},
          {patient:"TELEMETRY-04",unit:"Telemetry",tier:"Watch",score:6.2,driver:"HR variability",trend:"Stable",lead:"~2.8 hrs",workflow:"Monitoring",vitals:{HR:"104 bpm",SpO2:"94%",BP:"132/82",RR:"20/min",Temp:"98.8°F"},why:["HR variability is present but stable.","No critical tier signal in this snapshot.","Continue monitoring in pilot workflow."]},
          {patient:"STEPDOWN-09",unit:"Stepdown",tier:"Elevated",score:7.4,driver:"RR elevation",trend:"Stable / Watch",lead:"~3.1 hrs",workflow:"Assigned",vitals:{HR:"116 bpm",SpO2:"91%",BP:"142/88",RR:"27/min",Temp:"99.4°F"},why:["Respiratory rate is the main driver.","Trend supports elevated review priority.","Workflow state is assigned."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:5.8,driver:"BP trend",trend:"Stable",lead:"~2.2 hrs",workflow:"Monitoring",vitals:{HR:"92 bpm",SpO2:"96%",BP:"149/91",RR:"18/min",Temp:"98.6°F"},why:["BP trend is visible but stable.","Review score remains below elevated threshold.","Continue monitoring."]}
        ]
      },
      {
        label:"Snapshot 2/4",
        time:"03:49 PM",
        rows:[
          {patient:"TELEMETRY-04",unit:"Telemetry",tier:"Critical",score:8.4,driver:"HR instability",trend:"Worsening",lead:"~4.0 hrs",workflow:"Needs review",vitals:{HR:"132 bpm",SpO2:"94%",BP:"138/84",RR:"23/min",Temp:"98.7°F"},why:["HR instability increased.","Trend moved to worsening.","Telemetry case moved to top of queue."]},
          {patient:"ICU-12",unit:"ICU",tier:"Elevated",score:7.2,driver:"SpO₂ recovery watch",trend:"Stable / Watch",lead:"~3.2 hrs",workflow:"Acknowledged",vitals:{HR:"119 bpm",SpO2:"91%",BP:"154/94",RR:"25/min",Temp:"99.8°F"},why:["Oxygenation remains a review driver but is no longer the top queue item.","Trend is stable/watch rather than worsening.","Review state is acknowledged."]},
          {patient:"STEPDOWN-09",unit:"Stepdown",tier:"Elevated",score:7.0,driver:"RR elevation",trend:"Watch",lead:"~3.0 hrs",workflow:"Assigned",vitals:{HR:"112 bpm",SpO2:"92%",BP:"136/86",RR:"26/min",Temp:"99.1°F"},why:["RR elevation remains visible.","Tier stays elevated.","Assigned for workflow follow-up."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:6.1,driver:"BP trend",trend:"Stable",lead:"~2.5 hrs",workflow:"Monitoring",vitals:{HR:"96 bpm",SpO2:"95%",BP:"151/90",RR:"19/min",Temp:"98.9°F"},why:["BP trend remains present.","Stable trend keeps queue rank lower.","Monitoring continues."]}
        ]
      },
      {
        label:"Snapshot 3/4",
        time:"03:57 PM",
        rows:[
          {patient:"STEPDOWN-09",unit:"Stepdown",tier:"Critical",score:8.3,driver:"RR elevation",trend:"Worsening",lead:"~4.2 hrs",workflow:"Needs review",vitals:{HR:"121 bpm",SpO2:"90%",BP:"144/91",RR:"31/min",Temp:"100.1°F"},why:["Respiratory trend worsened.","RR is the primary driver.","Lead-time context moved this patient to top rank."]},
          {patient:"ICU-12",unit:"ICU",tier:"Watch",score:6.7,driver:"SpO₂ stable",trend:"Stable",lead:"~2.6 hrs",workflow:"Monitoring",vitals:{HR:"108 bpm",SpO2:"93%",BP:"145/88",RR:"21/min",Temp:"99.1°F"},why:["ICU-12 is no longer the highest-priority item in this snapshot.","SpO₂ is stable compared with prior snapshot.","Monitoring is appropriate as a review state."]},
          {patient:"TELEMETRY-04",unit:"Telemetry",tier:"Elevated",score:7.5,driver:"HR instability",trend:"Stable / Watch",lead:"~3.4 hrs",workflow:"Acknowledged",vitals:{HR:"124 bpm",SpO2:"95%",BP:"133/82",RR:"22/min",Temp:"98.6°F"},why:["HR instability remains visible.","Trend is no longer worsening.","Acknowledged workflow state."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:5.9,driver:"BP trend",trend:"Stable",lead:"~2.1 hrs",workflow:"Monitoring",vitals:{HR:"94 bpm",SpO2:"96%",BP:"146/89",RR:"18/min",Temp:"98.4°F"},why:["BP trend is stable.","Lower review score keeps this item lower in queue.","Monitoring continues."]}
        ]
      },
      {
        label:"Snapshot 4/4",
        time:"04:05 PM",
        rows:[
          {patient:"ICU-21",unit:"ICU",tier:"Critical",score:8.1,driver:"BP instability",trend:"Worsening",lead:"~3.9 hrs",workflow:"Needs review",vitals:{HR:"122 bpm",SpO2:"92%",BP:"172/96",RR:"24/min",Temp:"99.9°F"},why:["BP instability is the top driver.","Worsening trend raises priority.","Lead-time context supports top queue placement."]},
          {patient:"ICU-12",unit:"ICU",tier:"Elevated",score:7.6,driver:"SpO₂ watch",trend:"Stable / Watch",lead:"~3.3 hrs",workflow:"Assigned",vitals:{HR:"115 bpm",SpO2:"92%",BP:"150/91",RR:"24/min",Temp:"99.5°F"},why:["ICU-12 remains visible but not static.","Review score and tier changed from prior snapshots.","Assigned workflow state."]},
          {patient:"TELEMETRY-04",unit:"Telemetry",tier:"Elevated",score:7.1,driver:"HR variability",trend:"Stable",lead:"~2.9 hrs",workflow:"Acknowledged",vitals:{HR:"118 bpm",SpO2:"95%",BP:"129/80",RR:"21/min",Temp:"98.5°F"},why:["HR variability remains a driver.","Trend is stable.","Acknowledged workflow state."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:6.0,driver:"BP trend",trend:"Stable",lead:"~2.4 hrs",workflow:"Monitoring",vitals:{HR:"93 bpm",SpO2:"96%",BP:"148/90",RR:"18/min",Temp:"98.6°F"},why:["BP trend is stable.","Watch tier keeps this lower in review queue.","Monitoring continues."]}
        ]
      }
    ];

    let snapshotIndex = 0;
    let currentFilter = "all";
    let paused = false;
    let selectedPatient = null;

    function tierClass(tier){
      if(tier === "Critical") return "tier-critical";
      if(tier === "Elevated") return "tier-elevated";
      if(tier === "Watch") return "tier-watch";
      return "tier-low";
    }

    function medianLead(rows){
      const nums = rows.map(r => parseFloat(String(r.lead).replace(/[^0-9.]/g,""))).filter(n => !Number.isNaN(n));
      if(!nums.length) return "—";
      nums.sort((a,b)=>a-b);
      const mid = Math.floor(nums.length/2);
      const val = nums.length % 2 ? nums[mid] : (nums[mid-1] + nums[mid]) / 2;
      return val.toFixed(1) + "h";
    }

    function filteredRows(){
      const rows = snapshots[snapshotIndex].rows;
      return currentFilter === "all" ? rows : rows.filter(r => r.tier === currentFilter);
    }

    function render(){
      const snap = snapshots[snapshotIndex];
      const rows = filteredRows();
      const displayRows = rows.length ? rows : snap.rows;

      if(!selectedPatient || !displayRows.some(r => r.patient === selectedPatient)){
        selectedPatient = displayRows[0].patient;
      }

      document.getElementById("snapshotLabel").textContent = snap.label;
      document.getElementById("snapshotTime").textContent = snap.time;
      document.getElementById("statOpen").textContent = snap.rows.length;
      document.getElementById("statTier").textContent = snap.rows[0].tier;
      document.getElementById("statDriver").textContent = snap.rows[0].driver;
      document.getElementById("statDriverSub").textContent = snap.rows[0].patient + " • " + snap.rows[0].trend;
      document.getElementById("statLead").textContent = medianLead(snap.rows);

      const cards = document.getElementById("cards");
      cards.innerHTML = "";
      displayRows.forEach((r, idx) => {
        const card = document.createElement("div");
        card.className = "patient-card" + (r.patient === selectedPatient ? " selected" : "");
        card.onclick = () => { selectedPatient = r.patient; render(); };
        card.innerHTML = `
          <div class="patient-top">
            <div>
              <div class="rank-title">#${idx + 1} ${r.patient}</div>
              <div class="unit">${r.unit}</div>
            </div>
            <span class="pill ${tierClass(r.tier)}">${r.tier}</span>
          </div>
          <div class="score">${r.score.toFixed(1)}<small>/10</small></div>
          <div class="mini">
            <b>Driver:</b> ${r.driver}<br>
            <b>Trend:</b> ${r.trend}<br>
            <b>Lead:</b> ${r.lead}
          </div>
        `;
        cards.appendChild(card);
      });

      const body = document.getElementById("queueBody");
      body.innerHTML = "";
      displayRows.forEach((r, idx) => {
        const tr = document.createElement("tr");
        tr.onclick = () => { selectedPatient = r.patient; render(); };
        tr.innerHTML = `
          <td>#${idx + 1}</td>
          <td>${r.patient}</td>
          <td>${r.unit}</td>
          <td><span class="pill ${tierClass(r.tier)}">${r.tier}</span></td>
          <td>${r.score.toFixed(1)}/10</td>
          <td>${r.driver}</td>
          <td>${r.trend}</td>
          <td>${r.lead}</td>
          <td>${r.workflow}</td>
        `;
        body.appendChild(tr);
      });

      const selected = snap.rows.find(r => r.patient === selectedPatient) || displayRows[0];
      renderDetail(selected, snap.rows.indexOf(selected) + 1);
    }

    function renderDetail(r, rank){
      document.getElementById("detailPatient").textContent = r.patient;
      document.getElementById("detailSub").textContent = `${r.unit} • Queue rank #${rank} • First threshold context ${r.lead}`;
      const tier = document.getElementById("detailTier");
      tier.className = "pill " + tierClass(r.tier);
      tier.textContent = r.tier;
      document.getElementById("detailScore").innerHTML = `${r.score.toFixed(1)}<small>/10</small>`;

      const vitals = document.getElementById("vitals");
      vitals.innerHTML = Object.entries(r.vitals).map(([k,v]) => `
        <div class="vital"><small>${k}</small><b>${v}</b></div>
      `).join("");

      document.getElementById("reason").textContent =
        `${r.driver} is the selected primary driver. Trend: ${r.trend}. Lead-time context: ${r.lead}.`;

      document.getElementById("explainBullets").innerHTML =
        r.why.map(x => `<li>${x}</li>`).join("");
    }

    function rotate(){
      snapshotIndex = (snapshotIndex + 1) % snapshots.length;
      selectedPatient = null;
      render();
    }

    document.querySelectorAll("[data-filter]").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-filter]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentFilter = btn.dataset.filter;
        selectedPatient = null;
        render();
      });
    });

    document.getElementById("rotateBtn").addEventListener("click", rotate);
    document.getElementById("pauseBtn").addEventListener("click", () => {
      paused = !paused;
      document.getElementById("pauseBtn").textContent = paused ? "Resume" : "Pause";
    });

    render();
    setInterval(() => {
      if(!paused) rotate();
    }, 9000);
  </script>
</body>
</html>
"""

def command_center_page() -> Response:
    return Response(COMMAND_CENTER_HTML, mimetype="text/html")

@command_center_bp.route("/")
@command_center_bp.route("/command-center")
def command_center() -> Response:
    return command_center_page()

def register_command_center_routes(app):
    existing_rules = {str(rule.rule) for rule in app.url_map.iter_rules()}
    if "/command-center" not in existing_rules:
        app.add_url_rule("/command-center", "command_center_page", command_center_page)

# Compatibility aliases used by earlier builds.
bp = command_center_bp
get_command_center_html = lambda: COMMAND_CENTER_HTML
render_command_center = command_center_page

__all__ = [
    "command_center_bp",
    "bp",
    "command_center",
    "command_center_page",
    "register_command_center_routes",
    "get_command_center_html",
    "render_command_center",
    "COMMAND_CENTER_HTML",
]
