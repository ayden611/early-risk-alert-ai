from __future__ import annotations

from pathlib import Path
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
      --bg:#06101d;
      --panel:#101b2d;
      --panel2:#142238;
      --panel3:#0c1526;
      --line:rgba(184,211,255,.16);
      --text:#f7fbff;
      --muted:#a9b8cc;
      --green:#a7ff9a;
      --cyan:#9bd7ff;
      --amber:#ffd36d;
      --red:#ff8fa3;
      --blue:#9bc8ff;
      --purple:#d8b4fe;
      --shadow:0 18px 70px rgba(0,0,0,.34);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      min-height:100vh;
      overflow-x:hidden;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at 10% 0%,rgba(78,166,255,.12),transparent 28%),
        radial-gradient(circle at 88% 6%,rgba(167,255,154,.08),transparent 22%),
        linear-gradient(180deg,#050b14 0%,#07111c 46%,#050913 100%);
      color:var(--text);
    }
    a{color:inherit;text-decoration:none}
    .topbar{
      position:sticky;
      top:0;
      z-index:50;
      background:rgba(5,12,22,.96);
      backdrop-filter:blur(16px);
      border-bottom:1px solid rgba(184,211,255,.14);
      box-shadow:0 8px 28px rgba(0,0,0,.18);
    }
    .topbar-inner{
      max-width:1540px;
      margin:0 auto;
      padding:12px 18px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:16px;
      flex-wrap:wrap;
    }
    .brand-kicker{
      color:#b8ffd0;
      font-size:10px;
      letter-spacing:.18em;
      text-transform:uppercase;
      font-weight:1000;
    }
    .brand-title{
      margin-top:3px;
      font-size:clamp(24px,3vw,40px);
      line-height:.92;
      font-weight:1000;
      letter-spacing:-.06em;
    }
    .brand-sub{
      color:var(--muted);
      font-size:12px;
      font-weight:850;
      margin-top:5px;
    }
    .nav{
      display:flex;
      gap:7px;
      flex-wrap:wrap;
      justify-content:flex-end;
    }
    .nav a,.btn{
      border:1px solid var(--line);
      background:rgba(255,255,255,.06);
      color:var(--text);
      border-radius:999px;
      padding:8px 11px;
      font-weight:900;
      font-size:12px;
      cursor:pointer;
      transition:.15s ease;
      white-space:nowrap;
    }
    .nav a:hover,.btn:hover{
      background:rgba(255,255,255,.10);
      transform:translateY(-1px);
    }
    .wrap{
      max-width:1540px;
      margin:0 auto;
      padding:16px 18px 70px;
    }
    .pilot-banner{
      display:grid;
      grid-template-columns:1fr auto;
      gap:12px;
      align-items:center;
      padding:12px 14px;
      border:1px solid rgba(255,211,109,.26);
      background:linear-gradient(180deg,rgba(13,23,38,.98),rgba(10,18,31,.98));
      border-radius:18px;
      margin-bottom:14px;
      box-shadow:0 12px 34px rgba(0,0,0,.22);
    }
    .pilot-banner strong{
      color:#ffe29a;
      font-size:11px;
      letter-spacing:.14em;
      text-transform:uppercase;
      font-weight:1000;
    }
    .pilot-banner p{
      margin:4px 0 0;
      color:#f4f8ff;
      font-size:13px;
      line-height:1.42;
      font-weight:850;
      max-width:960px;
    }
    .status-pills{
      display:flex;
      gap:7px;
      flex-wrap:wrap;
      justify-content:flex-end;
    }
    .pill{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      border-radius:999px;
      padding:6px 9px;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.08em;
      font-weight:1000;
      border:1px solid var(--line);
      background:rgba(255,255,255,.08);
      white-space:nowrap;
      max-width:100%;
    }
    .pill.green{color:#d7ffd1;border-color:rgba(167,255,154,.42);background:rgba(167,255,154,.10)}
    .pill.amber{color:#ffe6a2;border-color:rgba(255,211,109,.52);background:rgba(255,211,109,.12)}
    .pill.blue{color:#d5e9ff;border-color:rgba(155,200,255,.42);background:rgba(155,200,255,.10)}
    .hero-grid{
      display:grid;
      grid-template-columns:minmax(0,1.7fr) minmax(320px,.75fr);
      gap:14px;
      align-items:start;
    }
    .panel{
      background:linear-gradient(180deg,rgba(16,27,45,.97),rgba(10,18,31,.97));
      border:1px solid rgba(184,211,255,.15);
      border-radius:22px;
      box-shadow:var(--shadow);
    }
    .queue-panel{
      padding:16px;
      border-color:rgba(167,255,154,.25);
    }
    .queue-head{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:12px;
      flex-wrap:wrap;
      margin-bottom:12px;
    }
    .eyebrow{
      display:inline-flex;
      align-items:center;
      gap:8px;
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
    .queue-head h1{
      font-size:clamp(31px,4vw,54px);
      line-height:.92;
      margin:11px 0 6px;
    }
    .queue-head p{
      color:#d8e3f2;
      margin:0;
      font-size:14px;
      line-height:1.42;
      font-weight:760;
      max-width:900px;
    }
    .controls{
      display:flex;
      gap:7px;
      flex-wrap:wrap;
      justify-content:flex-end;
      align-items:center;
    }
    .btn.active{
      background:rgba(167,255,154,.18);
      border-color:rgba(167,255,154,.52);
      color:#d9ffd2;
    }
    .btn.purple{
      background:rgba(216,180,254,.13);
      border-color:rgba(216,180,254,.35);
    }

    .scope-controls{
      align-items:center;
      gap:7px;
    }
    .control-label{
      color:#cfe0f4;
      font-size:10px;
      letter-spacing:.13em;
      text-transform:uppercase;
      font-weight:1000;
      padding:0 2px;
      opacity:.9;
      white-space:nowrap;
    }
    .control-sep{
      width:1px;
      min-height:26px;
      background:rgba(184,211,255,.22);
      margin:0 2px;
      display:inline-block;
    }
    .scope-note{
      margin:10px 0 12px;
      border:1px solid rgba(155,215,255,.34);
      background:rgba(155,215,255,.08);
      color:#dceeff;
      border-radius:14px;
      padding:9px 11px;
      font-size:12px;
      line-height:1.35;
      font-weight:850;
    }
    .scope-note strong{
      color:#bfe8ff;
      text-transform:uppercase;
      letter-spacing:.08em;
      font-size:10px;
      margin-right:5px;
    }

    .mini-metrics{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(165px,1fr));
      gap:10px;
      margin:12px 0;
    }
    .metric{
      background:rgba(255,255,255,.055);
      border:1px solid var(--line);
      border-radius:16px;
      padding:12px;
      min-height:86px;
    }
    .metric small{
      display:block;
      color:#b8c7da;
      text-transform:uppercase;
      letter-spacing:.13em;
      font-size:10px;
      font-weight:1000;
      margin-bottom:6px;
    }
    .metric b{
      display:block;
      font-size:27px;
      color:var(--green);
      letter-spacing:-.04em;
      line-height:1;
    }
    .metric span{
      display:block;
      color:var(--muted);
      font-size:12px;
      font-weight:800;
      line-height:1.25;
      margin-top:5px;
    }
    .main-tool{
      display:grid;
      grid-template-columns:minmax(0,1.2fr) minmax(300px,.62fr);
      gap:12px;
      align-items:start;
    }
    .queue-box{
      background:rgba(255,255,255,.03);
      border:1px solid var(--line);
      border-radius:18px;
      padding:12px;
      overflow:hidden;
    }
    .snapshot-meta{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      color:var(--muted);
      font-size:12px;
      font-weight:900;
      margin-bottom:10px;
      flex-wrap:wrap;
    }
    .card-row{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(165px,1fr));
      gap:10px;
      margin-bottom:10px;
      align-items:stretch;
    }
    .patient-card{
      border:1px solid var(--line);
      background:rgba(255,255,255,.045);
      border-radius:16px;
      padding:12px;
      cursor:pointer;
      min-height:0;
      overflow:hidden;
      transition:.15s ease;
    }
    .patient-card:hover,.patient-card.selected{
      transform:translateY(-1px);
      border-color:rgba(167,255,154,.55);
      box-shadow:0 0 0 1px rgba(167,255,154,.16),0 16px 45px rgba(0,0,0,.25);
    }
    .patient-top{
      display:grid;
      grid-template-columns:minmax(0,1fr) auto;
      gap:8px;
      align-items:start;
    }
    .patient-top > div{
      min-width:0;
    }
    .patient-top .pill{
      align-self:start;
      font-size:9px;
      padding:5px 8px;
      line-height:1;
      max-width:100%;
    }
    .rank-title{
      font-size:16px;
      font-weight:1000;
      letter-spacing:-.04em;
      line-height:1.05;
      word-break:break-word;
    }
    .unit{
      color:#c6d6e8;
      font-size:12px;
      font-weight:900;
      margin-top:2px;
    }
    .score{
      color:var(--green);
      font-size:31px;
      font-weight:1000;
      letter-spacing:-.06em;
      line-height:1;
      margin-top:12px;
    }
    .score small{
      font-size:14px;
      color:#dfffd9;
      letter-spacing:0;
    }
    .mini{
      margin-top:8px;
      color:#dbe7f5;
      font-size:11px;
      font-weight:850;
      line-height:1.35;
    }
    .tier-critical{background:rgba(255,143,163,.14);border-color:rgba(255,143,163,.55);color:#ffd2da}
    .tier-elevated{background:rgba(255,211,109,.14);border-color:rgba(255,211,109,.55);color:#ffe6a2}
    .tier-watch{background:rgba(155,215,255,.12);border-color:rgba(155,215,255,.45);color:#cceeff}
    .tier-low{background:rgba(167,255,154,.1);border-color:rgba(167,255,154,.42);color:#d7ffd1}
    .table-wrap{overflow-x:auto}
    table{
      width:100%;
      border-collapse:separate;
      border-spacing:0;
      min-width:880px;
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
      white-space:nowrap;
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
    .score-cell{color:var(--green);font-weight:1000;white-space:nowrap}
    .action-row{display:flex;gap:5px;flex-wrap:wrap}
    .small-action{
      border:1px solid var(--line);
      background:rgba(255,255,255,.07);
      color:#fff;
      border-radius:999px;
      padding:5px 8px;
      font-size:10px;
      font-weight:900;
      cursor:pointer;
      white-space:nowrap;
    }
    .small-action:hover{background:rgba(167,255,154,.13);border-color:rgba(167,255,154,.35)}
    .detail{
      padding:14px;
      background:rgba(255,255,255,.045);
      border:1px solid var(--line);
      border-radius:18px;
      min-height:100%;
    }
    .detail-title{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:12px;
      margin-bottom:10px;
    }
    .detail-title h3{
      font-size:26px;
      line-height:1;
    }
    .detail-sub{
      color:var(--muted);
      font-size:12px;
      font-weight:900;
      line-height:1.3;
      margin-top:4px;
    }
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
    .vital b{
      display:block;
      font-size:18px;
      margin-top:4px;
    }
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
    .side-panel{
      padding:14px;
    }
    .side-panel h2{
      font-size:24px;
      margin-bottom:8px;
    }
    .side-grid{
      display:grid;
      gap:10px;
    }
    .side-card{
      border:1px solid var(--line);
      background:rgba(255,255,255,.05);
      border-radius:16px;
      padding:12px;
    }
    .side-card small{
      display:block;
      color:#b8c7da;
      text-transform:uppercase;
      letter-spacing:.13em;
      font-size:10px;
      font-weight:1000;
      margin-bottom:6px;
    }
    .side-card b{
      color:var(--green);
      font-size:25px;
      letter-spacing:-.04em;
    }
    .side-card p{
      margin:5px 0 0;
      color:#d6e3f5;
      font-size:12px;
      line-height:1.35;
      font-weight:780;
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
    .footer-guardrail{
      margin-top:14px;
      border:1px solid rgba(255,211,109,.38);
      background:rgba(255,211,109,.075);
      color:#ffe6a2;
      border-radius:18px;
      padding:12px 14px;
      font-size:12px;
      line-height:1.45;
      font-weight:900;
      text-align:center;
    }
    .lower{
      margin-top:14px;
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:12px;
    }
    .lower .panel{
      padding:16px;
    }
    .lower h3{
      font-size:20px;
      margin-bottom:8px;
    }
    .lower p,.lower li{
      color:#d7e3f2;
      font-size:13px;
      line-height:1.45;
      font-weight:750;
    }
    .lower ul{margin:0;padding-left:18px}
    @media(max-width:1150px){
      .hero-grid,.main-tool,.lower{grid-template-columns:1fr}
      .card-row,.mini-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
      .side-panel{order:3}
    }
    @media(max-width:720px){
      .card-row,.mini-metrics,.vitals{grid-template-columns:1fr}
      .pilot-banner{grid-template-columns:1fr}
      .status-pills,.controls{justify-content:flex-start}
      .wrap{padding:12px}
      .patient-top{grid-template-columns:1fr}
      .patient-top .pill{justify-self:start}
    }
  
/* ERA_COMMAND_CENTER_VISIBILITY_FIX_START */

/* Dark production shell: prevents washed-out white/cream background behind the tool */
html,
body {
  background:
    radial-gradient(circle at 8% 0%, rgba(78,166,255,.16), transparent 30%),
    radial-gradient(circle at 90% 4%, rgba(167,255,154,.10), transparent 24%),
    linear-gradient(180deg, #050b16 0%, #06101d 48%, #030711 100%) !important;
  color: #f7fbff !important;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(90deg, rgba(255,255,255,.025), transparent 18%, transparent 82%, rgba(255,255,255,.02)),
    radial-gradient(circle at 50% 0%, rgba(255,255,255,.045), transparent 34%);
  z-index: -1;
}

.wrap,
main.wrap {
  background: transparent !important;
}

/* Restore clean platform identity */
.brand-title,
.topbar .brand-title {
  color: #ffffff !important;
  text-shadow: 0 2px 18px rgba(0,0,0,.62) !important;
}

.brand-kicker,
.topbar .brand-kicker {
  color: #caffbf !important;
  text-shadow: 0 1px 10px rgba(0,0,0,.6) !important;
}

.brand-sub,
.topbar .brand-sub {
  color: #dbe8f8 !important;
}

/* Top bar should match the darker validation/evidence pages */
.topbar {
  background:
    linear-gradient(180deg, rgba(8,16,29,.98), rgba(6,13,24,.96)) !important;
  border-bottom: 1px solid rgba(184,211,255,.18) !important;
  box-shadow: 0 12px 35px rgba(0,0,0,.32) !important;
}

.topbar-inner {
  background: transparent !important;
}

.nav a {
  background: rgba(18,31,52,.94) !important;
  color: #f7fbff !important;
  border: 1px solid rgba(184,211,255,.22) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06) !important;
}

.nav a:hover {
  background: rgba(34,52,82,.98) !important;
  border-color: rgba(167,255,154,.38) !important;
}

/* Fix washed-out pilot banner: readable, compact, dark, and professional */
.pilot-banner {
  background:
    linear-gradient(90deg, rgba(20,34,56,.98), rgba(12,24,40,.96)) !important;
  border: 1px solid rgba(255,211,109,.35) !important;
  box-shadow: 0 16px 45px rgba(0,0,0,.30) !important;
  color: #f7fbff !important;
  opacity: 1 !important;
  filter: none !important;
}

.pilot-banner *,
.pilot-banner p,
.pilot-banner strong {
  color: #f7fbff !important;
  opacity: 1 !important;
  text-shadow: 0 1px 10px rgba(0,0,0,.58) !important;
}

.pilot-banner strong {
  color: #ffe6a2 !important;
}

.pilot-banner p {
  color: #e3ecf8 !important;
}

/* Ensure pills are readable on top banner */
.status-pills .pill,
.pilot-banner .pill {
  background: rgba(8,16,29,.88) !important;
  color: #f7fbff !important;
  border-color: rgba(184,211,255,.28) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.07) !important;
}

.status-pills .pill.green,
.pilot-banner .pill.green {
  color: #d7ffd1 !important;
  border-color: rgba(167,255,154,.42) !important;
  background: rgba(49,93,62,.48) !important;
}

.status-pills .pill.blue,
.pilot-banner .pill.blue {
  color: #d5e9ff !important;
  border-color: rgba(155,200,255,.42) !important;
  background: rgba(43,69,105,.50) !important;
}

.status-pills .pill.amber,
.pilot-banner .pill.amber {
  color: #ffe6a2 !important;
  border-color: rgba(255,211,109,.45) !important;
  background: rgba(102,73,21,.45) !important;
}

/* Match the clean dark card look used by validation pages */
.panel,
.queue-panel,
.side-panel,
.queue-box,
.detail,
.lower .panel {
  background:
    linear-gradient(180deg, rgba(20,34,56,.96), rgba(9,18,33,.96)) !important;
  border-color: rgba(184,211,255,.18) !important;
  color: #f7fbff !important;
}

.queue-panel {
  border-color: rgba(167,255,154,.28) !important;
}

.metric,
.side-card,
.patient-card,
.vital {
  background: rgba(255,255,255,.055) !important;
  border-color: rgba(184,211,255,.17) !important;
}

.patient-card.selected,
.patient-card:hover {
  border-color: rgba(167,255,154,.58) !important;
  box-shadow: 0 0 0 1px rgba(167,255,154,.16), 0 16px 45px rgba(0,0,0,.30) !important;
}

/* Prevent any pale page area from showing behind the right side */
.hero-grid,
.lower {
  background: transparent !important;
}

.footer-guardrail,
.guardrail {
  background: rgba(255,211,109,.085) !important;
  border-color: rgba(255,211,109,.45) !important;
  color: #ffe6a2 !important;
}

/* Make the table feel more like a tool */
table {
  background: rgba(8,16,29,.35) !important;
}

th {
  background: rgba(154,190,255,.20) !important;
  color: #f7fbff !important;
}

td {
  color: #f7fbff !important;
}

/* Mobile/smaller laptop readability */
@media(max-width: 1150px) {
  .hero-grid {
    grid-template-columns: 1fr !important;
  }

  .side-panel {
    max-width: none !important;
  }
}

/* ERA_COMMAND_CENTER_VISIBILITY_FIX_END */


/* === ERA FINAL PILOT POLISH: compact clinical workboard hierarchy === */
.pilot-polish-marker{display:none}

.queue-panel{
  max-width:100%;
}

.queue-head h1{
  max-width:860px;
}

.queue-head p{
  max-width:760px;
}

.card-row{
  align-items:stretch;
}

.patient-card{
  position:relative;
  overflow:hidden;
  padding:10px 10px 11px;
  min-height:132px;
}

.patient-card .pill,
.patient-top .pill{
  position:relative;
  z-index:2;
  flex-shrink:0;
  max-width:104px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  padding:5px 7px;
  font-size:9px;
}

.rank-title{
  font-size:16px;
  line-height:1.02;
  max-width:130px;
  overflow-wrap:anywhere;
}

.unit{
  font-size:11px;
}

.patient-card .score{
  font-size:30px;
  margin-top:8px;
}

.patient-card .score small{
  font-size:12px;
}

.patient-card .mini{
  font-size:11px;
  line-height:1.22;
  margin-top:6px;
}

.patient-card .mini b{
  color:#eaf3ff;
}

.card-row .patient-card:first-child{
  border-color:rgba(167,255,154,.78);
  background:
    radial-gradient(circle at 8% 0%,rgba(167,255,154,.18),transparent 36%),
    linear-gradient(180deg,rgba(30,48,73,.98),rgba(12,22,38,.98));
  box-shadow:
    0 0 0 1px rgba(167,255,154,.24),
    0 18px 48px rgba(0,0,0,.34);
}

.card-row .patient-card:first-child::before{
  content:"TOP REVIEW";
  position:absolute;
  top:8px;
  right:10px;
  border:1px solid rgba(167,255,154,.45);
  color:#d7ffd1;
  background:rgba(167,255,154,.12);
  border-radius:999px;
  padding:4px 7px;
  font-size:8px;
  font-weight:1000;
  letter-spacing:.11em;
  z-index:1;
}

.card-row .patient-card:first-child .rank-title{
  font-size:21px;
  max-width:170px;
}

.card-row .patient-card:first-child .score{
  font-size:40px;
  text-shadow:0 0 24px rgba(167,255,154,.22);
}

.card-row .patient-card:first-child .pill{
  margin-top:22px;
}

.detail{
  box-shadow:inset 4px 0 0 rgba(167,255,154,.35);
}

.detail-title h3{
  font-size:30px;
}

.detail .score{
  font-size:42px;
  margin:4px 0 8px;
}

.reason{
  font-size:12px;
  line-height:1.28;
  padding:10px;
}

.detail ul{
  font-size:11px;
  line-height:1.38;
  margin-top:8px;
}

.vital{
  padding:8px;
}

.vital b{
  font-size:16px;
}

.tier-critical{
  box-shadow:0 0 0 1px rgba(255,143,163,.18);
}

.tier-elevated{
  box-shadow:0 0 0 1px rgba(255,211,109,.18);
}

.tier-watch{
  box-shadow:0 0 0 1px rgba(155,215,255,.15);
}

.lower{
  margin-top:10px;
}

.about-demo{
  margin-top:12px;
  border:1px solid rgba(184,211,255,.16);
  background:rgba(255,255,255,.035);
  border-radius:18px;
  overflow:hidden;
}

.about-demo summary{
  cursor:pointer;
  list-style:none;
  padding:12px 14px;
  color:#dfeaff;
  font-size:12px;
  font-weight:1000;
  letter-spacing:.08em;
  text-transform:uppercase;
  display:flex;
  align-items:center;
  justify-content:space-between;
}

.about-demo summary::-webkit-details-marker{
  display:none;
}

.about-demo summary::after{
  content:"Open";
  border:1px solid rgba(184,211,255,.22);
  background:rgba(255,255,255,.06);
  border-radius:999px;
  padding:5px 9px;
  color:#cfe0f4;
  font-size:10px;
  letter-spacing:.08em;
}

.about-demo[open] summary::after{
  content:"Close";
}

.about-demo .lower{
  padding:0 12px 12px;
}

.about-demo .panel{
  box-shadow:none;
}

.footer-guardrail{
  margin-top:10px;
  padding:10px 12px;
  font-size:11px;
}

@media(min-width:1000px){
  .card-row .patient-card:first-child{
    grid-column:span 2;
  }
}

@media(max-width:1150px){
  .card-row .patient-card:first-child{
    grid-column:auto;
  }
  .card-row .patient-card:first-child .rank-title{
    font-size:18px;
  }
  .card-row .patient-card:first-child .score{
    font-size:34px;
  }
}

</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="brand-kicker">HCP-facing decision support • rules-based • explainable • CITI certified</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Clinical Command Center</div>
      </div>
      <nav class="nav" aria-label="Platform navigation">
        <a href="/command-center">Command Center</a>
        <a href="/hospital-demo">Hospital Demo</a>
        <a href="/investor-intake">Investor Access</a>
        <a href="/executive-walkthrough">Executive Walkthrough</a>
        <a href="/admin/review">Admin Review</a>
        <a href="/pilot-docs">Pilot Docs</a>
        <a href="/command-center/deck">Governance Deck</a>
        <a href="/validation-evidence">Evidence Packet</a>
        <a href="/validation-intelligence">Validation Intelligence</a>
      </nav>
    </div>
  </header>

  <main class="wrap">
    <section class="pilot-banner">
      <div>
        <strong>Pilot mode • controlled demonstration • decision support only</strong>
        <p>
          This is a simulated pilot environment. Outputs support — not replace — clinical judgment.
          Review Score is a 0–10 prioritization score, not a clinical probability.
        </p>
      </div>
      <div class="status-pills">
        <span class="pill green">Role: Admin</span>
        <span class="pill blue">Scope: All Units</span>
        <span class="pill green">MFA Active</span>
        <span class="pill amber" id="lastUpdated">Updated now</span>
      </div>
    </section>

    <section class="hero-grid">
      <div class="panel queue-panel">
        <div class="queue-head">
          <div>
            <span class="eyebrow">Live review queue • simulated pilot view</span>
            <h1>Prioritized patient review queue</h1>
            <p>
              Compact workboard showing rank, unit, priority tier, review score, primary driver,
              trend, lead-time context, and workflow state in one glanceable view.
            </p>
          </div>
          <div class="controls scope-controls" aria-label="Demo queue filters">
            <span class="control-label">View scope</span>
            <button class="btn active" data-unit-filter="all">All Units</button>
            <button class="btn" data-unit-filter="ICU">ICU</button>
            <button class="btn" data-unit-filter="Telemetry">Telemetry</button>
            <button class="btn" data-unit-filter="Stepdown">Stepdown</button>
            <button class="btn" data-unit-filter="Ward">Ward</button>
            <span class="control-sep" aria-hidden="true"></span>
            <span class="control-label">Priority</span>
            <button class="btn active" data-filter="all">All Priorities</button>
            <button class="btn" data-filter="Critical">Critical</button>
            <button class="btn" data-filter="Elevated">Elevated</button>
            <button class="btn" data-filter="Watch">Watch</button>
            <span class="control-sep" aria-hidden="true"></span>
            <button class="btn purple" id="rotateBtn">Rotate snapshot</button>
            <button class="btn" id="pauseBtn">Pause</button>
          </div>
        </div>

        <div class="scope-note" id="scopeNote">
          <strong>View scope</strong>View scope: simulated unit filter for pilot demonstration. Production pilot access should be tied to authorized role and unit permissions.
        </div>

        <div class="mini-metrics">
          <div class="metric">
            <small>Open review items</small>
            <b id="statOpen">4</b>
            <span>Visible in current queue</span>
          </div>
          <div class="metric">
            <small>Critical items</small>
            <b id="statCritical">1</b>
            <span>Higher-priority items</span>
          </div>
          <div class="metric">
            <small>Average review score</small>
            <b id="statAverage">7.4/10</b>
            <span>0–10 queue scale</span>
          </div>
          <div class="metric">
            <small>System status</small>
            <b>Connected</b>
            <span>Role scoped • MFA active</span>
          </div>
        </div>

        <div class="main-tool">
          <div class="queue-box">
            <div class="snapshot-meta">
              <span id="snapshotLabel">Snapshot 1/4</span>
              <span>Sort: Queue Rank ↓ • Filter: <span id="filterLabel">All Units</span> • Operating Point: t=6.0 Conservative</span>
            </div>

            <div class="card-row" id="cards"></div>

            <div class="table-wrap">
              <table aria-label="Prioritized patient review queue">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Patient / Unit</th>
                    <th>Tier</th>
                    <th>Review Score</th>
                    <th>Primary Driver</th>
                    <th>Trend</th>
                    <th>Lead Time</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody id="queueBody"></tbody>
              </table>
            </div>
          </div>

          <aside class="detail" id="detail">
            <div class="detail-title">
              <div>
                <h3 id="detailPatient">ICU-12</h3>
                <div class="detail-sub" id="detailSub">ICU • Rank #1 • Critical</div>
              </div>
              <span class="pill tier-critical" id="detailTier">Critical</span>
            </div>

            <div class="score" id="detailScore">9.2<small>/10</small></div>
            <div class="vitals" id="vitals"></div>
            <div class="reason" id="reason"></div>
            <ul id="explainBullets"></ul>

            <div class="guardrail">
              Selected patient detail is review-support context only. Clinicians must independently review the patient and current clinical context.
            </div>
          </aside>
        </div>

        <div class="footer-guardrail">
          Decision support only. No diagnosis, treatment direction, clinician replacement, or independent escalation.
          Simulated demo data only. Retrospective MIMIC-IV + eICU validation remains aggregate and DUA-safe.
        </div>
      </div>

      <aside class="panel side-panel">
        <h2>Command summary</h2>
        <div class="side-grid">
          <div class="side-card">
            <small>Open review items</small>
            <b id="sideOpen">4</b>
            <p>Number of currently visible queue items.</p>
          </div>
          <div class="side-card">
            <small>Higher-priority items</small>
            <b id="sideHigh">1</b>
            <p>Critical tier items in current snapshot.</p>
          </div>
          <div class="side-card">
            <small>Avg review score</small>
            <b id="sideAverage">7.4/10</b>
            <p>Average score on the 0–10 queue scale.</p>
          </div>
          <div class="side-card">
            <small>System status</small>
            <b>Connected</b>
            <p>Admin role, all-units scope, MFA active.</p>
          </div>
        </div>
        <div class="guardrail">
          Workflow buttons are audit/workflow-state examples only.
        </div>
      </aside>
    </section>

    <details class="about-demo">
  <summary>About this demo, metric clarity, and governance links</summary>
  <section class="lower">
      <div class="panel">
        <h3>Explainability shown visually</h3>
        <ul>
          <li>Priority tier appears as a badge.</li>
          <li>Primary driver is visible in the queue.</li>
          <li>Trend and lead-time context are not buried in paragraphs.</li>
          <li>Selected patient detail shows vitals and short bullets.</li>
        </ul>
      </div>
      <div class="panel">
        <h3>Metric clarity</h3>
        <p>
          Review Score is always shown as 0–10. Aggregate validation percentages such as alert reduction,
          FPR, detection, and lead-time stay in validation pages and evidence packets.
        </p>
      </div>
      <div class="panel">
        <h3>Governance links</h3>
        <ul>
          <li><a href="/command-center/deck">Governance Deck</a></li>
          <li><a href="/model-card">Model Card</a></li>
          <li><a href="/pilot-success-guide">Pilot Success Guide</a></li>
          <li><a href="/validation-evidence">Validation Evidence</a></li>
        </ul>
      </div>
    </section>
</details>
  </main>

  <script>
    const snapshots = [
      {
        label:"Snapshot 1/4",
        rows:[
          {patient:"ICU-12",unit:"ICU",tier:"Critical",score:9.2,driver:"SpO2 decline",trend:"Worsening",lead:"~4.8 hrs",workflow:"Needs review",vitals:{SpO2:"88% ↓",HR:"127 bpm",BP:"168/99",RR:"29/min",Temp:"100.8°F"},why:["Oxygen saturation below threshold and trending down.","Heart rate elevated relative to baseline.","Respiratory rate outside normal range.","Lead-time context supports higher review priority."]},
          {patient:"ICU-07",unit:"ICU",tier:"Elevated",score:8.1,driver:"BP instability",trend:"Worsening",lead:"~4.1 hrs",workflow:"Acknowledged",vitals:{SpO2:"92%",HR:"118 bpm",BP:"91/58",RR:"24/min",Temp:"99.7°F"},why:["Blood-pressure instability is the primary driver.","Trend is worsening across the visible window.","Lead-time context remains relevant for review."]},
          {patient:"TEL-18",unit:"Telemetry",tier:"Elevated",score:7.5,driver:"HR instability",trend:"Stable",lead:"~3.4 hrs",workflow:"Acknowledged",vitals:{SpO2:"8.4/10",HR:"132 bpm",BP:"138/84",RR:"23/min",Temp:"99.1°F"},why:["Heart-rate instability remains visible.","Current trend is stable rather than worsening.","Elevated tier supports continued review."]},
          {patient:"SDU-04",unit:"Stepdown",tier:"Watch",score:6.8,driver:"RR elevation",trend:"Stable",lead:"~2.9 hrs",workflow:"Monitoring",vitals:{SpO2:"95%",HR:"101 bpm",BP:"128/76",RR:"25/min",Temp:"98.8°F"},why:["Respiratory rate is elevated but stable.","No critical driver is dominant in this snapshot.","Monitoring state remains appropriate."]}
        ]
      },
      {
        label:"Snapshot 2/4",
        rows:[
          {patient:"TEL-18",unit:"Telemetry",tier:"Critical",score:8.6,driver:"HR instability",trend:"Worsening",lead:"~4.0 hrs",workflow:"Needs review",vitals:{SpO2:"93%",HR:"138 bpm",BP:"141/86",RR:"24/min",Temp:"99.4°F"},why:["Heart-rate instability increased compared with prior snapshot.","Trend moved to worsening.","Telemetry case rises to top queue position."]},
          {patient:"ICU-12",unit:"ICU",tier:"Elevated",score:7.4,driver:"SpO2 recovery watch",trend:"Stable / Watch",lead:"~3.2 hrs",workflow:"Assigned",vitals:{SpO2:"91%",HR:"116 bpm",BP:"156/94",RR:"25/min",Temp:"99.8°F"},why:["ICU-12 remains visible but is no longer fixed as top priority.","Oxygenation remains under watch but trend is less severe.","Assigned workflow state is shown."]},
          {patient:"ICU-07",unit:"ICU",tier:"Elevated",score:7.1,driver:"BP trend",trend:"Watch",lead:"~3.0 hrs",workflow:"Acknowledged",vitals:{SpO2:"8.4/10",HR:"108 bpm",BP:"96/62",RR:"22/min",Temp:"99.0°F"},why:["Blood-pressure trend remains a review driver.","Trend supports watch/elevated review rather than critical tier.","Acknowledged workflow state remains visible."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:6.1,driver:"BP trend",trend:"Stable",lead:"~2.5 hrs",workflow:"Monitoring",vitals:{SpO2:"96%",HR:"94 bpm",BP:"149/90",RR:"19/min",Temp:"98.6°F"},why:["Blood-pressure trend is visible but stable.","Watch tier keeps this lower in the review queue.","Monitoring continues."]}
        ]
      },
      {
        label:"Snapshot 3/4",
        rows:[
          {patient:"SDU-04",unit:"Stepdown",tier:"Critical",score:8.4,driver:"RR elevation",trend:"Worsening",lead:"~4.2 hrs",workflow:"Needs review",vitals:{SpO2:"90%",HR:"121 bpm",BP:"144/91",RR:"31/min",Temp:"100.1°F"},why:["Respiratory trend worsened.","RR elevation is the primary driver.","Lead-time context moved this patient to top rank."]},
          {patient:"ICU-12",unit:"ICU",tier:"Watch",score:6.7,driver:"SpO2 stable",trend:"Stable",lead:"~2.6 hrs",workflow:"Monitoring",vitals:{SpO2:"93%",HR:"108 bpm",BP:"145/88",RR:"21/min",Temp:"99.1°F"},why:["ICU-12 is not highest priority in this snapshot.","SpO2 is stable compared with prior snapshot.","Monitoring is the current workflow state."]},
          {patient:"TEL-18",unit:"Telemetry",tier:"Elevated",score:7.5,driver:"HR instability",trend:"Stable / Watch",lead:"~3.4 hrs",workflow:"Acknowledged",vitals:{SpO2:"95%",HR:"124 bpm",BP:"133/82",RR:"22/min",Temp:"98.6°F"},why:["HR instability remains visible.","Trend is no longer worsening.","Acknowledged workflow state is maintained."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:5.9,driver:"BP trend",trend:"Stable",lead:"~2.1 hrs",workflow:"Monitoring",vitals:{SpO2:"96%",HR:"94 bpm",BP:"146/89",RR:"18/min",Temp:"98.4°F"},why:["Blood-pressure trend is stable.","Lower review score keeps this item lower in queue.","Monitoring continues."]}
        ]
      },
      {
        label:"Snapshot 4/4",
        rows:[
          {patient:"ICU-21",unit:"ICU",tier:"Critical",score:8.3,driver:"BP instability",trend:"Worsening",lead:"~3.9 hrs",workflow:"Needs review",vitals:{SpO2:"92%",HR:"122 bpm",BP:"172/96",RR:"24/min",Temp:"99.9°F"},why:["Blood-pressure instability is the top driver.","Worsening trend raises priority.","Lead-time context supports top queue placement."]},
          {patient:"ICU-12",unit:"ICU",tier:"Elevated",score:7.6,driver:"SpO2 watch",trend:"Stable / Watch",lead:"~3.3 hrs",workflow:"Assigned",vitals:{SpO2:"92%",HR:"115 bpm",BP:"150/91",RR:"24/min",Temp:"99.5°F"},why:["ICU-12 remains visible but not static.","Review score and tier changed from prior snapshots.","Assigned workflow state is shown."]},
          {patient:"TEL-18",unit:"Telemetry",tier:"Elevated",score:7.1,driver:"HR variability",trend:"Stable",lead:"~2.9 hrs",workflow:"Acknowledged",vitals:{SpO2:"95%",HR:"118 bpm",BP:"129/80",RR:"21/min",Temp:"98.5°F"},why:["HR variability remains a driver.","Trend is stable.","Acknowledged workflow state."]},
          {patient:"WARD-21",unit:"Ward",tier:"Watch",score:6.0,driver:"BP trend",trend:"Stable",lead:"~2.4 hrs",workflow:"Monitoring",vitals:{SpO2:"96%",HR:"93 bpm",BP:"148/90",RR:"18/min",Temp:"98.6°F"},why:["Blood-pressure trend is stable.","Watch tier keeps this lower in review queue.","Monitoring continues."]}
        ]
      }
    ];

    let snapshotIndex = 0;
    let currentFilter = "all";
    let currentUnitScope = "all";
    let paused = false;
    let selectedPatient = null;

    function tierClass(tier){
      if(tier === "Critical") return "tier-critical";
      if(tier === "Elevated") return "tier-elevated";
      if(tier === "Watch") return "tier-watch";
      return "tier-low";
    }

    function normalizeUnit(unit){
      const u = String(unit || "").toLowerCase();
      if(u.includes("icu")) return "ICU";
      if(u.includes("tele")) return "Telemetry";
      if(u.includes("step")) return "Stepdown";
      if(u.includes("ward")) return "Ward";
      return unit || "Unknown";
    }

    function rows(){
      const all = snapshots[snapshotIndex].rows;
      return all.filter(r => {
        const tierOk = currentFilter === "all" || r.tier === currentFilter;
        const unitOk = currentUnitScope === "all" || normalizeUnit(r.unit) === currentUnitScope;
        return tierOk && unitOk;
      });
    }

    function averageScore(list){
      if(!list.length) return "—";
      const avg = list.reduce((sum,r)=>sum+r.score,0)/list.length;
      return avg.toFixed(1) + "/10";
    }

    function render(){
      const snap = snapshots[snapshotIndex];
      const visible = rows();
      const displayRows = visible;

      if(!selectedPatient || !displayRows.some(r => r.patient === selectedPatient)){
        selectedPatient = displayRows[0].patient;
      }

      const critical = snap.rows.filter(r => r.tier === "Critical").length;

      document.getElementById("snapshotLabel").textContent = snap.label;
      const priorityText = currentFilter === "all" ? "All Priorities" : currentFilter;
      const unitText = currentUnitScope === "all" ? "All Units" : currentUnitScope;
      document.getElementById("filterLabel").textContent = unitText + " / " + priorityText;
      document.getElementById("statOpen").textContent = snap.rows.length;
      document.getElementById("statCritical").textContent = critical;
      document.getElementById("statAverage").textContent = averageScore(snap.rows);
      document.getElementById("sideOpen").textContent = snap.rows.length;
      document.getElementById("sideHigh").textContent = critical;
      document.getElementById("sideAverage").textContent = averageScore(snap.rows);
      document.getElementById("lastUpdated").textContent = "Updated: " + new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});

      const cards = document.getElementById("cards");
      cards.innerHTML = "";
      if(!displayRows.length){
        cards.innerHTML = `<div class="scope-note" style="grid-column:1/-1"><strong>No visible items</strong>No review items match the current view scope and priority filter.</div>`;
      }
      displayRows.forEach((r, idx) => {
        const card = document.createElement("article");
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
      if(!displayRows.length){
        body.innerHTML = `<tr><td colspan="8">No rows match the selected unit scope and priority filter.</td></tr>`;
      }
      displayRows.forEach((r, idx) => {
        const tr = document.createElement("tr");
        tr.onclick = () => { selectedPatient = r.patient; render(); };
        tr.innerHTML = `
          <td>#${idx + 1}</td>
          <td><strong>${r.patient}</strong><br><span style="color:#a9b8cc">${r.unit}</span></td>
          <td><span class="pill ${tierClass(r.tier)}">${r.tier}</span></td>
          <td class="score-cell">${r.score.toFixed(1)}/10</td>
          <td>${r.driver}</td>
          <td>${r.trend}</td>
          <td>${r.lead}</td>
          <td>
            <div class="action-row">
              <button class="small-action">Acknowledge</button>
              <button class="small-action">Assign</button>
              <button class="small-action">Escalate Review</button>
            </div>
          </td>
        `;
        body.appendChild(tr);
      });

      const selected = snap.rows.find(r => r.patient === selectedPatient) || displayRows[0];
      renderDetail(selected, snap.rows.indexOf(selected) + 1);
    }

    function renderDetail(r, rank){
      document.getElementById("detailPatient").textContent = r.patient;
      document.getElementById("detailSub").textContent = `${r.unit} • Rank #${rank} • ${r.workflow}`;
      const tier = document.getElementById("detailTier");
      tier.className = "pill " + tierClass(r.tier);
      tier.textContent = r.tier;
      document.getElementById("detailScore").innerHTML = `${r.score.toFixed(1)}<small>/10</small>`;

      document.getElementById("vitals").innerHTML = Object.entries(r.vitals).map(([k,v]) => `
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

    document.querySelectorAll("[data-unit-filter]").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-unit-filter]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentUnitScope = btn.dataset.unitFilter;
        selectedPatient = null;
        render();
      });
    });

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
  
/* === ERA FINAL PILOT POLISH RUNTIME === */
(function(){
  function polishCommandCenter(){
    try{
      var cards = Array.prototype.slice.call(document.querySelectorAll(".patient-card"));
      cards.forEach(function(card, index){
        card.classList.toggle("top-review-card", index === 0);
        var mini = card.querySelector(".mini");
        if(mini){
          mini.innerHTML = mini.innerHTML
            .replace(/<b>Driver:<\/b>/g, "<b>Driver</b>")
            .replace(/<b>Trend:<\/b>/g, "<b>Trend</b>")
            .replace(/<b>Lead:<\/b>/g, "<b>Lead</b>");
        }
      });

      var reason = document.getElementById("reason");
      if(reason){
        reason.innerHTML = reason.innerHTML
          .replace("is the selected primary driver.", "drives this review position.")
          .replace("Lead-time context:", "Lead:");
      }

      var lower = document.querySelector("section.lower");
      if(lower && !lower.closest("details.about-demo")){
        var details = document.createElement("details");
        details.className = "about-demo";
        var summary = document.createElement("summary");
        summary.textContent = "About this demo, metric clarity, and governance links";
        lower.parentNode.insertBefore(details, lower);
        details.appendChild(summary);
        details.appendChild(lower);
      }

      document.querySelectorAll(".patient-card .score, .score-cell, #detailScore").forEach(function(el){
        el.innerHTML = el.innerHTML
          .replace(/(\d{1,2}(?:\.\d)?)\s*\/\s*10/g, "$1<small>/10</small>")
          .replace(/%/g, "");
      });
    }catch(e){}
  }

  var originalRender = window.render;
  if(typeof originalRender === "function" && !window.__eraPilotPolishWrapped){
    window.render = function(){
      var result = originalRender.apply(this, arguments);
      setTimeout(polishCommandCenter, 0);
      return result;
    };
    window.__eraPilotPolishWrapped = true;
  }

  polishCommandCenter();
  setInterval(polishCommandCenter, 1200);
})();

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

@command_center_bp.route("/command-center/deck")
@command_center_bp.route("/command-center/governance-deck")
@command_center_bp.route("/governance-deck")
def command_center_deck() -> Response:
    deck_path = Path(__file__).resolve().parent / "command_center_deck.html"
    if deck_path.exists():
        return Response(deck_path.read_text(encoding="utf-8"), mimetype="text/html")
    fallback = """
    <!doctype html>
    <html>
      <head><title>Command Center Governance Deck</title></head>
      <body style="font-family:Arial;background:#08111f;color:white;padding:40px">
        <h1>Command Center Governance Deck</h1>
        <p>The governance deck file is not currently present. Use /pilot-docs, /model-card, and /pilot-success-guide for governance documentation.</p>
        <p><a style="color:#9bd7ff" href="/command-center">Back to Command Center</a></p>
      </body>
    </html>
    """
    return Response(fallback, mimetype="text/html")

def register_command_center_routes(app):
    existing_rules = {str(rule.rule) for rule in app.url_map.iter_rules()}
    if "/command-center" not in existing_rules:
        app.add_url_rule("/command-center", "command_center_page", command_center_page)
    if "/command-center/deck" not in existing_rules:
        app.add_url_rule("/command-center/deck", "command_center_deck_page", command_center_deck)

bp = command_center_bp
get_command_center_html = lambda: COMMAND_CENTER_HTML
render_command_center = command_center_page

__all__ = [
    "command_center_bp",
    "bp",
    "command_center",
    "command_center_page",
    "command_center_deck",
    "register_command_center_routes",
    "get_command_center_html",
    "render_command_center",
    "COMMAND_CENTER_HTML",
]


# ERA_COMMAND_CENTER_DECK_ROUTE_FIX_START

COMMAND_CENTER_DECK_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Early Risk Alert AI — Governance Deck</title>
  <style>
    :root{
      --bg:#06101d;--panel:#101b2d;--panel2:#142238;--line:rgba(184,211,255,.16);
      --text:#f7fbff;--muted:#a9b8cc;--green:#a7ff9a;--amber:#ffd36d;--blue:#9bd7ff;
    }
    *{box-sizing:border-box}
    body{
      margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);background:radial-gradient(circle at 10% 0%,rgba(78,166,255,.16),transparent 30%),linear-gradient(180deg,#06101d,#050912);
    }
    a{color:inherit;text-decoration:none}
    .wrap{max-width:1320px;margin:0 auto;padding:24px 18px 70px}
    .top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;margin-bottom:20px}
    .kicker{color:#b8ffd0;font-size:11px;letter-spacing:.18em;text-transform:uppercase;font-weight:1000}
    h1{margin:6px 0 8px;font-size:clamp(34px,5vw,64px);line-height:.9;letter-spacing:-.06em}
    p{color:#d8e3f2;font-size:14px;line-height:1.45;font-weight:760}
    .nav{display:flex;gap:8px;flex-wrap:wrap}
    .nav a,.pill{
      border:1px solid var(--line);background:rgba(255,255,255,.06);border-radius:999px;
      padding:8px 11px;font-weight:900;font-size:12px;white-space:nowrap;
    }
    .pill.green{color:#d7ffd1;border-color:rgba(167,255,154,.42);background:rgba(167,255,154,.10)}
    .pill.amber{color:#ffe6a2;border-color:rgba(255,211,109,.52);background:rgba(255,211,109,.12)}
    .hero{
      border:1px solid rgba(255,211,109,.34);background:linear-gradient(90deg,rgba(255,211,109,.10),rgba(167,255,154,.06));
      border-radius:22px;padding:18px;margin-bottom:16px;
    }
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
    .card{
      background:linear-gradient(180deg,rgba(20,34,56,.94),rgba(11,20,35,.94));
      border:1px solid var(--line);border-radius:20px;padding:18px;min-height:150px;
      box-shadow:0 18px 70px rgba(0,0,0,.25);
    }
    h2{margin:0 0 10px;font-size:24px;letter-spacing:-.04em}
    h3{margin:0 0 8px;color:#d8efff;font-size:13px;letter-spacing:.14em;text-transform:uppercase}
    ul{margin:0;padding-left:18px;color:#d8e3f2;font-size:14px;line-height:1.55;font-weight:760}
    table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border-radius:14px;border:1px solid var(--line);font-size:13px}
    th{background:rgba(154,190,255,.19);font-size:11px;text-transform:uppercase;letter-spacing:.12em;text-align:left;padding:10px}
    td{border-top:1px solid rgba(181,211,255,.12);padding:10px;color:#f5f9ff;font-weight:800;vertical-align:top}
    .footer{
      margin-top:18px;border:1px solid rgba(255,211,109,.38);background:rgba(255,211,109,.075);
      color:#ffe6a2;border-radius:18px;padding:13px;text-align:center;font-size:12px;font-weight:900;line-height:1.4
    }
    @media(max-width:850px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="top">
      <div>
        <div class="kicker">HCP-facing decision support • governance deck</div>
        <h1>Pilot Governance + Documentation</h1>
        <p>
          One stable page for intended use, support claims, visible limitations, role/unit scoping,
          audit controls, cybersecurity posture, and V&V-lite pilot readiness.
        </p>
      </div>
      <nav class="nav">
        <a href="/command-center">Command Center</a>
        <a href="/pilot-docs">Pilot Docs</a>
        <a href="/model-card">Model Card</a>
        <a href="/pilot-success-guide">Pilot Success Guide</a>
        <a href="/validation-evidence">Evidence Packet</a>
        <a href="/validation-intelligence">Validation Intelligence</a>
      </nav>
    </div>

    <section class="hero">
      <span class="pill green">Pilot governance route restored</span>
      <span class="pill amber">Decision support only</span>
      <p>
        This deck keeps the governance material separate from the live Command Center so the Command Center can stay compact,
        interactive, and queue-focused.
      </p>
    </section>

    <section class="grid">
      <div class="card">
        <h3>Frozen intended use</h3>
        <p>
          Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized
          healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization,
          and improve command-center operational awareness.
        </p>
        <p>
          It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.
        </p>
      </div>

      <div class="card">
        <h3>Pilot controls</h3>
        <ul>
          <li>Outputs are patient-prioritization support, not clinical directives.</li>
          <li>Review basis, confidence, limitations, freshness, and unknowns remain visible.</li>
          <li>Workflow controls record operational handling and user action logs only.</li>
          <li>Role, unit, and access scope should be confirmed before pilot launch.</li>
        </ul>
      </div>

      <div class="card">
        <h3>Approved support language</h3>
        <ul>
          <li>Supports patient prioritization across monitored patients.</li>
          <li>Provides explainable review-support context.</li>
          <li>Supports command-center workflow awareness and operational visibility.</li>
          <li>Helps identify patients who may warrant further review.</li>
        </ul>
      </div>

      <div class="card">
        <h3>Claims to avoid</h3>
        <ul>
          <li>Detects deterioration autonomously.</li>
          <li>Predicts who will clinically crash.</li>
          <li>Directs bedside intervention.</li>
          <li>Replaces clinician judgment.</li>
          <li>Independently triggers escalation.</li>
        </ul>
      </div>

      <div class="card">
        <h3>Risk register</h3>
        <table>
          <thead>
            <tr><th>ID</th><th>Risk</th><th>Mitigation</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td>R-001</td><td>Claims drift</td><td>Freeze intended-use language and review claims before release.</td><td>Open</td></tr>
            <tr><td>R-002</td><td>Explainability over-reliance</td><td>Show review basis, limitations, freshness, and unknowns.</td><td>In place</td></tr>
            <tr><td>R-003</td><td>Workflow confusion</td><td>Keep acknowledge, assign, escalate, and resolve as workflow states only.</td><td>In place</td></tr>
            <tr><td>R-004</td><td>Access scope</td><td>Maintain role and unit restrictions with filtered worklists.</td><td>In place</td></tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <h3>V&V-lite sheet</h3>
        <table>
          <thead>
            <tr><th>ID</th><th>Check</th><th>Evidence</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td>VV-001</td><td>Role/unit scope respected</td><td>Route and access-context checks.</td><td>Pass</td></tr>
            <tr><td>VV-002</td><td>Workflow actions remain operational</td><td>Audit/action routes and UI workflow controls.</td><td>Pass</td></tr>
            <tr><td>VV-003</td><td>Explainability fields visible</td><td>Queue + selected-patient detail panel.</td><td>Pass</td></tr>
            <tr><td>VV-004</td><td>Decision-support guardrails visible</td><td>Header, banner, page footer, and governance deck.</td><td>Pass</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI • HCP-facing decision support and workflow support • Not intended to diagnose, direct treatment,
      replace clinician judgment, or independently trigger escalation.
    </div>
  </main>
</body>
</html>
"""

def command_center_deck_page() -> Response:
    return Response(COMMAND_CENTER_DECK_HTML, mimetype="text/html")

try:
    command_center_bp.add_url_rule("/command-center/deck", "command_center_deck", command_center_deck_page)
except Exception:
    pass

def register_command_center_routes(app):
    existing_rules = {str(rule.rule) for rule in app.url_map.iter_rules()}
    if "/command-center" not in existing_rules:
        app.add_url_rule("/command-center", "command_center_page", command_center_page)
    if "/command-center/deck" not in existing_rules:
        app.add_url_rule("/command-center/deck", "command_center_deck_page", command_center_deck_page)

# ERA_COMMAND_CENTER_DECK_ROUTE_FIX_END

