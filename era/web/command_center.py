COMMAND_CENTER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — AI Hospital Command Center</title>
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
      --blue2:#5bd4ff;
      --green:#3ad38f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --purple:#b58cff;
      --shadow:0 20px 60px rgba(0,0,0,.34);
      --radius:24px;
      --max:1460px;
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
    .shell{max-width:var(--max);margin:0 auto;padding:22px 16px 48px}
    .topbar{
      position:sticky;top:0;z-index:50;
      border-bottom:1px solid var(--line);
      background:rgba(7,16,28,.82);
      backdrop-filter:blur(14px);
    }
    .topbar-inner{
      max-width:var(--max);margin:0 auto;padding:14px 16px;
      display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;
    }
    .brand-title{
      font-size:clamp(30px,3.4vw,48px);font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .brand-sub{font-size:14px;font-weight:800;color:var(--muted)}
    .nav-links{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
    .nav-links a{font-size:14px;font-weight:900;color:#dce9ff}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      border:1px solid transparent;
      transition:transform .18s ease, box-shadow .18s ease, opacity .18s ease;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#07101c;box-shadow:0 12px 30px rgba(91,212,255,.22);
    }
    .btn.secondary{
      background:rgba(255,255,255,.04);border-color:var(--line);color:var(--text);
    }

    .alert-actions-panel{
  display:flex;
  gap:10px;
  margin-top:12px;
}

.alert-action{
  border:1px solid rgba(255,255,255,.08);
  background:rgba(255,255,255,.03);
  color:#e6f0ff;
  padding:10px 14px;
  border-radius:10px;
  font-weight:800;
  letter-spacing:.06em;
  cursor:pointer;
  transition:all .18s ease;
}

.alert-action:hover{
  transform:translateY(-1px);
  border-color:rgba(91,212,255,.25);
}

.alert-action.ack{
  background:linear-gradient(135deg,#5bd38d,#2fbf71);
  color:#07101c;
}

.alert-action.escalate{
  background:linear-gradient(135deg,#ff6b6b,#ff8b8b);
  color:#07101c;
}

.alert-action.assign{
  background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
  color:#07101c;
}

    .ticker-wrap{
      margin-top:18px;
      border:1px solid var(--line);
      border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.82), rgba(8,15,28,.92));
      box-shadow:var(--shadow);
      overflow:hidden;
    }
    .ticker-track{
      display:flex;gap:16px;align-items:center;
      padding:12px 0;
      white-space:nowrap;
      min-width:max-content;
      animation:tickerMove 28s linear infinite;
    }
    .ticker-item{
      display:inline-flex;align-items:center;gap:10px;
      border:1px solid var(--line);
      border-radius:999px;
      padding:10px 14px;
      background:rgba(255,255,255,.03);
      margin-left:14px;
      font-size:14px;font-weight:800;color:#deebff;
    }
    .ticker-dot{
      width:10px;height:10px;border-radius:50%;
      box-shadow:0 0 20px currentColor;
      flex:0 0 auto;
    }
    .dot-green{color:var(--green);background:var(--green)}
    .dot-amber{color:var(--amber);background:var(--amber)}
    .dot-red{color:var(--red);background:var(--red)}
    .dot-blue{color:var(--blue2);background:var(--blue2)}
    @keyframes tickerMove{
      0%{transform:translateX(0)}
      100%{transform:translateX(-50%)}
    }

    .hero{
      margin-top:18px;
      border:1px solid var(--line);
      border-radius:30px;
      background:
        radial-gradient(circle at 20% 20%, rgba(91,212,255,.08), transparent 18%),
        radial-gradient(circle at 80% 0%, rgba(181,140,255,.08), transparent 18%),
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015)),
        linear-gradient(180deg, rgba(10,19,34,.86), rgba(7,14,26,.96));
      box-shadow:var(--shadow);
      padding:24px;
    }
    .hero-grid{
      display:grid;grid-template-columns:1.45fr .95fr;gap:18px;align-items:stretch;
    }
    .hero-copy{
      border:1px solid var(--line);
      border-radius:26px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:22px;
    }
    .hero-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#91ddff;margin-bottom:10px;
    }
    .hero-copy h1{
      margin:0 0 12px;font-size:clamp(40px,5.6vw,76px);line-height:.92;letter-spacing:-.06em;font-weight:1000;
    }
    .hero-copy p{
      margin:0;color:#d0def2;font-size:17px;line-height:1.68;max-width:820px;
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}
    .hero-mini-grid{
      margin-top:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
    }
    .hero-mini{
      border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
    }
    .hero-mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fdcff;margin-bottom:8px;
    }
    .hero-mini .v{
      font-size:18px;font-weight:1000;line-height:1.1;letter-spacing:-.03em;
    }

    .mission-panel{
      border:1px solid var(--line);
      border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;
      position:relative;
      overflow:hidden;
    }
    .mission-panel::before{
      content:"";
      position:absolute;inset:0;
      background:
        linear-gradient(120deg, transparent 30%, rgba(255,255,255,.04) 50%, transparent 70%);
      animation:panelSweep 9s linear infinite;
      pointer-events:none;
    }
    @keyframes panelSweep{
      0%{transform:translateX(-35%)}
      100%{transform:translateX(35%)}
    }
    .mission-header{
      display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px;
    }
    .mission-title{font-size:24px;font-weight:1000;letter-spacing:-.04em}
    .mission-sub{font-size:13px;color:var(--muted);line-height:1.55}
    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:10px 14px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.12);
      background:rgba(255,255,255,.05);
    }
    .status-pill.live{color:#0b1528;background:linear-gradient(135deg,var(--green),#7ef5c0);box-shadow:0 0 24px rgba(58,211,143,.18)}
    .status-pill.watch{color:#2a1b00;background:linear-gradient(135deg,var(--amber),#ffdba5)}
    .status-pill.critical{color:#fff3f5;background:linear-gradient(135deg,#ff667d,#ff8e99)}
    

    .cockpit-grid{
      margin-top:20px;
      display:grid;
      grid-template-columns:1.75fr 1.05fr;
      gap:18px;
      align-items:start;
    }

    .risk-actions{
      display:flex;
      gap:10px;
      margin-top:10px;
    }

    .risk-btn{
      border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.03);
      color:#eef4ff;
      padding:8px 12px;
      border-radius:8px;
      font-size:12px;
      font-weight:800;
      cursor:pointer;
    }

    .telemetry-wall,
    .intel-column,
    .ops-strip{
      display:grid;
      gap:18px;
    }

    .telemetry-top{
      display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
    }
    .stat-card{
      border:1px solid var(--line);
      border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(13,23,40,.90), rgba(8,15,28,.95));
      padding:16px;
      min-height:122px;
      position:relative;
      overflow:hidden;
    }
    .stat-card::after{
      content:"";
      position:absolute;right:-30px;top:-30px;width:120px;height:120px;border-radius:50%;
      background:radial-gradient(circle, rgba(91,212,255,.12), transparent 60%);
      pointer-events:none;
    }
    .stat-card .k{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8ddcff;margin-bottom:10px;
    }
    .stat-card .v{
      font-size:44px;font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .stat-card .hint{
      margin-top:8px;font-size:13px;line-height:1.5;color:#c8d8ef;
    }

    .monitor-grid{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:16px;
    }

    .monitor{
      border:1px solid var(--line);
      border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.012)),
        linear-gradient(180deg, rgba(11,20,35,.96), rgba(6,12,22,.98));
      box-shadow:0 16px 44px rgba(0,0,0,.32);
      padding:18px;
      position:relative;
      overflow:hidden;
      min-height:350px;
    }
    .monitor::before{
      content:"";
      position:absolute;inset:0;
      background:
        repeating-linear-gradient(
          to right,
          rgba(255,255,255,.03) 0,
          rgba(255,255,255,.03) 1px,
          transparent 1px,
          transparent 58px
        ),
        repeating-linear-gradient(
          to bottom,
          rgba(255,255,255,.025) 0,
          rgba(255,255,255,.025) 1px,
          transparent 1px,
          transparent 40px
        );
      opacity:.12;
      pointer-events:none;
    }
    .monitor-top{
      position:relative;z-index:1;
      display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
      margin-bottom:14px;
    }
    .monitor-bed{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#95ddff;
      margin-bottom:6px;
    }
    .monitor-title{
      font-size:34px;font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .monitor-wave{
      position:relative;z-index:1;
      border:1px solid rgba(255,255,255,.06);
      border-radius:20px;
      min-height:138px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01)),
        rgba(0,0,0,.18);
      overflow:hidden;
      display:flex;align-items:center;
      padding:8px 0;
    }
    .monitor-wave svg{width:100%;height:118px;display:block}
    .ecg-path{
      fill:none;
      stroke-width:4;
      stroke-linecap:round;
      stroke-linejoin:round;
      stroke-dasharray:920;
      stroke-dashoffset:0;
      animation:ecgMove 2.8s linear infinite;
      filter:drop-shadow(0 0 10px currentColor);
    }
    @keyframes ecgMove{
      0%{stroke-dashoffset:920}
      100%{stroke-dashoffset:0}
    }
    .ecg-green{stroke:#77ffb4;color:#77ffb4}
    .ecg-amber{stroke:#ffd96c;color:#ffd96c}
    .ecg-red{stroke:#ff7c8d;color:#ff7c8d}

    .monitor-metrics{
      position:relative;z-index:1;
      margin-top:14px;
      display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
    }
    .metric-box{
      border:1px solid rgba(255,255,255,.06);
      border-radius:16px;
      background:rgba(255,255,255,.03);
      padding:12px;
      min-height:78px;
      display:flex;flex-direction:column;justify-content:space-between;
    }
    .metric-k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb8dc;
    }
    .metric-v{
      font-size:18px;font-weight:1000;line-height:1;letter-spacing:-.03em;
    }
    .monitor-story{
      position:relative;z-index:1;
      margin-top:14px;
      display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
    }
    .story-text{
      font-size:13px;color:#cfe0f4;line-height:1.55;max-width:78%;
    }

    .wall-toolbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
  flex-wrap:wrap;
  margin-bottom:14px;
}

.wall-toolbar-left{
  display:flex;
  align-items:center;
  gap:12px;
}

.wall-title-block{
  display:flex;
  flex-direction:column;
  gap:4px;
}

.wall-kicker{
  font-size:11px;
  font-weight:900;
  letter-spacing:.16em;
  text-transform:uppercase;
  color:#8fdcff;
}

.wall-title{
  font-size:22px;
  font-weight:1000;
  letter-spacing:-.03em;
  color:#eef4ff;
}

.wall-toolbar-right{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}

.unit-filter{
  border:1px solid rgba(255,255,255,.08);
  background:rgba(255,255,255,.03);
  color:#dce9ff;
  border-radius:999px;
  padding:10px 14px;
  font-size:12px;
  font-weight:900;
  letter-spacing:.08em;
  text-transform:uppercase;
  cursor:pointer;
  transition:all .18s ease;
}

.unit-filter:hover{
  transform:translateY(-1px);
  border-color:rgba(91,212,255,.25);
}

.unit-filter.active{
  background:linear-gradient(135deg,var(--blue),var(--blue2));
  color:#07101c;
  border-color:transparent;
  box-shadow:0 10px 24px rgba(91,212,255,.20);
}

.command-wall-frame{
  border:1px solid var(--line);
  border-radius:24px;
  background:
    linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015)),
    linear-gradient(180deg, rgba(9,16,28,.96), rgba(6,12,22,.98));
  padding:16px;
  box-shadow:0 16px 44px rgba(0,0,0,.28);
}

@media (max-width:900px){
  .wall-toolbar{
    flex-direction:column;
    align-items:flex-start;
  }
}

    .intel-column{
      align-content:start;
    }
    .intel-card{
      border:1px solid var(--line);
      border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;
      box-shadow:0 14px 36px rgba(0,0,0,.26);
    }
    .intel-card h3{
      margin:0 0 12px;font-size:20px;font-weight:1000;letter-spacing:-.03em;
    }
    .intel-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
    }
    .intel-metric{
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
      min-height:100px;
    }
    .intel-label{
      display:block;font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb8dc;margin-bottom:8px;
    }
    .intel-value{
      display:block;font-size:38px;font-weight:1000;line-height:.92;letter-spacing:-.04em;
    }
    .intel-note{
      margin-top:8px;font-size:13px;line-height:1.5;color:#cadbf0;
    }

    .alert-feed{
      display:grid;gap:10px;
    }
    .alert-item{
      display:flex;align-items:flex-start;justify-content:space-between;gap:10px;
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
    }
    .alert-copy{
      font-size:14px;font-weight:900;line-height:1.45;color:#e4efff;
    }
    .alert-sub{
      font-size:12px;color:#acc0dd;font-weight:700;margin-top:4px;
    }

    .why-now{
      display:grid;gap:10px;
    }
    .why-line{
      border:1px solid rgba(255,255,255,.06);
      border-radius:16px;padding:12px;background:rgba(255,255,255,.03);
      font-size:14px;font-weight:800;line-height:1.5;color:#def0ff;
    }

    .ops-strip{
      margin-top:18px;
      display:grid;
      grid-template-columns:repeat(6,1fr);
      gap:16px;
    }
    .ops-card{
      border:1px solid var(--line);
      border-radius:22px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:16px;
      box-shadow:0 14px 34px rgba(0,0,0,.24);
      min-height:170px;
    }
    .ops-card h3{
      margin:0 0 12px;font-size:18px;font-weight:1000;letter-spacing:-.03em;
    }
    .trend-bar{
      position:relative;
      height:16px;border-radius:999px;overflow:hidden;
      background:rgba(255,255,255,.05);
      border:1px solid rgba(255,255,255,.06);
      margin-top:14px;
    }
    .trend-fill{
      position:absolute;left:-36%;top:0;bottom:0;width:60%;
      background:linear-gradient(90deg, rgba(91,212,255,.10), rgba(91,212,255,.82), rgba(91,212,255,.10));
      animation:trendMove 3.6s linear infinite;
      filter:blur(.2px);
    }
    @keyframes trendMove{
      0%{transform:translateX(0)}
      100%{transform:translateX(230%)}
    }
    .trend-labels{
      display:flex;justify-content:space-between;gap:8px;margin-top:12px;
      font-size:12px;font-weight:900;color:#b2c6e3;
    }

    .forecast{
      display:grid;gap:10px;margin-top:10px;
    }
    .forecast div{
      border:1px solid rgba(255,255,255,.06);
      border-radius:14px;padding:10px 12px;background:rgba(255,255,255,.03);
      font-size:14px;font-weight:900;color:#e3efff;
    }
    .forecast-mid{color:#ffd27d}
    .forecast-high{color:#ffb162}
    .forecast-critical{color:#ff7f91}

    .cluster{
      border:1px solid rgba(255,255,255,.06);
      border-radius:14px;padding:10px 12px;background:rgba(255,255,255,.03);
      font-size:14px;font-weight:900;color:#e3efff;margin-bottom:8px;
    }

    .capacity-block{text-align:left}
    .capacity-value{
      display:block;font-size:44px;font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .capacity-label{
      display:block;margin-top:8px;font-size:14px;font-weight:800;color:#d2e1f4;
    }
    .capacity-note{
      margin-top:10px;font-size:12px;line-height:1.55;color:#a9bedb;
    }

    .heatmap{
      display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px;
    }
    .heat{
      height:28px;border-radius:8px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.10);
    }
    .low{background:linear-gradient(135deg,#46d88f,#8cffc1)}
    .medium{background:linear-gradient(135deg,#ffd965,#ffe79e)}
    .high{background:linear-gradient(135deg,#ff7b71,#ff9f95)}

    .queue-list{display:grid;gap:10px}
    .queue-item{
      display:flex;align-items:center;justify-content:space-between;gap:10px;
      border:1px solid rgba(255,255,255,.06);
      border-radius:14px;padding:10px 12px;background:rgba(255,255,255,.03);
    }
    .queue-copy{
      font-size:13px;font-weight:900;line-height:1.45;color:#e3efff;
    }

    @media (max-width:1280px){
      .hero-grid{grid-template-columns:1fr}
      .cockpit-grid{grid-template-columns:1fr}
      .ops-strip{grid-template-columns:repeat(3,1fr)}
    }
    @media (max-width:900px){
      .hero-mini-grid{grid-template-columns:repeat(2,1fr)}
      .monitor-grid{grid-template-columns:1fr}
      .telemetry-top{grid-template-columns:1fr}
      .intel-grid{grid-template-columns:1fr}
      .ops-strip{grid-template-columns:1fr}
      .monitor-metrics{grid-template-columns:repeat(2,1fr)}
      .story-text{max-width:100%}
    }
    @media (max-width:640px){
      .shell{padding:16px 10px 36px}
      .hero{padding:16px}
      .hero-copy h1{font-size:clamp(32px,11vw,52px)}
      .hero-actions{flex-direction:column}
      .btn{width:100%}
      .hero-mini-grid{grid-template-columns:1fr}
      .monitor-metrics{grid-template-columns:1fr}
      .topbar-inner{padding:12px 10px}
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="brand-kicker">AI-powered predictive clinical intelligence</div>
        <div class="brand-title">Early Risk Alert AI</div>
        <div class="brand-sub">Hospitals · Clinics · Investors · Insurers · Patients</div>
      </div>
      <div class="nav-links">
        <a href="/">Overview</a>
        <a href="/hospital-demo">Hospitals</a>
        <a href="/investor-intake">Investors</a>
        <a href="/admin/review">Admin Review</a>
        <a class="btn primary" href="/admin/review">Open Operating Layer</a>
      </div>
    </div>
  </div>

  <div class="shell">

    <div class="ticker-wrap">
      <div class="ticker-track" id="ticker-track">
        <div class="ticker-item"><span class="ticker-dot dot-red"></span>Critical deterioration signal surfaced · p101 · Risk 9.3</div>
        <div class="ticker-item"><span class="ticker-dot dot-amber"></span>High-priority risk escalation · p104 · Risk 7.4</div>
        <div class="ticker-item"><span class="ticker-dot dot-green"></span>Stable patient trend · p105 · Risk 3.9</div>
        <div class="ticker-item"><span class="ticker-dot dot-blue"></span>Command center telemetry synced · Mission mode active</div>
        <div class="ticker-item"><span class="ticker-dot dot-green"></span>Respiratory cluster reduced · p103 · Recovery trend</div>

        <div class="ticker-item"><span class="ticker-dot dot-red"></span>Critical deterioration signal surfaced · p101 · Risk 9.3</div>
        <div class="ticker-item"><span class="ticker-dot dot-amber"></span>High-priority risk escalation · p104 · Risk 7.4</div>
        <div class="ticker-item"><span class="ticker-dot dot-green"></span>Stable patient trend · p105 · Risk 3.9</div>
        <div class="ticker-item"><span class="ticker-dot dot-blue"></span>Command center telemetry synced · Mission mode active</div>
        <div class="ticker-item"><span class="ticker-dot dot-green"></span>Respiratory cluster reduced · p103 · Recovery trend</div>
      </div>
    </div>

    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">Hospital mission control</div>
          <h1>AI Hospital Command Center</h1>
          <p>
            Real-time clinical intelligence, predictive monitoring, telemetry-style deterioration detection,
            investor-ready operating visibility, and hospital command-center storytelling in one polished environment.
          </p>
          <div class="hero-actions">
            <a class="btn primary" href="/admin/review">Open Admin Review</a>
            <a class="btn secondary" href="/hospital-demo">Hospital Demo Form</a>
            <a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a>
            <a class="btn secondary" href="/investor-intake">Investor Intake</a>
          </div>
          <div class="hero-mini-grid">
            <div class="hero-mini">
              <div class="k">Telemetry</div>
              <div class="v">ECG-style clinical monitors</div>
            </div>
            <div class="hero-mini">
              <div class="k">Prediction</div>
              <div class="v">AI risk + forecast signals</div>
            </div>
            <div class="hero-mini">
              <div class="k">Operations</div>
              <div class="v">Capacity + staffing + triage</div>
            </div>
            <div class="hero-mini">
              <div class="k">Commercial</div>
              <div class="v">Hospital + investor readiness</div>
            </div>
          </div>
        </div>

        <div class="mission-panel">
          <div class="mission-header">
            <div>
              <div class="mission-title">Mission Status</div>
              <div class="mission-sub">
                Command center online · predictive telemetry active · live intake flows synchronized
              </div>
            </div>
            <div class="status-pill live">Live</div>
          </div>

          <div class="telemetry-top">
            <div class="stat-card">
              <div class="k">Open Alerts</div>
              <div class="v" id="open-alerts">5</div>
              <div class="hint">Active alerts surfaced by the intelligence layer.</div>
            </div>
            <div class="stat-card">
              <div class="k">Critical Alerts</div>
              <div class="v" id="critical-alerts">1</div>
              <div class="hint">Highest urgency patients needing immediate attention.</div>
            </div>
            <div class="stat-card">
              <div class="k">Avg Risk Score</div>
              <div class="v" id="avg-risk">6.2</div>
              <div class="hint">Average intelligence-layer risk level across flagged patients.</div>
            </div>
          </div>
        </div>
      </div>

      <div class="cockpit-grid">
  <div class="telemetry-wall">
    <div class="wall-toolbar">
  <div class="wall-toolbar-left">
    <div class="wall-title-block">
      <div class="wall-kicker">Hospital Command Wall</div>
      <div class="wall-title">Live Unit Monitor View</div>
    </div>
  </div>

  <div class="wall-toolbar-right">
    <button class="unit-filter active" data-unit="all" type="button">All Units</button>
    <button class="unit-filter" data-unit="icu" type="button">ICU</button>
    <button class="unit-filter" data-unit="stepdown" type="button">Stepdown</button>
    <button class="unit-filter" data-unit="telemetry" type="button">Telemetry</button>
    <button class="unit-filter" data-unit="ward" type="button">Ward</button>
    <button class="unit-filter" data-unit="rpm" type="button">RPM</button>
  </div>
</div>

<div class="command-wall-frame">
  <div class="monitor-grid" id="wall"></div>
</div>
  </div>
<div class="intel-column">

  <div class="intel-card">
    <h3>Live Alerts Feed</h3>
    <div class="alert-feed" id="queue"></div>

<div class="alert-actions-panel">
  <button class="alert-action ack">ACK</button>
  <button class="alert-action escalate">ESCALATE</button>
  <button class="alert-action assign">ASSIGN NURSE</button>
</div>
  </div>

  <div class="intel-card">
    <h3>Top Risk Patients</h3>
    <div class="queue-list" id="top-risk-list"></div>

<div class="risk-actions">
  <button class="risk-btn ack-risk">Acknowledge</button>
  <button class="risk-btn escalate-risk">Escalate Case</button>
</div>
  </div>

  <div class="intel-card">
    <h3>AI Clinical Reasoning</h3>
    <div class="why-now" id="ai-reasoning-panel">
      <div class="why-line">Waiting for patient intelligence stream...</div>
    </div>
  </div>

  <div class="intel-card">
    <h3>Clinical Workflow</h3>
    <div class="queue-list" id="workflow-panel">
      <div class="queue-item">
        <div class="queue-copy">New Alerts <strong id="wf-new">0</strong></div>
        <div class="status-pill watch">Open</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">Acknowledged <strong id="wf-ack">0</strong></div>
        <div class="status-pill live">Tracked</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">Escalated <strong id="wf-escalated">0</strong></div>
        <div class="status-pill critical">Priority</div>
      </div>
    </div>
  </div>

  <div class="intel-card">
    <h3>System Health</h3>
    <div class="queue-list" id="system-health-panel">
      <div class="queue-item">
        <div class="queue-copy">Telemetry Devices Connected</div>
        <div class="status-pill live" id="health-devices">--</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">Active Patient Streams</div>
        <div class="status-pill live" id="health-streams">--</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">Data Latency</div>
        <div class="status-pill watch" id="health-latency">--</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">System Status</div>
        <div class="status-pill live" id="health-status">Operational</div>
      </div>
    </div>
  </div>

  <div class="intel-card">
    <h3>Staffing Load</h3>
    <div class="queue-list" id="staffing-panel">
      <div class="queue-item">
        <div class="queue-copy">Open Alerts</div>
        <div class="status-pill watch" id="staff-open-alerts">0</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">Critical Patients</div>
        <div class="status-pill critical" id="staff-critical-patients">0</div>
      </div>
      <div class="queue-item">
        <div class="queue-copy">Response Load</div>
        <div class="status-pill live" id="staff-response-load">Stable</div>
      </div>
    </div>
  </div>

</div>>
      </div>
    </section>
  </div>

  <div id="unit-filter-mount"></div>

<div class="intel-card">
  <h3>Top Risk Patients</h3>
  <div class="queue-list" id="top-risk-list"></div>
</div>

<div class="intel-card">
  <h3>AI Clinical Reasoning</h3>
  <div class="why-now" id="ai-reasoning-panel">
    <div class="why-line">Waiting for patient intelligence stream...</div>
  </div>
</div>

<div class="intel-card">
  <h3>Clinical Workflow</h3>
  <div class="queue-list" id="workflow-panel">
    <div class="queue-item">
      <div class="queue-copy">New Alerts <strong id="wf-new">0</strong></div>
      <div class="status-pill watch">Open</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">Acknowledged <strong id="wf-ack">0</strong></div>
      <div class="status-pill live">Tracked</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">Escalated <strong id="wf-esc">0</strong></div>
      <div class="status-pill critical">Priority</div>
    </div>
  </div>
</div>

<div class="intel-card">
  <h3>System Health</h3>
  <div class="queue-list">
    <div class="queue-item">
      <div class="queue-copy">Telemetry Devices Connected</div>
      <div class="status-pill live" id="health-devices">--</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">Active Patient Streams</div>
      <div class="status-pill live" id="health-streams">--</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">Data Latency</div>
      <div class="status-pill watch" id="health-latency">--</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">System Status</div>
      <div class="status-pill live" id="health-status">Operational</div>
    </div>
  </div>
</div>

<div class="intel-card">
  <h3>Staffing Load</h3>
  <div class="queue-list" id="staffing-panel">
    <div class="queue-item">
      <div class="queue-copy">Open Alerts</div>
      <div class="status-pill watch" id="staff-open-alerts">0</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">Critical Patients</div>
      <div class="status-pill critical" id="staff-critical-patients">0</div>
    </div>
    <div class="queue-item">
      <div class="queue-copy">Response Load</div>
      <div class="status-pill live" id="staff-response-load">Stable</div>
    </div>
  </div>
</div>

<div class="intel-card">
  <h3>Audit Trail</h3>
  <div class="queue-list" id="audit-panel"></div>
</div>

  <section class="roi-section">
  <div class="roi-card">

    <div class="roi-kicker">Hospital Value</div>

    <h2>Hospital Impact / ROI Metrics</h2>

    <p class="roi-lead">
      Early Risk Alert AI is designed to help hospitals identify deterioration sooner,
      improve monitoring visibility, support faster intervention, and strengthen operational efficiency.
    </p>

    <div class="roi-grid">

      <div class="roi-metric">
        <div class="roi-value">Earlier</div>
        <div class="roi-title">Intervention Timing</div>
        <p>
          Surfaces subtle deterioration signals sooner so clinical teams can respond before
          a patient reaches a more severe escalation threshold.
        </p>
      </div>

      <div class="roi-metric">
        <div class="roi-value">Lower</div>
        <div class="roi-title">Escalation Risk</div>
        <p>
          Supports proactive care decisions that may reduce avoidable respiratory or
          hemodynamic deterioration events.
        </p>
      </div>

      <div class="roi-metric">
        <div class="roi-value">Better</div>
        <div class="roi-title">Monitoring Visibility</div>
        <p>
          Gives command-center and clinical teams a clearer real-time view of which patients
          need attention first.
        </p>
      </div>

      <div class="roi-metric">
        <div class="roi-value">Stronger</div>
        <div class="roi-title">Operational Awareness</div>
        <p>
          Combines patient risk, alert prioritization, and dashboard visibility into one
          unified hospital workflow.
        </p>
      </div>

    </div>

    <div class="roi-bottom-grid">

      <div class="roi-panel">
        <h3>Potential Clinical Benefits</h3>
        <ul>
          <li>Earlier recognition of respiratory deterioration</li>
          <li>Faster response to rising-risk patients</li>
          <li>Improved prioritization across monitored patients</li>
          <li>More visible escalation trends for care teams</li>
        </ul>
      </div>

      <div class="roi-panel">
        <h3>Potential Operational Benefits</h3>
        <ul>
          <li>Stronger command-center oversight</li>
          <li>Improved staff attention allocation</li>
          <li>Better visibility into high-risk patient clusters</li>
          <li>Investor- and executive-ready workflow presentation</li>
        </ul>
      </div>

    </div>

  </div>
</section>

<style>
.roi-section{
  max-width:1460px;
  margin:40px auto;
  padding:0 16px 70px;
}

.roi-card{
  background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
  border:1px solid rgba(255,255,255,.08);
  border-radius:26px;
  padding:28px;
  box-shadow:0 20px 60px rgba(0,0,0,.34);
}

.roi-kicker{
  font-size:11px;
  font-weight:900;
  letter-spacing:.16em;
  text-transform:uppercase;
  color:#9adfff;
  margin-bottom:10px;
}

.roi-card h2{
  margin:0 0 12px;
  font-size:40px;
  font-weight:1000;
  letter-spacing:-.04em;
  line-height:1.05;
}

.roi-lead{
  margin:0 0 24px;
  font-size:16px;
  line-height:1.7;
  color:#cfe0f4;
  max-width:920px;
}

.roi-grid{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:16px;
  margin-bottom:18px;
}

.roi-metric{
  border:1px solid rgba(255,255,255,.07);
  border-radius:18px;
  padding:18px;
  background:rgba(255,255,255,.03);
}

.roi-value{
  font-size:30px;
  font-weight:1000;
  line-height:1;
  letter-spacing:-.03em;
  color:#7aa2ff;
  margin-bottom:10px;
}

.roi-title{
  font-size:18px;
  font-weight:900;
  margin-bottom:8px;
}

.roi-metric p{
  margin:0;
  font-size:14px;
  line-height:1.6;
  color:#c7d7ef;
}

.roi-bottom-grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:16px;
}

.roi-panel{
  border:1px solid rgba(255,255,255,.07);
  border-radius:18px;
  padding:20px;
  background:rgba(255,255,255,.03);
}

.roi-panel h3{
  margin:0 0 12px;
  font-size:20px;
  font-weight:900;
}

.roi-panel ul{
  margin:0;
  padding-left:18px;
  color:#d8e6f8;
  line-height:1.8;
  font-size:14px;
}

@media(max-width:1100px){
  .roi-grid{
    grid-template-columns:1fr 1fr;
  }
}

@media(max-width:760px){
  .roi-grid,
  .roi-bottom-grid{
    grid-template-columns:1fr;
  }

  .roi-card h2{
    font-size:32px;
  }
}
</style>

<div style="text-align:center;margin:40px 0 10px;font-size:14px;color:#9fb4d6;font-weight:700;">
Early Risk Alert AI — Predictive Monitoring Platform
Built for Hospitals, Care Networks, and Remote Monitoring Systems
</div>

<section class="who-section">
  <div class="who-card">

    <div class="who-kicker">Who this platform serves</div>

    <h2>Who This Is For</h2>

    <p class="who-lead">
      Early Risk Alert AI is designed for multiple healthcare and business audiences,
      giving each stakeholder a clear view of value, workflow impact, and platform relevance.
    </p>

    <div class="who-grid">

      <div class="who-box">
        <div class="who-icon">🏥</div>
        <h3>Hospitals</h3>
        <p>
          Built for hospitals that need earlier deterioration visibility, stronger monitoring workflows,
          and clearer command-center prioritization across patient populations.
        </p>
      </div>

      <div class="who-box">
        <div class="who-icon">🩺</div>
        <h3>Clinics</h3>
        <p>
          Helps outpatient and specialty clinics strengthen patient oversight, identify rising-risk trends,
          and improve escalation awareness in monitored care settings.
        </p>
      </div>

      <div class="who-box">
        <div class="who-icon">📡</div>
        <h3>RPM Providers</h3>
        <p>
          Supports remote patient monitoring programs by surfacing early warning patterns before patients
          cross critical alert thresholds.
        </p>
      </div>

      <div class="who-box">
        <div class="who-icon">💼</div>
        <h3>Executives & Operators</h3>
        <p>
          Gives leadership teams a polished operational view of alert pressure, monitoring visibility,
          workflow readiness, and enterprise value.
        </p>
      </div>

      <div class="who-box">
        <div class="who-icon">📈</div>
        <h3>Investors</h3>
        <p>
          Presents a healthcare AI platform with hospital relevance, command-center presentation,
          live workflow storytelling, and scalable enterprise SaaS positioning.
        </p>
      </div>

      <div class="who-box">
        <div class="who-icon">🧠</div>
        <h3>Care Teams</h3>
        <p>
          Helps nurses, respiratory teams, and clinicians identify which patients may need earlier
          intervention and closer follow-up attention.
        </p>
      </div>

    </div>

  </div>
</section>

<style>
.who-section{
  max-width:1460px;
  margin:0 auto 70px;
  padding:0 16px;
}

.who-card{
  background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
  border:1px solid rgba(255,255,255,.08);
  border-radius:26px;
  padding:28px;
  box-shadow:0 20px 60px rgba(0,0,0,.34);
}

.who-kicker{
  font-size:11px;
  font-weight:900;
  letter-spacing:.16em;
  text-transform:uppercase;
  color:#9adfff;
  margin-bottom:10px;
}

.who-card h2{
  margin:0 0 12px;
  font-size:40px;
  font-weight:1000;
  letter-spacing:-.04em;
  line-height:1.05;
}

.who-lead{
  margin:0 0 24px;
  font-size:16px;
  line-height:1.7;
  color:#cfe0f4;
  max-width:920px;
}

.who-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:16px;
}

.who-box{
  border:1px solid rgba(255,255,255,.07);
  border-radius:20px;
  padding:20px;
  background:rgba(255,255,255,.03);
  transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}

.who-box:hover{
  transform:translateY(-2px);
  border-color:rgba(122,162,255,.24);
  box-shadow:0 12px 28px rgba(0,0,0,.18);
}

.who-icon{
  font-size:28px;
  margin-bottom:12px;
}

.who-box h3{
  margin:0 0 10px;
  font-size:20px;
  font-weight:900;
  letter-spacing:-.02em;
}

.who-box p{
  margin:0;
  font-size:14px;
  line-height:1.65;
  color:#c7d7ef;
}

@media(max-width:1100px){
  .who-grid{
    grid-template-columns:1fr 1fr;
  }
}

@media(max-width:760px){
  .who-grid{
    grid-template-columns:1fr;
  }

  .who-card h2{
    font-size:32px;
  }
}
</style>

  <section class="use-case-section">
  <div class="use-case-card">

    <div class="use-case-kicker">Clinical Use Case Scenario</div>

    <h2>Early Detection of Respiratory Deterioration</h2>

    <p class="use-case-lead">
      Early Risk Alert AI enables care teams to detect respiratory decline sooner by identifying subtle warning
      patterns before traditional hospital alarm thresholds are reached.
    </p>

    <div class="use-case-body">

      <p>
        A patient is admitted with moderate respiratory risk and placed under standard monitoring.
        Continuous vital signals are collected including oxygen saturation, heart rate,
        respiratory rate, and blood pressure.
      </p>

      <p>
        Early Risk Alert AI analyzes these signals in real time using predictive risk models.
        The platform detects subtle deterioration patterns in oxygen saturation trends and
        respiratory variability that may not yet trigger standard hospital alarms.
      </p>

      <p>
        The system generates an early risk alert indicating an increasing probability of
        respiratory deterioration within the next monitoring window. Clinical staff receive
        the alert through the command center dashboard, providing earlier visibility into
        patient risk.
      </p>

      <p>
        Care teams can intervene sooner by adjusting oxygen support, increasing monitoring
        intensity, and reassessing the patient before the condition reaches a critical
        escalation threshold.
      </p>

    </div>

    <div class="use-case-results">

      <div class="result-box">
        Earlier intervention
      </div>

      <div class="result-box">
        Reduced escalation risk
      </div>

      <div class="result-box">
        Improved patient monitoring visibility
      </div>

    </div>

  </div>
</section>

<style>

.use-case-section{
max-width:1460px;
margin:40px auto;
padding:0 16px;
}

.use-case-card{
background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
border:1px solid rgba(255,255,255,.08);
border-radius:26px;
padding:28px;
box-shadow:0 20px 60px rgba(0,0,0,.34);
}

.use-case-kicker{
font-size:11px;
font-weight:900;
letter-spacing:.16em;
text-transform:uppercase;
color:#9adfff;
margin-bottom:10px;
}

.use-case-card h2{
margin:0 0 12px;
font-size:40px;
font-weight:1000;
letter-spacing:-.04em;
}

.use-case-lead{
margin-bottom:18px;
font-size:16px;
line-height:1.7;
color:#cfe0f4;
max-width:820px;
}

.use-case-body{
display:grid;
gap:14px;
}

.use-case-body p{
margin:0;
font-size:15px;
line-height:1.75;
color:#c7d7ef;
}

.use-case-results{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:14px;
margin-top:24px;
}

.result-box{
border:1px solid rgba(255,255,255,.07);
border-radius:18px;
padding:16px;
background:rgba(255,255,255,.03);
font-weight:900;
text-align:center;
color:#eef4ff;
}

@media(max-width:900px){
.use-case-results{
grid-template-columns:1fr;
}
}

</style>

<section class="ai-explain-section">
  <div class="ai-explain-card">

    <div class="ai-explain-kicker">AI Intelligence Layer</div>

    <h2>How Early Risk Alert AI Works</h2>

    <p class="ai-explain-lead">
      Early Risk Alert AI continuously analyzes patient vital signals using predictive models
      designed to detect deterioration patterns earlier than traditional monitoring systems.
    </p>

    <div class="ai-process-grid">

      <div class="ai-step">
        <div class="ai-step-number">1</div>
        <h3>Signal Collection</h3>
        <p>
          Continuous vital signals including oxygen saturation, heart rate, respiratory rate,
          and blood pressure are collected from patient monitoring systems.
        </p>
      </div>

      <div class="ai-step">
        <div class="ai-step-number">2</div>
        <h3>Pattern Analysis</h3>
        <p>
          The AI analyzes signal variability, trend drift, and subtle physiological patterns
          that may indicate early clinical deterioration.
        </p>
      </div>

      <div class="ai-step">
        <div class="ai-step-number">3</div>
        <h3>Risk Prediction</h3>
        <p>
          Predictive models calculate dynamic risk scores and estimate the probability of
          deterioration within upcoming monitoring windows.
        </p>
      </div>

      <div class="ai-step">
        <div class="ai-step-number">4</div>
        <h3>Clinical Alert</h3>
        <p>
          When risk thresholds increase, the platform generates an early alert visible
          in the command center dashboard for rapid clinical response.
        </p>
      </div>

    </div>

  </div>
</section>

<style>

.ai-explain-section{
max-width:1460px;
margin:40px auto;
padding:0 16px 60px;
}

.ai-explain-card{
background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
border:1px solid rgba(255,255,255,.08);
border-radius:26px;
padding:28px;
box-shadow:0 20px 60px rgba(0,0,0,.34);
}

.ai-explain-kicker{
font-size:11px;
font-weight:900;
letter-spacing:.16em;
text-transform:uppercase;
color:#9adfff;
margin-bottom:10px;
}

.ai-explain-card h2{
margin:0 0 12px;
font-size:40px;
font-weight:1000;
letter-spacing:-.04em;
}

.ai-explain-lead{
margin-bottom:24px;
font-size:16px;
line-height:1.7;
color:#cfe0f4;
max-width:900px;
}

.ai-process-grid{
display:grid;
grid-template-columns:repeat(4,1fr);
gap:16px;
}

.ai-step{
border:1px solid rgba(255,255,255,.07);
border-radius:18px;
padding:18px;
background:rgba(255,255,255,.03);
}

.ai-step-number{
font-size:24px;
font-weight:1000;
margin-bottom:10px;
color:#7aa2ff;
}

.ai-step h3{
margin:0 0 8px;
font-size:18px;
font-weight:900;
}

.ai-step p{
margin:0;
font-size:14px;
line-height:1.6;
color:#c7d7ef;
}

@media(max-width:1000px){
.ai-process-grid{
grid-template-columns:1fr 1fr;
}
}

@media(max-width:700px){
.ai-process-grid{
grid-template-columns:1fr;
}
}

</style>
<script>
const fallbackPatients = [
  {
    patient_id: "p101",
    name: "Patient 1042",
    bed: "ICU Bed A",
    unit: "icu",
    title: "Predictive Risk Monitor",
    heart_rate: 128,
    spo2: 89,
    bp_systolic: 165,
    bp_diastolic: 102,
    risk_score: 9.1,
    status: "Critical",
    story: "Oxygen trend sharply below target. Immediate bedside review recommended."
  },
  {
    patient_id: "p102",
    name: "Patient 2188",
    bed: "Stepdown Bed B",
    unit: "stepdown",
    title: "Predictive Risk Monitor",
    heart_rate: 100,
    spo2: 93,
    bp_systolic: 150,
    bp_diastolic: 90,
    risk_score: 3.3,
    status: "Stable",
    story: "Moderate signal drift detected. Continue close observation."
  },
  {
    patient_id: "p103",
    name: "Patient 3045",
    bed: "Telemetry Bed C",
    unit: "telemetry",
    title: "Predictive Risk Monitor",
    heart_rate: 84,
    spo2: 98,
    bp_systolic: 122,
    bp_diastolic: 80,
    risk_score: 1.6,
    status: "Stable",
    story: "Recovery pattern continues across monitored vitals."
  },
  {
    patient_id: "p104",
    name: "Patient 4172",
    bed: "Ward Bed D",
    unit: "ward",
    title: "Predictive Risk Monitor",
    heart_rate: 91,
    spo2: 95,
    bp_systolic: 136,
    bp_diastolic: 84,
    risk_score: 2.8,
    status: "Stable",
    story: "General floor visibility remains stable."
  },
  {
    patient_id: "p105",
    name: "Patient 5221",
    bed: "RPM Home A",
    unit: "rpm",
    title: "Remote Monitoring View",
    heart_rate: 87,
    spo2: 96,
    bp_systolic: 132,
    bp_diastolic: 82,
    risk_score: 2.1,
    status: "Stable",
    story: "Remote monitoring stream active with no immediate escalation."
  }
];

  const fallbackAlerts = [
    { patient_id: "p101", severity: "Critical", text: "Oxygen saturation critical", unit: "ICU-12" },
    { patient_id: "p102", severity: "High", text: "Escalation watch surfaced", unit: "Telemetry-09" },
    { patient_id: "p104", severity: "High", text: "Priority intervention requested", unit: "Stepdown-04" }
  ];

const wall = document.getElementById("wall");
let activeUnitFilter = "all";
const queue = document.getElementById("queue");
const topRiskList = document.getElementById("top-risk-list");
const aiReasoningPanel = document.getElementById("ai-reasoning-panel");
const workflowPanel = document.getElementById("workflow-panel");
const staffOpenAlerts = document.getElementById("staff-open-alerts");
const staffCriticalPatients = document.getElementById("staff-critical-patients");
const staffResponseLoad = document.getElementById("staff-response-load");
const healthDevices = document.getElementById("health-devices");
const healthStreams = document.getElementById("health-streams");
const healthLatency = document.getElementById("health-latency");
const healthStatus = document.getElementById("health-status");

let selectedUnit = "all";
let alertState = {};
let auditTrail = [];

function safe(v, fallback="--"){
  return v === undefined || v === null || v === "" ? fallback : v;
}

function statusClass(status){
  const s = String(status || "").toLowerCase();
  if (s === "critical") return "critical";
  if (s === "high") return "watch";
  return "live";
}

function ecgClass(status){
  const s = String(status || "").toLowerCase();
  if (s === "critical") return "ecg-red";
  if (s === "high") return "ecg-amber";
  return "ecg-green";
}

function pulseLabel(status){
  const s = String(status || "").toLowerCase();
  if (s === "critical") return "Critical";
  if (s === "high") return "High";
  return "Stable";
}

function riskSeverityLabel(severity){
  const s = String(severity || "").toLowerCase();
  if (s === "critical") return "Critical";
  if (s === "high") return "High";
  return "Stable";
}

function normalizePatient(raw) {
  if (!raw) return null;

  const vitals = raw.vitals || {};
  const risk = raw.risk || {};

  return {
    patient_id: raw.patient_id || "p---",
    name: raw.patient_name || raw.name || "Patient",
    bed: raw.room || raw.bed || "ICU Bed",
    unit: String(raw.unit || raw.program || raw.room || "icu").toLowerCase(),
    title:
      raw.title ||
      risk.alert_message ||
      (risk.alert_type ? String(risk.alert_type).replace(/_/g, " ") : "Predictive Risk Monitor"),
    heart_rate: raw.heart_rate ?? vitals.heart_rate ?? "--",
    spo2: raw.spo2 ?? vitals.spo2 ?? "--",
    bp_systolic: raw.bp_systolic ?? vitals.systolic_bp ?? "--",
    bp_diastolic: raw.bp_diastolic ?? vitals.diastolic_bp ?? "--",
    risk_score: raw.risk_score ?? risk.risk_score ?? "--",
    status: raw.status || risk.severity || "stable",
    story: raw.story || risk.recommended_action || "Predictive monitoring active.",
    unit: raw.room || raw.unit || raw.bed || "Unit",
    alert_text: risk.alert_message || raw.alert_text || "Clinical alert surfaced",
    clinical_priority: risk.clinical_priority || "Priority 4",
    confidence: risk.confidence ?? "--",
    trend: risk.trend_direction || "Stable",
    recommended_action: risk.recommended_action || "Continue routine monitoring."
  };
}

function normalizeAlert(alert) {
  if (!alert) return null;
  return {
    patient_id: alert.patient_id || "Patient",
    severity: alert.severity || "Stable",
    text: alert.message || alert.text || "Clinical alert surfaced",
    unit: alert.room || alert.unit || "Unit",
    risk_score: alert.risk_score ?? "--",
    confidence: alert.confidence ?? "--",
    recommended_action: alert.recommended_action || "Continue monitoring.",
    clinical_priority: alert.clinical_priority || "Priority 4",
    trend: alert.trend_direction || "Stable",
    created_at: alert.created_at || new Date().toISOString()
  };
}

function buildPath(points){
  return points.map((p, i) => (i === 0 ? "M" : "L") + p[0] + "," + p[1]).join(" ");
}

function waveformPoints(mode){
  if (mode === "critical") {
    return [[0,72],[30,72],[46,72],[58,70],[72,72],[86,72],[98,50],[106,100],[116,24],[126,108],[136,72],[162,72],[186,72],[204,70],[220,72],[236,72],[250,54],[258,96],[268,22],[278,106],[290,72],[318,72],[336,72],[352,70],[368,72],[382,72],[398,46],[406,104],[416,20],[428,110],[440,72],[466,72],[486,72],[504,70],[520,72],[536,72],[550,52],[558,98],[568,24],[578,106],[592,72],[620,72]];
  }
  if (mode === "high") {
    return [[0,76],[36,76],[60,76],[74,74],[88,76],[104,76],[118,56],[126,90],[136,38],[146,96],[158,76],[190,76],[214,76],[228,74],[242,76],[258,76],[272,54],[280,88],[290,40],[300,94],[314,76],[348,76],[372,76],[388,74],[402,76],[418,76],[432,58],[440,92],[452,42],[464,96],[478,76],[514,76],[538,76],[552,74],[566,76],[582,76],[596,60],[604,94],[614,44],[624,98],[640,76]];
  }
  return [[0,78],[48,78],[82,78],[96,76],[110,78],[126,78],[138,66],[146,84],[156,52],[166,88],[178,78],[220,78],[254,78],[268,76],[282,78],[298,78],[310,68],[318,84],[328,56],[338,88],[350,78],[392,78],[426,78],[440,76],[454,78],[470,78],[482,68],[490,84],[500,56],[510,88],[522,78],[566,78],[600,78]];
}

function buildUnitFilters(patients){
  const units = [...new Set((patients || []).map(p => safe(p.unit, "Unit")))].filter(Boolean);
  return `
    <div class="intel-card">
      <h3>Unit Filters</h3>
      <div class="queue-list">
        <div class="queue-item">
          <div class="queue-copy">All Units</div>
          <button class="status-pill ${selectedUnit === "all" ? "live" : "watch"}" onclick="setUnitFilter('all')">All</button>
        </div>
        ${units.map(unit => `
          <div class="queue-item">
            <div class="queue-copy">${unit}</div>
            <button class="status-pill ${selectedUnit === unit ? "live" : "watch"}" onclick="setUnitFilter(${JSON.stringify(unit)})">${selectedUnit === unit ? "Viewing" : "Filter"}</button>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

window.setUnitFilter = function(unit){
  selectedUnit = unit;
  refreshFallback();
};

window.ackAlert = function(patientId){
  if (!patientId) return;
  alertState[patientId] = "acknowledged";
  auditTrail.unshift({
    ts: new Date().toLocaleTimeString(),
    action: "Acknowledged alert",
    patient_id: patientId
  });
  renderWorkflowFromState();
  renderAuditTrail();
  refreshFallback();
};

window.escalateAlert = function(patientId){
  if (!patientId) return;
  alertState[patientId] = "escalated";
  auditTrail.unshift({
    ts: new Date().toLocaleTimeString(),
    action: "Escalated alert",
    patient_id: patientId
  });
  renderWorkflowFromState();
  renderAuditTrail();
  refreshFallback();
};

function renderMonitor(patient){
  const status = String(patient.status || "").toLowerCase();
  const ecg = ecgClass(status);
  const path = buildPath(
    waveformPoints(status === "critical" ? "critical" : status === "high" ? "high" : "stable")
  );

  const hr = Number(patient.heart_rate || 0);
  const spo2 = Number(patient.spo2 || 0);
  const sys = Number(patient.bp_systolic || 0);
  const dia = Number(patient.bp_diastolic || 0);
  const risk =
    typeof patient.risk_score === "number"
      ? patient.risk_score.toFixed(1)
      : safe(patient.risk_score);

  return `
    <div class="monitor">
      <div class="monitor-top">
        <div>
          <div class="monitor-bed">${safe(patient.bed, "ICU Bed")}</div>
          <div class="monitor-title">${safe(patient.title, safe(patient.name, "Predictive Risk Monitor"))}</div>
        </div>
        <div class="status-pill ${statusClass(patient.status)}">${pulseLabel(patient.status)}</div>
      </div>

      <div class="monitor-wave">
        <svg viewBox="0 0 640 120" preserveAspectRatio="none" aria-hidden="true">
          <path class="ecg-path ${ecg}" d="${path}"></path>
        </svg>
      </div>

      <div class="monitor-metrics">
        <div class="metric-box">
          <span class="metric-k">HR</span>
          <span class="metric-v">${hr ? Math.round(hr) : "--"}</span>
        </div>
        <div class="metric-box">
          <span class="metric-k">SpO₂</span>
          <span class="metric-v">${spo2 ? Math.round(spo2) : "--"}</span>
        </div>
        <div class="metric-box">
          <span class="metric-k">BP</span>
          <span class="metric-v">${sys ? Math.round(sys) : "--"}/${dia ? Math.round(dia) : "--"}</span>
        </div>
        <div class="metric-box">
          <span class="metric-k">Risk</span>
          <span class="metric-v">${risk}</span>
        </div>
      </div>

      <div class="monitor-story">
        <div class="story-text">${safe(patient.story, "Predictive monitoring active.")}</div>
        <div class="status-pill ${statusClass(patient.status)}">${safe(patient.patient_id, "p---")}</div>
      </div>
    </div>
  `;
}

function renderAlert(alert){
  const sev = String(alert.severity || "").toLowerCase();
  const pill = sev === "critical" ? "critical" : sev === "high" ? "watch" : "live";
  const state = alertState[alert.patient_id] || "new";
  const minutesOpen = Math.max(
    1,
    Math.floor((Date.now() - new Date(alert.created_at).getTime()) / 60000) || 1
  );

  return `
    <div class="alert-item">
      <div style="flex:1;">
        <div class="alert-copy">${safe(alert.text, "Clinical alert surfaced")}</div>
        <div class="alert-sub">
          ${safe(alert.patient_id, "Patient")} · ${safe(alert.unit, "Unit")} · ${safe(alert.severity, "Stable")}
          · ${safe(alert.clinical_priority, "Priority 4")} · ${minutesOpen}m open
        </div>
        <div class="alert-sub" style="margin-top:6px;">${safe(alert.recommended_action, "Continue monitoring.")}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
          <button class="status-pill live" onclick="ackAlert('${alert.patient_id}')">Acknowledge</button>
          <button class="status-pill critical" onclick="escalateAlert('${alert.patient_id}')">Escalate</button>
          <div class="status-pill ${pill}">${state}</div>
        </div>
      </div>
      <div class="status-pill ${pill}">${safe(alert.severity, "Live")}</div>
    </div>
  `;
}

function renderPatients(patients){
  const source = patients && patients.length ? patients : fallbackPatients;
  const normalized = source.map(normalizePatient).filter(Boolean);

  const filtered = activeUnitFilter === "all"
    ? normalized
    : normalized.filter(p => String(p.unit || "").toLowerCase().includes(activeUnitFilter));

  wall.innerHTML = filtered.slice(0, 8).map(renderMonitor).join("");

  if (!filtered.length) {
    wall.innerHTML = `
      <div class="intel-card" style="grid-column:1/-1;">
        <h3>No monitors in this unit</h3>
        <div class="intel-note">No live patients matched the selected hospital unit filter.</div>
      </div>
    `;
  }
}

function renderAlertsList(alerts){
  let source = (alerts && alerts.length) ? alerts.map(normalizeAlert).filter(Boolean) : fallbackAlerts;
  if (selectedUnit !== "all") {
    source = source.filter(a => safe(a.unit) === selectedUnit);
  }
  queue.innerHTML = source.slice(0, 6).map(renderAlert).join("");
}

function renderTopRiskPatients(patients){
  if (!topRiskList) return;
  let source = (patients && patients.length) ? patients.map(normalizePatient).filter(Boolean) : fallbackPatients;
  if (selectedUnit !== "all") {
    source = source.filter(p => safe(p.unit) === selectedUnit || safe(p.bed) === selectedUnit);
  }
  source.sort((a, b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));
  topRiskList.innerHTML = source.slice(0, 5).map((p, i) => `
    <div class="queue-item">
      <div class="queue-copy">#${i + 1} ${safe(p.patient_id)} · ${safe(p.unit)} · Risk ${Number(p.risk_score || 0).toFixed(1)}</div>
      <div class="status-pill ${statusClass(p.status)}">${pulseLabel(p.status)}</div>
    </div>
  `).join("");
}

function renderAIReasoning(patients){
  if (!aiReasoningPanel) return;
  let source = (patients && patients.length) ? patients.map(normalizePatient).filter(Boolean) : fallbackPatients;
  source.sort((a, b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));
  const p = source[0];
  aiReasoningPanel.innerHTML = `
    <div class="why-line">Highest risk patient: ${safe(p.patient_id)} in ${safe(p.unit)}</div>
    <div class="why-line">Primary signal: SpO₂ ${safe(p.spo2)} with HR ${safe(p.heart_rate)} and BP ${safe(p.bp_systolic)}/${safe(p.bp_diastolic)}</div>
    <div class="why-line">AI recommendation: ${safe(p.recommended_action)}</div>
    <div class="why-line">Trend: ${safe(p.trend)} · Confidence: ${safe(p.confidence)}%</div>
  `;
}

function renderWorkflowFromState(){
  const values = Object.values(alertState);
  const ack = values.filter(v => v === "acknowledged").length;
  const escalated = values.filter(v => v === "escalated").length;
  const newCount = Math.max(0, Object.keys(alertState).length - ack - escalated);

  const wfNew = document.getElementById("wf-new");
  const wfAck = document.getElementById("wf-ack");
  const wfEsc = document.getElementById("wf-esc");

  if (wfNew) wfNew.textContent = newCount;
  if (wfAck) wfAck.textContent = ack;
  if (wfEsc) wfEsc.textContent = escalated;
}

function renderWorkflow(alerts){
  let source = (alerts && alerts.length) ? alerts.map(normalizeAlert).filter(Boolean) : fallbackAlerts;
  source.forEach(a => {
    if (!alertState[a.patient_id]) {
      alertState[a.patient_id] = "new";
    }
  });
  renderWorkflowFromState();
}

function renderSystemHealth(patients, alerts){
  const sourcePatients = (patients && patients.length) ? patients.map(normalizePatient).filter(Boolean) : fallbackPatients;
  const sourceAlerts = (alerts && alerts.length) ? alerts.map(normalizeAlert).filter(Boolean) : fallbackAlerts;

  if (healthDevices) healthDevices.textContent = String(sourcePatients.length);
  if (healthStreams) healthStreams.textContent = String(sourcePatients.length);
  if (healthLatency) healthLatency.textContent = sourceAlerts.length ? "~2s" : "~0s";
  if (healthStatus) healthStatus.textContent = "Operational";
}

function renderStaffingLoad(patients, alerts){
  const sourcePatients = (patients && patients.length) ? patients.map(normalizePatient).filter(Boolean) : fallbackPatients;
  const sourceAlerts = (alerts && alerts.length) ? alerts.map(normalizeAlert).filter(Boolean) : fallbackAlerts;

  const criticalPatients = sourcePatients.filter(p => String(p.status).toLowerCase() === "critical").length;
  const responseLoad = sourceAlerts.length >= 5 ? "High" : sourceAlerts.length >= 3 ? "Moderate" : "Stable";

  if (staffOpenAlerts) staffOpenAlerts.textContent = String(sourceAlerts.length);
  if (staffCriticalPatients) staffCriticalPatients.textContent = String(criticalPatients);
  if (staffResponseLoad) staffResponseLoad.textContent = responseLoad;
}

function renderAuditTrail(){
  const auditPanel = document.getElementById("audit-panel");
  if (!auditPanel) return;
  auditPanel.innerHTML = auditTrail.slice(0, 6).map(item => `
    <div class="queue-item">
      <div class="queue-copy">${item.ts} · ${item.action} · ${item.patient_id}</div>
      <div class="status-pill live">Logged</div>
    </div>
  `).join("") || `<div class="queue-item"><div class="queue-copy">No actions logged yet.</div><div class="status-pill live">Ready</div></div>`;
}

function updateSummary(patients, alerts){
  const sourcePatients = (patients && patients.length) ? patients.map(normalizePatient).filter(Boolean) : fallbackPatients;
  const sourceAlerts = (alerts && alerts.length) ? alerts.map(normalizeAlert).filter(Boolean) : fallbackAlerts;

  const filteredPatients = selectedUnit === "all"
    ? sourcePatients
    : sourcePatients.filter(p => safe(p.unit) === selectedUnit || safe(p.bed) === selectedUnit);

  const filteredAlerts = selectedUnit === "all"
    ? sourceAlerts
    : sourceAlerts.filter(a => safe(a.unit) === selectedUnit);

  const openAlerts = filteredAlerts.length;
  const criticalAlerts = filteredAlerts.filter(a => String(a.severity || "").toLowerCase() === "critical").length;
  const avgRisk = filteredPatients.length
    ? filteredPatients.reduce((n, p) => n + Number(p.risk_score || 0), 0) / filteredPatients.length
    : 0;

  const openNode = document.getElementById("open-alerts");
  const criticalNode = document.getElementById("critical-alerts");
  const avgNode = document.getElementById("avg-risk");

  if (openNode) openNode.textContent = String(openAlerts);
  if (criticalNode) criticalNode.textContent = String(criticalAlerts);
  if (avgNode) avgNode.textContent = avgRisk.toFixed(1);

  const unitFilterMount = document.getElementById("unit-filter-mount");
  if (unitFilterMount) unitFilterMount.innerHTML = buildUnitFilters(filteredPatients);
}

function applyPayload(payload){
  if (payload && Array.isArray(payload.patients)) {
    renderPatients(payload.patients);
    renderAlertsList(payload.alerts || []);
    updateSummary(payload.patients, payload.alerts || []);
    renderTopRiskPatients(payload.patients);
    renderAIReasoning(payload.patients);
    renderWorkflow(payload.alerts || []);
    renderSystemHealth(payload.patients, payload.alerts || []);
    renderStaffingLoad(payload.patients, payload.alerts || []);
    renderAuditTrail();
    return;
  }

  if (Array.isArray(payload)) {
    renderPatients(payload);
    renderAlertsList([]);
    updateSummary(payload, []);
    renderTopRiskPatients(payload);
    renderAIReasoning(payload);
    renderWorkflow([]);
    renderSystemHealth(payload, []);
    renderStaffingLoad(payload, []);
    renderAuditTrail();
    return;
  }

  if (payload && (payload.patient_id || payload.vitals || payload.risk)) {
    const onePatient = [payload];
    const oneAlert = payload.risk ? [{
      patient_id: payload.patient_id || "Patient",
      severity: payload.risk.severity || "Stable",
      text: payload.risk.alert_message || "Clinical alert surfaced",
      unit: payload.room || "Unit",
      risk_score: payload.risk.risk_score || 0,
      confidence: payload.risk.confidence || "--",
      recommended_action: payload.risk.recommended_action || "Continue monitoring.",
      clinical_priority: payload.risk.clinical_priority || "Priority 4",
      trend_direction: payload.risk.trend_direction || "Stable",
      created_at: new Date().toISOString()
    }] : [];

    renderPatients(onePatient);
    renderAlertsList(oneAlert);
    updateSummary(onePatient, oneAlert);
    renderTopRiskPatients(onePatient);
    renderAIReasoning(onePatient);
    renderWorkflow(oneAlert);
    renderSystemHealth(onePatient, oneAlert);
    renderStaffingLoad(onePatient, oneAlert);
    renderAuditTrail();
    return;
  }

  renderPatients([]);
  renderAlertsList([]);
  updateSummary([], []);
  renderTopRiskPatients([]);
  renderAIReasoning([]);
  renderWorkflow([]);
  renderSystemHealth([], []);
  renderStaffingLoad([], []);
  renderAuditTrail();
}



  async function refreshFallback() {
    try {
      const res = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), {
        cache: "no-store"
      });
      if (!res.ok) {
        applyPayload({});
        return;
      }
      const payload = await res.json();
      applyPayload(payload);
    } catch (err) {
      console.error("Command center fallback refresh failed", err);
      applyPayload({});
    }
  }

  try {
    const evt = new EventSource("/api/command-center-stream");
    evt.onmessage = function(e) {
      try {
        const payload = JSON.parse(e.data || "{}");
        applyPayload(payload);
      } catch (err) {
        console.error("Command center stream parse error", err);
      }
    };
    evt.onerror = function() {
      console.warn("Command center stream error, fallback polling remains active.");
    };
  } catch (err) {
    console.warn("EventSource unavailable, fallback polling only.");
  }

  document.querySelectorAll(".unit-filter").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".unit-filter").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeUnitFilter = btn.dataset.unit || "all";
    refreshFallback();
  });
});

document.querySelectorAll(".alert-action.ack").forEach(btn=>{
  btn.addEventListener("click",()=>{
    console.log("Alert acknowledged");
  });
});

document.querySelectorAll(".alert-action.escalate").forEach(btn=>{
  btn.addEventListener("click",()=>{
    console.log("Alert escalated to critical care team");
  });
});

document.querySelectorAll(".alert-action.assign").forEach(btn=>{
  btn.addEventListener("click",()=>{
    console.log("Nurse assigned to patient case");
  });
});

document.querySelectorAll(".risk-btn.ack-risk").forEach(btn=>{
  btn.addEventListener("click",()=>{
    console.log("Top risk patient acknowledged");
  });
});

document.querySelectorAll(".risk-btn.escalate-risk").forEach(btn=>{
  btn.addEventListener("click",()=>{
    console.log("Risk escalation initiated");
  });
});

  applyPayload({});
  refreshFallback();
  setInterval(refreshFallback, 5000);
</script>
</body>
</html>
"""
