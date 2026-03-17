COMMAND_CENTER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Hospital Command Center</title>
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
      --max:1500px;
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
    .shell{max-width:var(--max);margin:0 auto;padding:20px 16px 56px}

    .topbar{
      position:sticky;top:0;z-index:80;
      border-bottom:1px solid var(--line);
      background:rgba(7,16,28,.86);
      backdrop-filter:blur(16px);
    }
    .topbar-inner{
      max-width:var(--max);margin:0 auto;padding:14px 16px;
      display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;
    }
    .brand-title{
      font-size:clamp(26px,3vw,44px);font-weight:1000;line-height:.92;letter-spacing:-.05em;
    }
    .brand-sub{font-size:14px;font-weight:800;color:var(--muted)}
    .nav-links{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
    .nav-links a{font-size:14px;font-weight:900;color:#dce9ff}

    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      border:1px solid transparent;cursor:pointer;
      transition:transform .18s ease, box-shadow .18s ease, opacity .18s ease, border-color .18s ease;
      font-family:inherit;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#07101c;box-shadow:0 12px 30px rgba(91,212,255,.22);
    }
    .btn.secondary{
      background:rgba(255,255,255,.04);border-color:var(--line);color:var(--text);
    }
    .btn.small{padding:9px 12px;border-radius:12px;font-size:12px}
    .btn.critical-btn{background:linear-gradient(135deg,#ff667d,#ff8fa0);color:#08111f}
    .btn.live-btn{background:linear-gradient(135deg,#3ad38f,#8ff3c1);color:#08111f}
    .btn.warn-btn{background:linear-gradient(135deg,#f4bd6a,#ffe09b);color:#2a1b00}
    .btn[disabled]{opacity:.45;cursor:not-allowed;transform:none !important}

    .status-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:10px 14px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.05);
    }
    .status-pill.live{color:#0b1528;background:linear-gradient(135deg,var(--green),#7ef5c0)}
    .status-pill.watch{color:#2a1b00;background:linear-gradient(135deg,var(--amber),#ffdba5)}
    .status-pill.critical{color:#fff3f5;background:linear-gradient(135deg,#ff667d,#ff8e99)}
    .status-pill.info{color:#07101c;background:linear-gradient(135deg,var(--blue),var(--blue2))}
    .status-pill.muted{color:#dce9ff;background:rgba(255,255,255,.05)}

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
      padding:22px;
      overflow:hidden;
    }
    .hero-grid{display:grid;grid-template-columns:1.18fr .82fr;gap:18px;align-items:stretch}
    .hero-copy,.hero-side{
      border:1px solid var(--line);
      border-radius:26px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:20px;
      min-width:0;
    }
    .hero-kicker{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#91ddff;margin-bottom:10px;
    }
    .hero-copy h1{
      margin:0 0 12px;
      font-size:clamp(32px,4.3vw,60px);
      line-height:.94;
      letter-spacing:-.055em;
      font-weight:1000;
      max-width:100%;
      word-break:break-word;
    }
    .hero-copy p,.hero-side p{margin:0;color:#d0def2;font-size:16px;line-height:1.68}
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}
    .hero-mini-grid{margin-top:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .hero-mini{
      border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
      min-width:0;
    }
    .hero-mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fdcff;margin-bottom:8px;
    }
    .hero-mini .v{font-size:18px;font-weight:1000;line-height:1.12;letter-spacing:-.03em}

    .pilot-banner{
      display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
      margin-bottom:14px;padding:14px 16px;border-radius:18px;
      background:rgba(244,189,106,.12);border:1px solid rgba(244,189,106,.18);color:#ffe4b6;
    }
    .pilot-banner strong{color:#fff0cf}
    .banner-links{display:flex;gap:10px;flex-wrap:wrap}
    .banner-links a{
      display:inline-flex;align-items:center;justify-content:center;padding:8px 12px;border-radius:999px;
      background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);font-size:12px;font-weight:900;
    }

    .toolbar{
      display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:18px;
    }
    .toolbar select{
      background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:14px;color:var(--text);
      padding:12px 14px;font:inherit;font-weight:800;
    }
    .health-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}

    .ticker-wrap{
      margin-top:18px;border:1px solid var(--line);border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.82), rgba(8,15,28,.92));
      overflow:hidden;
    }
    .ticker-track{
      display:flex;gap:16px;align-items:center;padding:12px 0;white-space:nowrap;min-width:max-content;
      animation:tickerMove 28s linear infinite;
    }
    .ticker-item{
      display:inline-flex;align-items:center;gap:10px;border:1px solid var(--line);border-radius:999px;
      padding:10px 14px;background:rgba(255,255,255,.03);margin-left:14px;font-size:14px;font-weight:800;color:#deebff;
    }
    .ticker-dot{width:10px;height:10px;border-radius:50%;box-shadow:0 0 20px currentColor;flex:0 0 auto}
    .dot-green{color:var(--green);background:var(--green)}
    .dot-amber{color:var(--amber);background:var(--amber)}
    .dot-red{color:var(--red);background:var(--red)}
    .dot-blue{color:var(--blue2);background:var(--blue2)}
    @keyframes tickerMove{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}

    .section{margin-top:18px}
    .section-card{
      border:1px solid var(--line);
      border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 14px 36px rgba(0,0,0,.26);
      padding:18px;
      overflow:hidden;
    }
    .section-head{
      display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:14px;
    }
    .section-title{font-size:28px;font-weight:1000;letter-spacing:-.04em;margin:0}
    .section-sub{font-size:14px;color:var(--muted);line-height:1.55;margin:6px 0 0}

    .telemetry-top{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
    .stat-card{
      border:1px solid var(--line);border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(13,23,40,.90), rgba(8,15,28,.95));
      padding:16px;min-height:122px;position:relative;overflow:hidden;
    }
    .stat-card::after{
      content:"";position:absolute;right:-30px;top:-30px;width:120px;height:120px;border-radius:50%;
      background:radial-gradient(circle, rgba(91,212,255,.12), transparent 60%);
      pointer-events:none;
    }
    .stat-card .k{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8ddcff;margin-bottom:10px;
    }
    .stat-card .v{font-size:42px;font-weight:1000;line-height:.92;letter-spacing:-.05em}
    .stat-card .hint{margin-top:8px;font-size:13px;line-height:1.5;color:#c8d8ef}

    .command-grid{display:grid;grid-template-columns:1.55fr .95fr;gap:18px;align-items:start}
    .monitor-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .monitor{
      border:1px solid var(--line);border-radius:26px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.012)),
        linear-gradient(180deg, rgba(11,20,35,.96), rgba(6,12,22,.98));
      box-shadow:0 16px 44px rgba(0,0,0,.32);
      padding:18px;position:relative;overflow:hidden;min-height:406px;
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
      font-size:clamp(24px,2.2vw,30px);
      font-weight:1000;line-height:.94;letter-spacing:-.05em;
      word-break:break-word;
    }
    .monitor-wave{
      position:relative;z-index:1;border:1px solid rgba(255,255,255,.06);border-radius:20px;min-height:138px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01)),
        rgba(0,0,0,.18);
      overflow:hidden;display:flex;align-items:center;padding:8px 0;
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
    .metric-v{font-size:18px;font-weight:1000;line-height:1;letter-spacing:-.03em;word-break:break-word}

    .monitor-story{
      position:relative;z-index:1;margin-top:14px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
    }
    .story-text{font-size:13px;color:#cfe0f4;line-height:1.55;max-width:74%}
    .monitor-actions{position:relative;z-index:1;margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}

    .side-stack{display:grid;gap:16px}
    .intel-card{
      border:1px solid var(--line);border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;box-shadow:0 14px 36px rgba(0,0,0,.26);
      overflow:hidden;
    }
    .intel-card h3{margin:0 0 12px;font-size:20px;font-weight:1000;letter-spacing:-.03em}

    .alert-feed,.queue-list,.timeline-panel,.why-now,.audit-list,.report-list,.threshold-list{display:grid;gap:10px}
    .alert-item,.queue-item,.timeline-item,.why-line,.audit-item,.report-item,.threshold-item{
      border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:12px;background:rgba(255,255,255,.03);
    }
    .alert-item,.queue-item,.audit-item,.report-item,.threshold-item{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
    .alert-copy,.queue-copy,.audit-copy,.report-copy{font-size:14px;font-weight:900;line-height:1.45;color:#e4efff}
    .alert-sub,.audit-sub,.report-sub{font-size:12px;color:#acc0dd;font-weight:700;margin-top:4px}

    .timeline-item{display:grid;grid-template-columns:78px 1fr;gap:12px;align-items:start}
    .timeline-time{font-size:12px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;color:#8fdcff}
    .timeline-copy{font-size:13px;line-height:1.55;color:#e4efff}

    .detail-drawer{
      position:fixed;top:0;right:-460px;width:440px;max-width:100%;height:100vh;z-index:120;
      background:linear-gradient(180deg, rgba(12,22,38,.98), rgba(7,14,26,.99));
      border-left:1px solid var(--line);box-shadow:-20px 0 60px rgba(0,0,0,.35);
      transition:right .25s ease;padding:18px;overflow:auto;
    }
    .detail-drawer.open{right:0}
    .drawer-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:16px}
    .drawer-title{font-size:28px;font-weight:1000;letter-spacing:-.04em;margin:0}
    .drawer-sub{font-size:13px;color:var(--muted);line-height:1.5;margin-top:6px}
    .drawer-block{
      border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);margin-bottom:12px;
    }
    .drawer-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9eb8dc;margin-bottom:8px}
    .drawer-v{font-size:18px;font-weight:1000;line-height:1.4;color:#eef4ff}
    .drawer-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .overlay{
      position:fixed;inset:0;background:rgba(0,0,0,.42);z-index:110;display:none;
    }
    .overlay.show{display:block}

    .ops-strip{margin-top:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
    .ops-card{
      border:1px solid var(--line);border-radius:22px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:16px;box-shadow:0 14px 34px rgba(0,0,0,.24);min-height:168px;
    }
    .ops-card h3{margin:0 0 12px;font-size:18px;font-weight:1000;letter-spacing:-.03em}
    .trend-bar{
      position:relative;height:16px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.05);
      border:1px solid rgba(255,255,255,.06);margin-top:14px;
    }
    .trend-fill{
      position:absolute;left:-36%;top:0;bottom:0;width:60%;
      background:linear-gradient(90deg, rgba(91,212,255,.10), rgba(91,212,255,.82), rgba(91,212,255,.10));
      animation:trendMove 3.6s linear infinite;
    }
    @keyframes trendMove{0%{transform:translateX(0)}100%{transform:translateX(230%)}}
    .trend-labels{
      display:flex;justify-content:space-between;gap:8px;margin-top:12px;font-size:12px;font-weight:900;color:#b2c6e3;
    }
    .forecast{display:grid;gap:10px;margin-top:10px}
    .forecast div{
      border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:10px 12px;background:rgba(255,255,255,.03);
      font-size:14px;font-weight:900;color:#e3efff;
    }
    .forecast-mid{color:#ffd27d}
    .forecast-high{color:#ffb162}
    .forecast-critical{color:#ff7f91}
    .capacity-value{display:block;font-size:44px;font-weight:1000;line-height:.92;letter-spacing:-.05em}
    .capacity-label{display:block;margin-top:8px;font-size:14px;font-weight:800;color:#d2e1f4}
    .capacity-note{margin-top:10px;font-size:12px;line-height:1.55;color:#a9bedb}
    .heatmap{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}
    .heat{height:28px;border-radius:8px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.10)}
    .low{background:linear-gradient(135deg,#46d88f,#8cffc1)}
    .medium{background:linear-gradient(135deg,#ffd965,#ffe79e)}
    .high{background:linear-gradient(135deg,#ff7b71,#ff9f95)}

    .grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
    .info-card{
      border:1px solid var(--line);border-radius:22px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;
      min-width:0;
    }
    .info-card h3,.info-card h4{margin:0 0 10px;font-weight:1000;letter-spacing:-.03em}
    .info-card h3{font-size:18px}
    .info-card h4{font-size:16px}
    .info-card p,.info-card li{
      color:#d0def2;font-size:14px;line-height:1.65;
    }
    .pill-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}
    .impact-pill{
      border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:14px;background:rgba(255,255,255,.03);
      text-align:center;font-weight:1000;color:#eef4ff;
    }
    .steps-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
    .step-card{
      border:1px solid rgba(255,255,255,.06);border-radius:20px;padding:18px;background:rgba(255,255,255,.03);
    }
    .step-num{
      width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#08111f;display:flex;align-items:center;justify-content:center;font-weight:1000;margin-bottom:12px;
    }
    .legal-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}

    .footer{
      margin-top:18px;border:1px solid var(--line);border-radius:22px;padding:18px;
      background:rgba(255,255,255,.03);color:#a7bddc;font-size:13px;line-height:1.6;text-align:center;
    }

    @media (max-width:1280px){
      .hero-grid,.command-grid{grid-template-columns:1fr}
      .ops-strip,.grid-4,.steps-grid{grid-template-columns:repeat(2,1fr)}
      .grid-3{grid-template-columns:1fr 1fr}
    }
    @media (max-width:980px){
      .telemetry-top{grid-template-columns:repeat(2,1fr)}
      .monitor-grid{grid-template-columns:1fr}
      .hero-mini-grid,.pill-row,.grid-4,.grid-3,.grid-2,.steps-grid,.legal-grid{grid-template-columns:1fr}
      .drawer-grid{grid-template-columns:1fr}
    }
    @media (max-width:700px){
      .shell{padding:14px 10px 40px}
      .hero{padding:16px}
      .hero-copy h1{font-size:clamp(30px,10vw,44px)}
      .hero-actions{flex-direction:column}
      .btn{width:100%}
      .hero-mini-grid,.telemetry-top,.ops-strip,.monitor-metrics,.pill-row,.grid-4,.grid-3,.grid-2,.steps-grid,.legal-grid{grid-template-columns:1fr}
      .story-text{max-width:100%}
      .timeline-item{grid-template-columns:1fr}
      .detail-drawer{width:100%}
      .brand-title{font-size:32px}
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
        <a href="/executive-walkthrough">Executives</a>
        <a href="/admin/review">Admin Review</a>
        <a href="/logout">Logout</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero">
      <div class="pilot-banner">
        <div>
          <strong>Pilot Mode / Evaluation Environment</strong><br>
          This environment is intended for pilot review, workflow testing, and stakeholder evaluation. It is not a substitute for clinician judgment, hospital policy, or emergency response systems.
        </div>
        <div class="banner-links">
          <a href="/terms">Terms</a>
          <a href="/privacy">Privacy</a>
          <a href="/pilot-disclaimer">Pilot Disclaimer</a>
        </div>
      </div>

      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">Hospital mission control</div>
          <h1>Real-time predictive monitoring wall for hospitals.</h1>
          <p>
            Early Risk Alert AI combines ICU-style telemetry monitors, live alert operations, top-risk queue
            visibility, clinical workflow panels, patient timeline tracking, pilot-safe operating controls,
            reporting, thresholds, and audit-ready hospital command-center workflows into one production-style command wall.
          </p>
          <div class="hero-actions">
            <a class="btn primary" href="/admin/review">Open Admin Review</a>
            <a class="btn secondary" href="/hospital-demo">Hospital Demo Form</a>
            <a class="btn secondary" href="/executive-walkthrough">Executive Walkthrough</a>
            <a class="btn secondary" href="/investor-intake">Investor Intake</a>
          </div>
          <div class="hero-mini-grid">
            <div class="hero-mini"><div class="k">Telemetry</div><div class="v">Live waveform monitors</div></div>
            <div class="hero-mini"><div class="k">Prediction</div><div class="v">AI risk + explainable reasons</div></div>
            <div class="hero-mini"><div class="k">Workflow</div><div class="v">Backend-saved action state</div></div>
            <div class="hero-mini"><div class="k">Audit</div><div class="v">Role-based intervention log</div></div>
          </div>
        </div>

        <div class="hero-side">
          <div class="hero-kicker">Pilot control layer</div>
          <p>
            Filter the telemetry wall by unit, review current user role, track feed freshness, monitor system health,
            and operate within a pilot-safe environment with visible legal, privacy, and disclaimer controls.
          </p>

          <div class="toolbar">
            <select id="unitFilter">
              <option value="all">All Units</option>
              <option value="icu">ICU</option>
              <option value="stepdown">Stepdown</option>
              <option value="telemetry">Telemetry</option>
              <option value="ward">Ward</option>
              <option value="rpm">RPM / Home</option>
            </select>
            <div class="status-pill info" id="selectedUnitPill">All Units</div>
            <div class="status-pill muted" id="currentRolePill">Role: --</div>
            <div class="status-pill muted" id="currentUserPill">User: --</div>
          </div>

          <div class="health-row">
            <div class="status-pill muted" id="pilotModePill">Pilot Mode</div>
            <div class="status-pill live" id="feedHealthPill">Feed Live</div>
            <div class="status-pill info" id="lastUpdatedPill">Last Updated --</div>
          </div>

          <div class="ticker-wrap">
            <div class="ticker-track">
              <div class="ticker-item"><span class="ticker-dot dot-red"></span>Critical deterioration signal surfaced</div>
              <div class="ticker-item"><span class="ticker-dot dot-amber"></span>High-priority patient moved to watch queue</div>
              <div class="ticker-item"><span class="ticker-dot dot-green"></span>Stable recovery trend confirmed</div>
              <div class="ticker-item"><span class="ticker-dot dot-blue"></span>Command center telemetry synchronized</div>
              <div class="ticker-item"><span class="ticker-dot dot-green"></span>Clinical workflow operational</div>

              <div class="ticker-item"><span class="ticker-dot dot-red"></span>Critical deterioration signal surfaced</div>
              <div class="ticker-item"><span class="ticker-dot dot-amber"></span>High-priority patient moved to watch queue</div>
              <div class="ticker-item"><span class="ticker-dot dot-green"></span>Stable recovery trend confirmed</div>
              <div class="ticker-item"><span class="ticker-dot dot-blue"></span>Command center telemetry synchronized</div>
              <div class="ticker-item"><span class="ticker-dot dot-green"></span>Clinical workflow operational</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Hospital Command Wall</h2>
            <div class="section-sub">ICU-style telemetry monitors, live triage signals, backend workflow persistence, patient detail drawer, and audit trail visibility.</div>
          </div>
          <div class="status-pill live" id="wallStatus">Live</div>
        </div>

        <div class="telemetry-top">
          <div class="stat-card">
            <div class="k">Open Alerts</div>
            <div class="v" id="open-alerts">0</div>
            <div class="hint">Active alerts surfaced by the intelligence layer.</div>
          </div>
          <div class="stat-card">
            <div class="k">Critical Alerts</div>
            <div class="v" id="critical-alerts">0</div>
            <div class="hint">Highest urgency patients needing immediate attention.</div>
          </div>
          <div class="stat-card">
            <div class="k">Avg Risk Score</div>
            <div class="v" id="avg-risk">0.0</div>
            <div class="hint">Average current risk across filtered monitored patients.</div>
          </div>
          <div class="stat-card">
            <div class="k">Filtered Unit</div>
            <div class="v" id="unit-count">All</div>
            <div class="hint">Active wall segmentation view.</div>
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
              <h3>Clinical Workflow</h3>
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
                  <div class="status-pill live">Resolved</div>
                </div>
              </div>
            </div>

            <div class="intel-card">
              <h3>System Health</h3>
              <div class="queue-list" id="system-health-list">
                <div class="queue-item">
                  <div class="queue-copy">Waiting for system health...</div>
                  <div class="status-pill muted">--</div>
                </div>
              </div>
            </div>

            <div class="intel-card">
              <h3>Live Patient Timeline</h3>
              <div class="timeline-panel" id="patient-timeline">
                <div class="timeline-item">
                  <div class="timeline-time">Now</div>
                  <div class="timeline-copy">Waiting for active patient timeline...</div>
                </div>
              </div>
            </div>

            <div class="intel-card">
              <h3>Audit Trail</h3>
              <div class="audit-list" id="audit-log">
                <div class="audit-item">
                  <div>
                    <div class="audit-copy">System initialized</div>
                    <div class="audit-sub">Awaiting user actions</div>
                  </div>
                  <div class="status-pill info">Log</div>
                </div>
              </div>
            </div>

            <div class="intel-card">
              <h3>Reporting Dashboard</h3>
              <div class="report-list" id="reporting-panel">
                <div class="report-item">
                  <div>
                    <div class="report-copy">Loading reporting...</div>
                    <div class="report-sub">Pilot metrics</div>
                  </div>
                  <div class="status-pill info">Report</div>
                </div>
              </div>
              <div class="hero-actions" style="margin-top:12px;">
                <button class="btn secondary small" id="refreshReportingBtn" type="button">Refresh Reporting</button>
                <button class="btn secondary small" id="exportAuditBtn" type="button">Export Audit Log</button>
              </div>
            </div>

            <div class="intel-card">
              <h3>Configurable Thresholds</h3>
              <div class="threshold-list" id="threshold-panel">
                <div class="threshold-item">
                  <div class="report-copy">Loading thresholds...</div>
                  <div class="status-pill info">Config</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="ops-strip">
          <div class="ops-card">
            <h3>Oxygen Trend</h3>
            <div class="trend-bar"><div class="trend-fill"></div></div>
            <div class="trend-labels"><span>Stable</span><span>Drift</span><span>Recovery</span></div>
          </div>

          <div class="ops-card">
            <h3>Risk Prediction Timeline</h3>
            <div class="trend-bar"><div class="trend-fill"></div></div>
            <div class="trend-labels"><span>15 min</span><span>1 hr</span><span>4 hr</span></div>
          </div>

          <div class="ops-card">
            <h3>Deterioration Forecast</h3>
            <div class="forecast">
              <div>15 min: <span class="forecast-mid">Moderate</span></div>
              <div>1 hr: <span class="forecast-high">Elevated</span></div>
              <div>4 hr: <span class="forecast-critical">Critical</span></div>
            </div>
          </div>

          <div class="ops-card">
            <h3>Hospital Capacity</h3>
            <span class="capacity-value">78%</span>
            <span class="capacity-label">Beds Occupied</span>
            <div class="capacity-note">Telemetry coverage and escalation pressure remain within controllable range.</div>
          </div>
        </div>

        <div class="ops-strip">
          <div class="ops-card">
            <h3>Predictive Alert Clusters</h3>
            <div class="forecast">
              <div>Respiratory: 3</div>
              <div>Cardiac: 2</div>
              <div>Hemodynamic: 1</div>
            </div>
          </div>

          <div class="ops-card">
            <h3>Unit Heatmap</h3>
            <div class="heatmap">
              <div class="heat low"></div>
              <div class="heat medium"></div>
              <div class="heat high"></div>
              <div class="heat medium"></div>
              <div class="heat medium"></div>
              <div class="heat low"></div>
              <div class="heat high"></div>
              <div class="heat medium"></div>
            </div>
          </div>

          <div class="ops-card">
            <h3>Workflow Readiness</h3>
            <div class="forecast">
              <div>Backend persistence: Enabled</div>
              <div>Role enforcement: Active</div>
              <div>Audit trail: Live</div>
            </div>
          </div>

          <div class="ops-card">
            <h3>Operations Wall Mode</h3>
            <div class="forecast">
              <div>Patient detail drawer: Enabled</div>
              <div>Stale feed detection: Enabled</div>
              <div>Pilot banner: Active</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="hero-kicker">Hospital value</div>
        <h2 class="section-title">Hospital Impact / ROI Metrics</h2>
        <div class="section-sub">
          Early Risk Alert AI is designed to help hospitals identify deterioration sooner, improve monitoring visibility,
          support faster intervention, and strengthen operational efficiency.
        </div>

        <div class="grid-4" style="margin-top:16px;">
          <div class="info-card">
            <h3>Earlier</h3>
            <h4>Intervention Timing</h4>
            <p>Surfaces subtle deterioration signals sooner so clinical teams can respond before a patient reaches a more severe escalation threshold.</p>
          </div>
          <div class="info-card">
            <h3>Lower</h3>
            <h4>Escalation Risk</h4>
            <p>Supports proactive care decisions that may reduce avoidable respiratory or hemodynamic deterioration events.</p>
          </div>
          <div class="info-card">
            <h3>Better</h3>
            <h4>Monitoring Visibility</h4>
            <p>Gives command-center and clinical teams a clearer real-time view of which patients need attention first.</p>
          </div>
          <div class="info-card">
            <h3>Stronger</h3>
            <h4>Operational Awareness</h4>
            <p>Combines patient risk, alert prioritization, and dashboard visibility into one unified hospital workflow.</p>
          </div>
        </div>

        <div class="grid-2" style="margin-top:16px;">
          <div class="info-card">
            <h3>Potential Clinical Benefits</h3>
            <ul>
              <li>Earlier recognition of respiratory deterioration</li>
              <li>Faster response to rising-risk patients</li>
              <li>Improved prioritization across monitored patients</li>
              <li>More visible escalation trends for care teams</li>
            </ul>
          </div>
          <div class="info-card">
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

    <section class="section">
      <div class="section-card">
        <div class="hero-kicker">Who this platform serves</div>
        <h2 class="section-title">Who This Is For</h2>
        <div class="section-sub">
          Early Risk Alert AI is designed for multiple healthcare and business audiences, giving each stakeholder a clear view of value,
          workflow impact, and platform relevance.
        </div>

        <div class="grid-3" style="margin-top:16px;">
          <div class="info-card"><h3>Hospitals</h3><p>Built for hospitals that need earlier deterioration visibility, stronger monitoring workflows, and clearer command-center prioritization across patient populations.</p></div>
          <div class="info-card"><h3>Clinics</h3><p>Helps outpatient and specialty clinics strengthen patient oversight, identify rising-risk trends, and improve escalation awareness in monitored care settings.</p></div>
          <div class="info-card"><h3>RPM Providers</h3><p>Supports remote patient monitoring programs by surfacing early warning patterns before patients cross critical alert thresholds.</p></div>
          <div class="info-card"><h3>Executives & Operators</h3><p>Gives leadership teams a polished operational view of alert pressure, monitoring visibility, workflow readiness, and enterprise value.</p></div>
          <div class="info-card"><h3>Investors</h3><p>Presents a healthcare AI platform with hospital relevance, command-center presentation, live workflow storytelling, and scalable enterprise SaaS positioning.</p></div>
          <div class="info-card"><h3>Care Teams</h3><p>Helps nurses, respiratory teams, and clinicians identify which patients may need earlier intervention and closer follow-up attention.</p></div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="hero-kicker">Clinical use case scenario</div>
        <h2 class="section-title">Early Detection of Respiratory Deterioration</h2>
        <div class="section-sub" style="max-width:none;color:#d0def2;">
          Early Risk Alert AI enables care teams to detect respiratory decline sooner by identifying subtle warning patterns
          before traditional hospital alarm thresholds are reached.
          <br><br>
          A patient is admitted with moderate respiratory risk and placed under standard monitoring. Continuous vital signals
          are collected including oxygen saturation, heart rate, respiratory rate, and blood pressure.
          <br><br>
          Early Risk Alert AI analyzes these signals in real time using predictive risk models. The platform detects subtle
          deterioration patterns in oxygen saturation trends and respiratory variability that may not yet trigger standard
          hospital alarms.
          <br><br>
          The system generates an early risk alert indicating an increasing probability of respiratory deterioration within the
          next monitoring window. Clinical staff receive the alert through the command center dashboard, providing earlier visibility into patient risk.
          <br><br>
          Care teams can intervene sooner by adjusting oxygen support, increasing monitoring intensity, and reassessing the
          patient before the condition reaches a critical escalation threshold.
        </div>

        <div class="pill-row">
          <div class="impact-pill">Earlier intervention</div>
          <div class="impact-pill">Reduced escalation risk</div>
          <div class="impact-pill">Improved patient monitoring visibility</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="hero-kicker">AI intelligence layer</div>
        <h2 class="section-title">How Early Risk Alert AI Works</h2>
        <div class="section-sub">
          Early Risk Alert AI continuously analyzes patient vital signals using predictive models designed to detect deterioration
          patterns earlier than traditional monitoring systems.
        </div>

        <div class="steps-grid" style="margin-top:16px;">
          <div class="step-card">
            <div class="step-num">1</div>
            <h3>Signal Collection</h3>
            <p>Continuous vital signals including oxygen saturation, heart rate, respiratory rate, and blood pressure are collected from patient monitoring systems.</p>
          </div>
          <div class="step-card">
            <div class="step-num">2</div>
            <h3>Pattern Analysis</h3>
            <p>The AI analyzes signal variability, trend drift, and subtle physiological patterns that may indicate early clinical deterioration.</p>
          </div>
          <div class="step-card">
            <div class="step-num">3</div>
            <h3>Risk Prediction</h3>
            <p>Predictive models calculate dynamic risk scores and estimate the probability of deterioration within upcoming monitoring windows.</p>
          </div>
          <div class="step-card">
            <div class="step-num">4</div>
            <h3>Clinical Alert</h3>
            <p>When risk thresholds increase, the platform generates an early alert visible in the command center dashboard for rapid clinical response.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="hero-kicker">Terms & privacy</div>
        <h2 class="section-title">Terms of Use + Privacy Notice</h2>
        <div class="legal-grid" style="margin-top:16px;">
          <div class="info-card">
            <h3>Terms of Use</h3>
            <p>
              This platform environment is intended for product demonstration, workflow evaluation, and pilot review.
              It is not intended to replace licensed clinical judgment, hospital policy, or emergency response procedures.
            </p>
            <p>
              Hospital operators, investors, executives, and reviewers should evaluate the platform in accordance with their
              own internal clinical, legal, compliance, procurement, and technical review processes.
            </p>
          </div>
          <div class="info-card">
            <h3>Privacy Notice</h3>
            <p>
              Demo content shown in this command center may include simulated patient records, generated risk events, and
              workflow state used for presentation purposes. Sensitive production patient data should only be used in secure,
              compliant deployment environments with appropriate access control, logging, and policy protections.
            </p>
            <p>
              Workflow actions, audit events, and pilot configuration changes may be stored for evaluation, reporting, and system testing purposes.
            </p>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC · Predictive clinical intelligence platform · Hospital command center wall · Backend workflow persistence · Patient detail panel · Audit trail · Pilot-safe environment
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
      <div class="drawer-k">Explainable Alert Reason</div>
      <div class="drawer-v" id="drawerExplainability">--</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Workflow Status</div>
      <div class="drawer-v" id="drawerWorkflow">No workflow action saved yet.</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Timeline Snapshot</div>
      <div class="timeline-panel" id="drawerTimeline">
        <div class="timeline-item">
          <div class="timeline-time">Now</div>
          <div class="timeline-copy">No patient selected.</div>
        </div>
      </div>
    </div>
  </aside>

  <script>
    const fallbackPatients = [
      {
        patient_id: "p101",
        patient_name: "Patient 1042",
        room: "ICU-12",
        program: "Cardiac",
        vitals: {heart_rate:128, systolic_bp:165, diastolic_bp:102, spo2:89},
        risk: {risk_score:9.1, severity:"critical", alert_message:"Oxygen saturation critical", recommended_action:"Escalate immediately to respiratory response and bedside reassessment."}
      },
      {
        patient_id: "p202",
        patient_name: "Patient 2188",
        room: "Stepdown-04",
        program: "Pulmonary",
        vitals: {heart_rate:100, systolic_bp:150, diastolic_bp:90, spo2:93},
        risk: {risk_score:3.3, severity:"high", alert_message:"Oxygen saturation below target", recommended_action:"Notify assigned clinician, review vitals, and reassess within 15 minutes."}
      },
      {
        patient_id: "p303",
        patient_name: "Patient 3045",
        room: "Telemetry-09",
        program: "Cardiac",
        vitals: {heart_rate:111, systolic_bp:162, diastolic_bp:97, spo2:95},
        risk: {risk_score:4.4, severity:"high", alert_message:"Blood pressure elevated", recommended_action:"Notify assigned clinician, review vitals, and reassess within 15 minutes."}
      },
      {
        patient_id: "p404",
        patient_name: "Patient 4172",
        room: "Ward-21",
        program: "Recovery",
        vitals: {heart_rate:84, systolic_bp:128, diastolic_bp:82, spo2:97},
        risk: {risk_score:1.8, severity:"stable", alert_message:"Vitals stable", recommended_action:"Continue routine monitoring."}
      }
    ];

    let activePatients = [];
    let activeAlerts = [];
    let currentUnitFilter = "all";
    let selectedPatientId = null;
    let workflowState = {};
    let auditLog = [];
    let reportingData = {};
    let thresholdData = {};
    let pilotStatusData = {};
    let systemHealthData = {};
    let lastSnapshotAt = null;
    let consecutiveFeedErrors = 0;

    function safe(v, fallback="--"){
      return v === undefined || v === null || v === "" ? fallback : v;
    }

    function statusClass(status){
      const s = String(status || "").toLowerCase();
      if (s === "critical") return "critical";
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
      if (s === "resolved") return "Resolved";
      return "Stable";
    }

    function ecgClass(status){
      const s = String(status || "").toLowerCase();
      if (s === "critical") return "ecg-red";
      if (s === "high" || s === "moderate") return "ecg-amber";
      return "ecg-green";
    }

    function roomToUnit(room){
      const r = String(room || "").toLowerCase();
      if (r.includes("icu")) return "icu";
      if (r.includes("stepdown")) return "stepdown";
      if (r.includes("telemetry")) return "telemetry";
      if (r.includes("ward")) return "ward";
      if (r.includes("rpm") || r.includes("home")) return "rpm";
      return "other";
    }

    function unitLabel(unit){
      if (unit === "icu") return "ICU";
      if (unit === "stepdown") return "Stepdown";
      if (unit === "telemetry") return "Telemetry";
      if (unit === "ward") return "Ward";
      if (unit === "rpm") return "RPM / Home";
      return "All Units";
    }

    function roleLabel(role){
      const r = String(role || "").toLowerCase();
      if (r === "viewer") return "Viewer";
      if (r === "operator") return "Operator";
      if (r === "nurse") return "Nurse";
      if (r === "physician") return "Physician";
      if (r === "admin") return "Admin";
      return "Viewer";
    }

    function hasPermission(action){
      const permissions = pilotStatusData.permissions || [];
      return permissions.includes(action) || permissions.includes("admin");
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

    function formatTime(iso){
      if (!iso) return "--";
      try{
        return new Date(iso).toLocaleTimeString();
      }catch(e){
        return "--";
      }
    }

    function secondsSince(iso){
      if (!iso) return 99999;
      try{
        const then = new Date(iso).getTime();
        const now = Date.now();
        return Math.floor((now - then) / 1000);
      }catch(e){
        return 99999;
      }
    }

    function explainabilityForPatient(patient){
      const parts = [];
      const spo2 = Number(patient.spo2 || 0);
      const hr = Number(patient.heart_rate || 0);
      const sbp = Number(patient.bp_systolic || 0);

      if (spo2 && spo2 < 90) parts.push("SpO₂ fell below 90%");
      else if (spo2 && spo2 < 94) parts.push("SpO₂ dropped below target range");

      if (hr && hr >= 130) parts.push("heart rate reached critical range");
      else if (hr && hr >= 110) parts.push("heart rate remained elevated");

      if (sbp && sbp >= 180) parts.push("systolic blood pressure reached crisis range");
      else if (sbp && sbp >= 160) parts.push("systolic blood pressure remained elevated");

      if (!parts.length) parts.push("combined vital pattern suggests current risk state");
      return parts.join(", ") + ".";
    }

    function normalizePatient(raw) {
      if (!raw) return null;
      const vitals = raw.vitals || {};
      const risk = raw.risk || {};
      const room = raw.room || raw.bed || "ICU Bed";
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

    function normalizeAlert(alert) {
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
        state: "new",
        ack: false,
        assigned: false,
        escalated: false,
        role: "",
        updated_at: ""
      };
    }

    function workflowText(patientId){
      const record = getWorkflowRecord(patientId);
      const parts = [];
      if (record.state) parts.push(String(record.state).replace(/_/g, " "));
      if (record.role) parts.push("by " + roleLabel(record.role));
      if (record.updated_at) parts.push("updated " + formatTime(record.updated_at));
      return parts.length ? parts.join(" · ") : "No workflow action saved yet.";
    }

    async function loadPilotStatus(){
      try{
        const res = await fetch("/api/pilot-status", {cache:"no-store"});
        if (!res.ok) return;
        pilotStatusData = await res.json();
      }catch(err){
        console.error("Failed to load pilot status", err);
      }
    }

    async function loadWorkflowState(){
      try{
        const res = await fetch("/api/workflow", {cache:"no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        workflowState = payload.records || {};
        auditLog = payload.audit_log || [];
      }catch(err){
        console.error("Failed to load workflow state", err);
      }
    }

    async function loadReporting(){
      try{
        const res = await fetch("/api/reporting", {cache:"no-store"});
        if (!res.ok) return;
        reportingData = await res.json();
      }catch(err){
        console.error("Failed to load reporting", err);
      }
    }

    async function loadThresholds(){
      try{
        const res = await fetch("/api/thresholds", {cache:"no-store"});
        if (!res.ok) return;
        thresholdData = await res.json();
      }catch(err){
        console.error("Failed to load thresholds", err);
      }
    }

    async function loadSystemHealth(){
      try{
        const res = await fetch("/api/system-health", {cache:"no-store"});
        if (!res.ok) return;
        systemHealthData = await res.json();
      }catch(err){
        console.error("Failed to load system health", err);
      }
    }

    async function postWorkflowAction(patientId, action){
      try{
        const res = await fetch("/api/workflow/action", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            patient_id: patientId,
            action: action
          })
        });

        const payload = await res.json();
        if (!res.ok || !payload.ok){
          alert(payload.error || "Action not permitted.");
          return false;
        }

        workflowState[String(patientId)] = payload.record || {};
        await loadWorkflowState();
        await loadReporting();
        return true;
      }catch(err){
        console.error("Workflow action request failed", err);
        alert("Unable to save workflow action.");
        return false;
      }
    }

    async function exportAudit(){
      try{
        const res = await fetch("/api/audit/export", {cache:"no-store"});
        const payload = await res.json();

        if (!res.ok){
          alert(payload.error || "Export unavailable.");
          return;
        }

        const blob = new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "early_risk_alert_audit_export.json";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }catch(err){
        console.error("Export failed", err);
        alert("Unable to export audit log.");
      }
    }

    function renderPilotStatus(){
      document.getElementById("currentRolePill").textContent = "Role: " + roleLabel(pilotStatusData.user_role || "--");
      document.getElementById("currentUserPill").textContent = "User: " + safe(pilotStatusData.user_name, "--");
      document.getElementById("pilotModePill").textContent = (pilotStatusData.pilot_mode ? "Pilot Mode On" : "Pilot Mode Off");
      document.getElementById("pilotModePill").className = pilotStatusData.pilot_mode ? "status-pill watch" : "status-pill muted";
    }

    function renderFeedStatus(){
      const lastUpdatedPill = document.getElementById("lastUpdatedPill");
      const feedHealthPill = document.getElementById("feedHealthPill");

      if (!lastSnapshotAt){
        lastUpdatedPill.textContent = "Last Updated --";
        feedHealthPill.textContent = "Feed Pending";
        feedHealthPill.className = "status-pill watch";
        return;
      }

      lastUpdatedPill.textContent = "Last Updated " + formatTime(lastSnapshotAt);
      const age = secondsSince(lastSnapshotAt);

      if (consecutiveFeedErrors >= 2 || age > 20){
        feedHealthPill.textContent = "Feed Stale";
        feedHealthPill.className = "status-pill critical";
      } else if (age > 10){
        feedHealthPill.textContent = "Feed Delayed";
        feedHealthPill.className = "status-pill watch";
      } else {
        feedHealthPill.textContent = "Feed Live";
        feedHealthPill.className = "status-pill live";
      }
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

    function monitorActionButtons(patient, record, pid){
      const canAck = hasPermission("ack");
      const canAssign = hasPermission("assign");
      const canEscalate = hasPermission("escalate");
      const canResolve = hasPermission("resolve");

      return `
        <button class="btn ${record.ack ? 'live-btn' : 'secondary'} small" onclick="ackPatient('${pid}')" ${canAck ? '' : 'disabled'}>${record.ack ? 'ACK Saved' : 'ACK'}</button>
        <button class="btn ${record.assigned ? 'warn-btn' : 'secondary'} small" onclick="assignPatient('${pid}')" ${canAssign ? '' : 'disabled'}>${record.assigned ? 'Assigned' : 'Assign'}</button>
        <button class="btn ${record.escalated ? 'critical-btn' : 'secondary'} small" onclick="escalatePatient('${pid}')" ${canEscalate ? '' : 'disabled'}>${record.escalated ? 'Escalated' : 'Escalate'}</button>
        <button class="btn ${record.state === 'resolved' ? 'live-btn' : 'secondary'} small" onclick="resolvePatient('${pid}')" ${canResolve ? '' : 'disabled'}>${record.state === 'resolved' ? 'Resolved' : 'Resolve'}</button>
        <button class="btn secondary small" onclick="openPatientDrawer('${pid}')">Details</button>
      `;
    }

    function renderMonitor(patient){
      const status = String(patient.status || "").toLowerCase();
      const ecg = ecgClass(status);
      const path = buildPath(waveformPoints(status === "critical" ? "critical" : status === "high" ? "high" : "stable"));
      const riskText = typeof patient.risk_score === "number" ? patient.risk_score.toFixed(1) : safe(patient.risk_score);
      const pid = safe(patient.patient_id);
      const record = getWorkflowRecord(pid);

      return `
        <div class="monitor">
          <div class="monitor-top">
            <div>
              <div class="monitor-bed">${safe(patient.bed, "ICU Bed")}</div>
              <div class="monitor-title">${safe(patient.title, safe(patient.name, "Predictive Risk Monitor"))}</div>
            </div>
            <div class="status-pill ${statusClass(record.state === 'resolved' ? 'stable' : patient.status)}">${pulseLabel(record.state === "new" ? patient.status : record.state)}</div>
          </div>

          <div class="monitor-wave" onclick="openPatientDrawer('${pid}')" style="cursor:pointer;">
            <svg viewBox="0 0 640 120" preserveAspectRatio="none" aria-hidden="true">
              <path class="ecg-path ${ecg}" d="${path}"></path>
            </svg>
          </div>

          <div class="monitor-metrics">
            <div class="metric-box">
              <span class="metric-k">HR</span>
              <span class="metric-v">${safe(Math.round(Number(patient.heart_rate || 0)) || "", "--")}</span>
            </div>
            <div class="metric-box">
              <span class="metric-k">SpO₂</span>
              <span class="metric-v">${safe(Math.round(Number(patient.spo2 || 0)) || "", "--")}</span>
            </div>
            <div class="metric-box">
              <span class="metric-k">BP</span>
              <span class="metric-v">${safe(Math.round(Number(patient.bp_systolic || 0)) || "", "--")}/${safe(Math.round(Number(patient.bp_diastolic || 0)) || "", "--")}</span>
            </div>
            <div class="metric-box">
              <span class="metric-k">Risk</span>
              <span class="metric-v">${riskText}</span>
            </div>
          </div>

          <div class="monitor-story">
            <div class="story-text">${safe(patient.story)}</div>
            <div class="status-pill ${statusClass(record.state === 'resolved' ? 'stable' : patient.status)}">${pid}</div>
          </div>

          <div class="monitor-actions">
            ${monitorActionButtons(patient, record, pid)}
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

    function renderPatients(patients){
      const source = filterPatientsByUnit((patients && patients.length ? patients : fallbackPatients))
        .sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))
        .slice(0, 4);
      document.getElementById("wall").innerHTML = source.map(renderMonitor).join("");
    }

    function renderAlertsList(alerts){
      const source = (alerts && alerts.length ? alerts : [
        {patient_id:"p101", severity:"critical", text:"Clinical alert surfaced", unit:"ICU-12"},
        {patient_id:"p202", severity:"high", text:"Clinical alert surfaced", unit:"Stepdown-04"},
        {patient_id:"p303", severity:"high", text:"Clinical alert surfaced", unit:"Telemetry-09"},
        {patient_id:"p404", severity:"stable", text:"Clinical alert surfaced", unit:"Ward-21"}
      ]).map(normalizeAlert).filter(Boolean).slice(0, 6);

      document.getElementById("queue").innerHTML = source.map(renderAlert).join("");
    }

    function renderTopRiskPatients(patients){
      const source = filterPatientsByUnit((patients && patients.length ? patients : fallbackPatients))
        .sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))
        .slice(0,4);

      document.getElementById("top-risk-list").innerHTML = source.map(p => `
        <div class="queue-item">
          <div class="queue-copy">
            ${safe(p.name)} · ${safe(p.patient_id)} · Risk ${typeof p.risk_score === "number" ? p.risk_score.toFixed(1) : safe(p.risk_score)}
            <div class="report-sub">${explainabilityForPatient(p)}</div>
          </div>
          <div class="status-pill ${statusClass(p.status)}">${pulseLabel(p.status)}</div>
        </div>
      `).join("");
    }

    function renderAIReasoning(patients){
      const source = filterPatientsByUnit((patients && patients.length ? patients : fallbackPatients))
        .sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));

      const top = source[0];
      const panel = document.getElementById("ai-reasoning-panel");

      if (!top) {
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

    function renderWorkflow(alerts){
      const source = (alerts && alerts.length ? alerts : []).map(normalizeAlert).filter(Boolean);
      const allRecords = Object.values(workflowState);
      const ackCount = allRecords.filter(r => r.ack).length;
      const escalatedCount = allRecords.filter(r => r.escalated).length;
      const assignedCount = allRecords.filter(r => r.assigned).length;
      const resolvedCount = allRecords.filter(r => r.state === "resolved").length;

      document.getElementById("wf-new").textContent = String(source.length);
      document.getElementById("wf-ack").textContent = String(ackCount);
      document.getElementById("wf-escalated").textContent = String(escalatedCount);
      document.getElementById("wf-assigned").textContent = String(assignedCount);
      document.getElementById("wf-resolved").textContent = String(resolvedCount);
    }

    function renderSystemHealth(){
      const target = document.getElementById("system-health-list");
      const status = systemHealthData.status || "unknown";
      const pilotMode = !!systemHealthData.pilot_mode;
      const timestamp = systemHealthData.timestamp || null;
      const staleAge = secondsSince(lastSnapshotAt);

      target.innerHTML = `
        <div class="queue-item">
          <div class="queue-copy">System Status<div class="report-sub">${safe(status)}</div></div>
          <div class="status-pill ${status === 'operational' ? 'live' : 'watch'}">${safe(status)}</div>
        </div>
        <div class="queue-item">
          <div class="queue-copy">Pilot Mode<div class="report-sub">${pilotMode ? 'Enabled' : 'Disabled'}</div></div>
          <div class="status-pill ${pilotMode ? 'watch' : 'muted'}">${pilotMode ? 'On' : 'Off'}</div>
        </div>
        <div class="queue-item">
          <div class="queue-copy">Workflow Storage<div class="report-sub">${safe(String(systemHealthData.workflow_storage), '--')}</div></div>
          <div class="status-pill ${systemHealthData.workflow_storage ? 'live' : 'critical'}">${systemHealthData.workflow_storage ? 'Ready' : 'Issue'}</div>
        </div>
        <div class="queue-item">
          <div class="queue-copy">Feed Freshness<div class="report-sub">age ${staleAge}s</div></div>
          <div class="status-pill ${staleAge > 20 ? 'critical' : staleAge > 10 ? 'watch' : 'live'}">${staleAge > 20 ? 'Stale' : staleAge > 10 ? 'Delayed' : 'Fresh'}</div>
        </div>
        <div class="queue-item">
          <div class="queue-copy">System Timestamp<div class="report-sub">${formatTime(timestamp)}</div></div>
          <div class="status-pill info">Clock</div>
        </div>
      `;
    }

    function renderPatientTimeline(patients){
      const source = filterPatientsByUnit((patients && patients.length ? patients : fallbackPatients))
        .sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));

      const top = source[0];
      const panel = document.getElementById("patient-timeline");

      if (!top) {
        panel.innerHTML = `
          <div class="timeline-item">
            <div class="timeline-time">Now</div>
            <div class="timeline-copy">Waiting for active patient timeline...</div>
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

      target.innerHTML = auditLog.slice(0, 8).map(entry => `
        <div class="audit-item">
          <div>
            <div class="audit-copy">${safe(entry.action)} · ${safe(entry.patient_id)}</div>
            <div class="audit-sub">${safe(entry.role)} · ${safe(entry.timestamp || entry.at)}${entry.note ? " · " + safe(entry.note) : ""}</div>
          </div>
          <div class="status-pill info">Log</div>
        </div>
      `).join("");
    }

    function renderReporting(){
      const target = document.getElementById("reporting-panel");
      if (!Object.keys(reportingData || {}).length){
        target.innerHTML = `
          <div class="report-item">
            <div>
              <div class="report-copy">Reporting unavailable</div>
              <div class="report-sub">Try refresh</div>
            </div>
            <div class="status-pill watch">Report</div>
          </div>
        `;
        return;
      }

      target.innerHTML = `
        <div class="report-item">
          <div>
            <div class="report-copy">Total Patients</div>
            <div class="report-sub">Tracked workflow records</div>
          </div>
          <div class="status-pill info">${safe(reportingData.total_patients, 0)}</div>
        </div>
        <div class="report-item">
          <div>
            <div class="report-copy">Acknowledged</div>
            <div class="report-sub">Pilot workflow progress</div>
          </div>
          <div class="status-pill live">${safe(reportingData.acknowledged, 0)}</div>
        </div>
        <div class="report-item">
          <div>
            <div class="report-copy">Assigned</div>
            <div class="report-sub">Clinical assignment state</div>
          </div>
          <div class="status-pill info">${safe(reportingData.assigned, 0)}</div>
        </div>
        <div class="report-item">
          <div>
            <div class="report-copy">Escalated</div>
            <div class="report-sub">Priority interventions</div>
          </div>
          <div class="status-pill critical">${safe(reportingData.escalated, 0)}</div>
        </div>
        <div class="report-item">
          <div>
            <div class="report-copy">Resolved</div>
            <div class="report-sub">Closed workflow state</div>
          </div>
          <div class="status-pill live">${safe(reportingData.resolved, 0)}</div>
        </div>
        <div class="report-item">
          <div>
            <div class="report-copy">Audit Events</div>
            <div class="report-sub">Logged pilot actions</div>
          </div>
          <div class="status-pill watch">${safe(reportingData.audit_events, 0)}</div>
        </div>
      `;
    }

    function renderThresholds(){
      const target = document.getElementById("threshold-panel");
      if (!Object.keys(thresholdData || {}).length){
        target.innerHTML = `
          <div class="threshold-item">
            <div class="report-copy">Thresholds unavailable</div>
            <div class="status-pill watch">Config</div>
          </div>
        `;
        return;
      }

      const entries = Object.entries(thresholdData);
      target.innerHTML = entries.map(([unit, settings]) => `
        <div class="threshold-item">
          <div>
            <div class="report-copy">${unitLabel(unit)}</div>
            <div class="report-sub">risk alert threshold ${safe(settings.risk_alert, '--')}</div>
          </div>
          <div class="status-pill info">Config</div>
        </div>
      `).join("");
    }

    function updateSummary(patients, alerts){
      const sourcePatients = filterPatientsByUnit((patients && patients.length ? patients : fallbackPatients));
      const sourceAlerts = (alerts && alerts.length ? alerts : []).map(normalizeAlert).filter(Boolean);

      const openAlerts = sourceAlerts.length;
      const criticalAlerts = sourceAlerts.filter(a => String(a.severity || "").toLowerCase() === "critical").length;
      const avgRisk = sourcePatients.length
        ? (sourcePatients.reduce((n, p) => n + Number(p.risk_score || 0), 0) / sourcePatients.length)
        : 0;

      document.getElementById("open-alerts").textContent = String(openAlerts);
      document.getElementById("critical-alerts").textContent = String(criticalAlerts);
      document.getElementById("avg-risk").textContent = avgRisk.toFixed(1);
      document.getElementById("unit-count").textContent = unitLabel(currentUnitFilter);
    }

    async function ackPatient(patientId){
      const ok = await postWorkflowAction(patientId, "ack");
      if (!ok) return;
      rerenderAll();
      if (selectedPatientId === patientId) openPatientDrawer(patientId);
    }

    async function assignPatient(patientId){
      const ok = await postWorkflowAction(patientId, "assign");
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

    function rerenderAll(){
      renderPilotStatus();
      renderFeedStatus();
      renderPatients(activePatients);
      renderAlertsList(activeAlerts);
      updateSummary(activePatients, activeAlerts);
      renderTopRiskPatients(activePatients);
      renderAIReasoning(activePatients);
      renderWorkflow(activeAlerts);
      renderSystemHealth();
      renderPatientTimeline(activePatients);
      renderAuditLog();
      renderReporting();
      renderThresholds();
    }

    function applyPayload(payload){
      if (payload && Array.isArray(payload.patients)) {
        activePatients = payload.patients.map(normalizePatient).filter(Boolean);
        activeAlerts = (payload.alerts || []).map(normalizeAlert).filter(Boolean);
        lastSnapshotAt = payload.generated_at || new Date().toISOString();
      } else {
        activePatients = fallbackPatients.map(normalizePatient).filter(Boolean);
        activeAlerts = [
          {patient_id:"p101", severity:"critical", text:"Clinical alert surfaced", unit:"ICU-12"},
          {patient_id:"p202", severity:"high", text:"Clinical alert surfaced", unit:"Stepdown-04"},
          {patient_id:"p303", severity:"high", text:"Clinical alert surfaced", unit:"Telemetry-09"},
          {patient_id:"p404", severity:"stable", text:"Clinical alert surfaced", unit:"Ward-21"}
        ];
        lastSnapshotAt = new Date().toISOString();
      }
      rerenderAll();
    }

    async function refreshSnapshot(){
      try{
        const res = await fetch("/api/v1/live-snapshot?tenant_id=demo&patient_id=p101&refresh=" + Date.now(), {cache:"no-store"});
        if (!res.ok) throw new Error("snapshot failed");
        const payload = await res.json();
        consecutiveFeedErrors = 0;
        applyPayload(payload);
      }catch(err){
        consecutiveFeedErrors += 1;
        console.error("Command center snapshot failed", err);
        applyPayload({});
      }
    }

    async function refreshAllSideData(){
      await Promise.all([
        loadPilotStatus(),
        loadWorkflowState(),
        loadReporting(),
        loadThresholds(),
        loadSystemHealth()
      ]);
      rerenderAll();
    }

    document.getElementById("unitFilter").addEventListener("change", function(){
      currentUnitFilter = this.value;
      document.getElementById("selectedUnitPill").textContent = unitLabel(currentUnitFilter);
      rerenderAll();
    });

    document.getElementById("refreshReportingBtn").addEventListener("click", async function(){
      await loadReporting();
      rerenderAll();
    });

    document.getElementById("exportAuditBtn").addEventListener("click", async function(){
      await exportAudit();
    });

    async function boot(){
      await refreshAllSideData();
      await refreshSnapshot();
      rerenderAll();

      setInterval(async () => {
        await refreshSnapshot();
        await loadSystemHealth();
        rerenderAll();
      }, 5000);

      setInterval(async () => {
        await loadWorkflowState();
        await loadReporting();
        rerenderAll();
      }, 10000);
    }

    boot();
  </script>
</body>
</html>
