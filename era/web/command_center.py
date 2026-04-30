COMMAND_CENTER_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Explainable Rules-Based Command-Center Platform</title>
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
      --live:#6bffb1;
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
    button{font:inherit}
    .shell{max-width:1480px;margin:0 auto;padding:18px 16px 56px}

    .topbar{
      position:sticky;top:0;z-index:70;
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

    .pilot-banner{
      margin-top:18px;
      border:1px solid rgba(255,255,255,.09);
      border-radius:24px;
      padding:18px;
      background:
        linear-gradient(180deg, rgba(91,212,255,.08), rgba(255,255,255,.02)),
        linear-gradient(180deg, rgba(14,24,41,.94), rgba(9,16,30,.98));
      box-shadow:var(--shadow);
    }
    .pilot-banner-grid{
      display:grid;grid-template-columns:1.35fr .95fr;gap:18px;align-items:start;
    }
    .pilot-kicker{
      font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px;
    }
    .pilot-title{
      margin:0 0 10px;
      font-size:clamp(28px,3vw,40px);
      line-height:.95;
      letter-spacing:-.05em;
      font-weight:1000;
    }
    .pilot-copy{
      margin:0;color:#d0def2;font-size:15px;line-height:1.72;
    }
    .policy-pills,.meta-pills,.hero-pills,.toolbar,.wall-tools,.mini-pills{
      display:flex;gap:10px;flex-wrap:wrap;align-items:center;
    }

    .policy-pill,.status-pill{
      display:inline-flex;align-items:center;justify-content:center;
      padding:10px 14px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;
      border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.05);
    }
    .policy-pill{cursor:default}
    .status-pill.live{color:#08111f;background:linear-gradient(135deg,var(--green),#8ff3c1)}
    .status-pill.watch{color:#2a1b00;background:linear-gradient(135deg,var(--amber),#ffdba5)}
    .status-pill.critical{color:#fff3f5;background:linear-gradient(135deg,#ff667d,#ff8e99)}
    .status-pill.info{color:#07101c;background:linear-gradient(135deg,var(--brand1),var(--brand2))}
    .status-pill.muted{color:#dce9ff;background:rgba(255,255,255,.06)}
    .status-pill.dark{color:#dce9ff;background:rgba(255,255,255,.03)}
    .status-pill.wall{color:#07101c;background:linear-gradient(135deg,#b58cff,#d5b9ff)}
    .status-pill.locked{color:#fff3f5;background:linear-gradient(135deg,#8f4fff,#c09cff)}
    .status-pill.live-dot{
      gap:8px;color:#08111f;background:linear-gradient(135deg,#64ffae,#9dffd0);
    }
    .status-pill.live-dot::before{
      content:"";width:8px;height:8px;border-radius:50%;background:#0b7d43;box-shadow:0 0 0 4px rgba(11,125,67,.14);
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
      display:grid;grid-template-columns:1.18fr .82fr;gap:18px;align-items:stretch;
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
    .hero-actions,.section-actions{
      display:flex;gap:12px;flex-wrap:wrap;margin-top:18px;
    }

    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      border:1px solid transparent;cursor:pointer;
      transition:transform .18s ease, box-shadow .18s ease, opacity .18s ease, border-color .18s ease, background .18s ease;
      color:var(--text);
      background:rgba(255,255,255,.04);
    }
    .btn:hover{transform:translateY(-2px)}
    .btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
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
    .btn.wall-btn{background:linear-gradient(135deg,#b58cff,#d8bfff);color:#08111f}
    .btn.wall-btn.active{box-shadow:0 0 0 3px rgba(181,140,255,.18), 0 12px 30px rgba(181,140,255,.2)}

    .toolbar select,.threshold-editor select,.threshold-editor input{
      background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:14px;color:var(--text);
      padding:12px 14px;font:inherit;font-weight:800;
    }
    .threshold-editor input{padding:10px 12px}

    .section{margin-top:18px}
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
      display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px;
    }
    .stat-card{
      border:1px solid var(--line);border-radius:20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(13,23,40,.90), rgba(8,15,28,.95));
      padding:16px;min-height:124px;position:relative;overflow:hidden;
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
      padding:18px;position:relative;overflow:hidden;min-height:386px;
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
    .intel-card,.module-card,.business-card{
      border:1px solid var(--line);border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      padding:18px;box-shadow:0 14px 36px rgba(0,0,0,.26);
    }
    .intel-card h3,.module-card h3,.business-card h3{
      margin:0 0 12px;font-size:20px;font-weight:1000;letter-spacing:-.03em;
    }

    .alert-feed,.queue-list,.timeline-panel,.audit-list,.why-now,.insight-list{
      display:grid;gap:10px;
    }
    .alert-item,.queue-item,.timeline-item,.audit-item,.why-line,.info-line,.threshold-row{
      border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:12px;background:rgba(255,255,255,.03);
    }
    .alert-item,.queue-item,.audit-item,.threshold-row{
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

    .modules-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
    }
    .module-head{
      display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:12px;
    }
    .module-title{
      font-size:19px;font-weight:1000;letter-spacing:-.03em;
    }
    .module-sub{
      font-size:12px;color:#9eb8dc;line-height:1.5;margin-top:4px;
    }
    .module-body{
      min-height:160px;
    }

    .bars{
      display:grid;gap:10px;
    }
    .bar-row{
      display:grid;grid-template-columns:108px 1fr 54px;gap:10px;align-items:center;
    }
    .bar-label{
      font-size:12px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:#dce9ff;
    }
    .bar-track{
      height:12px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden;border:1px solid rgba(255,255,255,.05);
    }
    .bar-fill{
      height:100%;border-radius:999px;background:linear-gradient(135deg,var(--brand1),var(--brand2));
    }
    .bar-fill.warn{background:linear-gradient(135deg,#f4bd6a,#ffdba5)}
    .bar-fill.critical{background:linear-gradient(135deg,#ff667d,#ff9aaa)}
    .bar-value{
      text-align:right;font-size:12px;font-weight:1000;color:#dce9ff;
    }

    .sparkline{
      width:100%;height:72px;border-radius:16px;border:1px solid rgba(255,255,255,.05);background:rgba(255,255,255,.02);
      overflow:hidden;padding:8px;
    }
    .sparkline.large{height:110px}
    .sparkline svg{width:100%;height:100%;display:block}
    .sparkline path.line{
      fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;
    }
    .sparkline.large path.line{stroke-width:3.5}
    .sparkline path.area{opacity:.14}

    .forecast-grid,.roi-grid,.audience-grid,.scenario-grid,.how-grid{
      display:grid;gap:14px;
    }
    .forecast-grid{grid-template-columns:repeat(2,1fr)}
    .roi-grid{grid-template-columns:repeat(4,1fr)}
    .audience-grid,.how-grid{grid-template-columns:repeat(4,1fr)}
    .scenario-grid{grid-template-columns:repeat(3,1fr)}

    .mini-card{
      border:1px solid rgba(255,255,255,.06);
      border-radius:18px;
      background:rgba(255,255,255,.03);
      padding:14px;
      min-height:116px;
    }
    .mini-k{
      font-size:11px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;color:#9adfff;margin-bottom:8px;
    }
    .mini-v{
      font-size:24px;font-weight:1000;letter-spacing:-.04em;line-height:1;
    }
    .mini-copy{
      margin-top:8px;font-size:13px;color:#cfe0f4;line-height:1.55;
    }

    .business-copy{
      color:#cfe0f4;font-size:14px;line-height:1.7;margin:0;
    }

    .detail-drawer{
      position:fixed;top:0;right:-500px;width:480px;max-width:100%;height:100vh;z-index:120;
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

    .terms-modal{
      position:fixed;inset:0;z-index:140;display:none;
      align-items:center;justify-content:center;
      background:rgba(0,0,0,.56);
      padding:20px;
    }
    .terms-modal.show{display:flex}
    .modal-card{
      width:min(760px,100%);
      max-height:88vh;
      overflow:auto;
      border:1px solid var(--line);
      border-radius:24px;
      background:linear-gradient(180deg, rgba(12,22,38,.98), rgba(7,14,26,.99));
      box-shadow:0 20px 60px rgba(0,0,0,.4);
      padding:20px;
    }
    .modal-title{
      margin:0 0 10px;
      font-size:32px;
      line-height:.95;
      letter-spacing:-.05em;
      font-weight:1000;
    }
    .modal-copy{
      color:#d0def2;
      font-size:15px;
      line-height:1.72;
    }

    .threshold-editor{
      display:grid;gap:10px;
    }
    .threshold-editor-grid{
      display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
    }
    .threshold-field{
      display:grid;gap:8px;
    }
    .threshold-field label{
      font-size:11px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;color:#9adfff;
    }
    .note-box{
      border:1px solid rgba(255,255,255,.06);
      border-radius:16px;
      background:rgba(255,255,255,.03);
      padding:12px;
      font-size:13px;
      line-height:1.6;
      color:#d8e6fb;
    }

    .footer{
      margin-top:18px;border:1px solid var(--line);border-radius:22px;padding:18px;
      background:rgba(255,255,255,.03);color:#a7bddc;font-size:13px;line-height:1.6;text-align:center;
    }

    .wall-mode .shell{max-width:100%;padding:10px 10px 34px}
    .wall-mode .topbar-inner{max-width:100%}
    .wall-mode .hero,
    .wall-mode .pilot-banner,
    .wall-mode .business-sections,
    .wall-mode .footer{
      display:none;
    }
    .wall-mode .section-card{
      padding:12px;
      border-radius:20px;
    }
    .wall-mode .command-grid{
      grid-template-columns:1.7fr .8fr;
      gap:12px;
    }
    .wall-mode .monitor{
      min-height:320px;
      padding:14px;
    }
    .wall-mode .side-stack{gap:12px}
    .wall-mode .intel-card{padding:14px}
    .wall-mode .telemetry-top{gap:10px;margin-bottom:12px}
    .wall-mode .stat-card{min-height:100px;padding:14px}

    @media (max-width:1280px){
      .hero-grid,.command-grid,.pilot-banner-grid{grid-template-columns:1fr}
      .modules-grid{grid-template-columns:repeat(2,1fr)}
      .roi-grid,.audience-grid,.scenario-grid,.how-grid{grid-template-columns:repeat(2,1fr)}
    }
    @media (max-width:980px){
      .telemetry-top{grid-template-columns:repeat(2,1fr)}
      .monitor-grid{grid-template-columns:1fr}
      .drawer-grid,.forecast-grid,.threshold-editor-grid{grid-template-columns:1fr}
      .modules-grid{grid-template-columns:1fr}
    }
    @media (max-width:700px){
      .shell{padding:14px 10px 40px}
      .hero{padding:16px}
      .hero-copy h1{font-size:clamp(32px,11vw,52px)}
      .btn{width:100%}
      .telemetry-top,.monitor-metrics,.roi-grid,.audience-grid,.scenario-grid,.how-grid{grid-template-columns:1fr}
      .story-text{max-width:100%}
      .timeline-item{grid-template-columns:1fr}
      .detail-drawer{width:100%}
      .bar-row{grid-template-columns:1fr}
      .bar-value{text-align:left}
    }
  
    /* ERA_COMMAND_PATIENT_EXPLAINABILITY_V2_START */
    .era-priority-context-grid{
      display:grid;
      grid-template-columns:repeat(2,1fr);
      gap:10px;
    }
    .era-priority-context-card{
      border:1px solid rgba(255,255,255,.06);
      border-radius:16px;
      background:rgba(255,255,255,.03);
      padding:12px;
    }
    .era-priority-context-card .context-k{
      font-size:10px;
      font-weight:1000;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#9adfff;
      margin-bottom:6px;
    }
    .era-priority-context-card .context-v{
      font-size:15px;
      font-weight:1000;
      color:#eef4ff;
      line-height:1.35;
    }
    .era-card-context{
      position:relative;
      z-index:2;
      margin-top:12px;
      display:grid;
      grid-template-columns:repeat(2,1fr);
      gap:8px;
    }
    .era-context-chip{
      border:1px solid rgba(255,255,255,.08);
      border-radius:999px;
      padding:7px 9px;
      background:rgba(255,255,255,.04);
      font-size:11px;
      font-weight:950;
      color:#dce9ff;
      line-height:1.2;
    }
    .era-context-chip.critical{background:rgba(255,102,125,.14);border-color:rgba(255,102,125,.32);color:#ffd0d6}
    .era-context-chip.elevated{background:rgba(244,189,106,.14);border-color:rgba(244,189,106,.32);color:#ffe5b8}
    .era-context-chip.watch{background:rgba(122,162,255,.14);border-color:rgba(122,162,255,.32);color:#dce7ff}
    .era-context-chip.low{background:rgba(58,211,143,.12);border-color:rgba(58,211,143,.26);color:#bff6dc}
    .era-help-note{
      margin-top:10px;
      border:1px solid rgba(138,180,255,.25);
      background:rgba(138,180,255,.08);
      border-radius:14px;
      padding:10px;
      font-size:12px;
      color:#dce9ff;
      line-height:1.5;
    }
    @media(max-width:700px){
      .era-priority-context-grid,.era-card-context{grid-template-columns:1fr}
    }
    /* ERA_COMMAND_PATIENT_EXPLAINABILITY_V2_END */

  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="brand-kicker" id="brandKicker">HCP-facing decision-support · Rules-based · Explainable · CITI certified</div>
        <div class="brand-title" id="brandTitle">Early Risk Alert AI</div>
        <div class="brand-sub" id="brandSub">Command-Center Platform</div>
      </div>
      <div class="nav-links">
        <a href="/command-center">Command Center</a>
        <a href="/hospital-demo">Hospital Demo</a>
        <a href="/investor-intake">Investor Access</a>
        <a href="/executive-walkthrough">Executive Walkthrough</a>
        <a href="/admin/review">Admin Review</a>
        <a href="/pilot-docs">Pilot Docs</a>
        <a href="/validation-evidence">Evidence Packet</a>
        <a href="/validation-intelligence">Validation Intelligence</a>
        <a href="/logout">Logout</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="pilot-banner">
      <div class="pilot-banner-grid">
        <div>
          <div class="pilot-kicker">Pilot mode · CITI certified · MFA implemented · MIMIC-IV validation published</div>
          <h2 class="pilot-title">Controlled demonstration environment for workflow evaluation, security review, and stakeholder readiness</h2>
          <p class="pilot-copy">
            This platform is presented in a controlled pilot environment for demonstration, workflow evaluation, and stakeholder review.
            It does not replace clinical judgment, hospital protocols, or emergency response systems.
            All review-priority outputs, monitored trends, and thresholds are presented to support — not replace — licensed clinical decision-making.
          </p>
        </div>

        <div style="display:grid;gap:12px;">
          <div class="policy-pills">
            <button class="policy-pill status-pill dark" type="button" onclick="openPolicyModal('terms')">Terms</button>
            <button class="policy-pill status-pill dark" type="button" onclick="openPolicyModal('privacy')">Privacy</button>
            <button class="policy-pill status-pill dark" type="button" onclick="openPolicyModal('disclaimer')">Pilot Disclaimer</button>
          </div>

          <div class="meta-pills">
            <div class="status-pill live-dot" id="liveSystemPill">Live System</div>
            <div class="status-pill muted" id="lastUpdatedPillTop">Last Updated --</div>
            <div class="status-pill info" id="topRiskPill">Top Review-Priority Patient --</div>
          </div>

          <div class="meta-pills">
            <div class="status-pill locked" id="unitLockPill">Unit Lock</div>
            <div class="status-pill dark" id="securityModePill">MFA Implemented · Pilot Security Layer</div>
          </div>
        </div>
      </div>
    </section>
<!-- ERA_COMMAND_CENTER_INTERACTIVE_QUEUE_V1_START -->
<section id="era-live-review-queue" class="era-live-review-queue" style="margin:22px 0 28px; padding:18px; border:1px solid rgba(166,255,194,.34); border-radius:24px; background:linear-gradient(180deg,rgba(9,18,32,.86),rgba(9,18,32,.68)); box-shadow:0 18px 45px rgba(0,0,0,.22);">
  <div style="display:flex; flex-wrap:wrap; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:14px;">
    <div>
      <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 11px; border-radius:999px; border:1px solid rgba(166,255,194,.42); color:#b8ffc8; font-weight:900; letter-spacing:.08em; text-transform:uppercase; font-size:.72rem;">Live Review Queue • Simulated Pilot View</div>
      <h2 style="margin:10px 0 5px; font-size:clamp(1.65rem,3vw,2.7rem); line-height:1.02;">Prioritized patient review queue</h2>
      <p style="margin:0; max-width:900px; opacity:.86;">Compact command-center view showing queue rank, priority tier, primary driver, trend, lead-time context, and workflow state. Decision-support only; not diagnosis or treatment direction.</p>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
      <button type="button" data-era-filter="all" class="era-q-btn era-q-active" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(166,255,194,.16); color:inherit; font-weight:800; cursor:pointer;">All</button>
      <button type="button" data-era-filter="critical" class="era-q-btn" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(255,255,255,.06); color:inherit; font-weight:800; cursor:pointer;">Critical</button>
      <button type="button" data-era-filter="elevated" class="era-q-btn" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(255,255,255,.06); color:inherit; font-weight:800; cursor:pointer;">Elevated</button>
      <select id="era-q-sort" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(255,255,255,.08); color:inherit; font-weight:800;">
        <option value="rank">Sort: Queue Rank</option>
        <option value="risk">Sort: Risk Score</option>
        <option value="lead">Sort: Lead Time</option>
      </select>
    </div>
  </div>

  <div id="era-q-cards" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-bottom:14px;"></div>

  <div style="overflow-x:auto; border:1px solid rgba(255,255,255,.10); border-radius:18px;">
    <table style="width:100%; min-width:980px; border-collapse:collapse;">
      <thead>
        <tr style="background:rgba(120,160,220,.18);">
          <th style="text-align:left; padding:11px;">Rank</th>
          <th style="text-align:left; padding:11px;">Patient</th>
          <th style="text-align:left; padding:11px;">Unit</th>
          <th style="text-align:left; padding:11px;">Priority Tier</th>
          <th style="text-align:left; padding:11px;">Risk</th>
          <th style="text-align:left; padding:11px;">Primary Driver</th>
          <th style="text-align:left; padding:11px;">Trend</th>
          <th style="text-align:left; padding:11px;">Lead Time</th>
          <th style="text-align:left; padding:11px;">Workflow</th>
          <th style="text-align:left; padding:11px;">Action</th>
        </tr>
      </thead>
      <tbody id="era-q-table"></tbody>
    </table>
  </div>

  <div id="era-q-detail" style="margin-top:14px; padding:14px; border-radius:18px; background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.10);">
    <strong>Selected review basis:</strong>
    <span id="era-q-detail-text">Select a row or card to view a concise explainability summary.</span>
  </div>

  <p style="margin:13px 0 0; padding:11px 13px; border-radius:16px; border:1px solid rgba(255,207,117,.45); color:#ffd977; background:rgba(255,207,117,.08); font-weight:700;">
    Guardrail: simulated de-identified demonstration queue. Workflow actions are review-state examples only. No diagnosis, treatment direction, clinician replacement, or autonomous escalation.
  </p>
</section>

<script>
(function(){
  if (window.__ERA_INTERACTIVE_QUEUE_V1__) return;
  window.__ERA_INTERACTIVE_QUEUE_V1__ = true;

  const patients = [
    {rank:1, id:"ICU-12", unit:"ICU", tier:"Critical", risk:99, driver:"SpO₂ decline", trend:"Worsening", lead:4.8, state:"Needs review", basis:"ICU-12 is queue-ranked first because the score is high, oxygenation is the primary driver, the trend is worsening, and the alert appears within retrospective lead-time context."},
    {rank:2, id:"ICU-07", unit:"ICU", tier:"Critical", risk:94, driver:"BP instability", trend:"Worsening", lead:4.1, state:"Acknowledged", basis:"ICU-07 remains high priority because blood-pressure instability is the dominant driver with worsening trend and conservative-threshold crossing context."},
    {rank:3, id:"TEL-18", unit:"Telemetry", tier:"Elevated", risk:86, driver:"HR instability", trend:"Stable / Watch", lead:3.6, state:"Assigned", basis:"TEL-18 is elevated rather than critical because the primary driver is heart-rate instability with a watchful but less severe trend pattern."},
    {rank:4, id:"SDU-04", unit:"Stepdown", tier:"Watch", risk:72, driver:"RR elevation", trend:"Stable", lead:2.9, state:"Monitoring", basis:"SDU-04 is monitored because respiratory-rate elevation contributes to risk, but the trend is currently stable and the queue rank is lower."}
  ];

  const tierStyle = {
    "Critical":"background:rgba(255,107,107,.16);border:1px solid rgba(255,107,107,.45);color:#ffd0d0;",
    "Elevated":"background:rgba(255,207,117,.14);border:1px solid rgba(255,207,117,.45);color:#ffdf91;",
    "Watch":"background:rgba(145,197,255,.14);border:1px solid rgba(145,197,255,.42);color:#cfe4ff;",
    "Low":"background:rgba(166,255,194,.12);border:1px solid rgba(166,255,194,.42);color:#caffd8;"
  };

  const cards = document.getElementById("era-q-cards");
  const tbody = document.getElementById("era-q-table");
  const detail = document.getElementById("era-q-detail-text");
  const sortEl = document.getElementById("era-q-sort");
  const buttons = Array.from(document.querySelectorAll("[data-era-filter]"));
  let filter = "all";

  function badge(text, style) {
    return '<span style="display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;font-weight:900;font-size:.78rem;white-space:nowrap;' + style + '">' + text + '</span>';
  }

  function workflowButton(label, id) {
    return '<button type="button" data-era-action="' + id + '" style="border:1px solid rgba(255,255,255,.22);border-radius:999px;padding:7px 10px;background:rgba(255,255,255,.07);color:inherit;font-weight:800;cursor:pointer;">' + label + '</button>';
  }

  function sortedRows() {
    let rows = patients.slice();
    if (filter !== "all") rows = rows.filter(p => p.tier.toLowerCase() === filter);
    const sort = sortEl ? sortEl.value : "rank";
    if (sort === "risk") rows.sort((a,b) => b.risk - a.risk);
    else if (sort === "lead") rows.sort((a,b) => b.lead - a.lead);
    else rows.sort((a,b) => a.rank - b.rank);
    return rows;
  }

  function render() {
    const rows = sortedRows();

    if (cards) {
      cards.innerHTML = rows.slice(0,3).map(p => `
        <button type="button" data-era-select="${p.id}" style="text-align:left; border:1px solid rgba(255,255,255,.13); border-radius:18px; padding:14px; background:rgba(255,255,255,.055); color:inherit; cursor:pointer;">
          <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px;">
            <strong style="font-size:1.25rem;">#${p.rank} ${p.id}</strong>
            ${badge(p.tier, tierStyle[p.tier] || "")}
          </div>
          <div style="font-size:2rem;line-height:1;font-weight:950;color:#b8ffc8;">${p.risk}%</div>
          <div style="margin-top:8px;"><strong>Driver:</strong> ${p.driver}</div>
          <div><strong>Trend:</strong> ${p.trend}</div>
          <div><strong>Lead time:</strong> ~${p.lead} hrs</div>
        </button>
      `).join("");
    }

    if (tbody) {
      tbody.innerHTML = rows.map(p => `
        <tr data-era-select="${p.id}" style="cursor:pointer;">
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:950;">#${p.rank}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:900;">${p.id}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${p.unit}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${badge(p.tier, tierStyle[p.tier] || "")}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:950;color:#b8ffc8;">${p.risk}%</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:850;">${p.driver}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${p.trend}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:900;">~${p.lead} hrs</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${p.state}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${workflowButton("Acknowledge", p.id)}</td>
        </tr>
      `).join("");
    }
  }

  function selectPatient(id) {
    const p = patients.find(x => x.id === id);
    if (!p || !detail) return;
    detail.textContent = p.basis;
  }

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      filter = btn.getAttribute("data-era-filter") || "all";
      buttons.forEach(b => {
        b.classList.remove("era-q-active");
        b.style.background = "rgba(255,255,255,.06)";
      });
      btn.classList.add("era-q-active");
      btn.style.background = "rgba(166,255,194,.16)";
      render();
    });
  });

  if (sortEl) sortEl.addEventListener("change", render);

  document.addEventListener("click", function(e){
    const action = e.target.closest("[data-era-action]");
    if (action) {
      const id = action.getAttribute("data-era-action");
      const p = patients.find(x => x.id === id);
      if (p) {
        p.state = p.state === "Acknowledged" ? "Assigned" : "Acknowledged";
        selectPatient(id);
        render();
      }
      return;
    }
    const row = e.target.closest("[data-era-select]");
    if (row) selectPatient(row.getAttribute("data-era-select"));
  });

  render();
  selectPatient("ICU-12");
})();
</script>
<!-- ERA_COMMAND_CENTER_INTERACTIVE_QUEUE_V1_END -->


    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">Pilot operating layer</div>
          <h1 id="heroTitle">Early Risk Alert AI — Explainable Rules-Based Command-Center Platform</h1>
          <p id="heroCopy">
            <strong style="color:#eef4ff">Rules-based prioritization with explainable deltas — not black-box AI.</strong> Retrospective MIMIC validation showed <strong style="color:#3ad38f">94.3% alert reduction at t=6.0</strong>, <strong style="color:#3ad38f">4.5% ERA false-positive rate</strong>, and <strong style="color:#7aa2ff">15.3% event-cluster detection at t=6.0</strong> at the conservative t=6.0 threshold. Lead-time analysis showed a median <strong style="color:#3ad38f">4.0-hour</strong> first-flag timing among detected event clusters within the 6-hour pre-event window.
          <br><span style="font-size:14px;color:#9fb4d6;margin-top:8px;display:block">FHIR R4 and HL7 integration remain on the product roadmap. The current pilot entry point is retrospective validation via de-identified CSV, which requires no EHR integration and can begin within days of data availability.</span>
          </p>

          <div class="hero-actions">
            <a class="btn primary" href="/hospital-demo">Request Live Demo</a>
            <a class="btn secondary" href="/investor-intake">Investor Access</a>
            <a class="btn secondary" href="/admin/review">Open Admin Review</a>
            <a class="btn secondary" href="/pilot-docs">Pilot Docs</a>
            <a class="btn secondary" href="/validation-intelligence">Command Validation</a>
            <a class="btn secondary" href="/model-card" target="_blank" rel="noopener" title="Validation methodology, signal weights, and performance results">Model Card</a>
          </div>

          <div class="hero-pills" style="margin-top:18px;">
            <div class="status-pill watch" id="pilotModePill">Pilot Mode</div>
            <div class="status-pill info" id="currentRolePill">Role</div>
            <div class="status-pill muted" id="currentUserPill">User</div>
            <div class="status-pill info" id="unitAccessPill">Unit Access</div>
            <div class="status-pill live" id="feedHealthPill">Feed Live</div>
            <div class="status-pill muted" id="lastUpdatedPill">Last Updated</div>
            <a class="status-pill info" href="/model-card" target="_blank" rel="noopener" style="text-decoration:none;cursor:pointer" title="Validation methodology, signal weights, performance results">Model Card</a>
            <a class="status-pill live" id="mimicValidationPill" href="/validation-intelligence" style="text-decoration:none;cursor:pointer" title="MIMIC retrospective validation intelligence">MIMIC Validation: Published</a>
          </div>
        </div>

        <div class="hero-side">
          <div class="hero-kicker">Access + scope</div>
          <p>
            Hospital pilot users can be restricted to a single unit such as ICU, telemetry, stepdown, ward, or RPM.
            Admin users retain full hospital-wide visibility across all units and all pilot accounts.
            Trend views, audit visibility, and thresholds are wired to backend routes and reflect your current access scope.
          </p>

          <div class="toolbar" style="margin-top:18px;">
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

          <div class="wall-tools" style="margin-top:14px;">
            <button class="btn wall-btn" id="wallModeBtn" type="button" onclick="toggleWallMode()">Enable Wall Mode</button>
            <div class="status-pill wall" id="wallModePill">Wall Mode Off</div>
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
            <div class="section-sub">Explainable rules-based prioritization — not black-box AI. Retrospective MIMIC validation showed 94.3% alert reduction at t=6.0, 4.5% ERA false-positive rate, and 15.3% event-cluster detection at t=6.0 at the conservative t=6.0 threshold. Lead-time analysis showed a median 4.0-hour first-flag timing among detected event clusters within the 6-hour pre-event window. Structured patient summaries, delta trend context, workflow handling, and persistent audit visibility remain decision-support only.</div>
          </div>
          <div class="mini-pills">
            <div class="status-pill live" id="wallStatus">Live</div>
            <div class="status-pill info" id="wallModeStatusPill">Standard View</div>
          </div>
        </div>

        <!-- PATIENT SEARCH BAR -->
        <div style="margin-bottom:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
          <div style="flex:1;min-width:200px;position:relative">
            <input id="patientSearchInput" type="text" placeholder="Search by patient ID, room, or unit..." oninput="filterPatientSearch()" style="width:100%;padding:10px 14px 10px 36px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.05);color:#eef4ff;font:inherit;font-size:13px;outline:none;">
            <span style="position:absolute;left:12px;top:50%;transform:translateY(-50%);font-size:14px;opacity:.5">&#x1F50D;</span>
          </div>
          <button id="demModeBtn" onclick="toggleDemoMode()" style="padding:10px 16px;border-radius:12px;border:1px solid rgba(244,189,106,.3);background:rgba(244,189,106,.08);color:#ffe7bf;font:inherit;font-size:12px;font-weight:900;cursor:pointer;white-space:nowrap;letter-spacing:.06em;text-transform:uppercase">&#x1F3AC; Live Demo Mode: OFF</button>
          <a href="/pilot-onboarding" target="_blank" style="padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);color:#9adfff;font-size:12px;font-weight:900;text-decoration:none;white-space:nowrap;letter-spacing:.06em;text-transform:uppercase">&#x1F4CB; Pilot Checklist</a>
        </div>

        <div class="telemetry-top">
          <div class="stat-card">
            <div class="k">Open Review Items</div>
            <div class="v" id="open-alerts">0</div>
            <div class="hint">Active alert count within your current access scope.</div>
          </div>
          <div class="stat-card">
            <div class="k">Higher-Priority Review Items</div>
            <div class="v" id="critical-alerts">0</div>
            <div class="hint">Highest-priority review items within the current access scope.</div>
          </div>
          <div class="stat-card">
            <div class="k">Avg Review Score</div>
            <div class="v" id="avg-risk">0%</div>
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
              <h3>Live Review Feed</h3>
              <div class="alert-feed" id="queue"></div>
            </div>

            <div class="intel-card">
              <h3>Top Review-Priority Queue</h3>
              <div class="queue-list" id="top-risk-list"></div>
            </div>

            <div class="intel-card">
              <h3>Explainable Review Basis</h3>
              <div class="why-now" id="ai-reasoning-panel">
                <div class="why-line">Waiting for patient intelligence stream...</div>
              </div>
            </div>

            <div class="intel-card">
              <h3>Review Insight</h3>
              <div class="insight-list" id="ai-insight-panel">
                <div class="info-line">Waiting for active review insight...</div>
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
              <h3>Security + Pilot Controls</h3>
              <div class="queue-list" id="security-controls-list"></div>
            </div>

            <div class="intel-card">
              <h3>Governance + Readiness</h3>
              <div class="queue-list" id="readiness-evidence-list"></div>
            </div>

            <div class="intel-card">
              <h3>Patient Timeline</h3>
              <div class="timeline-panel" id="patient-timeline"></div>
            </div>

            <div class="intel-card">
              <h3>Audit Trail</h3>
              <div class="audit-list" id="audit-log"></div>
            </div>

            <div class="intel-card">
              <h3>Audit Summary</h3>
              <div class="queue-list" id="audit-summary-list"></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Reporting + Operations Overview</h2>
            <div class="section-sub">Backend-connected operational modules powered by live snapshot data, workflow state, threshold storage, and true trend history when available.</div>
          </div>
          <div class="status-pill muted">Pilot / Evaluation Mode</div>
        </div>

        <div class="modules-grid">
          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Reporting Dashboard</div>
                <div class="module-sub">Snapshot-based operational summary</div>
              </div>
              <div class="status-pill info">Live</div>
            </div>
            <div class="module-body" id="reporting-dashboard-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Configurable Thresholds</div>
                <div class="module-sub">Admin-editable when backend threshold route is enabled</div>
              </div>
              <div class="status-pill watch" id="thresholdModePill">Read Only</div>
            </div>
            <div class="module-body" id="thresholds-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Oxygen Trend</div>
                <div class="module-sub">True trend history with fallback display</div>
              </div>
              <div class="status-pill muted" id="oxygenTrendModePill">Trend</div>
            </div>
            <div class="module-body" id="oxygen-trend-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Review Trend Timeline</div>
                <div class="module-sub">Near-term risk progression from stored trend history</div>
              </div>
              <div class="status-pill info">AI</div>
            </div>
            <div class="module-body" id="risk-timeline-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Hospital Capacity</div>
                <div class="module-sub">Visible-unit operational load estimate</div>
              </div>
              <div class="status-pill muted">Operational</div>
            </div>
            <div class="module-body" id="capacity-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Review Priority Outlook</div>
                <div class="module-sub">Supportive forecast, not clinical instruction</div>
              </div>
              <div class="status-pill critical">Priority</div>
            </div>
            <div class="module-body" id="forecast-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Review Pattern Clusters</div>
                <div class="module-sub">Grouped signal drivers across visible patients</div>
              </div>
              <div class="status-pill watch">Grouped</div>
            </div>
            <div class="module-body" id="clusters-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Unit Heatmap</div>
                <div class="module-sub">Relative concentration of risk by unit</div>
              </div>
              <div class="status-pill info">Command</div>
            </div>
            <div class="module-body" id="heatmap-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Workflow Readiness</div>
                <div class="module-sub">Operational response status across current scope</div>
              </div>
              <div class="status-pill live">Ready</div>
            </div>
            <div class="module-body" id="workflow-readiness-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Route + Document Status</div>
                <div class="module-sub">Checks core paths and pilot packet docs used in this bundle</div>
              </div>
              <div class="status-pill info">Stable</div>
            </div>
            <div class="module-body" id="route-status-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">Site-Specific Pilot Packet</div>
                <div class="module-sub">Hospital packet placeholders, responsibilities, support path, and closeout items</div>
              </div>
              <div class="status-pill watch">Packet</div>
            </div>
            <div class="module-body" id="pilot-packet-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">MIMIC Validation Intelligence</div>
                <div class="module-sub">Alert reduction, lead time, threshold profiles, and representative review examples</div>
              </div>
              <div class="status-pill green" style="background:rgba(58,211,143,.14);border:1px solid rgba(58,211,143,.3);color:#b6f5d9">Published</div>
            </div>
            <div class="module-body" id="validation-results-module"></div>
          </div>

          <div class="module-card">
            <div class="module-head">
              <div>
                <div class="module-title">No-Commitment Analysis</div>
                <div class="module-sub">Upload your de-identified CSV — receive results same session</div>
              </div>
              <div class="status-pill live">Open</div>
            </div>
            <div class="module-body" id="no-commit-module"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section business-sections">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Operations Wall Mode</h2>
            <div class="section-sub">High-visibility view for demonstrations, executive walkthroughs, and command-center presentations.</div>
          </div>
          <div class="section-actions" style="margin-top:0;">
            <button class="btn wall-btn" type="button" onclick="toggleWallMode()">Enable Wall Mode</button>
          </div>
        </div>
        <div class="forecast-grid" id="operations-wall-summary"></div>
      </div>
    </section>

    <section class="section business-sections">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Hospital Impact / ROI Metrics</h2>
            <div class="section-sub">Investor-ready and hospital-appropriate positioning without unsupported hard outcome claims.</div>
          </div>
          <div class="status-pill muted">Commercial Readiness</div>
        </div>
        <div class="roi-grid">
          <div class="mini-card">
            <div class="mini-k">Review Burden Reduction</div>
            <div class="mini-v" style="color:#3ad38f">71.6%</div>
            <div class="mini-copy">Reduction in unnecessary review interruptions vs standard threshold-based review rules (confirmed across 5 cohort sizes, 500–10,000 patients, April 2026). 6.2% false positive rate vs 20.4% for standard thresholds.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">False Positive Rate</div>
            <div class="mini-v" style="color:#3ad38f">6.2%</div>
            <div class="mini-copy">ERA false positive rate vs 20.4% for standard threshold-based review rules — a 14.2 point reduction in unnecessary interruptions.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">Monitoring Visibility</div>
            <div class="mini-v">Improved</div>
            <div class="mini-copy">Improves monitoring visibility across units, pilot users, and higher-priority patients in one command layer.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">No-Commitment Entry</div>
            <div class="mini-v" style="color:#3ad38f">Zero</div>
            <div class="mini-copy">Upload de-identified CSV → receive your hospital's own retrospective validation results in under 5 minutes. No EHR integration. No IT lift.</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section business-sections">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Who This Is For</h2>
            <div class="section-sub">Multi-audience platform positioning for hospital pilots, clinical operations, and stakeholder review.</div>
          </div>
        </div>
        <div class="audience-grid">
          <div class="mini-card">
            <div class="mini-k">Hospitals</div>
            <div class="mini-copy">Command-center visibility for monitored patients, response workflows, and unit-level patient prioritization.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">Clinical Leaders</div>
            <div class="mini-copy">Operational oversight for nursing, physician, and administrative response coordination.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">RPM Programs</div>
            <div class="mini-copy">Home monitoring visibility with supportive AI insights and escalation-aware workflow tracking.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">Investors / Partners</div>
            <div class="mini-copy">A polished demonstration of the platform’s product direction, market relevance, and pilot readiness.</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section business-sections">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Clinical Review Scenario</h2>
            <div class="section-sub">Demonstrates the product story: explainable review basis, supportive visibility, and structured workflow action.</div>
          </div>
        </div>
        <div class="scenario-grid" id="scenario-grid"></div>
      </div>
    </section>

    <section class="section business-sections">
      <div class="section-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">How It Works</h2>
            <div class="section-sub">Simple product explanation for hospital and investor conversations.</div>
          </div>
        </div>
        <div class="how-grid">
          <div class="mini-card">
            <div class="mini-k">1. Ingest</div>
            <div class="mini-copy">Patient vitals and monitoring signals are organized into a structured command workflow.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">2. Surface</div>
            <div class="mini-copy">Platform logic surfaces rising review priority, signal drift, and monitored patterns needing attention.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">3. Prioritize</div>
            <div class="mini-copy">Higher-priority patients rise to the top with explainable reasons, trend summaries, and suggested workflow notes.</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">4. Coordinate</div>
            <div class="mini-copy">Care teams can acknowledge, assign, escalate, resolve, and review the audit trail in one place.</div>
          </div>
        </div>
      </div>
    </section>

    <div class="footer">
      Early Risk Alert AI · Explainable Rules-Based Command-Center Platform · Unit-Based Pilot Access · Hospital Branded Accounts · Explainable Review Workflow · Persistent Audit Visibility · Pilot / Evaluation Environment
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

    <div class="drawer-block">
      <div class="drawer-k">Monitored Context</div>
      <div class="drawer-v">Values displayed as monitored context for authorized HCP review.</div>
    </div>

    <div class="drawer-grid">
      <div class="drawer-block">
        <div class="drawer-k">Heart Rate</div>
        <div class="drawer-v" id="drawerHr">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Oxygen Saturation</div>
        <div class="drawer-v" id="drawerSpo2">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Blood Pressure</div>
        <div class="drawer-v" id="drawerBp">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Respiratory Rate (RR)</div>
        <div class="drawer-v" id="drawerRr">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Temperature</div>
        <div class="drawer-v" id="drawerTemp">--</div>
      </div>
      <div class="drawer-block">
        <div class="drawer-k">Review Score</div>
        <div class="drawer-v" id="drawerRisk">--</div>
      </div>
    </div>


    <!-- ERA_DRAWER_PRIORITY_CONTEXT_V2_START -->
    <div class="drawer-block">
      <div class="drawer-k">Priority Context</div>
      <div class="era-priority-context-grid">
        <div class="era-priority-context-card">
          <div class="context-k">Priority Tier</div>
          <div class="context-v" id="drawerPriorityTier">--</div>
        </div>
        <div class="era-priority-context-card">
          <div class="context-k">Queue Rank</div>
          <div class="context-v" id="drawerQueueRank">--</div>
        </div>
        <div class="era-priority-context-card">
          <div class="context-k">Primary Driver</div>
          <div class="context-v" id="drawerPrimaryDriver">--</div>
        </div>
        <div class="era-priority-context-card">
          <div class="context-k">Trend Direction</div>
          <div class="context-v" id="drawerTrendDirection">--</div>
        </div>
      </div>
      <div class="era-help-note">
        How to read this: ERA displays a review-priority tier, queue rank, primary signal context, and trend direction so high-score patients can be distinguished without relying on raw score alone.
      </div>
    </div>
    <!-- ERA_DRAWER_PRIORITY_CONTEXT_V2_END -->

    <div class="drawer-block">
      <div class="drawer-k">Supportive Review Note</div>
      <div class="drawer-v" id="drawerAction">--</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Explainable Review Basis</div>
      <div class="drawer-v" id="drawerExplainability">--</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Workflow Status</div>
      <div class="drawer-v" id="drawerWorkflow">No workflow action saved yet.</div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Trend History</div>
      <div id="drawerTrendSparkline"></div>
      <div class="timeline-panel" id="drawerTrendTable" style="margin-top:12px;"></div>
    </div>

    <div class="drawer-block">
      <div class="drawer-k">Timeline Snapshot</div>
      <div class="timeline-panel" id="drawerTimeline"></div>
    </div>
  </aside>

  <div class="terms-modal" id="policyModal">
    <div class="modal-card">
      <div class="drawer-top">
        <div>
          <h3 class="modal-title" id="policyModalTitle">Policy</h3>
          <div class="drawer-sub">Pilot environment information</div>
        </div>
        <button class="btn secondary small" type="button" onclick="closePolicyModal()">Close</button>
      </div>
      <div class="modal-copy" id="policyModalBody"></div>
    </div>
  </div>

  <script>
    let activePatients = [];
    let activeAlerts = [];
    let workflowState = {};
    let workflowAuditLog = [];
    let persistentAuditLog = [];
    let auditLog = [];
    let currentUnitFilter = "all";
    let selectedPatientId = null;

    let accessRole = "viewer";
    let accessAssignedUnit = "all";
    let canViewAllUnits = true;
    let accessHospital = "Pilot";
    let brandName = "Early Risk Alert AI";
    let brandTagline = "Explainable Rules-Based Command-Center Platform";
    let brand1 = "#7aa2ff";
    let brand2 = "#5bd4ff";
    let pilotModeEnabled = true;
    let lastSystemHealth = {};
    let lastUpdatedAt = null;
    let wallModeEnabled = false;
    let thresholdStore = {};
    let trendCache = {};
    let trendsEndpointAvailable = false;
    let thresholdsEndpointAvailable = false;
    let pilotReadiness = {};

    const DEFAULT_THRESHOLDS = {
      icu: {spo2_low:92, hr_high:120, sbp_high:160},
      telemetry: {spo2_low:93, hr_high:110, sbp_high:150},
      stepdown: {spo2_low:93, hr_high:115, sbp_high:155},
      ward: {spo2_low:94, hr_high:110, sbp_high:150},
      rpm: {spo2_low:94, hr_high:105, sbp_high:145},
      all: {spo2_low:93, hr_high:112, sbp_high:152}
    };

    const POLICY_CONTENT = {
      terms: {
        title: "Terms",
        body: `
          This pilot environment is presented for product demonstration, workflow evaluation, and stakeholder review.
          Features, visualizations, and supportive review notes shown here represent a controlled command-center experience.
          Use of this environment does not establish a clinical system of record, emergency response capability, or replacement for hospital protocols.
        `
      },
      privacy: {
        title: "Privacy",
        body: `
          This pilot interface demonstrates how monitored patient data, risk signals, and workflow actions may be organized in a secure clinical operations environment.
          Displayed content in this environment is part of a controlled demonstration workflow and should be handled as evaluation material.
          Production privacy, security, and deployment requirements should be validated independently within hospital and regulatory standards.
        `
      },
      disclaimer: {
        title: "Pilot Disclaimer",
        body: `
          This platform is presented in a controlled pilot environment for demonstration, workflow evaluation, and stakeholder review.
          It is not intended to replace clinical judgment, hospital protocols, or emergency response systems.
          All insights, alerts, and review signals are designed to support — not replace — clinical decision-making.
        `
      }
    };

    function safe(v, fallback="--"){
      return v === undefined || v === null || v === "" ? fallback : v;
    }

    function safeNumber(v, fallback=0){
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    }

    function hasValue(v){
      return !(v === undefined || v === null || v === "");
    }

    function formatHeartRate(v){
      return hasValue(v) ? `${safeNumber(v)} bpm` : "--";
    }

    function formatSpO2(v){
      return hasValue(v) ? `${safeNumber(v)}%` : "--";
    }

    function formatBloodPressure(sys, dia){
      return hasValue(sys) && hasValue(dia) ? `${safeNumber(sys)}/${safeNumber(dia)} mmHg` : "--";
    }

    function formatRespiratoryRate(v, withRef=false){
      if (!hasValue(v)) return "--";
      const base = `${safeNumber(v)} breaths/min`;
      return withRef ? `${base} (Ref: 12–20)` : base;
    }

    function formatTemperature(v, withRef=false){
      if (!hasValue(v)) return "--";
      const num = safeNumber(v);
      const base = `${num.toFixed(1)} °F`;
      return withRef ? `${base} (Ref: 97.6–100.4)` : base;
    }

    function clamp(n, low, high){
      return Math.max(low, Math.min(high, n));
    }

    function formatTime(value){
      try{
        return new Date(value).toLocaleTimeString([], {hour:'numeric', minute:'2-digit'});
      }catch(err){
        return "--";
      }
    }

    function formatDateTime(value){
      try{
        return new Date(value).toLocaleString([], {month:'short', day:'numeric', hour:'numeric', minute:'2-digit'});
      }catch(err){
        return "--";
      }
    }

    function formatPercent(value){
      return Math.round(clamp(Number(value) || 0, 0, 100)) + "%";
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
      if (s.includes("critical") || s.includes("degraded") || s.includes("missing") || s.includes("failed")) return "critical";
      if (s.includes("pending") || s.includes("review") || s.includes("watch") || s.includes("moderate") || s.includes("acknowledged") || s.includes("assigned") || s.includes("documented") || s.includes("structured")) return "watch";
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
      if (s === "degraded") return "Degraded";
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

    function thresholdsForUnit(unit){
      const key = String(unit || "all").toLowerCase();
      return thresholdStore[key] || thresholdStore.all || DEFAULT_THRESHOLDS[key] || DEFAULT_THRESHOLDS.all;
    }

    function editableThresholdUnit(){
      if (currentUnitFilter !== "all") return currentUnitFilter;
      if (accessAssignedUnit !== "all") return accessAssignedUnit;
      return null;
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
      const notes = [];
      const spo2 = safeNumber(patient.spo2);
      const hr = safeNumber(patient.heart_rate);
      const sbp = safeNumber(patient.bp_systolic);
      const rr = safeNumber(patient.respiratory_rate);
      const temp = safeNumber(patient.temperature_f);
      const t = thresholdsForUnit(patient.unit || currentUnitFilter);

      if (spo2 && spo2 < safeNumber(t.spo2_low, 93)) notes.push("Oxygen saturation is below the current review threshold");
      else if (spo2 && spo2 < safeNumber(t.spo2_low, 93) + 2) notes.push("Oxygen saturation is trending below the current target range");

      if (hr && hr >= safeNumber(t.hr_high, 110)) notes.push("Heart rate is elevated relative to the current review threshold");
      else if (hr && hr >= safeNumber(t.hr_high, 110) - 5) notes.push("Heart rate trend is rising");

      if (sbp && sbp >= safeNumber(t.sbp_high, 150)) notes.push("Blood pressure is elevated relative to the current review threshold");

      if (rr){
        if (rr < 12 || rr > 20) notes.push("Respiratory rate is outside typical reference range — for HCP review");
        else notes.push("Respiratory rate trend is available for authorized HCP review");
      }

      if (temp){
        if (temp < 97.6 || temp > 100.4) notes.push("Temperature is outside typical reference range — for HCP review");
        else notes.push("Temperature trend is available for authorized HCP review");
      }

      if (!notes.length) notes.push("Combined monitored context remains available for authorized HCP review");
      return "Includes monitored vital context such as heart rate, oxygen saturation, respiratory rate, and temperature, along with trend behavior and workflow signals. " + notes.join("; ") + ".";
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
        title: risk.alert_message || raw.title || "Review Monitor",
        heart_rate: raw.heart_rate ?? vitals.heart_rate ?? "--",
        spo2: raw.spo2 ?? vitals.spo2 ?? "--",
        bp_systolic: raw.bp_systolic ?? vitals.systolic_bp ?? "--",
        bp_diastolic: raw.bp_diastolic ?? vitals.diastolic_bp ?? "--",
        respiratory_rate: raw.respiratory_rate ?? raw.rr ?? vitals.respiratory_rate ?? vitals.rr ?? "--",
        temperature_f: raw.temperature_f ?? raw.temp_f ?? raw.temperature ?? vitals.temperature_f ?? vitals.temp_f ?? vitals.temperature ?? "--",
        risk_score: raw.risk_score ?? risk.risk_score ?? "--",
        status: raw.status || risk.severity || "Stable",
        story: raw.story || risk.recommended_action || "Supportive review monitoring active.",
        alert_message: risk.alert_message || "Vitals stable",
        recommended_action: risk.recommended_action || "Continue routine monitoring.",
        reasons: risk.reasons || []
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

    function normalizeAuditEntry(entry){
      if (!entry) return null;
      return {
        time: entry.time || entry.timestamp || entry.created_at || null,
        patient_id: entry.patient_id || "--",
        action: entry.action || "--",
        role: entry.role || entry.user || "--",
        note: entry.note || entry.unit || "",
        user: entry.user || entry.role || "--"
      };
    }

    function mergeAuditLogs(){
      const map = new Map();
      [...workflowAuditLog, ...persistentAuditLog].forEach(item => {
        const row = normalizeAuditEntry(item);
        if (!row) return;
        const key = [row.time, row.patient_id, row.action, row.role].join("|");
        if (!map.has(key)) map.set(key, row);
      });
      auditLog = Array.from(map.values()).sort((a, b) => {
        return new Date(b.time || 0).getTime() - new Date(a.time || 0).getTime();
      });
    }

    function filterPatientsByUnit(patients){
      const source = (patients || []).map(normalizePatient).filter(Boolean);
      if (currentUnitFilter === "all") return source;
      return source.filter(p => p.unit === currentUnitFilter);
    }

    function filterAlertsByUnit(alerts){
      const source = (alerts || []).map(normalizeAlert).filter(Boolean);
      if (currentUnitFilter === "all") return source;
      return source.filter(a => roomToUnit(a.unit) === currentUnitFilter);
    }

    function findPatient(patientId){
      return activePatients.find(p => String(p.patient_id) === String(patientId)) || null;
    }

    function getTopRiskPatient(){
      return filterPatientsByUnit(activePatients)
        .slice()
        .sort((a,b) => safeNumber(b.risk_score) - safeNumber(a.risk_score))[0] || null;
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
      if (record.updated_at) parts.push(formatTime(record.updated_at));
      return parts.join(" · ");
    }

    function patientRiskPercent(patient){
      return clamp(Math.round((safeNumber(patient.risk_score) / 9) * 100), 0, 99);
    }

    function fallbackPatientTrendSeries(patient, metric){
      const riskPct = patientRiskPercent(patient);
      const spo2 = safeNumber(patient.spo2, 96);
      const hr = safeNumber(patient.heart_rate, 84);
      const rr = safeNumber(patient.respiratory_rate, 18);
      const temp = safeNumber(patient.temperature_f, 98.6);
      const status = String(patient.status || "").toLowerCase();

      if (metric === "spo2"){
        if (status === "critical") return [spo2 + 5, spo2 + 3, spo2 + 2, spo2 + 1, spo2, spo2 - 1].map(v => clamp(v, 82, 100));
        if (status === "high") return [spo2 + 3, spo2 + 2, spo2 + 1, spo2, spo2, spo2 - 1].map(v => clamp(v, 88, 100));
        return [spo2 - 1, spo2, spo2, spo2 + 1, spo2, spo2].map(v => clamp(v, 90, 100));
      }

      if (metric === "risk"){
        const base = clamp(riskPct - 18, 6, 95);
        return [base - 12, base - 7, base - 3, base + 4, riskPct - 3, riskPct].map(v => clamp(v, 0, 100));
      }

      if (metric === "hr"){
        if (status === "critical") return [hr - 12, hr - 6, hr - 4, hr + 1, hr + 4, hr].map(v => clamp(v, 60, 150));
        if (status === "high") return [hr - 8, hr - 5, hr - 2, hr + 1, hr + 2, hr].map(v => clamp(v, 60, 140));
        return [hr - 3, hr - 2, hr - 1, hr, hr, hr].map(v => clamp(v, 55, 130));
      }

      if (metric === "rr"){
        if (status === "critical") return [rr - 4, rr - 2, rr - 1, rr + 1, rr + 2, rr].map(v => clamp(v, 10, 34));
        if (status === "high") return [rr - 3, rr - 2, rr - 1, rr, rr + 1, rr].map(v => clamp(v, 10, 30));
        return [rr - 1, rr - 1, rr, rr, rr + 1, rr].map(v => clamp(v, 10, 28));
      }

      if (metric === "temp"){
        const series = status === "critical"
          ? [temp - 0.6, temp - 0.3, temp - 0.1, temp + 0.1, temp + 0.2, temp]
          : status === "high"
            ? [temp - 0.4, temp - 0.2, temp, temp, temp + 0.1, temp]
            : [temp - 0.1, temp, temp, temp + 0.1, temp, temp];
        return series.map(v => Number(Math.max(96, Math.min(103, v)).toFixed(1)));
      }

      return [0,0,0,0,0,0];
    }

    function sparklineSvg(series, colorClass, large=false){
      const values = (series || []).map(v => safeNumber(v));
      if (!values.length){
        return `<div class="note-box">Trend history not available.</div>`;
      }

      const width = 300;
      const height = large ? 92 : 56;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min || 1;
      const pts = values.map((v, i) => {
        const x = (i / Math.max(values.length - 1, 1)) * width;
        const y = height - (((v - min) / range) * (height - 10)) - 5;
        return [x, y];
      });
      const line = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
      const area = line + " L " + width + "," + height + " L 0," + height + " Z";

      const stroke =
        colorClass === "critical" ? "#ff7c8d" :
        colorClass === "watch" ? "#ffd96c" : "#77ffb4";

      return `
        <div class="sparkline ${large ? 'large' : ''}">
          <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
            <path class="area" d="${area}" fill="${stroke}"></path>
            <path class="line" d="${line}" stroke="${stroke}"></path>
          </svg>
        </div>
      `;
    }

    function applyBranding(data){
      brandName = safe(data.brand_name, "Early Risk Alert AI");
      brandTagline = safe(data.brand_tagline, "Explainable Rules-Based Command-Center Platform");
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
      if (heroTitle) heroTitle.textContent = brandName + " — Explainable Rules-Based Command-Center Platform";
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
        const unitLockPill = document.getElementById("unitLockPill");

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
        if (unitLockPill) {
          unitLockPill.textContent = canViewAllUnits ? "Hospital Scope" : "Unit Locked: " + unitLabel(accessAssignedUnit);
        }
      }catch(err){
        console.error("access context failed", err);
      }
    }

    async function loadWorkflowState(){
      try{
        const res = await fetch("/api/workflow", {cache:"no-store"});
        const payload = await res.json();
        workflowState = payload.records || {};
        workflowAuditLog = payload.audit_log || [];
        mergeAuditLogs();
      }catch(err){
        console.error("workflow load failed", err);
      }
    }

    async function loadPersistentAudit(){
      try{
        const res = await fetch("/api/audit", {cache:"no-store"});
        if (!res.ok) throw new Error("audit unavailable");
        const payload = await res.json();
        persistentAuditLog = Array.isArray(payload) ? payload : [];
      }catch(err){
        persistentAuditLog = [];
      }
      mergeAuditLogs();
    }

    async function loadThresholds(){
      thresholdStore = JSON.parse(JSON.stringify(DEFAULT_THRESHOLDS));
      try{
        const res = await fetch("/api/thresholds", {cache:"no-store"});
        if (!res.ok) throw new Error("threshold route unavailable");
        const payload = await res.json();
        thresholdsEndpointAvailable = true;
        Object.keys(payload || {}).forEach(unit => {
          thresholdStore[unit] = Object.assign({}, thresholdStore[unit] || {}, payload[unit] || {});
        });
      }catch(err){
        thresholdsEndpointAvailable = false;
      }
    }

    async function saveThresholds(){
      if (accessRole !== "admin" || !thresholdsEndpointAvailable) return;

      const unit = editableThresholdUnit();
      if (!unit){
        alert("Select a single unit to edit thresholds.");
        return;
      }

      const spo2_low = safeNumber(document.getElementById("thr-spo2-low")?.value, thresholdsForUnit(unit).spo2_low);
      const hr_high = safeNumber(document.getElementById("thr-hr-high")?.value, thresholdsForUnit(unit).hr_high);
      const sbp_high = safeNumber(document.getElementById("thr-sbp-high")?.value, thresholdsForUnit(unit).sbp_high);

      const body = {};
      body[unit] = {spo2_low, hr_high, sbp_high};

      try{
        const res = await fetch("/api/thresholds", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify(body)
        });
        const payload = await res.json();
        if (!res.ok || !payload.ok){
          alert(payload.error || "Unable to save thresholds.");
          return;
        }
        await loadThresholds();
        renderThresholdsModule();
      }catch(err){
        console.error("threshold save failed", err);
        alert("Unable to save thresholds.");
      }
    }

    async function loadSystemHealth(){
      try{
        const res = await fetch("/api/system-health", {cache:"no-store"});
        lastSystemHealth = await res.json();
      }catch(err){
        console.error("system health failed", err);
        lastSystemHealth = {status:"degraded"};
      }
    }

    async function loadPilotReadiness(){
      try{
        const res = await fetch("/api/pilot-readiness", {cache:"no-store"});
        if (!res.ok) throw new Error("pilot readiness route unavailable");
        pilotReadiness = await res.json();
      }catch(err){
        console.error("pilot readiness failed", err);
        pilotReadiness = {};
      }
      renderReadinessEvidence();
      renderRouteStatusModule();
      renderPilotPacketModule();
    }

    async function refreshSnapshot(){
      try{
        const res = await fetch("/api/v1/live-snapshot?refresh=" + Date.now(), {cache:"no-store"});
        const payload = await res.json();
        activePatients = (payload.patients || []).map(normalizePatient).filter(Boolean);
        activeAlerts = (payload.alerts || []).map(normalizeAlert).filter(Boolean);
        lastUpdatedAt = payload.generated_at || new Date().toISOString();
        rerenderAll();
        primeTrendCache();
      }catch(err){
        console.error("snapshot failed", err);
      }
    }

    async function loadPatientTrend(patientId){
      if (!patientId) return [];
      try{
        const res = await fetch("/api/trends/" + encodeURIComponent(patientId), {cache:"no-store"});
        if (!res.ok) throw new Error("trend route unavailable");
        const payload = await res.json();
        trendsEndpointAvailable = true;
        trendCache[patientId] = Array.isArray(payload) ? payload : [];
      }catch(err){
        trendsEndpointAvailable = false;
        if (!trendCache[patientId]) trendCache[patientId] = [];
      }
      return trendCache[patientId] || [];
    }

    async function primeTrendCache(){
      const visible = filterPatientsByUnit(activePatients)
        .slice()
        .sort((a,b) => safeNumber(b.risk_score) - safeNumber(a.risk_score))
        .slice(0, 4);

      await Promise.all(visible.map(p => loadPatientTrend(p.patient_id)));
      renderModules();
      if (selectedPatientId){
        const current = selectedPatientId;
        const drawerOpen = document.getElementById("patientDrawer").classList.contains("open");
        if (drawerOpen) openPatientDrawer(current);
      }
    }

    function getPatientTrendSeries(patient, metric){
      const records = trendCache[patient.patient_id] || [];
      if (records.length > 1){
        if (metric === "spo2") return records.map(r => safeNumber(r.spo2));
        if (metric === "risk") return records.map(r => clamp(Math.round((safeNumber(r.risk) / 9) * 100), 0, 99));
        if (metric === "hr") return records.map(r => safeNumber(r.hr));
        if (metric === "rr") return records.map(r => safeNumber(r.rr));
        if (metric === "temp") return records.map(r => safeNumber(r.temp_f));
      }
      return fallbackPatientTrendSeries(patient, metric);
    }

    function buildPatientHistoryTimeline(patient){
      const records = (trendCache[patient.patient_id] || []).slice(-5).reverse();
      if (!records.length){
        return `
          <div class="timeline-item">
            <div class="timeline-time">Live</div>
            <div class="timeline-copy">Stored trend history is not yet available. Snapshot-based monitoring is active for authorized HCP review.</div>
          </div>
        `;
      }
      return records.map(r => `
        <div class="timeline-item">
          <div class="timeline-time">${formatTime(r.time)}</div>
          <div class="timeline-copy">HR ${formatHeartRate(r.hr)} · SpO₂ ${formatSpO2(r.spo2)} · RR ${formatRespiratoryRate(r.rr)} · Temp ${formatTemperature(r.temp_f)} · Review score ${clamp(Math.round((safeNumber(r.risk) / 9) * 100), 0, 99)}%</div>
        </div>
      `).join("");
    }

    function buildDrawerTrendBlocks(patient){
      const hrSeries = getPatientTrendSeries(patient, "hr");
      const rrSeries = getPatientTrendSeries(patient, "rr");
      const tempSeries = getPatientTrendSeries(patient, "temp");
      const klass = statusClass(patient.status);
      return `
        <div class="drawer-k" style="margin-top:4px;">HR Trend</div>
        ${sparklineSvg(hrSeries, klass, true)}
        <div class="drawer-k" style="margin-top:12px;">RR Trend</div>
        ${sparklineSvg(rrSeries, klass)}
        <div class="drawer-k" style="margin-top:12px;">Temperature Trend</div>
        ${sparklineSvg(tempSeries, klass)}
      `;
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
        await loadPersistentAudit();
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

    async function openPatientDrawer(patientId){
      const patient = findPatient(patientId);
      if (!patient) return;
      selectedPatientId = patientId;

      await loadPatientTrend(patientId);

      document.getElementById("drawerPatientName").textContent = safe(patient.name);
      document.getElementById("drawerPatientSub").textContent = `${safe(patient.patient_id)} · ${safe(patient.bed)} · ${safe(patient.program)}`;
      document.getElementById("drawerSummary").textContent = safe(patient.alert_message);
      document.getElementById("drawerHr").textContent = formatHeartRate(patient.heart_rate);
      document.getElementById("drawerSpo2").textContent = formatSpO2(patient.spo2);
      document.getElementById("drawerBp").textContent = formatBloodPressure(patient.bp_systolic, patient.bp_diastolic);
      document.getElementById("drawerRr").textContent = formatRespiratoryRate(patient.respiratory_rate, true);
      document.getElementById("drawerTemp").textContent = formatTemperature(patient.temperature_f, true);
      document.getElementById("drawerRisk").textContent = patientRiskPercent(patient) + "% (" + (typeof patient.risk_score === "number" ? patient.risk_score.toFixed(1) : safe(patient.risk_score)) + ")";
      document.getElementById("drawerAction").textContent = safe(patient.recommended_action);
      document.getElementById("drawerExplainability").textContent = explainabilityForPatient(patient);
      document.getElementById("drawerWorkflow").textContent = workflowText(patientId);

      document.getElementById("drawerTrendSparkline").innerHTML = buildDrawerTrendBlocks(patient);
      document.getElementById("drawerTrendTable").innerHTML = buildPatientHistoryTimeline(patient);

      const records = (trendCache[patient.patient_id] || []).slice(-3).reverse();
      const timelineHtml = records.length ? records.map(r => `
        <div class="timeline-item">
          <div class="timeline-time">${formatTime(r.time)}</div>
          <div class="timeline-copy">SpO₂ ${formatSpO2(r.spo2)} · HR ${formatHeartRate(r.hr)} · RR ${formatRespiratoryRate(r.rr)} · Temp ${formatTemperature(r.temp_f)} · Review score ${clamp(Math.round((safeNumber(r.risk) / 9) * 100), 0, 99)}%</div>
        </div>
      `).join("") : `
        <div class="timeline-item">
          <div class="timeline-time">Now</div>
          <div class="timeline-copy">${safe(patient.name)} is currently ${safe(patient.status)} with displayed review score ${patientRiskPercent(patient)}%. Values displayed as monitored context for authorized HCP review.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">Trend</div>
          <div class="timeline-copy">Stored history is not yet available, so the current snapshot is driving the display.</div>
        </div>
      `;

      document.getElementById("drawerTimeline").innerHTML = timelineHtml + `
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

    function openPolicyModal(kind){
      const item = POLICY_CONTENT[kind] || POLICY_CONTENT.disclaimer;
      document.getElementById("policyModalTitle").textContent = item.title;
      document.getElementById("policyModalBody").innerHTML = item.body.trim().replace(/\n/g, "<br><br>");
      document.getElementById("policyModal").classList.add("show");
    }

    function closePolicyModal(){
      document.getElementById("policyModal").classList.remove("show");
    }

    function toggleWallMode(){
      wallModeEnabled = !wallModeEnabled;
      document.body.classList.toggle("wall-mode", wallModeEnabled);
      const btn = document.getElementById("wallModeBtn");
      const pill = document.getElementById("wallModePill");
      const status = document.getElementById("wallModeStatusPill");
      if (btn){
        btn.classList.toggle("active", wallModeEnabled);
        btn.textContent = wallModeEnabled ? "Disable Wall Mode" : "Enable Wall Mode";
      }
      if (pill) pill.textContent = wallModeEnabled ? "Wall Mode On" : "Wall Mode Off";
      if (status) status.textContent = wallModeEnabled ? "Wall View" : "Standard View";
    }

    function renderActionButtons(record, pid){
      const ackDisabled = accessRole === "viewer" ? "disabled" : "";
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
      const riskText = patientRiskPercent(patient) + "%";
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
            <div class="metric-box"><span class="metric-k">HR</span><span class="metric-v">${formatHeartRate(patient.heart_rate)}</span></div>
            <div class="metric-box"><span class="metric-k">SpO₂</span><span class="metric-v">${formatSpO2(patient.spo2)}</span></div>
            <div class="metric-box"><span class="metric-k">BP</span><span class="metric-v">${formatBloodPressure(patient.bp_systolic, patient.bp_diastolic)}</span></div>
            <div class="metric-box"><span class="metric-k">RR</span><span class="metric-v">${formatRespiratoryRate(patient.respiratory_rate)}</span></div>
            <div class="metric-box"><span class="metric-k">Temp</span><span class="metric-v">${formatTemperature(patient.temperature_f)}</span></div>
            <div class="metric-box"><span class="metric-k">Risk</span><span class="metric-v">${riskText}</span></div>
          </div>

          <div class="monitor-story">
            <div class="story-text">${safe(patient.story)}<br><span class="alert-sub">Values displayed as monitored context for authorized HCP review.</span></div>
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
      const source = applySearchFilter(filterPatientsByUnit(activePatients))
        .sort((a,b) => safeNumber(b.risk_score) - safeNumber(a.risk_score))
        .slice(0,8);
      const noMsg = patientSearchTerm
        ? `<div class="monitor"><div class="monitor-title">No patients match "${patientSearchTerm}".</div></div>`
        : `<div class="monitor"><div class="monitor-title">No patients visible in current access scope.</div></div>`;
      document.getElementById("wall").innerHTML = source.map(renderMonitor).join("") || noMsg;
    }

    function renderAlertsList(){
      const source = filterAlertsByUnit(activeAlerts).slice(0,6);
      document.getElementById("queue").innerHTML = source.map(renderAlert).join("") || `
        <div class="alert-item"><div><div class="alert-copy">No active alerts in current scope.</div></div><div class="status-pill live">Clear</div></div>
      `;
    }

    function renderTopRiskPatients(){
      const source = filterPatientsByUnit(activePatients)
        .sort((a,b) => safeNumber(b.risk_score) - safeNumber(a.risk_score))
        .slice(0,4);

      document.getElementById("top-risk-list").innerHTML = source.map(p => `
        <div class="queue-item">
          <div class="queue-copy">${safe(p.name)} · ${safe(p.patient_id)} · Review score ${patientRiskPercent(p)}%</div>
          <div class="status-pill ${statusClass(p.status)}">${pulseLabel(p.status)}</div>
        </div>
      `).join("") || `<div class="queue-item"><div class="queue-copy">No risk queue visible.</div><div class="status-pill muted">Scope</div></div>`;
    }

    function renderAIReasoning(){
      const top = getTopRiskPatient();
      const panel = document.getElementById("ai-reasoning-panel");

      if (!top){
        panel.innerHTML = `<div class="why-line">Waiting for patient intelligence stream...</div>`;
        return;
      }

      panel.innerHTML = `
        <div class="why-line">Highest review-priority patient: ${safe(top.name)} (${safe(top.patient_id)})</div>
        <div class="why-line">Current review priority: ${safe(top.status)} · Review score ${patientRiskPercent(top)}%</div>
        <div class="why-line">${safe(top.story)}</div>
        <div class="why-line">Explainable reason: ${explainabilityForPatient(top)}</div>
      `;
    }

    function renderAIInsight(){
      const top = getTopRiskPatient();
      const panel = document.getElementById("ai-insight-panel");

      if (!top){
        panel.innerHTML = `<div class="info-line">Waiting for active review insight...</div>`;
        return;
      }

      const spo2 = safeNumber(top.spo2, 96);
      const hr = safeNumber(top.heart_rate, 82);
      const rr = safeNumber(top.respiratory_rate, 18);
      const temp = safeNumber(top.temperature_f, 98.6);
      const reviewAttentionLevel = safeNumber(top.risk_score) >= 7 ? "elevated" : "active";

      panel.innerHTML = `
        <div class="info-line"><strong>Review Insight:</strong> Oxygen saturation is ${spo2 < 94 ? "trending downward" : "being closely monitored"} with ${hr >= 110 ? "rising heart-rate pressure" : "emerging physiologic drift"}.</div>
        <div class="info-line">Monitored context also includes RR ${formatRespiratoryRate(rr, true)} and Temperature ${formatTemperature(temp, true)} for authorized HCP review.</div>
        <div class="info-line">Suggested workflow note: Current review attention is ${reviewAttentionLevel}. ${safe(top.recommended_action)}</div>
      `;
    }

    function renderWorkflow(){
      const records = Object.values(workflowState || {});
      const currentAlerts = filterAlertsByUnit(activeAlerts);
      document.getElementById("wf-new").textContent = String(currentAlerts.length);
      document.getElementById("wf-ack").textContent = String(records.filter(r => r.ack).length);
      document.getElementById("wf-assigned").textContent = String(records.filter(r => r.assigned).length);
      document.getElementById("wf-escalated").textContent = String(records.filter(r => r.escalated).length);
      document.getElementById("wf-resolved").textContent = String(records.filter(r => r.resolved).length);
    }

    function renderSystemHealth(){
      const status = String(lastSystemHealth.status || "unknown").toLowerCase();
      const healthy = status === "ok" || status === "connected";
      const list = document.getElementById("system-health-list");
      list.innerHTML = `
        <div class="queue-item"><div class="queue-copy">Hospital</div><div class="status-pill info">${safe(accessHospital)}</div></div>
        <div class="queue-item"><div class="queue-copy">Role</div><div class="status-pill info">${safe(accessRole)}</div></div>
        <div class="queue-item"><div class="queue-copy">Assigned Unit</div><div class="status-pill watch">${unitLabel(accessAssignedUnit)}</div></div>
        <div class="queue-item"><div class="queue-copy">System Status</div><div class="status-pill ${healthy ? 'live' : 'critical'}">${healthy ? 'Connected' : safe(status, 'unknown')}</div></div>
        <div class="queue-item"><div class="queue-copy">Patients in Feed</div><div class="status-pill muted">${safe(lastSystemHealth.patient_count, activePatients.length)}</div></div>
        <div class="queue-item"><div class="queue-copy">Higher-Priority Review Items</div><div class="status-pill critical">${safe(lastSystemHealth.critical_alerts, 0)}</div></div>
      `;
    }

    function renderSecurityControls(){
      const list = document.getElementById("security-controls-list");
      const unitState = canViewAllUnits ? "Hospital Scope" : "Locked to " + unitLabel(accessAssignedUnit);
      const thresholdMode = thresholdsEndpointAvailable ? (accessRole === "admin" ? "Admin Editable" : "Read Only") : "Fallback Mode";
      const trendMode = trendsEndpointAvailable ? "Stored History" : "Snapshot Fallback";
      const routeStatus = pilotReadiness.route_status || {};
      const mfaRows = pilotReadiness.mfa_access_log || [];
      const mfaRow = mfaRows[0] || {};
      const docsSummary = routeStatus.docs_checked ? `${safeNumber(routeStatus.docs_available, 0)}/${safeNumber(routeStatus.docs_checked, 0)} docs visible` : "Doc review pending";

      list.innerHTML = `
        <div class="queue-item">
          <div>
            <div class="queue-copy">Pilot Protection</div>
            <div class="alert-sub">Decision support only · not a medical device</div>
          </div>
          <div class="status-pill critical">Protected</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Unit Scope Control</div>
            <div class="alert-sub">${unitState}</div>
          </div>
          <div class="status-pill ${canViewAllUnits ? 'info' : 'locked'}">${canViewAllUnits ? 'Scoped' : 'Locked'}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Threshold Controls</div>
            <div class="alert-sub">${thresholdMode}</div>
          </div>
          <div class="status-pill ${thresholdsEndpointAvailable ? 'watch' : 'muted'}">${thresholdsEndpointAvailable ? 'Connected' : 'Fallback'}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Trend History</div>
            <div class="alert-sub">${trendMode}</div>
          </div>
          <div class="status-pill ${trendsEndpointAvailable ? 'live' : 'muted'}">${trendsEndpointAvailable ? 'History' : 'Snapshot'}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">MFA &amp; Admin Access</div>
            <div class="alert-sub">Phishing-resistant MFA implemented on all administrative systems — email, source control, hosting, DNS · April 10, 2026</div>
          </div>
          <div class="status-pill live">Implemented</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">CITI Research Ethics Training</div>
            <div class="alert-sub">Data or Specimens Only Research · Conflict of Interest · Completed April 10, 2026</div>
          </div>
          <div class="status-pill live">Completed</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">MIMIC-IV Data Access</div>
            <div class="alert-sub">PhysioNet application submitted April 10, 2026 · Awaiting approval · Real de-identified ICU data validation pending</div>
          </div>
          <div class="status-pill watch">Pending</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Docs + Packet Status</div>
            <div class="alert-sub">${docsSummary}</div>
          </div>
          <div class="status-pill ${routeStatus.missing_routes && routeStatus.missing_routes.length ? 'critical' : 'live'}">${routeStatus.missing_routes && routeStatus.missing_routes.length ? 'Attention' : 'Ready'}</div>
        </div>
      `;
    }

    function renderReadinessEvidence(){
      const list = document.getElementById("readiness-evidence-list");
      if (!list) return;

      const routeStatus = pilotReadiness.route_status || {};
      const owners = pilotReadiness.support_owners || [];
      const backup = (pilotReadiness.backup_restore_log || [])[0] || {};
      const patch = (pilotReadiness.patch_log || [])[0] || {};
      const accessReview = (pilotReadiness.access_review_log || [])[0] || {};
      const tabletop = (pilotReadiness.tabletop_log || [])[0] || {};
      const training = (pilotReadiness.training_ack_log || [])[0] || {};
      const mfa = (pilotReadiness.mfa_access_log || [])[0] || {};
      const ownerNames = owners.slice(0, 3).map(item => safe(item.owner || item.name)).join(" · ");
      const routeCopy = routeStatus.routes_checked
        ? `${safeNumber(routeStatus.routes_available, 0)}/${safeNumber(routeStatus.routes_checked, 0)} core routes present`
        : "Route audit pending";
      const latestRelease = ((pilotReadiness.release_notes || [])[0] || {}).version || ((pilotReadiness.release_notes || []).slice(-1)[0] || {}).version || "--";
      const validationCount = (pilotReadiness.dated_validation_evidence || []).length;
      const buildState = safe(pilotReadiness.pilot_build_state, "Locked Stable Pilot Build");

      list.innerHTML = `
        <div class="queue-item">
          <div>
            <div class="queue-copy">Full Governance Packet</div>
            <div class="alert-sub">Pilot docs, claims control, change control, validation packet, cybersecurity summary, and site packet placeholders.</div>
          </div>
          <a class="btn secondary small" href="/pilot-docs" target="_blank" rel="noopener">Open</a>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Governance Build State</div>
            <div class="alert-sub">${safe(pilotReadiness.pilot_version, '--')} · ${buildState}</div>
          </div>
          <div class="status-pill info">Locked</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Intended-Use Governance</div>
            <div class="alert-sub">HCP-facing review support posture retained across pilot materials.</div>
          </div>
          <div class="status-pill live">Active</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Route Stability</div>
            <div class="alert-sub">${routeCopy}</div>
          </div>
          <div class="status-pill ${routeStatus.missing_routes && routeStatus.missing_routes.length ? 'critical' : 'live'}">${routeStatus.missing_routes && routeStatus.missing_routes.length ? 'Review' : 'Pass'}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">MFA + Access Evidence</div>
            <div class="alert-sub">Implemented · Phishing-resistant MFA on all admin systems · April 10, 2026</div>
          </div>
          <div class="status-pill ${statusClass(mfa.status)}">${safe(mfa.status, 'Review')}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Backup / Restore</div>
            <div class="alert-sub">${safe(backup.last_tested || backup.last_reviewed, 'test date needed')} · ${safe(backup.notes, 'Record restore evidence before pilot.')}</div>
          </div>
          <div class="status-pill ${statusClass(backup.status)}">${safe(backup.status, 'Pending')}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Patch + Access Review</div>
            <div class="alert-sub">${safe(patch.last_patched || patch.last_reviewed, 'patch date needed')} · Access review ${safe(accessReview.last_reviewed, 'pending')}</div>
          </div>
          <div class="status-pill ${statusClass(patch.status || accessReview.status)}">${safe(patch.status || accessReview.status, 'Review')}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Tabletop + Training</div>
            <div class="alert-sub">Tabletop ${safe(tabletop.exercise_date, 'pending')} · Training ${safe(training.ack_date, 'pending')}</div>
          </div>
          <div class="status-pill ${statusClass(tabletop.status || training.status)}">${safe(tabletop.status || training.status, 'Pending')}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Release + Validation</div>
            <div class="alert-sub">Latest release ${latestRelease} · Validation records ${validationCount}</div>
          </div>
          <div class="status-pill info">Tracked</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Named Owners</div>
            <div class="alert-sub">${ownerNames || 'Add site sponsor and clinical reviewer before pilot start.'}</div>
          </div>
          <div class="status-pill info">${owners.length || 0}</div>
        </div>
      `;
    }

function renderRouteStatusModule(){
      const target = document.getElementById("route-status-module");
      if (!target) return;
      const routeStatus = pilotReadiness.route_status || {};
      const docs = routeStatus.documents || {};
      const docEntries = Object.entries(docs);
      const missing = routeStatus.missing_routes || [];

      target.innerHTML = `
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">Routes</div>
            <div class="bar-track"><div class="bar-fill ${missing.length ? 'critical' : ''}" style="width:${routeStatus.routes_checked ? Math.round((safeNumber(routeStatus.routes_available, 0) / safeNumber(routeStatus.routes_checked, 1)) * 100) : 0}%"></div></div>
            <div class="bar-value">${safeNumber(routeStatus.routes_available, 0)}/${safeNumber(routeStatus.routes_checked, 0)}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Docs</div>
            <div class="bar-track"><div class="bar-fill ${routeStatus.docs_available < routeStatus.docs_checked ? 'warn' : ''}" style="width:${routeStatus.docs_checked ? Math.round((safeNumber(routeStatus.docs_available, 0) / safeNumber(routeStatus.docs_checked, 1)) * 100) : 0}%"></div></div>
            <div class="bar-value">${safeNumber(routeStatus.docs_available, 0)}/${safeNumber(routeStatus.docs_checked, 0)}</div>
          </div>
        </div>
        <div class="queue-list" style="margin-top:12px;">
          ${missing.length ? `<div class="queue-item"><div class="queue-copy">Missing route references</div><div class="status-pill critical">${missing.length}</div></div>` : `<div class="queue-item"><div class="queue-copy">Core command-center routes present</div><div class="status-pill live">Pass</div></div>`}
          ${docEntries.map(([name, ok]) => `<div class="queue-item"><div class="queue-copy">${name.replaceAll('_', ' ')}</div><div class="status-pill ${ok ? 'live' : 'critical'}">${ok ? 'Ready' : 'Missing'}</div></div>`).join('')}
        </div>
      `;
    }

    function renderPilotPacketModule(){
      const target = document.getElementById("pilot-packet-module");
      if (!target) return;
      const packet = pilotReadiness.site_packet_template || [];
      if (!packet.length){
        target.innerHTML = `<div class="mini-copy">Site packet template will appear when pilot-readiness data is available.</div>`;
        return;
      }

      target.innerHTML = `
        <div class="queue-list">
          ${packet.slice(0, 6).map(item => `
            <div class="queue-item">
              <div>
                <div class="queue-copy">${safe(item.section)}</div>
                <div class="alert-sub">${safe(item.summary)}</div>
              </div>
              <div class="status-pill ${statusClass(item.status)}">${safe(item.status, 'Fill')}</div>
            </div>
          `).join('')}
        </div>
      `;
    }

    function renderPatientTimeline(){
      const top = getTopRiskPatient();
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

      const records = (trendCache[top.patient_id] || []).slice(-4).reverse();
      if (records.length){
        panel.innerHTML = records.map(r => `
          <div class="timeline-item">
            <div class="timeline-time">${formatTime(r.time)}</div>
            <div class="timeline-copy">SpO₂ ${formatSpO2(r.spo2)} · HR ${formatHeartRate(r.hr)} · RR ${formatRespiratoryRate(r.rr)} · Temp ${formatTemperature(r.temp_f)} · Review score ${clamp(Math.round((safeNumber(r.risk) / 9) * 100), 0, 99)}%</div>
          </div>
        `).join("") + `
          <div class="timeline-item">
            <div class="timeline-time">Action</div>
            <div class="timeline-copy">${safe(top.recommended_action)}</div>
          </div>
        `;
        return;
      }

      panel.innerHTML = `
        <div class="timeline-item">
          <div class="timeline-time">Now</div>
          <div class="timeline-copy">${safe(top.name)} is currently in ${safe(top.status)} review priority with review score ${patientRiskPercent(top)}%.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">-15 Min</div>
          <div class="timeline-copy">Trend drift detected across oxygen saturation, respiratory rate, temperature, and heart-rate signals.</div>
        </div>
        <div class="timeline-item">
          <div class="timeline-time">-30 Min</div>
          <div class="timeline-copy">Supportive review logic elevated review attention before standard threshold-only workflow.</div>
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
            <div class="audit-sub">${safe(entry.role)} · ${safe(entry.note)} · ${formatDateTime(entry.time)}</div>
          </div>
          <div class="status-pill info">Log</div>
        </div>
      `).join("");
    }

    function renderAuditSummary(){
      const target = document.getElementById("audit-summary-list");
      const total = auditLog.length;
      const last = auditLog[0];
      const acks = auditLog.filter(a => String(a.action).toLowerCase().includes("ack")).length;
      const escalations = auditLog.filter(a => String(a.action).toLowerCase().includes("escalate")).length;

      target.innerHTML = `
        <div class="queue-item">
          <div class="queue-copy">Visible audit entries</div>
          <div class="status-pill info">${total}</div>
        </div>
        <div class="queue-item">
          <div class="queue-copy">Acknowledgments</div>
          <div class="status-pill live">${acks}</div>
        </div>
        <div class="queue-item">
          <div class="queue-copy">Escalations</div>
          <div class="status-pill critical">${escalations}</div>
        </div>
        <div class="queue-item">
          <div>
            <div class="queue-copy">Most recent activity</div>
            <div class="alert-sub">${last ? safe(last.action) + " · " + formatDateTime(last.time) : "No activity yet"}</div>
          </div>
          <div class="status-pill muted">${last ? safe(last.role) : "Idle"}</div>
        </div>
      `;
    }

    function renderReportingDashboardModule(){
      const patients = filterPatientsByUnit(activePatients);
      const alerts = filterAlertsByUnit(activeAlerts);
      const avgRisk = patients.length ? patients.reduce((n,p)=>n+patientRiskPercent(p),0)/patients.length : 0;
      const top = getTopRiskPatient();

      document.getElementById("reporting-dashboard-module").innerHTML = `
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">Patients</div>
            <div class="bar-track"><div class="bar-fill" style="width:${clamp(patients.length * 18, 8, 100)}%"></div></div>
            <div class="bar-value">${patients.length}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Alerts</div>
            <div class="bar-track"><div class="bar-fill warn" style="width:${clamp(alerts.length * 22, 8, 100)}%"></div></div>
            <div class="bar-value">${alerts.length}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Avg Risk</div>
            <div class="bar-track"><div class="bar-fill critical" style="width:${avgRisk}%"></div></div>
            <div class="bar-value">${Math.round(avgRisk)}%</div>
          </div>
        </div>
        <div class="mini-copy" style="margin-top:12px;">
          ${top ? `Top review-priority patient: ${safe(top.bed)} (${patientRiskPercent(top)}% review score).` : `No top review-priority patient in current scope.`}
        </div>
      `;
    }

    function renderThresholdsModule(){
      const pill = document.getElementById("thresholdModePill");
      const unit = editableThresholdUnit();
      const current = thresholdsForUnit(unit || currentUnitFilter);

      if (accessRole === "admin" && thresholdsEndpointAvailable && unit){
        if (pill) pill.textContent = "Admin Editable";
        document.getElementById("thresholds-module").innerHTML = `
          <div class="threshold-editor">
            <div class="note-box">Editing thresholds for <strong>${unitLabel(unit)}</strong>. Changes post to the backend threshold route.</div>
            <div class="threshold-editor-grid">
              <div class="threshold-field">
                <label>SpO₂ Low</label>
                <input id="thr-spo2-low" type="number" value="${safeNumber(current.spo2_low)}">
              </div>
              <div class="threshold-field">
                <label>HR High</label>
                <input id="thr-hr-high" type="number" value="${safeNumber(current.hr_high)}">
              </div>
              <div class="threshold-field">
                <label>SBP High</label>
                <input id="thr-sbp-high" type="number" value="${safeNumber(current.sbp_high)}">
              </div>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
              <button class="btn primary" type="button" onclick="saveThresholds()">Save Thresholds</button>
              <button class="btn secondary" type="button" onclick="renderThresholdsModule()">Reset View</button>
            </div>
          </div>
        `;
        return;
      }

      if (pill) pill.textContent = thresholdsEndpointAvailable ? "Read Only" : "Fallback";
      if (accessRole === "admin" && thresholdsEndpointAvailable && !unit){
        document.getElementById("thresholds-module").innerHTML = `
          <div class="note-box">Select a single unit from the Unit Filter to edit thresholds. All Units view is display-only.</div>
        `;
        return;
      }

      document.getElementById("thresholds-module").innerHTML = `
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">SpO₂ Low</div>
            <div class="bar-track"><div class="bar-fill warn" style="width:${safeNumber(current.spo2_low)}%"></div></div>
            <div class="bar-value">${safeNumber(current.spo2_low)}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">HR High</div>
            <div class="bar-track"><div class="bar-fill" style="width:${Math.min(safeNumber(current.hr_high)/1.4,100)}%"></div></div>
            <div class="bar-value">${safeNumber(current.hr_high)}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">SBP High</div>
            <div class="bar-track"><div class="bar-fill critical" style="width:${Math.min(safeNumber(current.sbp_high)/1.8,100)}%"></div></div>
            <div class="bar-value">${safeNumber(current.sbp_high)}</div>
          </div>
        </div>
        <div class="mini-copy" style="margin-top:12px;">
          ${thresholdsEndpointAvailable ? "Thresholds are loaded from the backend and displayed in read-only mode for this role." : "Threshold route not available, so fallback threshold defaults are being displayed."}
        </div>
      `;
    }

    function renderOxygenTrendModule(){
      const top = getTopRiskPatient();
      const pill = document.getElementById("oxygenTrendModePill");
      if (!top){
        document.getElementById("oxygen-trend-module").innerHTML = `<div class="mini-copy">No oxygen trend available in current scope.</div>`;
        return;
      }
      const series = getPatientTrendSeries(top, "spo2");
      if (pill) pill.textContent = trendsEndpointAvailable ? "Stored Trend" : "Fallback";
      document.getElementById("oxygen-trend-module").innerHTML = `
        ${sparklineSvg(series, statusClass(top.status))}
        <div class="mini-copy" style="margin-top:12px;">
          ${safe(top.name)} · Current SpO₂ ${safe(top.spo2)} · ${trendsEndpointAvailable ? "Stored trend history is active." : "Trend route unavailable, showing fallback history view."}
        </div>
      `;
    }

    function renderRiskTimelineModule(){
      const top = getTopRiskPatient();
      if (!top){
        document.getElementById("risk-timeline-module").innerHTML = `<div class="mini-copy">No risk timeline available in current scope.</div>`;
        return;
      }
      const series = getPatientTrendSeries(top, "risk");
      const history = (trendCache[top.patient_id] || []).slice(-3);
      document.getElementById("risk-timeline-module").innerHTML = `
        ${sparklineSvg(series, statusClass(top.status))}
        <div class="timeline-panel" style="margin-top:12px;">
          ${history.length ? history.map(r => `
            <div class="timeline-item">
              <div class="timeline-time">${formatTime(r.time)}</div>
              <div class="timeline-copy">Review score ${clamp(Math.round((safeNumber(r.risk) / 9) * 100), 0, 99)}% · HR ${formatHeartRate(r.hr)} · SpO₂ ${formatSpO2(r.spo2)} · RR ${formatRespiratoryRate(r.rr)} · Temp ${formatTemperature(r.temp_f)}</div>
            </div>
          `).join("") : `
            <div class="timeline-item">
              <div class="timeline-time">-30</div>
              <div class="timeline-copy">Baseline monitoring state remained within normal workflow review.</div>
            </div>
            <div class="timeline-item">
              <div class="timeline-time">-15</div>
              <div class="timeline-copy">Supportive AI logic detected visible signal drift and elevated attention level.</div>
            </div>
            <div class="timeline-item">
              <div class="timeline-time">Now</div>
              <div class="timeline-copy">Current displayed review score for ${safe(top.patient_id)} is ${patientRiskPercent(top)}%.</div>
            </div>
          `}
        </div>
      `;
    }

    function renderCapacityModule(){
      const patients = filterPatientsByUnit(activePatients);
      const alerts = filterAlertsByUnit(activeAlerts);
      const visibleUnitCount = currentUnitFilter === "all" ? 5 : 1;
      const capacity = clamp(42 + (patients.length * 8) + (alerts.length * 6) + (visibleUnitCount * 4), 18, 96);

      document.getElementById("capacity-module").innerHTML = `
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">Load</div>
            <div class="bar-track"><div class="bar-fill ${capacity >= 80 ? 'critical' : capacity >= 60 ? 'warn' : ''}" style="width:${capacity}%"></div></div>
            <div class="bar-value">${capacity}%</div>
          </div>
        </div>
        <div class="mini-copy" style="margin-top:12px;">
          Capacity view is presented as an operational estimate tied to visible patients, alerts, and access scope.
        </div>
      `;
    }

    function renderForecastModule(){
      const top = getTopRiskPatient();
      const target = document.getElementById("forecast-module");
      if (!top){
        target.innerHTML = `<div class="mini-copy">No review priority outlook available in current scope.</div>`;
        return;
      }

      const trend = getPatientTrendSeries(top, "risk");
      const latest = trend[trend.length - 1] || patientRiskPercent(top);
      const prev = trend[Math.max(trend.length - 2, 0)] || latest;
      const slope = latest - prev;
      const forecastPct = clamp(latest + Math.max(slope, 4), 0, 99);
      const reviewAttentionLevel = forecastPct >= 85 ? "High" : forecastPct >= 65 ? "Elevated" : "Moderate";

      target.innerHTML = `
        <div class="forecast-grid">
          <div class="mini-card">
            <div class="mini-k">Primary Patient</div>
            <div class="mini-v">${safe(top.patient_id)}</div>
            <div class="mini-copy">${safe(top.bed)} · ${safe(top.status)}</div>
          </div>
          <div class="mini-card">
            <div class="mini-k">Review Attention Level</div>
            <div class="mini-v">${reviewAttentionLevel}</div>
            <div class="mini-copy">Increased review attention supported by current monitored trend.</div>
          </div>
        </div>
      `;
    }

    function renderClustersModule(){
      const patients = filterPatientsByUnit(activePatients);
      let oxygen = 0, pressure = 0, hr = 0;

      patients.forEach(p => {
        const t = thresholdsForUnit(p.unit);
        if (safeNumber(p.spo2, 100) < safeNumber(t.spo2_low, 93)) oxygen += 1;
        if (safeNumber(p.bp_systolic, 0) >= safeNumber(t.sbp_high, 150)) pressure += 1;
        if (safeNumber(p.heart_rate, 0) >= safeNumber(t.hr_high, 110)) hr += 1;
      });

      document.getElementById("clusters-module").innerHTML = `
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">Oxygen</div>
            <div class="bar-track"><div class="bar-fill warn" style="width:${patients.length ? (oxygen/patients.length)*100 : 6}%"></div></div>
            <div class="bar-value">${oxygen}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Pressure</div>
            <div class="bar-track"><div class="bar-fill critical" style="width:${patients.length ? (pressure/patients.length)*100 : 6}%"></div></div>
            <div class="bar-value">${pressure}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Heart Rate</div>
            <div class="bar-track"><div class="bar-fill" style="width:${patients.length ? (hr/patients.length)*100 : 6}%"></div></div>
            <div class="bar-value">${hr}</div>
          </div>
        </div>
      `;
    }

    function renderHeatmapModule(){
      const unitStats = {
        icu: {label:"ICU", risk:0, count:0},
        telemetry: {label:"Telemetry", risk:0, count:0},
        stepdown: {label:"Stepdown", risk:0, count:0},
        ward: {label:"Ward", risk:0, count:0},
        rpm: {label:"RPM", risk:0, count:0}
      };

      activePatients.forEach(p => {
        const u = p.unit;
        if (unitStats[u]){
          unitStats[u].count += 1;
          unitStats[u].risk += patientRiskPercent(p);
        }
      });

      const rows = Object.values(unitStats).map(u => {
        const avg = u.count ? Math.round(u.risk / u.count) : 0;
        const klass = avg >= 70 ? "critical" : avg >= 45 ? "warn" : "";
        return `
          <div class="bar-row">
            <div class="bar-label">${u.label}</div>
            <div class="bar-track"><div class="bar-fill ${klass}" style="width:${Math.max(avg,4)}%"></div></div>
            <div class="bar-value">${avg}%</div>
          </div>
        `;
      }).join("");

      document.getElementById("heatmap-module").innerHTML = `<div class="bars">${rows}</div>`;
    }

    function renderWorkflowReadinessModule(){
      const records = Object.values(workflowState || {});
      const total = Math.max(records.length, 1);
      const ack = Math.round((records.filter(r => r.ack).length / total) * 100);
      const assigned = Math.round((records.filter(r => r.assigned).length / total) * 100);
      const escalated = Math.round((records.filter(r => r.escalated).length / total) * 100);
      const resolved = Math.round((records.filter(r => r.resolved).length / total) * 100);

      document.getElementById("workflow-readiness-module").innerHTML = `
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">ACK</div>
            <div class="bar-track"><div class="bar-fill" style="width:${ack}%"></div></div>
            <div class="bar-value">${ack}%</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Assign</div>
            <div class="bar-track"><div class="bar-fill warn" style="width:${assigned}%"></div></div>
            <div class="bar-value">${assigned}%</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Escalate</div>
            <div class="bar-track"><div class="bar-fill critical" style="width:${escalated}%"></div></div>
            <div class="bar-value">${escalated}%</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Resolve</div>
            <div class="bar-track"><div class="bar-fill" style="width:${resolved}%"></div></div>
            <div class="bar-value">${resolved}%</div>
          </div>
        </div>
      `;
    }

    function renderOperationsWallSummary(){
      const top = getTopRiskPatient();
      const alerts = filterAlertsByUnit(activeAlerts);
      const patients = filterPatientsByUnit(activePatients);

      document.getElementById("operations-wall-summary").innerHTML = `
        <div class="mini-card">
          <div class="mini-k">Live State</div>
          <div class="mini-v">Operational</div>
          <div class="mini-copy">Current environment is presenting a live pilot command-center experience.</div>
        </div>
        <div class="mini-card">
          <div class="mini-k">Visible Scope</div>
          <div class="mini-v">${unitLabel(currentUnitFilter)}</div>
          <div class="mini-copy">${patients.length} monitored patients visible in this view.</div>
        </div>
        <div class="mini-card">
          <div class="mini-k">Top Risk Highlight</div>
          <div class="mini-v">${top ? safe(top.bed) : "--"}</div>
          <div class="mini-copy">${top ? `Review score ${patientRiskPercent(top)}% for ${safe(top.patient_id)}.` : `No higher-priority patient in view.`}</div>
        </div>
        <div class="mini-card">
          <div class="mini-k">Alerts in Scope</div>
          <div class="mini-v">${alerts.length}</div>
          <div class="mini-copy">Alert workload displayed for wall mode and executive walkthrough presentation.</div>
        </div>
      `;
    }

    function renderScenarioGrid(){
      const top = getTopRiskPatient();
      const target = document.getElementById("scenario-grid");
      if (!top){
        target.innerHTML = `
          <div class="mini-card"><div class="mini-k">Signal Change</div><div class="mini-copy">No monitored signal currently visible.</div></div>
          <div class="mini-card"><div class="mini-k">AI Review Logic</div><div class="mini-copy">AI detection panel waits for active patient risk.</div></div>
          <div class="mini-card"><div class="mini-k">Workflow Action</div><div class="mini-copy">Workflow action becomes available when patient risk is visible.</div></div>
        `;
        return;
      }

      target.innerHTML = `
        <div class="mini-card">
          <div class="mini-k">Signal Change</div>
          <div class="mini-copy">${safe(top.name)} shows visible monitored-context drift across oxygen saturation, respiratory rate, temperature, heart rate, or blood pressure signals.</div>
        </div>
        <div class="mini-card">
          <div class="mini-k">AI Review Logic</div>
          <div class="mini-copy">Supportive AI review logic elevates review attention and places the patient in the top risk view.</div>
        </div>
        <div class="mini-card">
          <div class="mini-k">Workflow Action</div>
          <div class="mini-copy">Care team can acknowledge, assign, escalate, or resolve from a single command workflow.</div>
        </div>
      `;
    }

    function updateSummary(){
      const sourcePatients = filterPatientsByUnit(activePatients);
      const sourceAlerts = filterAlertsByUnit(activeAlerts);
      const openAlerts = sourceAlerts.length;
      const criticalAlerts = sourceAlerts.filter(a => String(a.severity || "").toLowerCase() === "critical").length;
      const avgRisk = sourcePatients.length
        ? (sourcePatients.reduce((n, p) => n + patientRiskPercent(p), 0) / sourcePatients.length)
        : 0;

      document.getElementById("open-alerts").textContent = String(openAlerts);
      document.getElementById("critical-alerts").textContent = String(criticalAlerts);
      document.getElementById("avg-risk").textContent = Math.round(avgRisk) + "%";
      document.getElementById("unit-count").textContent = unitLabel(currentUnitFilter);

      const timeText = "Last Updated " + (lastUpdatedAt ? formatTime(lastUpdatedAt) : "--");
      const lastUpdatedPill = document.getElementById("lastUpdatedPill");
      const lastUpdatedPillTop = document.getElementById("lastUpdatedPillTop");
      if (lastUpdatedPill) lastUpdatedPill.textContent = timeText;
      if (lastUpdatedPillTop) lastUpdatedPillTop.textContent = timeText;

      const top = getTopRiskPatient();
      const topRiskText = top ? `Top Review Priority: ${safe(top.bed)} (${patientRiskPercent(top)}% review score)` : "Top Review-Priority Patient --";
      const topRiskPill = document.getElementById("topRiskPill");
      if (topRiskPill) topRiskPill.textContent = topRiskText;

      const healthy = ["ok", "connected"].includes(String(lastSystemHealth.status || "").toLowerCase());
      const liveSystemPill = document.getElementById("liveSystemPill");
      const feedHealthPill = document.getElementById("feedHealthPill");
      const wallStatus = document.getElementById("wallStatus");
      if (liveSystemPill) liveSystemPill.textContent = healthy ? "Live System" : "System Review";
      if (feedHealthPill){
        feedHealthPill.className = healthy ? "status-pill live" : "status-pill critical";
        feedHealthPill.textContent = healthy ? "Feed Live" : "Feed Review";
      }
      if (wallStatus){
        wallStatus.className = healthy ? "status-pill live" : "status-pill critical";
        wallStatus.textContent = healthy ? "Live" : "Review";
      }
    }

    function renderModules(){
      renderReportingDashboardModule();
      renderThresholdsModule();
      renderOxygenTrendModule();
      renderRiskTimelineModule();
      renderCapacityModule();
      renderForecastModule();
      renderClustersModule();
      renderHeatmapModule();
      renderWorkflowReadinessModule();
      renderRouteStatusModule();
      renderPilotPacketModule();
      renderOperationsWallSummary();
      renderScenarioGrid();
      renderValidationResultsModule();
      renderNoCommitModule();
    }

    function renderValidationResultsModule(){
      const target = document.getElementById("validation-results-module");
      if (!target) return;

      target.innerHTML = `<div class="mini-copy">Loading MIMIC Validation Intelligence...</div>`;

      fetch("/api/validation/milestone", {cache:"no-store"})
        .then(r => r.json())
        .then(data => {
          if (!data || data.ok === false){
            target.innerHTML = `
              <div class="mini-copy">Validation Intelligence is not available yet.</div>
              <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
                <a class="btn secondary small" href="/validation-intelligence">Open Validation Page</a>
              </div>
            `;
            return;
          }

          const d = data.default_threshold || {};
          const metrics = data.next_validation_metrics || {};
          const comparison = data.operational_alert_burden_comparison || {};
          const lead = data.computed_validation_metrics && data.computed_validation_metrics.lead_time_before_event
            ? data.computed_validation_metrics.lead_time_before_event
            : {};

          const profiles = data.threshold_profiles || [];
          const examples = (lead.detected_examples || []).slice(0, 8).sort((a,b) => Number(b.score || 0) - Number(a.score || 0));

          function pct(v){
            if (v === undefined || v === null || v === "") return "—";
            return String(v).replace(".0","") + "%";
          }

          function safeText(v){
            if (v === undefined || v === null || v === "") return "—";
            return String(v)
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;");
          }

          function cleanDriver(v){
            if (!v) return "Review context";
            if (String(v).toLowerCase() === "no dominant driver") return "Composite multi-signal pattern";
            return v;
          }

          function leadHours(v){
            if (v === undefined || v === null || v === "") return "—";
            const n = Number(v);
            if (!Number.isFinite(n)) return safeText(v);
            return (Math.round(n * 10) / 10).toFixed(n % 1 === 0 ? 0 : 1) + " hrs";
          }

          target.innerHTML = `
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:12px;">
              <div class="mini-card" style="min-height:auto;padding:12px;">
                <div class="mini-k">Alert Reduction</div>
                <div class="mini-v" style="color:#3ad38f">${pct(d.alert_reduction_pct)}</div>
                <div class="mini-copy">Conservative t=6.0 review queue; intentionally selective for low alert burden.</div>
              </div>
              <div class="mini-card" style="min-height:auto;padding:12px;">
                <div class="mini-k">Median Lead Time</div>
                <div class="mini-v" style="color:#3ad38f">${metrics.median_lead_time_before_event_hours || "—"} hrs</div>
                <div class="mini-copy">Among detected event clusters within the 6-hour pre-event window.</div>
              </div>
              <div class="mini-card" style="min-height:auto;padding:12px;">
                <div class="mini-k">ERA FPR</div>
                <div class="mini-v" style="color:#3ad38f">${pct(d.era_fpr_pct)}</div>
                <div class="mini-copy">Compared with ${pct(d.threshold_fpr_pct)} for standard threshold alerting.</div>
              </div>
              <div class="mini-card" style="min-height:auto;padding:12px;">
                <div class="mini-k">Event-Cluster Detection</div>
                <div class="mini-v" style="color:#7aa2ff">${pct(d.era_patient_sensitivity_pct)}</div>
                <div class="mini-copy">Event-cluster detection at conservative t=6.0 in the 6-hour pre-event window.</div>
              </div>
            </div>

            <div class="note-box" style="margin-bottom:12px;">
              <strong style="color:#eef4ff">Command-center validation headline:</strong>
              ${safeText(data.lead_time_headline || "Retrospective validation supports configurable review prioritization with reduced alert burden.")}
            </div>

            <div class="note-box" style="margin-bottom:12px;">
              <strong>Operational alert burden:</strong>
              ERA displayed approximately ${metrics.alerts_per_patient_day || "—"} alerts per patient-day versus approximately ${comparison.standard_threshold_alerts_per_patient_day_approx || "—"} alerts per patient-day under standard threshold-only alerting.
            </div>

            <div style="overflow-x:auto;margin-top:10px">
              <table style="width:100%;border-collapse:collapse;font-size:12px">
                <thead>
                  <tr style="border-bottom:1px solid rgba(255,255,255,.12)">
                    <th style="text-align:left;padding:7px;color:#9adfff;font-weight:900">Threshold</th>
                    <th style="text-align:left;padding:7px;color:#9adfff;font-weight:900">Use Case</th>
                    <th style="text-align:right;padding:7px;color:#9adfff;font-weight:900">Detection</th>
                    <th style="text-align:right;padding:7px;color:#9adfff;font-weight:900">FPR</th>
                    <th style="text-align:right;padding:7px;color:#9adfff;font-weight:900">Alert Red.</th>
                  </tr>
                </thead>
                <tbody>
                  ${profiles.map(p => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                      <td style="padding:7px;color:#eef4ff;font-weight:900">t=${safeText(p.threshold)}</td>
                      <td style="padding:7px;color:#dce9ff">${safeText(p.unit_profile || p.framing || "Configurable")}</td>
                      <td style="padding:7px;text-align:right;color:#7aa2ff;font-weight:900">${pct(p.era_patient_sensitivity_pct)}</td>
                      <td style="padding:7px;text-align:right;color:#3ad38f;font-weight:900">${pct(p.era_fpr_pct)}</td>
                      <td style="padding:7px;text-align:right;color:#3ad38f;font-weight:900">${pct(p.alert_reduction_pct)}</td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>

            <div style="margin-top:14px;">
              <div class="mini-k">Representative Review Examples</div>
              <div class="mini-copy" style="margin-bottom:8px;">
                Queue-ranked retrospective examples showing lead time, priority tier, and primary driver.
              </div>
              <div style="overflow-x:auto">
                <table style="width:100%;border-collapse:collapse;font-size:12px">
                  <thead>
                    <tr style="border-bottom:1px solid rgba(255,255,255,.12)">
                      <th style="text-align:left;padding:7px;color:#9adfff;font-weight:900">Rank</th>
                      <th style="text-align:left;padding:7px;color:#9adfff;font-weight:900">Patient</th>
                      <th style="text-align:right;padding:7px;color:#9adfff;font-weight:900">Lead</th>
                      <th style="text-align:left;padding:7px;color:#9adfff;font-weight:900">Tier</th>
                      <th style="text-align:left;padding:7px;color:#9adfff;font-weight:900">Primary Driver</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${examples.length ? examples.map((x, idx) => `
                      <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                        <td style="padding:7px;color:#eef4ff;font-weight:1000">#${idx + 1}</td>
                        <td style="padding:7px;color:#dce9ff">${safeText(x.patient_id)}</td>
                        <td style="padding:7px;text-align:right;color:#3ad38f;font-weight:900">${leadHours(x.lead_hours)}</td>
                        <td style="padding:7px;color:#eef4ff">${safeText(x.priority_tier)}</td>
                        <td style="padding:7px;color:#dce9ff">${safeText(cleanDriver(x.primary_driver))}</td>
                      </tr>
                    `).join("") : `
                      <tr><td colspan="5" style="padding:7px;color:#9fb4d6">No representative examples available.</td></tr>
                    `}
                  </tbody>
                </table>
              </div>
            </div>

            <div class="note-box" style="margin-top:12px;">
              <strong>Operating-point framing:</strong> t=6.0 is intentionally selective for telemetry / stepdown review. t=4.0 is the high-acuity operating point when higher detection is preferred with more alerts.<br><br><strong>Decision-support framing:</strong>
              Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable event-cluster detection in a 6-hour pre-event window. Independent clinical review required.
            </div>

            <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
              <a class="btn secondary small" href="/validation-intelligence">Open Full Validation Intelligence</a>
              <a class="btn secondary small" href="/validation-evidence">Evidence Packet</a>
              <a class="btn secondary small" href="/validation-evidence/examples.csv">Examples CSV</a>
              <a class="btn secondary small" href="/data-ingest">Run New Retrospective Analysis</a>
              <a class="btn secondary small" href="/pilot-docs">Pilot Docs</a>
            </div>
          `;
        })
        .catch(err => {
          console.error("Validation Intelligence load failed:", err);
          target.innerHTML = `
            <div class="mini-copy">Validation Intelligence could not load from /api/validation/milestone.</div>
            <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
              <a class="btn secondary small" href="/validation-intelligence">Open Validation Page</a>
            </div>
          `;
        });
    }


    function renderNoCommitModule(){
      const target = document.getElementById("no-commit-module");
      if (!target) return;
      target.innerHTML = `
        <div class="note-box" style="margin-bottom:12px">
          Upload your hospital's de-identified vital-sign CSV and receive a retrospective analysis
          showing how the ERA prioritization logic would have performed against your documented
          clinical events — compared to standard threshold-based review rules.
          <strong style="color:#dce9ff">No EHR integration. No IT lift. No cost to evaluate.</strong>
        </div>
        <div class="bars">
          <div class="bar-row">
            <div class="bar-label">IT Lift</div>
            <div class="bar-track"><div class="bar-fill" style="width:4%;background:linear-gradient(135deg,#3ad38f,#8ff3c1)"></div></div>
            <div class="bar-value" style="color:#3ad38f">None</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Time to Results</div>
            <div class="bar-track"><div class="bar-fill" style="width:8%;background:linear-gradient(135deg,#3ad38f,#8ff3c1)"></div></div>
            <div class="bar-value" style="color:#3ad38f">&lt;5 min</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">Commitment</div>
            <div class="bar-track"><div class="bar-fill" style="width:2%;background:linear-gradient(135deg,#3ad38f,#8ff3c1)"></div></div>
            <div class="bar-value" style="color:#3ad38f">Zero</div>
          </div>
        </div>
        <div style="margin-top:14px">
          <a class="btn primary" href="/retro-upload" style="width:100%;display:flex">Upload De-Identified CSV</a>
        </div>
      `;
    }

    // ── LIVE DEMO MODE ──────────────────────────────────────────────
    let demoModeActive = false;
    let demoInterval = null;
    let patientSearchTerm = "";

    function toggleDemoMode(){
      demoModeActive = !demoModeActive;
      const btn = document.getElementById("demModeBtn");
      if (demoModeActive){
        btn.textContent = "\uD83C\uDFAC Live Demo Mode: ON";
        btn.style.background = "rgba(58,211,143,.15)";
        btn.style.borderColor = "rgba(58,211,143,.4)";
        btn.style.color = "#b6f5d9";
        demoInterval = setInterval(applyDemoDrift, 45000);
        applyDemoDrift();
      } else {
        btn.textContent = "\uD83C\uDFAC Live Demo Mode: OFF";
        btn.style.background = "rgba(244,189,106,.08)";
        btn.style.borderColor = "rgba(244,189,106,.3)";
        btn.style.color = "#ffe7bf";
        if (demoInterval){ clearInterval(demoInterval); demoInterval = null; }
      }
    }

    function applyDemoDrift(){
      if (!activePatients || !activePatients.length) return;
      const archetypes = [
        {hr:+3, spo2:-1.2, sbp:+2, rr:+1.5, temp:+0.15, label:"Sepsis drift"},
        {hr:+2, spo2:-2.0, sbp:-1, rr:+2.0, temp:+0.08, label:"Resp drift"},
        {hr:+4, spo2:-0.5, sbp:+3, rr:+0.8, temp:+0.05, label:"Cardiac drift"},
        {hr:-1, spo2:+0.3, sbp:-1, rr:-0.5, temp:-0.05, label:"Recovering"},
      ];
      activePatients = activePatients.map((p, i) => {
        const arch = archetypes[i % archetypes.length];
        const jitter = v => v + (Math.random() - 0.5) * 1.5;
        const clamp = (v,lo,hi) => Math.max(lo, Math.min(hi, v));
        const hr  = clamp(safeNumber(p.heart_rate)  + jitter(arch.hr),  28,  200);
        const spo = clamp(safeNumber(p.spo2)        + jitter(arch.spo2), 70, 100);
        const sbp = clamp(safeNumber(p.bp_systolic) + jitter(arch.sbp),  60, 220);
        const rr  = clamp(safeNumber(p.resp_rate)   + jitter(arch.rr),    4,  50);
        const tmp = clamp(safeNumber(p.temperature) + jitter(arch.temp), 94, 106);
        let score = 0;
        score += Math.max(0, hr - 90) * 0.035;
        score += Math.max(0, 94 - spo) * 0.75;
        score += Math.max(0, sbp - 140) * 0.02;
        score += Math.max(0, rr - 20) * 0.12;
        score += Math.max(0, tmp - 99.0) * 0.7;
        score = Math.max(0.8, Math.min(9.9, score));
        const status = score >= 8.5 ? "Critical" : score >= 6.2 ? "High" : "Stable";
        return {...p,
          heart_rate: Math.round(hr),
          spo2: Math.round(spo * 10)/10,
          bp_systolic: Math.round(sbp),
          resp_rate: Math.round(rr * 10)/10,
          temperature: Math.round(tmp * 100)/100,
          risk_score: Math.round(score * 10)/10,
          status,
          _demo_drift: arch.label,
        };
      });
      renderPatients();
      renderTopRiskPatients();
      renderAlertsList();
      const el = document.getElementById("lastUpdatedPill");
      if (el) el.textContent = "Demo drift " + new Date().toLocaleTimeString();
    }

    // ── PATIENT SEARCH FILTER ────────────────────────────────────────
    function filterPatientSearch(){
      patientSearchTerm = (document.getElementById("patientSearchInput")?.value || "").toLowerCase().trim();
      renderPatients();
      renderTopRiskPatients();
    }

    function applySearchFilter(patients){
      if (!patientSearchTerm) return patients;
      return patients.filter(p => {
        const pid   = (p.patient_id || "").toLowerCase();
        const name  = (p.name || "").toLowerCase();
        const room  = (p.room || "").toLowerCase();
        const unit  = (p.unit || "").toLowerCase();
        return pid.includes(patientSearchTerm) ||
               name.includes(patientSearchTerm) ||
               room.includes(patientSearchTerm) ||
               unit.includes(patientSearchTerm);
      });
    }

    function rerenderAll(){
      renderPatients();
      renderAlertsList();
      renderTopRiskPatients();
      renderAIReasoning();
      renderAIInsight();
      renderWorkflow();
      renderSystemHealth();
      renderSecurityControls();
      renderReadinessEvidence();
      renderPatientTimeline();
      renderAuditLog();
      renderAuditSummary();
      renderModules();
      updateSummary();
    }

    async function boot(){
      await loadAccessContext();
      await loadThresholds();
      await loadWorkflowState();
      await loadPersistentAudit();
      await loadSystemHealth();
      await loadPilotReadiness();
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
        await loadPersistentAudit();
        await loadThresholds();
        await loadSystemHealth();
        await loadPilotReadiness();
        await refreshSnapshot();
      }, 5000);
    }

    boot();
  
    // ERA_COMMAND_PATIENT_EXPLAINABILITY_V2_START
    function eraReviewScore(patient){
      return safeNumber(patient && patient.risk_score, 0);
    }

    function eraPriorityTier(patient){
      const score = eraReviewScore(patient);
      const status = String((patient && patient.status) || "").toLowerCase();
      if (status.includes("critical") || score >= 85 || score >= 8.5) return "Critical";
      if (status.includes("high") || status.includes("escalated") || score >= 70 || score >= 7) return "Elevated";
      if (status.includes("moderate") || score >= 45 || score >= 4.5) return "Watch";
      return "Low";
    }

    function eraTierClass(tier){
      const t = String(tier || "").toLowerCase();
      if (t.includes("critical")) return "critical";
      if (t.includes("elevated")) return "elevated";
      if (t.includes("watch")) return "watch";
      return "low";
    }

    function eraPrimaryDriver(patient){
      if (!patient) return "Review context";
      const reasons = Array.isArray(patient.reasons) ? patient.reasons.join(" ") : "";
      const text = [
        patient.alert_message,
        patient.recommended_action,
        patient.story,
        patient.title,
        reasons
      ].filter(Boolean).join(" ").toLowerCase();

      const spo2 = safeNumber(patient.spo2);
      const hr = safeNumber(patient.heart_rate);
      const sbp = safeNumber(patient.bp_systolic);
      const rr = safeNumber(patient.respiratory_rate);
      const temp = safeNumber(patient.temperature_f);

      if (text.includes("spo2") || text.includes("oxygen") || (spo2 && spo2 < 93)) return "SpO₂ decline";
      if (text.includes("respiratory") || text.includes("rr") || (rr && (rr < 12 || rr > 20))) return "RR instability";
      if (text.includes("heart") || text.includes("hr") || (hr && (hr >= 110 || hr <= 55))) return "HR instability";
      if (text.includes("blood pressure") || text.includes("bp") || (sbp && (sbp >= 150 || sbp < 100))) return "BP trend concern";
      if (text.includes("temp") || (temp && (temp < 97.6 || temp > 100.4))) return "Temperature trend";
      return "Composite multi-signal pattern";
    }

    function eraTrendDirection(patient){
      if (!patient) return "Stable";
      const text = [
        patient.alert_message,
        patient.recommended_action,
        patient.story,
        patient.title
      ].filter(Boolean).join(" ").toLowerCase();

      if (text.includes("worsen") || text.includes("rising") || text.includes("decline") || text.includes("drop") || text.includes("elevated")) return "Worsening";
      if (text.includes("improv") || text.includes("resolved") || text.includes("recover")) return "Improving";
      const score = eraReviewScore(patient);
      if (score >= 7 || score >= 70) return "Worsening";
      return "Stable";
    }

    function eraQueueRank(patient){
      if (!patient || !Array.isArray(activePatients)) return "--";
      const sorted = activePatients.slice().sort((a,b) => eraReviewScore(b) - eraReviewScore(a));
      const idx = sorted.findIndex(p => String(p.patient_id) === String(patient.patient_id));
      return idx >= 0 ? "#" + (idx + 1) : "--";
    }

    function eraFindSelectedPatient(){
      if (!Array.isArray(activePatients) || !activePatients.length) return null;
      if (selectedPatientId){
        const found = activePatients.find(p => String(p.patient_id) === String(selectedPatientId));
        if (found) return found;
      }
      const drawerTitle = document.getElementById("drawerPatientName");
      const txt = drawerTitle ? drawerTitle.textContent : "";
      return activePatients.find(p => txt.includes(p.patient_id) || txt.includes(p.name)) || activePatients[0];
    }

    function eraUpdateDrawerPriorityContext(){
      const patient = eraFindSelectedPatient();
      if (!patient) return;

      const tier = eraPriorityTier(patient);
      const rank = eraQueueRank(patient);
      const driver = eraPrimaryDriver(patient);
      const trend = eraTrendDirection(patient);

      const tierEl = document.getElementById("drawerPriorityTier");
      const rankEl = document.getElementById("drawerQueueRank");
      const driverEl = document.getElementById("drawerPrimaryDriver");
      const trendEl = document.getElementById("drawerTrendDirection");

      if (tierEl) tierEl.textContent = tier;
      if (rankEl) rankEl.textContent = rank;
      if (driverEl) driverEl.textContent = driver;
      if (trendEl) trendEl.textContent = trend;

      const explain = document.getElementById("drawerExplainability");
      if (explain && patient){
        const base = explain.textContent && explain.textContent !== "--" ? explain.textContent : explainabilityForPatient(patient);
        if (!base.includes("Priority tier:")){
          explain.textContent = base + " Priority tier: " + tier + ". Queue rank: " + rank + ". Primary driver: " + driver + ". Trend direction: " + trend + ".";
        }
      }
    }

    function eraDecoratePatientCards(){
      const wall = document.getElementById("wall");
      if (!wall || !Array.isArray(activePatients) || !activePatients.length) return;

      const monitors = Array.from(wall.querySelectorAll(".monitor"));
      if (!monitors.length) return;

      activePatients.forEach(patient => {
        const id = String(patient.patient_id || "");
        const name = String(patient.name || "");
        const card = monitors.find(m => m.textContent.includes(id) || (name && m.textContent.includes(name)));
        if (!card) return;

        let box = card.querySelector(".era-card-context");
        if (!box){
          box = document.createElement("div");
          box.className = "era-card-context";
          const story = card.querySelector(".monitor-story") || card.querySelector(".monitor-actions") || card;
          if (story && story.parentNode){
            story.parentNode.insertBefore(box, story.nextSibling);
          } else {
            card.appendChild(box);
          }
        }

        const tier = eraPriorityTier(patient);
        const cls = eraTierClass(tier);
        box.innerHTML = `
          <div class="era-context-chip ${cls}">Tier: ${tier}</div>
          <div class="era-context-chip">Rank: ${eraQueueRank(patient)}</div>
          <div class="era-context-chip">Driver: ${eraPrimaryDriver(patient)}</div>
          <div class="era-context-chip">Trend: ${eraTrendDirection(patient)}</div>
        `;
      });
    }

    function eraInstallPatientExplainabilityUpgrade(){
      try{
        const originalOpen = typeof openPatientDrawer === "function" ? openPatientDrawer : null;
        if (originalOpen && !originalOpen.__eraWrapped){
          const wrapped = function(){
            const out = originalOpen.apply(this, arguments);
            setTimeout(eraUpdateDrawerPriorityContext, 80);
            setTimeout(eraDecoratePatientCards, 120);
            return out;
          };
          wrapped.__eraWrapped = true;
          openPatientDrawer = wrapped;
        }
      }catch(err){
        console.warn("ERA drawer wrapper skipped", err);
      }

      setTimeout(eraDecoratePatientCards, 400);
      setTimeout(eraUpdateDrawerPriorityContext, 500);
    }

    document.addEventListener("DOMContentLoaded", eraInstallPatientExplainabilityUpgrade);
    setInterval(eraDecoratePatientCards, 5000);
    setInterval(eraUpdateDrawerPriorityContext, 5000);
    // ERA_COMMAND_PATIENT_EXPLAINABILITY_V2_END

  </script>

<!-- ERA_MINIMAL_EXPLAINABILITY_VISIBILITY_V2_START -->

<section id="explainability-review-queue" style="margin: 28px 0; padding: 22px; border: 1px solid rgba(164,255,190,.32); border-radius: 24px; background: rgba(9,18,32,.72);">
  <div style="display:inline-block; padding: 6px 12px; border-radius:999px; border:1px solid rgba(164,255,190,.45); color:#b8ffc8; font-weight:800; letter-spacing:.08em; font-size:.75rem; text-transform:uppercase;">Explainability visibility</div>
  <h2 style="margin:12px 0 8px; font-size: clamp(1.6rem, 3vw, 2.6rem); line-height:1.05;">Explainable review queue</h2>
  <p style="max-width:980px; opacity:.9; margin-bottom:18px;">This command-center view shows review prioritization context: queue rank, priority tier, primary driver, trend direction, lead-time context, and workflow state. It is decision-support only and does not diagnose, direct treatment, or independently trigger escalation.</p>

  <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; min-width:960px;">
      <thead>
        <tr style="background:rgba(120,160,220,.18);">
          <th style="text-align:left; padding:12px;">Queue Rank</th>
          <th style="text-align:left; padding:12px;">Case</th>
          <th style="text-align:left; padding:12px;">Unit</th>
          <th style="text-align:left; padding:12px;">Priority Tier</th>
          <th style="text-align:left; padding:12px;">Risk Score</th>
          <th style="text-align:left; padding:12px;">Primary Driver</th>
          <th style="text-align:left; padding:12px;">Trend</th>
          <th style="text-align:left; padding:12px;">Lead-Time Context</th>
          <th style="text-align:left; padding:12px;">First Threshold Crossing</th>
          <th style="text-align:left; padding:12px;">Workflow State</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12); font-weight:800;">#1</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Demo-001</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">ICU</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Critical</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">9.2</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">SpO2 decline</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Worsening</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">~4.0 hrs</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">t=6.0 crossed</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Needs review</td>
        </tr>
        <tr>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12); font-weight:800;">#2</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Demo-002</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Telemetry</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Critical</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">8.7</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">BP instability</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Worsening</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">~3.5 hrs</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">t=6.0 crossed</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Acknowledged</td>
        </tr>
        <tr>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12); font-weight:800;">#3</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Demo-003</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Stepdown</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Elevated</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">7.9</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">HR instability</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Stable / Watch</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">~3.0 hrs</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">t=5.0 crossed</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Assigned</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-top:16px;">
    <div style="padding:14px; border-radius:18px; background:rgba(255,255,255,.06);"><strong>Priority Tier</strong><br>Low / Watch / Elevated / Critical</div>
    <div style="padding:14px; border-radius:18px; background:rgba(255,255,255,.06);"><strong>Queue Rank</strong><br>Relative review ordering</div>
    <div style="padding:14px; border-radius:18px; background:rgba(255,255,255,.06);"><strong>Primary Driver</strong><br>SpO2 / HR / BP / RR context</div>
    <div style="padding:14px; border-radius:18px; background:rgba(255,255,255,.06);"><strong>Lead-Time Context</strong><br>Retrospective timing context</div>
  </div>

  <p style="margin-top:14px; padding:12px 14px; border-radius:16px; border:1px solid rgba(255,205,110,.45); color:#ffd778; background:rgba(255,205,110,.08);">Guardrail: simulated examples only. No real patient IDs, no row-level timestamps, and no raw MIMIC/eICU data are displayed.</p>
</section>
<!-- ERA_MINIMAL_EXPLAINABILITY_VISIBILITY_V2_END -->
</body>
</html>
"""
