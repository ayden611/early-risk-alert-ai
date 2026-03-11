from __future__ import annotations

import os
from flask import Flask

INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder, Early Risk Alert AI"
YOUTUBE_EMBED_URL = "https://www.youtube.com/embed/z4SbeYwwm7k"
PROD_BASE_URL = "https://early-risk-alert-ai-1.onrender.com"

MAIN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#07101c;
      --bg2:#0c1425;
      --panel:#101a2d;
      --panel2:#13203a;
      --line:rgba(255,255,255,.08);
      --line2:rgba(255,255,255,.05);
      --text:#eef4ff;
      --muted:#a8bddc;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --green:#38d39f;
      --amber:#f4bd6a;
      --red:#ff667d;
      --shadow:0 20px 60px rgba(0,0,0,.30);
      --radius:24px;
      --max:1360px;
    }

    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.16), transparent 22%),
        radial-gradient(circle at 82% 12%, rgba(91,212,255,.10), transparent 20%),
        linear-gradient(180deg, var(--bg), var(--bg2));
      overflow-x:hidden;
    }
    a{text-decoration:none;color:inherit}
    img{max-width:100%;display:block}
    .shell{max-width:var(--max);margin:0 auto;padding:22px 16px 56px}
    .nav{
      position:sticky;top:0;z-index:1000;
      background:rgba(7,16,28,.82);
      backdrop-filter:blur(14px);
      border-bottom:1px solid var(--line);
    }
    .nav-inner{
      max-width:var(--max);
      margin:0 auto;
      padding:14px 16px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:18px;
      flex-wrap:wrap;
    }
    .brand-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8cd7ef;
    }
    .brand-title{
      font-size:clamp(26px,3vw,40px);font-weight:1000;line-height:.95;letter-spacing:-.05em;
    }
    .brand-sub{
      font-size:14px;color:var(--muted);font-weight:800;
    }
    .nav-links{
      display:flex;align-items:center;gap:16px;flex-wrap:wrap;
    }
    .nav-links a{font-size:14px;font-weight:900}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;gap:8px;
      padding:13px 18px;border-radius:16px;font-size:14px;font-weight:900;cursor:pointer;
      border:1px solid transparent;
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, opacity .18s ease;
    }
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{
      background:linear-gradient(135deg,var(--blue),var(--blue2));
      color:#07101c;box-shadow:0 12px 30px rgba(91,212,255,.22);
    }
    .btn.secondary{
      background:rgba(255,255,255,.04);
      color:var(--text);border-color:var(--line);
    }
    .hero{
      position:relative;overflow:hidden;border:1px solid var(--line);border-radius:32px;
      box-shadow:var(--shadow);
      background:
        linear-gradient(180deg, rgba(10,18,31,.26), rgba(7,16,28,.78)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
      min-height:720px;
    }
    .hero::before{
      content:"";position:absolute;inset:0;
      background:
        radial-gradient(circle at 50% 10%, rgba(91,212,255,.24), transparent 18%),
        linear-gradient(110deg, transparent 35%, rgba(255,255,255,.06) 50%, transparent 65%);
      animation:heroSweep 9s linear infinite;
      pointer-events:none;
    }
    @keyframes heroSweep{
      0%{transform:translateX(-40px)}
      50%{transform:translateX(40px)}
      100%{transform:translateX(-40px)}
    }
    .hero-inner{
      position:relative;z-index:2;min-height:720px;display:flex;align-items:flex-end;padding:36px;
    }
    .hero-grid{
      width:100%;display:grid;grid-template-columns:1.08fr .92fr;gap:18px;align-items:end;
    }
    .glass{
      border:1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      border-radius:28px;box-shadow:0 16px 42px rgba(0,0,0,.24);backdrop-filter:blur(14px);
    }
    .hero-copy{padding:28px}
    .hero-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px;
    }
    .hero-copy h1{
      margin:0 0 14px;font-size:clamp(40px,6vw,82px);line-height:.92;letter-spacing:-.06em;font-weight:1000;max-width:760px;text-wrap:balance;
    }
    .hero-copy p{
      margin:0;color:#d0ddf0;font-size:18px;line-height:1.68;max-width:760px;
    }
    .hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}
    .hero-mini-grid{margin-top:20px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .hero-mini{
      border:1px solid var(--line2);border-radius:18px;padding:14px;background:rgba(255,255,255,.03);
    }
    .hero-mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:8px;
    }
    .hero-mini .v{font-size:16px;font-weight:1000;line-height:1.1}
    .demo-card{overflow:hidden}
    .demo-stage{
      position:relative;aspect-ratio:16/9;min-height:360px;
      background:
        linear-gradient(180deg, rgba(0,0,0,.12), rgba(0,0,0,.40)),
        url('/static/images/ai-command-center.jpg') center/cover no-repeat;
    }
    .demo-badge{
      position:absolute;top:18px;left:18px;z-index:3;padding:9px 13px;border-radius:999px;
      background:rgba(7,16,28,.62);border:1px solid var(--line);color:#eaf3ff;
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;backdrop-filter:blur(12px);
    }
    .demo-play{
      position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:3;
    }
    .demo-play-btn{
      width:96px;height:96px;border-radius:50%;border:1px solid rgba(255,255,255,.18);
      background:rgba(7,16,28,.58);backdrop-filter:blur(12px);
      box-shadow:0 20px 50px rgba(0,0,0,.32), 0 0 34px rgba(91,212,255,.18);
      display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative;
    }
    .demo-play-btn::before{
      content:"";position:absolute;inset:-14px;border-radius:50%;
      border:1px solid rgba(91,212,255,.20);animation:ring 2.8s ease-out infinite;
    }
    @keyframes ring{
      0%{transform:scale(.7);opacity:1}
      100%{transform:scale(1.35);opacity:0}
    }
    .demo-play-btn svg{width:34px;height:34px;margin-left:5px;fill:#eef4ff}
    .demo-caption{
      position:absolute;left:0;right:0;bottom:0;z-index:2;padding:22px;
      background:linear-gradient(180deg, transparent, rgba(7,16,28,.82));
    }
    .demo-caption h3{
      margin:0 0 8px;font-size:30px;line-height:1;font-weight:1000;letter-spacing:-.04em;
    }
    .demo-caption p{margin:0;color:#d4e2f3;font-size:15px;line-height:1.6}
    .demo-bottom{padding:18px;border-top:1px solid var(--line)}
    .demo-note{color:#bdd0eb;font-size:14px;line-height:1.55;margin-bottom:14px}
    .demo-btns{display:flex;gap:10px;flex-wrap:wrap}
    .route-grid{
      margin-top:18px;
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:14px;
    }
    .route-card{
      border:1px solid var(--line);
      border-radius:24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:20px;
    }
    .route-label{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;
    }
    .route-card h3{
      margin:10px 0 10px;font-size:30px;line-height:1;font-weight:1000;letter-spacing:-.04em;
    }
    .route-card p{
      margin:0;color:#d0ddf0;font-size:15px;line-height:1.64;
    }
    .route-card .route-actions{
      margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;
    }
    .section{
      margin-top:22px;
      border:1px solid var(--line);
      border-radius:28px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018)),
        linear-gradient(180deg, rgba(12,22,38,.76), rgba(9,16,30,.88));
      box-shadow:0 16px 42px rgba(0,0,0,.24);
      padding:26px;
    }
    .section-kicker{
      font-size:11px;font-weight:900;letter-spacing:.16em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px;
    }
    .section h2{
      margin:0 0 10px;font-size:clamp(34px,5vw,58px);line-height:.95;letter-spacing:-.05em;font-weight:1000;
    }
    .section p{
      margin:0;color:#d0ddf0;font-size:16px;line-height:1.7;max-width:980px;
    }
    .list{
      margin-top:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
    }
    .mini{
      border:1px solid var(--line2);border-radius:18px;padding:16px;background:rgba(255,255,255,.03);
    }
    .mini .k{
      font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8fd7ff;margin-bottom:10px;
    }
    .mini .v{
      display:block;font-size:20px;font-weight:1000;line-height:1.1;letter-spacing:-.03em;
    }
    .mini p{
      margin:10px 0 0;color:#c7d8ef;font-size:14px;line-height:1.6;
    }
    .footer{
      margin-top:22px;
      padding:18px 0 8px;
      color:#9fb4d6;
      font-size:14px;
      line-height:1.6;
    }

    @media (max-width:1100px){
      .hero-grid{grid-template-columns:1fr}
      .hero-mini-grid{grid-template-columns:repeat(2,1fr)}
      .route-grid{grid-template-columns:1fr}
      .list{grid-template-columns:1fr}
    }

    @media (max-width:760px){
      .nav-inner{padding:12px 14px}
      .nav-links{gap:12px}
      .hero{min-height:auto}
      .hero-inner{min-height:auto;padding:16px}
      .hero-copy{padding:18px}
      .hero-copy h1{font-size:clamp(34px,11vw,54px)}
      .hero-copy p{font-size:16px}
      .hero-actions,.demo-btns,.route-card .route-actions{flex-direction:column}
      .btn{width:100%}
      .demo-stage{min-height:250px}
      .hero-mini-grid{grid-template-columns:1fr}
      .section{padding:18px}
    }
  </style>
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
        <a href="https://early-risk-alert-ai-1.onrender.com/#overview">Overview</a>
        <a href="https://early-risk-alert-ai-1.onrender.com/#hospital-story">Hospital Story</a>
        <a href="https://early-risk-alert-ai-1.onrender.com/#product-walkthrough">Live Platform</a>
        <a href="https://early-risk-alert-ai-1.onrender.com/#commercial-path">Investor Story</a>
        <a href="https://early-risk-alert-ai-1.onrender.com/#dashboard">Command Center</a>
        <a class="btn primary" href="https://early-risk-alert-ai-1.onrender.com/#commercial-path">Pitch Deck</a>
      </div>
    </div>
  </div>

  <div class="shell">
    <section class="hero" id="overview">
      <div class="hero-inner">
        <div class="hero-grid">
          <div class="glass hero-copy">
            <div class="hero-kicker">Single-domain production experience</div>
            <h1>See the platform. Open the demo. Move hospitals and investors into the right path.</h1>
            <p>
              Early Risk Alert AI is a professional predictive clinical intelligence platform for hospitals,
              care operations, and investors. This opening experience keeps everything on one clean production domain
              and uses your brand image as the polished entry to the live platform story.
            </p>

            <div class="hero-actions">
              <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
              <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#hospital-story">Open Hospital Story</a>
              <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#commercial-path">Open Investor Story</a>
              <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#dashboard">Open Command Center</a>
            </div>

            <div class="hero-mini-grid">
              <div class="hero-mini">
                <div class="k">Hospital Path</div>
                <div class="v">Clinical workflow review</div>
              </div>
              <div class="hero-mini">
                <div class="k">Live Platform</div>
                <div class="v">Alerts, scoring, confidence</div>
              </div>
              <div class="hero-mini">
                <div class="k">Investor Path</div>
                <div class="v">Commercial positioning</div>
              </div>
              <div class="hero-mini">
                <div class="k">Production Lock</div>
                <div class="v">Single trusted domain</div>
              </div>
            </div>
          </div>

          <div class="glass demo-card" id="demo">
            <div class="demo-stage">
              <div class="demo-badge">Platform Demo</div>

              <div class="demo-play">
                <a class="demo-play-btn" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer" aria-label="Watch demo video">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M8 5v14l11-7z"></path>
                  </svg>
                </a>
              </div>

              <div class="demo-caption">
                <h3>Early Risk Alert AI Master Demo</h3>
                <p>A clean public-facing demo entry using your branded image as the thumbnail and your walkthrough as the video destination.</p>
              </div>
            </div>

            <div class="demo-bottom">
              <div class="demo-note">
                Clicking play opens the full walkthrough. All platform paths below stay on the working homepage and do not break.
              </div>
              <div class="demo-btns">
                <a class="btn primary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Play Demo</a>
                <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#hospital-story">Hospital Demo</a>
                <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#commercial-path">Investor View</a>
                <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#dashboard">Command Center</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="route-grid">
      <div class="route-card" id="hospital-story">
        <div class="route-label">Hospital Story</div>
        <h3>Clinical Buyer Path</h3>
        <p>Show workflows, command-center visibility, AI risk scoring, and operational value for hospitals and care teams.</p>
        <div class="route-actions">
          <a class="btn secondary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Watch Demo</a>
          <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#dashboard">Open Command Center</a>
        </div>
      </div>

      <div class="route-card" id="product-walkthrough">
        <div class="route-label">Live Platform</div>
        <h3>Product Walkthrough</h3>
        <p>Present alerts, patient focus, confidence indicators, recommended action, and the live platform experience.</p>
        <div class="route-actions">
          <a class="btn secondary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Watch Demo</a>
          <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#dashboard">Open Command Center</a>
        </div>
      </div>

      <div class="route-card" id="commercial-path">
        <div class="route-label">Investor Story</div>
        <h3>Commercial Path</h3>
        <p>Guide investors through product readiness, founder credibility, market relevance, and enterprise SaaS positioning.</p>
        <div class="route-actions">
          <a class="btn secondary" href="https://youtu.be/z4SbeYwwm7k" target="_blank" rel="noopener noreferrer">Watch Demo</a>
          <a class="btn secondary" href="https://early-risk-alert-ai-1.onrender.com/#contact">Founder Contact</a>
        </div>
      </div>
    </div>

    <section class="section" id="dashboard">
      <div class="section-kicker">Clinical Command Center</div>
      <h2>Live command-center style presentation from one stable page.</h2>
      <p>
        This section becomes the working command center entry point while the public site stays stable.
        Use it to present live alerts, AI scoring, confidence indicators, and hospital operations story from the same homepage.
      </p>

      <div class="list">
        <div class="mini">
          <div class="k">Monitoring</div>
          <span class="v">Real-time alert visibility</span>
          <p>Show live patient status, alert flow, and risk prioritization in one executive-facing section.</p>
        </div>
        <div class="mini">
          <div class="k">AI Scoring</div>
          <span class="v">Confidence and action signals</span>
          <p>Present severity, confidence, and next-step reasoning in a clean command-center narrative.</p>
        </div>
        <div class="mini">
          <div class="k">Operations</div>
          <span class="v">Hospital-ready workflow story</span>
          <p>Use this as the stable platform explanation while deeper route-specific pages are rebuilt later.</p>
        </div>
      </div>
    </section>

    <section class="section" id="contact">
      <div class="section-kicker">Founder & Contact</div>
      <h2>Milton Munroe</h2>
      <p>
        Founder, Early Risk Alert AI · <a href="mailto:info@earlyriskalertai.com">info@earlyriskalertai.com</a> ·
        <a href="tel:7327247267">732-724-7267</a>
      </p>
    </section>

    <div class="footer">
      Early Risk Alert AI LLC · Predictive clinical intelligence platform · Hospitals · Clinics · Investors · Patients
    </div>
  </div>

  <script>
    (function () {
      var allowedHost = "early-risk-alert-ai-1.onrender.com";
      if (window.location.hostname !== allowedHost && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
        var target = "https://" + allowedHost + window.location.pathname + window.location.search + window.location.hash;
        window.location.replace(target);
        return;
      }

      var routeMap = {
        "/": "overview",
        "/overview": "overview",
        "/investors": "commercial-path",
        "/investor": "commercial-path",
        "/investor-view": "commercial-path",
        "/hospital-demo": "hospital-story",
        "/hospital": "hospital-story",
        "/hospital-intake": "hospital-story",
        "/dashboard": "dashboard",
        "/command-center": "dashboard",
        "/deck": "commercial-path",
        "/demo": "demo"
      };

      var targetId = routeMap[window.location.pathname];
      if (targetId) {
        window.addEventListener("load", function () {
          var el = document.getElementById(targetId);
          if (el) {
            setTimeout(function () {
              el.scrollIntoView({ behavior: "smooth", block: "start" });
            }, 120);
          }
        });
      }
    })();
  </script>
</body>
</html>
"""

INVESTOR_HTML = MAIN_HTML
HOSPITAL_HTML = MAIN_HTML
COMMAND_CENTER_HTML = MAIN_HTML


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "era-dev-secret")

    try:
        from era.web.routes import web_bp
        app.register_blueprint(web_bp)
    except Exception as exc:
        @app.get("/")
        def startup_error():
            return f"<h1>Startup Error</h1><pre>{exc}</pre>", 500

    try:
        from era.api.routes import api_bp
        app.register_blueprint(api_bp)
    except Exception:
        pass

    return app
