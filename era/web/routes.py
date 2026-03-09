from flask import Blueprint, render_template, render_template_string, send_file
import os

web_bp = Blueprint("web", __name__)

@web_bp.route("/pitch-deck")
def pitch_deck():
    return send_from_directory(
        "static",
        "Early_Risk_Alert_AI_Pitch_Deck.pdf",
        as_attachment=True
    )

COMMAND_CENTER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert Command Center</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --text:#ecf3ff;
      --muted:#92a6c8;
      --border:rgba(255,255,255,.08);
      --accent:#7aa2ff;
      --accent2:#5b84f7;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.10), transparent 22%),
        radial-gradient(circle at top right, rgba(91,211,141,.08), transparent 20%),
        var(--bg);
      color:var(--text);
    }
    .shell{max-width:1400px;margin:0 auto;padding:24px 18px 60px}
    .top{display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:22px}
    .kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;font-weight:900;color:#dbe7ff}
    .title{font-size:46px;font-weight:950;letter-spacing:-.04em;margin:6px 0}
    .sub{color:var(--muted);font-size:15px;font-weight:700}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:12px 18px;border-radius:12px;text-decoration:none;font-weight:900;
      background:linear-gradient(180deg,var(--accent),var(--accent2));
      color:#08111f;border:none
    }
    .btn.secondary{
      background:#111b2f;color:var(--text);border:1px solid var(--border)
    }
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:22px;
      padding:22px;
    }
    .card h3{margin:0 0 10px;font-size:24px}
    .card p{margin:0;color:#cbd8ef;line-height:1.6}
    .live{
      display:inline-flex;align-items:center;gap:10px;
      background:rgba(91,211,141,.11);
      border:1px solid rgba(91,211,141,.22);
      color:#d8ffe8;padding:8px 12px;border-radius:999px;
      font-size:12px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;
      margin-top:14px
    }
    .dot{width:10px;height:10px;border-radius:999px;background:#5bd38d}
    @media (max-width:900px){ .grid{grid-template-columns:1fr} .title{font-size:38px} }
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div>
        <div class="kicker">Predictive Clinical Intelligence</div>
        <div class="title">Early Risk Alert</div>
        <div class="sub">Hospital UI + Investor Demo + Command Center</div>
        <div class="live"><span class="dot"></span> Live System</div>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <a class="btn secondary" href="/investors">Investor View</a>
        <a class="btn secondary" href="/deck">Download Pitch Deck</a>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Live Alerts Feed</h3>
        <p>Clinical alerts, patient prioritization, and rapid visibility for command center operations.</p>
      </div>
      <div class="card">
        <h3>Patient Focus</h3>
        <p>Focused patient view for vital signs, risk indicators, and rollup metrics.</p>
      </div>
      <div class="card">
        <h3>System Panels</h3>
        <p>Operational stream health, channels, and command center summary views.</p>
      </div>
    </div>
  </div>
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
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --text:#ecf3ff;
      --muted:#92a6c8;
      --border:rgba(255,255,255,.08);
      --accent:#7aa2ff;
      --accent2:#5b84f7;
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,.10), transparent 22%),
        radial-gradient(circle at top right, rgba(91,211,141,.08), transparent 20%),
        var(--bg);
      color:var(--text);
    }
    .shell{max-width:1440px;margin:0 auto;padding:22px 18px 60px}
    .nav{
      display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
      margin-bottom:28px
    }
    .logo-kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;font-weight:900;color:#dbe7ff}
    .logo{font-size:34px;font-weight:950;letter-spacing:-.04em;margin-top:4px}
    .nav-links{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
    .nav-links a{text-decoration:none;color:#dfe9ff;font-weight:800;font-size:14px}
    .btn{
      display:inline-flex;align-items:center;justify-content:center;
      padding:13px 18px;border-radius:12px;text-decoration:none;font-weight:900;
      background:linear-gradient(180deg,var(--accent),var(--accent2));
      color:#08111f;border:none
    }
    .btn.secondary{background:#111b2f;color:var(--text);border:1px solid var(--border)}
    .btn.outline{background:transparent;color:#dfe9ff;border:1px solid var(--border)}
    .hero{
      display:grid;grid-template-columns:1.1fr .9fr;gap:20px;align-items:stretch
    }
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);
      border-radius:26px;
      padding:28px;
    }
    .hero-kicker{font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:900;color:#dce7ff;margin-bottom:16px}
    .headline{font-size:76px;line-height:.94;font-weight:950;letter-spacing:-.065em;margin:0 0 18px}
    .body{font-size:22px;line-height:1.45;color:#d3def2}
    .sub{margin-top:16px;font-size:17px;line-height:1.65;color:#a9bad8}
    .actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}
    .side-title{font-size:34px;font-weight:950;letter-spacing:-.03em;margin:0 0 14px}
    .side-text{color:#d5e1f5;font-size:18px;line-height:1.6;margin-bottom:18px}
    .grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
    .mini{
      border:1px solid rgba(255,255,255,.05);
      background:rgba(255,255,255,.02);
      border-radius:18px;padding:16px;min-height:120px
    }
    .mini-k{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#9cb1d6;font-weight:900}
    .mini-v{font-size:28px;font-weight:950;letter-spacing:-.04em;margin-top:10px}
    .mini-s{margin-top:6px;color:#b7c7e4;font-size:13px;line-height:1.45;font-weight:700}
    .section{margin-top:34px}
    .section-title{font-size:40px;font-weight:950;letter-spacing:-.04em;margin:0 0 18px}
    .cards3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .cards2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
    .box h3{margin:0 0 10px;font-size:24px}
    .box p{margin:0;color:#c9d7ef;font-size:16px;line-height:1.6}
    .band{
      background:linear-gradient(180deg, rgba(122,162,255,.08), rgba(122,162,255,.03));
      border:1px solid rgba(122,162,255,.16);
      border-radius:26px;padding:28px
    }
    .quote{font-size:28px;line-height:1.45;color:#eaf2ff;font-weight:800;letter-spacing:-.02em}
    .contact-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .contact-card{
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
      border:1px solid var(--border);border-radius:20px;padding:20px
    }
    .contact-label{font-size:11px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;color:#9fb3d8;margin-bottom:10px}
    .contact-value{font-size:28px;font-weight:900;letter-spacing:-.03em}
    .contact-value.small{font-size:20px;line-height:1.4;word-break:break-word}
    .contact-link{color:#9bc0ff;text-decoration:none;font-weight:900}
    .footer{
      margin-top:42px;padding-top:18px;border-top:1px solid rgba(255,255,255,.06);
      color:#88a0c8;font-size:14px;text-align:center
    }
    @media (max-width:1200px){
      .hero{grid-template-columns:1fr}
      .headline{font-size:56px}
      .cards3{grid-template-columns:1fr 1fr}
      .contact-grid{grid-template-columns:1fr 1fr}
    }
    @media (max-width:900px){
      .cards3,.cards2,.grid-2,.contact-grid{grid-template-columns:1fr}
      .headline{font-size:42px}
      .logo{font-size:28px}
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="nav">
      <div>
        <div class="logo-kicker">Investor Overview</div>
        <div class="logo">Early Risk Alert AI</div>
      </div>
      <div class="nav-links">
        <a href="#problem">Problem</a>
        <a href="#solution">Solution</a>
        <a href="#revenue">Revenue Model</a>
        <a href="#compliance">Compliance</a>
        <a href="#partnership">Partnership</a>
        <a href="#founder">Founder</a>
        <a class="btn secondary" href="/">Hospital Command Center</a>
        <a class="btn" href="#contact">Contact Founder</a>
      </div>
    </div>

    <div class="hero">
      <div class="card">
        <div class="hero-kicker">AI-Powered Predictive Clinical Intelligence</div>
        <h1 class="headline">Detect patient deterioration earlier. Strengthen hospital response at scale.</h1>
        <div class="body">
          Early Risk Alert AI is a clinical intelligence platform designed to help hospitals, health systems,
          and remote monitoring programs identify elevated-risk patients in real time.
        </div>
        <div class="sub">
          Our platform transforms fragmented monitoring into a unified command-center model —
          bringing together live patient visibility, AI-assisted risk prioritization, and enterprise-grade operational oversight.
        </div>
        <div class="actions">
          <a class="btn" href="/">View Live Product Demo</a>
          <a class="btn secondary" href="/">Open Hospital Command Center</a>
          <a class="btn outline" href="/deck">Download Pitch Deck PDF</a>
        </div>
      </div>

      <div class="card">
        <div class="side-title">Investment Highlights</div>
        <div class="side-text">
          Positioned at the intersection of hospital efficiency, predictive monitoring,
          and scalable healthcare software infrastructure.
        </div>
        <div class="grid-2">
          <div class="mini">
            <div class="mini-k">Model</div>
            <div class="mini-v">Enterprise SaaS</div>
            <div class="mini-s">Recurring platform revenue with enterprise expansion potential.</div>
          </div>
          <div class="mini">
            <div class="mini-k">Primary Buyer</div>
            <div class="mini-v">Hospitals</div>
            <div class="mini-s">Built for hospital systems, command centers, and remote monitoring teams.</div>
          </div>
          <div class="mini">
            <div class="mini-k">Use Case</div>
            <div class="mini-v">Early Risk Detection</div>
            <div class="mini-s">Identify deterioration signals before emergencies escalate.</div>
          </div>
          <div class="mini">
            <div class="mini-k">Expansion</div>
            <div class="mini-v">Multi-Site Rollout</div>
            <div class="mini-s">Designed for scale across facilities and enterprise networks.</div>
          </div>
        </div>
      </div>
    </div>

    <div id="problem" class="section">
      <div class="section-title">The Problem</div>
      <div class="cards3">
        <div class="card box"><h3>Reactive Care Models</h3><p>Many healthcare environments still depend on delayed intervention after patient risk has already escalated.</p></div>
        <div class="card box"><h3>Operational Fragmentation</h3><p>Clinical teams often work across disconnected monitoring tools, scattered dashboards, and inconsistent alert workflows.</p></div>
        <div class="card box"><h3>Scaling Constraints</h3><p>Hospitals face increasing patient loads, staffing shortages, and rising pressure to improve intervention speed.</p></div>
      </div>
    </div>

    <div id="solution" class="section">
      <div class="section-title">The Solution</div>
      <div class="band">
        <div class="quote">
          Early Risk Alert AI centralizes live patient monitoring, AI-assisted risk detection,
          and command-center visibility into one premium clinical intelligence platform.
        </div>
      </div>
    </div>

    <div id="revenue" class="section">
      <div class="section-title">Revenue Model</div>
      <div class="cards3">
        <div class="card box"><h3>Enterprise Platform Licensing</h3><p>Hospitals and health systems subscribe to command-center access, monitoring dashboards, and platform infrastructure.</p></div>
        <div class="card box"><h3>Per-Patient Program Revenue</h3><p>Remote monitoring and distributed care programs create scalable usage-based commercial opportunities.</p></div>
        <div class="card box"><h3>Integration & Deployment Services</h3><p>Additional revenue can be generated through custom integrations, implementation support, and enterprise onboarding.</p></div>
      </div>
    </div>

    <div id="compliance" class="section">
      <div class="section-title">Compliance Positioning</div>
      <div class="cards3">
        <div class="card box"><h3>HIPAA-Ready Architecture</h3><p>Designed with secure healthcare deployment principles suitable for regulated clinical environments.</p></div>
        <div class="card box"><h3>Enterprise Deployment Readiness</h3><p>Built for cloud-based delivery, operational resilience, and future hospital system integration pathways.</p></div>
        <div class="card box"><h3>Trust & Credibility</h3><p>Premium product presentation, live demos, and clear operating model improve institutional confidence during outreach.</p></div>
      </div>
    </div>

    <div id="partnership" class="section">
      <div class="section-title">Hospital Partnership Opportunity</div>
      <div class="band">
        <div class="quote">
          Early Risk Alert AI is positioned for strategic partnerships with hospitals, care networks,
          pilot programs, and investors seeking exposure to modern healthcare intelligence infrastructure.
        </div>
      </div>
    </div>

    <div id="founder" class="section">
      <div class="section-title">Founder</div>
      <div class="cards2">
        <div class="card box">
          <h3>Milton Munroe</h3>
          <p>
            Milton Munroe founded Early Risk Alert AI to advance predictive healthcare through intelligent monitoring infrastructure.
            The company vision centers on earlier detection, stronger operational visibility, and scalable clinical intelligence systems.
          </p>
        </div>
        <div class="card box">
          <h3>Founder Mission</h3>
          <p>
            Early Risk Alert AI represents a broader vision for proactive healthcare: identify patient risk sooner,
            support clinicians with better operational tools, and create more scalable systems for hospital response.
          </p>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="band">
        <div class="section-title" style="margin-top:0">Partner with Early Risk Alert AI</div>
        <div class="quote" style="font-size:22px">
          Early Risk Alert AI is building a premium predictive clinical intelligence platform for hospitals,
          health systems, and remote monitoring programs. The company is open to investor discussions,
          strategic partnerships, and product demonstration conversations.
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:22px">
          <a class="btn" href="#contact">Contact Founder</a>
          <a class="btn secondary" href="/">Open Hospital Command Center</a>
          <a class="btn outline" href="/deck">Download Pitch Deck</a>
        </div>
      </div>
    </div>

    <div id="contact" class="section">
      <div class="section-title">Investor Contact</div>
      <div class="contact-grid">
        <div class="contact-card">
          <div class="contact-label">Founder</div>
          <div class="contact-value">Milton Munroe</div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Email</div>
          <div class="contact-value small">info@earlyriskalertai.com</div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Business Phone</div>
          <div class="contact-value">732-724-7267</div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Request Investor Deck</div>
          <div class="contact-value small">
            <a class="contact-link" href="mailto:info@earlyriskalertai.com?subject=Investor%20Inquiry%20-%20Early%20Risk%20Alert%20AI">info@earlyriskalertai.com</a>
          </div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Pitch Deck Download</div>
          <div class="contact-value small">
            <a class="contact-link" href="/deck">Download Pitch Deck PDF</a>
          </div>
        </div>
        <div class="contact-card">
          <div class="contact-label">Live Demo Access</div>
          <div class="contact-value small">
            <a class="contact-link" href="/">Open Hospital Command Center</a>
          </div>
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

@web_bp.get("/deck")
def deck():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_path = os.path.join(root_dir, "static", "Early_Risk_Alert_AI_Pitch_Deck.pdf")
    if not os.path.exists(pdf_path):
        return f"Pitch deck not found at: {pdf_path}", 404
    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf",
    )
