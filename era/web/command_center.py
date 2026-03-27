from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import NotFound


web_bp = Blueprint("web", __name__)

APP_NAME = "Early Risk Alert AI"
INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
DECK_FILENAME = "Early_Risk_Alert_AI_Pitch_Deck.pdf"
DECK_ALIASES = (
    "/deck",
    "/pitch-deck",
    "/hospital-deck",
    "/investor-deck",
    "/pilot-deck",
)


BASE_CSS = """
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
    --green:#39d39f;
    --amber:#f4bd6a;
    --red:#ff667d;
    --shadow:0 20px 60px rgba(0,0,0,.34);
    --radius:24px;
    --max:1380px;
  }
  *{box-sizing:border-box}
  body{
    margin:0;
    font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:radial-gradient(circle at top, #10203d 0%, var(--bg) 35%, #050b14 100%);
    color:var(--text);
  }
  a{color:inherit;text-decoration:none}
  .wrap{max-width:var(--max);margin:0 auto;padding:28px 18px 64px}
  .nav{
    display:flex;align-items:center;justify-content:space-between;gap:16px;
    margin-bottom:22px;
  }
  .brand{
    display:flex;align-items:center;gap:12px;font-weight:800;letter-spacing:.08em;
    text-transform:uppercase;font-size:13px;color:#dbe7ff;
  }
  .brand-badge{
    width:42px;height:42px;border-radius:14px;
    background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
    display:grid;place-items:center;color:#07101c;font-weight:900;box-shadow:var(--shadow);
  }
  .nav-links{display:flex;flex-wrap:wrap;gap:10px}
  .chip{
    padding:10px 14px;border-radius:999px;background:rgba(255,255,255,.05);
    border:1px solid var(--line);color:var(--muted);font-weight:700;font-size:13px;
  }
  .hero{
    background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
    border:1px solid var(--line);border-radius:30px;padding:34px;box-shadow:var(--shadow);
    margin-bottom:20px;
  }
  .eyebrow{
    color:#d7e6ff;font-size:13px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;
    margin-bottom:14px;
  }
  h1,h2,h3{margin:0}
  h1{font-size:clamp(34px,5vw,66px);line-height:0.96;letter-spacing:-.04em;max-width:920px}
  .lede{margin-top:18px;max-width:980px;color:var(--muted);font-size:18px;line-height:1.65}
  .banner-grid{display:grid;grid-template-columns:1fr;gap:14px;margin-top:22px}
  .callout{
    padding:18px 20px;border-radius:20px;border:1px solid var(--line);
    background:rgba(122,162,255,.08);color:#eef4ff;font-weight:700;line-height:1.55;
  }
  .callout.warn{background:rgba(244,189,106,.08);border-color:rgba(244,189,106,.24);color:#ffe6ba}
  .cta-row{display:flex;flex-wrap:wrap;gap:12px;margin-top:24px}
  .btn{
    display:inline-flex;align-items:center;justify-content:center;gap:10px;
    padding:14px 18px;border-radius:16px;border:1px solid var(--line);
    background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;font-weight:900;
    box-shadow:var(--shadow);
  }
  .btn.secondary{background:rgba(255,255,255,.04);color:var(--text)}
  .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px;margin-top:20px}
  .card{
    grid-column:span 12;background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02));
    border:1px solid var(--line);border-radius:24px;padding:22px;box-shadow:var(--shadow);
  }
  .card h2{font-size:28px;letter-spacing:-.03em;margin-bottom:10px}
  .card p{color:var(--muted);line-height:1.6}
  .mini-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:16px}
  .mini{
    padding:16px;border:1px solid var(--line2);background:rgba(255,255,255,.03);
    border-radius:18px;
  }
  .mini strong{display:block;margin-bottom:6px}
  .list{display:grid;gap:10px;margin-top:16px}
  .row{
    padding:14px 16px;border-radius:14px;background:rgba(255,255,255,.03);
    border:1px solid var(--line2);color:#dfe9ff;
  }
  .footer{
    margin-top:20px;padding:18px 0 0;color:var(--muted);font-size:14px;border-top:1px solid var(--line);
  }
  .form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:18px}
  label{display:block;font-size:13px;font-weight:800;margin-bottom:8px;color:#dbe7ff}
  input,select,textarea{
    width:100%;padding:14px 16px;border-radius:14px;border:1px solid var(--line);
    background:#0b1528;color:var(--text);font:inherit;
  }
  textarea{min-height:120px;resize:vertical}
  .full{grid-column:1/-1}
  .helper{font-size:13px;color:var(--muted);margin-top:6px}
  .status{
    display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;
    background:rgba(57,211,159,.12);border:1px solid rgba(57,211,159,.25);color:#baf7de;font-weight:800;
    font-size:13px;
  }
  .two-up{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:16px}
  @media (max-width:920px){
    .mini-grid,.form-grid,.two-up{grid-template-columns:1fr}
  }
</style>
"""


COMMAND_CENTER_HTML = BASE_CSS + """
<div class="wrap">
  <div class="nav">
    <div class="brand">
      <div class="brand-badge">ER</div>
      <div>{{ app_name }}</div>
    </div>
    <div class="nav-links">
      <a class="chip" href="{{ url_for('web.command_center') }}">Command Center</a>
      <a class="chip" href="{{ url_for('web.pilot_docs') }}">Pilot Docs</a>
      <a class="chip" href="{{ url_for('web.deck') }}">Pitch Deck</a>
      <a class="chip" href="{{ url_for('web.hospital_demo') }}">Hospital Demo</a>
      <a class="chip" href="{{ url_for('web.investor_intake') }}">Investor Access</a>
    </div>
  </div>

  <section class="hero">
    <div class="eyebrow">{{ app_name }} · HCP-Facing Clinical Command Center</div>
    <h1>Supporting Monitored Patient Visibility</h1>
    <div class="lede">
      {{ app_name }} is an HCP-facing decision-support and workflow-support software platform intended to assist authorized
      health care professionals in identifying patients who may warrant further clinical evaluation, supporting patient prioritization,
      and improving command-center operational awareness.
    </div>
    <div class="banner-grid">
      <div class="callout">
        {{ app_name }} is positioned for controlled pilot evaluation and hospital-facing workflow support with visible review basis,
        confidence, limitations, data freshness, and workflow-state separation.
      </div>
      <div class="callout warn">
        The platform does not replace clinician judgment and is not intended to diagnose, direct treatment,
        or independently trigger escalation.
      </div>
    </div>
    <div class="cta-row">
      <a class="btn" href="{{ url_for('web.deck') }}">Open Pitch Deck</a>
      <a class="btn secondary" href="{{ url_for('web.pilot_docs') }}">Open Pilot Docs</a>
      <a class="btn secondary" href="{{ url_for('web.hospital_demo') }}">Request Live Demo</a>
      <a class="btn secondary" href="{{ url_for('web.executive_walkthrough') }}">Executive Walkthrough</a>
      <a class="btn secondary" href="{{ url_for('web.investor_intake') }}">Investor Access</a>
    </div>
  </section>

  <section class="grid">
    <article class="card" style="grid-column:span 7;">
      <h2>Clinical Command Center</h2>
      <p>
        Real-time monitored patient visibility, review priority summary, supportive review context,
        and operational workflow tracking in one disciplined pilot-ready interface.
      </p>
      <div class="mini-grid">
        <div class="mini"><strong>Monitored Patient Visibility</strong>Live visibility across monitored care environments.</div>
        <div class="mini"><strong>Review Priority Summary</strong>Supportive review ordering across monitored patients.</div>
        <div class="mini"><strong>Workflow Tracking</strong>Status and progress monitoring kept separate from clinical review.</div>
        <div class="mini"><strong>Supportive Review Context</strong>Contextual information with explainability and limitations.</div>
      </div>
    </article>

    <article class="card" style="grid-column:span 5;">
      <h2>Current Deck Route</h2>
      <p>The public deck route now points to the current file in <code>static/{{ deck_filename }}</code>.</p>
      <div class="list">
        <div class="row"><strong>/deck</strong>Primary public route</div>
        <div class="row"><strong>/pitch-deck</strong>Alias route</div>
        <div class="row"><strong>/hospital-deck</strong>Alias route</div>
        <div class="row"><strong>/investor-deck</strong>Alias route</div>
        <div class="row"><strong>/pilot-deck</strong>Legacy alias route</div>
      </div>
    </article>

    <article class="card" style="grid-column:span 12;">
      <h2>Deck + Pilot Readiness</h2>
      <div class="two-up">
        <div>
          <div class="status">Current deck connected</div>
          <div class="list">
            <div class="row">Pitch deck buttons now route directly to the current PDF file.</div>
            <div class="row">Old deck aliases can stay active so older links do not break.</div>
            <div class="row">The command center links, pilot docs, and intake routes all stay connected from one file.</div>
          </div>
        </div>
        <div>
          <div class="status">Governance posture visible</div>
          <div class="list">
            <div class="row">Conservative HCP-facing decision-support and workflow-support language</div>
            <div class="row">Review basis, confidence, limitations, freshness, and workflow separation</div>
            <div class="row">Controlled pilot evaluation posture without diagnostic or treatment claims</div>
          </div>
        </div>
      </div>
      <div class="footer">
        Contact: {{ info_email }} · {{ business_phone }}
      </div>
    </article>
  </section>
</div>
"""


PILOT_DOCS_HTML = BASE_CSS + """
<div class="wrap">
  <div class="nav">
    <div class="brand"><div class="brand-badge">ER</div><div>Pilot Docs</div></div>
    <div class="nav-links">
      <a class="chip" href="{{ url_for('web.command_center') }}">Back to Command Center</a>
      <a class="chip" href="{{ url_for('web.deck') }}">Open Pitch Deck</a>
    </div>
  </div>

  <section class="hero">
    <div class="eyebrow">Stable Pilot Positioning Bundle</div>
    <h1>Governance and Reviewability</h1>
    <div class="lede">
      Freeze one intended-use statement everywhere, keep outputs supportive rather than directive,
      and keep explainability, limitations, scoping, and audit visibility easy to review.
    </div>
  </section>

  <section class="grid">
    <article class="card" style="grid-column:span 12;">
      <h2>Frozen Intended Use</h2>
      <p>
        {{ app_name }} is an HCP-facing decision-support and workflow-support software platform intended to assist authorized
        health care professionals in identifying patients who may warrant further clinical evaluation, supporting patient prioritization,
        and improving command-center operational awareness.
      </p>
      <div class="callout warn" style="margin-top:16px;">
        It does not replace clinician judgment and is not intended to diagnose, direct treatment,
        or independently trigger escalation.
      </div>
    </article>

    <article class="card" style="grid-column:span 6;">
      <h2>Approved Language</h2>
      <div class="list">
        <div class="row">Near-Term Review Context</div>
        <div class="row">Supportive Review Visibility</div>
        <div class="row">Review Priority Summary</div>
        <div class="row">Workflow Tracking Status</div>
      </div>
    </article>

    <article class="card" style="grid-column:span 6;">
      <h2>Banned or Avoid Claims</h2>
      <div class="list">
        <div class="row">Priority Review Forecast</div>
        <div class="row">Forecast Attention 99%</div>
        <div class="row">Anything implying the software decides urgency on its own</div>
        <div class="row">Anything implying diagnosis, treatment direction, or autonomous escalation</div>
      </div>
    </article>
  </section>
</div>
"""


FORM_HTML = BASE_CSS + """
<div class="wrap">
  <div class="nav">
    <div class="brand"><div class="brand-badge">ER</div><div>{{ title }}</div></div>
    <div class="nav-links">
      <a class="chip" href="{{ url_for('web.command_center') }}">Command Center</a>
      <a class="chip" href="{{ url_for('web.deck') }}">Pitch Deck</a>
      <a class="chip" href="{{ url_for('web.pilot_docs') }}">Pilot Docs</a>
    </div>
  </div>

  <section class="hero">
    <div class="eyebrow">{{ app_name }}</div>
    <h1>{{ title }}</h1>
    <div class="lede">{{ subtitle }}</div>
    <div class="banner-grid">
      <div class="callout">
        {{ app_name }} is an HCP-facing decision-support and workflow-support platform intended to assist authorized
        health care professionals in identifying patients who may warrant further clinical evaluation,
        supporting patient prioritization, and improving command-center operational awareness.
      </div>
      <div class="callout warn">
        The platform is intended for controlled pilot evaluation and hospital-facing workflow support.
        It does not replace clinician judgment and is not intended to diagnose, direct treatment,
        or independently trigger escalation.
      </div>
    </div>
  </section>

  <section class="card">
    {% if submitted %}
      <div class="status">Request captured</div>
      <p style="margin-top:16px;">
        Thank you. Your request was received for controlled follow-up. This placeholder page confirms the route works.
      </p>
      <div class="list">
        <div class="row"><strong>Name</strong> {{ payload.get('full_name', '') }}</div>
        <div class="row"><strong>Email</strong> {{ payload.get('work_email', '') }}</div>
        <div class="row"><strong>Organization</strong> {{ payload.get('organization', '') }}</div>
      </div>
      <div class="cta-row">
        <a class="btn" href="{{ url_for('web.command_center') }}">Back to Command Center</a>
        <a class="btn secondary" href="{{ url_for('web.deck') }}">Open Pitch Deck</a>
      </div>
    {% else %}
      <form method="post">
        <div class="form-grid">
          <div>
            <label>Full Name</label>
            <input name="full_name" placeholder="Enter your full name" required>
          </div>
          <div>
            <label>Organization</label>
            <input name="organization" placeholder="Enter your organization" required>
          </div>
          <div>
            <label>Work Email</label>
            <input name="work_email" type="email" placeholder="Enter your work email" required>
          </div>
          <div>
            <label>Role</label>
            <input name="role" placeholder="Enter your role">
          </div>
          <div>
            <label>Timeline</label>
            <select name="timeline">
              <option>Immediate</option>
              <option>30-60 Days</option>
              <option>This Quarter</option>
              <option>Exploratory</option>
            </select>
          </div>
          <div>
            <label>Focus Area</label>
            <select name="interest">
              <option>Monitored patient visibility</option>
              <option>Patient prioritization support</option>
              <option>Command-center workflow support</option>
              <option>Platform overview</option>
            </select>
          </div>
          <div class="full">
            <label>Notes</label>
            <textarea name="message" placeholder="Share your goals, pilot interests, or discussion notes"></textarea>
            <div class="helper">This placeholder form keeps the route live and connected while you continue refining the full workflow.</div>
          </div>
        </div>
        <div class="cta-row">
          <button class="btn" type="submit">Submit Request</button>
          <a class="btn secondary" href="{{ url_for('web.command_center') }}">Cancel</a>
        </div>
      </form>
    {% endif %}
  </section>
</div>
"""


def _static_root() -> Path:
    static_folder = current_app.static_folder
    if not static_folder:
        raise NotFound("Static folder is not configured.")
    return Path(static_folder)


def _send_static_pdf(filename: str) -> Response:
    static_root = _static_root()
    file_path = static_root / filename
    if not file_path.exists():
        raise NotFound(f"Missing PDF in static/: {filename}")
    return send_from_directory(static_root, filename)


@web_bp.get("/")
def home() -> Response:
    return redirect(url_for("web.command_center"))


@web_bp.get("/command-center")
def command_center() -> str:
    return render_template_string(
        COMMAND_CENTER_HTML,
        app_name=APP_NAME,
        deck_filename=DECK_FILENAME,
        info_email=INFO_EMAIL,
        business_phone=BUSINESS_PHONE,
    )


@web_bp.get("/pilot-docs")
def pilot_docs() -> str:
    return render_template_string(PILOT_DOCS_HTML, app_name=APP_NAME)


@web_bp.get("/deck")
def deck() -> Response:
    return _send_static_pdf(DECK_FILENAME)


@web_bp.get("/pitch-deck")
def pitch_deck() -> Response:
    return deck()


@web_bp.get("/hospital-deck")
def hospital_deck() -> Response:
    return deck()


@web_bp.get("/investor-deck")
def investor_deck() -> Response:
    return deck()


@web_bp.get("/pilot-deck")
def pilot_deck() -> Response:
    return deck()


@web_bp.route("/hospital-demo", methods=["GET", "POST"])
def hospital_demo() -> str:
    payload: Dict[str, Any] = request.form.to_dict() if request.method == "POST" else {}
    return render_template_string(
        FORM_HTML,
        app_name=APP_NAME,
        title="Request a Live Command Center Demonstration",
        subtitle="Schedule a guided demonstration for monitored patient visibility, supportive review context, and command-center workflow support.",
        submitted=request.method == "POST",
        payload=payload,
    )


@web_bp.route("/executive-walkthrough", methods=["GET", "POST"])
def executive_walkthrough() -> str:
    payload: Dict[str, Any] = request.form.to_dict() if request.method == "POST" else {}
    return render_template_string(
        FORM_HTML,
        app_name=APP_NAME,
        title="Request an Executive Walkthrough",
        subtitle="Request a higher-level review of platform positioning, pilot readiness, governance, and hospital-facing deployment posture.",
        submitted=request.method == "POST",
        payload=payload,
    )


@web_bp.route("/investor-intake", methods=["GET", "POST"])
def investor_intake() -> str:
    payload: Dict[str, Any] = request.form.to_dict() if request.method == "POST" else {}
    return render_template_string(
        FORM_HTML,
        app_name=APP_NAME,
        title="Request Investor Access",
        subtitle="Request investor materials, platform overview, and controlled partnership discussion access.",
        submitted=request.method == "POST",
        payload=payload,
    )


@web_bp.get("/api/platform-positioning")
def platform_positioning() -> Response:
    return jsonify(
        {
            "app_name": APP_NAME,
            "intended_use": (
                "HCP-facing decision-support and workflow-support software platform intended to assist authorized health care "
                "professionals in identifying patients who may warrant further clinical evaluation, supporting patient prioritization, "
                "and improving command-center operational awareness."
            ),
            "limitations": [
                "Does not replace clinician judgment.",
                "Not intended to diagnose, direct treatment, or independently trigger escalation.",
                "Controlled pilot evaluation posture.",
            ],
            "approved_terms": [
                "Near-Term Review Context",
                "Supportive Review Visibility",
                "Review Priority Summary",
                "Workflow Tracking Status",
            ],
            "deck_route": url_for("web.deck", _external=False),
            "pilot_docs_route": url_for("web.pilot_docs", _external=False),
        }
    )
