from pathlib import Path
import re

init_path = Path("era/__init__.py")
cc_path = Path("era/web/command_center.py")

init_s = init_path.read_text(encoding="utf-8")
cc_s = cc_path.read_text(encoding="utf-8")

# 1) Make sure command_center.py has a deck page function available.
if "def command_center_deck_page" not in cc_s:
    deck_block = r'''

# ERA_COMMAND_CENTER_DECK_FALLBACK_START

COMMAND_CENTER_DECK_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Early Risk Alert AI — Governance Deck</title>
  <style>
    body{margin:0;background:#06101d;color:#f7fbff;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
    .wrap{max-width:1200px;margin:0 auto;padding:28px 18px 70px}
    a{color:inherit;text-decoration:none}
    .kicker{color:#b8ffd0;font-size:11px;letter-spacing:.18em;text-transform:uppercase;font-weight:1000}
    h1{font-size:clamp(34px,5vw,60px);line-height:.9;letter-spacing:-.06em;margin:8px 0 12px}
    p,li{color:#d8e3f2;font-size:14px;line-height:1.5;font-weight:760}
    .nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 20px}
    .nav a{border:1px solid rgba(184,211,255,.16);background:rgba(255,255,255,.06);border-radius:999px;padding:8px 11px;font-weight:900;font-size:12px}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
    .card{background:linear-gradient(180deg,rgba(20,34,56,.94),rgba(11,20,35,.94));border:1px solid rgba(184,211,255,.16);border-radius:20px;padding:18px}
    h2{margin:0 0 10px;font-size:22px}
    .guard{border:1px solid rgba(255,211,109,.42);background:rgba(255,211,109,.08);color:#ffe6a2;border-radius:18px;padding:13px;margin:18px 0;font-weight:900}
    @media(max-width:800px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="kicker">HCP-facing decision support • governance deck</div>
    <h1>Pilot Governance + Documentation</h1>
    <p>Governance material is kept separate from the compact live review queue so the Command Center stays tool-like and the documentation remains easy to review.</p>

    <nav class="nav">
      <a href="/command-center">Command Center</a>
      <a href="/pilot-docs">Pilot Docs</a>
      <a href="/model-card">Model Card</a>
      <a href="/pilot-success-guide">Pilot Success Guide</a>
      <a href="/validation-evidence">Evidence Packet</a>
      <a href="/validation-intelligence">Validation Intelligence</a>
    </nav>

    <div class="guard">
      Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
    </div>

    <section class="grid">
      <div class="card">
        <h2>Frozen Intended Use</h2>
        <p>Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.</p>
      </div>

      <div class="card">
        <h2>Pilot Controls</h2>
        <ul>
          <li>Outputs are patient-prioritization support, not clinical directives.</li>
          <li>Review basis, limitations, confidence, freshness, and unknowns remain visible.</li>
          <li>Workflow controls record operational handling and user action logs only.</li>
        </ul>
      </div>

      <div class="card">
        <h2>Approved Claims</h2>
        <ul>
          <li>Supports patient prioritization across monitored patients.</li>
          <li>Provides explainable review-support context.</li>
          <li>Supports command-center workflow awareness and operational visibility.</li>
        </ul>
      </div>

      <div class="card">
        <h2>Claims To Avoid</h2>
        <ul>
          <li>Predicts clinical deterioration.</li>
          <li>Detects crises autonomously.</li>
          <li>Directs bedside intervention.</li>
          <li>Replaces clinician judgment.</li>
        </ul>
      </div>

      <div class="card">
        <h2>Risk Register</h2>
        <ul>
          <li>Claims drift — controlled by frozen intended-use language.</li>
          <li>Explainability over-reliance — controlled by visible limitations and review basis.</li>
          <li>Workflow confusion — controlled by audit/workflow-only action framing.</li>
        </ul>
      </div>

      <div class="card">
        <h2>V&V-Lite Checks</h2>
        <ul>
          <li>Role/unit scope respected.</li>
          <li>Workflow actions remain operational.</li>
          <li>Explainability fields visible.</li>
          <li>Decision-support guardrails visible.</li>
        </ul>
      </div>
    </section>
  </main>
</body>
</html>
"""

def command_center_deck_page():
    return Response(COMMAND_CENTER_DECK_HTML, mimetype="text/html")

# ERA_COMMAND_CENTER_DECK_FALLBACK_END
'''
    if "from flask import" in cc_s and "Response" not in cc_s.split("from flask import",1)[1].split("\n",1)[0]:
        cc_s = re.sub(r"from flask import ([^\n]+)", r"from flask import \1, Response", cc_s, count=1)
    elif "from flask import" not in cc_s:
        cc_s = "from flask import Response\n" + cc_s

    cc_s = cc_s.rstrip() + "\n" + deck_block + "\n"
    cc_path.write_text(cc_s, encoding="utf-8")
    print("Added fallback command_center_deck_page to command_center.py")
else:
    print("command_center_deck_page already exists")

# 2) Remove previous direct route block if present.
init_s = re.sub(
    r"\n\s*# ERA_DIRECT_COMMAND_CENTER_DECK_ROUTE_START.*?# ERA_DIRECT_COMMAND_CENTER_DECK_ROUTE_END\n",
    "\n",
    init_s,
    flags=re.S,
)

# 3) Insert direct app.add_url_rule before the final return app.
matches = list(re.finditer(r"^(\s*)return app\s*$", init_s, flags=re.M))
if not matches:
    raise SystemExit("Could not find 'return app' in era/__init__.py")

m = matches[-1]
indent = m.group(1)

block = f'''
{indent}# ERA_DIRECT_COMMAND_CENTER_DECK_ROUTE_START
{indent}try:
{indent}    from era.web.command_center import command_center_deck_page
{indent}    existing_rules = {{str(rule.rule) for rule in app.url_map.iter_rules()}}
{indent}    if "/command-center/deck" not in existing_rules:
{indent}        app.add_url_rule(
{indent}            "/command-center/deck",
{indent}            "command_center_deck_direct",
{indent}            command_center_deck_page,
{indent}        )
{indent}except Exception as deck_route_error:
{indent}    app.logger.warning("Command Center deck route registration skipped: %s", deck_route_error)
{indent}# ERA_DIRECT_COMMAND_CENTER_DECK_ROUTE_END
'''

init_s = init_s[:m.start()] + block + init_s[m.start():]
init_path.write_text(init_s, encoding="utf-8")

print("Patched era/__init__.py to directly register /command-center/deck")
