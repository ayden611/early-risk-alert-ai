#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

APP = Path("era/__init__.py")
WEB = Path("era/web")
DATA = Path("data/validation")
DOCS = Path("docs/validation")

ROUTE_START = "# ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1_START"
ROUTE_END = "# ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1_END"

STAMP = "ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1"

CSS = """
<style>
:root{--bg:#07101f;--panel:#101b2e;--line:rgba(170,205,255,.22);--text:#f6fbff;--muted:#b8c6d8;--green:#8fffd2;--gold:#ffe2a5;--red:#ff9da8}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 10% 0%,rgba(139,184,255,.15),transparent 28%),var(--bg);color:var(--text);font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{width:min(1180px,calc(100% - 28px));margin:0 auto;padding:24px 0 56px}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--green);font-weight:950}
h1{font-size:clamp(36px,6vw,68px);letter-spacing:-.07em;line-height:.95;margin:8px 0 14px}
h2{font-size:clamp(24px,3vw,36px);letter-spacing:-.05em;margin:0 0 12px}
p{line-height:1.55}
a{color:#eaf4ff;text-decoration:none;font-weight:850}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}
nav a{border:1px solid rgba(170,205,255,.18);background:rgba(255,255,255,.06);border-radius:999px;padding:9px 12px;font-size:12px}
.card,.hero{border:1px solid var(--line);background:linear-gradient(145deg,rgba(16,27,46,.96),rgba(7,16,31,.96));border-radius:28px;padding:24px;box-shadow:0 22px 65px rgba(0,0,0,.28);margin-top:18px}
.grid{display:grid;gap:14px}.cols4{grid-template-columns:repeat(4,1fr)}.cols3{grid-template-columns:repeat(3,1fr)}.cols2{grid-template-columns:repeat(2,1fr)}
.metric{border:1px solid rgba(170,205,255,.18);background:rgba(255,255,255,.055);border-radius:20px;padding:18px}
.metric span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.13em;font-weight:950}
.metric strong{display:block;font-size:clamp(26px,4vw,44px);margin-top:8px;color:var(--green);letter-spacing:-.06em}
.metric em{display:block;color:var(--muted);font-style:normal;font-size:12px;margin-top:6px}
.callout{border-left:4px solid var(--green);background:rgba(143,255,210,.09);border-radius:16px;padding:14px 16px;font-weight:850}
.guardrail{border:1px solid rgba(255,226,165,.35);background:rgba(255,226,165,.09);color:var(--gold);border-radius:18px;padding:14px 16px;font-weight:780}
table{width:100%;border-collapse:collapse;border-radius:18px;overflow:hidden;border:1px solid rgba(170,205,255,.2);background:rgba(255,255,255,.035);margin-top:12px}
th,td{text-align:left;padding:12px 14px;border-bottom:1px solid rgba(170,205,255,.14);vertical-align:top;font-size:14px}
th{background:rgba(139,184,255,.12);font-size:11px;text-transform:uppercase;letter-spacing:.12em}
tr:last-child td{border-bottom:0}
.pill{display:inline-flex;padding:6px 10px;border-radius:999px;border:1px solid rgba(143,255,210,.35);background:rgba(143,255,210,.11);color:var(--green);font-size:11px;font-weight:950;letter-spacing:.08em;text-transform:uppercase}
.red{border-color:rgba(255,157,168,.35);background:rgba(255,157,168,.12);color:var(--red)}
.muted{color:var(--muted)}
.footer{margin-top:24px;color:var(--muted);font-size:12px}
@media(max-width:900px){.cols4,.cols3,.cols2{grid-template-columns:1fr}table{display:block;overflow-x:auto;white-space:nowrap}}
</style>
"""

NAV = """
<nav>
  <a href="/command-center">Command Center</a>
  <a href="/validation-intelligence">Validation Intelligence</a>
  <a href="/validation-evidence">Evidence Packet</a>
  <a href="/validation-runs">Validation Runs</a>
  <a href="/model-card">Model Card</a>
  <a href="/pilot-success-guide">Pilot Success Guide</a>
</nav>
"""

TOP_LINE = "Early Risk Alert AI has retrospective evidence across two de-identified ICU datasets: MIMIC-IV strict clinical-event cross-cohort validation plus a separate eICU outcome-proxy check."
BEHAVIOR = "Across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives."
GUARDRAIL = "MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context."

def html(title: str, eyebrow: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Early Risk Alert AI</title>
<meta name="era-page-stamp" content="{STAMP}">
{CSS}
</head>
<body data-era-stamp="{STAMP}">
<div class="wrap">
<div class="eyebrow">Early Risk Alert AI · {eyebrow}</div>
<h1>{title}</h1>
{NAV}
{body}
<div class="footer">Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.</div>
</div>
</body>
</html>
"""

mimic_rows = """
<tr><td><strong>t=4.0</strong></td><td>ICU / high-acuity</td><td>80.3%</td><td>14.4%</td><td>37.9%</td><td>2.22</td><td>4.0 hrs</td></tr>
<tr><td><strong>t=5.0</strong></td><td>Mixed units / balanced</td><td>89.1%</td><td>8.0%</td><td>24.6%</td><td>1.23</td><td>4.0 hrs</td></tr>
<tr><td><strong>t=6.0</strong></td><td>Telemetry / stepdown conservative</td><td>94.3%</td><td>4.2%</td><td>15.3%</td><td>0.65</td><td>4.0 hrs</td></tr>
"""

eicu_rows = """
<tr><td><strong>t=4.0</strong></td><td>ICU / high-acuity</td><td>87.7%</td><td>7.3%</td><td>81.3%</td><td>—</td><td>4.89 hrs</td></tr>
<tr><td><strong>t=5.0</strong></td><td>Mixed units / balanced</td><td>93.5%</td><td>3.7%</td><td>75.1%</td><td>—</td><td>4.32 hrs</td></tr>
<tr><td><strong>t=6.0</strong></td><td>Telemetry / stepdown conservative</td><td>96.8%</td><td>1.8%</td><td>66.6%</td><td>—</td><td>3.41 hrs</td></tr>
"""

validation_intelligence = html("Validation Intelligence", "Retrospective Evidence", f"""
<section class="hero">
  <span class="pill">Published · DUA-safe aggregate evidence</span>
  <h2 style="margin-top:18px">Multi-dataset retrospective robustness story</h2>
  <p class="callout">{TOP_LINE}</p>
  <p><strong>{BEHAVIOR}</strong></p>
  <p class="muted">At the conservative t=6.0 setting, MIMIC-IV showed 4.0 hrs median lead-time context across the locked cross-cohort release, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.</p>
  <div class="grid cols4">
    <div class="metric"><span>MIMIC-IV t=6.0 Alert Reduction</span><strong>94%–94.9%</strong><em>Strict clinical-event evidence</em></div>
    <div class="metric"><span>MIMIC-IV t=6.0 FPR</span><strong>3.7%–4.4%</strong><em>Cross-cohort release</em></div>
    <div class="metric"><span>eICU t=6.0 Alert Reduction</span><strong>96.8%</strong><em>Outcome-proxy check</em></div>
    <div class="metric"><span>Lead-Time Context</span><strong>3.4–4.0 hrs</strong><em>Retrospective timing context</em></div>
  </div>
</section>
<section class="card">
  <h2>Side-by-Side Conservative Operating Point</h2>
  <p class="muted">The strongest public story is consistent threshold-direction behavior, not direct equality between different event definitions.</p>
  <table>
    <thead><tr><th>Dataset</th><th>Evidence Role</th><th>Event Definition</th><th>Cases</th><th>Rows</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th></tr></thead>
    <tbody>
      <tr><td><strong>MIMIC-IV v3.1</strong></td><td>Strict clinical-event cross-cohort validation release</td><td>Strict clinical-event labels / event clusters</td><td>577–1,705</td><td>456,453 in full validation cohort</td><td>94%–94.9%</td><td>3.7%–4.4%</td><td>14.1%–16%</td><td>4 hrs</td></tr>
      <tr><td><strong>eICU v2.0</strong></td><td>Second-dataset outcome-proxy check</td><td>Mortality/discharge-derived outcome proxy</td><td>2,394</td><td>2,023,962</td><td>96.8%</td><td>1.8%</td><td>66.6%</td><td>3.41 hrs</td></tr>
    </tbody>
  </table>
  <p class="guardrail">{GUARDRAIL}</p>
</section>
<section class="card">
  <h2>Explainability Context</h2>
  <p class="muted">The command-center display should make prioritization understandable without relying on raw score alone.</p>
  <div class="grid cols4">
    <div class="metric"><span>Priority Tier</span><strong>Critical</strong><em>Low / Watch / Elevated / Critical</em></div>
    <div class="metric"><span>Queue Rank</span><strong>#1</strong><em>Relative review ordering</em></div>
    <div class="metric"><span>Primary Driver</span><strong>SpO₂</strong><em>SpO₂ / HR / BP / RR</em></div>
    <div class="metric"><span>Trend</span><strong>Worsening</strong><em>Context for review</em></div>
  </div>
</section>
""")

validation_evidence = html("Pilot Evidence Packet", "Validation Evidence", f"""
<section class="hero">
  <span class="pill">No Loading Placeholders · Locked Aggregate Evidence</span>
  <h2 style="margin-top:18px">Printable validation evidence summary</h2>
  <p class="callout">{TOP_LINE}</p>
  <p><strong>{BEHAVIOR}</strong></p>
  <p class="guardrail">{GUARDRAIL}</p>
</section>
<section class="card">
  <h2>MIMIC-IV Threshold Matrix</h2>
  <p class="muted">Strict clinical-event retrospective validation. Conservative t=6.0 is optimized for low review burden.</p>
  <table>
    <thead><tr><th>Threshold</th><th>Suggested Setting</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>ERA Alerts / Patient-Day</th><th>Median Lead Time</th></tr></thead>
    <tbody>{mimic_rows}</tbody>
  </table>
</section>
<section class="card">
  <h2>eICU Threshold Matrix</h2>
  <p class="muted">Second-dataset retrospective outcome-proxy check using mortality/discharge-derived event context.</p>
  <table>
    <thead><tr><th>Threshold</th><th>Suggested Setting</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>ERA Alerts / Patient-Day</th><th>Median Lead Time</th></tr></thead>
    <tbody>{eicu_rows}</tbody>
  </table>
</section>
<section class="card">
  <h2>Representative Review Examples</h2>
  <p class="muted">Sanitized examples only. No real restricted IDs, no row-level timestamps, and no raw MIMIC/eICU data.</p>
  <table>
    <thead><tr><th>Queue Rank</th><th>Case</th><th>Tier</th><th>Primary Driver</th><th>Trend</th><th>Lead-Time Context</th><th>Meaning</th></tr></thead>
    <tbody>
      <tr><td>#1</td><td>Case-001</td><td>Critical</td><td>SpO₂ decline</td><td>Worsening</td><td>~4 hrs</td><td>May warrant prioritized HCP review</td></tr>
      <tr><td>#2</td><td>Case-002</td><td>Critical</td><td>BP instability</td><td>Worsening</td><td>~3.5 hrs</td><td>Review queue prioritization context</td></tr>
      <tr><td>#3</td><td>Case-003</td><td>Elevated</td><td>HR instability</td><td>Stable / watch</td><td>~3 hrs</td><td>Monitor trend and reassess</td></tr>
    </tbody>
  </table>
</section>
<section class="card">
  <h2>Downloads</h2>
  <ul>
    <li><a href="/api/validation/multi-dataset-checkpoint">Locked multi-dataset checkpoint JSON</a></li>
    <li><a href="/api/validation/multi-dataset-public-framing">Multi-dataset public framing JSON</a></li>
    <li><a href="/validation-evidence/model-card.md">Model Card markdown</a></li>
    <li><a href="/validation-evidence/pilot-success-guide.md">Pilot Success Guide markdown</a></li>
  </ul>
</section>
""")

validation_runs = html("Validation Runs", "Run Registry", f"""
<section class="hero">
  <span class="pill">Run Registry</span>
  <h2 style="margin-top:18px">Validation run history</h2>
  <p class="callout">{BEHAVIOR}</p>
</section>
<section class="card">
  <h2>Current Locked Runs</h2>
  <table>
    <thead><tr><th>Run</th><th>Dataset</th><th>Event Definition</th><th>Cases</th><th>Rows</th><th>Threshold</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th><th>Status</th></tr></thead>
    <tbody>
      <tr><td><strong>MIMIC-IV full cohort</strong></td><td>MIMIC-IV v3.1</td><td>Strict clinical-event labels</td><td>1,705</td><td>456,453</td><td>t=6.0</td><td>94.3%</td><td>4.2%</td><td>15.3%</td><td>4.0 hrs</td><td><span class="pill">Locked</span></td></tr>
      <tr><td><strong>MIMIC-IV subcohort B</strong></td><td>MIMIC-IV v3.1</td><td>Strict clinical-event labels</td><td>577</td><td>~159K</td><td>t=6.0</td><td>94.0%</td><td>4.4%</td><td>16.0%</td><td>4.0 hrs</td><td><span class="pill">Locked</span></td></tr>
      <tr><td><strong>MIMIC-IV subcohort C</strong></td><td>MIMIC-IV v3.1</td><td>Strict clinical-event labels</td><td>621</td><td>~159K</td><td>t=6.0</td><td>94.9%</td><td>3.7%</td><td>14.1%</td><td>4.0 hrs</td><td><span class="pill">Locked</span></td></tr>
      <tr><td><strong>eICU outcome-proxy cohort</strong></td><td>eICU v2.0</td><td>Mortality/discharge-derived outcome proxy</td><td>2,394</td><td>2,023,962</td><td>t=6.0</td><td>96.8%</td><td>1.8%</td><td>66.6%</td><td>3.41 hrs</td><td><span class="pill">Outcome-proxy check</span></td></tr>
    </tbody>
  </table>
</section>
<section class="card">
  <h2>Checkpoint</h2>
  <p><span class="pill">Locked</span></p>
  <p><strong>multi-dataset-retrospective-robustness-checkpoint-2026-04-30</strong></p>
  <p class="muted">MIMIC-IV strict clinical-event evidence and eICU second-dataset outcome-proxy evidence are locked as the current validation checkpoint.</p>
</section>
""")

command_center = html("Command Center", "Pilot Demo", f"""
<section class="hero">
  <span class="pill">Pilot Mode · Frontend QA Corrected</span>
  <h2 style="margin-top:18px">Explainable rules-based command-center platform</h2>
  <p class="callout">This demo emphasizes patient-prioritization context: tier, queue rank, primary driver, trend, and lead-time context.</p>
  <div class="grid cols4">
    <div class="metric"><span>Route Status</span><strong>Ready</strong><em>Core routes available</em></div>
    <div class="metric"><span>Model Card</span><strong>Ready</strong><em><a href="/model-card">Open Model Card</a></em></div>
    <div class="metric"><span>Pilot Success Guide</span><strong>Ready</strong><em><a href="/pilot-success-guide">Open Guide</a></em></div>
    <div class="metric"><span>Evidence Packet</span><strong>Ready</strong><em><a href="/validation-evidence">Open Evidence</a></em></div>
  </div>
</section>
<section class="card">
  <h2>Hospital Command Wall</h2>
  <p class="muted">Sanitized demonstration data only. Shows review prioritization, not diagnosis or treatment direction.</p>
  <table>
    <thead><tr><th>Queue</th><th>Patient</th><th>Priority Tier</th><th>Score</th><th>Primary Driver</th><th>Trend</th><th>Lead-Time Context</th><th>Workflow State</th></tr></thead>
    <tbody>
      <tr><td>#1</td><td>Demo-001</td><td>Critical</td><td>9.2</td><td>SpO₂ decline</td><td>Worsening</td><td>~4 hrs</td><td>Needs review</td></tr>
      <tr><td>#2</td><td>Demo-002</td><td>Critical</td><td>8.7</td><td>BP instability</td><td>Worsening</td><td>~3.5 hrs</td><td>Acknowledged</td></tr>
      <tr><td>#3</td><td>Demo-003</td><td>Elevated</td><td>7.9</td><td>HR instability</td><td>Stable / watch</td><td>~3 hrs</td><td>Assigned</td></tr>
      <tr><td>#4</td><td>Demo-004</td><td>Watch</td><td>6.4</td><td>RR elevation</td><td>Stable</td><td>—</td><td>Monitoring</td></tr>
    </tbody>
  </table>
</section>
<section class="card">
  <h2>MIMIC Validation Intelligence</h2>
  <div class="grid cols4">
    <div class="metric"><span>Alert Reduction</span><strong>94.3%</strong><em>t=6.0 conservative</em></div>
    <div class="metric"><span>ERA FPR</span><strong>4.2%</strong><em>Low false-positive burden</em></div>
    <div class="metric"><span>Detection</span><strong>15.3%</strong><em>Strict clinical-event labels</em></div>
    <div class="metric"><span>Median Lead Time</span><strong>4 hrs</strong><em>Among detected event clusters</em></div>
  </div>
</section>
<section class="card">
  <h2>Multi-Dataset Robustness</h2>
  <p class="guardrail">{GUARDRAIL}</p>
</section>
""")

model_card = html("Model Card", "Governance", f"""
<section class="hero"><span class="pill">Ready</span><h2 style="margin-top:18px">Model disclosure</h2><p class="callout">Early Risk Alert AI is HCP-facing decision support and workflow support.</p><p class="guardrail">No diagnosis, no treatment direction, no autonomous escalation.</p></section>
<section class="card"><h2>Evidence Status</h2><table><thead><tr><th>Dataset</th><th>Role</th><th>Event Definition</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th></tr></thead><tbody><tr><td>MIMIC-IV</td><td>Strict clinical-event validation</td><td>Clinical-event labels</td><td>94%–94.9%</td><td>3.7%–4.4%</td><td>14.1%–16%</td><td>4 hrs</td></tr><tr><td>eICU</td><td>Outcome-proxy check</td><td>Mortality/discharge proxy</td><td>96.8%</td><td>1.8%</td><td>66.6%</td><td>3.41 hrs</td></tr></tbody></table></section>
""")

pilot_guide = html("Pilot Success Guide", "Pilot Readiness", f"""
<section class="hero"><span class="pill">Ready</span><h2 style="margin-top:18px">Pilot success framework</h2><p class="callout">Measure explainable prioritization, review burden, workflow visibility, and HCP usability.</p><p class="guardrail">Avoid unsupported hard ROI or clinical outcome claims before site-specific pilot data.</p></section>
<section class="card"><h2>Suggested Success Metrics</h2><table><thead><tr><th>Area</th><th>Metric</th></tr></thead><tbody><tr><td>Review burden</td><td>Alerts or reviews per patient-day</td></tr><tr><td>Prioritization clarity</td><td>Tier, driver, trend, queue rank, and lead-time context visible</td></tr><tr><td>Workflow</td><td>Acknowledge, assign, escalate, resolve workflow states</td></tr><tr><td>Safety</td><td>No autonomous escalation, diagnosis, or treatment direction</td></tr></tbody></table></section>
""")

pages = {
    "validation_intelligence_forced.html": validation_intelligence,
    "validation_evidence_forced.html": validation_evidence,
    "validation_runs_forced.html": validation_runs,
    "command_center_forced.html": command_center,
    "model_card_forced.html": model_card,
    "pilot_success_guide_forced.html": pilot_guide,
}

for name, content in pages.items():
    (WEB / name).write_text(content, encoding="utf-8")
    print("WROTE:", WEB / name)

manifest = {
    "ok": True,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "stamp": STAMP,
    "purpose": "Force public routes to serve corrected frontend files before older route handlers.",
    "routes_forced": {
        "/validation-intelligence": "validation_intelligence_forced.html",
        "/validation-evidence": "validation_evidence_forced.html",
        "/validation-runs": "validation_runs_forced.html",
        "/command-center": "command_center_forced.html",
        "/model-card": "model_card_forced.html",
        "/pilot-success-guide": "pilot_success_guide_forced.html",
    },
    "claim_boundary": "Cross-dataset retrospective robustness evidence across de-identified ICU datasets; decision support only.",
}
(DATA / "authoritative_frontend_override_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

DOCS.joinpath("authoritative_frontend_override.md").write_text(
    f"""# Authoritative Frontend Override

Generated: {datetime.now(timezone.utc).isoformat()}

## Purpose

The backend evidence was correct, but the live app was still serving older frontend route content on some URLs.

This patch forces the public routes to serve corrected frontend files before older route handlers can respond.

## Forced Routes

- `/validation-intelligence`
- `/validation-evidence`
- `/validation-runs`
- `/command-center`
- `/model-card`
- `/pilot-success-guide`

## What This Fixes

- Removes stale `Loading...` placeholders.
- Makes Model Card and Pilot Success Guide show as ready.
- Makes Command Center visibly show priority tier, queue rank, primary driver, trend, and lead-time context.
- Keeps MIMIC-IV and eICU event definitions separate.

## Not Added

- No hard ROI claims.
- No proven generalizability claim.
- No direct equality claim between MIMIC-IV and eICU detection rates.
- No diagnosis, treatment, prevention, or autonomous escalation claims.
""",
    encoding="utf-8",
)

app_text = APP.read_text(encoding="utf-8")

while ROUTE_START in app_text:
    a = app_text.find(ROUTE_START)
    b = app_text.find(ROUTE_END, a)
    if b == -1:
        raise SystemExit("Route override marker start without end.")
    app_text = app_text[:a] + app_text[b + len(ROUTE_END):]

tree = ast.parse(app_text)
create_app_node = None
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "create_app":
        create_app_node = node
        break

if create_app_node is None:
    raise SystemExit("ERROR: create_app() not found.")

return_lines = []
for node in ast.walk(create_app_node):
    if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "app":
        return_lines.append(node.lineno)

if not return_lines:
    raise SystemExit("ERROR: return app not found inside create_app().")

insert_line = max(return_lines)

override_block = r'''
    # ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1_START
    @app.before_request
    def era_force_public_page_override_v1():
        from pathlib import Path
        from flask import request, send_from_directory

        forced_pages = {
            "/validation-intelligence": "validation_intelligence_forced.html",
            "/validation-evidence": "validation_evidence_forced.html",
            "/validation-runs": "validation_runs_forced.html",
            "/command-center": "command_center_forced.html",
            "/model-card": "model_card_forced.html",
            "/pilot-success-guide": "pilot_success_guide_forced.html",
        }

        path = request.path.rstrip("/") or "/"
        if path in forced_pages:
            web_dir = Path(__file__).resolve().parent / "web"
            return send_from_directory(str(web_dir), forced_pages[path])
    # ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1_END

'''

lines = app_text.splitlines()
lines = lines[:insert_line - 1] + override_block.splitlines() + lines[insert_line - 1:]
APP.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("PATCHED:", APP)
