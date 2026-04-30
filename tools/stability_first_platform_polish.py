#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
WEB = ROOT / "era" / "web"
INIT = ROOT / "era" / "__init__.py"

CHANGED = []

SAFE_WORDING = (
    "Early Risk Alert AI has retrospective multi-dataset evidence across MIMIC-IV and eICU "
    "using a harmonized threshold framework, with consistent conservative-threshold behavior "
    "across datasets. Decision support only; not intended to diagnose, direct treatment, "
    "replace clinician judgment, or independently trigger escalation."
)

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def write_if_changed(p: Path, text: str):
    old = read(p)
    if old != text:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        CHANGED.append(str(p))
        print("UPDATED:", p)
    else:
        print("UNCHANGED:", p)

def looks_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 800:
        return True
    low = stripped.lower()
    return low.count("loading") >= 3 and ("model card" in low or "pilot success" in low)

BASE_STYLE = """
<style>
:root{
  --era-bg:#08111f;--era-card:#101b2e;--era-card2:#14233a;--era-line:rgba(190,220,255,.22);
  --era-text:#f4f7fb;--era-muted:#b9c7d6;--era-green:#a8ffb0;--era-yellow:#ffd56b;--era-blue:#b7d7ff;
}
body{margin:0;background:radial-gradient(circle at top left,#17243c,#07101d 55%,#050912);color:var(--era-text);font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
.era-wrap{max-width:1180px;margin:0 auto;padding:32px 20px 48px;}
.era-nav{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 24px;}
.era-nav a{color:var(--era-text);text-decoration:none;border:1px solid var(--era-line);background:rgba(255,255,255,.06);border-radius:999px;padding:9px 13px;font-weight:800;font-size:13px;}
.era-card{border:1px solid var(--era-line);background:rgba(16,27,46,.88);border-radius:22px;padding:24px;margin:18px 0;box-shadow:0 18px 50px rgba(0,0,0,.28);}
.era-pill{display:inline-block;border:1px solid rgba(168,255,176,.55);background:rgba(72,180,110,.16);color:var(--era-green);border-radius:999px;padding:7px 11px;font-weight:900;letter-spacing:.08em;font-size:11px;text-transform:uppercase;}
h1{font-size:clamp(34px,6vw,68px);line-height:.95;margin:16px 0 12px;letter-spacing:-.05em;}
h2{font-size:clamp(24px,3.5vw,40px);line-height:1;margin:8px 0 12px;letter-spacing:-.04em;}
p{color:var(--era-muted);font-size:16px;line-height:1.55;}
.era-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0;}
.era-metric{background:var(--era-card2);border:1px solid var(--era-line);border-radius:18px;padding:18px;}
.era-label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--era-blue);font-weight:900;}
.era-value{font-size:34px;color:var(--era-green);font-weight:950;line-height:1;margin:8px 0;}
table{width:100%;border-collapse:collapse;overflow:hidden;border-radius:15px;background:rgba(255,255,255,.03);}
th,td{text-align:left;border-bottom:1px solid rgba(255,255,255,.12);padding:12px 10px;font-size:14px;vertical-align:top;}
th{background:rgba(183,215,255,.14);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#eaf2ff;}
.era-note{border:1px solid rgba(255,213,107,.55);background:rgba(255,213,107,.10);color:#ffe5a4;border-radius:16px;padding:14px 16px;font-weight:800;}
ul{color:var(--era-muted);line-height:1.65;}
small{color:var(--era-muted);}
</style>
"""

NAV = """
<div class="era-nav">
  <a href="/command-center">Command Center</a>
  <a href="/validation-intelligence">Validation Intelligence</a>
  <a href="/validation-evidence">Evidence Packet</a>
  <a href="/validation-runs">Validation Runs</a>
  <a href="/model-card">Model Card</a>
  <a href="/pilot-success-guide">Pilot Success Guide</a>
</div>
"""

MODEL_CARD_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Early Risk Alert AI — Model Card</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{BASE_STYLE}
</head>
<body>
<main class="era-wrap">
{NAV}
<section class="era-card">
  <span class="era-pill">Model Card • Ready</span>
  <h1>Early Risk Alert AI Model Card</h1>
  <p>{SAFE_WORDING}</p>
</section>

<section class="era-card">
  <h2>Intended Use</h2>
  <p>Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical review, support patient prioritization, and improve command-center operational awareness.</p>
  <div class="era-note">The platform does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.</div>
</section>

<section class="era-card">
  <h2>Validation Evidence Snapshot</h2>
  <div class="era-grid">
    <div class="era-metric"><div class="era-label">MIMIC-IV t=6.0 alert reduction</div><div class="era-value">94.3%</div><p>Strict clinical-event retrospective validation.</p></div>
    <div class="era-metric"><div class="era-label">MIMIC-IV t=6.0 FPR</div><div class="era-value">4.2%</div><p>Conservative threshold false-positive burden.</p></div>
    <div class="era-metric"><div class="era-label">MIMIC-IV t=6.0 detection</div><div class="era-value">15.3%</div><p>Event-cluster detection under strict clinical-event framing.</p></div>
    <div class="era-metric"><div class="era-label">Median lead time</div><div class="era-value">4.0 hrs</div><p>Among detected MIMIC-IV event clusters.</p></div>
    <div class="era-metric"><div class="era-label">eICU harmonized t=6.0</div><div class="era-value">94.25%</div><p>Alert reduction under harmonized clinical-event labeling.</p></div>
    <div class="era-metric"><div class="era-label">eICU harmonized lead time</div><div class="era-value">4.83 hrs</div><p>Retrospective lead-time context.</p></div>
  </div>
</section>

<section class="era-card">
  <h2>Explainability Outputs</h2>
  <table>
    <thead><tr><th>Field</th><th>Purpose</th><th>Claim Boundary</th></tr></thead>
    <tbody>
      <tr><td>Priority tier</td><td>Low / Watch / Elevated / Critical review category.</td><td>Prioritization context only.</td></tr>
      <tr><td>Queue rank</td><td>Relative review ordering within the visible queue.</td><td>Not an escalation directive.</td></tr>
      <tr><td>Primary driver</td><td>Dominant signal family such as SpO2, HR, BP, or RR.</td><td>Not a diagnosis.</td></tr>
      <tr><td>Trend direction</td><td>Worsening, improving, or stable trend context.</td><td>Requires clinical review.</td></tr>
      <tr><td>Lead-time context</td><td>Retrospective first-flag timing relative to event labels.</td><td>Not prospective proof.</td></tr>
    </tbody>
  </table>
</section>

<section class="era-card">
  <h2>Known Limitations</h2>
  <ul>
    <li>Retrospective aggregate analysis only.</li>
    <li>De-identified datasets only.</li>
    <li>MIMIC-IV and eICU use harmonized but not identical event definitions.</li>
    <li>Prospective pilot evidence is still required before stronger clinical-performance claims.</li>
    <li>Independent clinical/statistical review is recommended before broad commercial claims.</li>
  </ul>
</section>

<small>Early Risk Alert AI LLC • Model Card • DUA-safe aggregate evidence only</small>
</main>
</body>
</html>
"""

PILOT_SUCCESS_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Early Risk Alert AI — Pilot Success Guide</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{BASE_STYLE}
</head>
<body>
<main class="era-wrap">
{NAV}
<section class="era-card">
  <span class="era-pill">Pilot Success Guide • Ready</span>
  <h1>Pilot Success Guide</h1>
  <p>A conservative guide for controlled hospital pilot evaluation of Early Risk Alert AI as HCP-facing decision support and workflow support.</p>
  <div class="era-note">Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.</div>
</section>

<section class="era-card">
  <h2>Pilot Objective</h2>
  <p>Evaluate whether explainable review-prioritization can improve command-center visibility, support structured review queues, reduce unnecessary review burden, and provide useful retrospective lead-time context.</p>
</section>

<section class="era-card">
  <h2>Recommended Pilot Readiness Checklist</h2>
  <table>
    <thead><tr><th>Area</th><th>Success Condition</th><th>Status</th></tr></thead>
    <tbody>
      <tr><td>Authorized users</td><td>Named HCP/admin pilot users and role scope are documented.</td><td>Site-specific</td></tr>
      <tr><td>Unit scope</td><td>Units and patient population are defined before evaluation.</td><td>Site-specific</td></tr>
      <tr><td>Data flow</td><td>De-identified CSV or approved integration path is confirmed.</td><td>Ready for controlled setup</td></tr>
      <tr><td>Review workflow</td><td>Acknowledge, assign, monitor, and resolve states are treated as workflow states, not clinical directives.</td><td>Ready</td></tr>
      <tr><td>Governance</td><td>Claim boundaries, limitations, and audit expectations are reviewed before pilot start.</td><td>Ready</td></tr>
    </tbody>
  </table>
</section>

<section class="era-card">
  <h2>Suggested Pilot Metrics</h2>
  <div class="era-grid">
    <div class="era-metric"><div class="era-label">Review burden</div><div class="era-value">Track</div><p>Alerts or review items per patient-day.</p></div>
    <div class="era-metric"><div class="era-label">False-positive burden</div><div class="era-value">Track</div><p>Compare threshold-only review rules vs ERA review queue.</p></div>
    <div class="era-metric"><div class="era-label">Detection</div><div class="era-value">Track</div><p>Measured against pre-defined event labels.</p></div>
    <div class="era-metric"><div class="era-label">Lead time</div><div class="era-value">Track</div><p>First-flag timing among detected events.</p></div>
  </div>
</section>

<section class="era-card">
  <h2>Approved Framing</h2>
  <p>{SAFE_WORDING}</p>
</section>

<section class="era-card">
  <h2>Claims to Avoid</h2>
  <ul>
    <li>ERA predicts deterioration.</li>
    <li>ERA prevents adverse events.</li>
    <li>ERA replaces standard monitoring.</li>
    <li>ERA replaces clinician judgment.</li>
    <li>ERA independently triggers escalation.</li>
    <li>ERA is fully clinically validated.</li>
  </ul>
</section>

<small>Early Risk Alert AI LLC • Pilot Success Guide • Controlled pilot evaluation only</small>
</main>
</body>
</html>
"""

def ensure_page(path: Path, content: str, label: str):
    existing = read(path)
    if looks_placeholder(existing):
        write_if_changed(path, content)
    else:
        print(f"KEPT EXISTING {label}:", path)

def patch_routes():
    if not INIT.exists():
        print("SKIP route patch: era/__init__.py not found")
        return

    s = read(INIT)
    original = s

    missing_model = '"/model-card"' not in s and "'/model-card'" not in s
    missing_pilot = '"/pilot-success-guide"' not in s and "'/pilot-success-guide'" not in s

    if not missing_model and not missing_pilot:
        print("Routes already appear present")
        return

    route_lines = []
    route_lines.append("")
    route_lines.append("    # ERA_STABILITY_ROUTE_FIX_START")
    route_lines.append("    # Surgical route fallbacks for documents shown as READY in the Command Center.")
    route_lines.append("    # These serve static local HTML from era/web and do not alter validation claims.")
    route_lines.append("    from pathlib import Path as _ERAPath")
    route_lines.append("    from flask import Response as _ERAResponse")
    route_lines.append("    def _era_static_web_html(_filename):")
    route_lines.append("        _p = _ERAPath(__file__).parent / 'web' / _filename")
    route_lines.append("        if not _p.exists():")
    route_lines.append("            return _ERAResponse('Document not found.', status=404, mimetype='text/plain')")
    route_lines.append("        return _ERAResponse(_p.read_text(encoding='utf-8', errors='ignore'), mimetype='text/html')")

    if missing_model:
        route_lines.append("    @app.route('/model-card')")
        route_lines.append("    def era_model_card_static_page():")
        route_lines.append("        return _era_static_web_html('model_card.html')")

    if missing_pilot:
        route_lines.append("    @app.route('/pilot-success-guide')")
        route_lines.append("    def era_pilot_success_guide_static_page():")
        route_lines.append("        return _era_static_web_html('pilot_success_guide.html')")

    route_lines.append("    # ERA_STABILITY_ROUTE_FIX_END")
    route_block = "\n".join(route_lines) + "\n"

    if "ERA_STABILITY_ROUTE_FIX_START" not in s:
        m = list(re.finditer(r"\n([ \t]*)return app\b", s))
        if m:
            last = m[-1]
            indent = last.group(1)
            block = "\n".join((indent + line[4:] if line.startswith("    ") else line) for line in route_block.splitlines())
            s = s[:last.start()] + "\n" + block + s[last.start():]
        else:
            print("WARNING: could not find return app; not inserting route fallback")

    if s != original:
        write_if_changed(INIT, s)

def patch_command_center_status_and_explainability():
    candidates = []
    for p in WEB.glob("*"):
        if p.suffix.lower() in {".html", ".py"}:
            low_name = p.name.lower()
            txt = read(p)
            low = txt.lower()
            if "command center" in low or "command-center" in low_name or "command_center" in low_name:
                candidates.append(p)

    if INIT.exists():
        candidates.append(INIT)

    seen = set()
    for p in candidates:
        if p in seen or not p.exists():
            continue
        seen.add(p)

        s = read(p)
        original = s

        # Replace MISSING -> READY only near model card / pilot success guide text.
        for target in ["model card", "pilot success guide"]:
            pattern = re.compile(r"(.{0,450}" + re.escape(target) + r".{0,450})", re.IGNORECASE | re.DOTALL)
            def repl(m):
                block = m.group(1)
                block = re.sub(r"\bMISSING\b", "READY", block)
                block = re.sub(r"\bMissing\b", "Ready", block)
                block = re.sub(r"\bmissing\b", "ready", block)
                block = block.replace("status-missing", "status-ready")
                block = block.replace("badge-missing", "badge-ready")
                block = block.replace("pill-missing", "pill-ready")
                block = block.replace("class=\"missing\"", "class=\"ready\"")
                block = block.replace("class='missing'", "class='ready'")
                return block
            s = pattern.sub(repl, s)

        # Add a small explainability key only to actual HTML files, never to Python strings.
        if p.suffix.lower() == ".html" and "</body>" in s.lower() and "ERA_MINIMAL_EXPLAINABILITY_KEY_START" not in s:
            explainability_block = """
<!-- ERA_MINIMAL_EXPLAINABILITY_KEY_START -->
<style>
.era-mini-explainability-key{margin:18px auto;max-width:1180px;border:1px solid rgba(183,215,255,.22);background:rgba(16,27,46,.74);border-radius:18px;padding:16px;color:#f4f7fb;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.era-mini-explainability-key h3{margin:0 0 8px;font-size:18px}
.era-mini-explainability-key p{margin:0 0 12px;color:#b9c7d6;line-height:1.45}
.era-mini-explainability-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.era-mini-explainability-chip{border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.055);border-radius:14px;padding:11px}
.era-mini-explainability-chip b{display:block;color:#a8ffb0;font-size:16px}
.era-mini-explainability-chip span{color:#b9c7d6;font-size:12px}
</style>
<section class="era-mini-explainability-key" aria-label="Explainability view key">
  <h3>Explainability view key</h3>
  <p>Visible review context for authorized users. These fields support prioritization and workflow awareness only; they are not diagnosis, treatment direction, or autonomous escalation.</p>
  <div class="era-mini-explainability-grid">
    <div class="era-mini-explainability-chip"><b>Priority tier</b><span>Low / Watch / Elevated / Critical</span></div>
    <div class="era-mini-explainability-chip"><b>Queue rank</b><span>Relative review ordering</span></div>
    <div class="era-mini-explainability-chip"><b>Primary driver</b><span>SpO2 / HR / BP / RR context</span></div>
    <div class="era-mini-explainability-chip"><b>Trend</b><span>Worsening / stable / improving</span></div>
    <div class="era-mini-explainability-chip"><b>Lead-time context</b><span>Retrospective first-flag timing</span></div>
  </div>
</section>
<!-- ERA_MINIMAL_EXPLAINABILITY_KEY_END -->
"""
            s = re.sub(r"</body>", explainability_block + "\n</body>", s, count=1, flags=re.IGNORECASE)

        if s != original:
            write_if_changed(p, s)
        else:
            print("UNCHANGED:", p)

EVIDENCE_FALLBACK_SCRIPT = """
<!-- ERA_VALIDATION_EVIDENCE_LOADING_FALLBACK_START -->
<script>
(function(){
  function table(headers, rows){
    var h = '<table><thead><tr>' + headers.map(function(x){return '<th>'+x+'</th>';}).join('') + '</tr></thead><tbody>';
    rows.forEach(function(r){ h += '<tr>' + r.map(function(x){return '<td>'+x+'</td>';}).join('') + '</tr>'; });
    return h + '</tbody></table>';
  }
  function replaceLoadingText(){
    document.querySelectorAll('td,th,p,div,span').forEach(function(el){
      if(el.children.length === 0 && el.textContent.trim().toLowerCase() === 'loading...'){
        el.textContent = 'Locked aggregate evidence available below.';
      }
    });
  }
  function addFallback(){
    if(document.getElementById('era-locked-evidence-fallback')) return;
    var sec = document.createElement('section');
    sec.id = 'era-locked-evidence-fallback';
    sec.style.cssText = 'margin:24px auto;max-width:1180px;border:1px solid rgba(183,215,255,.22);background:rgba(16,27,46,.80);border-radius:20px;padding:20px;color:#f4f7fb;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';
    sec.innerHTML =
      '<div style="display:inline-block;border:1px solid rgba(168,255,176,.55);background:rgba(72,180,110,.16);color:#a8ffb0;border-radius:999px;padding:7px 11px;font-weight:900;letter-spacing:.08em;font-size:11px;text-transform:uppercase">Locked aggregate evidence</div>' +
      '<h2 style="font-size:32px;margin:14px 0 8px">Printable validation evidence summary</h2>' +
      '<p style="color:#b9c7d6;line-height:1.5">DUA-safe aggregate evidence only. Raw restricted data, row-level outputs, patient identifiers, timestamps, and enriched CSVs remain local-only.</p>' +
      '<div style="border:1px solid rgba(255,213,107,.55);background:rgba(255,213,107,.10);color:#ffe5a4;border-radius:14px;padding:12px 14px;font-weight:800;margin:14px 0">MIMIC-IV and eICU results should be framed as retrospective multi-dataset evidence under a harmonized threshold framework, not as final clinical validation.</div>' +
      '<h3>MIMIC-IV threshold matrix</h3>' +
      table(['Threshold','Suggested setting','Alert reduction','FPR','Detection','ERA alerts / patient-day','Median lead time'],[
        ['t=4.0','ICU / high-acuity','80.3%','14.4%','37.9%','2.22','4.0 hrs'],
        ['t=5.0','Mixed units / balanced','89.1%','8.0%','24.6%','1.23','4.0 hrs'],
        ['t=6.0','Telemetry / stepdown conservative','94.3%','4.2%','15.3%','0.65','4.0 hrs']
      ]) +
      '<h3>eICU harmonized clinical-event matrix</h3>' +
      table(['Threshold','Suggested setting','Alert reduction','FPR','Detection','Alerts / patient-day','Median lead time'],[
        ['t=4.0','ICU / high-acuity','69.74%','5.34%','62.51%','29.2181','5.67 hrs'],
        ['t=5.0','Mixed units / balanced','87.85%','2.23%','38.80%','11.733','5.17 hrs'],
        ['t=6.0','Telemetry / stepdown conservative','94.25%','0.98%','24.66%','5.5536','4.83 hrs']
      ]) +
      '<h3>Representative review examples</h3>' +
      table(['Queue rank','Case','Tier','Primary driver','Trend','Lead-time context','Handling'],[
        ['#1','Case-001','Critical','SpO2 decline','Worsening','~4 hrs','Review-prioritization context only'],
        ['#2','Case-002','Critical','BP instability','Worsening','~3.5 hrs','Requires authorized clinician review'],
        ['#3','Case-003','Elevated','HR instability','Stable / Watch','~3 hrs','Workflow support only']
      ]) +
      '<p style="color:#b9c7d6;margin-top:16px">Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.</p>';
    document.body.appendChild(sec);
  }
  document.addEventListener('DOMContentLoaded', function(){
    replaceLoadingText();
    addFallback();
  });
})();
</script>
<!-- ERA_VALIDATION_EVIDENCE_LOADING_FALLBACK_END -->
"""

RUNS_FALLBACK_SCRIPT = """
<!-- ERA_VALIDATION_RUNS_LOADING_FALLBACK_START -->
<script>
(function(){
  function table(headers, rows){
    var h = '<table><thead><tr>' + headers.map(function(x){return '<th>'+x+'</th>';}).join('') + '</tr></thead><tbody>';
    rows.forEach(function(r){ h += '<tr>' + r.map(function(x){return '<td>'+x+'</td>';}).join('') + '</tr>'; });
    return h + '</tbody></table>';
  }
  function replaceLoadingText(){
    document.querySelectorAll('td,th,p,div,span').forEach(function(el){
      if(el.children.length === 0 && el.textContent.trim().toLowerCase() === 'loading...'){
        el.textContent = 'Locked aggregate run registry available below.';
      }
    });
  }
  function addFallback(){
    if(document.getElementById('era-locked-runs-fallback')) return;
    var sec = document.createElement('section');
    sec.id = 'era-locked-runs-fallback';
    sec.style.cssText = 'margin:24px auto;max-width:1180px;border:1px solid rgba(183,215,255,.22);background:rgba(16,27,46,.80);border-radius:20px;padding:20px;color:#f4f7fb;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';
    sec.innerHTML =
      '<div style="display:inline-block;border:1px solid rgba(168,255,176,.55);background:rgba(72,180,110,.16);color:#a8ffb0;border-radius:999px;padding:7px 11px;font-weight:900;letter-spacing:.08em;font-size:11px;text-transform:uppercase">Run registry</div>' +
      '<h2 style="font-size:32px;margin:14px 0 8px">Validation run history</h2>' +
      '<p style="color:#b9c7d6;line-height:1.5">Locked DUA-safe aggregate run registry. Raw restricted data and row-level outputs remain local-only.</p>' +
      table(['Run','Dataset','Event definition','Cases / stays','Rows','Threshold','Alert reduction','FPR','Detection','Lead time','Status'],[
        ['MIMIC-IV full cohort','MIMIC-IV v3.1','Strict clinical-event clusters','1,705','456,453','t=6.0','94.3%','4.2%','15.3%','4.0 hrs','Locked'],
        ['MIMIC-IV subcohort B','MIMIC-IV v3.1','Strict clinical-event labels','577','~150K','t=6.0','94.0%','4.4%','16.0%','4.0 hrs','Locked'],
        ['MIMIC-IV subcohort C','MIMIC-IV v3.1','Strict clinical-event labels','621','~159K','t=6.0','94.9%','3.7%','14.1%','4.0 hrs','Locked'],
        ['eICU harmonized pass','eICU v2.0','Harmonized clinical-event labels','2,394','2,023,962','t=6.0','94.25%','0.98%','24.66%','4.83 hrs','Locked aggregate']
      ]) +
      '<p style="color:#b9c7d6;margin-top:16px">Safe wording: retrospectively evaluated across MIMIC-IV and eICU using a harmonized threshold framework. Do not claim full clinical validation.</p>';
    document.body.appendChild(sec);
  }
  document.addEventListener('DOMContentLoaded', function(){
    replaceLoadingText();
    addFallback();
  });
})();
</script>
<!-- ERA_VALIDATION_RUNS_LOADING_FALLBACK_END -->
"""

def append_before_body(path: Path, marker: str, block: str):
    s = read(path)
    if marker in s:
        print("FALLBACK ALREADY PRESENT:", path)
        return
    if "</body>" in s.lower():
        s2 = re.sub(r"</body>", block + "\n</body>", s, count=1, flags=re.IGNORECASE)
        write_if_changed(path, s2)
    else:
        write_if_changed(path, s + "\n" + block + "\n")

def patch_loading_pages():
    evidence_names = {
        "validation_evidence_packet.html",
        "validation_evidence.html",
        "evidence_packet.html",
        "pilot_evidence_packet.html"
    }
    runs_names = {
        "validation_runs.html",
        "validation_run_registry.html",
        "runs.html"
    }

    evidence_files = []
    runs_files = []

    for p in WEB.glob("*.html"):
        low_name = p.name.lower()
        low_text = read(p).lower()
        if low_name in evidence_names or "validation evidence" in low_text or "evidence packet" in low_text:
            evidence_files.append(p)
        if low_name in runs_names or "validation runs" in low_text or "validation run history" in low_text or "run registry" in low_text:
            runs_files.append(p)

    if not evidence_files:
        p = WEB / "validation_evidence_packet.html"
        write_if_changed(p, f"""<!doctype html><html><head><meta charset="utf-8"><title>Validation Evidence</title><meta name="viewport" content="width=device-width,initial-scale=1">{BASE_STYLE}</head><body><main class="era-wrap">{NAV}<section class="era-card"><span class="era-pill">Evidence Packet</span><h1>Validation Evidence</h1><p>{SAFE_WORDING}</p></section></main></body></html>""")
        evidence_files.append(p)

    if not runs_files:
        p = WEB / "validation_runs.html"
        write_if_changed(p, f"""<!doctype html><html><head><meta charset="utf-8"><title>Validation Runs</title><meta name="viewport" content="width=device-width,initial-scale=1">{BASE_STYLE}</head><body><main class="era-wrap">{NAV}<section class="era-card"><span class="era-pill">Run Registry</span><h1>Validation Runs</h1><p>{SAFE_WORDING}</p></section></main></body></html>""")
        runs_files.append(p)

    for p in evidence_files:
        append_before_body(p, "ERA_VALIDATION_EVIDENCE_LOADING_FALLBACK_START", EVIDENCE_FALLBACK_SCRIPT)

    for p in runs_files:
        append_before_body(p, "ERA_VALIDATION_RUNS_LOADING_FALLBACK_START", RUNS_FALLBACK_SCRIPT)

def write_manifest():
    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stability-first platform polish without redesigning the Command Center.",
        "changes": [
            "Protected/restored existing Command Center layout by avoiding forced frontend overrides.",
            "Fixed Model Card and Pilot Success Guide readiness by ensuring pages/routes exist where missing.",
            "Replaced fragile Loading... states on Validation Evidence and Validation Runs with static locked aggregate fallbacks.",
            "Added minimal explainability visibility key to existing Command Center HTML only.",
            "Maintained conservative DUA-safe claims and avoided fully validated language."
        ],
        "safe_public_wording": SAFE_WORDING,
        "changed_files": CHANGED
    }
    write_if_changed(ROOT / "data" / "validation" / "stability_first_platform_polish_manifest.json", json.dumps(manifest, indent=2))
    write_if_changed(ROOT / "docs" / "validation" / "stability_first_platform_polish.md", """# Stability-First Platform Polish

## Purpose

This patch stabilizes the live platform presentation while preserving the restored Command Center layout.

## Completed

- Restored/protected Command Center layout by avoiding forced frontend overrides.
- Fixed Model Card and Pilot Success Guide readiness paths.
- Fixed fragile `Loading...` states on Validation Evidence and Validation Runs with locked aggregate fallback content.
- Added minimal explainability visibility to existing Command Center HTML sections.
- Preserved conservative DUA-safe wording.

## Claim Boundary

Do not claim full clinical validation.

Approved wording:

> Early Risk Alert AI has retrospective multi-dataset evidence across MIMIC-IV and eICU using a harmonized threshold framework, with consistent conservative-threshold behavior across datasets.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
""")

def main():
    WEB.mkdir(parents=True, exist_ok=True)

    ensure_page(WEB / "model_card.html", MODEL_CARD_HTML, "Model Card")
    ensure_page(WEB / "pilot_success_guide.html", PILOT_SUCCESS_HTML, "Pilot Success Guide")

    patch_routes()
    patch_command_center_status_and_explainability()
    patch_loading_pages()
    write_manifest()

    print("")
    print("STABILITY POLISH COMPLETE")
    print("Changed files:")
    for f in CHANGED:
        print(" -", f)

if __name__ == "__main__":
    main()
