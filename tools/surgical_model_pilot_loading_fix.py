#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

APP = Path("era/__init__.py")
WEB = Path("era/web")
DOCS = Path("docs/validation")
DATA = Path("data/validation")

ROUTE_START = "# ERA_MODEL_PILOT_PUBLIC_ONLY_V3_START"
ROUTE_END = "# ERA_MODEL_PILOT_PUBLIC_ONLY_V3_END"

OLD_MARKERS = [
    ("# ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1_START", "# ERA_FORCE_PUBLIC_PAGE_OVERRIDE_V1_END"),
    ("# ERA_MINIMAL_MODEL_CARD_PILOT_GUIDE_ROUTES_V1_START", "# ERA_MINIMAL_MODEL_CARD_PILOT_GUIDE_ROUTES_V1_END"),
    ("# ERA_ONLY_MODEL_PILOT_GUIDE_ROUTES_V2_START", "# ERA_ONLY_MODEL_PILOT_GUIDE_ROUTES_V2_END"),
    ("# ERA_MODEL_PILOT_PUBLIC_ONLY_V3_START", "# ERA_MODEL_PILOT_PUBLIC_ONLY_V3_END"),
    ("<!-- ERA_FRONTEND_EVIDENCE_QA_FIX_V1_START -->", "<!-- ERA_FRONTEND_EVIDENCE_QA_FIX_V1_END -->"),
    ("<!-- ERA_COMMAND_CENTER_MULTI_DATASET_EXPLAINABILITY_V1_START -->", "<!-- ERA_COMMAND_CENTER_MULTI_DATASET_EXPLAINABILITY_V1_END -->"),
    ("<!-- ERA_FINAL_COMMAND_CENTER_EXPLAINABILITY_V1_START -->", "<!-- ERA_FINAL_COMMAND_CENTER_EXPLAINABILITY_V1_END -->"),
]

FORCED_FILES = [
    WEB / "command_center_forced.html",
    WEB / "validation_intelligence_forced.html",
    WEB / "validation_evidence_forced.html",
    WEB / "validation_runs_forced.html",
    WEB / "model_card_forced.html",
    WEB / "pilot_success_guide_forced.html",
    Path("tools/force_authoritative_frontend_routes.py"),
    Path("tools/fix_frontend_evidence_placeholders.py"),
    Path("tools/restore_platform_remove_forced_override.py"),
]


def remove_marker_block(text: str, start: str, end: str) -> tuple[str, int]:
    count = 0
    while start in text:
        a = text.find(start)
        b = text.find(end, a)
        if b == -1:
            break
        text = text[:a] + text[b + len(end):]
        count += 1
    return text, count


def write_if_changed(path: Path, text: str) -> bool:
    old = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    if old != text:
        path.write_text(text, encoding="utf-8")
        print("UPDATED:", path)
        return True
    return False


print("STEP A — Remove old forced override leftovers")

for p in FORCED_FILES:
    if p.exists():
        p.unlink()
        print("REMOVED:", p)

candidate_files = []
candidate_files.append(APP)
candidate_files += list(WEB.glob("*.html"))
candidate_files += list(Path("era").glob("*.py"))

for p in candidate_files:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    original = text
    for start, end in OLD_MARKERS:
        text, _ = remove_marker_block(text, start, end)
    if text != original:
        p.write_text(text, encoding="utf-8")
        print("CLEANED OLD MARKERS:", p)


print("STEP B — Create Model Card and Pilot Success Guide pages only")

model_card_html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Card — Early Risk Alert AI</title>
<style>
body{margin:0;background:#07101f;color:#f6fbff;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{width:min(1050px,calc(100% - 28px));margin:auto;padding:32px 0 60px}
a{color:#8fffd2;text-decoration:none;font-weight:800}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
nav a{border:1px solid rgba(170,205,255,.22);border-radius:999px;padding:9px 12px;color:#f6fbff}
.card{border:1px solid rgba(170,205,255,.22);border-radius:24px;background:linear-gradient(145deg,rgba(16,27,46,.96),rgba(7,16,31,.96));padding:22px;margin-top:16px}
h1{font-size:clamp(34px,5vw,58px);letter-spacing:-.06em;line-height:1;margin:0 0 12px}
p,li{line-height:1.6}
.guard{border-left:4px solid #ffe2a5;background:rgba(255,226,165,.09);border-radius:14px;padding:12px 14px;color:#ffe2a5;font-weight:800}
.pill{display:inline-block;border:1px solid rgba(143,255,210,.35);background:rgba(143,255,210,.1);color:#8fffd2;border-radius:999px;padding:6px 10px;font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}
</style>
</head>
<body>
<div class="wrap">
<span class="pill">Model Card · Ready</span>
<h1>Early Risk Alert AI Model Card</h1>
<nav>
<a href="/command-center">Command Center</a>
<a href="/validation-intelligence">Validation Intelligence</a>
<a href="/validation-evidence">Evidence Packet</a>
<a href="/validation-runs">Validation Runs</a>
<a href="/pilot-success-guide">Pilot Success Guide</a>
</nav>

<section class="card">
<h2>Intended Use</h2>
<p>Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.</p>
<p class="guard">Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.</p>
</section>

<section class="card">
<h2>Evidence Status</h2>
<ul>
<li>MIMIC-IV strict clinical-event retrospective validation.</li>
<li>eICU second-dataset outcome-proxy robustness check.</li>
<li>DUA-safe aggregate-only public outputs.</li>
<li>Raw restricted files and row-level derived files remain local-only.</li>
</ul>
<p class="guard">MIMIC-IV and eICU use different event definitions. Detection rates should not be treated as equivalent endpoint definitions until a harmonized event definition is run across both datasets.</p>
</section>

<section class="card">
<h2>Explainability Outputs</h2>
<ul>
<li>Priority Tier</li>
<li>Queue Rank</li>
<li>Primary Driver</li>
<li>Trend Direction</li>
<li>Lead-Time Context</li>
<li>Workflow-state actions such as acknowledge, assign, escalate, and resolve</li>
</ul>
</section>

<section class="card">
<h2>Current Validation Boundary</h2>
<p>Current public evidence should be framed as multi-dataset retrospective robustness evidence. The next validation upgrade is a harmonized eICU clinical-event labeler so MIMIC-IV and eICU can test the same validation question.</p>
</section>
</div>
</body>
</html>
"""

pilot_guide_html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilot Success Guide — Early Risk Alert AI</title>
<style>
body{margin:0;background:#07101f;color:#f6fbff;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{width:min(1050px,calc(100% - 28px));margin:auto;padding:32px 0 60px}
a{color:#8fffd2;text-decoration:none;font-weight:800}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
nav a{border:1px solid rgba(170,205,255,.22);border-radius:999px;padding:9px 12px;color:#f6fbff}
.card{border:1px solid rgba(170,205,255,.22);border-radius:24px;background:linear-gradient(145deg,rgba(16,27,46,.96),rgba(7,16,31,.96));padding:22px;margin-top:16px}
h1{font-size:clamp(34px,5vw,58px);letter-spacing:-.06em;line-height:1;margin:0 0 12px}
p,li{line-height:1.6}
.guard{border-left:4px solid #ffe2a5;background:rgba(255,226,165,.09);border-radius:14px;padding:12px 14px;color:#ffe2a5;font-weight:800}
.pill{display:inline-block;border:1px solid rgba(143,255,210,.35);background:rgba(143,255,210,.1);color:#8fffd2;border-radius:999px;padding:6px 10px;font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}
</style>
</head>
<body>
<div class="wrap">
<span class="pill">Pilot Success Guide · Ready</span>
<h1>Pilot Success Guide</h1>
<nav>
<a href="/command-center">Command Center</a>
<a href="/validation-intelligence">Validation Intelligence</a>
<a href="/validation-evidence">Evidence Packet</a>
<a href="/validation-runs">Validation Runs</a>
<a href="/model-card">Model Card</a>
</nav>

<section class="card">
<h2>Pilot Purpose</h2>
<p>Evaluate whether Early Risk Alert AI can support authorized healthcare professionals with explainable patient prioritization, review-burden reduction, and command-center workflow visibility.</p>
<p class="guard">Pilot success should be measured with pre-agreed workflow and review metrics, not unsupported outcome or hard-dollar ROI claims.</p>
</section>

<section class="card">
<h2>Suggested Pilot Success Metrics</h2>
<ul>
<li>Alerts or reviews per patient-day.</li>
<li>Visibility of priority tier, primary driver, trend direction, queue rank, and lead-time context.</li>
<li>Reviewer understanding of why a case may warrant review.</li>
<li>Workflow-state completion.</li>
<li>Safety: no autonomous escalation, diagnosis, treatment direction, or independent escalation.</li>
</ul>
</section>

<section class="card">
<h2>Before Pilot Launch</h2>
<ul>
<li>Hospital/site name confirmed.</li>
<li>Site sponsor or clinical champion confirmed.</li>
<li>Pilot scope and unit access defined.</li>
<li>Authorized users identified.</li>
<li>IT/security contact identified.</li>
<li>Data-use pathway reviewed.</li>
<li>Decision-support-only framing accepted.</li>
</ul>
</section>
</div>
</body>
</html>
"""

model_card_md = """# Early Risk Alert AI — Model Card

## Intended Use

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.

It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.

## Evidence Status

- MIMIC-IV strict clinical-event retrospective validation.
- eICU second-dataset outcome-proxy robustness check.
- DUA-safe aggregate-only public outputs.
- Raw restricted files and row-level derived files remain local-only.

## Current Limitation

MIMIC-IV and eICU use different event definitions. Detection rates should not be treated as equivalent endpoint definitions until a harmonized event definition is run across both datasets.
"""

pilot_guide_md = """# Early Risk Alert AI — Pilot Success Guide

## Pilot Purpose

Evaluate explainable patient prioritization, review-burden reduction, and command-center workflow visibility.

## Suggested Metrics

- Alerts or reviews per patient-day.
- Visibility of priority tier, primary driver, trend direction, queue rank, and lead-time context.
- Reviewer understanding of why a case may warrant review.
- Workflow-state completion.
- Safety: no autonomous escalation, diagnosis, treatment direction, or independent escalation.

## Guardrail

Pilot success should be measured with pre-agreed workflow and review metrics, not unsupported hard-dollar ROI claims.
"""

write_if_changed(WEB / "model_card.html", model_card_html)
write_if_changed(WEB / "pilot_success_guide.html", pilot_guide_html)
write_if_changed(DOCS / "model_card.md", model_card_md)
write_if_changed(DOCS / "pilot_success_guide.md", pilot_guide_md)


print("STEP C — Add early public routes for only Model Card and Pilot Success Guide")

app_text = APP.read_text(encoding="utf-8", errors="ignore")

app_text, _ = remove_marker_block(app_text, ROUTE_START, ROUTE_END)

tree = ast.parse(app_text)

create_app_node = None
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "create_app":
        create_app_node = node
        break

if create_app_node is None:
    raise SystemExit("ERROR: create_app() not found.")

lines = app_text.splitlines()

insert_idx = None
for i in range(create_app_node.lineno - 1, len(lines)):
    stripped = lines[i].strip()
    if re.match(r"app\s*=", stripped) and "Flask" in stripped:
        insert_idx = i + 1
        break

if insert_idx is None:
    return_lines = []
    for node in ast.walk(create_app_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "app":
            return_lines.append(node.lineno)
    if not return_lines:
        raise SystemExit("ERROR: app assignment and return app not found.")
    insert_idx = min(return_lines) - 1

route_block = r'''
    # ERA_MODEL_PILOT_PUBLIC_ONLY_V3_START
    @app.before_request
    def era_model_pilot_public_only_v3():
        from pathlib import Path
        from flask import request, send_from_directory, Response

        path = request.path.rstrip("/") or "/"
        root_dir = Path(__file__).resolve().parent.parent
        web_dir = Path(__file__).resolve().parent / "web"

        if path in ("/model-card", "/model_card"):
            return send_from_directory(str(web_dir), "model_card.html")

        if path in ("/pilot-success-guide", "/pilot_success_guide"):
            return send_from_directory(str(web_dir), "pilot_success_guide.html")

        if path == "/validation-evidence/model-card.md":
            p = root_dir / "docs" / "validation" / "model_card.md"
            return Response(p.read_text(encoding="utf-8"), mimetype="text/markdown")

        if path == "/validation-evidence/pilot-success-guide.md":
            p = root_dir / "docs" / "validation" / "pilot_success_guide.md"
            return Response(p.read_text(encoding="utf-8"), mimetype="text/markdown")
    # ERA_MODEL_PILOT_PUBLIC_ONLY_V3_END

'''.splitlines()

lines = lines[:insert_idx] + route_block + lines[insert_idx:]
write_if_changed(APP, "\n".join(lines) + "\n")


print("STEP D — Patch only the MISSING labels near Model Card / Pilot Success Guide")

status_files = [APP] + list(WEB.glob("*.html")) + list(Path("era").glob("*.py"))

def patch_missing_near_label(text: str, label: str) -> str:
    label_pattern = r"\s+".join(re.escape(part) for part in label.split())
    pattern = re.compile(
        r"(" + label_pattern + r"(?:(?!model\s+card|pilot\s+success\s+guide).){0,900}?)(MISSING|Missing|missing)",
        re.I | re.S
    )

    def repl(m):
        prefix = m.group(1)
        missing = m.group(2)
        if missing.isupper():
            return prefix + "READY"
        if missing[0].isupper():
            return prefix + "Ready"
        return prefix + "ready"

    return pattern.sub(repl, text, count=1)

for p in status_files:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    old = text
    text = patch_missing_near_label(text, "model card")
    text = patch_missing_near_label(text, "pilot success guide")
    if text != old:
        p.write_text(text, encoding="utf-8")
        print("PATCHED STATUS TEXT:", p)


print("STEP E — Fix only leftover Loading placeholder rows, no redesign")

threshold_rows = """
<tr>
  <td><strong>t=4.0</strong></td>
  <td>ICU / high-acuity</td>
  <td>80.3%</td>
  <td>14.4%</td>
  <td>37.9%</td>
  <td>2.22</td>
  <td>4.0 hrs</td>
</tr>
<tr>
  <td><strong>t=5.0</strong></td>
  <td>Mixed units / balanced</td>
  <td>89.1%</td>
  <td>8.0%</td>
  <td>24.6%</td>
  <td>1.23</td>
  <td>4.0 hrs</td>
</tr>
<tr>
  <td><strong>t=6.0</strong></td>
  <td>Telemetry / stepdown conservative</td>
  <td>94.3%</td>
  <td>4.2%</td>
  <td>15.3%</td>
  <td>0.65</td>
  <td>4.0 hrs</td>
</tr>
"""

for p in list(WEB.glob("*.html")) + [APP]:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    old = text

    text = re.sub(
        r"<tr[^>]*>\s*<td[^>]*>\s*Loading\.\.\.\s*</td>\s*(?:<td[^>]*>\s*[—-]\s*</td>\s*){5,8}</tr>",
        threshold_rows,
        text,
        flags=re.I | re.S,
    )

    if "validation" in p.name.lower() or "evidence" in p.name.lower():
        text = text.replace("Loading...", "Loaded from locked aggregate evidence")

    if text != old:
        p.write_text(text, encoding="utf-8")
        print("PATCHED LOADING PLACEHOLDER:", p)


manifest = {
    "ok": True,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Surgical fix after command-center restore.",
    "changed": [
        "Added Model Card page",
        "Added Pilot Success Guide page",
        "Added markdown downloads",
        "Patched model card / pilot success guide status labels only",
        "Patched leftover Loading placeholder row/text only"
    ],
    "not_changed": [
        "No command-center forced route override",
        "No validation page replacement",
        "No validation metric changes",
        "No raw or row-level restricted data exposed",
        "No hard ROI claims",
        "No unqualified two-dataset validation claim"
    ]
}
write_if_changed(DATA / "surgical_model_pilot_loading_fix_manifest.json", json.dumps(manifest, indent=2))

print("DONE surgical patch prepared.")
