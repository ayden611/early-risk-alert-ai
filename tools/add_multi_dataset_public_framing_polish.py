#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

DATA = Path("data/validation")
DOCS = Path("docs/validation")
WEB = Path("era/web")
APP_FILE = Path("era/__init__.py")
README = Path("README.md")

HTML_START = "<!-- ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_V1_START -->"
HTML_END = "<!-- ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_V1_END -->"

CC_START = "<!-- ERA_COMMAND_CENTER_MULTI_DATASET_EXPLAINABILITY_V1_START -->"
CC_END = "<!-- ERA_COMMAND_CENTER_MULTI_DATASET_EXPLAINABILITY_V1_END -->"

README_START = "<!-- ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_README_V1_START -->"
README_END = "<!-- ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_README_V1_END -->"

ROUTE_START = "# ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_ROUTES_V1_START"
ROUTE_END = "# ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_ROUTES_V1_END"


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def num(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def fmt(x, suffix=""):
    n = num(x)
    if n is None:
        return "—"
    if abs(n - round(n)) < 0.001:
        return f"{n:.0f}{suffix}"
    return f"{n:.2f}".rstrip("0").rstrip(".") + suffix


def marker_replace(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


multi = load_json(DATA / "multi_dataset_robustness_summary.json")
eicu_summary = load_json(DATA / "eicu_validation_summary.json")
cross = load_json(DATA / "cross_cohort_validation_summary.json")

datasets = multi.get("datasets", {})
mimic = datasets.get("mimic_iv", {})
eicu = datasets.get("eicu", {})

mimic_alert = mimic.get("alert_reduction", "94.0%–94.9%")
mimic_fpr = mimic.get("fpr", "3.7%–4.4%")
mimic_detection = mimic.get("detection", "14.1%–16.0%")
mimic_lead = mimic.get("lead_time", "4.0 hrs")
mimic_cases = mimic.get("cases", "577–1,705")
mimic_rows = mimic.get("rows", "456,453 in full validation cohort")

eicu_alert = eicu.get("alert_reduction", "96.8%")
eicu_fpr = eicu.get("fpr", "1.8%")
eicu_detection = eicu.get("detection", "66.6%")
eicu_lead = eicu.get("lead_time", "3.41 hrs")
eicu_cases = eicu.get("cases", "2,394")
eicu_rows = eicu.get("rows", "2,023,962")
eicu_events = eicu.get("events", "772")

TOP_LINE = (
    "Early Risk Alert AI now has retrospective evidence across two de-identified ICU datasets: "
    "MIMIC-IV strict clinical-event cross-cohort validation plus a separate eICU outcome-proxy check."
)

BEHAVIOR_LINE = (
    "Across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, "
    "while conservative thresholds reduced review burden and false positives."
)

GUARDRAIL_LINE = (
    "MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses "
    "stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context."
)

LEAD_LINE = (
    f"At the conservative t=6.0 setting, MIMIC-IV showed {mimic_lead} median lead-time context across the locked "
    f"cross-cohort release, while eICU showed {eicu_lead} median lead-time context in the outcome-proxy check."
)

public_payload = {
    "ok": True,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "title": "Multi-Dataset Public Framing Polish",
    "top_line": TOP_LINE,
    "behavior_line": BEHAVIOR_LINE,
    "lead_time_context": LEAD_LINE,
    "event_definition_guardrail": GUARDRAIL_LINE,
    "mimic_iv": {
        "evidence_role": "Strict clinical-event cross-cohort retrospective validation release",
        "cases": mimic_cases,
        "rows": mimic_rows,
        "event_definition": "Strict clinical-event labels / event clusters",
        "threshold": "t=6.0",
        "alert_reduction": mimic_alert,
        "fpr": mimic_fpr,
        "detection": mimic_detection,
        "lead_time": mimic_lead,
    },
    "eicu": {
        "evidence_role": "Second-dataset retrospective outcome-proxy check",
        "cases": eicu_cases,
        "rows": eicu_rows,
        "event_definition": "Outcome-proxy labels from mortality/discharge-derived event context",
        "event_proxy_rows": eicu_events,
        "threshold": "t=6.0",
        "alert_reduction": eicu_alert,
        "fpr": eicu_fpr,
        "detection": eicu_detection,
        "lead_time": eicu_lead,
    },
    "approved_claim": "Cross-dataset retrospective robustness evidence across de-identified ICU datasets.",
    "avoid_claims": [
        "validated on two datasets without qualification",
        "proven generalizability",
        "prospective validation",
        "ERA predicts deterioration",
        "ERA detects crises early",
        "ERA prevents adverse events",
        "ERA diagnoses",
        "ERA directs treatment",
        "ERA replaces clinician judgment",
        "ERA independently triggers escalation",
        "direct superiority over standard monitoring"
    ],
    "public_policy": "Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.",
    "notice": "Decision support only. Retrospective aggregate analysis only."
}

(DATA / "multi_dataset_public_framing_polish.json").write_text(json.dumps(public_payload, indent=2), encoding="utf-8")


HTML_BLOCK = f"""
{HTML_START}
<section class="card full" style="
  border:1px solid rgba(143,255,210,.34);
  border-left:5px solid #8fffd2;
  background:linear-gradient(145deg, rgba(13,24,42,.95), rgba(6,11,21,.95));
  border-radius:28px;
  padding:24px;
  margin-top:18px;
">
  <div style="color:#8fffd2;text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:950;margin-bottom:8px;">
    Multi-Dataset Robustness — Public Framing
  </div>

  <h2 style="margin:0 0 12px;">Retrospective evidence now spans MIMIC-IV and eICU.</h2>

  <p class="callout" style="
    border-left:4px solid #8fffd2;
    padding:12px 16px;
    border-radius:14px;
    background:rgba(143,255,210,.09);
    color:#eaf4ff;
    font-weight:850;
  ">
    {TOP_LINE}
  </p>

  <p class="muted"><strong>{BEHAVIOR_LINE}</strong></p>
  <p class="muted">{LEAD_LINE}</p>

  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Evidence Role</th>
        <th>Event Definition</th>
        <th>Cases</th>
        <th>Rows</th>
        <th>t=6.0 Alert Reduction</th>
        <th>t=6.0 FPR</th>
        <th>t=6.0 Detection</th>
        <th>Lead-Time Context</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>MIMIC-IV v3.1</strong></td>
        <td>Strict clinical-event cross-cohort validation release</td>
        <td>Clinical-event labels / event clusters</td>
        <td>{mimic_cases}</td>
        <td>{mimic_rows}</td>
        <td>{mimic_alert}</td>
        <td>{mimic_fpr}</td>
        <td>{mimic_detection}</td>
        <td>{mimic_lead}</td>
      </tr>
      <tr>
        <td><strong>eICU v2.0</strong></td>
        <td>Second-dataset outcome-proxy check</td>
        <td>Mortality/discharge-derived outcome proxy</td>
        <td>{eicu_cases}</td>
        <td>{eicu_rows}</td>
        <td>{eicu_alert}</td>
        <td>{eicu_fpr}</td>
        <td>{eicu_detection}</td>
        <td>{eicu_lead}</td>
      </tr>
    </tbody>
  </table>

  <div style="
    margin-top:14px;
    padding:12px 14px;
    border-radius:16px;
    border:1px solid rgba(255,215,138,.34);
    background:rgba(255,215,138,.09);
    color:#ffe7b7;
    font-weight:780;
    line-height:1.5;
  ">
    Interpretation guardrail: {GUARDRAIL_LINE}
  </div>

  <p class="muted" style="margin-top:14px;">
    Approved framing: cross-dataset retrospective robustness evidence across de-identified ICU datasets. Avoid claims of proven generalizability, prospective validation, diagnosis, treatment direction, prevention, autonomous escalation, or replacement of clinician judgment.
  </p>

  <div class="era-release-lock-links" style="margin-top:12px;">
    <a href="/api/validation/multi-dataset-public-framing">Public Framing JSON</a>
    <a href="/validation-evidence/multi-dataset-pilot-summary.md">Pilot One-Pager</a>
    <a href="/api/validation/multi-dataset-checkpoint">Locked Checkpoint</a>
  </div>
</section>
{HTML_END}
"""

CC_BLOCK = f"""
{CC_START}
<section id="era-command-center-explainability-alignment" class="era-release-lock-panel" style="
  width:min(1180px, calc(100% - 28px));
  margin:16px auto;
  border:1px solid rgba(139,184,255,.34);
  border-left:5px solid #8bb8ff;
  border-radius:24px;
  background:linear-gradient(145deg, rgba(13,24,42,.96), rgba(6,11,21,.96));
  box-shadow:0 22px 65px rgba(0,0,0,.32);
  color:#f7fbff;
  padding:18px;
  position:relative;
  z-index:5;
">
  <div style="color:#8bb8ff;text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:950;margin-bottom:6px;">
    Command Center Explainability Alignment
  </div>

  <h2 style="margin:0;font-size:clamp(20px,2.5vw,30px);letter-spacing:-.04em;">
    Live demo aligned to multi-dataset robustness evidence.
  </h2>

  <p style="margin:10px 0 0;color:#dbe8fb;font-size:14px;line-height:1.55;font-weight:750;">
    {BEHAVIOR_LINE} The live view should emphasize prioritization context, not raw score alone.
  </p>

  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:14px;">
    <div style="border:1px solid rgba(155,190,255,.22);background:rgba(255,255,255,.055);border-radius:16px;padding:12px;">
      <span style="display:block;color:#b8c6d8;font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:900;">Priority Tier</span>
      <b style="display:block;font-size:20px;margin-top:8px;">Critical / Elevated / Watch</b>
    </div>
    <div style="border:1px solid rgba(155,190,255,.22);background:rgba(255,255,255,.055);border-radius:16px;padding:12px;">
      <span style="display:block;color:#b8c6d8;font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:900;">Queue Rank</span>
      <b style="display:block;font-size:20px;margin-top:8px;">#1, #2, #3</b>
    </div>
    <div style="border:1px solid rgba(155,190,255,.22);background:rgba(255,255,255,.055);border-radius:16px;padding:12px;">
      <span style="display:block;color:#b8c6d8;font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:900;">Primary Driver</span>
      <b style="display:block;font-size:20px;margin-top:8px;">SpO₂ / HR / BP / RR</b>
    </div>
    <div style="border:1px solid rgba(155,190,255,.22);background:rgba(255,255,255,.055);border-radius:16px;padding:12px;">
      <span style="display:block;color:#b8c6d8;font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:900;">Trend Direction</span>
      <b style="display:block;font-size:20px;margin-top:8px;">Worsening / Stable</b>
    </div>
    <div style="border:1px solid rgba(155,190,255,.22);background:rgba(255,255,255,.055);border-radius:16px;padding:12px;">
      <span style="display:block;color:#b8c6d8;font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:900;">Lead-Time Context</span>
      <b style="display:block;font-size:20px;margin-top:8px;">3.4–4.0 hrs</b>
    </div>
  </div>

  <div style="margin-top:12px;border:1px solid rgba(255,215,138,.35);background:rgba(255,215,138,.095);color:#ffe2a5;border-radius:14px;padding:10px 12px;font-size:12px;font-weight:780;line-height:1.45;">
    Guardrail: MIMIC-IV and eICU use different event definitions. Use this module to explain threshold-direction behavior and review-queue prioritization, not diagnosis, treatment direction, or autonomous escalation.
  </div>
</section>
{CC_END}
"""


def patch_html(path: Path, block: str, start: str, end: str):
    if not path.exists():
        print("SKIP missing HTML:", path)
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = marker_replace(text, start, end, "")

    idx = text.lower().rfind("</main>")
    if idx == -1:
        idx = text.lower().rfind("</body>")
    if idx == -1:
        print("SKIP no insertion point:", path)
        return

    text = text[:idx] + "\n" + block + "\n" + text[idx:]
    path.write_text(text, encoding="utf-8")
    print("PATCHED:", path)


def score_command_center(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return -1

    name = path.name.lower()
    score = 0
    if "command" in name:
        score += 50
    if "center" in name:
        score += 40
    for token in ["Command Center", "patient", "risk", "priority", "alert", "workflow", "ACK", "Escalate"]:
        if token in text:
            score += 6
    return score


def patch_command_center():
    candidates = list(WEB.glob("*.html"))
    if not candidates:
        print("No HTML files found.")
        return

    ranked = sorted(((score_command_center(p), p) for p in candidates), reverse=True, key=lambda x: x[0])
    score, target = ranked[0]

    if score < 30:
        print("Could not confidently identify command center HTML. Candidates:")
        for s, p in ranked:
            print(s, p)
        return

    text = target.read_text(encoding="utf-8", errors="ignore")
    text = marker_replace(text, CC_START, CC_END, "")

    insert_after = None
    header = re.search(r"</header>", text, flags=re.I)
    nav = re.search(r"</nav>", text, flags=re.I)
    if header:
        insert_after = header.end()
    elif nav:
        insert_after = nav.end()

    if insert_after:
        text = text[:insert_after] + "\n" + CC_BLOCK + "\n" + text[insert_after:]
    else:
        idx = text.lower().rfind("</body>")
        if idx == -1:
            print("No command center insertion point:", target)
            return
        text = text[:idx] + "\n" + CC_BLOCK + "\n" + text[idx:]

    target.write_text(text, encoding="utf-8")
    print("PATCHED COMMAND CENTER:", target)


pilot_md = f"""# Early Risk Alert AI — Multi-Dataset Pilot Summary

## Top-Line Summary

{TOP_LINE}

{BEHAVIOR_LINE}

{LEAD_LINE}

## Why This Matters

The evidence package no longer rests on a single dataset. MIMIC-IV supports strict clinical-event cross-cohort retrospective stability, while eICU provides a separate second-dataset outcome-proxy check.

## Conservative t=6.0 Summary

| Dataset | Event Definition | Cases | Rows | Alert Reduction | FPR | Detection | Lead-Time Context |
|---|---|---:|---:|---:|---:|---:|---:|
| MIMIC-IV v3.1 | Strict clinical-event labels / event clusters | {mimic_cases} | {mimic_rows} | {mimic_alert} | {mimic_fpr} | {mimic_detection} | {mimic_lead} |
| eICU v2.0 | Outcome-proxy labels from mortality/discharge-derived event context | {eicu_cases} | {eicu_rows} | {eicu_alert} | {eicu_fpr} | {eicu_detection} | {eicu_lead} |

## Interpretation Guardrail

{GUARDRAIL_LINE}

The detection percentages are not meant to be compared directly as identical clinical endpoints. The stronger finding is that ERA preserved the same threshold-direction behavior across both datasets.

## Approved Framing

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

## Avoid

- validated on two datasets without qualification
- proven generalizability
- prospective validation
- ERA predicts deterioration
- ERA prevents adverse events
- ERA diagnoses
- ERA directs treatment
- ERA replaces clinician judgment
- ERA independently triggers escalation
- direct superiority over standard monitoring

## Decision-Support Boundary

Early Risk Alert AI is HCP-facing decision support and workflow support. It is not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

(DOCS / "multi_dataset_pilot_summary.md").write_text(pilot_md, encoding="utf-8")

framing_md = f"""# Multi-Dataset Public Framing Update

Generated: {datetime.now(timezone.utc).isoformat()}

## Agreed Updates

- Add clean MIMIC-IV vs eICU t=6.0 side-by-side summary.
- Clearly label MIMIC as strict clinical-event evidence and eICU as outcome-proxy evidence.
- Use the shared threshold-direction behavior as the main cross-dataset story.
- Add Command Center explainability alignment module.
- Create a one-page pilot summary.

## Deliberately Avoided

- “Validated on two datasets” without qualification.
- “Proven generalizability.”
- Direct comparison of MIMIC detection rate vs eICU detection rate as identical endpoints.
- Early-detection, diagnosis, treatment, prevention, or autonomous-escalation claims.
"""

(DOCS / "multi_dataset_public_framing_update.md").write_text(framing_md, encoding="utf-8")


patch_html(WEB / "validation_intelligence.html", HTML_BLOCK, HTML_START, HTML_END)
patch_html(WEB / "validation_evidence.html", HTML_BLOCK, HTML_START, HTML_END)
patch_command_center()


readme_text = README.read_text(encoding="utf-8", errors="ignore") if README.exists() else "# Early Risk Alert AI\n"

readme_block = f"""
{README_START}
## Multi-Dataset Public Framing

**{TOP_LINE}**

{BEHAVIOR_LINE}

{LEAD_LINE}

### Critical Interpretation Guardrail

{GUARDRAIL_LINE}

### Approved Claim

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

### Do Not Claim

- validated on two datasets without qualification
- proven generalizability
- prospective validation
- diagnosis, treatment direction, prevention, or autonomous escalation

Public evidence remains aggregate-only. Raw restricted files and row-level outputs remain local-only.
{README_END}
"""

readme_text = marker_replace(readme_text, README_START, README_END, readme_block)
if README_START not in readme_text:
    readme_text = readme_text.rstrip() + "\n\n" + readme_block + "\n"

README.write_text(readme_text, encoding="utf-8")


app_text = APP_FILE.read_text(encoding="utf-8")

while ROUTE_START in app_text:
    a = app_text.find(ROUTE_START)
    b = app_text.find(ROUTE_END, a)
    if b == -1:
        raise SystemExit("ERROR: route marker start without end.")
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
    raise SystemExit("ERROR: return app not found in create_app().")

insert_line = max(return_lines)

route_block = r'''
    # ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_ROUTES_V1_START
    @app.get("/api/validation/multi-dataset-public-framing")
    def era_validation_multi_dataset_public_framing_api():
        import json
        from pathlib import Path
        from flask import jsonify

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "data" / "validation" / "multi_dataset_public_framing_polish.json"

        if not p.exists():
            return jsonify({"ok": False, "error": "Multi-dataset public framing summary not found."}), 404

        data = json.loads(p.read_text(encoding="utf-8"))
        data["ok"] = True
        return jsonify(data)

    @app.get("/validation-evidence/multi-dataset-pilot-summary.md")
    def era_validation_multi_dataset_pilot_summary_md():
        from pathlib import Path
        from flask import Response

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "docs" / "validation" / "multi_dataset_pilot_summary.md"

        if not p.exists():
            return Response("Multi-dataset pilot summary not found.", status=404, mimetype="text/plain")

        return Response(
            p.read_text(encoding="utf-8"),
            mimetype="text/markdown",
            headers={
                "Content-Disposition": "inline; filename=early-risk-alert-ai-multi-dataset-pilot-summary.md"
            }
        )
    # ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_ROUTES_V1_END

'''

lines = app_text.splitlines()
lines = lines[:insert_line - 1] + route_block.splitlines() + lines[insert_line - 1:]
APP_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("DONE — multi-dataset public framing polish applied.")
