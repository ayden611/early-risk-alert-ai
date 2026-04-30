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

ROUTE_START = "# ERA_FINAL_FRONTEND_EVIDENCE_POLISH_ROUTES_V1_START"
ROUTE_END = "# ERA_FINAL_FRONTEND_EVIDENCE_POLISH_ROUTES_V1_END"

CC_START = "<!-- ERA_FINAL_COMMAND_CENTER_EXPLAINABILITY_V1_START -->"
CC_END = "<!-- ERA_FINAL_COMMAND_CENTER_EXPLAINABILITY_V1_END -->"

NAV = """
<nav class="topnav">
  <a href="/command-center">Command Center</a>
  <a href="/validation-intelligence">Validation Intelligence</a>
  <a href="/validation-evidence">Evidence Packet</a>
  <a href="/validation-runs">Validation Runs</a>
  <a href="/model-card">Model Card</a>
  <a href="/pilot-success-guide">Pilot Success Guide</a>
</nav>
"""

CSS = """
<style>
:root{
  --bg:#07101f;
  --panel:#101b2e;
  --panel2:#0d1626;
  --line:rgba(170,205,255,.22);
  --text:#f6fbff;
  --muted:#b8c6d8;
  --green:#8fffd2;
  --blue:#8bb8ff;
  --gold:#ffe2a5;
  --red:#ff9da8;
}
*{box-sizing:border-box}
body{
  margin:0;
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 10% 0%, rgba(139,184,255,.16), transparent 28%),
    radial-gradient(circle at 90% 10%, rgba(143,255,210,.12), transparent 28%),
    var(--bg);
  color:var(--text);
}
a{color:var(--green);text-decoration:none;font-weight:850}
.wrapper{width:min(1180px, calc(100% - 28px));margin:0 auto;padding:28px 0 52px}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--green);font-weight:950}
h1{font-size:clamp(34px,6vw,64px);letter-spacing:-.07em;line-height:.95;margin:8px 0 14px}
h2{font-size:clamp(24px,3vw,36px);letter-spacing:-.05em;margin:0 0 12px}
h3{margin:22px 0 10px;letter-spacing:-.03em}
p{line-height:1.56}
.muted{color:var(--muted)}
.hero,.card{
  border:1px solid var(--line);
  background:linear-gradient(145deg, rgba(16,27,46,.95), rgba(7,16,31,.95));
  border-radius:28px;
  padding:24px;
  box-shadow:0 22px 65px rgba(0,0,0,.28);
}
.hero{margin-top:20px}
.grid{display:grid;gap:14px}
.grid.cols4{grid-template-columns:repeat(4,1fr)}
.grid.cols3{grid-template-columns:repeat(3,1fr)}
.grid.cols2{grid-template-columns:repeat(2,1fr)}
.metric{
  border:1px solid rgba(170,205,255,.18);
  background:rgba(255,255,255,.055);
  border-radius:20px;
  padding:18px;
}
.metric span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.13em;font-weight:950}
.metric strong{display:block;font-size:clamp(26px,4vw,44px);margin-top:8px;color:var(--green);letter-spacing:-.06em}
.metric em{display:block;color:var(--muted);font-style:normal;font-size:12px;margin-top:6px}
.callout{
  border-left:4px solid var(--green);
  background:rgba(143,255,210,.09);
  border-radius:16px;
  padding:14px 16px;
  font-weight:850;
}
.guardrail{
  border:1px solid rgba(255,226,165,.35);
  background:rgba(255,226,165,.09);
  color:var(--gold);
  border-radius:18px;
  padding:14px 16px;
  font-weight:780;
}
table{
  width:100%;
  border-collapse:collapse;
  overflow:hidden;
  border-radius:18px;
  border:1px solid rgba(170,205,255,.20);
  background:rgba(255,255,255,.035);
  margin-top:12px;
}
th,td{
  text-align:left;
  padding:12px 14px;
  border-bottom:1px solid rgba(170,205,255,.14);
  vertical-align:top;
  font-size:14px;
}
th{
  color:#dce9ff;
  background:rgba(139,184,255,.12);
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.12em;
}
tr:last-child td{border-bottom:0}
.pill{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:6px 10px;
  border-radius:999px;
  border:1px solid rgba(143,255,210,.35);
  background:rgba(143,255,210,.11);
  color:var(--green);
  font-size:11px;
  font-weight:950;
  letter-spacing:.08em;
  text-transform:uppercase;
}
.warn{border-color:rgba(255,226,165,.35);background:rgba(255,226,165,.10);color:var(--gold)}
.topnav{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin:18px 0;
}
.topnav a{
  color:#eaf4ff;
  border:1px solid rgba(170,205,255,.18);
  background:rgba(255,255,255,.06);
  border-radius:999px;
  padding:9px 12px;
  font-size:12px;
}
.section{margin-top:18px}
ul{line-height:1.7}
.footer{margin-top:24px;color:var(--muted);font-size:12px}
@media(max-width:900px){
  .grid.cols4,.grid.cols3,.grid.cols2{grid-template-columns:1fr}
  table{display:block;overflow-x:auto;white-space:nowrap}
}
@media print{
  body{background:#fff;color:#111}
  .hero,.card,.metric{box-shadow:none;background:#fff;color:#111;border-color:#ccc}
  .topnav{display:none}
  a{color:#111}
}
</style>
"""


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
        if isinstance(x, str) and x:
            return x
        return "—"
    if abs(n - round(n)) < 0.001:
        return f"{n:.0f}{suffix}"
    return f"{n:.2f}".rstrip("0").rstrip(".") + suffix


def page(title: str, eyebrow: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} — Early Risk Alert AI</title>
  {CSS}
</head>
<body>
  <div class="wrapper">
    <div class="eyebrow">Early Risk Alert AI · {eyebrow}</div>
    <h1>{title}</h1>
    {NAV}
    {body}
    <div class="footer">
      Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
    </div>
  </div>
</body>
</html>
"""


multi = load_json(DATA / "multi_dataset_public_framing_polish.json") or load_json(DATA / "multi_dataset_robustness_summary.json")
checkpoint = load_json(DATA / "multi_dataset_retrospective_robustness_checkpoint_2026_04_30.json")
eicu_summary = load_json(DATA / "eicu_validation_summary.json")
registry = load_json(DATA / "validation_run_registry.json")

TOP_LINE = multi.get("top_line") or checkpoint.get("top_line") or (
    "Early Risk Alert AI now has retrospective evidence across two de-identified ICU datasets: "
    "MIMIC-IV strict clinical-event cross-cohort validation plus a separate eICU outcome-proxy check."
)
BEHAVIOR_LINE = multi.get("behavior_line") or (
    "Across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, "
    "while conservative thresholds reduced review burden and false positives."
)
GUARDRAIL_LINE = multi.get("event_definition_guardrail") or checkpoint.get("interpretation_guardrail") or (
    "MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses "
    "stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context."
)
LEAD_LINE = multi.get("lead_time_context") or checkpoint.get("lead_time_context") or (
    "At conservative operating points, median lead-time context remained approximately 3.4–4.0 hours across MIMIC-IV and eICU evidence."
)

mimic = (multi.get("mimic_iv") or checkpoint.get("datasets", {}).get("mimic_iv") or {})
eicu = (multi.get("eicu") or checkpoint.get("datasets", {}).get("eicu") or {})

MIMIC_T6 = {
    "dataset": "MIMIC-IV v3.1",
    "role": "Strict clinical-event cross-cohort validation release",
    "cases": mimic.get("cases", "577–1,705"),
    "rows": mimic.get("rows", "456,453 full cohort"),
    "event_definition": "Strict clinical-event labels / event clusters",
    "alert_reduction": mimic.get("alert_reduction", "94.0%–94.9%"),
    "fpr": mimic.get("fpr", "3.7%–4.4%"),
    "detection": mimic.get("detection", "14.1%–16.0%"),
    "lead": mimic.get("lead_time", "4.0 hrs"),
}

EICU_T6 = {
    "dataset": "eICU v2.0",
    "role": "Second-dataset outcome-proxy check",
    "cases": eicu.get("cases", "2,394"),
    "rows": eicu.get("rows", "2,023,962"),
    "event_definition": "Mortality/discharge-derived outcome proxy",
    "alert_reduction": eicu.get("alert_reduction", "96.8%"),
    "fpr": eicu.get("fpr", "1.8%"),
    "detection": eicu.get("detection", "66.6%"),
    "lead": eicu.get("lead_time", "3.41 hrs"),
}

MIMIC_THRESHOLDS = [
    {"t": "4.0", "setting": "ICU / high-acuity", "alert": "80.3%", "fpr": "14.4%", "detection": "37.9%", "lead": "4.0 hrs", "alerts_day": "2.22"},
    {"t": "5.0", "setting": "Mixed units", "alert": "89.1%", "fpr": "8.0%", "detection": "24.6%", "lead": "4.0 hrs", "alerts_day": "1.23"},
    {"t": "6.0", "setting": "Telemetry / stepdown", "alert": "94.3%", "fpr": "4.2%", "detection": "15.3%", "lead": "4.0 hrs", "alerts_day": "0.65"},
]

EICU_THRESHOLDS = [
    {"t": "4.0", "setting": "ICU / high-acuity", "alert": "87.7%", "fpr": "7.3%", "detection": "81.3%", "lead": "4.89 hrs"},
    {"t": "5.0", "setting": "Mixed units", "alert": "93.5%", "fpr": "3.7%", "detection": "75.1%", "lead": "4.32 hrs"},
    {"t": "6.0", "setting": "Telemetry / stepdown", "alert": "96.8%", "fpr": "1.8%", "detection": "66.6%", "lead": "3.41 hrs"},
]

actual_eicu = []
for r in sorted(eicu_summary.get("runs", []), key=lambda x: x.get("threshold", 999)):
    actual_eicu.append({
        "t": fmt(r.get("threshold")),
        "setting": r.get("suggested_setting") or "—",
        "alert": fmt(r.get("alert_reduction_pct"), "%"),
        "fpr": fmt(r.get("era_fpr_pct"), "%"),
        "detection": fmt(r.get("detection_pct"), "%"),
        "lead": fmt(r.get("median_lead_time_hours"), " hrs"),
    })
if len(actual_eicu) >= 3:
    EICU_THRESHOLDS = actual_eicu


def threshold_rows(rows):
    out = ""
    for r in rows:
        out += f"""
        <tr>
          <td><strong>t={r['t']}</strong></td>
          <td>{r.get('setting','—')}</td>
          <td>{r.get('alert','—')}</td>
          <td>{r.get('fpr','—')}</td>
          <td>{r.get('detection','—')}</td>
          <td>{r.get('lead','—')}</td>
          <td>{r.get('alerts_day','—')}</td>
        </tr>
        """
    return out


def run_history_rows():
    rows = [
        {
            "run": "MIMIC-IV full cohort",
            "dataset": "MIMIC-IV v3.1",
            "event": "Strict clinical-event labels",
            "cases": "1,705",
            "rows": "456,453",
            "threshold": "t=6.0",
            "alert": "94.3%",
            "fpr": "4.2%",
            "detection": "15.3%",
            "lead": "4.0 hrs",
            "status": "Locked",
        },
        {
            "run": "MIMIC-IV subcohort B",
            "dataset": "MIMIC-IV v3.1",
            "event": "Strict clinical-event labels",
            "cases": "577",
            "rows": "~159K",
            "threshold": "t=6.0",
            "alert": "94.0%",
            "fpr": "4.4%",
            "detection": "16.0%",
            "lead": "4.0 hrs",
            "status": "Locked",
        },
        {
            "run": "MIMIC-IV subcohort C",
            "dataset": "MIMIC-IV v3.1",
            "event": "Strict clinical-event labels",
            "cases": "621",
            "rows": "~159K",
            "threshold": "t=6.0",
            "alert": "94.9%",
            "fpr": "3.7%",
            "detection": "14.1%",
            "lead": "4.0 hrs",
            "status": "Locked",
        },
        {
            "run": "eICU outcome-proxy cohort",
            "dataset": "eICU v2.0",
            "event": "Mortality/discharge-derived outcome proxy",
            "cases": "2,394",
            "rows": "2,023,962",
            "threshold": "t=6.0",
            "alert": EICU_T6["alert_reduction"],
            "fpr": EICU_T6["fpr"],
            "detection": EICU_T6["detection"],
            "lead": EICU_T6["lead"],
            "status": "Outcome-proxy check",
        },
    ]

    html = ""
    for r in rows:
        html += f"""
        <tr>
          <td><strong>{r['run']}</strong></td>
          <td>{r['dataset']}</td>
          <td>{r['event']}</td>
          <td>{r['cases']}</td>
          <td>{r['rows']}</td>
          <td>{r['threshold']}</td>
          <td>{r['alert']}</td>
          <td>{r['fpr']}</td>
          <td>{r['detection']}</td>
          <td>{r['lead']}</td>
          <td><span class="pill">{r['status']}</span></td>
        </tr>
        """
    return html


validation_intelligence_body = f"""
<section class="hero">
  <span class="pill">Published · DUA-safe aggregate evidence</span>
  <h2 style="margin-top:18px;">Multi-dataset retrospective robustness story</h2>
  <p class="callout">{TOP_LINE}</p>
  <p class="muted"><strong>{BEHAVIOR_LINE}</strong></p>
  <p class="muted">{LEAD_LINE}</p>
  <div class="grid cols4 section">
    <div class="metric"><span>MIMIC-IV t=6.0 Alert Reduction</span><strong>{MIMIC_T6['alert_reduction']}</strong><em>Strict clinical-event evidence</em></div>
    <div class="metric"><span>MIMIC-IV t=6.0 FPR</span><strong>{MIMIC_T6['fpr']}</strong><em>Cross-cohort release</em></div>
    <div class="metric"><span>eICU t=6.0 Alert Reduction</span><strong>{EICU_T6['alert_reduction']}</strong><em>Outcome-proxy check</em></div>
    <div class="metric"><span>Lead-Time Context</span><strong>3.4–4.0 hrs</strong><em>Retrospective timing context</em></div>
  </div>
</section>

<section class="card section">
  <h2>Side-by-Side Conservative Operating Point</h2>
  <p class="muted">The strongest public story is consistent threshold-direction behavior, not direct equality between different event definitions.</p>
  <table>
    <thead><tr><th>Dataset</th><th>Evidence Role</th><th>Event Definition</th><th>Cases</th><th>Rows</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th></tr></thead>
    <tbody>
      <tr><td><strong>{MIMIC_T6['dataset']}</strong></td><td>{MIMIC_T6['role']}</td><td>{MIMIC_T6['event_definition']}</td><td>{MIMIC_T6['cases']}</td><td>{MIMIC_T6['rows']}</td><td>{MIMIC_T6['alert_reduction']}</td><td>{MIMIC_T6['fpr']}</td><td>{MIMIC_T6['detection']}</td><td>{MIMIC_T6['lead']}</td></tr>
      <tr><td><strong>{EICU_T6['dataset']}</strong></td><td>{EICU_T6['role']}</td><td>{EICU_T6['event_definition']}</td><td>{EICU_T6['cases']}</td><td>{EICU_T6['rows']}</td><td>{EICU_T6['alert_reduction']}</td><td>{EICU_T6['fpr']}</td><td>{EICU_T6['detection']}</td><td>{EICU_T6['lead']}</td></tr>
    </tbody>
  </table>
  <p class="guardrail">{GUARDRAIL_LINE}</p>
</section>

<section class="card section">
  <h2>Explainability Context</h2>
  <p class="muted">The command-center display should make prioritization understandable without relying on raw score alone.</p>
  <div class="grid cols4">
    <div class="metric"><span>Priority Tier</span><strong>Critical</strong><em>Low / Watch / Elevated / Critical</em></div>
    <div class="metric"><span>Queue Rank</span><strong>#1</strong><em>Relative review ordering</em></div>
    <div class="metric"><span>Primary Driver</span><strong>SpO₂</strong><em>SpO₂ / HR / BP / RR</em></div>
    <div class="metric"><span>Trend</span><strong>Worsening</strong><em>Context for review</em></div>
  </div>
</section>
"""

(WEB / "validation_intelligence.html").write_text(
    page("Validation Intelligence", "Retrospective Evidence", validation_intelligence_body),
    encoding="utf-8"
)


validation_evidence_body = f"""
<section class="hero">
  <span class="pill">Pilot Evidence Packet</span>
  <h2 style="margin-top:18px;">Printable validation evidence summary</h2>
  <p class="callout">{TOP_LINE}</p>
  <p class="muted">{BEHAVIOR_LINE}</p>
  <p class="guardrail">{GUARDRAIL_LINE}</p>
</section>

<section class="card section">
  <h2>MIMIC-IV Threshold Matrix</h2>
  <p class="muted">Strict clinical-event retrospective validation. Conservative t=6.0 is optimized for low review burden.</p>
  <table>
    <thead><tr><th>Threshold</th><th>Suggested Setting</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th><th>ERA Alerts / Patient-Day</th></tr></thead>
    <tbody>{threshold_rows(MIMIC_THRESHOLDS)}</tbody>
  </table>
</section>

<section class="card section">
  <h2>eICU Threshold Matrix</h2>
  <p class="muted">Second-dataset retrospective outcome-proxy check using mortality/discharge-derived event context.</p>
  <table>
    <thead><tr><th>Threshold</th><th>Suggested Setting</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th><th>ERA Alerts / Patient-Day</th></tr></thead>
    <tbody>{threshold_rows(EICU_THRESHOLDS)}</tbody>
  </table>
</section>

<section class="card section">
  <h2>Representative Explainability Display</h2>
  <p class="muted">Sanitized examples demonstrate what reviewers see without exposing row-level restricted data.</p>
  <table>
    <thead><tr><th>Queue Rank</th><th>Case Label</th><th>Priority Tier</th><th>Primary Driver</th><th>Trend</th><th>Lead-Time Context</th><th>Review Meaning</th></tr></thead>
    <tbody>
      <tr><td>#1</td><td>Case-001</td><td>Critical</td><td>SpO₂ decline</td><td>Worsening</td><td>~4 hrs</td><td>May warrant prioritized clinician review</td></tr>
      <tr><td>#2</td><td>Case-002</td><td>Critical</td><td>BP instability</td><td>Worsening</td><td>~3.5 hrs</td><td>Review queue prioritization context</td></tr>
      <tr><td>#3</td><td>Case-003</td><td>Elevated</td><td>HR instability</td><td>Stable / watch</td><td>~3 hrs</td><td>Monitor trend and reassess</td></tr>
    </tbody>
  </table>
</section>

<section class="card section">
  <h2>Downloads</h2>
  <ul>
    <li><a href="/api/validation/multi-dataset-checkpoint">Locked multi-dataset checkpoint JSON</a></li>
    <li><a href="/api/validation/multi-dataset-public-framing">Multi-dataset public framing JSON</a></li>
    <li><a href="/validation-evidence/multi-dataset-pilot-summary.md">Multi-dataset pilot one-pager</a></li>
    <li><a href="/validation-evidence/model-card.md">Model Card markdown</a></li>
    <li><a href="/validation-evidence/pilot-success-guide.md">Pilot Success Guide markdown</a></li>
  </ul>
</section>
"""

(WEB / "validation_evidence.html").write_text(
    page("Pilot Evidence Packet", "Validation Evidence", validation_evidence_body),
    encoding="utf-8"
)


validation_runs_body = f"""
<section class="hero">
  <span class="pill">Run Registry</span>
  <h2 style="margin-top:18px;">Validation run history</h2>
  <p class="muted">This page replaces placeholder “Loading…” views with the current locked evidence registry.</p>
  <p class="callout">{BEHAVIOR_LINE}</p>
</section>

<section class="card section">
  <h2>Current Locked Runs</h2>
  <table>
    <thead><tr><th>Run</th><th>Dataset</th><th>Event Definition</th><th>Cases</th><th>Rows</th><th>Threshold</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th><th>Status</th></tr></thead>
    <tbody>{run_history_rows()}</tbody>
  </table>
</section>

<section class="card section">
  <h2>Checkpoint</h2>
  <p><span class="pill">Locked</span></p>
  <p><strong>multi-dataset-retrospective-robustness-checkpoint-2026-04-30</strong></p>
  <p class="muted">MIMIC-IV strict clinical-event evidence and eICU second-dataset outcome-proxy evidence are locked as the current validation checkpoint.</p>
</section>
"""

(WEB / "validation_runs.html").write_text(
    page("Validation Runs", "Run Registry", validation_runs_body),
    encoding="utf-8"
)


model_card_md = f"""# Early Risk Alert AI — Model Card

## Intended Use

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.

It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.

## Current Evidence Status

The current evidence package includes:

- MIMIC-IV strict clinical-event cross-cohort retrospective validation.
- eICU second-dataset retrospective outcome-proxy check.
- DUA-safe aggregate-only public outputs.
- Local-only handling of restricted raw data and row-level derived outputs.

## Dataset Roles

| Dataset | Role | Event Definition |
|---|---|---|
| MIMIC-IV v3.1 | Strict clinical-event cross-cohort retrospective validation release | Clinical-event labels / event clusters |
| eICU v2.0 | Separate second-dataset retrospective outcome-proxy check | Mortality/discharge-derived outcome proxy |

## Conservative Operating Point

| Dataset | Alert Reduction | FPR | Detection | Lead-Time Context |
|---|---:|---:|---:|---:|
| MIMIC-IV t=6.0 | {MIMIC_T6['alert_reduction']} | {MIMIC_T6['fpr']} | {MIMIC_T6['detection']} | {MIMIC_T6['lead']} |
| eICU t=6.0 | {EICU_T6['alert_reduction']} | {EICU_T6['fpr']} | {EICU_T6['detection']} | {EICU_T6['lead']} |

## Interpretation Guardrail

{GUARDRAIL_LINE}

## Explainability Outputs

The platform should display:

- Priority Tier
- Queue Rank
- Primary Driver
- Trend Direction
- Lead-Time Context
- Review-state workflow actions

## Limitations

- Retrospective analysis only.
- No prospective clinical validation yet.
- MIMIC-IV and eICU event definitions are not identical.
- Detection rates should not be compared as equivalent endpoints.
- The system is not autonomous and does not direct treatment.

## Governance Controls

- DUA-safe aggregate public summaries.
- Raw restricted files kept local-only.
- Row-level enriched exports kept local-only.
- Conservative claim guardrails.
- Decision-support-only disclaimers.

## Claims to Avoid

- Proven generalizability.
- Prospective validation.
- Diagnosis.
- Treatment direction.
- Prevention of adverse events.
- Replacement of clinician judgment.
- Autonomous escalation.
- Direct superiority over standard monitoring.
"""

(DOCS / "model_card.md").write_text(model_card_md, encoding="utf-8")

model_card_body = f"""
<section class="hero">
  <span class="pill">Model Card</span>
  <h2 style="margin-top:18px;">Current model and validation disclosure</h2>
  <p class="callout">Early Risk Alert AI is decision support and workflow support for authorized healthcare professionals.</p>
  <p class="guardrail">It is not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.</p>
</section>

<section class="card section">
  <h2>Evidence Status</h2>
  <table>
    <thead><tr><th>Dataset</th><th>Role</th><th>Event Definition</th><th>Alert Reduction</th><th>FPR</th><th>Detection</th><th>Lead Time</th></tr></thead>
    <tbody>
      <tr><td>MIMIC-IV v3.1</td><td>Strict clinical-event validation</td><td>Clinical-event labels / clusters</td><td>{MIMIC_T6['alert_reduction']}</td><td>{MIMIC_T6['fpr']}</td><td>{MIMIC_T6['detection']}</td><td>{MIMIC_T6['lead']}</td></tr>
      <tr><td>eICU v2.0</td><td>Outcome-proxy check</td><td>Mortality/discharge-derived proxy</td><td>{EICU_T6['alert_reduction']}</td><td>{EICU_T6['fpr']}</td><td>{EICU_T6['detection']}</td><td>{EICU_T6['lead']}</td></tr>
    </tbody>
  </table>
</section>

<section class="card section">
  <h2>Limitations and Guardrails</h2>
  <ul>
    <li>Retrospective aggregate evidence only.</li>
    <li>No prospective clinical validation yet.</li>
    <li>MIMIC-IV and eICU event definitions differ.</li>
    <li>Detection rates are not equivalent endpoint definitions.</li>
    <li>Decision support only; no diagnosis or treatment direction.</li>
  </ul>
</section>
"""

(WEB / "model_card.html").write_text(
    page("Model Card", "Governance", model_card_body),
    encoding="utf-8"
)


pilot_success_md = f"""# Early Risk Alert AI — Pilot Success Guide

## Pilot Purpose

Evaluate whether Early Risk Alert AI can support authorized healthcare professionals with explainable patient prioritization and command-center workflow visibility.

## Recommended Pilot Scope

- Controlled pilot evaluation.
- Authorized HCP users only.
- De-identified or approved data flow during initial review.
- No autonomous clinical escalation.
- No diagnosis or treatment direction.
- Workflow-support evaluation only.

## Suggested Success Metrics

| Area | Metric |
|---|---|
| Review burden | Alerts/reviews per patient-day |
| Prioritization clarity | Percentage of high-priority reviews with visible tier, driver, trend, and queue rank |
| Explainability | Reviewer understanding of why a case was prioritized |
| Workflow | Acknowledged, assigned, escalated, resolved workflow-state completion |
| Safety | No autonomous escalation or treatment-direction claims |
| Operational adoption | Reviewer feedback and command-center usefulness |

## Evidence Baseline

{TOP_LINE}

{BEHAVIOR_LINE}

## Pilot Readiness Checklist

- Hospital/site name identified.
- Site sponsor or clinical champion identified.
- Pilot unit/scope defined.
- Named pilot users and roles defined.
- IT/security contact defined.
- Data-use pathway reviewed.
- Decision-support-only disclaimer accepted.
- Success metrics agreed before pilot launch.

## Approved Framing

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

## Avoid

- Claims of diagnosis, treatment, prevention, or autonomous escalation.
- Hard financial ROI claims before site-specific pilot data.
- Direct comparison of MIMIC and eICU detection rates as identical endpoint definitions.
"""

(DOCS / "pilot_success_guide.md").write_text(pilot_success_md, encoding="utf-8")

pilot_success_body = f"""
<section class="hero">
  <span class="pill">Pilot Success Guide</span>
  <h2 style="margin-top:18px;">Hospital pilot readiness framework</h2>
  <p class="callout">The goal is to evaluate explainable patient prioritization, review-burden reduction, and command-center workflow visibility.</p>
  <p class="guardrail">Pilot success should be measured with pre-agreed workflow and review metrics, not unsupported outcome or ROI claims.</p>
</section>

<section class="card section">
  <h2>Suggested Pilot Success Metrics</h2>
  <table>
    <thead><tr><th>Area</th><th>Metric</th></tr></thead>
    <tbody>
      <tr><td>Review burden</td><td>Alerts or reviews per patient-day</td></tr>
      <tr><td>Prioritization clarity</td><td>Priority tier, primary driver, trend, queue rank, and lead-time context visible</td></tr>
      <tr><td>Explainability</td><td>Reviewer can understand why a patient may warrant review</td></tr>
      <tr><td>Workflow</td><td>Acknowledge, assign, escalate, resolve workflow-state tracking</td></tr>
      <tr><td>Safety</td><td>No autonomous escalation, diagnosis, or treatment direction</td></tr>
    </tbody>
  </table>
</section>

<section class="card section">
  <h2>Pilot Readiness Checklist</h2>
  <ul>
    <li>Hospital/site name identified.</li>
    <li>Site sponsor or clinical champion identified.</li>
    <li>Pilot unit/scope defined.</li>
    <li>Named pilot users and roles defined.</li>
    <li>IT/security contact defined.</li>
    <li>Data-use pathway reviewed.</li>
    <li>Success metrics agreed before launch.</li>
  </ul>
</section>
"""

(WEB / "pilot_success_guide.html").write_text(
    page("Pilot Success Guide", "Pilot Readiness", pilot_success_body),
    encoding="utf-8"
)


CC_BLOCK = f"""
{CC_START}
<section class="era-final-explainability-module" style="
  width:min(1180px, calc(100% - 28px));
  margin:16px auto;
  border:1px solid rgba(143,255,210,.34);
  border-left:5px solid #8fffd2;
  border-radius:24px;
  background:linear-gradient(145deg, rgba(13,24,42,.96), rgba(6,11,21,.96));
  box-shadow:0 22px 65px rgba(0,0,0,.32);
  color:#f7fbff;
  padding:18px;
  position:relative;
  z-index:5;
">
  <div style="color:#8fffd2;text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:950;margin-bottom:6px;">
    Explainable Clinical Prioritization
  </div>

  <h2 style="margin:0;font-size:clamp(22px,2.8vw,34px);letter-spacing:-.05em;">
    Priority context is visible before workflow action.
  </h2>

  <p style="margin:10px 0 0;color:#dbe8fb;font-size:14px;line-height:1.55;font-weight:750;">
    {BEHAVIOR_LINE} The live view should show why a patient may warrant review, not just a raw risk score.
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
    Guardrail: this is review-prioritization context, not diagnosis, treatment direction, or autonomous escalation.
  </div>
</section>
{CC_END}
"""


def marker_replace(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


def score_command_center(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return -1

    name = path.name.lower()
    score = 0
    if "command" in name:
        score += 60
    if "center" in name:
        score += 40
    for token in ["Command Center", "ROI", "Wall Mode", "Route + Document Status", "Pilot Docs", "MIMIC Validation"]:
        if token in text:
            score += 10
    return score


def update_missing_status(text: str, label: str) -> str:
    pattern = re.compile(r"(" + re.escape(label).replace("\\ ", r"\s+") + r"(?:(?!\n\n).){0,700}?)(MISSING)", re.I | re.S)
    def repl(m):
        return m.group(1).replace("missing", "ready").replace("MISSING", "READY") + "READY"
    text = pattern.sub(repl, text, count=1)
    text = re.sub(r"(?i)(model\s*card(?:(?!\n\n).){0,700}?class=[\"'][^\"']*)missing", r"\1ready", text, count=1)
    text = re.sub(r"(?i)(pilot\s*success\s*guide(?:(?!\n\n).){0,700}?class=[\"'][^\"']*)missing", r"\1ready", text, count=1)
    return text


def patch_command_center():
    candidates = sorted(WEB.glob("*.html"))
    ranked = sorted(((score_command_center(p), p) for p in candidates), reverse=True, key=lambda x: x[0])
    if not ranked or ranked[0][0] < 50:
        print("WARNING: could not confidently identify command center HTML.")
        return None

    target = ranked[0][1]
    text = target.read_text(encoding="utf-8", errors="ignore")
    text = marker_replace(text, CC_START, CC_END, "")
    text = update_missing_status(text, "model card")
    text = update_missing_status(text, "pilot success guide")

    if "/model-card" not in text:
        text = re.sub(r"(</nav>)", r'<a href="/model-card">Model Card</a><a href="/pilot-success-guide">Pilot Success Guide</a>\1', text, count=1, flags=re.I)

    header = re.search(r"</header>", text, flags=re.I)
    nav = re.search(r"</nav>", text, flags=re.I)
    if header:
        idx = header.end()
        text = text[:idx] + "\n" + CC_BLOCK + "\n" + text[idx:]
    elif nav:
        idx = nav.end()
        text = text[:idx] + "\n" + CC_BLOCK + "\n" + text[idx:]
    else:
        idx = text.lower().rfind("</body>")
        if idx != -1:
            text = text[:idx] + "\n" + CC_BLOCK + "\n" + text[idx:]

    target.write_text(text, encoding="utf-8")
    print("PATCHED COMMAND CENTER:", target)
    return str(target)


command_center_file = patch_command_center()

manifest = {
    "ok": True,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "updates": [
        "Rebuilt validation intelligence page with cleaner MIMIC-IV vs eICU framing.",
        "Rebuilt validation evidence page with populated matrices and no Loading placeholders.",
        "Rebuilt validation runs page with real locked run history.",
        "Created model card page and markdown.",
        "Created pilot success guide page and markdown.",
        "Patched command center with visible explainability alignment module.",
        "Updated command-center route status for model card and pilot success guide where detectable.",
    ],
    "not_added": [
        "No hard ROI dollar claims.",
        "No proven generalizability claim.",
        "No direct equality claim between MIMIC and eICU detection rates.",
        "No raw or row-level restricted data exposure.",
    ],
    "command_center_file": command_center_file,
    "safe_claim": "Cross-dataset retrospective robustness evidence across de-identified ICU datasets.",
}
(DATA / "final_frontend_evidence_polish_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    raise SystemExit("ERROR: return app not found inside create_app().")

insert_line = max(return_lines)

route_block = r'''
    # ERA_FINAL_FRONTEND_EVIDENCE_POLISH_ROUTES_V1_START
    @app.get("/validation-runs")
    def era_validation_runs_page_final():
        from pathlib import Path
        from flask import send_from_directory

        web_dir = Path(__file__).resolve().parent / "web"
        return send_from_directory(str(web_dir), "validation_runs.html")

    @app.get("/model-card")
    def era_model_card_page_final():
        from pathlib import Path
        from flask import send_from_directory

        web_dir = Path(__file__).resolve().parent / "web"
        return send_from_directory(str(web_dir), "model_card.html")

    @app.get("/pilot-success-guide")
    def era_pilot_success_guide_page_final():
        from pathlib import Path
        from flask import send_from_directory

        web_dir = Path(__file__).resolve().parent / "web"
        return send_from_directory(str(web_dir), "pilot_success_guide.html")

    @app.get("/validation-evidence/model-card.md")
    def era_model_card_markdown_final():
        from pathlib import Path
        from flask import Response

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "docs" / "validation" / "model_card.md"
        if not p.exists():
            return Response("Model card not found.", status=404, mimetype="text/plain")
        return Response(p.read_text(encoding="utf-8"), mimetype="text/markdown")

    @app.get("/validation-evidence/pilot-success-guide.md")
    def era_pilot_success_guide_markdown_final():
        from pathlib import Path
        from flask import Response

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "docs" / "validation" / "pilot_success_guide.md"
        if not p.exists():
            return Response("Pilot success guide not found.", status=404, mimetype="text/plain")
        return Response(p.read_text(encoding="utf-8"), mimetype="text/markdown")

    @app.get("/api/validation/final-frontend-polish")
    def era_final_frontend_polish_manifest_api():
        import json
        from pathlib import Path
        from flask import jsonify

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "data" / "validation" / "final_frontend_evidence_polish_manifest.json"
        if not p.exists():
            return jsonify({"ok": False, "error": "Final frontend polish manifest not found."}), 404
        data = json.loads(p.read_text(encoding="utf-8"))
        data["ok"] = True
        return jsonify(data)
    # ERA_FINAL_FRONTEND_EVIDENCE_POLISH_ROUTES_V1_END

'''

lines = app_text.splitlines()
lines = lines[:insert_line - 1] + route_block.splitlines() + lines[insert_line - 1:]
APP_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


readme = README.read_text(encoding="utf-8", errors="ignore") if README.exists() else "# Early Risk Alert AI\n"
block_start = "<!-- ERA_FINAL_FRONTEND_EVIDENCE_POLISH_README_V1_START -->"
block_end = "<!-- ERA_FINAL_FRONTEND_EVIDENCE_POLISH_README_V1_END -->"
readme_block = f"""
{block_start}
## Final Frontend Evidence Polish

The live platform now includes:

- `/validation-intelligence`
- `/validation-evidence`
- `/validation-runs`
- `/model-card`
- `/pilot-success-guide`
- `/api/validation/final-frontend-polish`

The public story is:

**{TOP_LINE}**

{BEHAVIOR_LINE}

Important guardrail:

{GUARDRAIL_LINE}

No raw restricted files or row-level outputs are published.
{block_end}
"""
readme = marker_replace(readme, block_start, block_end, readme_block)
if block_start not in readme:
    readme = readme.rstrip() + "\n\n" + readme_block + "\n"
README.write_text(readme, encoding="utf-8")

print("DONE — final frontend evidence polish created.")
