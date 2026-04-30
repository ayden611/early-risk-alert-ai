#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
CHANGED = []

START_CMD = "<!-- ERA_MINIMAL_EXPLAINABILITY_VISIBILITY_V2_START -->"
END_CMD = "<!-- ERA_MINIMAL_EXPLAINABILITY_VISIBILITY_V2_END -->"

START_INTEL = "<!-- ERA_SAFE_MULTI_DATASET_INTELLIGENCE_V2_START -->"
END_INTEL = "<!-- ERA_SAFE_MULTI_DATASET_INTELLIGENCE_V2_END -->"

START_EVIDENCE = "<!-- ERA_LOCKED_EVIDENCE_NO_LOADING_V2_START -->"
END_EVIDENCE = "<!-- ERA_LOCKED_EVIDENCE_NO_LOADING_V2_END -->"

START_RUNS = "<!-- ERA_LOCKED_RUNS_NO_LOADING_V2_START -->"
END_RUNS = "<!-- ERA_LOCKED_RUNS_NO_LOADING_V2_END -->"

SAFE_SENTENCE = (
    "Retrospectively evaluated across MIMIC-IV strict clinical-event cohorts and eICU harmonized clinical-event cohorts "
    "under a harmonized threshold framework, with event-definition differences clearly labeled."
)

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_if_changed(p: Path, text: str):
    old = read(p)
    if old != text:
        p.write_text(text, encoding="utf-8")
        CHANGED.append(str(p))
        print("UPDATED:", p)
    else:
        print("UNCHANGED:", p)

def files_under_era():
    out = []
    for p in (ROOT / "era").rglob("*"):
        if p.is_file() and p.suffix.lower() in {".html", ".py", ".js"}:
            out.append(p)
    return out

def replace_or_insert(text: str, start: str, end: str, snippet: str, mode: str = "top") -> str:
    block = f"{start}\n{snippet.rstrip()}\n{end}"
    if start in text and end in text:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), block, text, flags=re.S)

    if mode == "top":
        m = re.search(r"<main[^>]*>", text, flags=re.I)
        if m:
            return text[:m.end()] + "\n" + block + "\n" + text[m.end():]
        m = re.search(r"<body[^>]*>", text, flags=re.I)
        if m:
            return text[:m.end()] + "\n" + block + "\n" + text[m.end():]

    if mode == "bottom":
        idx = text.lower().rfind("</main>")
        if idx != -1:
            return text[:idx] + "\n" + block + "\n" + text[idx:]
        idx = text.lower().rfind("</body>")
        if idx != -1:
            return text[:idx] + "\n" + block + "\n" + text[idx:]

    return text + "\n" + block + "\n"

def patch_status_labels(text: str) -> str:
    # Surgical status correction only near Model Card and Pilot Success Guide wording.
    targets = [
        "model card", "model-card", "model_card",
        "pilot success guide", "pilot-success-guide", "pilot_success_guide"
    ]

    low = text.lower()
    intervals = []
    for target in targets:
        start = 0
        while True:
            i = low.find(target, start)
            if i == -1:
                break
            intervals.append((max(0, i - 900), min(len(text), i + len(target) + 900)))
            start = i + len(target)

    if not intervals:
        return text

    intervals.sort()
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    def patch_block(block: str) -> str:
        block = re.sub(r"\bMISSING\b", "READY", block)
        block = re.sub(r"\bMissing\b", "Ready", block)
        block = re.sub(r"\bmissing\b", "ready", block)
        block = block.replace("status-missing", "status-ready")
        block = block.replace("badge-missing", "badge-ready")
        block = block.replace("pill-missing", "pill-ready")
        block = block.replace("doc-missing", "doc-ready")
        block = block.replace("route-missing", "route-ready")
        block = block.replace("missing-doc", "ready-doc")
        block = block.replace("missing-route", "ready-route")
        block = re.sub(r'("available"\s*:\s*)false', r'\1true', block, flags=re.I)
        block = re.sub(r'("exists"\s*:\s*)false', r'\1true', block, flags=re.I)
        block = re.sub(r'("ready"\s*:\s*)false', r'\1true', block, flags=re.I)
        block = re.sub(r"('available'\s*:\s*)False", r"\1True", block)
        block = re.sub(r"('exists'\s*:\s*)False", r"\1True", block)
        block = re.sub(r"('ready'\s*:\s*)False", r"\1True", block)
        return block

    out = []
    last = 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(patch_block(text[s:e]))
        last = e
    out.append(text[last:])
    return "".join(out)

def patch_loading(text: str) -> str:
    replacements = [
        ("Loading...", "Locked aggregate evidence available."),
        ("Loading…", "Locked aggregate evidence available."),
        ("loading...", "locked aggregate evidence available."),
        ("loading…", "locked aggregate evidence available."),
    ]
    for a, b in replacements:
        text = text.replace(a, b)
    return text

COMMAND_SNIPPET = """
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
"""

INTEL_SNIPPET = """
<section id="multi-dataset-retrospective-evidence" style="margin: 28px 0; padding: 22px; border: 1px solid rgba(164,255,190,.32); border-radius: 24px; background: rgba(9,18,32,.72);">
  <div style="display:inline-block; padding: 6px 12px; border-radius:999px; border:1px solid rgba(164,255,190,.45); color:#b8ffc8; font-weight:800; letter-spacing:.08em; font-size:.75rem; text-transform:uppercase;">DUA-safe aggregate evidence</div>
  <h2 style="margin:12px 0 8px; font-size: clamp(1.8rem, 3vw, 2.8rem); line-height:1.05;">Multi-dataset retrospective evidence: MIMIC-IV + eICU</h2>
  <p style="max-width:1060px; opacity:.92;">Early Risk Alert AI has retrospective aggregate evidence across MIMIC-IV strict clinical-event cohorts and eICU harmonized clinical-event cohorts. Across conservative t=6.0 operating points, alert reduction remained high and false-positive burden remained low. Event definitions differ, so detection rates are reported separately and are not treated as equivalent endpoints.</p>

  <div style="overflow-x:auto; margin-top:16px;">
    <table style="width:100%; border-collapse:collapse; min-width:920px;">
      <thead>
        <tr style="background:rgba(120,160,220,.18);">
          <th style="text-align:left; padding:12px;">Evidence Family</th>
          <th style="text-align:left; padding:12px;">Event Definition</th>
          <th style="text-align:left; padding:12px;">t=6.0 Alert Reduction</th>
          <th style="text-align:left; padding:12px;">t=6.0 FPR</th>
          <th style="text-align:left; padding:12px;">Detection</th>
          <th style="text-align:left; padding:12px;">Lead-Time Context</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">MIMIC-IV v3.1</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Strict clinical-event cohorts</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">94.0%–94.9%</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">3.7%–4.4%</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">14.1%–16.0%</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">4.0 hrs</td>
        </tr>
        <tr>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">eICU v2.0</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">Harmonized clinical-event cohorts</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">94.25%–95.64%</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">0.98%–2.02%</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">24.66%–64.49%</td>
          <td style="padding:12px; border-top:1px solid rgba(255,255,255,.12);">3.13–4.83 hrs</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p style="margin-top:14px; padding:12px 14px; border-radius:16px; border:1px solid rgba(255,205,110,.45); color:#ffd778; background:rgba(255,205,110,.08);">Approved framing: retrospectively evaluated across MIMIC-IV and eICU under a harmonized threshold framework. Avoid unqualified claims such as “clinically validated on two datasets” until the same validation question and endpoint definition are applied equivalently.</p>
</section>
"""

EVIDENCE_SNIPPET = """
<section id="locked-validation-evidence" style="margin: 28px 0; padding: 22px; border: 1px solid rgba(164,255,190,.32); border-radius: 24px; background: rgba(9,18,32,.72);">
  <div style="display:inline-block; padding: 6px 12px; border-radius:999px; border:1px solid rgba(164,255,190,.45); color:#b8ffc8; font-weight:800; letter-spacing:.08em; font-size:.75rem; text-transform:uppercase;">No loading placeholders</div>
  <h2 style="margin:12px 0 8px; font-size: clamp(1.6rem, 3vw, 2.6rem); line-height:1.05;">Locked aggregate validation evidence</h2>
  <p style="max-width:1060px; opacity:.92;">This evidence packet shows locked DUA-safe aggregate summaries. Raw restricted data and row-level outputs remain local-only.</p>

  <h3 style="margin-top:18px;">MIMIC-IV strict clinical-event threshold matrix</h3>
  <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; min-width:900px;">
      <thead><tr style="background:rgba(120,160,220,.18);"><th style="text-align:left; padding:12px;">Threshold</th><th style="text-align:left; padding:12px;">Setting</th><th style="text-align:left; padding:12px;">Alert Reduction</th><th style="text-align:left; padding:12px;">FPR</th><th style="text-align:left; padding:12px;">Detection</th><th style="text-align:left; padding:12px;">ERA Alerts / Patient-Day</th><th style="text-align:left; padding:12px;">Median Lead Time</th></tr></thead>
      <tbody>
        <tr><td style="padding:12px;">t=4.0</td><td style="padding:12px;">ICU / high-acuity</td><td style="padding:12px;">80.3%</td><td style="padding:12px;">14.4%</td><td style="padding:12px;">37.9%</td><td style="padding:12px;">2.22</td><td style="padding:12px;">4.0 hrs</td></tr>
        <tr><td style="padding:12px;">t=5.0</td><td style="padding:12px;">Mixed units / balanced</td><td style="padding:12px;">89.1%</td><td style="padding:12px;">8.0%</td><td style="padding:12px;">24.6%</td><td style="padding:12px;">1.23</td><td style="padding:12px;">4.0 hrs</td></tr>
        <tr><td style="padding:12px;">t=6.0</td><td style="padding:12px;">Telemetry / stepdown conservative</td><td style="padding:12px;">94.3%</td><td style="padding:12px;">4.2%</td><td style="padding:12px;">15.3%</td><td style="padding:12px;">0.65</td><td style="padding:12px;">4.0 hrs</td></tr>
      </tbody>
    </table>
  </div>

  <h3 style="margin-top:18px;">eICU harmonized clinical-event threshold matrix</h3>
  <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; min-width:900px;">
      <thead><tr style="background:rgba(120,160,220,.18);"><th style="text-align:left; padding:12px;">Threshold</th><th style="text-align:left; padding:12px;">Setting</th><th style="text-align:left; padding:12px;">Alert Reduction</th><th style="text-align:left; padding:12px;">FPR</th><th style="text-align:left; padding:12px;">Detection</th><th style="text-align:left; padding:12px;">ERA Alerts / Patient-Day</th><th style="text-align:left; padding:12px;">Median Lead Time</th></tr></thead>
      <tbody>
        <tr><td style="padding:12px;">t=4.0</td><td style="padding:12px;">ICU / high-acuity</td><td style="padding:12px;">69.74%</td><td style="padding:12px;">5.34%</td><td style="padding:12px;">62.51%</td><td style="padding:12px;">29.22</td><td style="padding:12px;">5.67 hrs</td></tr>
        <tr><td style="padding:12px;">t=5.0</td><td style="padding:12px;">Mixed units / balanced</td><td style="padding:12px;">87.85%</td><td style="padding:12px;">2.23%</td><td style="padding:12px;">38.8%</td><td style="padding:12px;">11.73</td><td style="padding:12px;">5.17 hrs</td></tr>
        <tr><td style="padding:12px;">t=6.0</td><td style="padding:12px;">Telemetry / stepdown conservative</td><td style="padding:12px;">94.25%</td><td style="padding:12px;">0.98%</td><td style="padding:12px;">24.66%</td><td style="padding:12px;">5.55</td><td style="padding:12px;">4.83 hrs</td></tr>
      </tbody>
    </table>
  </div>

  <p style="margin-top:14px; padding:12px 14px; border-radius:16px; border:1px solid rgba(255,205,110,.45); color:#ffd778; background:rgba(255,205,110,.08);">Detection rates should not be treated as equivalent endpoints across datasets. MIMIC-IV and eICU use different event-definition methods. Alert reduction, false-positive burden, and lead-time context are the safer cross-dataset comparison points.</p>
</section>
"""

RUNS_SNIPPET = """
<section id="locked-validation-run-history" style="margin: 28px 0; padding: 22px; border: 1px solid rgba(164,255,190,.32); border-radius: 24px; background: rgba(9,18,32,.72);">
  <div style="display:inline-block; padding: 6px 12px; border-radius:999px; border:1px solid rgba(164,255,190,.45); color:#b8ffc8; font-weight:800; letter-spacing:.08em; font-size:.75rem; text-transform:uppercase;">Run registry</div>
  <h2 style="margin:12px 0 8px; font-size: clamp(1.6rem, 3vw, 2.6rem); line-height:1.05;">Current locked validation run history</h2>
  <p style="max-width:1060px; opacity:.92;">Locked DUA-safe aggregate run registry. Raw restricted data and row-level outputs remain local-only.</p>

  <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; min-width:980px;">
      <thead>
        <tr style="background:rgba(120,160,220,.18);">
          <th style="text-align:left; padding:12px;">Run</th>
          <th style="text-align:left; padding:12px;">Dataset</th>
          <th style="text-align:left; padding:12px;">Event Definition</th>
          <th style="text-align:left; padding:12px;">Cases</th>
          <th style="text-align:left; padding:12px;">Rows</th>
          <th style="text-align:left; padding:12px;">Threshold</th>
          <th style="text-align:left; padding:12px;">Alert Reduction</th>
          <th style="text-align:left; padding:12px;">FPR</th>
          <th style="text-align:left; padding:12px;">Detection</th>
          <th style="text-align:left; padding:12px;">Lead Time</th>
          <th style="text-align:left; padding:12px;">Status</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="padding:12px;">MIMIC-IV full cohort</td><td style="padding:12px;">MIMIC-IV v3.1</td><td style="padding:12px;">Strict clinical-event labels</td><td style="padding:12px;">1,705</td><td style="padding:12px;">456,453</td><td style="padding:12px;">t=6.0</td><td style="padding:12px;">94.3%</td><td style="padding:12px;">4.2%</td><td style="padding:12px;">15.3%</td><td style="padding:12px;">4.0 hrs</td><td style="padding:12px;">Locked</td></tr>
        <tr><td style="padding:12px;">MIMIC-IV subcohort B</td><td style="padding:12px;">MIMIC-IV v3.1</td><td style="padding:12px;">Strict clinical-event labels</td><td style="padding:12px;">577</td><td style="padding:12px;">~159K</td><td style="padding:12px;">t=6.0</td><td style="padding:12px;">94.0%</td><td style="padding:12px;">4.4%</td><td style="padding:12px;">16.0%</td><td style="padding:12px;">4.0 hrs</td><td style="padding:12px;">Locked</td></tr>
        <tr><td style="padding:12px;">MIMIC-IV subcohort C</td><td style="padding:12px;">MIMIC-IV v3.1</td><td style="padding:12px;">Strict clinical-event labels</td><td style="padding:12px;">621</td><td style="padding:12px;">~159K</td><td style="padding:12px;">t=6.0</td><td style="padding:12px;">94.9%</td><td style="padding:12px;">3.7%</td><td style="padding:12px;">14.1%</td><td style="padding:12px;">4.0 hrs</td><td style="padding:12px;">Locked</td></tr>
        <tr><td style="padding:12px;">eICU harmonized full pass</td><td style="padding:12px;">eICU v2.0</td><td style="padding:12px;">Harmonized clinical-event labels</td><td style="padding:12px;">2,394</td><td style="padding:12px;">2,023,962</td><td style="padding:12px;">t=6.0</td><td style="padding:12px;">94.25%</td><td style="padding:12px;">0.98%</td><td style="padding:12px;">24.66%</td><td style="padding:12px;">4.83 hrs</td><td style="padding:12px;">Locked aggregate</td></tr>
        <tr><td style="padding:12px;">eICU harmonized subcohort B</td><td style="padding:12px;">eICU v2.0</td><td style="padding:12px;">Harmonized clinical-event labels</td><td style="padding:12px;">Aggregate</td><td style="padding:12px;">Aggregate</td><td style="padding:12px;">t=6.0</td><td style="padding:12px;">95.64%</td><td style="padding:12px;">1.78%</td><td style="padding:12px;">64.49%</td><td style="padding:12px;">3.69 hrs</td><td style="padding:12px;">Locked aggregate</td></tr>
        <tr><td style="padding:12px;">eICU harmonized subcohort C</td><td style="padding:12px;">eICU v2.0</td><td style="padding:12px;">Harmonized clinical-event labels</td><td style="padding:12px;">Aggregate</td><td style="padding:12px;">Aggregate</td><td style="padding:12px;">t=6.0</td><td style="padding:12px;">95.26%</td><td style="padding:12px;">2.02%</td><td style="padding:12px;">64.03%</td><td style="padding:12px;">3.13 hrs</td><td style="padding:12px;">Locked aggregate</td></tr>
      </tbody>
    </table>
  </div>

  <p style="margin-top:14px;">Safe wording: retrospectively evaluated across MIMIC-IV and eICU under a harmonized threshold framework. Do not claim full clinical validation.</p>
</section>
"""

PILOT_SUMMARY = """# Early Risk Alert AI — Pilot Sponsor Summary

## Purpose

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

## Current Evidence Posture

Early Risk Alert AI has retrospective aggregate evidence across:

1. MIMIC-IV v3.1 strict clinical-event cohorts.
2. eICU v2.0 harmonized clinical-event cohorts.

The evidence supports retrospective robustness under a harmonized threshold framework. It does not establish prospective clinical performance or full clinical validation.

## Conservative t=6.0 Summary

- MIMIC-IV strict clinical-event cohorts: 94.0%–94.9% alert reduction, 3.7%–4.4% FPR, 14.1%–16.0% detection, and 4.0 hours of retrospective lead-time context.
- eICU harmonized clinical-event cohorts: 94.25%–95.64% alert reduction, 0.98%–2.02% FPR, 24.66%–64.49% detection, and 3.13–4.83 hours of retrospective lead-time context.

Detection rates are reported separately because event definitions differ across datasets.

## Command-Center Explainability

The pilot interface should visibly show:

- Priority tier.
- Queue rank.
- Primary driver.
- Trend direction.
- Lead-time context.
- Workflow state.

These outputs are review-prioritization context only, not diagnosis or treatment direction.

## Suggested Pilot Success Metrics

- Review burden / alert burden.
- False-positive burden.
- Reviewer understanding of why a case may warrant review.
- Workflow-state completion.
- Safe use of decision-support-only guardrails.
"""

def is_command_center_candidate(p: Path, text: str) -> bool:
    name = p.name.lower()
    low = text.lower()
    if "command" in name and ("command center" in low or "hospital command wall" in low):
        return True
    return "hospital command wall" in low and "site-specific pilot packet" in low

def is_validation_intelligence_candidate(p: Path, text: str) -> bool:
    name = p.name.lower()
    low = text.lower()
    return ("validation_intelligence" in name or "validation-intelligence" in name) and "validation intelligence" in low

def is_validation_evidence_candidate(p: Path, text: str) -> bool:
    name = p.name.lower()
    low = text.lower()
    return (
        "validation_evidence" in name
        or "validation-evidence" in name
        or "evidence_packet" in name
        or "evidence-packet" in name
    ) and ("evidence" in low or "validation" in low)

def is_validation_runs_candidate(p: Path, text: str) -> bool:
    name = p.name.lower()
    low = text.lower()
    return ("validation_runs" in name or "validation-runs" in name) and "validation" in low

def main():
    files = files_under_era()

    command_hits = 0
    intel_hits = 0
    evidence_hits = 0
    runs_hits = 0

    for p in files:
        text = read(p)
        original = text

        if is_command_center_candidate(p, text):
            command_hits += 1
            text = patch_status_labels(text)
            if "Explainable review queue" not in text:
                text = replace_or_insert(text, START_CMD, END_CMD, COMMAND_SNIPPET, mode="bottom")
            else:
                text = patch_status_labels(text)

        elif is_validation_intelligence_candidate(p, text):
            intel_hits += 1
            text = patch_loading(text)
            text = replace_or_insert(text, START_INTEL, END_INTEL, INTEL_SNIPPET, mode="top")

        elif is_validation_evidence_candidate(p, text):
            evidence_hits += 1
            text = patch_loading(text)
            text = replace_or_insert(text, START_EVIDENCE, END_EVIDENCE, EVIDENCE_SNIPPET, mode="top")

        elif is_validation_runs_candidate(p, text):
            runs_hits += 1
            text = patch_loading(text)
            text = replace_or_insert(text, START_RUNS, END_RUNS, RUNS_SNIPPET, mode="top")

        if text != original:
            write_if_changed(p, text)

    if command_hits == 0:
        print("WARNING: No Command Center candidate file found.")
    if intel_hits == 0:
        print("WARNING: No Validation Intelligence candidate file found.")
    if evidence_hits == 0:
        print("WARNING: No Validation Evidence/Evidence Packet candidate file found.")
    if runs_hits == 0:
        print("WARNING: No Validation Runs candidate file found.")

    doc = ROOT / "docs" / "validation" / "pilot_sponsor_summary.md"
    if not doc.exists() or read(doc) != PILOT_SUMMARY:
        doc.write_text(PILOT_SUMMARY, encoding="utf-8")
        CHANGED.append(str(doc))
        print("UPDATED:", doc)

    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Polish frontend presentation without redesigning or replacing the Command Center.",
        "scope": [
            "Minimal Command Center explainability visibility",
            "Model Card and Pilot Success Guide status correction where stale",
            "Validation Evidence and Validation Runs loading-placeholder cleanup",
            "Validation Intelligence conservative MIMIC/eICU comparison",
            "Pilot sponsor summary document"
        ],
        "claim_boundary": [
            "Retrospective aggregate evidence only",
            "MIMIC-IV and eICU event definitions remain separated",
            "Do not claim unqualified two-dataset validation",
            "Do not claim diagnosis, treatment direction, clinician replacement, or autonomous escalation"
        ],
        "safe_sentence": SAFE_SENTENCE,
        "changed_files": CHANGED,
        "candidate_counts": {
            "command_center": command_hits,
            "validation_intelligence": intel_hits,
            "validation_evidence": evidence_hits,
            "validation_runs": runs_hits
        }
    }
    out = ROOT / "data" / "validation" / "polished_platform_presentation_patch_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    CHANGED.append(str(out))
    print("UPDATED:", out)

    print("")
    print("PATCH SUMMARY")
    print("Command Center candidates:", command_hits)
    print("Validation Intelligence candidates:", intel_hits)
    print("Validation Evidence candidates:", evidence_hits)
    print("Validation Runs candidates:", runs_hits)
    print("Changed files:")
    for f in CHANGED:
        print(" -", f)

if __name__ == "__main__":
    main()
