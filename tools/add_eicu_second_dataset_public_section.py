#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

DATA = Path("data/validation")
WEB = Path("era/web")
DOCS = Path("docs/validation")

HTML_START = "<!-- ERA_EICU_SECOND_DATASET_PUBLIC_V1_START -->"
HTML_END = "<!-- ERA_EICU_SECOND_DATASET_PUBLIC_V1_END -->"

MD_START = "<!-- ERA_EICU_SECOND_DATASET_PUBLIC_V1_START -->"
MD_END = "<!-- ERA_EICU_SECOND_DATASET_PUBLIC_V1_END -->"

summary_path = DATA / "eicu_validation_summary.json"
if not summary_path.exists():
    raise SystemExit("ERROR: data/validation/eicu_validation_summary.json not found.")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
runs = summary.get("runs", [])

if len(runs) < 3:
    raise SystemExit("ERROR: expected at least 3 eICU threshold runs.")

def n(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def fmt(x, suffix=""):
    y = n(x)
    if y is None:
        return "—"
    if abs(y - round(y)) < 0.001:
        s = f"{y:.0f}"
    else:
        s = f"{y:.2f}".rstrip("0").rstrip(".")
    return s + suffix

def get_t(threshold):
    for r in runs:
        try:
            if float(r.get("threshold")) == float(threshold):
                return r
        except Exception:
            pass
    return {}

t4 = get_t(4.0)
t5 = get_t(5.0)
t6 = get_t(6.0)

def row_html(r):
    return f"""
      <tr>
        <td><strong>t={fmt(r.get("threshold"))}</strong></td>
        <td>{r.get("suggested_setting") or "—"}</td>
        <td>{fmt(r.get("rows"))}</td>
        <td>{fmt(r.get("case_count"))}</td>
        <td>{fmt(r.get("event_count"))}</td>
        <td>{fmt(r.get("alert_reduction_pct"), "%")}</td>
        <td>{fmt(r.get("era_fpr_pct"), "%")}</td>
        <td>{fmt(r.get("detection_pct"), "%")}</td>
        <td>{fmt(r.get("median_lead_time_hours"), " hrs")}</td>
        <td>{fmt(r.get("era_alerts_per_patient_day"))}</td>
      </tr>
    """

rows_html = "\n".join(row_html(r) for r in sorted(runs, key=lambda x: x.get("threshold", 999)))

best_t6_alert = fmt(t6.get("alert_reduction_pct"), "%")
best_t6_fpr = fmt(t6.get("era_fpr_pct"), "%")
best_t6_det = fmt(t6.get("detection_pct"), "%")
best_t6_lead = fmt(t6.get("median_lead_time_hours"), " hrs")

headline = (
    f"On a separate eICU outcome-proxy cohort, ERA preserved the same threshold pattern: "
    f"lower thresholds produced higher detection, while the conservative t=6.0 setting emphasized low-burden review behavior."
)

html_block = f"""
{HTML_START}
<section class="card full" style="
  border:1px solid rgba(139,184,255,.30);
  border-left:5px solid #8bb8ff;
  background:linear-gradient(145deg, rgba(13,24,42,.92), rgba(6,11,21,.92));
  border-radius:26px;
  padding:24px;
  margin-top:18px;
">
  <div style="
    color:#8bb8ff;
    text-transform:uppercase;
    letter-spacing:.14em;
    font-size:11px;
    font-weight:950;
    margin-bottom:8px;
  ">Second-Dataset Outcome-Proxy Check</div>

  <h2 style="margin:0 0 12px;">eICU Retrospective Outcome-Proxy Validation</h2>

  <p class="callout" style="
    border-left:4px solid #8bb8ff;
    padding:12px 16px;
    border-radius:14px;
    background:rgba(139,184,255,.09);
    color:#eaf4ff;
    font-weight:800;
  ">
    {headline}
  </p>

  <p class="muted">
    This eICU run is intentionally framed separately from the locked MIMIC-IV evidence release because it uses outcome-proxy event labels from eICU mortality/discharge status fields and relative ICU offsets.
  </p>

  <div class="metrics" style="margin-top:16px;">
    <div class="metric">
      <span>t=6.0 Alert Reduction</span>
      <strong>{best_t6_alert}</strong>
      <em>eICU outcome-proxy cohort</em>
    </div>
    <div class="metric">
      <span>t=6.0 ERA FPR</span>
      <strong>{best_t6_fpr}</strong>
      <em>conservative setting</em>
    </div>
    <div class="metric">
      <span>t=6.0 Detection</span>
      <strong>{best_t6_det}</strong>
      <em>outcome-proxy event context</em>
    </div>
    <div class="metric">
      <span>t=6.0 Lead-Time Context</span>
      <strong>{best_t6_lead}</strong>
      <em>retrospective timing context</em>
    </div>
  </div>

  <h3 style="margin-top:20px;">eICU Threshold Matrix</h3>
  <table>
    <thead>
      <tr>
        <th>Threshold</th>
        <th>Setting</th>
        <th>Rows</th>
        <th>Cases</th>
        <th>Event-Proxy Rows</th>
        <th>Alert Reduction</th>
        <th>ERA FPR</th>
        <th>Detection</th>
        <th>Lead Time</th>
        <th>Alerts / Patient-Day</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <p class="muted" style="margin-top:14px;">
    Safe claim: cross-dataset retrospective outcome-proxy evaluation on de-identified ICU data. Avoid claiming proven generalizability, prospective validation, diagnosis, treatment direction, prevention, or autonomous escalation.
  </p>

  <div class="era-release-lock-links" style="margin-top:12px;">
    <a href="/api/validation/eicu-validation">eICU Validation JSON</a>
    <a href="/validation-evidence/eicu-validation.json">Download eICU Summary</a>
  </div>
</section>
{HTML_END}
"""

md_block = f"""
{MD_START}

## eICU Second-Dataset Outcome-Proxy Check

**{headline}**

This eICU run is intentionally framed separately from the locked MIMIC-IV evidence release because it uses outcome-proxy event labels from eICU mortality/discharge status fields and relative ICU offsets.

| Threshold | Setting | Rows | Cases | Event-Proxy Rows | Alert Reduction | ERA FPR | Detection | Lead Time | Alerts / Patient-Day |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""

for r in sorted(runs, key=lambda x: x.get("threshold", 999)):
    md_block += (
        f"| t={fmt(r.get('threshold'))} "
        f"| {r.get('suggested_setting') or '—'} "
        f"| {fmt(r.get('rows'))} "
        f"| {fmt(r.get('case_count'))} "
        f"| {fmt(r.get('event_count'))} "
        f"| {fmt(r.get('alert_reduction_pct'), '%')} "
        f"| {fmt(r.get('era_fpr_pct'), '%')} "
        f"| {fmt(r.get('detection_pct'), '%')} "
        f"| {fmt(r.get('median_lead_time_hours'), ' hrs')} "
        f"| {fmt(r.get('era_alerts_per_patient_day'))} |\n"
    )

md_block += """

Safe claim: cross-dataset retrospective outcome-proxy evaluation on de-identified ICU data.

Avoid claiming proven generalizability, prospective validation, diagnosis, treatment direction, prevention, or autonomous escalation.

""" + MD_END + "\n"

def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text

def patch_html(path: Path):
    if not path.exists():
        raise SystemExit(f"ERROR: missing file {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = replace_block(text, HTML_START, HTML_END, "")

    idx = text.lower().rfind("</main>")
    if idx == -1:
        idx = text.lower().rfind("</body>")
    if idx == -1:
        raise SystemExit(f"ERROR: no insertion point found in {path}")

    text = text[:idx] + "\n" + html_block + "\n" + text[idx:]
    path.write_text(text, encoding="utf-8")
    print("PATCHED:", path)

def patch_md(path: Path):
    if not path.exists():
        raise SystemExit(f"ERROR: missing file {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = replace_block(text, MD_START, MD_END, "")

    text = text.rstrip() + "\n\n" + md_block
    path.write_text(text, encoding="utf-8")
    print("PATCHED:", path)

patch_html(WEB / "validation_intelligence.html")
patch_html(WEB / "validation_evidence.html")
patch_md(Path("docs/validation/eicu_validation_summary.md"))

progress = DOCS / "eicu_second_dataset_public_section.md"
progress.write_text(
    f"""# eICU Second-Dataset Public Section

Generated: {datetime.now(timezone.utc).isoformat()}

## Added Public Framing

{headline}

## Updated Pages

- /validation-intelligence
- /validation-evidence

## Guardrail

The eICU result is framed as a second-dataset retrospective outcome-proxy check, not as prospective validation or proven generalizability.
""",
    encoding="utf-8",
)

print("WROTE:", progress)
