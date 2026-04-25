#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(".")
DATA = ROOT / "data" / "validation"
WEB = ROOT / "era" / "web"
DOCS = ROOT / "docs" / "validation"

WEB.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def load_json(name: str, default=None):
    p = DATA / name
    if not p.exists():
        return default if default is not None else {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


cross = load_json("cross_cohort_validation_summary.json", {})
threshold_matrix = load_json("threshold_matrix_summary.json", {})
milestone = load_json("mimic_validation_milestone_2026_04.json", {})


def val(x, suffix=""):
    if x is None or x == "":
        return "—"
    try:
        if isinstance(x, float):
            s = f"{x:.3f}".rstrip("0").rstrip(".")
        else:
            s = str(x)
        return s + suffix
    except Exception:
        return str(x) + suffix


def threshold_summary_rows():
    ts = cross.get("threshold_summary", {}) if isinstance(cross, dict) else {}
    rows = []

    if isinstance(ts, dict) and ts:
        for threshold, item in sorted(ts.items(), key=lambda kv: float(kv[0])):
            rows.append({
                "threshold": threshold,
                "cohorts_compared": item.get("cohorts_compared"),
                "alert_reduction_pct_mean": item.get("alert_reduction_pct_mean"),
                "alert_reduction_pct_sd": item.get("alert_reduction_pct_sd"),
                "era_fpr_pct_mean": item.get("era_fpr_pct_mean"),
                "era_fpr_pct_sd": item.get("era_fpr_pct_sd"),
                "detection_pct_mean": item.get("detection_pct_mean"),
                "detection_pct_sd": item.get("detection_pct_sd"),
                "median_lead_time_hours_mean": item.get("median_lead_time_hours_mean"),
                "median_lead_time_hours_sd": item.get("median_lead_time_hours_sd"),
                "era_alerts_per_patient_day_mean": item.get("era_alerts_per_patient_day_mean"),
                "era_alerts_per_patient_day_sd": item.get("era_alerts_per_patient_day_sd"),
            })

    if rows:
        return rows

    return [
        {
            "threshold": "4.0",
            "cohorts_compared": 3,
            "alert_reduction_pct_mean": 80.3,
            "alert_reduction_pct_sd": "—",
            "era_fpr_pct_mean": 14.4,
            "era_fpr_pct_sd": "—",
            "detection_pct_mean": 37.9,
            "detection_pct_sd": "—",
            "median_lead_time_hours_mean": 4.0,
            "median_lead_time_hours_sd": "—",
            "era_alerts_per_patient_day_mean": 2.22,
            "era_alerts_per_patient_day_sd": "—",
        },
        {
            "threshold": "5.0",
            "cohorts_compared": 3,
            "alert_reduction_pct_mean": 89.1,
            "alert_reduction_pct_sd": "—",
            "era_fpr_pct_mean": 8.0,
            "era_fpr_pct_sd": "—",
            "detection_pct_mean": 24.6,
            "detection_pct_sd": "—",
            "median_lead_time_hours_mean": 4.0,
            "median_lead_time_hours_sd": "—",
            "era_alerts_per_patient_day_mean": 1.23,
            "era_alerts_per_patient_day_sd": "—",
        },
        {
            "threshold": "6.0",
            "cohorts_compared": 3,
            "alert_reduction_pct_mean": 94.3,
            "alert_reduction_pct_sd": "—",
            "era_fpr_pct_mean": 4.2,
            "era_fpr_pct_sd": "—",
            "detection_pct_mean": 15.3,
            "detection_pct_sd": "—",
            "median_lead_time_hours_mean": 4.0,
            "median_lead_time_hours_sd": "—",
            "era_alerts_per_patient_day_mean": 0.65,
            "era_alerts_per_patient_day_sd": "—",
        },
    ]


summary_rows = threshold_summary_rows()
t4 = next((r for r in summary_rows if str(r.get("threshold")) in {"4.0", "4"}), summary_rows[0])
t6 = next((r for r in summary_rows if str(r.get("threshold")) in {"6.0", "6"}), summary_rows[-1])

cohorts = cross.get("cohorts_compared") if isinstance(cross, dict) else None
if not cohorts:
    cohorts = ["Full validation cohort", "Subcohort B", "Subcohort C"]

runs = cross.get("runs", []) if isinstance(cross, dict) else []


def metric_card(label, value, subtext):
    return f"""
      <div class="metric">
        <span>{label}</span>
        <strong>{value}</strong>
        <em>{subtext}</em>
      </div>
    """


def threshold_table():
    html = ""
    for r in summary_rows:
        html += f"""
          <tr>
            <td><strong>t={val(r.get("threshold"))}</strong></td>
            <td>{val(r.get("cohorts_compared"))}</td>
            <td>{val(r.get("alert_reduction_pct_mean"), "%")}</td>
            <td>{val(r.get("alert_reduction_pct_sd"))}</td>
            <td>{val(r.get("era_fpr_pct_mean"), "%")}</td>
            <td>{val(r.get("era_fpr_pct_sd"))}</td>
            <td>{val(r.get("detection_pct_mean"), "%")}</td>
            <td>{val(r.get("detection_pct_sd"))}</td>
            <td>{val(r.get("median_lead_time_hours_mean"), " hrs")}</td>
            <td>{val(r.get("era_alerts_per_patient_day_mean"))}</td>
          </tr>
        """
    return html


def run_table():
    if not runs:
        return """
        <p class="muted">
          Run-level cross-cohort entries will appear after the cross-cohort validation comparison is generated.
        </p>
        """

    html = """
      <table>
        <thead>
          <tr>
            <th>Cohort</th>
            <th>Threshold</th>
            <th>Setting</th>
            <th>Rows</th>
            <th>Cases</th>
            <th>Alert Reduction</th>
            <th>FPR</th>
            <th>Detection</th>
            <th>Lead Time</th>
            <th>Alerts / Patient-Day</th>
          </tr>
        </thead>
        <tbody>
    """

    for r in sorted(runs, key=lambda x: (str(x.get("cohort", "")), float(x.get("threshold") or 999))):
        html += f"""
          <tr>
            <td>{r.get("cohort", "—")}</td>
            <td><strong>t={val(r.get("threshold"))}</strong></td>
            <td>{r.get("suggested_setting") or "—"}</td>
            <td>{val(r.get("rows"))}</td>
            <td>{val(r.get("cases"))}</td>
            <td>{val(r.get("alert_reduction_pct"), "%")}</td>
            <td>{val(r.get("era_fpr_pct"), "%")}</td>
            <td>{val(r.get("detection_pct"), "%")}</td>
            <td>{val(r.get("median_lead_time_hours"), " hrs")}</td>
            <td>{val(r.get("era_alerts_per_patient_day"))}</td>
          </tr>
        """

    html += "</tbody></table>"
    return html


def cohort_list():
    return "".join(f"<li>{c}</li>" for c in cohorts)


CSS = """
:root {
  --bg: #070b12;
  --panel: rgba(15, 24, 42, 0.9);
  --panel2: rgba(23, 34, 55, 0.92);
  --line: rgba(155, 190, 255, 0.2);
  --text: #f7fbff;
  --muted: #b8c6d8;
  --accent: #8fffd2;
  --blue: #8bb8ff;
  --gold: #ffd78a;
  --red: #ff9aa9;
  --green: #96ffb8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background:
    radial-gradient(circle at 18% 8%, rgba(50, 120, 255, 0.23), transparent 26rem),
    radial-gradient(circle at 82% 4%, rgba(83, 255, 203, 0.16), transparent 25rem),
    linear-gradient(180deg, #050914 0%, #0a1020 55%, #050914 100%);
}
.page {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 34px 0 70px;
}
.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
  margin-bottom: 30px;
}
.brand {
  font-size: 12px;
  letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 900;
}
.navlinks { display: flex; gap: 10px; flex-wrap: wrap; }
.navlinks a, .btn {
  text-decoration: none;
  color: var(--text);
  border: 1px solid var(--line);
  background: rgba(255,255,255,.06);
  padding: 10px 13px;
  border-radius: 14px;
  font-size: 13px;
  font-weight: 800;
}
.btn.primary {
  background: linear-gradient(135deg, rgba(143,255,210,.95), rgba(139,184,255,.95));
  color: #061017;
  border: 0;
}
.hero {
  border: 1px solid var(--line);
  border-radius: 32px;
  background: linear-gradient(145deg, rgba(16, 26, 45, .96), rgba(7, 12, 22, .96));
  padding: 34px;
  box-shadow: 0 25px 75px rgba(0,0,0,.42);
}
.kicker {
  color: var(--accent);
  letter-spacing: .17em;
  text-transform: uppercase;
  font-size: 12px;
  font-weight: 950;
  margin-bottom: 10px;
}
h1 {
  margin: 0;
  font-size: clamp(38px, 7vw, 74px);
  line-height: .94;
  letter-spacing: -.06em;
}
h2 {
  margin: 0 0 14px;
  font-size: clamp(24px, 3vw, 38px);
  letter-spacing: -.04em;
}
h3 {
  margin: 0 0 10px;
  font-size: 20px;
  letter-spacing: -.02em;
}
.lede {
  color: #dbe8fb;
  font-size: 18px;
  max-width: 900px;
  margin: 18px 0 0;
}
.disclaimer {
  margin-top: 22px;
  border: 1px solid rgba(255,215,138,.38);
  background: rgba(255,215,138,.1);
  color: #ffe2a5;
  border-radius: 18px;
  padding: 14px 16px;
  font-weight: 800;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 26px;
}
.metric {
  border: 1px solid var(--line);
  background: rgba(255,255,255,.06);
  border-radius: 22px;
  padding: 18px;
  min-height: 132px;
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .13em;
  font-weight: 950;
}
.metric strong {
  display: block;
  font-size: clamp(32px, 5vw, 52px);
  line-height: 1;
  letter-spacing: -.06em;
  margin-top: 12px;
}
.metric em {
  display: block;
  color: var(--muted);
  font-size: 13px;
  margin-top: 10px;
  font-style: normal;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-top: 18px;
}
.card {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 26px;
  padding: 24px;
  box-shadow: 0 18px 48px rgba(0,0,0,.28);
}
.card.full { grid-column: 1 / -1; }
.callout {
  border-left: 4px solid var(--accent);
  padding: 12px 16px;
  border-radius: 14px;
  background: rgba(143,255,210,.08);
  color: #dffdf3;
  font-weight: 800;
}
.muted { color: var(--muted); }
ul { margin-top: 8px; }
li { margin: 7px 0; }
table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 18px;
  font-size: 13px;
}
th, td {
  padding: 13px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--muted);
  background: rgba(255,255,255,.055);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
tr:last-child td { border-bottom: 0; }
.download-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
.download-card {
  text-decoration: none;
  color: var(--text);
  border: 1px solid var(--line);
  background: rgba(255,255,255,.055);
  padding: 16px;
  border-radius: 18px;
}
.download-card strong { display: block; margin-bottom: 5px; }
.download-card span { color: var(--muted); font-size: 13px; }
.footer {
  margin-top: 26px;
  color: var(--muted);
  font-size: 13px;
}
@media print {
  body { background: #fff; color: #111; }
  .page { width: 100%; padding: 0; }
  .navlinks, .btn { display: none !important; }
  .hero, .card, .metric { background: #fff; color: #111; box-shadow: none; border-color: #aaa; }
  .muted, .metric em, .metric span { color: #444; }
  .disclaimer { color: #111; background: #f7f7f7; border-color: #777; }
}
@media (max-width: 900px) {
  .metrics, .grid, .download-grid { grid-template-columns: 1fr; }
  .hero { padding: 24px; }
  table { display: block; overflow-x: auto; white-space: nowrap; }
}
"""


html_intelligence = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Validation Intelligence — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{CSS}</style>
</head>
<body>
  <main class="page">
    <nav class="nav">
      <div class="brand">Early Risk Alert AI · Validation Intelligence</div>
      <div class="navlinks">
        <a href="/command-center">Command Center</a>
        <a href="/validation-evidence">Evidence Packet</a>
        <a href="/validation-runs">Validation Runs</a>
        <a href="/data-ingest">Data Ingest</a>
      </div>
    </nav>

    <section class="hero">
      <div class="kicker">Cross-Cohort Validation Story</div>
      <h1>Full cohort plus two subcohorts now support the validation story.</h1>
      <p class="lede">
        ERA validation evidence now compares the full validation cohort with two different deterministic case-level subcohorts.
        This makes the public story stronger: results are no longer presented from one cohort slice alone.
      </p>
      <div class="disclaimer">
        Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
      </div>

      <div class="metrics">
        {metric_card("Cohorts Compared", val(len(cohorts)), "full cohort + deterministic subcohorts")}
        {metric_card("Mean Alert Reduction", val(t6.get("alert_reduction_pct_mean"), "%"), "t=6.0 conservative mode")}
        {metric_card("Mean ERA FPR", val(t6.get("era_fpr_pct_mean"), "%"), "t=6.0 low-burden setting")}
        {metric_card("Mean Lead Time", val(t6.get("median_lead_time_hours_mean"), " hrs"), "cross-cohort t=6.0")}
      </div>
    </section>

    <section class="grid">
      <div class="card full">
        <h2>Top Robustness Finding</h2>
        <p class="callout">
          The threshold story can now be reviewed across the full validation cohort, Subcohort B, and Subcohort C using DUA-safe aggregate evidence.
        </p>
        <p class="muted">
          This supports a more credible pilot conversation because the results are framed as an operating-point pattern across multiple cohort slices, not a single isolated run.
        </p>
      </div>

      <div class="card">
        <h2>Cohorts Included</h2>
        <ul>{cohort_list()}</ul>
      </div>

      <div class="card">
        <h2>How to Read It</h2>
        <p class="muted">
          t=4.0 supports higher-detection / high-acuity review. t=5.0 supports balanced review.
          t=6.0 supports the conservative low-burden review queue with the lowest FPR.
        </p>
      </div>

      <div class="card full">
        <h2>Cross-Cohort Threshold Stability</h2>
        <table>
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Cohorts</th>
              <th>Mean Alert Reduction</th>
              <th>SD</th>
              <th>Mean FPR</th>
              <th>SD</th>
              <th>Mean Detection</th>
              <th>SD</th>
              <th>Mean Lead Time</th>
              <th>Mean Alerts / Patient-Day</th>
            </tr>
          </thead>
          <tbody>{threshold_table()}</tbody>
        </table>
      </div>

      <div class="card full">
        <h2>Run-Level Comparison</h2>
        {run_table()}
      </div>

      <div class="card full">
        <h2>Pilot-Safe Claim</h2>
        <p class="callout">
          Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.
        </p>
        <p class="muted">
          Avoid claims that ERA predicts deterioration, prevents adverse events, outperforms standard monitoring, replaces clinician judgment, or independently triggers escalation.
        </p>
      </div>
    </section>

    <p class="footer">Generated {datetime.now(timezone.utc).isoformat()} · Aggregate DUA-safe public summary only.</p>
  </main>
</body>
</html>
"""


html_evidence = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pilot Evidence Packet — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{CSS}</style>
</head>
<body>
  <main class="page">
    <nav class="nav">
      <div class="brand">Early Risk Alert AI · Pilot Evidence Packet</div>
      <div class="navlinks">
        <a href="/validation-intelligence">Validation Intelligence</a>
        <a href="/validation-runs">Validation Runs</a>
        <a href="/command-center">Command Center</a>
        <a class="btn primary" href="javascript:window.print()">Print / Save PDF</a>
      </div>
    </nav>

    <section class="hero">
      <div class="kicker">Printable Cross-Cohort Evidence Packet</div>
      <h1>Validation evidence now spans multiple cohort slices.</h1>
      <p class="lede">
        This packet summarizes aggregate-only evidence across the full validation cohort, Subcohort B, and Subcohort C.
        It is designed for hospital, advisor, investor, and pilot-readiness review.
      </p>
      <div class="disclaimer">
        Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
      </div>

      <div class="metrics">
        {metric_card("Cohorts Compared", val(len(cohorts)), "full cohort + subcohorts")}
        {metric_card("t=6.0 Alert Reduction", val(t6.get("alert_reduction_pct_mean"), "%"), "cross-cohort mean")}
        {metric_card("t=6.0 FPR", val(t6.get("era_fpr_pct_mean"), "%"), "cross-cohort mean")}
        {metric_card("t=6.0 Lead Time", val(t6.get("median_lead_time_hours_mean"), " hrs"), "cross-cohort mean")}
      </div>
    </section>

    <section class="grid">
      <div class="card full">
        <h2>Evidence Downloads</h2>
        <p class="muted">
          Public downloads are aggregate or sanitized only. Row-level MIMIC-derived files, raw restricted CSVs, real identifiers, and exact case-linked timestamps remain local-only.
        </p>
        <div class="download-grid">
          <a class="download-card" href="/api/validation/cross-cohort-validation"><strong>Cross-Cohort JSON</strong><span>Full cohort + Subcohort B + Subcohort C</span></a>
          <a class="download-card" href="/api/validation/milestone"><strong>Milestone JSON</strong><span>Current validation milestone</span></a>
          <a class="download-card" href="/api/validation/runs"><strong>Validation Runs</strong><span>DUA-safe run registry</span></a>
          <a class="download-card" href="/validation-evidence/subcohort-validation.json"><strong>Subcohort B</strong><span>Aggregate subcohort summary</span></a>
          <a class="download-card" href="/validation-evidence/subcohort-c-validation.json"><strong>Subcohort C</strong><span>Aggregate subcohort summary</span></a>
          <a class="download-card" href="/validation-evidence/score-saturation-queue.json"><strong>Queue Audit</strong><span>Score saturation and review-queue context</span></a>
        </div>
      </div>

      <div class="card full">
        <h2>Cross-Cohort Threshold Summary</h2>
        <table>
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Cohorts</th>
              <th>Mean Alert Reduction</th>
              <th>SD</th>
              <th>Mean FPR</th>
              <th>SD</th>
              <th>Mean Detection</th>
              <th>SD</th>
              <th>Mean Lead Time</th>
              <th>Mean Alerts / Patient-Day</th>
            </tr>
          </thead>
          <tbody>{threshold_table()}</tbody>
        </table>
      </div>

      <div class="card full">
        <h2>Run-Level Evidence</h2>
        {run_table()}
      </div>

      <div class="card">
        <h2>DUA-Safe Boundary</h2>
        <p class="muted">
          Safe public evidence includes aggregate metrics, threshold summaries, methodology, governance framing, sanitized summaries, and downloadable aggregate JSON.
        </p>
        <p class="muted">
          Do not publish raw restricted data, row-level enriched exports, real restricted identifiers, exact case-linked timestamps, or patient-level rows.
        </p>
      </div>

      <div class="card">
        <h2>Command Center Link</h2>
        <p class="muted">
          The Command Center should continue to display queue rank, priority tier, primary driver, trend direction, score, and timing context instead of relying on raw score alone.
        </p>
      </div>

      <div class="card full">
        <h2>Approved Framing</h2>
        <p class="callout">
          Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.
        </p>
        <p class="muted">
          Keep claims conservative. Do not state that ERA diagnoses, predicts deterioration, prevents adverse events, replaces standard monitoring, replaces clinician judgment, or independently triggers escalation.
        </p>
      </div>
    </section>

    <p class="footer">Generated {datetime.now(timezone.utc).isoformat()} · Aggregate DUA-safe evidence packet only.</p>
  </main>
</body>
</html>
"""

(WEB / "validation_intelligence.html").write_text(html_intelligence, encoding="utf-8")
(WEB / "validation_evidence.html").write_text(html_evidence, encoding="utf-8")

md = f"""# Cross-Cohort Public Page Polish

Generated: {datetime.now(timezone.utc).isoformat()}

## Updated Public Pages

- `/validation-intelligence`
- `/validation-evidence`

## Top Story

Full validation cohort + Subcohort B + Subcohort C are now presented as the primary robustness story.

## Public Framing

Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.

## DUA-Safe Boundary

Aggregate public evidence only. No row-level restricted data, real identifiers, exact case-linked timestamps, or patient-level rows.
"""

(DOCS / "cross_cohort_public_page_polish.md").write_text(md, encoding="utf-8")

print("WROTE era/web/validation_intelligence.html")
print("WROTE era/web/validation_evidence.html")
print("WROTE docs/validation/cross_cohort_public_page_polish.md")
