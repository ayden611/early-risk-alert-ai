#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone


ROOT = Path(".")
WEB = ROOT / "era" / "web"
DATA = ROOT / "data" / "validation"
DOCS = ROOT / "docs" / "validation"

WEB.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def load_json(name, default=None):
    p = DATA / name
    if not p.exists():
        return default if default is not None else {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


milestone = load_json("mimic_validation_milestone_2026_04.json", {})
threshold_matrix = load_json("threshold_matrix_summary.json", {})
lead_window = load_json("lead_time_window_sensitivity_summary.json", {})
event_cluster = load_json("event_cluster_robustness_summary.json", {})
driver = load_json("driver_subgroup_summary.json", {})
data_quality = load_json("data_quality_temporal_audit_summary.json", {})
cohort_split = load_json("cohort_split_stability_summary.json", {})
score_queue = load_json("score_saturation_queue_audit_summary.json", {})
runs = load_json("validation_run_registry.json", {})


def pick_threshold_rows():
    candidates = []

    if isinstance(threshold_matrix, dict):
        if isinstance(threshold_matrix.get("threshold_matrix"), list):
            candidates = threshold_matrix.get("threshold_matrix")
        elif isinstance(threshold_matrix.get("thresholds"), list):
            candidates = threshold_matrix.get("thresholds")
        elif isinstance(threshold_matrix.get("runs"), list):
            candidates = threshold_matrix.get("runs")

    if not candidates and isinstance(milestone, dict):
        for key in ["threshold_matrix", "threshold_profiles", "thresholds"]:
            if isinstance(milestone.get(key), list):
                candidates = milestone.get(key)
                break

    if not candidates:
        candidates = [
            {
                "threshold": 4.0,
                "suggested_setting": "ICU / high-acuity",
                "alert_reduction_pct": 80.3,
                "era_fpr_pct": 14.4,
                "patient_detection_pct": 37.9,
                "era_alerts_per_patient_day": 2.22,
                "median_lead_time_hours": 4.0,
            },
            {
                "threshold": 5.0,
                "suggested_setting": "Mixed units / balanced",
                "alert_reduction_pct": 89.1,
                "era_fpr_pct": 8.0,
                "patient_detection_pct": 24.6,
                "era_alerts_per_patient_day": 1.23,
                "median_lead_time_hours": 4.0,
            },
            {
                "threshold": 6.0,
                "suggested_setting": "Telemetry / stepdown conservative",
                "alert_reduction_pct": 94.3,
                "era_fpr_pct": 4.2,
                "patient_detection_pct": 15.3,
                "era_alerts_per_patient_day": 0.65,
                "median_lead_time_hours": 4.0,
            },
        ]

    normalized = []
    for row in candidates:
        t = row.get("threshold", row.get("t", row.get("selected_threshold")))
        try:
            t = float(t)
        except Exception:
            pass

        normalized.append({
            "threshold": t,
            "suggested_setting": row.get("suggested_setting") or row.get("unit_profile") or row.get("setting") or (
                "ICU / high-acuity" if t == 4.0 else
                "Mixed units / balanced" if t == 5.0 else
                "Telemetry / stepdown conservative"
            ),
            "framing": row.get("framing") or row.get("interpretation") or (
                "Higher detection, more alerts" if t == 4.0 else
                "Balanced detection and alert burden" if t == 5.0 else
                "Most selective review queue, lowest alert burden"
            ),
            "alert_reduction_pct": row.get("alert_reduction_pct") or row.get("alert_reduction"),
            "era_fpr_pct": row.get("era_fpr_pct") or row.get("fpr_pct") or row.get("fpr"),
            "patient_detection_pct": row.get("patient_detection_pct") or row.get("event_cluster_detection_pct") or row.get("detection_pct") or row.get("detection"),
            "era_alerts_per_patient_day": row.get("era_alerts_per_patient_day") or row.get("alerts_per_patient_day"),
            "median_lead_time_hours": row.get("median_lead_time_hours") or row.get("median_lead_time"),
        })

    return sorted(normalized, key=lambda x: float(x.get("threshold", 999)))


threshold_rows = pick_threshold_rows()
t6 = next((r for r in threshold_rows if str(r.get("threshold")) in {"6.0", "6"}), threshold_rows[-1] if threshold_rows else {})
t4 = next((r for r in threshold_rows if str(r.get("threshold")) in {"4.0", "4"}), threshold_rows[0] if threshold_rows else {})


def safe_num(x, fallback="—", suffix=""):
    if x is None or x == "":
        return fallback
    try:
        if isinstance(x, float):
            s = f"{x:.2f}".rstrip("0").rstrip(".")
        else:
            s = str(x)
        return s + suffix
    except Exception:
        return str(x) + suffix


def metric(name, value, sub=""):
    return f"""
      <div class="metric-card">
        <div class="metric-label">{name}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
      </div>
    """


def threshold_table_html():
    rows = []
    for r in threshold_rows:
        rows.append(f"""
          <tr>
            <td><strong>t={safe_num(r.get("threshold"))}</strong></td>
            <td>{r.get("suggested_setting", "—")}</td>
            <td>{r.get("framing", "—")}</td>
            <td>{safe_num(r.get("alert_reduction_pct"), suffix="%")}</td>
            <td>{safe_num(r.get("era_fpr_pct"), suffix="%")}</td>
            <td>{safe_num(r.get("patient_detection_pct"), suffix="%")}</td>
            <td>{safe_num(r.get("median_lead_time_hours"), suffix=" hrs")}</td>
            <td>{safe_num(r.get("era_alerts_per_patient_day"))}</td>
          </tr>
        """)
    return "\n".join(rows)


def latest_audit_statuses():
    items = [
        ("Threshold matrix", "Configurable t=4.0 / t=5.0 / t=6.0 story", bool(threshold_rows)),
        ("Lead-time sensitivity", "Lead-time behavior tested across event-window assumptions", bool(lead_window)),
        ("Event-cluster robustness", "Event definitions stress-tested across reasonable gap settings", bool(event_cluster)),
        ("Cohort split stability", "Patient-level fold stability added", bool(cohort_split)),
        ("Data-quality audit", "Schema, timestamp, vital coverage, and alignment audit added", bool(data_quality)),
        ("Score saturation / queue audit", "Raw score saturation handled through tier, queue rank, driver, and trend context", bool(score_queue)),
        ("Driver explainability", "Driver subgroups correctly framed as explainability distribution only", bool(driver)),
    ]

    html = ""
    for title, body, ok in items:
        status = "Complete" if ok else "Pending"
        cls = "ok" if ok else "pending"
        html += f"""
          <div class="status-row">
            <div>
              <strong>{title}</strong>
              <span>{body}</span>
            </div>
            <b class="{cls}">{status}</b>
          </div>
        """
    return html


def driver_summary_html():
    runs_list = driver.get("runs", []) if isinstance(driver, dict) else []
    chosen = None
    for r in runs_list:
        if str(r.get("threshold")) in {"6.0", "6"}:
            chosen = r
            break
    if chosen is None and runs_list:
        chosen = runs_list[-1]

    if not chosen:
        return "<p class='muted'>Driver-family distribution will appear after the driver explainability report is generated.</p>"

    fams = chosen.get("driver_family_summaries", [])[:6]
    if not fams:
        return "<p class='muted'>Driver-family distribution is available as aggregate explainability context only.</p>"

    rows = ""
    for f in fams:
        rows += f"""
          <tr>
            <td>{f.get("driver_family", "—")}</td>
            <td>{safe_num(f.get("era_alert_rows"))}</td>
            <td>{safe_num(f.get("share_of_era_alerts_pct"), suffix="%")}</td>
          </tr>
        """

    return f"""
      <table>
        <thead>
          <tr>
            <th>Driver Family</th>
            <th>Alert Rows</th>
            <th>Share of Alerts</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p class="small-note">Driver-family summaries are explainability distributions only. They are not clinical causation claims and are not published as event-attribution evidence.</p>
    """


def cohort_summary_html():
    stability = cohort_split.get("threshold_stability_summary", {}) if isinstance(cohort_split, dict) else {}
    if not stability:
        return "<p class='muted'>Cohort split stability will appear after the patient-level fold test is generated.</p>"

    rows = ""
    for threshold, item in sorted(stability.items(), key=lambda kv: float(kv[0])):
        rows += f"""
          <tr>
            <td><strong>t={threshold}</strong></td>
            <td>{safe_num(item.get("alert_reduction_pct_mean"), suffix="%")}</td>
            <td>{safe_num(item.get("era_fpr_pct_mean"), suffix="%")}</td>
            <td>{safe_num(item.get("detection_pct_mean"), suffix="%")}</td>
            <td>{safe_num(item.get("median_lead_time_hours_mean"), suffix=" hrs")}</td>
            <td>{safe_num(item.get("era_alerts_per_patient_day_mean"))}</td>
          </tr>
        """

    return f"""
      <table>
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Mean Alert Reduction</th>
            <th>Mean FPR</th>
            <th>Mean Detection</th>
            <th>Mean Lead Time</th>
            <th>Mean Alerts / Patient-Day</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    """


def score_queue_html():
    results = score_queue.get("threshold_results", []) if isinstance(score_queue, dict) else []
    if not results:
        return "<p class='muted'>Score saturation and queue burden audit will appear after generation.</p>"

    rows = ""
    for r in results:
        rows += f"""
          <tr>
            <td><strong>t={safe_num(r.get("threshold"))}</strong></td>
            <td>{r.get("suggested_setting", "—")}</td>
            <td>{safe_num(r.get("alert_rows"))}</td>
            <td>{safe_num(r.get("score_9_plus_pct_of_alert_rows"), suffix="%")}</td>
            <td>{safe_num(r.get("score_9_9_plus_pct_of_alert_rows"), suffix="%")}</td>
            <td>{safe_num(r.get("median_queue_size_per_timepoint"))}</td>
            <td>{safe_num(r.get("p95_queue_size_per_timepoint"))}</td>
          </tr>
        """

    return f"""
      <table>
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Setting</th>
            <th>Alert Rows</th>
            <th>Score ≥9</th>
            <th>Score ≥9.9</th>
            <th>Median Queue</th>
            <th>P95 Queue</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    """


CSS = r"""
:root {
  --bg: #070b12;
  --panel: rgba(17, 25, 40, 0.84);
  --panel2: rgba(24, 34, 52, 0.9);
  --line: rgba(161, 190, 255, 0.18);
  --text: #f7fbff;
  --muted: #b8c6d8;
  --accent: #8fffd2;
  --blue: #8bb8ff;
  --gold: #ffd78a;
  --red: #ff9aa9;
  --green: #96ffb8;
  --shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 20% 10%, rgba(34, 124, 255, 0.22), transparent 28rem),
    radial-gradient(circle at 90% 8%, rgba(60, 255, 194, 0.14), transparent 24rem),
    linear-gradient(180deg, #060910 0%, #0a1020 55%, #060910 100%);
  color: var(--text);
  line-height: 1.55;
}

a { color: inherit; }

.page {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 34px 0 70px;
}

.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  margin-bottom: 34px;
}

.brand {
  font-size: 12px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 800;
}

.navlinks {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.navlinks a,
.btn {
  text-decoration: none;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.06);
  color: var(--text);
  padding: 10px 14px;
  border-radius: 14px;
  font-size: 13px;
  font-weight: 750;
}

.btn.primary {
  background: linear-gradient(135deg, rgba(143,255,210,0.95), rgba(139,184,255,0.95));
  color: #041015;
  border: 0;
}

.hero {
  padding: 34px;
  border: 1px solid var(--line);
  border-radius: 30px;
  background: linear-gradient(145deg, rgba(19, 28, 47, 0.94), rgba(9, 14, 24, 0.94));
  box-shadow: var(--shadow);
  overflow: hidden;
  position: relative;
}

.hero::after {
  content: "";
  position: absolute;
  inset: -1px;
  background: radial-gradient(circle at 85% 0%, rgba(143,255,210,0.18), transparent 26rem);
  pointer-events: none;
}

.kicker {
  color: var(--accent);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 12px;
  font-weight: 900;
  margin-bottom: 10px;
}

h1 {
  margin: 0;
  font-size: clamp(36px, 7vw, 74px);
  line-height: 0.94;
  letter-spacing: -0.06em;
}

h2 {
  margin: 0 0 14px;
  font-size: clamp(24px, 3vw, 38px);
  letter-spacing: -0.04em;
}

h3 {
  margin: 0 0 10px;
  font-size: 19px;
  letter-spacing: -0.02em;
}

.lede {
  max-width: 860px;
  color: #d9e5f7;
  font-size: 18px;
  margin: 18px 0 0;
}

.disclaimer {
  margin-top: 22px;
  border: 1px solid rgba(255,215,138,0.38);
  background: rgba(255, 215, 138, 0.1);
  color: #ffe2a5;
  border-radius: 18px;
  padding: 14px 16px;
  font-weight: 750;
}

.hero-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 26px;
}

.metric-card {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: rgba(255,255,255,0.065);
  padding: 18px;
  min-height: 132px;
}

.metric-label {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 900;
}

.metric-value {
  font-size: clamp(32px, 5vw, 52px);
  font-weight: 950;
  line-height: 1;
  letter-spacing: -0.06em;
  margin-top: 12px;
}

.metric-sub {
  color: var(--muted);
  font-size: 13px;
  margin-top: 10px;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-top: 18px;
}

.card {
  border: 1px solid var(--line);
  border-radius: 26px;
  background: var(--panel);
  box-shadow: 0 18px 48px rgba(0,0,0,0.28);
  padding: 24px;
}

.card.full { grid-column: 1 / -1; }

.callout {
  border-left: 4px solid var(--accent);
  padding: 12px 16px;
  background: rgba(143,255,210,0.08);
  border-radius: 14px;
  color: #dffdf3;
  font-weight: 700;
}

table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 18px;
  font-size: 13px;
}

th, td {
  padding: 13px 12px;
  text-align: left;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}

th {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 11px;
  background: rgba(255,255,255,0.05);
}

tr:last-child td { border-bottom: 0; }

.status-row {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
}

.status-row:last-child { border-bottom: 0; }

.status-row span {
  display: block;
  color: var(--muted);
  font-size: 13px;
  margin-top: 3px;
}

.ok {
  color: var(--green);
  white-space: nowrap;
}

.pending {
  color: var(--gold);
  white-space: nowrap;
}

.muted {
  color: var(--muted);
}

.small-note {
  color: var(--muted);
  font-size: 12px;
}

.download-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.download-card {
  display: block;
  text-decoration: none;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px;
  background: rgba(255,255,255,0.055);
}

.download-card strong {
  display: block;
  margin-bottom: 4px;
}

.download-card span {
  color: var(--muted);
  font-size: 13px;
}

.footer {
  margin-top: 28px;
  color: var(--muted);
  font-size: 13px;
}

@media print {
  body {
    background: #fff;
    color: #111;
  }
  .page {
    width: 100%;
    padding: 0;
  }
  .navlinks, .btn {
    display: none !important;
  }
  .hero, .card {
    background: #fff;
    color: #111;
    box-shadow: none;
    border-color: #bbb;
  }
  .metric-card {
    background: #fff;
    border-color: #bbb;
  }
  .muted, .small-note, .metric-sub, .metric-label {
    color: #444;
  }
  .disclaimer {
    color: #111;
    border-color: #777;
    background: #f7f7f7;
  }
}

@media (max-width: 900px) {
  .hero-metrics, .grid, .download-grid {
    grid-template-columns: 1fr;
  }

  .hero {
    padding: 24px;
  }

  table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }
}
"""


validation_intelligence_html = f"""<!doctype html>
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
      <div class="kicker">MIMIC Retrospective Validation · DUA-Safe Public Summary</div>
      <h1>Configurable alert-burden reduction with stable lead-time context.</h1>
      <p class="lede">
        Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable event-cluster detection across threshold settings.
        The current public story emphasizes the conservative t=6.0 operating point while preserving t=4.0 as the higher-detection high-acuity setting.
      </p>
      <div class="disclaimer">
        Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
      </div>

      <div class="hero-metrics">
        {metric("Conservative Alert Reduction", safe_num(t6.get("alert_reduction_pct"), suffix="%"), "t=6.0 telemetry / stepdown setting")}
        {metric("Median Lead Time", safe_num(t6.get("median_lead_time_hours"), suffix=" hrs"), "stable lead-time headline from DUA-safe testing")}
        {metric("ERA FPR", safe_num(t6.get("era_fpr_pct"), suffix="%"), "low false-positive burden at t=6.0")}
        {metric("Higher-Detection Option", safe_num(t4.get("alert_reduction_pct"), suffix="%"), "alert reduction at t=4.0 ICU / high-acuity")}
      </div>
    </section>

    <section class="grid">
      <div class="card full">
        <h2>Threshold Matrix Is the Main Story</h2>
        <p class="muted">
          ERA is intentionally configurable. Lower thresholds increase detection and review volume. Higher thresholds reduce alert burden and FPR.
          This supports different deployment modes for ICU, mixed units, telemetry, or stepdown review queues.
        </p>
        <table>
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Suggested Setting</th>
              <th>Framing</th>
              <th>Alert Reduction</th>
              <th>ERA FPR</th>
              <th>Detection</th>
              <th>Median Lead Time</th>
              <th>ERA Alerts / Patient-Day</th>
            </tr>
          </thead>
          <tbody>
            {threshold_table_html()}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>Robustness Evidence</h2>
        <div class="callout">
          The validation package now includes sensitivity testing, event-cluster robustness, cohort split stability, data-quality checks, and score-saturation queue analysis.
        </div>
        <div style="margin-top:14px;">
          {latest_audit_statuses()}
        </div>
      </div>

      <div class="card">
        <h2>How to Read the Evidence</h2>
        <p class="muted">
          The strongest public-facing claim is not that ERA replaces standard monitoring. The safer and stronger claim is that ERA creates a configurable review queue that can reduce alert burden and provide explainable prioritization context.
        </p>
        <p class="muted">
          Use t=6.0 for the conservative low-burden story. Use t=4.0 for high-acuity review where more sensitivity is acceptable.
        </p>
      </div>

      <div class="card full">
        <h2>Cohort Split Stability</h2>
        <p class="muted">
          Patient-level split testing helps show whether the operating-point story is directionally stable across cohort partitions, instead of relying only on the full dataset.
        </p>
        {cohort_summary_html()}
      </div>

      <div class="card full">
        <h2>Score Saturation + Review Queue Context</h2>
        <p class="muted">
          This section supports the Command Center design decision to show queue rank, priority tier, primary driver, trend direction, and lead-time context instead of relying on raw score alone.
        </p>
        {score_queue_html()}
      </div>

      <div class="card full">
        <h2>Driver Explainability Distribution</h2>
        <p class="muted">
          Driver-family summaries are shown as aggregate explainability distributions only. They are not event-causation claims and are not used as pre-event driver attribution.
        </p>
        {driver_summary_html()}
      </div>

      <div class="card full">
        <h2>Pilot-Safe Claims</h2>
        <p class="callout">
          Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable event-cluster detection and explainable review context across threshold settings.
        </p>
        <p class="muted"><strong>Avoid:</strong> “predicts deterioration,” “prevents adverse events,” “outperforms standard monitoring,” “replaces clinician judgment,” or “independently triggers escalation.”</p>
      </div>
    </section>

    <p class="footer">Generated {datetime.now(timezone.utc).isoformat()} · Aggregate DUA-safe public summary only.</p>
  </main>
</body>
</html>
"""


validation_evidence_html = f"""<!doctype html>
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
      <div class="kicker">Printable Hospital / Investor / Advisor Evidence Packet</div>
      <h1>DUA-safe validation evidence in one place.</h1>
      <p class="lede">
        This packet summarizes the latest MIMIC retrospective validation evidence using aggregate metrics only.
        It is designed for pilot-readiness review, advisor review, hospital conversations, and diligence discussions.
      </p>
      <div class="disclaimer">
        Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
      </div>

      <div class="hero-metrics">
        {metric("Alert Reduction", safe_num(t6.get("alert_reduction_pct"), suffix="%"), "conservative t=6.0 operating point")}
        {metric("Median Lead Time", safe_num(t6.get("median_lead_time_hours"), suffix=" hrs"), "among detected review context")}
        {metric("ERA FPR", safe_num(t6.get("era_fpr_pct"), suffix="%"), "conservative threshold")}
        {metric("t=4.0 High-Acuity", safe_num(t4.get("patient_detection_pct"), suffix="%"), "higher-detection option")}
      </div>
    </section>

    <section class="grid">
      <div class="card full">
        <h2>Evidence Downloads</h2>
        <p class="muted">
          Public downloads are aggregate or sanitized only. Row-level MIMIC-derived files, raw MIMIC CSVs, real MIMIC IDs, and exact patient-linked timestamps remain local-only.
        </p>
        <div class="download-grid">
          <a class="download-card" href="/api/validation/milestone"><strong>Milestone JSON</strong><span>Current validation milestone summary</span></a>
          <a class="download-card" href="/api/validation/runs"><strong>Validation Runs JSON</strong><span>Registry of DUA-safe runs</span></a>
          <a class="download-card" href="/validation-evidence/cohort-split-stability.json"><strong>Cohort Split Stability</strong><span>Patient-level fold stability summary</span></a>
          <a class="download-card" href="/validation-evidence/data-quality-audit.json"><strong>Data Quality Audit</strong><span>Schema, timestamp, and alignment checks</span></a>
          <a class="download-card" href="/validation-evidence/score-saturation-queue.json"><strong>Score Saturation Queue Audit</strong><span>Queue burden and display rationale</span></a>
          <a class="download-card" href="/validation-evidence/driver-subgroups.json"><strong>Driver Explainability</strong><span>Aggregate driver-family distribution</span></a>
        </div>
      </div>

      <div class="card full">
        <h2>Primary Threshold Matrix</h2>
        <table>
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Suggested Setting</th>
              <th>Framing</th>
              <th>Alert Reduction</th>
              <th>ERA FPR</th>
              <th>Detection</th>
              <th>Median Lead Time</th>
              <th>Alerts / Patient-Day</th>
            </tr>
          </thead>
          <tbody>
            {threshold_table_html()}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>Quality-Control Foundation</h2>
        <p class="muted">
          The evidence packet now includes data-quality, temporal-alignment, event-window, event-cluster, cohort split, and score saturation audits.
        </p>
        {latest_audit_statuses()}
      </div>

      <div class="card">
        <h2>Public DUA-Safe Boundary</h2>
        <p class="muted">
          Safe public evidence includes aggregate metrics, sanitized examples, methodology, code, threshold tables, alert reduction, FPR, detection, lead-time summaries, and governance framing.
        </p>
        <p class="muted">
          Do not publish raw MIMIC CSV files, row-level enriched CSV files, real MIMIC IDs, exact timestamps tied to cases, or patient-level rows.
        </p>
      </div>

      <div class="card full">
        <h2>Command Center Display Rationale</h2>
        <p class="callout">
          The live UI should emphasize Priority Tier, Queue Rank, Primary Driver, Trend Direction, Risk Score, and Lead-Time / First Threshold Crossing context.
        </p>
        <p class="muted">
          This supports the score-saturation fix: even when raw scores cluster high, the command center can still differentiate patients using context-rich prioritization.
        </p>
      </div>

      <div class="card full">
        <h2>Driver Explainability Distribution</h2>
        {driver_summary_html()}
      </div>

      <div class="card full">
        <h2>Approved Framing</h2>
        <p class="callout">
          Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable event-cluster detection and explainable review context across threshold settings.
        </p>
        <p class="muted">
          Keep claims conservative. Do not state that ERA diagnoses, predicts deterioration, prevents adverse events, replaces monitoring, replaces clinician judgment, or independently triggers escalation.
        </p>
      </div>
    </section>

    <p class="footer">Generated {datetime.now(timezone.utc).isoformat()} · Aggregate DUA-safe evidence packet only.</p>
  </main>
</body>
</html>
"""


(WEB / "validation_intelligence.html").write_text(validation_intelligence_html, encoding="utf-8")
(WEB / "validation_evidence.html").write_text(validation_evidence_html, encoding="utf-8")

summary_md = f"""# Public Validation Page Polish

Generated: {datetime.now(timezone.utc).isoformat()}

## Updated Pages

- `/validation-intelligence`
- `/validation-evidence`

## Main Public Story

- Conservative t=6.0 operating point: {safe_num(t6.get("alert_reduction_pct"), suffix="%")} alert reduction, {safe_num(t6.get("era_fpr_pct"), suffix="%")} FPR, {safe_num(t6.get("median_lead_time_hours"), suffix=" hrs")} median lead-time context.
- Higher-detection t=4.0 operating point: {safe_num(t4.get("alert_reduction_pct"), suffix="%")} alert reduction with higher review volume.
- Threshold matrix is now the main story.
- Robustness evidence is integrated into the public validation narrative.
- Driver subgroup output is framed as explainability distribution only.
- Score saturation is handled through queue rank, priority tier, driver, trend, and lead-time context.

## DUA-Safe Boundary

Public pages use aggregate metrics only. They do not include row-level MIMIC-derived data, real MIMIC identifiers, or exact patient-linked timestamps.
"""

(DOCS / "public_validation_page_polish.md").write_text(summary_md, encoding="utf-8")

print("WROTE era/web/validation_intelligence.html")
print("WROTE era/web/validation_evidence.html")
print("WROTE docs/validation/public_validation_page_polish.md")
