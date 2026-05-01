from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

WEB = Path("era/web")
DOCS = Path("docs/validation")
DATA = Path("data/validation")

runs_path = WEB / "validation_runs.html"

if not runs_path.exists():
    raise SystemExit("ERROR: era/web/validation_runs.html not found.")

canonical = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": "validation_runs_metric_alignment_resolved",
    "authoritative_public_eicu_track": {
        "track": "eICU outcome-proxy check",
        "threshold": "t=6.0",
        "alert_reduction": "96.8%",
        "fpr": "1.8%",
        "detection": "66.6%",
        "lead_time": "3.41 hrs",
        "source_note": "Authoritative terminal/evidence output for the eICU outcome-proxy track."
    },
    "separate_eicu_harmonized_track": {
        "track": "eICU harmonized clinical-event labeling pass",
        "threshold": "t=6.0",
        "alert_reduction": "94.25%",
        "fpr": "0.98%",
        "detection": "24.66%",
        "lead_time": "4.83 hrs",
        "source_note": "Separate harmonized clinical-event alignment track. Not merged with the outcome-proxy track."
    },
    "eicu_subcohort_bc_internal_robustness": {
        "track": "eICU harmonized subcohorts B/C",
        "threshold": "t=6.0",
        "alert_reduction_range": "95.26%–95.64%",
        "fpr_range": "1.78%–2.02%",
        "detection_range": "64.03%–64.49%",
        "lead_time_range": "3.13–3.69 hrs",
        "source_note": "Internal robustness checkpoint; aggregate only."
    },
    "mimic_track": {
        "track": "MIMIC-IV strict clinical-event evidence",
        "threshold": "t=6.0",
        "alert_reduction_range": "94.0%–94.9%",
        "fpr_range": "3.7%–4.4%",
        "detection_range": "14.1%–16.0%",
        "lead_time": "4.0 hrs",
        "source_note": "Strict clinical-event label/event-cluster retrospective track."
    },
    "guardrail": "Detection rates must not be merged across event-definition tracks. These are retrospective aggregate results only and are not prospective clinical validation."
}

DATA.mkdir(parents=True, exist_ok=True)
(DATA / "validation_runs_authoritative_metric_alignment.json").write_text(
    json.dumps(canonical, indent=2),
    encoding="utf-8"
)

html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Early Risk Alert AI — Validation Runs</title>
  <style>
    :root{
      --bg:#06101d;
      --panel:#111d31;
      --panel2:#15243b;
      --line:rgba(184,211,255,.18);
      --text:#f7fbff;
      --muted:#a9b8cc;
      --green:#a7ff9a;
      --amber:#ffd36d;
      --cyan:#9bd7ff;
      --red:#ff8fa3;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at 10% 0%,rgba(78,166,255,.16),transparent 30%),
        radial-gradient(circle at 86% 5%,rgba(167,255,154,.10),transparent 26%),
        linear-gradient(180deg,#06101d 0%,#07111f 52%,#050912 100%);
      color:var(--text);
    }
    a{color:inherit;text-decoration:none}
    .wrap{max-width:1360px;margin:0 auto;padding:24px 18px 70px}
    .top{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:18px;
      flex-wrap:wrap;
      margin-bottom:18px;
    }
    .kicker{
      color:#b8ffd0;
      font-size:11px;
      letter-spacing:.16em;
      text-transform:uppercase;
      font-weight:1000;
    }
    h1{
      margin:8px 0 8px;
      font-size:clamp(34px,5vw,64px);
      line-height:.9;
      letter-spacing:-.065em;
    }
    .sub{
      max-width:920px;
      color:#d8e3f2;
      font-size:15px;
      line-height:1.45;
      font-weight:760;
    }
    .nav{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      justify-content:flex-end;
      margin-top:4px;
    }
    .nav a{
      border:1px solid var(--line);
      background:rgba(255,255,255,.06);
      border-radius:999px;
      padding:8px 11px;
      font-weight:900;
      font-size:12px;
      white-space:nowrap;
    }
    .badge{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      border-radius:999px;
      padding:7px 10px;
      font-size:10px;
      letter-spacing:.11em;
      text-transform:uppercase;
      font-weight:1000;
      border:1px solid var(--line);
      background:rgba(255,255,255,.06);
      margin:0 6px 6px 0;
    }
    .badge.green{color:#d7ffd1;border-color:rgba(167,255,154,.45);background:rgba(167,255,154,.10)}
    .badge.amber{color:#ffe6a2;border-color:rgba(255,211,109,.50);background:rgba(255,211,109,.10)}
    .badge.blue{color:#d6efff;border-color:rgba(155,215,255,.45);background:rgba(155,215,255,.10)}
    .panel{
      background:linear-gradient(180deg,rgba(20,34,56,.94),rgba(11,20,35,.94));
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:0 18px 70px rgba(0,0,0,.35);
      padding:18px;
      margin:14px 0;
    }
    .panel h2{
      margin:0 0 8px;
      font-size:26px;
      letter-spacing:-.045em;
    }
    .panel p{
      color:#d7e3f3;
      line-height:1.45;
      font-size:14px;
      font-weight:750;
      margin:0 0 12px;
    }
    .callout{
      border:1px solid rgba(255,211,109,.45);
      background:rgba(255,211,109,.09);
      color:#ffe6a2;
      border-radius:16px;
      padding:12px 14px;
      font-size:13px;
      line-height:1.42;
      font-weight:850;
      margin-top:12px;
    }
    .resolved{
      border:1px solid rgba(167,255,154,.45);
      background:rgba(167,255,154,.08);
      color:#dfffd9;
      border-radius:16px;
      padding:12px 14px;
      font-size:13px;
      line-height:1.42;
      font-weight:850;
      margin-top:12px;
    }
    table{
      width:100%;
      border-collapse:separate;
      border-spacing:0;
      overflow:hidden;
      border-radius:16px;
      border:1px solid var(--line);
      margin-top:12px;
      font-size:13px;
    }
    th{
      background:rgba(154,190,255,.18);
      color:#f3f8ff;
      text-align:left;
      padding:11px 10px;
      font-size:10px;
      letter-spacing:.12em;
      text-transform:uppercase;
      font-weight:1000;
      white-space:nowrap;
    }
    td{
      padding:12px 10px;
      border-top:1px solid rgba(181,211,255,.12);
      color:#f7fbff;
      font-weight:820;
      vertical-align:top;
    }
    td small{
      display:block;
      color:#a9b8cc;
      margin-top:4px;
      font-weight:760;
      line-height:1.35;
    }
    .num{
      color:var(--green);
      font-weight:1000;
      white-space:nowrap;
    }
    .grid{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:14px;
    }
    .mini{
      border:1px solid var(--line);
      background:rgba(255,255,255,.05);
      border-radius:18px;
      padding:14px;
    }
    .mini h3{
      margin:0 0 8px;
      font-size:18px;
      letter-spacing:-.03em;
    }
    .mini ul{
      margin:0;
      padding-left:18px;
      color:#d7e3f3;
      line-height:1.45;
      font-weight:760;
      font-size:13px;
    }
    .footer{
      text-align:center;
      color:#ffe6a2;
      border:1px solid rgba(255,211,109,.38);
      background:rgba(255,211,109,.075);
      border-radius:18px;
      padding:12px 14px;
      font-size:12px;
      line-height:1.4;
      font-weight:900;
      margin-top:18px;
    }
    @media(max-width:900px){
      .grid{grid-template-columns:1fr}
      table{font-size:12px}
      th,td{padding:9px 8px}
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="top">
      <div>
        <div class="kicker">DUA-safe aggregate evidence • corrected validation-runs alignment</div>
        <h1>Validation run history</h1>
        <div class="sub">
          This page now separates eICU evidence tracks so the public validation-runs view matches the authoritative terminal/evidence output without merging different event definitions.
        </div>
      </div>
      <nav class="nav" aria-label="Validation navigation">
        <a href="/command-center">Command Center</a>
        <a href="/validation-intelligence">Validation Intelligence</a>
        <a href="/validation-evidence">Evidence Packet</a>
        <a href="/model-card">Model Card</a>
        <a href="/pilot-success-guide">Pilot Success Guide</a>
      </nav>
    </section>

    <section class="panel">
      <span class="badge green">Discrepancy resolved</span>
      <span class="badge blue">Aggregate only</span>
      <span class="badge amber">Event definitions separated</span>
      <h2>Authoritative correction</h2>
      <p>
        The earlier validation-runs page displayed one eICU row using harmonized clinical-event numbers while the terminal log and evidence page showed the eICU outcome-proxy check. This has now been corrected by separating the eICU tracks instead of presenting them as one interchangeable result.
      </p>
      <div class="resolved">
        Authoritative public eICU outcome-proxy track at t=6.0: <strong>96.8% alert reduction, 1.8% FPR, 66.6% detection, and 3.41 hours lead-time context.</strong>
      </div>
      <div class="callout">
        Guardrail: Detection rates must not be merged across event definitions. MIMIC-IV strict clinical-event results, eICU outcome-proxy results, and eICU harmonized clinical-event results answer related but different validation questions.
      </div>
    </section>

    <section class="panel">
      <h2>Corrected validation matrix</h2>
      <table>
        <thead>
          <tr>
            <th>Evidence track</th>
            <th>Event definition</th>
            <th>Threshold</th>
            <th>Alert reduction</th>
            <th>FPR</th>
            <th>Detection</th>
            <th>Lead-time context</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              MIMIC-IV strict clinical-event evidence
              <small>Full cohort plus subcohort robustness range.</small>
            </td>
            <td>Strict clinical-event labels / event clusters</td>
            <td>t=6.0</td>
            <td class="num">94.0%–94.9%</td>
            <td class="num">3.7%–4.4%</td>
            <td class="num">14.1%–16.0%</td>
            <td class="num">4.0 hrs</td>
            <td>Locked aggregate</td>
          </tr>
          <tr>
            <td>
              eICU outcome-proxy check
              <small>Authoritative terminal/evidence output for the public eICU outcome-proxy track.</small>
            </td>
            <td>Mortality/discharge-derived outcome-proxy event context</td>
            <td>t=6.0</td>
            <td class="num">96.8%</td>
            <td class="num">1.8%</td>
            <td class="num">66.6%</td>
            <td class="num">3.41 hrs</td>
            <td>Corrected primary eICU public row</td>
          </tr>
          <tr>
            <td>
              eICU harmonized clinical-event pass
              <small>Separate alignment pass intended to better match the MIMIC clinical-event framework.</small>
            </td>
            <td>Harmonized clinical-event labeling pass</td>
            <td>t=6.0</td>
            <td class="num">94.25%</td>
            <td class="num">0.98%</td>
            <td class="num">24.66%</td>
            <td class="num">4.83 hrs</td>
            <td>Separate track — not merged</td>
          </tr>
          <tr>
            <td>
              eICU harmonized subcohorts B/C
              <small>Internal robustness checkpoint for harmonized eICU subcohorts.</small>
            </td>
            <td>Harmonized clinical-event labeling pass, subcohort robustness</td>
            <td>t=6.0</td>
            <td class="num">95.26%–95.64%</td>
            <td class="num">1.78%–2.02%</td>
            <td class="num">64.03%–64.49%</td>
            <td class="num">3.13–3.69 hrs</td>
            <td>Internal aggregate checkpoint</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="grid">
      <div class="mini">
        <h3>What changed</h3>
        <ul>
          <li>The eICU outcome-proxy row now matches the terminal/evidence output.</li>
          <li>The harmonized eICU clinical-event pass remains visible as a separate track.</li>
          <li>The page no longer implies that different eICU event definitions are the same endpoint.</li>
          <li>All values are aggregate and DUA-safe.</li>
        </ul>
      </div>
      <div class="mini">
        <h3>Approved interpretation</h3>
        <ul>
          <li>Retrospective evidence spans MIMIC-IV and eICU.</li>
          <li>Alert-reduction and lead-time behavior are directionally consistent.</li>
          <li>Detection rates must be interpreted only within their event-definition track.</li>
          <li>This is not prospective clinical validation.</li>
        </ul>
      </div>
    </section>

    <div class="footer">
      Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation. Retrospective aggregate evidence only.
    </div>
  </main>
</body>
</html>
"""

runs_path.write_text(html, encoding="utf-8")

DOCS.mkdir(parents=True, exist_ok=True)
(DOCS / "validation_runs_discrepancy_resolved.md").write_text(
    """# Validation Runs Page Discrepancy Resolved

## Status

Resolved.

The `/validation-runs` page has been corrected so the eICU outcome-proxy row matches the authoritative terminal/evidence output:

- t=6.0 alert reduction: 96.8%
- t=6.0 FPR: 1.8%
- t=6.0 detection: 66.6%
- t=6.0 lead-time context: 3.41 hours

## Important Separation

The prior issue occurred because the page displayed eICU harmonized clinical-event numbers as if they were the same as the eICU outcome-proxy terminal/evidence output.

The corrected page now separates:

1. MIMIC-IV strict clinical-event evidence.
2. eICU outcome-proxy check.
3. eICU harmonized clinical-event pass.
4. eICU harmonized subcohort B/C robustness checkpoint.

## Guardrail

Detection rates must not be merged across event-definition tracks. These are retrospective aggregate results only and are not prospective clinical validation.

## Public Wording

Approved conservative wording:

Early Risk Alert AI has retrospective aggregate evidence across MIMIC-IV and eICU, with event-definition tracks clearly separated. Alert-reduction and lead-time behavior are directionally consistent, while detection rates should be interpreted only within their respective event-definition context.
""",
    encoding="utf-8"
)

print("WROTE: era/web/validation_runs.html")
print("WROTE: data/validation/validation_runs_authoritative_metric_alignment.json")
print("WROTE: docs/validation/validation_runs_discrepancy_resolved.md")

# Content checks
s = runs_path.read_text(encoding="utf-8")
required = [
    "96.8%",
    "1.8%",
    "66.6%",
    "3.41 hrs",
    "94.25%",
    "0.98%",
    "24.66%",
    "4.83 hrs",
    "Detection rates must not be merged",
    "Discrepancy resolved",
    "Corrected primary eICU public row"
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit(f"ERROR: missing required corrected content: {missing}")

if "Loading..." in s:
    raise SystemExit("ERROR: validation_runs.html still contains Loading...")

print("VALIDATION RUNS CONTENT CHECK OK")
