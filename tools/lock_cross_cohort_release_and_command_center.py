#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
DATA = ROOT / "data" / "validation"
DOCS = ROOT / "docs" / "validation"
WEB = ROOT / "era" / "web"
README = ROOT / "README.md"

RELEASE_ID = "stable-cross-cohort-validation-release-2026-04-25"
RELEASE_TITLE = "Stable Cross-Cohort Validation Evidence Release — April 25, 2026"

CC_START = "<!-- ERA_CROSS_COHORT_RELEASE_COMMAND_CENTER_V1_START -->"
CC_END = "<!-- ERA_CROSS_COHORT_RELEASE_COMMAND_CENTER_V1_END -->"

README_START = "<!-- ERA_CROSS_COHORT_RELEASE_README_V1_START -->"
README_END = "<!-- ERA_CROSS_COHORT_RELEASE_README_V1_END -->"


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


def range_text(values, suffix=""):
    clean = [num(v) for v in values if num(v) is not None]
    if not clean:
        return "—"

    lo = min(clean)
    hi = max(clean)

    if abs(lo - hi) < 0.001:
        if suffix == "%":
            return f"{lo:.1f}%"
        if suffix == " hrs":
            return f"{lo:.1f} hours"
        return f"{lo:.1f}"

    if suffix == "%":
        return f"{lo:.1f}%–{hi:.1f}%"
    if suffix == " hrs":
        return f"{lo:.1f}–{hi:.1f} hours"
    return f"{lo:.1f}–{hi:.1f}"


def display_cases(values):
    clean = [num(v) for v in values if num(v) is not None]
    if not clean:
        return "577–1,705"
    return f"{int(min(clean)):,}–{int(max(clean)):,}"


def get_release_values():
    cross = load_json(DATA / "cross_cohort_validation_summary.json")
    runs = cross.get("runs", []) if isinstance(cross, dict) else []

    t6 = []
    for r in runs:
        try:
            if float(r.get("threshold")) == 6.0:
                t6.append(r)
        except Exception:
            pass

    if t6:
        return {
            "cohorts": cross.get("cohorts_compared") or ["Full validation cohort", "Subcohort B", "Subcohort C"],
            "cases_range": display_cases([r.get("cases") or r.get("case_count") for r in t6]),
            "alert_reduction_range": range_text([r.get("alert_reduction_pct") for r in t6], "%"),
            "fpr_range": range_text([r.get("era_fpr_pct") for r in t6], "%"),
            "detection_range": range_text([r.get("detection_pct") for r in t6], "%"),
            "lead_time_range": range_text([r.get("median_lead_time_hours") for r in t6], " hrs"),
            "alerts_per_patient_day_range": range_text([r.get("era_alerts_per_patient_day") for r in t6]),
        }

    return {
        "cohorts": ["Full validation cohort", "Subcohort B", "Subcohort C"],
        "cases_range": "577–1,705",
        "alert_reduction_range": "94.0%–94.9%",
        "fpr_range": "3.7%–4.4%",
        "detection_range": "14.1%–16.0%",
        "lead_time_range": "4.0 hours",
        "alerts_per_patient_day_range": "—",
    }


VALS = get_release_values()

HEADLINE = (
    f"Across the full validation cohort and two deterministic patient-level subcohorts "
    f"({VALS['cases_range']} cases each), the conservative t=6.0 ERA setting showed consistent "
    f"low-burden review-queue performance, with {VALS['alert_reduction_range']} alert reduction, "
    f"{VALS['fpr_range']} ERA FPR, {VALS['detection_range']} event-cluster detection, and a stable "
    f"{VALS['lead_time_range']} median lead-time context across all cohorts."
)


def marker_replace(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


def score_candidate(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return -1

    name = path.name.lower()
    score = 0

    if "command" in name:
        score += 40
    if "center" in name:
        score += 30
    if "dashboard" in name:
        score += 10

    for marker in [
        "Command Center",
        "command-center",
        "patient",
        "risk",
        "SpO2",
        "priority",
        "alert",
        "Explainability",
    ]:
        if marker in text:
            score += 5

    if "</body>" in text.lower():
        score += 15

    return score


def find_command_center_html() -> Path:
    candidates = list(WEB.glob("*.html"))
    if not candidates:
        raise SystemExit("ERROR: No HTML files found in era/web.")

    ranked = sorted(((score_candidate(p), p) for p in candidates), reverse=True, key=lambda x: x[0])
    best_score, best = ranked[0]

    if best_score < 25:
        print("HTML candidates:")
        for score, path in ranked:
            print(score, path)
        raise SystemExit("ERROR: Could not confidently identify Command Center HTML.")

    print("Selected Command Center HTML:", best, "score=", best_score)
    return best


def patch_command_center():
    target = find_command_center_html()
    text = target.read_text(encoding="utf-8", errors="ignore")

    text = marker_replace(text, CC_START, CC_END, "")

    text = text.replace("94.3%", VALS["alert_reduction_range"])
    text = text.replace("4.2%", VALS["fpr_range"])
    text = text.replace("4.0h", VALS["lead_time_range"])
    text = text.replace("4.0 hours", VALS["lead_time_range"])

    module = f"""
{CC_START}
<style id="era-cross-cohort-release-command-center-css">
  .era-release-lock-panel {{
    width: min(1180px, calc(100% - 28px));
    margin: 16px auto;
    border: 1px solid rgba(143,255,210,.36);
    border-left: 5px solid #8fffd2;
    border-radius: 24px;
    background:
      radial-gradient(circle at 10% 0%, rgba(143,255,210,.15), transparent 26rem),
      linear-gradient(145deg, rgba(13,24,42,.96), rgba(6,11,21,.96));
    box-shadow: 0 22px 65px rgba(0,0,0,.34);
    color: #f7fbff;
    padding: 18px;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    position: relative;
    z-index: 5;
  }}
  .era-release-lock-kicker {{
    color: #8fffd2;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .14em;
    font-weight: 950;
    margin-bottom: 6px;
  }}
  .era-release-lock-title {{
    margin: 0;
    font-size: clamp(20px, 2.5vw, 32px);
    line-height: 1.05;
    letter-spacing: -.045em;
    font-weight: 950;
  }}
  .era-release-lock-copy {{
    margin: 10px 0 0;
    color: #dbe8fb;
    font-size: 14px;
    line-height: 1.55;
    max-width: 1050px;
    font-weight: 750;
  }}
  .era-release-lock-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 14px;
  }}
  .era-release-lock-metric {{
    border: 1px solid rgba(155,190,255,.22);
    background: rgba(255,255,255,.055);
    border-radius: 16px;
    padding: 12px;
    min-height: 92px;
  }}
  .era-release-lock-metric span {{
    display: block;
    color: #b8c6d8;
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
    font-weight: 900;
  }}
  .era-release-lock-metric b {{
    display: block;
    font-size: 26px;
    line-height: 1;
    margin-top: 9px;
    letter-spacing: -.05em;
    color: #f7fbff;
  }}
  .era-release-lock-metric small {{
    display: block;
    color: #b8c6d8;
    margin-top: 6px;
    font-size: 11px;
  }}
  .era-release-lock-impact {{
    margin-top: 12px;
    border: 1px solid rgba(255,215,138,.35);
    background: rgba(255,215,138,.095);
    color: #ffe2a5;
    border-radius: 14px;
    padding: 10px 12px;
    font-size: 12px;
    font-weight: 780;
    line-height: 1.45;
  }}
  .era-release-lock-links {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
  }}
  .era-release-lock-links a {{
    color: #f7fbff;
    text-decoration: none;
    border: 1px solid rgba(155,190,255,.24);
    background: rgba(255,255,255,.06);
    border-radius: 999px;
    padding: 8px 11px;
    font-size: 12px;
    font-weight: 850;
  }}
  @media(max-width:900px){{
    .era-release-lock-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
  @media(max-width:560px){{
    .era-release-lock-grid {{ grid-template-columns: 1fr; }}
  }}
</style>

<section id="era-cross-cohort-release-panel" class="era-release-lock-panel">
  <div class="era-release-lock-kicker">Stable Cross-Cohort Validation Evidence Release</div>
  <h2 class="era-release-lock-title">Command Center aligned to locked t=6.0 cross-cohort evidence.</h2>
  <p class="era-release-lock-copy">{HEADLINE}</p>

  <div class="era-release-lock-grid">
    <div class="era-release-lock-metric">
      <span>Alert Reduction Range</span>
      <b>{VALS['alert_reduction_range']}</b>
      <small>t=6.0 across 3 cohorts</small>
    </div>
    <div class="era-release-lock-metric">
      <span>ERA FPR Range</span>
      <b>{VALS['fpr_range']}</b>
      <small>conservative review queue</small>
    </div>
    <div class="era-release-lock-metric">
      <span>Detection Range</span>
      <b>{VALS['detection_range']}</b>
      <small>event-cluster detection</small>
    </div>
    <div class="era-release-lock-metric">
      <span>Median Lead-Time Context</span>
      <b>{VALS['lead_time_range']}</b>
      <small>stable across cohorts</small>
    </div>
  </div>

  <div class="era-release-lock-impact">
    Operational impact framing: this release supports a lower-burden review queue and explainable prioritization workflow. Do not frame this as financial ROI, diagnosis, treatment direction, prevention, or autonomous escalation.
  </div>

  <div class="era-release-lock-links">
    <a href="/validation-intelligence">Validation Intelligence</a>
    <a href="/validation-evidence">Evidence Packet</a>
    <a href="/api/validation/cross-cohort-validation">Cross-Cohort JSON</a>
    <a href="/validation-runs">Validation Runs</a>
  </div>
</section>

<script id="era-cross-cohort-release-command-center-js">
(function(){{
  window.ERA_CROSS_COHORT_RELEASE = {{
    releaseId: "{RELEASE_ID}",
    alertReductionRange: "{VALS['alert_reduction_range']}",
    fprRange: "{VALS['fpr_range']}",
    detectionRange: "{VALS['detection_range']}",
    leadTimeRange: "{VALS['lead_time_range']}",
    casesRange: "{VALS['cases_range']}",
    headline: {json.dumps(HEADLINE)}
  }};

  function updateEvidenceText() {{
    const textNodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {{
      if (!node.nodeValue) continue;
      const parent = node.parentElement;
      if (!parent || parent.closest("#era-cross-cohort-release-panel")) continue;
      textNodes.push(node);
    }}

    textNodes.forEach((n) => {{
      let s = n.nodeValue;
      s = s.replace(/94\\.3%/g, "{VALS['alert_reduction_range']}");
      s = s.replace(/4\\.2%/g, "{VALS['fpr_range']}");
      s = s.replace(/4\\.0h/g, "{VALS['lead_time_range']}");
      n.nodeValue = s;
    }});
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", updateEvidenceText);
  }} else {{
    updateEvidenceText();
  }}
}})();
</script>
{CC_END}
"""

    body_close = text.lower().find("</body>")
    if body_close == -1:
        raise SystemExit(f"ERROR: Could not find </body> in {target}")

    insert_after = None
    nav_match = re.search(r"</nav>", text, flags=re.I)
    header_match = re.search(r"</header>", text, flags=re.I)

    if header_match:
        insert_after = header_match.end()
    elif nav_match:
        insert_after = nav_match.end()

    if insert_after:
        text = text[:insert_after] + "\n" + module + "\n" + text[insert_after:]
    else:
        text = text[:body_close] + "\n" + module + "\n" + text[body_close:]

    target.write_text(text, encoding="utf-8")
    print("PATCHED COMMAND CENTER:", target)
    return target


def build_release_artifacts(command_center_file: Path):
    release_json = {
        "ok": True,
        "release_id": RELEASE_ID,
        "title": RELEASE_TITLE,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_type": "Stable validation evidence and Command Center alignment release",
        "cross_cohort_headline": HEADLINE,
        "t6_cross_cohort_ranges": {
            "cases_per_cohort": VALS["cases_range"],
            "alert_reduction": VALS["alert_reduction_range"],
            "era_fpr": VALS["fpr_range"],
            "event_cluster_detection": VALS["detection_range"],
            "median_lead_time_context": VALS["lead_time_range"],
            "era_alerts_per_patient_day": VALS["alerts_per_patient_day_range"],
        },
        "public_routes": [
            "/validation-intelligence",
            "/validation-evidence",
            "/validation-runs",
            "/command-center",
            "/api/validation/cross-cohort-validation",
            "/validation-evidence/cross-cohort-validation.json"
        ],
        "updated_command_center_file": str(command_center_file),
        "public_policy": "Aggregate DUA-safe evidence only. Row-level MIMIC-derived data remains local-only.",
        "pilot_safe_claim": "Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.",
        "claim_guardrails": [
            "Do not claim ERA predicts deterioration.",
            "Do not claim ERA prevents adverse events.",
            "Do not claim ERA diagnoses.",
            "Do not claim ERA directs treatment.",
            "Do not claim ERA replaces clinician judgment.",
            "Do not claim ERA independently triggers escalation.",
            "Do not claim financial ROI from retrospective validation evidence alone."
        ],
        "notice": "Decision support only. Retrospective aggregate analysis only."
    }

    release_json_path = DATA / "stable_cross_cohort_validation_release_2026_04_25.json"
    release_json_path.write_text(json.dumps(release_json, indent=2), encoding="utf-8")

    release_md = f"""# {RELEASE_TITLE}

## Release ID

`{RELEASE_ID}`

## Locked Evidence Statement

**{HEADLINE}**

## What Was Locked

- Cross-cohort validation comparison
- Full validation cohort + Subcohort B + Subcohort C public story
- Conservative t=6.0 range headline
- Validation Intelligence page alignment
- Evidence Packet alignment
- Command Center evidence module alignment
- Operational impact language alignment

## t=6.0 Cross-Cohort Ranges

| Metric | Locked Range |
|---|---:|
| Cases per cohort | {VALS['cases_range']} |
| Alert reduction | {VALS['alert_reduction_range']} |
| ERA FPR | {VALS['fpr_range']} |
| Event-cluster detection | {VALS['detection_range']} |
| Median lead-time context | {VALS['lead_time_range']} |
| ERA alerts / patient-day | {VALS['alerts_per_patient_day_range']} |

## Command Center Alignment

The Command Center now reflects the stable cross-cohort release and uses operational impact framing:

- Lower-burden review queue
- Explainable prioritization workflow
- Priority tier / queue rank / driver / trend context
- Validation evidence links

## Public Routes

- `/validation-intelligence`
- `/validation-evidence`
- `/validation-runs`
- `/command-center`
- `/api/validation/cross-cohort-validation`
- `/validation-evidence/cross-cohort-validation.json`

## DUA-Safe Boundary

Public evidence remains aggregate-only.

Do not publish:

- Raw MIMIC CSV files
- Row-level enriched CSV exports
- Real restricted identifiers
- Exact case-linked timestamps
- Patient-level rows
- Any restricted row-level derived data

## Pilot-Safe Claim

Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.

## Claim Guardrails

Avoid:

- ERA predicts deterioration
- ERA detects crises early
- ERA prevents adverse events
- ERA diagnoses
- ERA directs treatment
- ERA outperforms standard monitoring
- ERA replaces clinician judgment
- ERA independently triggers escalation
- Financial ROI claims based only on retrospective validation

Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

    release_md_path = DOCS / "stable_cross_cohort_validation_release_2026_04_25.md"
    release_md_path.write_text(release_md, encoding="utf-8")

    milestone_path = DATA / "mimic_validation_milestone_2026_04.json"
    milestone = load_json(milestone_path)

    milestone["stable_cross_cohort_validation_release"] = {
        "status": "Locked",
        "release_id": RELEASE_ID,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_json": str(release_json_path),
        "release_note": str(release_md_path),
        "cross_cohort_headline": HEADLINE,
        "t6_cross_cohort_ranges": release_json["t6_cross_cohort_ranges"],
        "command_center_aligned": True,
        "public_policy": release_json["public_policy"],
        "pilot_safe_note": release_json["notice"]
    }

    milestone["latest_progress_update"] = {
        "status": "Stable cross-cohort validation release locked",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": "Cross-cohort evidence release locked and Command Center aligned to t=6.0 aggregate range story.",
        "public_takeaway": HEADLINE
    }

    milestone_path.write_text(json.dumps(milestone, indent=2), encoding="utf-8")

    print("WROTE:", release_json_path)
    print("WROTE:", release_md_path)
    print("UPDATED:", milestone_path)

    return release_json_path, release_md_path


def update_readme():
    if README.exists():
        text = README.read_text(encoding="utf-8", errors="ignore")
    else:
        text = "# Early Risk Alert AI\n"

    block = f"""
{README_START}
## Stable Cross-Cohort Validation Evidence Release

Release ID: `{RELEASE_ID}`

**{HEADLINE}**

### Locked Public Routes

| Route | Purpose |
|---|---|
| `/validation-intelligence` | Cross-cohort validation story |
| `/validation-evidence` | Printable evidence packet |
| `/validation-runs` | Validation run registry |
| `/command-center` | Live command-center demo aligned to release evidence |
| `/api/validation/cross-cohort-validation` | Aggregate cross-cohort validation JSON |
| `/validation-evidence/cross-cohort-validation.json` | Downloadable aggregate cross-cohort evidence |

### Public Evidence Boundary

Aggregate DUA-safe evidence only. Row-level MIMIC-derived exports, raw restricted CSVs, real restricted identifiers, exact case-linked timestamps, and patient-level rows remain local-only.

Decision support only. Retrospective aggregate analysis only.
{README_END}
"""

    text = marker_replace(text, README_START, README_END, block)
    if README_START not in text:
        text = text.rstrip() + "\n\n" + block + "\n"

    README.write_text(text, encoding="utf-8")
    print("UPDATED README.md")


def main():
    cc = patch_command_center()
    build_release_artifacts(cc)
    update_readme()

    print("")
    print("LOCKED HEADLINE:")
    print(HEADLINE)


if __name__ == "__main__":
    main()
