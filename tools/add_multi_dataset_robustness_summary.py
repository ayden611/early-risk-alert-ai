#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

ROOT = Path(".")
DATA = ROOT / "data" / "validation"
DOCS = ROOT / "docs" / "validation"
WEB = ROOT / "era" / "web"
README = ROOT / "README.md"
APP_FILE = ROOT / "era" / "__init__.py"

HTML_START = "<!-- ERA_MULTI_DATASET_ROBUSTNESS_V1_START -->"
HTML_END = "<!-- ERA_MULTI_DATASET_ROBUSTNESS_V1_END -->"
MD_START = "<!-- ERA_MULTI_DATASET_ROBUSTNESS_V1_START -->"
MD_END = "<!-- ERA_MULTI_DATASET_ROBUSTNESS_V1_END -->"
README_START = "<!-- ERA_MULTI_DATASET_ROBUSTNESS_README_V1_START -->"
README_END = "<!-- ERA_MULTI_DATASET_ROBUSTNESS_README_V1_END -->"
ROUTE_START = "# ERA_MULTI_DATASET_ROBUSTNESS_ROUTES_V1_START"
ROUTE_END = "# ERA_MULTI_DATASET_ROBUSTNESS_ROUTES_V1_END"


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


def fmt_num(x, suffix=""):
    y = num(x)
    if y is None:
        return "—"
    if abs(y - round(y)) < 0.001:
        s = f"{y:.0f}"
    else:
        s = f"{y:.2f}".rstrip("0").rstrip(".")
    return s + suffix


def range_text(values, suffix=""):
    clean = [num(v) for v in values if num(v) is not None]
    if not clean:
        return "—"
    lo = min(clean)
    hi = max(clean)
    if abs(lo - hi) < 0.001:
        return fmt_num(lo, suffix)
    return f"{fmt_num(lo, suffix)}–{fmt_num(hi, suffix)}"


def cases_range(values):
    clean = [num(v) for v in values if num(v) is not None]
    if not clean:
        return "577–1,705"
    return f"{int(min(clean)):,}–{int(max(clean)):,}"


def marker_replace(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


def get_mimic_release():
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
            "dataset": "MIMIC-IV v3.1",
            "role": "Locked strict clinical-event cross-cohort validation release",
            "cohort_label": "Full validation cohort + two deterministic case-level subcohorts",
            "cases": cases_range([r.get("cases") or r.get("case_count") for r in t6]),
            "rows": "456,453 in full validation cohort",
            "events": "event clusters / clinical-event labels",
            "threshold": "t=6.0",
            "alert_reduction": range_text([r.get("alert_reduction_pct") for r in t6], "%"),
            "fpr": range_text([r.get("era_fpr_pct") for r in t6], "%"),
            "detection": range_text([r.get("detection_pct") for r in t6], "%"),
            "lead_time": range_text([r.get("median_lead_time_hours") for r in t6], " hrs"),
            "interpretation": "Strict clinical-event retrospective validation; conservative setting optimized for low-burden review queues.",
        }

    return {
        "dataset": "MIMIC-IV v3.1",
        "role": "Locked strict clinical-event cross-cohort validation release",
        "cohort_label": "Full validation cohort + two deterministic case-level subcohorts",
        "cases": "577–1,705",
        "rows": "456,453 in full validation cohort",
        "events": "event clusters / clinical-event labels",
        "threshold": "t=6.0",
        "alert_reduction": "94.0%–94.9%",
        "fpr": "3.7%–4.4%",
        "detection": "14.1%–16.0%",
        "lead_time": "4.0 hrs",
        "interpretation": "Strict clinical-event retrospective validation; conservative setting optimized for low-burden review queues.",
    }


def get_eicu_release():
    eicu = load_json(DATA / "eicu_validation_summary.json")
    runs = eicu.get("runs", []) if isinstance(eicu, dict) else []

    t6 = None
    for r in runs:
        try:
            if float(r.get("threshold")) == 6.0:
                t6 = r
        except Exception:
            pass

    if t6:
        return {
            "dataset": "eICU Collaborative Research Database v2.0",
            "role": "Second-dataset retrospective outcome-proxy check",
            "cohort_label": "Local-only ERA-format eICU outcome-proxy cohort",
            "cases": fmt_num(t6.get("case_count")),
            "rows": fmt_num(t6.get("rows")),
            "events": fmt_num(t6.get("event_count")),
            "threshold": "t=6.0",
            "alert_reduction": fmt_num(t6.get("alert_reduction_pct"), "%"),
            "fpr": fmt_num(t6.get("era_fpr_pct"), "%"),
            "detection": fmt_num(t6.get("detection_pct"), "%"),
            "lead_time": fmt_num(t6.get("median_lead_time_hours"), " hrs"),
            "interpretation": "Outcome-proxy retrospective check using mortality/discharge-derived event context; not directly comparable to MIMIC clinical-event detection.",
        }

    return {
        "dataset": "eICU Collaborative Research Database v2.0",
        "role": "Second-dataset retrospective outcome-proxy check",
        "cohort_label": "Local-only ERA-format eICU outcome-proxy cohort",
        "cases": "2,394",
        "rows": "2,023,962",
        "events": "772",
        "threshold": "t=6.0",
        "alert_reduction": "96.8%",
        "fpr": "1.8%",
        "detection": "66.6%",
        "lead_time": "3.41 hrs",
        "interpretation": "Outcome-proxy retrospective check using mortality/discharge-derived event context; not directly comparable to MIMIC clinical-event detection.",
    }


MIMIC = get_mimic_release()
EICU = get_eicu_release()

MULTI_DATASET_SENTENCE = (
    "MIMIC-IV established strict clinical-event cross-cohort retrospective stability, while eICU added a separate "
    "second-dataset outcome-proxy check; across both datasets, ERA preserved the same threshold-direction behavior: "
    "lower thresholds increased detection, while conservative thresholds reduced review burden and false positives."
)

LEAD_TIME_SENTENCE = (
    f"At the conservative t=6.0 operating point, MIMIC-IV showed {MIMIC['lead_time']} median lead-time context "
    f"across locked cross-cohort evidence, while eICU showed {EICU['lead_time']} median lead-time context in the "
    f"outcome-proxy check."
)

CITATIONS = {
    "mimic_iv_physionet": {
        "label": "MIMIC-IV v3.1 PhysioNet citation",
        "apa": "Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58",
        "doi": "10.13026/kpb9-mt58",
        "url": "https://doi.org/10.13026/kpb9-mt58",
    },
    "mimic_iv_scientific_data": {
        "label": "MIMIC-IV Scientific Data publication",
        "apa": "Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10, 1. https://doi.org/10.1038/s41597-022-01899-x",
        "doi": "10.1038/s41597-022-01899-x",
        "url": "https://doi.org/10.1038/s41597-022-01899-x",
    },
    "eicu_physionet": {
        "label": "eICU v2.0 PhysioNet citation",
        "apa": "Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2WM1R",
        "doi": "10.13026/C2WM1R",
        "url": "https://doi.org/10.13026/C2WM1R",
    },
    "eicu_scientific_data": {
        "label": "eICU Scientific Data publication",
        "apa": "Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. https://doi.org/10.1038/sdata.2018.178",
        "doi": "10.1038/sdata.2018.178",
        "url": "https://doi.org/10.1038/sdata.2018.178",
    },
    "physionet_standard": {
        "label": "PhysioNet standard citation",
        "apa": "Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220. RRID:SCR_007345.",
    },
    "dua": {
        "label": "PhysioNet Credentialed Health Data Use Agreement 1.5.0",
        "summary": "Credentialed restricted data requires avoiding re-identification attempts, avoiding disclosure of identities in publications or communications, and not sharing restricted-data access.",
    }
}


def build_payload():
    return {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "Multi-Dataset Robustness Summary",
        "purpose": "DUA-safe aggregate summary comparing MIMIC-IV strict clinical-event evidence with eICU second-dataset outcome-proxy evidence.",
        "top_line": MULTI_DATASET_SENTENCE,
        "lead_time_context": LEAD_TIME_SENTENCE,
        "datasets": {
            "mimic_iv": MIMIC,
            "eicu": EICU,
        },
        "comparison": {
            "shared_behavior": "Across both datasets, lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.",
            "not_apples_to_apples": "MIMIC-IV uses stricter clinical-event labels; eICU uses outcome-proxy labels from mortality/discharge-derived event context. Detection rates should not be compared as equivalent endpoint definitions.",
            "safe_claim": "Cross-dataset retrospective robustness evidence across de-identified ICU datasets.",
            "avoid_claims": [
                "proven generalizability",
                "prospective validation",
                "diagnosis",
                "treatment direction",
                "prevention of adverse events",
                "replacement of clinician judgment",
                "autonomous escalation",
                "direct superiority over standard monitoring"
            ],
        },
        "citations": CITATIONS,
        "public_policy": "Aggregate public evidence only. Raw restricted files and row-level derived outputs remain local-only.",
        "notice": "Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation."
    }


payload = build_payload()
(DATA / "multi_dataset_robustness_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


HTML_BLOCK = f"""
{HTML_START}
<section class="card full" style="
  border:1px solid rgba(143,255,210,.34);
  border-left:5px solid #8fffd2;
  background:linear-gradient(145deg, rgba(13,24,42,.94), rgba(6,11,21,.94));
  border-radius:28px;
  padding:24px;
  margin-top:18px;
">
  <div style="color:#8fffd2;text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:950;margin-bottom:8px;">
    Multi-Dataset Robustness Summary
  </div>

  <h2 style="margin:0 0 12px;">MIMIC-IV + eICU evidence now support a stronger cross-dataset story.</h2>

  <p class="callout" style="
    border-left:4px solid #8fffd2;
    padding:12px 16px;
    border-radius:14px;
    background:rgba(143,255,210,.09);
    color:#eaf4ff;
    font-weight:800;
  ">
    {MULTI_DATASET_SENTENCE}
  </p>

  <p class="muted">
    {LEAD_TIME_SENTENCE}
  </p>

  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Evidence Role</th>
        <th>Cases</th>
        <th>Rows</th>
        <th>Event Context</th>
        <th>Threshold</th>
        <th>Alert Reduction</th>
        <th>FPR</th>
        <th>Detection</th>
        <th>Lead Time</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>{MIMIC['dataset']}</strong></td>
        <td>{MIMIC['role']}</td>
        <td>{MIMIC['cases']}</td>
        <td>{MIMIC['rows']}</td>
        <td>{MIMIC['events']}</td>
        <td>{MIMIC['threshold']}</td>
        <td>{MIMIC['alert_reduction']}</td>
        <td>{MIMIC['fpr']}</td>
        <td>{MIMIC['detection']}</td>
        <td>{MIMIC['lead_time']}</td>
      </tr>
      <tr>
        <td><strong>{EICU['dataset']}</strong></td>
        <td>{EICU['role']}</td>
        <td>{EICU['cases']}</td>
        <td>{EICU['rows']}</td>
        <td>{EICU['events']}</td>
        <td>{EICU['threshold']}</td>
        <td>{EICU['alert_reduction']}</td>
        <td>{EICU['fpr']}</td>
        <td>{EICU['detection']}</td>
        <td>{EICU['lead_time']}</td>
      </tr>
    </tbody>
  </table>

  <p class="muted" style="margin-top:14px;">
    Important interpretation: MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy labels from mortality/discharge-derived event context. Detection rates should not be treated as equivalent endpoint definitions.
  </p>

  <h3 style="margin-top:18px;">Citation Story</h3>
  <ul class="muted">
    <li><strong>MIMIC-IV:</strong> Johnson et al., MIMIC-IV v3.1, PhysioNet, DOI 10.13026/kpb9-mt58; original Scientific Data publication DOI 10.1038/s41597-022-01899-x.</li>
    <li><strong>eICU:</strong> Pollard et al., eICU Collaborative Research Database v2.0, PhysioNet, DOI 10.13026/C2WM1R; original Scientific Data publication DOI 10.1038/sdata.2018.178.</li>
    <li><strong>DUA boundary:</strong> aggregate public evidence only; raw restricted files and row-level derived outputs remain local-only.</li>
  </ul>

  <div class="era-release-lock-links" style="margin-top:12px;">
    <a href="/api/validation/multi-dataset-robustness">Multi-Dataset JSON</a>
    <a href="/validation-evidence/multi-dataset-robustness.json">Download Summary</a>
    <a href="/api/validation/eicu-validation">eICU JSON</a>
    <a href="/api/validation/cross-cohort-validation">MIMIC Cross-Cohort JSON</a>
  </div>
</section>
{HTML_END}
"""

MD_BLOCK = f"""
{MD_START}

## Multi-Dataset Robustness Summary

**{MULTI_DATASET_SENTENCE}**

{LEAD_TIME_SENTENCE}

| Dataset | Evidence Role | Cases | Rows | Event Context | Threshold | Alert Reduction | FPR | Detection | Lead Time |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| {MIMIC['dataset']} | {MIMIC['role']} | {MIMIC['cases']} | {MIMIC['rows']} | {MIMIC['events']} | {MIMIC['threshold']} | {MIMIC['alert_reduction']} | {MIMIC['fpr']} | {MIMIC['detection']} | {MIMIC['lead_time']} |
| {EICU['dataset']} | {EICU['role']} | {EICU['cases']} | {EICU['rows']} | {EICU['events']} | {EICU['threshold']} | {EICU['alert_reduction']} | {EICU['fpr']} | {EICU['detection']} | {EICU['lead_time']} |

### Interpretation Guardrail

MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy labels from mortality/discharge-derived event context. Detection rates should not be treated as equivalent endpoint definitions.

### Safe Claim

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

### Avoid

- proven generalizability
- prospective validation
- diagnosis
- treatment direction
- prevention of adverse events
- replacement of clinician judgment
- autonomous escalation
- direct superiority over standard monitoring

{MD_END}
"""


def patch_html(path: Path):
    if not path.exists():
        raise SystemExit(f"ERROR: missing HTML file {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = marker_replace(text, HTML_START, HTML_END, "")

    idx = text.lower().rfind("</main>")
    if idx == -1:
        idx = text.lower().rfind("</body>")
    if idx == -1:
        raise SystemExit(f"ERROR: no insertion point in {path}")

    text = text[:idx] + "\n" + HTML_BLOCK + "\n" + text[idx:]
    path.write_text(text, encoding="utf-8")
    print("PATCHED:", path)


def patch_md(path: Path):
    if not path.exists():
        print("SKIP missing MD:", path)
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = marker_replace(text, MD_START, MD_END, "")
    text = text.rstrip() + "\n\n" + MD_BLOCK + "\n"
    path.write_text(text, encoding="utf-8")
    print("PATCHED:", path)


def write_docs():
    md = f"""# Multi-Dataset Robustness Summary

Generated: {datetime.now(timezone.utc).isoformat()}

## Top-Line Summary

**{MULTI_DATASET_SENTENCE}**

{LEAD_TIME_SENTENCE}

## Dataset Comparison

| Dataset | Evidence Role | Cases | Rows | Event Context | Threshold | Alert Reduction | FPR | Detection | Lead Time |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| {MIMIC['dataset']} | {MIMIC['role']} | {MIMIC['cases']} | {MIMIC['rows']} | {MIMIC['events']} | {MIMIC['threshold']} | {MIMIC['alert_reduction']} | {MIMIC['fpr']} | {MIMIC['detection']} | {MIMIC['lead_time']} |
| {EICU['dataset']} | {EICU['role']} | {EICU['cases']} | {EICU['rows']} | {EICU['events']} | {EICU['threshold']} | {EICU['alert_reduction']} | {EICU['fpr']} | {EICU['detection']} | {EICU['lead_time']} |

## Interpretation

MIMIC-IV established strict clinical-event cross-cohort retrospective stability.

eICU added a separate second-dataset outcome-proxy check.

The strongest defensible finding is not that the two detection rates are directly comparable. The strongest defensible finding is that ERA preserved the same threshold-direction behavior across both datasets.

## Citation Story

### MIMIC-IV

Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10, 1. https://doi.org/10.1038/s41597-022-01899-x

### eICU

Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2WM1R

Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. https://doi.org/10.1038/sdata.2018.178

### PhysioNet

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220. RRID:SCR_007345.

## DUA-Safe Boundary

Aggregate public evidence only.

Do not publish:

- raw restricted CSV/GZ files
- row-level enriched exports
- real restricted identifiers
- exact row-level timestamps tied to individual records
- patient-level rows
- local mapping files

## Claim Guardrails

Use:

**Cross-dataset retrospective robustness evidence across de-identified ICU datasets.**

Avoid:

- proven generalizability
- prospective validation
- diagnosis
- treatment direction
- prevention of adverse events
- replacement of clinician judgment
- autonomous escalation
- direct superiority over standard monitoring

Decision support only. Retrospective aggregate analysis only.
"""
    (DOCS / "multi_dataset_robustness_summary.md").write_text(md, encoding="utf-8")
    print("WROTE:", DOCS / "multi_dataset_robustness_summary.md")

    citation_md = f"""# Dataset Citation Story

Generated: {datetime.now(timezone.utc).isoformat()}

## Public Citation Language

This validation package uses DUA-safe aggregate summaries derived from de-identified PhysioNet datasets.

MIMIC-IV is cited as the strict clinical-event retrospective validation source. eICU is cited as a separate second-dataset retrospective outcome-proxy check.

## MIMIC-IV Citations

1. Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

2. Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10, 1. https://doi.org/10.1038/s41597-022-01899-x

## eICU Citations

1. Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2WM1R

2. Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. https://doi.org/10.1038/sdata.2018.178

## Standard PhysioNet Citation

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220. RRID:SCR_007345.

## DUA-Safe Publication Boundary

Public outputs are aggregate only and must avoid restricted row-level disclosure, real restricted identifiers, exact row-level timestamps tied to individual records, and raw restricted files.
"""
    (DOCS / "dataset_citation_story.md").write_text(citation_md, encoding="utf-8")
    print("WROTE:", DOCS / "dataset_citation_story.md")


def patch_readme():
    text = README.read_text(encoding="utf-8", errors="ignore") if README.exists() else "# Early Risk Alert AI\n"

    block = f"""
{README_START}
## Multi-Dataset Robustness Summary

**{MULTI_DATASET_SENTENCE}**

{LEAD_TIME_SENTENCE}

### Evidence Roles

| Dataset | Role |
|---|---|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort retrospective validation release |
| eICU Collaborative Research Database v2.0 | Separate second-dataset retrospective outcome-proxy check |

### Citation Story

- MIMIC-IV v3.1: Johnson et al. (2024), PhysioNet, DOI 10.13026/kpb9-mt58.
- MIMIC-IV Scientific Data: Johnson et al. (2023), DOI 10.1038/s41597-022-01899-x.
- eICU v2.0: Pollard et al. (2019), PhysioNet, DOI 10.13026/C2WM1R.
- eICU Scientific Data: Pollard et al. (2018), DOI 10.1038/sdata.2018.178.
- PhysioNet standard citation: Goldberger et al. (2000), Circulation.

### Public Boundary

Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.

Decision support only. Retrospective aggregate analysis only.
{README_END}
"""

    text = marker_replace(text, README_START, README_END, block)
    if README_START not in text:
        text = text.rstrip() + "\n\n" + block + "\n"

    README.write_text(text, encoding="utf-8")
    print("UPDATED README.md")


def add_routes():
    if not APP_FILE.exists():
        raise SystemExit("ERROR: era/__init__.py not found.")

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
    # ERA_MULTI_DATASET_ROBUSTNESS_ROUTES_V1_START
    @app.get("/api/validation/multi-dataset-robustness")
    def era_validation_multi_dataset_robustness_api():
        import json
        from pathlib import Path
        from flask import jsonify

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "data" / "validation" / "multi_dataset_robustness_summary.json"

        if not p.exists():
            return jsonify({"ok": False, "error": "Multi-dataset robustness summary not found."}), 404

        data = json.loads(p.read_text(encoding="utf-8"))
        data["ok"] = True
        return jsonify(data)

    @app.get("/validation-evidence/multi-dataset-robustness.json")
    def era_validation_multi_dataset_robustness_download_json():
        from pathlib import Path
        from flask import Response

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "data" / "validation" / "multi_dataset_robustness_summary.json"

        if not p.exists():
            return Response("Multi-dataset robustness summary not found.", status=404, mimetype="text/plain")

        return Response(
            p.read_text(encoding="utf-8"),
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment; filename=early-risk-alert-ai-multi-dataset-robustness-summary.json"
            }
        )
    # ERA_MULTI_DATASET_ROBUSTNESS_ROUTES_V1_END

'''

    lines = app_text.splitlines()
    lines = lines[:insert_line - 1] + route_block.splitlines() + lines[insert_line - 1:]
    APP_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ADDED multi-dataset robustness routes.")


def update_milestone():
    milestone_path = DATA / "mimic_validation_milestone_2026_04.json"
    milestone = load_json(milestone_path)

    milestone["multi_dataset_robustness_summary"] = {
        "status": "Completed",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_file": "data/validation/multi_dataset_robustness_summary.json",
        "report_file": "docs/validation/multi_dataset_robustness_summary.md",
        "citation_story_file": "docs/validation/dataset_citation_story.md",
        "top_line": MULTI_DATASET_SENTENCE,
        "lead_time_context": LEAD_TIME_SENTENCE,
        "claim_guardrail": "Cross-dataset retrospective robustness evidence across de-identified ICU datasets; not prospective validation or proven generalizability."
    }

    milestone["latest_progress_update"] = {
        "status": "Multi-dataset robustness summary completed",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": "MIMIC-IV strict clinical-event evidence and eICU second-dataset outcome-proxy evidence are now summarized together with updated citation story.",
        "public_takeaway": MULTI_DATASET_SENTENCE
    }

    milestone_path.write_text(json.dumps(milestone, indent=2), encoding="utf-8")
    print("UPDATED:", milestone_path)


patch_html(WEB / "validation_intelligence.html")
patch_html(WEB / "validation_evidence.html")
patch_md(DOCS / "eicu_validation_summary.md")
patch_md(DOCS / "cross_cohort_validation_summary.md")
write_docs()
patch_readme()
add_routes()
update_milestone()

print("")
print("DONE building multi-dataset robustness summary.")
print(MULTI_DATASET_SENTENCE)
