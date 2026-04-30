#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re
from statistics import mean

ROOT = Path(".")
DATA = ROOT / "data" / "validation"
DOCS = ROOT / "docs" / "validation"

NOW = datetime.now(timezone.utc).isoformat()
CHECKPOINT_ID = "eicu-harmonized-subcohort-bc-robustness-checkpoint-2026-04-30"
ROLLUP_ID = "mimic-eicu-conservative-t6-aggregate-rollup-2026-04-30"

# Locked values from completed runs.
# Keep event-definition families separate. Do not imply identical endpoints.
MIMIC_T6 = [
    {
        "run": "MIMIC-IV full cohort",
        "dataset": "MIMIC-IV v3.1",
        "event_definition": "Strict clinical-event labels / event clusters",
        "cases": 1705,
        "rows": 456453,
        "threshold": 6.0,
        "alert_reduction_pct": 94.3,
        "fpr_pct": 4.2,
        "detection_pct": 15.3,
        "median_lead_time_hours": 4.0,
        "status": "locked aggregate"
    },
    {
        "run": "MIMIC-IV subcohort B",
        "dataset": "MIMIC-IV v3.1",
        "event_definition": "Strict clinical-event labels / event clusters",
        "cases": 577,
        "rows": "~159K",
        "threshold": 6.0,
        "alert_reduction_pct": 94.0,
        "fpr_pct": 4.4,
        "detection_pct": 16.0,
        "median_lead_time_hours": 4.0,
        "status": "locked aggregate"
    },
    {
        "run": "MIMIC-IV subcohort C",
        "dataset": "MIMIC-IV v3.1",
        "event_definition": "Strict clinical-event labels / event clusters",
        "cases": 621,
        "rows": "~159K",
        "threshold": 6.0,
        "alert_reduction_pct": 94.9,
        "fpr_pct": 3.7,
        "detection_pct": 14.1,
        "median_lead_time_hours": 4.0,
        "status": "locked aggregate"
    },
]

EICU_T6 = [
    {
        "run": "eICU harmonized full pass",
        "dataset": "eICU v2.0",
        "event_definition": "Harmonized clinical-event labels derived from eICU available fields",
        "cases": 2394,
        "rows": 2023962,
        "threshold": 6.0,
        "alert_reduction_pct": 94.25,
        "fpr_pct": 0.98,
        "detection_pct": 24.66,
        "median_lead_time_hours": 4.83,
        "status": "locked aggregate"
    },
    {
        "run": "eICU harmonized subcohort B",
        "dataset": "eICU v2.0",
        "event_definition": "Harmonized clinical-event labels derived from eICU available fields",
        "cases": None,
        "rows": None,
        "threshold": 6.0,
        "alert_reduction_pct": 95.64,
        "fpr_pct": 1.78,
        "detection_pct": 64.49,
        "median_lead_time_hours": 3.69,
        "status": "locked aggregate"
    },
    {
        "run": "eICU harmonized subcohort C",
        "dataset": "eICU v2.0",
        "event_definition": "Harmonized clinical-event labels derived from eICU available fields",
        "cases": None,
        "rows": None,
        "threshold": 6.0,
        "alert_reduction_pct": 95.26,
        "fpr_pct": 2.02,
        "detection_pct": 64.03,
        "median_lead_time_hours": 3.13,
        "status": "locked aggregate"
    },
]

EICU_BC_SENTENCE = (
    "Across eICU harmonized clinical-event subcohorts B and C, the conservative t=6.0 "
    "operating point showed stable aggregate behavior, with 95.26%–95.64% alert reduction, "
    "1.78%–2.02% FPR, 64.03%–64.49% detection, and 3.13–3.69 hours of retrospective "
    "lead-time context."
)

SAFE_CROSS_DATASET_SENTENCE = (
    "Across MIMIC-IV strict clinical-event cohorts and eICU harmonized clinical-event cohorts, "
    "the conservative t=6.0 operating point showed directionally consistent aggregate behavior: "
    "high alert-reduction, low false-positive burden, and measurable retrospective lead-time context, "
    "while detection rates are not treated as directly equivalent because the datasets use different "
    "event-definition methods."
)

def pct_range(items, key):
    vals = [float(x[key]) for x in items if x.get(key) is not None]
    return {
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "mean": round(mean(vals), 2),
    }

def lead_range(items):
    vals = [float(x["median_lead_time_hours"]) for x in items if x.get("median_lead_time_hours") is not None]
    return {
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "mean": round(mean(vals), 2),
    }

def write_json(path: Path, obj: dict):
    text = json.dumps(obj, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    print("WROTE:", path)

def write_text(path: Path, text: str):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print("WROTE:", path)

def md_table(rows):
    lines = []
    lines.append("| Run | Dataset | Event definition | Cases | Rows | t=6 alert reduction | t=6 FPR | t=6 detection | Lead-time context | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        cases = "—" if r["cases"] is None else f'{r["cases"]:,}'
        rows_val = r["rows"]
        rows = "—" if rows_val is None else (rows_val if isinstance(rows_val, str) else f"{rows_val:,}")
        lines.append(
            f'| {r["run"]} | {r["dataset"]} | {r["event_definition"]} | {cases} | {rows} | '
            f'{r["alert_reduction_pct"]}% | {r["fpr_pct"]}% | {r["detection_pct"]}% | '
            f'{r["median_lead_time_hours"]} hrs | {r["status"]} |'
        )
    return "\n".join(lines)

def main():
    eicu_bc = EICU_T6[1:]
    all_rows = MIMIC_T6 + EICU_T6

    eicu_checkpoint = {
        "ok": True,
        "checkpoint_id": CHECKPOINT_ID,
        "generated_at_utc": NOW,
        "purpose": "Lock DUA-safe eICU harmonized clinical-event subcohort B/C robustness as an internal validation checkpoint.",
        "claim_boundary": [
            "Aggregate retrospective robustness only.",
            "Raw eICU files and row-level outputs remain local-only.",
            "Do not claim full clinical validation.",
            "Do not imply diagnosis, treatment direction, clinician replacement, or autonomous escalation."
        ],
        "conservative_sentence": EICU_BC_SENTENCE,
        "threshold": 6.0,
        "subcohort_range_at_t6": {
            "alert_reduction_pct": pct_range(eicu_bc, "alert_reduction_pct"),
            "fpr_pct": pct_range(eicu_bc, "fpr_pct"),
            "detection_pct": pct_range(eicu_bc, "detection_pct"),
            "median_lead_time_hours": lead_range(eicu_bc),
        },
        "runs": eicu_bc,
    }

    rollup = {
        "ok": True,
        "rollup_id": ROLLUP_ID,
        "generated_at_utc": NOW,
        "purpose": "Create one conservative aggregate t=6.0 rollup across MIMIC-IV and eICU while clearly separating event-definition differences.",
        "claim_boundary": [
            "MIMIC-IV and eICU are not treated as identical endpoint definitions.",
            "Detection rates should not be compared as equivalent endpoints unless harmonized event definitions are demonstrably identical.",
            "This supports retrospective multi-dataset robustness under a harmonized threshold framework.",
            "This does not establish prospective clinical performance or full clinical validation."
        ],
        "safe_cross_dataset_sentence": SAFE_CROSS_DATASET_SENTENCE,
        "threshold": 6.0,
        "mimic_strict_clinical_event_range": {
            "alert_reduction_pct": pct_range(MIMIC_T6, "alert_reduction_pct"),
            "fpr_pct": pct_range(MIMIC_T6, "fpr_pct"),
            "detection_pct": pct_range(MIMIC_T6, "detection_pct"),
            "median_lead_time_hours": lead_range(MIMIC_T6),
        },
        "eicu_harmonized_clinical_event_range": {
            "alert_reduction_pct": pct_range(EICU_T6, "alert_reduction_pct"),
            "fpr_pct": pct_range(EICU_T6, "fpr_pct"),
            "detection_pct": pct_range(EICU_T6, "detection_pct"),
            "median_lead_time_hours": lead_range(EICU_T6),
        },
        "runs": all_rows,
    }

    write_json(DATA / "eicu_bc_internal_robustness_checkpoint.json", eicu_checkpoint)
    write_json(DATA / "mimic_eicu_conservative_t6_aggregate_rollup.json", rollup)

    eicu_md = f"""# eICU Harmonized Clinical-Event Subcohort B/C Internal Robustness Checkpoint

## Status

Locked internal validation checkpoint.

## Conservative Summary Sentence

{EICU_BC_SENTENCE}

## What This Supports

This supports an internal robustness conclusion that eICU harmonized clinical-event behavior remains directionally stable across deterministic patient-level subcohorts B and C at the conservative t=6.0 operating point.

## What This Does Not Claim

- It does not claim full clinical validation.
- It does not claim prospective performance.
- It does not claim diagnosis, treatment direction, clinician replacement, or autonomous escalation.
- It does not publish raw eICU data or row-level outputs.

## Locked t=6.0 Subcohort Results

{md_table(eicu_bc)}

## Internal Interpretation

The eICU B/C subcohort results are stable enough to support a DUA-safe internal robustness checkpoint. They should be used as supporting evidence behind the broader validation narrative, not as a standalone clinical-performance claim.
"""
    write_text(DOCS / "eicu_bc_internal_robustness_checkpoint.md", eicu_md)

    rollup_md = f"""# MIMIC-IV + eICU Conservative t=6.0 Aggregate Rollup

## Status

Locked aggregate rollup.

## Safe Cross-Dataset Sentence

{SAFE_CROSS_DATASET_SENTENCE}

## Why Event Definitions Are Separated

MIMIC-IV and eICU should not be described as identical validation endpoints. MIMIC-IV results are framed as strict clinical-event retrospective validation. eICU results are framed as harmonized clinical-event retrospective robustness using fields available in eICU.

The correct public-facing phrasing is:

> Retrospectively evaluated across MIMIC-IV and eICU under a harmonized threshold framework, with event-definition differences clearly labeled.

Avoid saying:

> Validated on two datasets.

unless the same clinical-event definition and the same validation question are applied equivalently across both datasets.

## Conservative t=6.0 Aggregate Table

{md_table(all_rows)}

## Aggregate Ranges by Evidence Family

### MIMIC-IV strict clinical-event cohorts

- Alert reduction: {pct_range(MIMIC_T6, "alert_reduction_pct")["min"]}%–{pct_range(MIMIC_T6, "alert_reduction_pct")["max"]}%
- FPR: {pct_range(MIMIC_T6, "fpr_pct")["min"]}%–{pct_range(MIMIC_T6, "fpr_pct")["max"]}%
- Detection: {pct_range(MIMIC_T6, "detection_pct")["min"]}%–{pct_range(MIMIC_T6, "detection_pct")["max"]}%
- Median lead-time context: {lead_range(MIMIC_T6)["min"]}–{lead_range(MIMIC_T6)["max"]} hours

### eICU harmonized clinical-event cohorts

- Alert reduction: {pct_range(EICU_T6, "alert_reduction_pct")["min"]}%–{pct_range(EICU_T6, "alert_reduction_pct")["max"]}%
- FPR: {pct_range(EICU_T6, "fpr_pct")["min"]}%–{pct_range(EICU_T6, "fpr_pct")["max"]}%
- Detection: {pct_range(EICU_T6, "detection_pct")["min"]}%–{pct_range(EICU_T6, "detection_pct")["max"]}%
- Median lead-time context: {lead_range(EICU_T6)["min"]}–{lead_range(EICU_T6)["max"]} hours

## Approved Claim Language

Early Risk Alert AI has retrospective aggregate evidence across MIMIC-IV strict clinical-event cohorts and eICU harmonized clinical-event cohorts. Across both evidence families, the conservative t=6.0 operating point preserved directionally consistent behavior: high alert-reduction, low false-positive burden, and measurable retrospective lead-time context. Detection rates are reported separately because event definitions differ between datasets.

## Claims To Avoid

- Validated on two datasets.
- Clinically validated on two datasets.
- Predicts deterioration.
- Detects crises early.
- Prevents adverse events.
- Outperforms standard monitoring.
- Replaces clinician judgment.
- Independently triggers escalation.

## DUA-Safe Boundary

This document contains aggregate metrics only. Raw MIMIC/eICU data, row-level outputs, timestamps, patient identifiers, and enriched restricted-data files remain local-only.
"""
    write_text(DOCS / "mimic_eicu_conservative_t6_aggregate_rollup.md", rollup_md)

    # Optional registry entry, preserving existing registry if present.
    registry_path = DATA / "validation_run_registry.json"
    registry = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    if not isinstance(registry, dict):
        registry = {}

    entries = registry.get("internal_checkpoints", [])
    if not isinstance(entries, list):
        entries = []

    new_entry = {
        "id": ROLLUP_ID,
        "created_at_utc": NOW,
        "type": "aggregate_rollup",
        "scope": "MIMIC-IV strict clinical-event cohorts plus eICU harmonized clinical-event cohorts at t=6.0",
        "public_claim_boundary": "Retrospective aggregate robustness under harmonized threshold framework; event-definition differences labeled; not full clinical validation.",
        "docs": [
            "docs/validation/eicu_bc_internal_robustness_checkpoint.md",
            "docs/validation/mimic_eicu_conservative_t6_aggregate_rollup.md"
        ],
        "data": [
            "data/validation/eicu_bc_internal_robustness_checkpoint.json",
            "data/validation/mimic_eicu_conservative_t6_aggregate_rollup.json"
        ]
    }

    entries = [e for e in entries if not (isinstance(e, dict) and e.get("id") == ROLLUP_ID)]
    entries.append(new_entry)
    registry["internal_checkpoints"] = entries
    registry["updated_at_utc"] = NOW
    write_json(registry_path, registry)

    print("")
    print("DONE — checkpoint + rollup files written.")
    print("Review:")
    print(" - docs/validation/eicu_bc_internal_robustness_checkpoint.md")
    print(" - docs/validation/mimic_eicu_conservative_t6_aggregate_rollup.md")
    print(" - data/validation/eicu_bc_internal_robustness_checkpoint.json")
    print(" - data/validation/mimic_eicu_conservative_t6_aggregate_rollup.json")

if __name__ == "__main__":
    main()
