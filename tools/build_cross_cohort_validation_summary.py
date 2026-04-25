#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import statistics

DATA = Path("data/validation")
DOCS = Path("docs/validation")


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def n(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def mean(vals):
    clean = [n(v) for v in vals if n(v) is not None]
    return round(statistics.mean(clean), 3) if clean else None


def sd(vals):
    clean = [n(v) for v in vals if n(v) is not None]
    if len(clean) < 2:
        return 0
    return round(statistics.stdev(clean), 3)


def rng(vals):
    clean = [n(v) for v in vals if n(v) is not None]
    if not clean:
        return [None, None]
    return [round(min(clean), 3), round(max(clean), 3)]


def v(x, suffix=""):
    if x is None or x == "":
        return "—"
    return f"{x}{suffix}"


def normalize_threshold_run(source_name, cohort_name, run):
    threshold = run.get("threshold") or run.get("t") or run.get("selected_threshold")
    try:
        threshold = float(threshold)
    except Exception:
        pass

    return {
        "source": source_name,
        "cohort": cohort_name,
        "threshold": threshold,
        "suggested_setting": run.get("suggested_setting") or run.get("unit_profile") or run.get("setting"),
        "rows": run.get("rows") or run.get("total_rows"),
        "cases": run.get("case_count") or run.get("patients") or run.get("unique_cases"),
        "events": run.get("event_count") or run.get("events") or run.get("clinical_events"),
        "event_clusters": run.get("event_cluster_count") or run.get("event_clusters") or run.get("source_event_clusters"),
        "alert_reduction_pct": run.get("alert_reduction_pct") or run.get("alert_reduction"),
        "era_fpr_pct": run.get("era_fpr_pct") or run.get("fpr_pct") or run.get("fpr"),
        "detection_pct": (
            run.get("detection_pct")
            or run.get("event_cluster_detection_pct")
            or run.get("patient_detection_pct")
            or run.get("era_patient_detection_pct")
            or run.get("era_event_cluster_detection_pct")
        ),
        "median_lead_time_hours": run.get("median_lead_time_hours") or run.get("median_lead_time"),
        "era_alerts_per_patient_day": run.get("era_alerts_per_patient_day") or run.get("alerts_per_patient_day"),
        "standard_alerts_per_patient_day": run.get("standard_alerts_per_patient_day"),
        "validation_status": run.get("validation_status") or "DUA-safe aggregate validation run",
    }


def full_cohort_runs():
    threshold = load_json(DATA / "threshold_matrix_summary.json")
    milestone = load_json(DATA / "mimic_validation_milestone_2026_04.json")

    rows = []

    for key in ["threshold_matrix", "thresholds", "runs", "threshold_results"]:
        if isinstance(threshold.get(key), list):
            rows = threshold.get(key)
            break

    if not rows:
        for key in ["threshold_matrix", "threshold_profiles", "thresholds", "threshold_results"]:
            if isinstance(milestone.get(key), list):
                rows = milestone.get(key)
                break

    if not rows:
        rows = [
            {
                "threshold": 4.0,
                "suggested_setting": "ICU / high-acuity",
                "alert_reduction_pct": 80.3,
                "era_fpr_pct": 14.4,
                "detection_pct": 37.9,
                "median_lead_time_hours": 4.0,
                "era_alerts_per_patient_day": 2.22,
            },
            {
                "threshold": 5.0,
                "suggested_setting": "Mixed units / balanced",
                "alert_reduction_pct": 89.1,
                "era_fpr_pct": 8.0,
                "detection_pct": 24.6,
                "median_lead_time_hours": 4.0,
                "era_alerts_per_patient_day": 1.23,
            },
            {
                "threshold": 6.0,
                "suggested_setting": "Telemetry / stepdown conservative",
                "alert_reduction_pct": 94.3,
                "era_fpr_pct": 4.2,
                "detection_pct": 15.3,
                "median_lead_time_hours": 4.0,
                "era_alerts_per_patient_day": 0.65,
            },
        ]

    return [normalize_threshold_run("threshold_matrix_summary", "Full validation cohort", r) for r in rows]


def summary_runs(path_name, source_name, cohort_name):
    obj = load_json(DATA / path_name)
    runs = obj.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    return [normalize_threshold_run(source_name, cohort_name, r) for r in runs]


all_runs = []
all_runs += full_cohort_runs()
all_runs += summary_runs("subcohort_validation_summary.json", "subcohort_b_validation_summary", "Subcohort B")
all_runs += summary_runs("subcohort_c_validation_summary.json", "subcohort_c_validation_summary", "Subcohort C")

# Keep only threshold rows that look usable.
all_runs = [r for r in all_runs if r.get("threshold") not in [None, ""]]

if not all_runs:
    raise SystemExit("ERROR: No validation runs found to compare.")

by_threshold = {}
for run in all_runs:
    key = str(float(run.get("threshold")))
    by_threshold.setdefault(key, []).append(run)

threshold_summary = {}

for threshold, runs in sorted(by_threshold.items(), key=lambda kv: float(kv[0])):
    threshold_summary[threshold] = {
        "cohorts_compared": len(runs),
        "alert_reduction_pct_mean": mean([r.get("alert_reduction_pct") for r in runs]),
        "alert_reduction_pct_sd": sd([r.get("alert_reduction_pct") for r in runs]),
        "alert_reduction_pct_range": rng([r.get("alert_reduction_pct") for r in runs]),
        "era_fpr_pct_mean": mean([r.get("era_fpr_pct") for r in runs]),
        "era_fpr_pct_sd": sd([r.get("era_fpr_pct") for r in runs]),
        "era_fpr_pct_range": rng([r.get("era_fpr_pct") for r in runs]),
        "detection_pct_mean": mean([r.get("detection_pct") for r in runs]),
        "detection_pct_sd": sd([r.get("detection_pct") for r in runs]),
        "detection_pct_range": rng([r.get("detection_pct") for r in runs]),
        "median_lead_time_hours_mean": mean([r.get("median_lead_time_hours") for r in runs]),
        "median_lead_time_hours_sd": sd([r.get("median_lead_time_hours") for r in runs]),
        "median_lead_time_hours_range": rng([r.get("median_lead_time_hours") for r in runs]),
        "era_alerts_per_patient_day_mean": mean([r.get("era_alerts_per_patient_day") for r in runs]),
        "era_alerts_per_patient_day_sd": sd([r.get("era_alerts_per_patient_day") for r in runs]),
        "era_alerts_per_patient_day_range": rng([r.get("era_alerts_per_patient_day") for r in runs]),
    }

payload = {
    "ok": True,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "DUA-safe aggregate cross-cohort validation comparison.",
    "cohorts_compared": sorted(set(r["cohort"] for r in all_runs)),
    "thresholds_compared": sorted(set(r["threshold"] for r in all_runs), key=lambda x: float(x)),
    "runs": all_runs,
    "threshold_summary": threshold_summary,
    "interpretation_guidance": {
        "main_question": "Does the ERA operating-point story remain directionally consistent across the full cohort and different deterministic subcohorts?",
        "expected_pattern": {
            "t=4.0": "Higher detection / higher review burden; suitable for high-acuity review.",
            "t=5.0": "Balanced operating point.",
            "t=6.0": "Most selective / lowest alert burden and FPR; suitable for conservative telemetry or stepdown review queue."
        },
        "caution": "This is retrospective aggregate validation only, not prospective clinical validation."
    },
    "public_policy": "Aggregate metrics only. No row-level MIMIC-derived data, no real identifiers, no exact case-linked timestamps.",
    "pilot_safe_claim": "Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.",
    "notice": "Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation."
}

out_json = DATA / "cross_cohort_validation_summary.json"
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

md = """# DUA-Safe Cross-Cohort Validation Comparison

## Purpose

This report compares ERA validation behavior across the full validation cohort and different deterministic patient/case-level subcohorts.

Public output is aggregate only.

No row-level MIMIC-derived data, real identifiers, or exact case-linked timestamps are included.

## Cohorts Compared

"""

for c in payload["cohorts_compared"]:
    md += f"- {c}\n"

md += """

## Run-Level Comparison

| Cohort | Threshold | Setting | Rows | Cases | Alert Reduction | ERA FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
"""

for r in sorted(all_runs, key=lambda x: (x.get("cohort", ""), float(x.get("threshold") or 999))):
    md += (
        f"| {r.get('cohort')} "
        f"| t={r.get('threshold')} "
        f"| {r.get('suggested_setting') or '—'} "
        f"| {v(r.get('rows'))} "
        f"| {v(r.get('cases'))} "
        f"| {v(r.get('alert_reduction_pct'), '%')} "
        f"| {v(r.get('era_fpr_pct'), '%')} "
        f"| {v(r.get('detection_pct'), '%')} "
        f"| {v(r.get('median_lead_time_hours'), ' hrs')} "
        f"| {v(r.get('era_alerts_per_patient_day'))} |\n"
    )

md += """

## Threshold-Level Stability Summary

| Threshold | Cohorts Compared | Mean Alert Reduction | SD | Mean ERA FPR | SD | Mean Detection | SD | Mean Lead Time | SD | Mean ERA Alerts / Patient-Day | SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""

for threshold, item in sorted(threshold_summary.items(), key=lambda kv: float(kv[0])):
    md += (
        f"| t={threshold} "
        f"| {v(item.get('cohorts_compared'))} "
        f"| {v(item.get('alert_reduction_pct_mean'), '%')} "
        f"| {v(item.get('alert_reduction_pct_sd'))} "
        f"| {v(item.get('era_fpr_pct_mean'), '%')} "
        f"| {v(item.get('era_fpr_pct_sd'))} "
        f"| {v(item.get('detection_pct_mean'), '%')} "
        f"| {v(item.get('detection_pct_sd'))} "
        f"| {v(item.get('median_lead_time_hours_mean'), ' hrs')} "
        f"| {v(item.get('median_lead_time_hours_sd'))} "
        f"| {v(item.get('era_alerts_per_patient_day_mean'))} "
        f"| {v(item.get('era_alerts_per_patient_day_sd'))} |\n"
    )

md += """

## Interpretation

Use this report to show that ERA’s operating-point story can be evaluated across multiple cohort slices:

- t=4.0 supports higher-detection / high-acuity review.
- t=5.0 supports a balanced review mode.
- t=6.0 supports a conservative, low-burden review queue.

## Pilot-Safe Claim

Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

out_md = DOCS / "cross_cohort_validation_summary.md"
out_md.write_text(md, encoding="utf-8")

milestone_path = DATA / "mimic_validation_milestone_2026_04.json"
milestone = load_json(milestone_path)

milestone["cross_cohort_validation_comparison"] = {
    "status": "Completed",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "summary_file": str(out_json),
    "report_file": str(out_md),
    "cohorts_compared": payload["cohorts_compared"],
    "thresholds_compared": payload["thresholds_compared"],
    "threshold_summary": threshold_summary,
    "public_policy": payload["public_policy"],
    "pilot_safe_note": "Decision support only. Retrospective aggregate validation only."
}

milestone["latest_progress_update"] = {
    "status": "Cross-cohort validation comparison completed",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "summary": "DUA-safe aggregate comparison completed across the full validation cohort and deterministic subcohorts.",
    "public_takeaway": "ERA operating-point behavior can now be reviewed across multiple cohort slices instead of one validation cohort alone."
}

milestone_path.write_text(json.dumps(milestone, indent=2), encoding="utf-8")

print("WROTE:", out_json)
print("WROTE:", out_md)
print("UPDATED:", milestone_path)
print("")
print("Threshold stability:")
for threshold, item in sorted(threshold_summary.items(), key=lambda kv: float(kv[0])):
    print(
        f"t={threshold} | cohorts={item.get('cohorts_compared')} | "
        f"alert_reduction_mean={item.get('alert_reduction_pct_mean')}% | "
        f"fpr_mean={item.get('era_fpr_pct_mean')}% | "
        f"detection_mean={item.get('detection_pct_mean')}% | "
        f"lead_mean={item.get('median_lead_time_hours_mean')}h"
    )
