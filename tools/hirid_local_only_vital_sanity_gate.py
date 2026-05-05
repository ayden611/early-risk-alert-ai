#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(".")
OUT_DIR = ROOT / "data" / "validation" / "local_private" / "hirid" / "aggregate_outputs"

SUMMARY_JSON = OUT_DIR / "hirid_local_only_aggregate_readiness_summary.json"
SANITY_JSON = OUT_DIR / "hirid_local_only_vital_sanity_gate.json"
SANITY_MD = OUT_DIR / "hirid_local_only_vital_sanity_gate.md"

EXPECTED_RANGES = {
    "heart_rate": {
        "mean_min": 40,
        "mean_max": 130,
        "threshold_positive_rate_max": 40,
        "note": "Heart-rate aggregate appears plausible if mean is within expected adult ICU range."
    },
    "respiratory_rate": {
        "mean_min": 8,
        "mean_max": 35,
        "threshold_positive_rate_max": 60,
        "note": "Respiratory-rate aggregate appears plausible if mean and threshold-positive behavior are reasonable."
    },
    "systolic_bp": {
        "mean_min": 70,
        "mean_max": 170,
        "threshold_positive_rate_max": 60,
        "note": "Systolic BP aggregate appears plausible if mean is within expected ICU range."
    },
    "diastolic_bp": {
        "mean_min": 35,
        "mean_max": 100,
        "threshold_positive_rate_max": 70,
        "note": "Diastolic BP aggregate appears plausible if mean is within expected ICU range."
    },
    "temperature": {
        "mean_min": 34,
        "mean_max": 41,
        "threshold_positive_rate_max": 50,
        "note": "Temperature aggregate appears plausible if Celsius conversion is correct."
    },
    "spo2": {
        "mean_min": 70,
        "mean_max": 100,
        "threshold_positive_rate_max": 60,
        "note": "SpO2 should usually aggregate near physiologic percentage values. A mean far below 70% strongly suggests mapping/unit/value-column issue."
    },
}

BLOCK_PUBLIC_TERMS = [
    "HiRID validated",
    "three-dataset validation completed",
    "clinical validation",
    "prospective validation",
    "confirmed performance",
]

def pct_value(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def evaluate_vital(name: str, stats: dict) -> dict:
    spec = EXPECTED_RANGES.get(name, {})
    mean = pct_value(stats.get("mean"))
    tpr = pct_value(stats.get("threshold_positive_rate"))
    rows = int(stats.get("rows", 0) or 0)
    patients = int(stats.get("unique_patients", 0) or 0)

    issues = []
    status = "PASS"

    if rows <= 0:
        issues.append("No rows found.")
        status = "FAIL"

    if patients <= 0:
        issues.append("No unique patients found.")
        status = "FAIL"

    if spec and mean is not None:
        if mean < spec["mean_min"] or mean > spec["mean_max"]:
            issues.append(
                f"Mean {mean} is outside expected sanity range "
                f"{spec['mean_min']}–{spec['mean_max']}."
            )
            status = "FAIL"

    if spec and tpr is not None:
        if tpr > spec["threshold_positive_rate_max"]:
            issues.append(
                f"Threshold-positive rate {tpr}% is above sanity ceiling "
                f"{spec['threshold_positive_rate_max']}%."
            )
            if status != "FAIL":
                status = "WARN"

    if name == "spo2":
        if mean is not None and mean < 70:
            issues.append(
                "SpO2 aggregate is not physiologically plausible. "
                "Do not run or publish HiRID operating-point validation until SpO2 mapping/value handling is corrected."
            )
            status = "FAIL"
        if tpr is not None and tpr > 80:
            issues.append(
                "SpO2 threshold-positive rate is extremely high. This usually indicates incorrect variable mapping, wrong unit, or wrong numeric value column."
            )
            status = "FAIL"

    return {
        "vital": name,
        "status": status,
        "rows": rows,
        "unique_patients": patients,
        "mean": mean,
        "threshold_positive_rate": tpr,
        "issues": issues,
        "note": spec.get("note", "No expected range configured."),
    }

def main() -> None:
    if not SUMMARY_JSON.exists():
        raise SystemExit(f"Missing summary JSON: {SUMMARY_JSON}")

    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    vital_summary = data.get("vital_aggregate_summary", {})

    results = []
    for vital in ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "temperature"]:
        if vital in vital_summary:
            results.append(evaluate_vital(vital, vital_summary[vital]))
        else:
            results.append({
                "vital": vital,
                "status": "FAIL",
                "rows": 0,
                "unique_patients": 0,
                "mean": None,
                "threshold_positive_rate": None,
                "issues": ["Vital missing from aggregate summary."],
                "note": "Required vital missing.",
            })

    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    pass_count = sum(1 for r in results if r["status"] == "PASS")

    overall_status = "BLOCK_PUBLIC_VALIDATION" if fail_count else ("REVIEW_WARNINGS" if warn_count else "PASS_LOCAL_READINESS")

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_json": str(SUMMARY_JSON),
        "overall_status": overall_status,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "policy": {
            "raw_data": "No raw rows reviewed or exported by this sanity gate.",
            "public_claim": "Do not publish HiRID performance metrics until all vital sanity checks pass and aggregate operating-point validation is reviewed.",
            "blocked_terms": BLOCK_PUBLIC_TERMS,
        },
        "results": results,
    }

    SANITY_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = []
    lines.append("# HiRID Local-Only Vital Sanity Gate")
    lines.append("")
    lines.append(f"Timestamp: {output['timestamp_utc']}")
    lines.append(f"Overall Status: **{overall_status}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- PASS: {pass_count}")
    lines.append(f"- WARN: {warn_count}")
    lines.append(f"- FAIL: {fail_count}")
    lines.append("")
    lines.append("## Vital Checks")
    lines.append("")

    for r in results:
        lines.append(f"### {r['vital']}")
        lines.append("")
        lines.append(f"- Status: **{r['status']}**")
        lines.append(f"- Rows: {r['rows']:,}")
        lines.append(f"- Unique patients: {r['unique_patients']:,}")
        lines.append(f"- Mean: {r['mean']}")
        lines.append(f"- Threshold-positive rate: {r['threshold_positive_rate']}%")
        lines.append(f"- Note: {r['note']}")
        if r["issues"]:
            lines.append("- Issues:")
            for issue in r["issues"]:
                lines.append(f"  - {issue}")
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    if overall_status == "BLOCK_PUBLIC_VALIDATION":
        lines.append("Do **not** run public-facing HiRID validation claims yet. Correct the failed vital mapping first, especially SpO2.")
    elif overall_status == "REVIEW_WARNINGS":
        lines.append("Review warnings before moving to threshold operating-point validation.")
    else:
        lines.append("Local-only aggregate readiness appears acceptable for the next private operating-point validation step.")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append("This is local-only aggregate readiness. It is not clinical validation, prospective validation, or public performance evidence.")

    SANITY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("VITAL SANITY GATE COMPLETE")
    print("Overall status:", overall_status)
    print("PASS:", pass_count, "WARN:", warn_count, "FAIL:", fail_count)
    print("")
    print("Written:")
    print("-", SANITY_JSON)
    print("-", SANITY_MD)

    for r in results:
        print("")
        print(r["vital"], "=>", r["status"])
        if r["issues"]:
            for issue in r["issues"]:
                print(" -", issue)

if __name__ == "__main__":
    main()
