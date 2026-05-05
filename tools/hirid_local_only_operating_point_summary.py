#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
READINESS_JSON = OUT_DIR / "hirid_local_only_aggregate_readiness_summary.json"
SANITY_JSON = OUT_DIR / "hirid_local_only_vital_sanity_gate.json"
OUT_JSON = OUT_DIR / "hirid_local_only_operating_point_summary.json"
OUT_MD = OUT_DIR / "hirid_local_only_operating_point_summary.md"

THRESHOLD_PROFILES = {
    "t=4.0": {
        "profile": "higher-sensitivity review setting",
        "intended_demo_scope": "ICU / high-acuity private aggregate review simulation",
    },
    "t=5.0": {
        "profile": "balanced review setting",
        "intended_demo_scope": "mixed-unit private aggregate review simulation",
    },
    "t=6.0": {
        "profile": "conservative review setting",
        "intended_demo_scope": "telemetry / stepdown conservative private aggregate review simulation",
    },
}

NOT_ALLOWED_PUBLIC_CLAIMS = [
    "HiRID validated",
    "three-dataset validation completed",
    "clinical validation",
    "prospective validation",
    "final HiRID performance",
    "diagnosis or treatment direction",
    "independent escalation",
]

def pct(x):
    if x is None:
        return None
    try:
        return round(float(x), 4)
    except Exception:
        return None

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def load_json(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    readiness = load_json(READINESS_JSON)
    sanity = load_json(SANITY_JSON)

    fail_count = safe_int(sanity.get("fail_count", 0))
    failed = [
        r for r in sanity.get("results", [])
        if str(r.get("status", "")).upper() == "FAIL"
    ]

    if fail_count > 0 or failed or sanity.get("overall_status") == "BLOCK_PUBLIC_VALIDATION":
        raise SystemExit("Sanity gate has actual FAIL results. Stop before readiness summary.")

    vitals = readiness.get("vital_aggregate_summary", {})
    required = [
        "heart_rate",
        "spo2",
        "respiratory_rate",
        "systolic_bp",
        "diastolic_bp",
        "temperature",
    ]

    missing = [v for v in required if v not in vitals]
    if missing:
        raise SystemExit(f"Missing required vital summaries: {missing}")

    total_rows = safe_int(readiness.get("rows_scanned_total", 0))
    matched_rows = safe_int(readiness.get("vital_rows_matched_total", 0))

    vital_table = {}
    for vital in required:
        s = vitals.get(vital, {})

        rate_percent = s.get("threshold_positive_rate")
        if rate_percent is None and s.get("threshold_positive_rate_among_numeric_rows") is not None:
            rate_percent = float(s.get("threshold_positive_rate_among_numeric_rows", 0)) * 100

        vital_table[vital] = {
            "rows": s.get("rows"),
            "numeric_rows": s.get("numeric_rows"),
            "unique_patients_count": s.get("unique_patients_count"),
            "mean": s.get("mean"),
            "min": s.get("min"),
            "max": s.get("max"),
            "threshold_positive_rows": s.get("threshold_positive_rows"),
            "threshold_positive_rate_percent": pct(rate_percent),
            "out_of_sanity_range_rows": s.get("out_of_sanity_range_rows"),
            "variable_id_counts": s.get("variable_id_counts"),
        }

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_only_operating_point_readiness_summary_not_public_validation",
        "dataset": "HiRID v1.1.1",
        "allowed_public_wording_now": "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        "not_allowed_public_claims": NOT_ALLOWED_PUBLIC_CLAIMS,
        "privacy_policy": {
            "raw_rows_exported": False,
            "patient_level_outputs_exported": False,
            "timestamps_exported": False,
            "aggregate_only": True,
            "local_private_only": True,
        },
        "overall_counts": {
            "rows_scanned_total": total_rows,
            "matched_vital_rows_total": matched_rows,
        },
        "sanity_gate": {
            "overall_status": sanity.get("overall_status"),
            "fail_count": fail_count,
            "passed_for_private_readiness_summary": True,
        },
        "vital_aggregate_summary": vital_table,
        "threshold_profiles_for_next_private_run": THRESHOLD_PROFILES,
        "next_step": (
            "After review, run a private aggregate-only threshold operating-point pass for "
            "t=4.0, t=5.0, and t=6.0. Export aggregate tables only. Do not export raw rows, "
            "timestamps, patient-level records, patient-level predictions, or restricted files."
        ),
    }

    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# HiRID Local-Only Operating-Point Readiness Summary",
        "",
        f"Timestamp UTC: {output['timestamp_utc']}",
        "",
        "**Status:** Local-only operating-point readiness summary. This is **not public validation**.",
        "",
        "## Allowed Public Wording Right Now",
        "",
        "> HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        "",
        "## Overall Counts",
        "",
        f"- Rows scanned total: {total_rows:,}",
        f"- Matched vital rows total: {matched_rows:,}",
        "",
        "## Sanity Gate",
        "",
        f"- Overall status: {sanity.get('overall_status')}",
        f"- Fail count: {fail_count}",
        "- Result: Passed for private readiness summary",
        "",
        "## Vital Aggregate Summary",
        "",
    ]

    for vital, s in vital_table.items():
        rows = s.get("rows")
        numeric_rows = s.get("numeric_rows")
        patients = s.get("unique_patients_count")
        threshold_rows = s.get("threshold_positive_rows")
        out_range = s.get("out_of_sanity_range_rows")

        lines.extend([
            f"### {vital}",
            "",
            f"- Rows: {rows:,}" if isinstance(rows, int) else f"- Rows: {rows}",
            f"- Numeric rows: {numeric_rows:,}" if isinstance(numeric_rows, int) else f"- Numeric rows: {numeric_rows}",
            f"- Unique patients count: {patients:,}" if isinstance(patients, int) else f"- Unique patients count: {patients}",
            f"- Mean: {s.get('mean')}",
            f"- Min: {s.get('min')}",
            f"- Max: {s.get('max')}",
            f"- Threshold-positive rows: {threshold_rows:,}" if isinstance(threshold_rows, int) else f"- Threshold-positive rows: {threshold_rows}",
            f"- Threshold-positive rate: {s.get('threshold_positive_rate_percent')}%",
            f"- Out-of-sanity-range rows: {out_range:,}" if isinstance(out_range, int) else f"- Out-of-sanity-range rows: {out_range}",
            "",
        ])

    lines.extend([
        "## Threshold Profiles for the Next Private Run",
        "",
        "| Threshold | Profile | Demo Scope |",
        "|---|---|---|",
    ])

    for t, meta in THRESHOLD_PROFILES.items():
        lines.append(f"| {t} | {meta['profile']} | {meta['intended_demo_scope']} |")

    lines.extend([
        "",
        "## Guardrail",
        "",
        "Do not publish this as HiRID validation. This is local-only aggregate readiness for a private operating-point pass.",
        "",
        "Do not export raw rows, timestamps, patient-level records, patient-level predictions, or restricted files.",
        "",
        "## Not Allowed Public Claims",
        "",
    ])

    for claim in NOT_ALLOWED_PUBLIC_CLAIMS:
        lines.append(f"- {claim}")

    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("DONE — local-only operating-point readiness summary written.")
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")
    print("")
    print("Current public wording remains:")
    print("HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.")
    print("")
    print("Next technical step after reviewing this file:")
    print("Run a private aggregate-only threshold operating-point pass for t=4.0, t=5.0, and t=6.0.")

if __name__ == "__main__":
    main()
