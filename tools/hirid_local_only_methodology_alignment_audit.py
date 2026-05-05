#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
AUDIT_DIR = Path("data/validation/local_private/hirid/audit")

READINESS_JSON = OUT_DIR / "hirid_local_only_aggregate_readiness_summary.json"
SANITY_JSON = OUT_DIR / "hirid_local_only_vital_sanity_gate.json"
OPERATING_SUMMARY_JSON = OUT_DIR / "hirid_local_only_operating_point_summary.json"
THRESHOLD_JSON = OUT_DIR / "hirid_local_only_threshold_operating_points.json"
THRESHOLD_MD = OUT_DIR / "hirid_local_only_threshold_operating_points.md"

REVIEW_GATE_JSON = AUDIT_DIR / "hirid_local_only_threshold_review_gate.json"
METHODOLOGY_JSON = AUDIT_DIR / "hirid_local_only_methodology_alignment_audit.json"
METHODOLOGY_MD = AUDIT_DIR / "hirid_local_only_methodology_alignment_audit.md"

ALLOWED_PUBLIC_WORDING = "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."

BANNED_PUBLIC_CLAIMS = [
    "HiRID validated",
    "three-dataset validation completed",
    "clinical validation",
    "prospective validation",
    "final HiRID performance",
    "predicts deterioration",
    "prevents adverse events",
    "FPR",
    "detection rate",
    "lead time",
    "diagnosis",
    "treatment direction",
    "independent escalation",
]

REQUIRED_THRESHOLDS = {4.0, 5.0, 6.0}

def load_json(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise SystemExit(f"Missing required file: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def safe_int(x, default=0):
    try:
        return int(x or 0)
    except Exception:
        return default

def find_text_flags(text: str) -> list[str]:
    flags = []
    lowered = text.lower()

    forbidden_patterns = [
        "clinical validation",
        "prospective validation",
        "final hirid performance",
        "hirid validated",
        "three-dataset validation completed",
        "predicts deterioration",
        "prevents adverse events",
        "diagnosis",
        "treatment direction",
        "independent escalation",
    ]

    for phrase in forbidden_patterns:
        if phrase in lowered and "not allowed public claims" not in lowered:
            flags.append(f"Potential overclaim phrase found outside claim-control context: {phrase}")

    return flags

def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    readiness = load_json(READINESS_JSON)
    sanity = load_json(SANITY_JSON)
    operating_summary = load_json(OPERATING_SUMMARY_JSON, required=False)
    threshold = load_json(THRESHOLD_JSON)
    review_gate = load_json(REVIEW_GATE_JSON, required=False)

    issues = []
    warnings = []
    passes = []

    for p in [READINESS_JSON, SANITY_JSON, THRESHOLD_JSON, THRESHOLD_MD]:
        if p.exists():
            passes.append(f"Required aggregate/audit file found: {p}")
        else:
            issues.append(f"Missing required aggregate/audit file: {p}")

    sanity_fail_count = safe_int(sanity.get("fail_count", 0))
    failed_vitals = [
        r.get("vital")
        for r in sanity.get("results", [])
        if str(r.get("status", "")).upper() == "FAIL"
    ]

    if sanity_fail_count == 0 and not failed_vitals:
        passes.append("Sanity gate has no actual failed vitals.")
    else:
        issues.append(f"Sanity gate still has failed vitals: {failed_vitals}")

    if review_gate:
        review_status = str(review_gate.get("review_status", ""))
        issue_count = safe_int(review_gate.get("issue_count", 0))
        warning_count = safe_int(review_gate.get("warning_count", 0))

        if review_status == "PASS_PRIVATE_REVIEW_READY" and issue_count == 0:
            passes.append("Threshold review gate is PASS_PRIVATE_REVIEW_READY with zero issues.")
        else:
            issues.append(f"Threshold review gate not clean: status={review_status}, issues={issue_count}, warnings={warning_count}")
    else:
        warnings.append("Threshold review gate JSON not found. Continuing audit based on threshold output only.")

    privacy = threshold.get("privacy_policy", {})
    privacy_expected_false = [
        "raw_rows_exported",
        "patient_level_outputs_exported",
        "timestamps_exported",
        "patient_ids_exported",
    ]

    for key in privacy_expected_false:
        if privacy.get(key) is False:
            passes.append(f"Privacy control OK: {key}=False")
        else:
            issues.append(f"Privacy control problem: expected {key}=False")

    if privacy.get("aggregate_only") is True and privacy.get("local_private_only") is True:
        passes.append("Privacy scope OK: aggregate_only=True and local_private_only=True")
    else:
        issues.append("Privacy scope problem: aggregate_only/local_private_only not both true.")

    method_note = str(threshold.get("method_note", ""))
    method_note_lower = method_note.lower()

    if "does not calculate fpr" in method_note_lower and "detection" in method_note_lower and "lead-time" in method_note_lower:
        passes.append("Method note correctly states this pass does not calculate FPR, detection, or lead time.")
    else:
        warnings.append("Method note should clearly state this does not calculate FPR, detection, or lead time.")

    threshold_results = threshold.get("threshold_results", [])
    thresholds_found = set()
    for r in threshold_results:
        try:
            thresholds_found.add(float(r.get("threshold")))
        except Exception:
            pass

    if REQUIRED_THRESHOLDS.issubset(thresholds_found):
        passes.append("Required threshold profiles present: t=4.0, t=5.0, t=6.0.")
    else:
        issues.append(f"Missing required thresholds. Found={sorted(thresholds_found)}")

    overall_counts = threshold.get("overall_counts", {})
    rows_scanned = safe_int(overall_counts.get("rows_scanned_total"))
    matched_vitals = safe_int(overall_counts.get("matched_vital_rows_total"))
    scored_rows = safe_int(overall_counts.get("scored_signal_rows_total"))
    total_bins = safe_int(overall_counts.get("total_scored_patient_hour_bins"))
    any_signal_bins = safe_int(overall_counts.get("any_signal_bins"))
    unique_any_signal = safe_int(overall_counts.get("unique_patients_any_signal"))

    if rows_scanned > 0 and matched_vitals > 0 and total_bins > 0:
        passes.append("Aggregate counts are nonzero and suitable for private methodology review.")
    else:
        issues.append("Aggregate counts are missing or zero.")

    if unique_any_signal > 0:
        passes.append("Unique patient aggregate count is above zero.")
    else:
        issues.append("Unique patient aggregate count is zero.")

    allowed = str(threshold.get("allowed_public_wording_now", ""))

    if allowed == ALLOWED_PUBLIC_WORDING:
        passes.append("Allowed public wording is conservative and unchanged.")
    else:
        warnings.append("Allowed public wording differs from the approved conservative sentence.")

    not_allowed = threshold.get("not_allowed_public_claims", [])
    missing_banned = [c for c in BANNED_PUBLIC_CLAIMS if c not in not_allowed]

    if not missing_banned:
        passes.append("Not-allowed public claims list includes all required banned claim categories.")
    else:
        warnings.append(f"Some banned claim categories are missing from not_allowed_public_claims: {missing_banned}")

    if THRESHOLD_MD.exists():
        md_text = THRESHOLD_MD.read_text(encoding="utf-8", errors="ignore")
        text_flags = find_text_flags(md_text)

        if text_flags:
            warnings.extend(text_flags)
        else:
            passes.append("Threshold markdown does not appear to contain uncontrolled public overclaim wording.")

    if issues:
        overall_status = "METHOD_REVIEW_BLOCKED"
        decision = (
            "Do not proceed to any public-facing HiRID wording. Fix the methodology issues first. "
            "This remains local-only aggregate research work."
        )
    elif warnings:
        overall_status = "METHOD_ALIGNED_WITH_WARNINGS_PRIVATE_ONLY"
        decision = (
            "Methodology appears aligned for private aggregate review, but warnings should be reviewed before any next-stage claims. "
            "Do not publish HiRID validation language."
        )
    else:
        overall_status = "METHOD_ALIGNED_PRIVATE_AGGREGATE_REVIEW_READY"
        decision = (
            "Methodology is aligned for private aggregate review only. This supports internal methodology documentation, "
            "not public HiRID validation claims."
        )

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "decision": decision,
        "scope": "HiRID v1.1.1 local-only aggregate methodology alignment audit",
        "allowed_public_wording_now": ALLOWED_PUBLIC_WORDING,
        "not_allowed_public_claims": BANNED_PUBLIC_CLAIMS,
        "files_reviewed": {
            "readiness_json": str(READINESS_JSON),
            "sanity_json": str(SANITY_JSON),
            "operating_summary_json": str(OPERATING_SUMMARY_JSON) if OPERATING_SUMMARY_JSON.exists() else None,
            "threshold_json": str(THRESHOLD_JSON),
            "threshold_md": str(THRESHOLD_MD),
            "review_gate_json": str(REVIEW_GATE_JSON) if REVIEW_GATE_JSON.exists() else None,
        },
        "aggregate_counts_reviewed": {
            "rows_scanned_total": rows_scanned,
            "matched_vital_rows_total": matched_vitals,
            "scored_signal_rows_total": scored_rows,
            "total_scored_patient_hour_bins": total_bins,
            "any_signal_bins": any_signal_bins,
            "unique_patients_any_signal": unique_any_signal,
        },
        "thresholds_found": sorted(thresholds_found),
        "passes": passes,
        "warnings": warnings,
        "issues": issues,
        "guardrail": {
            "not_public_validation": True,
            "not_clinical_validation": True,
            "not_prospective_validation": True,
            "no_fpr_detection_or_lead_time_claim": True,
            "no_raw_rows": True,
            "no_patient_level_outputs": True,
            "no_timestamps_exported": True,
        }
    }

    METHODOLOGY_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# HiRID Local-Only Methodology Alignment Audit",
        "",
        f"Timestamp UTC: {audit['timestamp_utc']}",
        "",
        f"**Overall Status:** {overall_status}",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Current Allowed Public Wording",
        "",
        f"> {ALLOWED_PUBLIC_WORDING}",
        "",
        "## Methodology Scope",
        "",
        "- Local-only retrospective aggregate review.",
        "- Rules-based private operating-point behavior using aggregate patient-hour review bins.",
        "- Aggregate summaries only.",
        "- No raw rows, timestamps, patient IDs, patient-level records, or patient-level predictions exported.",
        "- Not clinical validation.",
        "- Not prospective validation.",
        "- Not final HiRID performance evidence.",
        "- Does not calculate FPR, detection rate, or lead time.",
        "",
        "## Aggregate Counts Reviewed",
        "",
        f"- Rows scanned total: {rows_scanned:,}",
        f"- Matched vital rows total: {matched_vitals:,}",
        f"- Scored signal rows total: {scored_rows:,}",
        f"- Total scored patient-hour bins: {total_bins:,}",
        f"- Any-signal bins: {any_signal_bins:,}",
        f"- Unique patients with any signal: {unique_any_signal:,}",
        "",
        "## Thresholds Found",
        "",
        f"- {', '.join('t=' + str(x) for x in sorted(thresholds_found))}",
        "",
        "## Pass Checks",
        "",
    ]

    if passes:
        lines.extend([f"- PASS: {p}" for p in passes])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])

    if warnings:
        lines.extend([f"- WARNING: {w}" for w in warnings])
    else:
        lines.append("- None")

    lines.extend(["", "## Issues", ""])

    if issues:
        lines.extend([f"- ISSUE: {i}" for i in issues])
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Not Allowed Public Claims",
        "",
    ])

    for claim in BANNED_PUBLIC_CLAIMS:
        lines.append(f"- {claim}")

    lines.extend([
        "",
        "## Guardrail",
        "",
        "Do not update public pages or claim HiRID validation yet.",
        "",
        "Correct current wording remains:",
        "",
        f"> {ALLOWED_PUBLIC_WORDING}",
        "",
    ])

    METHODOLOGY_MD.write_text("\n".join(lines), encoding="utf-8")

    print("METHODOLOGY ALIGNMENT AUDIT COMPLETE")
    print("Overall status:", overall_status)
    print("Issues:", len(issues))
    print("Warnings:", len(warnings))
    print("JSON:", METHODOLOGY_JSON)
    print("MD:", METHODOLOGY_MD)

if __name__ == "__main__":
    main()
