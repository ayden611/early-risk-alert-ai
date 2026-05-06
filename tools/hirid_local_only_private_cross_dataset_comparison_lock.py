#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

AUDIT_DIR = Path("data/validation/local_private/hirid/audit")

CROSS_JSON = AUDIT_DIR / "hirid_local_only_private_cross_dataset_comparison_packet.json"
CROSS_MD = AUDIT_DIR / "hirid_local_only_private_cross_dataset_comparison_packet.md"
HIRID_LOCK_JSON = AUDIT_DIR / "hirid_local_only_outcome_label_methodology_lock.json"

LOCK_JSON = AUDIT_DIR / "hirid_local_only_private_cross_dataset_comparison_lock.json"
LOCK_MD = AUDIT_DIR / "hirid_local_only_private_cross_dataset_comparison_lock.md"

PUBLIC_WORDING = "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."

ACCEPTABLE_HIRID_LOCK_DECISIONS = {
    "LOCKED_LOCAL_ONLY_OUTCOME_LABEL_METHODOLOGY_PACKET",
    "LOCKED_LOCAL_ONLY_OUTCOME_LABEL_METHODOLOGY_PACKET_WITH_WARNINGS",
}

REQUIRED_BANNED_CLAIMS = [
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

REQUIRED_PRIVACY_FALSE = [
    "raw_rows_exported",
    "patient_level_outputs_exported",
    "timestamps_exported",
    "patient_ids_exported",
]

REQUIRED_PRIVACY_TRUE = [
    "aggregate_only",
    "local_private_only",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def has_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def main():
    failures = []
    warnings = []

    for path in [CROSS_JSON, CROSS_MD, HIRID_LOCK_JSON]:
        if not path.exists():
            failures.append(f"Missing required file: {path}")

    if failures:
        raise SystemExit("\n".join(failures))

    cross = load_json(CROSS_JSON)
    hirid_lock = load_json(HIRID_LOCK_JSON)
    md_text = CROSS_MD.read_text(encoding="utf-8")

    status = str(cross.get("status", ""))
    if "not_public_validation" not in status:
        failures.append(f"Cross-dataset packet status is not safely marked private/not-public: {status}")

    if cross.get("allowed_public_wording_now") != PUBLIC_WORDING:
        failures.append("Allowed public wording does not match conservative approved wording.")

    privacy = cross.get("privacy_policy", {})

    for key in REQUIRED_PRIVACY_FALSE:
        if privacy.get(key) is not False:
            failures.append(f"Privacy control failed: {key} must be False")

    for key in REQUIRED_PRIVACY_TRUE:
        if privacy.get(key) is not True:
            failures.append(f"Privacy control failed: {key} must be True")

    banned = cross.get("not_allowed_public_claims", [])

    for claim in REQUIRED_BANNED_CLAIMS:
        if claim not in banned:
            failures.append(f"Missing banned public claim category: {claim}")

    method_boundary = cross.get("methodology_boundary", {})
    boundary_text = json.dumps(method_boundary, indent=2).lower()

    required_boundary_phrases = [
        "not directly equivalent",
        "do not combine",
        "three-dataset validation claim",
        "discharge_status",
        "event-window",
    ]

    for phrase in required_boundary_phrases:
        if phrase not in boundary_text:
            failures.append(f"Methodology boundary missing required phrase/context: {phrase}")

    hirid_decision = str(hirid_lock.get("decision", ""))

    if hirid_decision not in ACCEPTABLE_HIRID_LOCK_DECISIONS:
        failures.append(f"HiRID outcome-label lock decision is not acceptable: {hirid_decision}")

    hirid_issues = hirid_lock.get("issues") or []
    hirid_warnings = hirid_lock.get("warnings") or []

    if hirid_issues:
        failures.append(f"HiRID outcome-label lock has issues: {hirid_issues}")

    for w in hirid_warnings:
        warnings.append(f"HiRID lock warning carried forward: {w}")

    hirid_summary = cross.get("hirid_outcome_proxy_summary", {})
    threshold_results = hirid_summary.get("threshold_results", [])

    if not threshold_results or len(threshold_results) < 3:
        failures.append("HiRID outcome-proxy threshold results missing or incomplete.")

    thresholds_seen = {str(r.get("threshold")) for r in threshold_results}

    for required in ["4.0", "5.0", "6.0"]:
        if required not in thresholds_seen:
            failures.append(f"Missing HiRID threshold result: t={required}")

    sorted_rows = sorted(threshold_results, key=lambda r: float(r.get("threshold", 0)))
    positive_bins = [int(r.get("positive_bins", 0) or 0) for r in sorted_rows]

    if positive_bins != sorted(positive_bins, reverse=True):
        failures.append(f"Positive review bins are not monotonic decreasing across thresholds: {positive_bins}")

    mimic_files = cross.get("mimic_candidate_aggregate_files", [])
    eicu_files = cross.get("eicu_candidate_aggregate_files", [])

    if not mimic_files:
        warnings.append("No MIMIC aggregate candidate files detected in private comparison packet.")

    if not eicu_files:
        warnings.append("No eICU aggregate candidate files detected in private comparison packet.")

    overclaim_phrases = [
        "hirid validated",
        "three-dataset validation completed",
        "clinically validated",
        "prospectively validated",
        "final hirid performance",
    ]

    exact_banned_lower = [c.lower() for c in REQUIRED_BANNED_CLAIMS]

    for phrase in overclaim_phrases:
        if has_phrase(md_text, phrase) and phrase not in exact_banned_lower:
            failures.append(f"Potential uncontrolled public overclaim phrase found in markdown: {phrase}")

    decision = (
        "LOCKED_LOCAL_ONLY_PRIVATE_CROSS_DATASET_COMPARISON_PACKET"
        if not failures
        else "BLOCK_PRIVATE_CROSS_DATASET_LOCK_REVIEW_REQUIRED"
    )

    manifest_paths = [
        CROSS_JSON,
        CROSS_MD,
        HIRID_LOCK_JSON,
    ]

    locked_manifest = []

    for path in manifest_paths:
        locked_manifest.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "issues": failures,
        "warnings": warnings,
        "allowed_public_wording_now": PUBLIC_WORDING,
        "privacy_policy_confirmed": {
            "raw_rows_exported": False,
            "patient_level_outputs_exported": False,
            "timestamps_exported": False,
            "patient_ids_exported": False,
            "aggregate_only": True,
            "local_private_only": True,
        },
        "methodology_boundary_locked": {
            "mimic_iv": "Procedure/event-window aggregate metrics remain methodologically separate.",
            "eicu": "Outcome-proxy or harmonized aggregate metrics remain methodologically separate.",
            "hirid": "Discharge-status outcome-proxy metrics remain local/private and are not equivalent to event-window FPR/detection/lead-time.",
            "public_claim_boundary": "Do not publish HiRID validation or three-dataset validation completed claims.",
        },
        "private_cross_dataset_summary": {
            "mimic_candidate_aggregate_files_count": len(mimic_files),
            "eicu_candidate_aggregate_files_count": len(eicu_files),
            "hirid_threshold_results_count": len(threshold_results),
            "hirid_thresholds_seen": sorted(thresholds_seen),
            "hirid_positive_bins_monotonic": positive_bins == sorted(positive_bins, reverse=True),
            "hirid_lock_decision": hirid_decision,
            "hirid_lock_warnings_count": len(hirid_warnings),
        },
        "not_allowed_public_claims": REQUIRED_BANNED_CLAIMS,
        "locked_file_manifest": locked_manifest,
    }

    LOCK_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# Private Cross-Dataset Comparison Lock Packet",
        "",
        f"Timestamp UTC: {output['timestamp_utc']}",
        "",
        f"**Decision:** {decision}",
        "",
        f"**Issues:** {len(failures)}",
        f"**Warnings:** {len(warnings)}",
        "",
        "## What This Lock Confirms",
        "",
        "- The private cross-dataset comparison packet exists.",
        "- The HiRID outcome-label methodology lock is acceptable for private/internal comparison.",
        "- HiRID remains discharge-status outcome-proxy only.",
        "- MIMIC/eICU and HiRID are methodologically separated.",
        "- The packet does not support a public HiRID validation claim.",
        "- The packet does not support a public three-dataset validation completed claim.",
        "",
        "## Private Cross-Dataset Summary",
        "",
        f"- MIMIC candidate aggregate files found: {len(mimic_files)}",
        f"- eICU candidate aggregate files found: {len(eicu_files)}",
        f"- HiRID threshold rows found: {len(threshold_results)}",
        f"- HiRID thresholds seen: {', '.join(sorted(thresholds_seen))}",
        f"- HiRID positive bins monotonic across thresholds: {positive_bins == sorted(positive_bins, reverse=True)}",
        f"- HiRID lock decision: {hirid_decision}",
        f"- HiRID lock warnings carried forward: {len(hirid_warnings)}",
        "",
        "## Current Safe Public Wording",
        "",
        f"> {PUBLIC_WORDING}",
        "",
        "## Issues",
        "",
    ]

    if failures:
        for f in failures:
            lines.append(f"- ISSUE: {f}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Warnings",
        "",
    ])

    if warnings:
        for w in warnings:
            lines.append(f"- WARNING: {w}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Methodology Boundary",
        "",
        "| Dataset | Locked Boundary |",
        "|---|---|",
        "| MIMIC-IV | Procedure/event-window aggregate metrics remain separate. |",
        "| eICU | Outcome-proxy or harmonized aggregate metrics remain separate. |",
        "| HiRID | Discharge-status outcome-proxy metrics remain private/local and not equivalent to event-window FPR/detection/lead-time. |",
        "",
        "## Not Allowed Public Claims",
        "",
    ])

    for claim in REQUIRED_BANNED_CLAIMS:
        lines.append(f"- {claim}")

    lines.extend([
        "",
        "## Privacy Controls",
        "",
        "- raw_rows_exported: false",
        "- patient_level_outputs_exported: false",
        "- timestamps_exported: false",
        "- patient_ids_exported: false",
        "- aggregate_only: true",
        "- local_private_only: true",
        "",
        "## Locked File Manifest",
        "",
        "| File | SHA-256 | Bytes |",
        "|---|---|---:|",
    ])

    for rec in locked_manifest:
        lines.append(f"| {rec['path']} | {rec['sha256']} | {rec['bytes']} |")

    lines.extend([
        "",
        "## Guardrail",
        "",
        "Do not update public pages or claim HiRID validation yet.",
        "",
        "Correct current wording remains:",
        "",
        f"> {PUBLIC_WORDING}",
        "",
    ])

    LOCK_MD.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("PRIVATE CROSS-DATASET COMPARISON LOCK RESULT")
    print("=" * 80)
    print("Decision:", decision)
    print("Issues:", len(failures))
    print("Warnings:", len(warnings))
    print("MIMIC candidate aggregate files:", len(mimic_files))
    print("eICU candidate aggregate files:", len(eicu_files))
    print("HiRID threshold rows:", len(threshold_results))
    print("HiRID lock decision:", hirid_decision)
    print("HiRID lock warnings:", len(hirid_warnings))
    print("Lock JSON:", LOCK_JSON)
    print("Lock MD:", LOCK_MD)

    if failures:
        print("")
        print("ISSUES:")
        for f in failures:
            print("-", f)
        raise SystemExit("STOP: private cross-dataset comparison lock has issues.")


if __name__ == "__main__":
    main()
