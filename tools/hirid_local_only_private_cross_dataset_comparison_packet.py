#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(".")
DATA = ROOT / "data" / "validation"
AUDIT_DIR = DATA / "local_private" / "hirid" / "audit"

HIRID_LOCK_JSON = AUDIT_DIR / "hirid_local_only_outcome_label_methodology_lock.json"
OUT_JSON = AUDIT_DIR / "hirid_local_only_private_cross_dataset_comparison_packet.json"
OUT_MD = AUDIT_DIR / "hirid_local_only_private_cross_dataset_comparison_packet.md"

PUBLIC_WORDING = "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."

ACCEPTABLE_HIRID_LOCK_DECISIONS = {
    "LOCKED_LOCAL_ONLY_OUTCOME_LABEL_METHODOLOGY_PACKET",
    "LOCKED_LOCAL_ONLY_OUTCOME_LABEL_METHODOLOGY_PACKET_WITH_WARNINGS",
}

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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_num(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return None


def find_dataset_files(dataset_key: str):
    """
    Best-effort aggregate JSON discovery only.
    This intentionally ignores obvious raw/private row-level file formats.
    """
    matches = []

    for p in DATA.rglob("*.json"):
        s = str(p).lower()

        if any(bad in s for bad in [
            ".csv",
            ".parquet",
            ".sqlite",
            ".tar.gz",
            ".zip",
            "/raw/",
            "patient_level",
            "row_level",
            "rows",
            "patient_ids",
            "timestamps",
        ]):
            continue

        if dataset_key in s and any(k in s for k in [
            "validation",
            "evidence",
            "threshold",
            "operating",
            "summary",
            "release",
            "lock",
            "aggregate",
        ]):
            matches.append(p)

    return sorted(set(matches))


def summarize_file(path: Path):
    data = load_json(path)

    if not isinstance(data, dict):
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "status": "json_not_dictionary_or_unreadable",
        }

    found_metrics = {}

    metric_patterns = {
        "alert_reduction_percent": [
            "alert_reduction_percent",
            "alert_reduction",
            "estimated_reduction_vs_any_single_signal_bins_percent",
            "estimated_reduction",
        ],
        "false_positive_rate_percent": [
            "false_positive_rate_percent",
            "fpr_percent",
            "fpr",
        ],
        "median_lead_time_hours": [
            "median_lead_time_hours",
            "lead_time_hours",
            "median_lead_time",
        ],
        "threshold": [
            "threshold",
        ],
        "detection_percent": [
            "detection_percent",
            "patient_detection_percent",
            "event_capture_proxy_percent",
        ],
    }

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()

                for metric, names in metric_patterns.items():
                    if key in names or any(n in key for n in names):
                        if isinstance(v, (int, float, str)):
                            n = safe_num(v)
                            if n is not None:
                                found_metrics.setdefault(metric, []).append(n)

                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    safe_metrics = {}

    for k, vals in found_metrics.items():
        nums = [v for v in vals if isinstance(v, (int, float))]
        if nums:
            safe_metrics[k] = {
                "min": round(min(nums), 4),
                "max": round(max(nums), 4),
                "count": len(nums),
            }

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "status": str(data.get("status", "aggregate_json_found")),
        "dataset": str(data.get("dataset", "")),
        "safe_metric_ranges_detected": safe_metrics,
    }


def main():
    if not HIRID_LOCK_JSON.exists():
        raise SystemExit(f"Missing HiRID lock JSON: {HIRID_LOCK_JSON}")

    hirid_lock = load_json(HIRID_LOCK_JSON)

    if not isinstance(hirid_lock, dict):
        raise SystemExit("HiRID lock JSON could not be read.")

    decision = str(hirid_lock.get("decision", ""))
    issues = hirid_lock.get("issues", [])
    warnings = hirid_lock.get("warnings", [])

    if decision not in ACCEPTABLE_HIRID_LOCK_DECISIONS:
        raise SystemExit(f"HiRID outcome-label lock is not acceptable for comparison packet. Decision: {decision}")

    if issues:
        raise SystemExit(f"HiRID outcome-label lock has issues: {issues}")

    hirid_results = hirid_lock.get("normalized_threshold_results", [])

    mimic_files = find_dataset_files("mimic")
    eicu_files = find_dataset_files("eicu")

    mimic_summaries = [summarize_file(p) for p in mimic_files[:12]]
    eicu_summaries = [summarize_file(p) for p in eicu_files[:12]]

    packet = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "private_cross_dataset_comparison_packet_not_public_validation",
        "allowed_public_wording_now": PUBLIC_WORDING,
        "not_allowed_public_claims": BANNED_PUBLIC_CLAIMS,
        "privacy_policy": {
            "raw_rows_exported": False,
            "patient_level_outputs_exported": False,
            "timestamps_exported": False,
            "patient_ids_exported": False,
            "aggregate_only": True,
            "local_private_only": True,
        },
        "methodology_boundary": {
            "mimic_iv": "Use only previously locked aggregate procedure/event-window metrics if present.",
            "eicu": "Use only previously locked aggregate outcome-proxy or harmonized metrics if present.",
            "hirid": "Use discharge_status outcome-proxy metrics only; not directly equivalent to MIMIC event-window FPR/detection/lead-time.",
            "important_limitation": "Do not combine these into a single public three-dataset validation claim.",
        },
        "hirid_outcome_proxy_summary": {
            "lock_decision": decision,
            "lock_warnings": warnings,
            "threshold_results": hirid_results,
        },
        "mimic_candidate_aggregate_files": mimic_summaries,
        "eicu_candidate_aggregate_files": eicu_summaries,
        "internal_assessment": {
            "what_is_supported": [
                "HiRID private aggregate outcome-proxy behavior is locked and audit-ready.",
                "HiRID t=6.0 conservative threshold shows lower overall labeled review burden than t=4.0 and t=5.0.",
                "HiRID uses discharge_status outcome proxy, so it must be discussed separately from event-window FPR/detection methods.",
            ],
            "what_is_not_supported": [
                "Public HiRID validation claim.",
                "Public three-dataset validation completed claim.",
                "Clinical validation claim.",
                "Prospective validation claim.",
                "Final HiRID performance claim.",
                "Claim that HiRID FPR/detection/lead-time is directly comparable to MIMIC procedure-event-window metrics.",
            ],
            "safe_next_step": "Use this packet as an internal comparison aid only. Public wording should remain unchanged.",
        },
    }

    OUT_JSON.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    lines = [
        "# Private Cross-Dataset Comparison Packet",
        "",
        f"Timestamp UTC: {packet['timestamp_utc']}",
        "",
        "**Status:** Private/internal aggregate comparison aid. This is **not public validation**.",
        "",
        "## Current Safe Public Wording",
        "",
        f"> {PUBLIC_WORDING}",
        "",
        "## Methodology Boundary",
        "",
        "| Dataset | Safe Interpretation | Public Limitation |",
        "|---|---|---|",
        "| MIMIC-IV | Procedure/event-window aggregate metrics may be discussed only if already locked and DUA-safe. | Do not merge with HiRID as identical methodology. |",
        "| eICU | Outcome-proxy or harmonized aggregate metrics may be discussed only if already locked and DUA-safe. | Do not overstate clinical validation. |",
        "| HiRID | Discharge-status outcome-proxy behavior is privately locked. | Not public validation; not equivalent to event-window FPR/detection/lead-time. |",
        "",
        "## HiRID Lock Status",
        "",
        f"- Decision: {decision}",
        f"- Issues: {len(issues)}",
        f"- Warnings: {len(warnings)}",
        "",
    ]

    if warnings:
        lines.append("### HiRID Lock Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- WARNING: {w}")
        lines.append("")

    lines.extend([
        "## HiRID Outcome-Proxy Threshold Results",
        "",
        "| Threshold | Positive bins | Labeled patients reviewed | Event reviewed | Non-event reviewed | Event capture proxy | Non-event review proxy | Overall labeled review rate | Event mix among reviewed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in hirid_results:
        lines.append(
            f"| t={r.get('threshold')} | "
            f"{int(r.get('positive_bins', 0)):,} | "
            f"{int(r.get('labeled_patients_reviewed', 0)):,} | "
            f"{int(r.get('event_patients_reviewed', 0)):,} | "
            f"{int(r.get('non_event_patients_reviewed', 0)):,} | "
            f"{r.get('event_capture_proxy_percent')}% | "
            f"{r.get('non_event_review_proxy_percent')}% | "
            f"{r.get('overall_labeled_review_rate_percent')}% | "
            f"{r.get('event_mix_among_reviewed_percent')}% |"
        )

    lines.extend([
        "",
        "## Internal Read",
        "",
        "- HiRID t=6.0 remains the conservative operating point.",
        "- HiRID t=6.0 shows lower overall labeled review burden compared with t=4.0 and t=5.0.",
        "- HiRID must remain framed as discharge-status outcome-proxy analysis, not event-window clinical validation.",
        "- The public site should not say HiRID validated or three-dataset validation completed.",
        "",
        "## MIMIC-IV Aggregate Candidate Files Found",
        "",
    ])

    if mimic_summaries:
        for s in mimic_summaries:
            lines.append(f"- {s['path']} | status: {s.get('status')} | bytes: {s.get('bytes')}")
    else:
        lines.append("- None found by filename scan. This does not mean MIMIC evidence is missing; it means no obvious MIMIC aggregate JSON was found by this script.")

    lines.extend([
        "",
        "## eICU Aggregate Candidate Files Found",
        "",
    ])

    if eicu_summaries:
        for s in eicu_summaries:
            lines.append(f"- {s['path']} | status: {s.get('status')} | bytes: {s.get('bytes')}")
    else:
        lines.append("- None found by filename scan. This does not mean eICU evidence is missing; it means no obvious eICU aggregate JSON was found by this script.")

    lines.extend([
        "",
        "## What This Supports Internally",
        "",
        "- HiRID private aggregate outcome-proxy behavior is locked and audit-ready.",
        "- The HiRID workflow is privacy-controlled and aggregate-only.",
        "- The result can strengthen internal validation planning and investor/hospital diligence preparation.",
        "",
        "## What This Does Not Support Publicly",
        "",
    ])

    for claim in BANNED_PUBLIC_CLAIMS:
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
        "## Guardrail",
        "",
        "Do not update public pages or claim HiRID validation yet.",
        "",
        "Correct current wording remains:",
        "",
        f"> {PUBLIC_WORDING}",
        "",
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("PRIVATE CROSS-DATASET COMPARISON PACKET COMPLETE")
    print("=" * 80)
    print("JSON:", OUT_JSON)
    print("MD:", OUT_MD)
    print("MIMIC candidate aggregate files:", len(mimic_summaries))
    print("eICU candidate aggregate files:", len(eicu_summaries))
    print("")
    print("Do NOT publish this as HiRID validation.")
    print("Correct public wording remains:")
    print(PUBLIC_WORDING)


if __name__ == "__main__":
    main()
