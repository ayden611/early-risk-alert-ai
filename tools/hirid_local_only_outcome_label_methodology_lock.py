#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
AUDIT_DIR = Path("data/validation/local_private/hirid/audit")

OUTCOME_JSON = OUT_DIR / "hirid_local_only_outcome_label_operating_points.json"
OUTCOME_MD = OUT_DIR / "hirid_local_only_outcome_label_operating_points.md"
SANITY_JSON = OUT_DIR / "hirid_local_only_vital_sanity_gate.json"
GENERAL_SCHEMA_JSON = AUDIT_DIR / "hirid_local_only_general_table_schema_audit.json"

LOCK_JSON = AUDIT_DIR / "hirid_local_only_outcome_label_methodology_lock.json"
LOCK_MD = AUDIT_DIR / "hirid_local_only_outcome_label_methodology_lock.md"

PUBLIC_WORDING = "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."

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

EXPECTED_THRESHOLDS = [4.0, 5.0, 6.0]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_num(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip().replace("%", "").replace(",", "")
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def pick(d: dict, names: list[str]):
    for name in names:
        if name in d and d.get(name) is not None:
            return d.get(name)
    return None


def pct(numer, denom):
    n = safe_num(numer)
    d = safe_num(denom)
    if n is None or d in (None, 0):
        return None
    return round((n / d) * 100.0, 4)


def threshold_float(x):
    if x is None:
        return None
    try:
        return float(str(x).replace("t=", "").strip())
    except Exception:
        return None


def find_threshold_rows(obj):
    found = []

    def walk(x):
        if isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                if any("threshold" in i or "t" in i for i in x):
                    found.append(x)
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)

    if not found:
        return []

    found.sort(key=len, reverse=True)
    return found[0]


def main():
    failures = []
    warnings = []

    outcome = load_json(OUTCOME_JSON)
    sanity = load_json(SANITY_JSON)

    schema = {}
    if GENERAL_SCHEMA_JSON.exists():
        schema = load_json(GENERAL_SCHEMA_JSON)
    else:
        warnings.append(f"General schema audit JSON not found: {GENERAL_SCHEMA_JSON}")

    sanity_fail_count = int(sanity.get("fail_count", 0) or 0)
    sanity_failed = [
        r for r in sanity.get("results", [])
        if str(r.get("status", "")).upper() == "FAIL"
    ]

    if sanity_fail_count > 0 or sanity_failed or sanity.get("overall_status") == "BLOCK_PUBLIC_VALIDATION":
        failures.append("Sanity gate has actual failed vitals.")

    privacy = outcome.get("privacy_policy", {})
    expected_privacy = {
        "raw_rows_exported": False,
        "patient_level_outputs_exported": False,
        "timestamps_exported": False,
        "patient_ids_exported": False,
        "aggregate_only": True,
        "local_private_only": True,
    }

    for key, expected in expected_privacy.items():
        if privacy.get(key) != expected:
            failures.append(f"Privacy control mismatch: {key} expected {expected}, found {privacy.get(key)}")

    status_text = str(outcome.get("status", ""))
    if "not_public_validation" not in status_text:
        warnings.append("Outcome JSON status does not explicitly include not_public_validation.")

    method_note = str(outcome.get("method_note", "")).lower()
    if "does not" not in method_note:
        warnings.append("Method note should explicitly say what this pass does not establish.")

    overall = outcome.get("overall_counts", {})

    total_labeled = pick(overall, [
        "labeled_patients_total",
        "total_labeled_patients",
        "patients_with_outcome_label_total",
        "outcome_labeled_patients_total",
    ])

    event_total = pick(overall, [
        "event_patients_total",
        "labeled_event_patients_total",
        "event_patient_total",
    ])

    non_event_total = pick(overall, [
        "non_event_patients_total",
        "labeled_non_event_patients_total",
        "non_event_patient_total",
    ])

    rows = find_threshold_rows(outcome)
    if not rows:
        failures.append("No threshold results found in outcome JSON.")

    normalized = []
    thresholds_seen = []

    for r in rows:
        t = threshold_float(pick(r, ["threshold", "t"]))

        if t is None:
            failures.append(f"Threshold row missing readable threshold: {r}")
            continue

        thresholds_seen.append(t)

        positive_bins = pick(r, [
            "positive_bins",
            "positive_review_bins",
            "positive_review_patient_hour_bins",
        ])

        labeled_reviewed = pick(r, [
            "labeled_patients_reviewed",
            "labeled_patient_reviewed_count",
            "patients_reviewed_with_label",
            "reviewed_labeled_patients",
        ])

        event_reviewed = pick(r, [
            "event_patients_reviewed",
            "event_patient_reviewed_count",
            "reviewed_event_patients",
        ])

        non_event_reviewed = pick(r, [
            "non_event_patients_reviewed",
            "non_event_patient_reviewed_count",
            "reviewed_non_event_patients",
        ])

        event_capture = pick(r, [
            "event_capture_proxy_percent",
            "event_capture_rate_percent",
            "event_capture_proxy",
        ])

        non_event_review = pick(r, [
            "non_event_review_proxy_percent",
            "non_event_review_rate_percent",
            "non_event_review_proxy",
        ])

        overall_rate = pick(r, [
            "overall_labeled_review_rate_percent",
            "overall_labeled_review_rate",
            "overall_review_rate_percent",
            "overall_review_rate",
        ])

        event_mix = pick(r, [
            "event_mix_among_reviewed_percent",
            "event_mix_among_reviewed",
            "event_mix_percent",
        ])

        if safe_num(event_capture) is None:
            event_capture = pct(event_reviewed, event_total)

        if safe_num(non_event_review) is None:
            non_event_review = pct(non_event_reviewed, non_event_total)

        if safe_num(overall_rate) is None:
            overall_rate = pct(labeled_reviewed, total_labeled)

        if safe_num(event_mix) is None:
            event_mix = pct(event_reviewed, labeled_reviewed)

        checks = {
            "positive_bins": positive_bins,
            "labeled_patients_reviewed": labeled_reviewed,
            "event_patients_reviewed": event_reviewed,
            "non_event_patients_reviewed": non_event_reviewed,
            "event_capture_proxy_percent": event_capture,
            "non_event_review_proxy_percent": non_event_review,
            "overall_labeled_review_rate_percent": overall_rate,
            "event_mix_among_reviewed_percent": event_mix,
        }

        for key, value in checks.items():
            if safe_num(value) is None:
                failures.append(f"t={t}: missing or nonnumeric {key}")

        normalized.append({
            "threshold": t,
            "positive_bins": int(safe_num(positive_bins) or 0),
            "labeled_patients_reviewed": int(safe_num(labeled_reviewed) or 0),
            "event_patients_reviewed": int(safe_num(event_reviewed) or 0),
            "non_event_patients_reviewed": int(safe_num(non_event_reviewed) or 0),
            "event_capture_proxy_percent": round(safe_num(event_capture), 4) if safe_num(event_capture) is not None else None,
            "non_event_review_proxy_percent": round(safe_num(non_event_review), 4) if safe_num(non_event_review) is not None else None,
            "overall_labeled_review_rate_percent": round(safe_num(overall_rate), 4) if safe_num(overall_rate) is not None else None,
            "event_mix_among_reviewed_percent": round(safe_num(event_mix), 4) if safe_num(event_mix) is not None else None,
        })

    missing_thresholds = sorted(set(EXPECTED_THRESHOLDS) - set(thresholds_seen))
    if missing_thresholds:
        failures.append(f"Missing expected thresholds: {missing_thresholds}")

    normalized.sort(key=lambda x: x["threshold"])

    for prev, cur in zip(normalized, normalized[1:]):
        if cur["positive_bins"] > prev["positive_bins"]:
            failures.append("Positive bins are not monotonic as threshold increases.")

        if cur["overall_labeled_review_rate_percent"] is not None and prev["overall_labeled_review_rate_percent"] is not None:
            if cur["overall_labeled_review_rate_percent"] > prev["overall_labeled_review_rate_percent"]:
                failures.append("Overall labeled review rate is not monotonic as threshold increases.")

    not_allowed = outcome.get("not_allowed_public_claims", [])
    if isinstance(not_allowed, list):
        missing_banned = [c for c in REQUIRED_BANNED_CLAIMS if c not in not_allowed]
        if missing_banned:
            warnings.append(f"Outcome JSON not_allowed_public_claims missing: {missing_banned}")
    else:
        warnings.append("Outcome JSON not_allowed_public_claims is not a list.")

    lock_manifest_files = [
        OUTCOME_JSON,
        OUTCOME_MD,
        SANITY_JSON,
        GENERAL_SCHEMA_JSON,
        Path("tools/hirid_local_only_outcome_label_methodology_lock.py"),
    ]

    manifest = []
    for p in lock_manifest_files:
        if p.exists():
            manifest.append({
                "path": str(p),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            })

    if failures:
        decision = "BLOCK_OUTCOME_LABEL_LOCK_REVIEW_REQUIRED"
    elif warnings:
        decision = "LOCKED_LOCAL_ONLY_OUTCOME_LABEL_METHODOLOGY_PACKET_WITH_WARNINGS"
    else:
        decision = "LOCKED_LOCAL_ONLY_OUTCOME_LABEL_METHODOLOGY_PACKET"

    lock = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "issues": failures,
        "warnings": warnings,
        "dataset": "HiRID v1.1.1",
        "allowed_public_wording_now": PUBLIC_WORDING,
        "privacy_policy": expected_privacy,
        "methodology_boundary": {
            "label_type": "discharge_status outcome proxy",
            "not_clinical_validation": True,
            "not_prospective_validation": True,
            "not_final_hirid_performance": True,
            "does_not_establish_time_window_fpr": True,
            "does_not_establish_time_window_detection": True,
            "does_not_establish_lead_time": True,
        },
        "overall_counts": overall,
        "normalized_threshold_results": normalized,
        "not_allowed_public_claims": REQUIRED_BANNED_CLAIMS,
        "locked_file_manifest": manifest,
        "next_recommended_step": (
            "Create a private cross-dataset comparison packet that clearly separates "
            "MIMIC procedure/event-window metrics from HiRID discharge-status outcome-proxy metrics."
        ),
    }

    LOCK_JSON.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    lines = [
        "# HiRID Outcome-Label Methodology Lock Packet",
        "",
        f"Timestamp UTC: {lock['timestamp_utc']}",
        "",
        f"**Decision:** {decision}",
        "",
        "## Methodology Boundary",
        "",
        "- This is a private/local aggregate outcome-proxy review.",
        "- This uses discharge_status as an outcome proxy.",
        "- This is not clinical validation.",
        "- This is not prospective validation.",
        "- This is not final HiRID performance.",
        "- This does not establish time-window FPR, time-window detection, or lead time.",
        "",
        "## Allowed Public Wording Right Now",
        "",
        f"> {PUBLIC_WORDING}",
        "",
        "## Normalized Threshold Results",
        "",
        "| Threshold | Positive bins | Labeled patients reviewed | Event reviewed | Non-event reviewed | Event capture proxy | Non-event review proxy | Overall labeled review rate | Event mix among reviewed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in normalized:
        lines.append(
            f"| t={r['threshold']} | "
            f"{r['positive_bins']:,} | "
            f"{r['labeled_patients_reviewed']:,} | "
            f"{r['event_patients_reviewed']:,} | "
            f"{r['non_event_patients_reviewed']:,} | "
            f"{r['event_capture_proxy_percent']}% | "
            f"{r['non_event_review_proxy_percent']}% | "
            f"{r['overall_labeled_review_rate_percent']}% | "
            f"{r['event_mix_among_reviewed_percent']}% |"
        )

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
        "## Issues",
        "",
    ])

    if failures:
        for f in failures:
            lines.append(f"- ISSUE: {f}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Privacy Controls",
        "",
    ])

    for k, v in expected_privacy.items():
        lines.append(f"- {k}: {str(v).lower()}")

    lines.extend([
        "",
        "## Locked File Manifest",
        "",
        "| File | SHA-256 | Bytes |",
        "|---|---|---:|",
    ])

    for rec in manifest:
        lines.append(f"| {rec['path']} | {rec['sha256']} | {rec['bytes']} |")

    lines.extend([
        "",
        "## Not Allowed Public Claims",
        "",
    ])

    for claim in REQUIRED_BANNED_CLAIMS:
        lines.append(f"- {claim}")

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
        "Next recommended step after review: create a private cross-dataset comparison packet that clearly separates MIMIC procedure/event-window metrics from HiRID discharge-status outcome-proxy metrics.",
        "",
    ])

    LOCK_MD.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("OUTCOME-LABEL METHODOLOGY LOCK RESULT")
    print("=" * 80)
    print("Decision:", decision)
    print("Issues:", len(failures))
    print("Warnings:", len(warnings))
    print("Lock JSON:", LOCK_JSON)
    print("Lock MD:", LOCK_MD)

    if warnings:
        print("")
        print("WARNINGS:")
        for w in warnings:
            print(" -", w)

    if failures:
        print("")
        print("ISSUES:")
        for f in failures:
            print(" -", f)
        raise SystemExit("STOP: outcome-label methodology lock has issues.")


if __name__ == "__main__":
    main()
