#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "validation"
PRIVATE_DIR = DATA_DIR / "local_private"

PUBLIC_JSON_FILES = [
    DATA_DIR / "mimic_validation_milestone_2026_04.json",
    DATA_DIR / "latest_era_validation_summary.json",
    DATA_DIR / "validation_run_registry.json",
]

PUBLIC_EXAMPLES_JSON = DATA_DIR / "public_representative_examples.json"

SENSITIVE_KEYS_DROP = {
    "patient_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "event_time",
    "first_alert_time",
    "timestamp",
    "charttime",
    "threshold_crossed_at",
}

LOCAL_ONLY_KEYS = {
    "source_file",
    "enriched_csv",
    "summary_file",
    "local_file",
    "raw_file",
}

CASE_MAP = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def backup_private(path: Path):
    if not path.exists():
        return

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dst = PRIVATE_DIR / f"{path.stem}_private_backup_{stamp}{path.suffix}"
    shutil.copy2(path, dst)


def case_id_for(value, index=None):
    raw = str(value or "").strip()
    if not raw:
        raw = f"unknown-{index if index is not None else len(CASE_MAP)+1}"

    if raw not in CASE_MAP:
        CASE_MAP[raw] = f"Case-{len(CASE_MAP)+1:03d}"

    return CASE_MAP[raw]


def clean_driver(v):
    if not v:
        return "Review context"
    if str(v).strip().lower() == "no dominant driver":
        return "Composite multi-signal pattern"
    return str(v)


def round_hours(v):
    try:
        return round(float(v), 1)
    except Exception:
        return v


def sanitize_example(ex: dict, index: int):
    raw_id = ex.get("patient_id") or ex.get("case_id") or f"example-{index}"
    return {
        "case_id": case_id_for(raw_id, index),
        "lead_hours": round_hours(ex.get("lead_hours")),
        "priority_tier": ex.get("priority_tier") or "Review",
        "primary_driver": clean_driver(ex.get("primary_driver")),
        "trend_direction": ex.get("trend_direction") or "Review context",
        "score": ex.get("score"),
        "queue_rank": ex.get("queue_rank"),
    }


def looks_like_example_list(value):
    if not isinstance(value, list) or not value:
        return False

    for item in value[:5]:
        if isinstance(item, dict) and (
            "patient_id" in item
            or "case_id" in item
            or "event_time" in item
            or "first_alert_time" in item
            or "lead_hours" in item
        ):
            return True

    return False


def sanitize_obj(obj):
    if isinstance(obj, list):
        if looks_like_example_list(obj):
            return [sanitize_example(x, i + 1) for i, x in enumerate(obj) if isinstance(x, dict)]
        return [sanitize_obj(x) for x in obj]

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in SENSITIVE_KEYS_DROP:
                if k == "patient_id":
                    out["case_id"] = case_id_for(v)
                continue

            if k in LOCAL_ONLY_KEYS:
                out[k] = "local-only-not-public"
                continue

            if k == "detected_examples" and isinstance(v, list):
                out[k] = [sanitize_example(x, i + 1) for i, x in enumerate(v) if isinstance(x, dict)]
                continue

            if k == "patient_examples" and isinstance(v, list):
                out[k] = [sanitize_example(x, i + 1) for i, x in enumerate(v) if isinstance(x, dict)]
                continue

            out[k] = sanitize_obj(v)

        return out

    if isinstance(obj, str):
        s = obj
        s = s.replace("no dominant driver", "Composite multi-signal pattern")
        s = s.replace("No dominant driver", "Composite multi-signal pattern")

        # Do not publish local absolute paths.
        if "/Users/" in s and ("mimic" in s.lower() or ".csv" in s.lower()):
            return "local-only-not-public"

        return s

    return obj


def find_public_examples(data):
    paths = [
        ["selected_threshold_metrics", "detected_examples"],
        ["computed_validation_metrics", "lead_time_before_event", "detected_examples"],
        ["threshold_results", "6.0", "detected_examples"],
        ["threshold_results", "6", "detected_examples"],
        ["threshold_results", "4.0", "detected_examples"],
    ]

    for path in paths:
        cur = data
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break

        if ok and isinstance(cur, list) and cur:
            return [sanitize_example(x, i + 1) for i, x in enumerate(cur) if isinstance(x, dict)]

    return []


def add_compliance_metadata(data):
    if not isinstance(data, dict):
        return data

    data["mimic_dua_public_safety"] = {
        "status": "Applied",
        "applied_at_utc": utc_now(),
        "public_policy": "Only aggregate metrics, sanitized case examples, validation methodology, and code are public-facing. Row-level MIMIC-derived exports remain local-only.",
        "removed_from_public_outputs": [
            "raw MIMIC files",
            "row-level enriched CSV downloads",
            "patient IDs",
            "subject IDs",
            "hospital admission/stay IDs",
            "exact patient-linked timestamps",
            "source file paths",
            "patient-level rows"
        ],
        "safe_public_outputs": [
            "aggregate validation metrics",
            "threshold comparison tables",
            "alert reduction percentage",
            "false-positive rate",
            "patient-level detection percentage",
            "median lead-time summary",
            "sanitized case examples",
            "validation methodology",
            "code used to produce results"
        ],
        "notice": "Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation."
    }
    return data


def sanitize_public_files(quiet=False):
    all_public_examples = []

    for path in PUBLIC_JSON_FILES:
        if not path.exists():
            continue

        raw = load_json(path)
        if raw is None:
            continue

        backup_private(path)

        examples = find_public_examples(raw)
        if examples and not all_public_examples:
            all_public_examples = examples

        sanitized = sanitize_obj(raw)
        sanitized = add_compliance_metadata(sanitized)
        save_json(path, sanitized)

        if not quiet:
            print("SANITIZED:", path)

    if not all_public_examples:
        # Try again from sanitized milestone.
        milestone = load_json(DATA_DIR / "mimic_validation_milestone_2026_04.json") or {}
        all_public_examples = find_public_examples(milestone)

    public_payload = {
        "ok": True,
        "generated_at_utc": utc_now(),
        "policy": "Sanitized public representative examples only. No MIMIC patient IDs, subject IDs, hospital admission IDs, stay IDs, or exact patient-linked timestamps.",
        "examples": all_public_examples[:20],
    }

    save_json(PUBLIC_EXAMPLES_JSON, public_payload)

    docs = ROOT / "docs" / "validation"
    docs.mkdir(parents=True, exist_ok=True)

    (docs / "mimic_dua_public_safety.md").write_text("""# MIMIC DUA Public Safety Policy

## Public-Facing Rule

Only aggregate metrics, sanitized case examples, validation methodology, and code should be public-facing.

## Safe to Publish

- Aggregate validation metrics
- Code used to generate results
- High-level threshold tables
- Alert reduction percentage
- False-positive rate
- Patient detection percentage
- Median lead-time summary
- Pilot-safe evidence packet
- Validation methodology
- Sanitized case examples using Case-001 style labels

## Local-Only / Not Public

- Raw MIMIC CSV files
- Row-level enriched CSV files
- Patient-level rows
- MIMIC patient IDs
- Subject IDs
- Hospital admission IDs
- Stay IDs
- Exact timestamps tied to cases/patients
- Representative examples with real MIMIC IDs or exact timestamps
- Any file that could be treated as derived restricted data

## Public Example Format

- Case ID, such as Case-001
- Lead time only
- Priority tier
- Primary driver
- Trend direction
- Score
- Queue rank

No MIMIC IDs. No exact timestamps. No row-level export.

## Notice

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
""", encoding="utf-8")

    if not quiet:
        print("WROTE:", PUBLIC_EXAMPLES_JSON)
        print("WROTE:", docs / "mimic_dua_public_safety.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    sanitize_public_files(quiet=args.quiet)


if __name__ == "__main__":
    main()
