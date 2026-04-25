#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def find_col(headers, candidates):
    lower = {str(h).lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def bucket_for_case(case_value):
    digest = hashlib.sha256(str(case_value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--out-csv", default="data/validation/local_private/subcohorts/mimic_subcohort_b_hash_33_65.csv")
    ap.add_argument("--manifest-json", default="data/validation/local_private/subcohorts/mimic_subcohort_b_manifest.json")
    ap.add_argument("--bucket-start", type=int, default=33)
    ap.add_argument("--bucket-end", type=int, default=65)
    args = ap.parse_args()

    src = Path(args.source_csv)
    out_csv = Path(args.out_csv)
    manifest_path = Path(args.manifest_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise SystemExit("ERROR: source CSV has no rows.")

    case_col = find_col(headers, ["patient_id", "case_id", "subject_id"])
    event_col = find_col(headers, ["clinical_event", "event", "event_flag", "has_event", "clinical_event_flag"])

    if not case_col:
        raise SystemExit(f"ERROR: could not find case identifier column. Found: {headers}")

    selected = []
    cases = set()
    all_cases = set()
    event_rows = 0

    for row in rows:
        case_value = str(row.get(case_col, "")).strip()
        if not case_value:
            continue

        all_cases.add(case_value)
        bucket = bucket_for_case(case_value)

        if args.bucket_start <= bucket <= args.bucket_end:
            selected.append(row)
            cases.add(case_value)

            if event_col:
                v = str(row.get(event_col, "")).strip().lower()
                if v not in {"", "0", "0.0", "false", "no", "none", "nan", "null"}:
                    event_rows += 1

    if not selected:
        raise SystemExit("ERROR: selected subcohort has no rows.")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(selected)

    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local-only deterministic patient/case-level MIMIC subcohort for DUA-safe aggregate validation.",
        "subset_name": "subcohort_b_hash_33_65",
        "bucket_start": args.bucket_start,
        "bucket_end": args.bucket_end,
        "source_rows": len(rows),
        "source_case_count": len(all_cases),
        "selected_rows": len(selected),
        "selected_case_count": len(cases),
        "selected_event_labeled_rows": event_rows,
        "local_subset_file": str(out_csv),
        "public_policy": "Local-only row-level subset. Do not commit or publish this CSV."
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("WROTE local subcohort CSV:", out_csv)
    print("WROTE local manifest:", manifest_path)
    print("Selected rows:", len(selected))
    print("Selected cases:", len(cases))
    print("Selected event-labeled rows:", event_rows)


if __name__ == "__main__":
    main()
