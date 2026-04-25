#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def find_col(headers, candidates):
    lower = {str(h).lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def stable_fold(case_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out-dir", default="data/validation/local_private/cohort_folds")
    ap.add_argument("--summary-json", default="data/validation/local_private/cohort_fold_manifest.json")
    args = ap.parse_args()

    source = Path(args.source_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise SystemExit("ERROR: source CSV has no rows.")

    case_col = find_col(headers, ["patient_id", "case_id", "subject_id"])
    if not case_col:
        raise SystemExit(f"ERROR: no patient/case identifier column found. Found: {headers}")

    fold_rows = {i: [] for i in range(args.folds)}
    fold_cases = {i: set() for i in range(args.folds)}

    for row in rows:
        case_id = str(row.get(case_col, "")).strip()
        if not case_id:
            continue
        fold = stable_fold(case_id, args.folds)
        fold_rows[fold].append(row)
        fold_cases[fold].add(case_id)

    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local-only patient-level cohort folds for DUA-safe aggregate validation.",
        "folds": args.folds,
        "source_rows": len(rows),
        "source_cases": len(set(str(r.get(case_col, '')).strip() for r in rows if str(r.get(case_col, '')).strip())),
        "public_policy": "Manifest remains local-only because fold files are row-level derived data.",
        "folds_detail": []
    }

    for fold in range(args.folds):
        out_path = out_dir / f"cohort_fold_{fold+1}_of_{args.folds}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(fold_rows[fold])

        manifest["folds_detail"].append({
            "fold": fold + 1,
            "rows": len(fold_rows[fold]),
            "cases": len(fold_cases[fold]),
            "local_file": str(out_path)
        })

        print(f"WROTE fold {fold+1}: rows={len(fold_rows[fold])}, cases={len(fold_cases[fold])}, file={out_path}")

    Path(args.summary_json).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE local manifest:", args.summary_json)


if __name__ == "__main__":
    main()
