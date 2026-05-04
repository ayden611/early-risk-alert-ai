#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(".").resolve()
RAW = ROOT / "data" / "validation" / "local_private" / "hirid" / "raw"
AUDIT = ROOT / "data" / "validation" / "local_private" / "hirid" / "audit"
PUBLIC_SUMMARY = ROOT / "data" / "validation" / "hirid_download_audit_public_summary.json"
PUBLIC_MD = ROOT / "docs" / "validation" / "hirid_download_readiness_summary.md"

RESTRICTED_SUFFIXES = {
    ".csv",
    ".gz",
    ".zip",
    ".parquet",
    ".h5",
    ".hdf5",
    ".feather",
    ".pkl",
    ".pickle",
    ".tar",
}

EXPECTED_HINTS = [
    "observation",
    "pharma",
    "general",
    "apache",
    "patient",
    "variable",
    "hirid",
]


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def is_restricted_file(path: Path) -> bool:
    name = path.name.lower()
    suffixes = "".join(path.suffixes).lower()
    if any(name.endswith(x) for x in [".csv.gz", ".tar.gz"]):
        return True
    if path.suffix.lower() in RESTRICTED_SUFFIXES:
        return True
    if suffixes.endswith(".csv.gz") or suffixes.endswith(".tar.gz"):
        return True
    return False


def scan_raw() -> Dict:
    RAW.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    files: List[Path] = [p for p in RAW.rglob("*") if p.is_file()]
    restricted = [p for p in files if is_restricted_file(p)]

    total_bytes = sum(p.stat().st_size for p in files)
    restricted_bytes = sum(p.stat().st_size for p in restricted)

    detected_hints = sorted(
        {
            hint
            for hint in EXPECTED_HINTS
            for p in files
            if hint in p.name.lower()
        }
    )

    private_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_folder": str(RAW),
        "file_count": len(files),
        "restricted_file_count": len(restricted),
        "total_bytes": total_bytes,
        "restricted_bytes": restricted_bytes,
        "detected_name_hints": detected_hints,
        "files_private_local_only": [
            {
                "relative_path": str(p.relative_to(RAW)),
                "size_bytes": p.stat().st_size,
                "sha256_short": sha256_short(p),
            }
            for p in files
        ],
        "raw_data_policy": "Private local-only audit manifest. Do not commit this file."
    }

    private_path = AUDIT / "hirid_private_download_manifest.json"
    private_path.write_text(json.dumps(private_manifest, indent=2), encoding="utf-8")

    public_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "HiRID v1.1.1",
        "status": "local_download_audit_completed" if files else "awaiting_local_download",
        "local_raw_file_count": len(files),
        "local_restricted_file_count": len(restricted),
        "local_total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "detected_name_hints": detected_hints,
        "public_safety": "This public summary contains only aggregate file counts and size totals. It does not include filenames, paths, row-level data, timestamps, identifiers, or patient-level outputs.",
        "validation_status": "HiRID validation not yet completed."
    }

    PUBLIC_SUMMARY.write_text(json.dumps(public_summary, indent=2), encoding="utf-8")

    md = f"""# HiRID Download Readiness Summary

## Status

**{public_summary["status"]}**

HiRID v1.1.1 access has been approved. This summary records only aggregate local download readiness information.

## Local Audit Summary

| Field | Value |
|---|---:|
| Local raw file count | {public_summary["local_raw_file_count"]} |
| Local restricted file count | {public_summary["local_restricted_file_count"]} |
| Local total size MB | {public_summary["local_total_size_mb"]} |

## Safety Statement

This public summary contains only aggregate file counts and size totals. It does not include filenames, raw rows, timestamps, identifiers, patient-level outputs, or restricted data.

## Validation Status

HiRID validation has **not** been completed yet. Current public validation evidence remains MIMIC-IV + eICU retrospective aggregate evidence only.
"""
    PUBLIC_MD.write_text(md, encoding="utf-8")

    return {
        "private_manifest": str(private_path),
        "public_summary": str(PUBLIC_SUMMARY),
        "public_markdown": str(PUBLIC_MD),
        "file_count": len(files),
        "restricted_file_count": len(restricted),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-complete", action="store_true", help="Run after HiRID files are copied into the local raw folder.")
    args = parser.parse_args()

    result = scan_raw()

    print("HIRID LOCAL-ONLY DOWNLOAD AUDIT")
    print(f"Raw folder: {RAW}")
    print(f"Local file count: {result['file_count']}")
    print(f"Restricted file count: {result['restricted_file_count']}")
    print(f"Total size MB: {result['total_mb']}")
    print(f"Private manifest: {result['private_manifest']}")
    print(f"Public aggregate summary: {result['public_summary']}")
    print(f"Public markdown summary: {result['public_markdown']}")

    if result["file_count"] == 0:
        print("")
        print("NEXT: Download approved HiRID files through PhysioNet, then copy them into:")
        print(f"  {RAW}")
        print("")
        print("After copying files, rerun:")
        print("  python3 tools/hirid_local_only_download_audit.py --copy-complete")
    else:
        print("")
        print("DOWNLOAD AUDIT COMPLETE.")
        print("Do not commit the private manifest or raw files.")
        print("Next step after review: run HiRID harmonized retrospective aggregate evaluation.")


if __name__ == "__main__":
    main()
