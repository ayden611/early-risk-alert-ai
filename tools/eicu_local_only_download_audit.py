#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = [
    "patient.csv.gz",
    "vitalPeriodic.csv.gz",
    "vitalAperiodic.csv.gz",
    "diagnosis.csv.gz",
    "apachePatientResult.csv.gz",
]

RAW_DIR = Path("data/validation/local_private/eicu/raw")
MANIFEST_DIR = Path("data/validation/local_private/eicu/manifests")
LOG_DIR = Path("data/validation/local_private/eicu/logs")


def human_size(n: int | None) -> str:
    if n is None:
        return "missing"
    size = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def read_header(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            sample = []
            for _, row in zip(range(3), reader):
                sample.append(row[:8])
        return {
            "ok": True,
            "columns": header,
            "column_count": len(header),
            "sample_rows_read": len(sample),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "columns": [],
            "column_count": 0,
            "sample_rows_read": 0,
        }


def count_rows(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        for _ in f:
            count += 1
    return max(count - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads-dir", default=str(Path.home() / "Downloads"))
    ap.add_argument("--copy-complete", action="store_true")
    ap.add_argument("--full-row-count", action="store_true")
    args = ap.parse_args()

    downloads = Path(args.downloads_dir)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    all_present = True
    any_downloading = False
    all_headers_ok = True

    for name in REQUIRED:
        src = downloads / name
        partials = list(downloads.glob(name + "*.crdownload")) + list(downloads.glob(name.replace(".gz", "") + "*.crdownload"))

        status = "missing"
        size = None
        header_info = {
            "ok": False,
            "error": "not checked",
            "columns": [],
            "column_count": 0,
            "sample_rows_read": 0,
        }

        if partials:
            status = "downloading"
            any_downloading = True
            all_present = False
            if src.exists():
                size = src.stat().st_size
        elif src.exists():
            status = "present"
            size = src.stat().st_size
            header_info = read_header(src)
            if not header_info["ok"]:
                all_headers_ok = False
        else:
            all_present = False

        row_count = None
        if args.full_row_count and status == "present" and header_info["ok"]:
            row_count = count_rows(src)

        results.append({
            "file": name,
            "status": status,
            "source_path": str(src) if src.exists() else None,
            "size_bytes": size,
            "size_human": human_size(size),
            "header_ok": header_info["ok"],
            "column_count": header_info["column_count"],
            "columns": header_info["columns"],
            "sample_rows_read": header_info["sample_rows_read"],
            "row_count": row_count,
        })

    can_copy = all_present and not any_downloading and all_headers_ok

    copied = []
    if args.copy_complete:
        if not can_copy:
            print("COPY SKIPPED: all files are not complete/header-readable yet.")
        else:
            for item in results:
                src = Path(item["source_path"])
                dst = RAW_DIR / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst))
            print("COPIED completed eICU files into local-only raw folder.")

    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local-only eICU download integrity and schema audit before ERA extraction.",
        "downloads_dir": str(downloads),
        "required_files": REQUIRED,
        "all_present": all_present,
        "any_downloading": any_downloading,
        "all_headers_ok": all_headers_ok,
        "ready_for_extraction": can_copy,
        "copied_to_local_private_raw": copied,
        "files": results,
        "public_policy": "Local-only audit. Raw eICU files and row-level outputs must not be committed or published.",
    }

    out = MANIFEST_DIR / "eicu_download_audit_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("")
    print("eICU DOWNLOAD AUDIT")
    print("===================")
    for item in results:
        print(f"{item['file']}: {item['status']} | {item['size_human']} | header_ok={item['header_ok']} | columns={item['column_count']}")

    print("")
    print("all_present:", all_present)
    print("any_downloading:", any_downloading)
    print("all_headers_ok:", all_headers_ok)
    print("ready_for_extraction:", can_copy)
    print("manifest:", out)

    if any_downloading:
        print("")
        print("WAIT: one or more files are still downloading. Do not extract yet.")
    elif can_copy:
        print("")
        print("READY: all files appear complete/header-readable. Next step will be copy/extraction.")
    else:
        print("")
        print("NOT READY: missing file or header issue detected.")


if __name__ == "__main__":
    main()
