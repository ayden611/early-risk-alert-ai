#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

LOCAL_RAW = Path("data/validation/local_private/hirid/raw")
LOCAL_MANIFESTS = Path("data/validation/local_private/hirid/manifests")
PUBLIC_SUMMARY = Path("data/validation/hirid_download_readiness_summary.json")

DOWNLOAD_SUFFIXES = (
    ".csv", ".csv.gz", ".tsv", ".tsv.gz",
    ".parquet", ".parquet.gzip", ".zip", ".tar.gz", ".tgz",
    ".json", ".txt", ".pdf"
)

INCOMPLETE_SUFFIXES = (
    ".crdownload", ".download", ".part", ".tmp"
)

KEY_HINTS = (
    "hirid",
    "observation",
    "observations",
    "pharma",
    "general",
    "variable_reference",
    "ordinal",
    "imputed",
    "merged",
    "index",
    "schemata"
)


def size_mb(p: Path) -> float:
    return round(p.stat().st_size / (1024 * 1024), 3)


def is_incomplete(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(s) for s in INCOMPLETE_SUFFIXES)


def stable_size(p: Path, pause: float = 1.5) -> bool:
    try:
        s1 = p.stat().st_size
        time.sleep(pause)
        s2 = p.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False


def looks_like_hirid_file(p: Path) -> bool:
    name = p.name.lower()
    if is_incomplete(p):
        return False
    if not name.endswith(DOWNLOAD_SUFFIXES):
        return False
    return any(h in name for h in KEY_HINTS)


def read_header(path: Path) -> dict:
    name = path.name.lower()
    result = {"header_readable": False, "columns": None, "first_columns": []}

    try:
        if name.endswith(".csv.gz"):
            f = gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
        elif name.endswith(".csv"):
            f = open(path, "r", encoding="utf-8", errors="ignore", newline="")
        elif name.endswith(".tsv.gz"):
            f = gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
        elif name.endswith(".tsv"):
            f = open(path, "r", encoding="utf-8", errors="ignore", newline="")
        else:
            return result

        with f:
            sample = f.readline()
            if not sample:
                return result
            delimiter = "\t" if name.endswith(".tsv") or name.endswith(".tsv.gz") else ","
            row = next(csv.reader([sample], delimiter=delimiter))
            result["header_readable"] = True
            result["columns"] = len(row)
            result["first_columns"] = row[:12]
            return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def copy_completed_downloads(download_dir: Path, local_raw: Path) -> list[dict]:
    copied = []
    if not download_dir.exists():
        return copied

    local_raw.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in download_dir.iterdir() if p.is_file() and looks_like_hirid_file(p)]
    for src in candidates:
        if not stable_size(src):
            continue
        dst = local_raw / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            copied.append({"file": src.name, "status": "already_present", "size_mb": size_mb(dst)})
            continue
        shutil.copy2(src, dst)
        copied.append({"file": src.name, "status": "copied", "size_mb": size_mb(dst)})
    return copied


def audit_local_raw(local_raw: Path) -> list[dict]:
    local_raw.mkdir(parents=True, exist_ok=True)

    rows = []
    files = sorted([p for p in local_raw.iterdir() if p.is_file()])
    for p in files:
        row = {
            "file": p.name,
            "size_mb": size_mb(p),
            "incomplete": is_incomplete(p),
            "kind": "unknown",
        }

        n = p.name.lower()
        if "general" in n:
            row["kind"] = "general"
        elif "observation" in n or "observations" in n:
            row["kind"] = "observations"
        elif "pharma" in n:
            row["kind"] = "pharma"
        elif "variable_reference" in n:
            row["kind"] = "variable_reference"
        elif "ordinal" in n:
            row["kind"] = "ordinal_reference"
        elif "imputed" in n:
            row["kind"] = "preprocessed_imputed"
        elif "merged" in n:
            row["kind"] = "preprocessed_merged"
        elif "index" in n:
            row["kind"] = "index"
        elif "schemata" in n:
            row["kind"] = "schema"

        row.update(read_header(p))
        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default=str(Path.home() / "Downloads"))
    ap.add_argument("--copy-complete", action="store_true")
    args = ap.parse_args()

    downloads = Path(args.downloads)
    LOCAL_RAW.mkdir(parents=True, exist_ok=True)
    LOCAL_MANIFESTS.mkdir(parents=True, exist_ok=True)

    copied = []
    if args.copy_complete:
        copied = copy_completed_downloads(downloads, LOCAL_RAW)

    rows = audit_local_raw(LOCAL_RAW)

    total_size = round(sum(r["size_mb"] for r in rows), 3)
    kinds = sorted(set(r["kind"] for r in rows if not r["incomplete"]))

    has_reference = any(r["kind"] in {"variable_reference", "ordinal_reference", "schema"} for r in rows)
    has_core_data = any(r["kind"] in {"observations", "preprocessed_imputed", "preprocessed_merged", "general"} for r in rows)
    ready_for_extractor = bool(rows) and has_reference and has_core_data and not any(r["incomplete"] for r in rows)

    private_manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_source_checked": str(downloads),
        "local_raw_dir": str(LOCAL_RAW),
        "copied": copied,
        "files": rows,
        "total_files": len(rows),
        "total_size_mb": total_size,
        "ready_for_extractor": ready_for_extractor,
        "notes": [
            "This private manifest may contain local paths and should remain local-only.",
            "Raw HiRID files and row-level outputs must not be committed."
        ]
    }

    private_path = LOCAL_MANIFESTS / "hirid_download_audit_private_manifest.json"
    private_path.write_text(json.dumps(private_manifest, indent=2), encoding="utf-8")

    public_summary = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "HiRID",
        "purpose": "Third-dataset retrospective validation readiness audit",
        "public_safety": "Aggregate file-readiness summary only. Raw HiRID files remain local-only.",
        "total_files_detected": len(rows),
        "total_size_mb": total_size,
        "file_kinds_detected": kinds,
        "ready_for_extractor": ready_for_extractor,
        "next_step": "Download/access complete files, then build HiRID-to-ERA cohort extractor.",
        "claim_boundary": "Do not claim three-dataset validation until HiRID extraction, scoring, harmonized labeling, and aggregate review are complete."
    }

    PUBLIC_SUMMARY.write_text(json.dumps(public_summary, indent=2), encoding="utf-8")

    print("")
    print("HiRID DOWNLOAD READINESS AUDIT")
    print("==============================")
    print(f"Files detected locally: {len(rows)}")
    print(f"Total local size: {total_size} MB")
    print(f"Kinds detected: {', '.join(kinds) if kinds else 'none yet'}")
    print(f"Ready for extractor: {ready_for_extractor}")
    print("")
    if copied:
        print("Copied completed files:")
        for item in copied:
            print(f" - {item['file']} | {item['status']} | {item['size_mb']} MB")
        print("")
    print("Local-only raw folder:")
    print(f" - {LOCAL_RAW}")
    print("Private manifest:")
    print(f" - {private_path}")
    print("Public aggregate summary:")
    print(f" - {PUBLIC_SUMMARY}")
    print("")
    if not ready_for_extractor:
        print("NEXT: Finish PhysioNet HiRID access/downloads, then rerun:")
        print("python3 tools/hirid_local_only_download_audit.py --copy-complete")


if __name__ == "__main__":
    main()
