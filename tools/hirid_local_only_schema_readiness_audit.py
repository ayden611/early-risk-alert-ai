#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import tarfile

RAW_DIR = Path("data/validation/local_private/hirid/raw")
OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
AUDIT_DIR = Path("data/validation/local_private/hirid/audit")

OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def safe_decode(b: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="replace")

def read_header_from_tar_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> list[str]:
    f = tf.extractfile(member)
    if f is None:
        return []
    sample = f.read(1024 * 256)
    text = safe_decode(sample)
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if not first_line:
        return []
    try:
        return next(csv.reader([first_line]))
    except Exception:
        return [first_line[:200]]

def summarize_tar(path: Path) -> dict:
    summary = {
        "file": str(path.relative_to(RAW_DIR)),
        "size_bytes": path.stat().st_size,
        "members_total": 0,
        "members_preview": [],
        "csv_like_headers": [],
    }

    try:
        with tarfile.open(path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            summary["members_total"] = len(members)
            summary["members_preview"] = [m.name for m in members[:25]]

            for m in members:
                lower = m.name.lower()
                if lower.endswith(".csv") or lower.endswith(".txt"):
                    header = read_header_from_tar_member(tf, m)
                    summary["csv_like_headers"].append({
                        "member": m.name,
                        "size_bytes": m.size,
                        "header_preview": header[:40],
                    })
                    if len(summary["csv_like_headers"]) >= 12:
                        break

    except Exception as e:
        summary["error"] = repr(e)

    return summary

def main() -> None:
    tarballs = sorted(RAW_DIR.rglob("*.tar.gz"))
    txt_pdf = sorted([*RAW_DIR.rglob("*.txt"), *RAW_DIR.rglob("*.pdf")])

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "schema_readiness_only_no_extraction_no_analysis",
        "raw_dir": str(RAW_DIR),
        "policy": {
            "raw_files_remain_local_only": True,
            "row_level_outputs_written": False,
            "public_claims_updated": False,
            "current_public_wording": "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        },
        "tarballs_found": [str(p.relative_to(RAW_DIR)) for p in tarballs],
        "support_files_found": [str(p.relative_to(RAW_DIR)) for p in txt_pdf],
        "tarball_summaries": [],
        "next_required_before_validation": [
            "Confirm which extracted/streamed table contains patient id, time, vital-sign variable, and value fields.",
            "Map HiRID vital variables to ERA review inputs: SpO2, HR, RR, systolic BP, diastolic BP, temperature.",
            "Run aggregate-only local validation after mapping is confirmed.",
            "Publish only aggregate summaries after local review.",
        ],
    }

    for p in tarballs:
        print(f"Inspecting tarball metadata/header only: {p}")
        result["tarball_summaries"].append(summarize_tar(p))

    out = OUT_DIR / "hirid_schema_readiness_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = OUT_DIR / "hirid_schema_readiness_audit.md"
    lines = [
        "# HiRID Local-Only Schema Readiness Audit",
        "",
        f"Timestamp UTC: {result['timestamp_utc']}",
        "",
        "Status: schema/readiness only. No extraction, no row-level output, no public validation claim.",
        "",
        "Current allowed public wording:",
        "",
        "> HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        "",
        "## Tarballs Found",
    ]

    for t in result["tarballs_found"]:
        lines.append(f"- {t}")

    lines.extend(["", "## Header Previews"])

    for s in result["tarball_summaries"]:
        lines.append(f"### {s['file']}")
        lines.append(f"- Members: {s.get('members_total')}")
        for h in s.get("csv_like_headers", [])[:8]:
            cols = ", ".join(h.get("header_preview", [])[:16])
            lines.append(f"- `{h['member']}`: {cols}")

    lines.extend([
        "",
        "## Next Step",
        "Use this schema audit to confirm the correct HiRID table and field mapping before running aggregate validation.",
        "Do not publish HiRID performance metrics until local-only aggregate validation is completed and reviewed.",
    ])

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print(f"Wrote local JSON audit: {out}")
    print(f"Wrote local markdown audit: {md}")
    print("")
    print("DONE — schema readiness audit complete.")
    print("No raw data was extracted or published.")

if __name__ == "__main__":
    main()
