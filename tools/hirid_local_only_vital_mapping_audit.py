#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import io
import json
import re
import tarfile

RAW_DIR = Path("data/validation/local_private/hirid/raw")
OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "heart_rate": [
        "heart rate", "hr", "pulse", "cardiac frequency"
    ],
    "spo2": [
        "spo2", "oxygen saturation", "o2 saturation", "peripheral oxygen saturation",
        "sao2", "arterial oxygen saturation"
    ],
    "respiratory_rate": [
        "respiratory rate", "resp rate", "respiration rate", "breathing rate", "rr"
    ],
    "systolic_bp": [
        "systolic", "systolic blood pressure", "systolic arterial pressure", "sbp"
    ],
    "diastolic_bp": [
        "diastolic", "diastolic blood pressure", "diastolic arterial pressure", "dbp"
    ],
    "temperature": [
        "temperature", "body temperature", "temp"
    ],
}

NEGATIVE_HINTS = [
    "urine", "drug", "dose", "fluid", "calculated", "comment",
    "height", "weight", "score", "apache", "glasgow", "gcs"
]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def csv_rows_from_tar_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> list[dict]:
    f = tf.extractfile(member)
    if f is None:
        return []
    raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)

def find_reference_csvs() -> list[tuple[Path, str, list[dict]]]:
    found = []
    for tar_path in sorted(RAW_DIR.rglob("*.tar.gz")):
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    lower = m.name.lower()
                    if lower.endswith(".csv") and (
                        "variable_reference" in lower
                        or "ordinal_vars_ref" in lower
                        or "reference" in lower
                    ):
                        rows = csv_rows_from_tar_member(tf, m)
                        found.append((tar_path, m.name, rows))
        except Exception as e:
            print(f"WARNING: could not inspect {tar_path}: {e}")
    return found

def candidate_score(row: dict, target_words: list[str]) -> int:
    combined = " | ".join(norm(v) for v in row.values())
    score = 0
    for w in target_words:
        if norm(w) in combined:
            score += 10
    for bad in NEGATIVE_HINTS:
        if bad in combined:
            score -= 4
    return score

def compact_row(row: dict) -> dict:
    keep = {}
    for k, v in row.items():
        if k is None:
            continue
        kk = str(k).strip()
        vv = str(v).strip()
        if vv:
            keep[kk] = vv[:300]
    return keep

def main() -> None:
    refs = find_reference_csvs()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_only_reference_mapping_no_patient_rows",
        "raw_dir": str(RAW_DIR),
        "policy": {
            "patient_rows_read": False,
            "row_level_outputs_written": False,
            "public_claims_updated": False,
            "current_public_wording": "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        },
        "reference_files_found": [],
        "mapping_candidates": {},
        "recommended_next_step": "Review candidates, select variable IDs for HR, SpO2, RR, SBP, DBP, and temperature, then run aggregate-only validation.",
    }

    for tar_path, member, rows in refs:
        result["reference_files_found"].append({
            "tarball": str(tar_path.relative_to(RAW_DIR)),
            "member": member,
            "rows": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
        })

    for target, words in TARGETS.items():
        scored = []
        for tar_path, member, rows in refs:
            for row in rows:
                s = candidate_score(row, words)
                if s > 0:
                    scored.append({
                        "score": s,
                        "source_tarball": str(tar_path.relative_to(RAW_DIR)),
                        "member": member,
                        "row": compact_row(row),
                    })
        scored.sort(key=lambda x: x["score"], reverse=True)
        result["mapping_candidates"][target] = scored[:25]

    out_json = OUT_DIR / "hirid_vital_variable_mapping_candidates.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    out_md = OUT_DIR / "hirid_vital_variable_mapping_candidates.md"
    lines = [
        "# HiRID Vital Variable Mapping Candidates",
        "",
        f"Timestamp UTC: {result['timestamp_utc']}",
        "",
        "Status: local-only reference metadata mapping. No patient rows were analyzed.",
        "",
        "Allowed public wording remains:",
        "",
        "> HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        "",
        "## Reference Files Found",
    ]

    for rf in result["reference_files_found"]:
        lines.append(f"- `{rf['tarball']} :: {rf['member']}` — rows: {rf['rows']} — columns: {', '.join(rf['columns'])}")

    for target, candidates in result["mapping_candidates"].items():
        lines.append("")
        lines.append(f"## Candidate mappings for `{target}`")
        if not candidates:
            lines.append("- No candidates found.")
            continue

        for c in candidates[:12]:
            row = c["row"]
            identifier = (
                row.get("Variable id")
                or row.get("variableid")
                or row.get("ID")
                or row.get("id")
                or row.get("code")
                or "unknown-id"
            )
            name = (
                row.get("Variable Name")
                or row.get("Name")
                or row.get("name")
                or row.get("stringvalue")
                or "unknown-name"
            )
            unit = row.get("Unit") or row.get("unit") or ""
            source = row.get("Source Table") or row.get("Source") or ""
            lines.append(f"- score {c['score']} | id `{identifier}` | name `{name}` | unit `{unit}` | source `{source}` | member `{c['member']}`")

    lines.extend([
        "",
        "## Next step",
        "Use the candidate IDs above to lock a conservative mapping for ERA inputs.",
        "Do not publish HiRID validation metrics until aggregate-only validation is completed and reviewed.",
    ])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("REFERENCE FILES FOUND")
    print("=" * 80)
    for rf in result["reference_files_found"]:
        print(f"- {rf['tarball']} :: {rf['member']} | rows={rf['rows']} | columns={rf['columns']}")

    print("")
    print("TOP VARIABLE MAPPING CANDIDATES")
    print("=" * 80)

    for target, candidates in result["mapping_candidates"].items():
        print(f"\n{target.upper()}")
        print("-" * 80)
        if not candidates:
            print("No candidates found.")
            continue

        for c in candidates[:10]:
            row = c["row"]
            identifier = (
                row.get("Variable id")
                or row.get("variableid")
                or row.get("ID")
                or row.get("id")
                or row.get("code")
                or "unknown-id"
            )
            name = (
                row.get("Variable Name")
                or row.get("Name")
                or row.get("name")
                or row.get("stringvalue")
                or "unknown-name"
            )
            unit = row.get("Unit") or row.get("unit") or ""
            source = row.get("Source Table") or row.get("Source") or ""
            print(f"score={c['score']} | id={identifier} | name={name} | unit={unit} | source={source} | member={c['member']}")

    print("")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")
    print("")
    print("DONE — local-only vital mapping audit complete.")
    print("No patient rows were analyzed.")

if __name__ == "__main__":
    main()
