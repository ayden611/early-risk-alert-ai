#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import io
import json
import re
import tarfile
from collections import Counter, defaultdict

RAW_DIR = Path("data/validation/local_private/hirid/raw")
OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
AUDIT_DIR = Path("data/validation/local_private/hirid/audit")

OUT_JSON = AUDIT_DIR / "hirid_local_only_general_table_schema_audit.json"
OUT_MD = AUDIT_DIR / "hirid_local_only_general_table_schema_audit.md"

PUBLIC_WORDING = "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."

BLOCKED_RAW_PATTERNS = re.compile(
    r"(patient|subject|stay|encounter|id|time|date|timestamp|charttime|admissiontime|dischargetime)",
    re.IGNORECASE,
)

EVENT_LABEL_PATTERNS = re.compile(
    r"(discharge|status|death|deceased|mortality|surviv|outcome|endpoint|label|event|reason|icu|unit|ward)",
    re.IGNORECASE,
)

NOT_ALLOWED_PUBLIC_CLAIMS = [
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

def find_reference_tarball() -> Path:
    candidates = sorted(RAW_DIR.rglob("*reference*.tar.gz"))
    if not candidates:
        candidates = sorted(RAW_DIR.rglob("*general*.tar.gz"))
    if not candidates:
        raise SystemExit("ERROR: reference_data tarball not found under local_private HiRID raw folder.")
    return candidates[0]

def safe_value_for_distribution(column: str, value: str):
    """
    Only allow low-risk category values.
    Never print IDs/times/free-text long values.
    """
    col = column.lower()
    v = str(value).strip()

    if not v:
        return None

    if BLOCKED_RAW_PATTERNS.search(col):
        return None

    if len(v) > 80:
        return None

    # Allow simple categories/codes only.
    if re.fullmatch(r"[A-Za-z0-9 _./:+-]{1,80}", v):
        return v

    return None

def looks_numeric(v: str) -> bool:
    try:
        float(str(v).strip())
        return True
    except Exception:
        return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    ref_tar = find_reference_tarball()

    with tarfile.open(ref_tar, "r:gz") as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and m.name.lower().endswith("general_table.csv")
        ]

        if not members:
            members = [
                m for m in tf.getmembers()
                if m.isfile() and "general" in m.name.lower() and m.name.lower().endswith(".csv")
            ]

        if not members:
            raise SystemExit("ERROR: general_table.csv not found inside reference tarball.")

        member = members[0]
        f = tf.extractfile(member)
        if f is None:
            raise SystemExit("ERROR: could not read general_table.csv from tarball.")

        text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.DictReader(text)

        columns = list(reader.fieldnames or [])
        row_count = 0
        nonempty_counts = Counter()
        numeric_counts = Counter()
        candidate_counts = defaultdict(Counter)
        unique_tracker = defaultdict(set)
        unique_capped = defaultdict(bool)

        for row in reader:
            row_count += 1

            for col in columns:
                value = str(row.get(col, "")).strip()
                if value:
                    nonempty_counts[col] += 1

                    if looks_numeric(value):
                        numeric_counts[col] += 1

                    if not unique_capped[col]:
                        unique_tracker[col].add(value)
                        if len(unique_tracker[col]) > 200:
                            unique_capped[col] = True
                            unique_tracker[col].clear()

                    if EVENT_LABEL_PATTERNS.search(col):
                        safe_v = safe_value_for_distribution(col, value)
                        if safe_v is not None and len(candidate_counts[col]) <= 75:
                            candidate_counts[col][safe_v] += 1

        column_profiles = []
        for col in columns:
            nonempty = nonempty_counts[col]
            numeric = numeric_counts[col]
            unique_count = None if unique_capped[col] else len(unique_tracker[col])

            if nonempty == 0:
                observed_type = "empty"
            elif numeric / max(nonempty, 1) > 0.95:
                observed_type = "mostly_numeric"
            else:
                observed_type = "mostly_text_or_category"

            column_profiles.append({
                "column": col,
                "nonempty_count": nonempty,
                "nonempty_rate_percent": round(nonempty / max(row_count, 1) * 100.0, 4),
                "observed_type": observed_type,
                "unique_count_capped_at_200": ">200" if unique_capped[col] else unique_count,
                "possible_event_label_column": bool(EVENT_LABEL_PATTERNS.search(col)),
                "distribution_printed": bool(candidate_counts.get(col)),
            })

        candidate_event_label_columns = []
        for col, counts in candidate_counts.items():
            if not counts:
                continue

            candidate_event_label_columns.append({
                "column": col,
                "top_category_counts": dict(counts.most_common(30)),
                "note": "Aggregate category counts only. No row-level records printed."
            })

        if candidate_event_label_columns:
            decision = "READY_FOR_EVENT_LABEL_MAPPING_REVIEW"
        else:
            decision = "NEEDS_MANUAL_REVIEW_NO_CLEAR_EVENT_LABEL_COLUMN_FOUND"

        output = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "dataset": "HiRID v1.1.1",
            "source_tarball": str(ref_tar),
            "source_member": member.name,
            "public_wording_allowed_now": PUBLIC_WORDING,
            "privacy_policy": {
                "raw_rows_printed": False,
                "raw_rows_exported": False,
                "patient_level_outputs_exported": False,
                "patient_ids_exported": False,
                "timestamps_exported": False,
                "aggregate_only": True,
                "local_private_only": True,
            },
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "column_profiles": column_profiles,
            "candidate_event_label_columns": candidate_event_label_columns,
            "not_allowed_public_claims": NOT_ALLOWED_PUBLIC_CLAIMS,
            "next_step": (
                "Review candidate event-label columns. Only after selecting an appropriate aggregate event label "
                "should an event-window FPR/detection pass be run."
            ),
        }

        OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

        lines = [
            "# HiRID Local-Only General Table Schema Audit",
            "",
            f"Timestamp UTC: {output['timestamp_utc']}",
            "",
            f"Decision: **{decision}**",
            "",
            "**Status:** Schema and aggregate category audit only. This is **not public validation**.",
            "",
            "## Allowed Public Wording Right Now",
            "",
            f"> {PUBLIC_WORDING}",
            "",
            "## Source",
            "",
            f"- Reference tarball: `{ref_tar}`",
            f"- CSV member: `{member.name}`",
            "",
            "## Aggregate Table Shape",
            "",
            f"- Rows: {row_count:,}",
            f"- Columns: {len(columns):,}",
            "",
            "## Columns",
            "",
        ]

        for col in columns:
            lines.append(f"- `{col}`")

        lines.extend([
            "",
            "## Candidate Event / Status Columns",
            "",
        ])

        if candidate_event_label_columns:
            for c in candidate_event_label_columns:
                lines.append(f"### `{c['column']}`")
                lines.append("")
                lines.append("| Category | Count |")
                lines.append("|---|---:|")
                for k, v in c["top_category_counts"].items():
                    lines.append(f"| `{k}` | {v:,} |")
                lines.append("")
        else:
            lines.append("- No clear low-risk event/status category column found from schema audit.")
            lines.append("")

        lines.extend([
            "## Column Profiles",
            "",
            "| Column | Nonempty | Nonempty % | Type | Unique Count | Candidate Label? |",
            "|---|---:|---:|---|---:|---|",
        ])

        for p in column_profiles:
            lines.append(
                f"| `{p['column']}` | "
                f"{p['nonempty_count']:,} | "
                f"{p['nonempty_rate_percent']}% | "
                f"{p['observed_type']} | "
                f"{p['unique_count_capped_at_200']} | "
                f"{p['possible_event_label_column']} |"
            )

        lines.extend([
            "",
            "## Guardrail",
            "",
            "Do not update public pages or claim HiRID validation yet.",
            "",
            "This audit does not calculate FPR, detection rate, lead time, clinical validation, prospective validation, diagnosis, treatment direction, or independent escalation.",
            "",
            "Do not export raw rows, timestamps, patient-level records, patient-level predictions, patient IDs, or restricted files.",
            "",
            "## Not Allowed Public Claims",
            "",
        ])

        for claim in NOT_ALLOWED_PUBLIC_CLAIMS:
            lines.append(f"- {claim}")

        lines.append("")

        OUT_MD.write_text("\n".join(lines), encoding="utf-8")

        print("")
        print("GENERAL TABLE SCHEMA AUDIT COMPLETE")
        print("=" * 80)
        print("Decision:", decision)
        print("Rows:", f"{row_count:,}")
        print("Columns:", len(columns))
        print("Candidate event/status columns:", len(candidate_event_label_columns))
        print("JSON:", OUT_JSON)
        print("MD:", OUT_MD)
        print("")
        print("Allowed public wording remains:")
        print(PUBLIC_WORDING)

if __name__ == "__main__":
    main()
