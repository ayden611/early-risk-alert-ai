#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import io
import json
import re
import tarfile
from collections import defaultdict

ROOT = Path(".")
RAW = ROOT / "data" / "validation" / "local_private" / "hirid" / "raw"
OUT = ROOT / "data" / "validation" / "local_private" / "hirid" / "aggregate_outputs"

OUT_JSON = OUT / "hirid_spo2_mapping_normalization_diagnostic.json"
OUT_MD = OUT / "hirid_spo2_mapping_normalization_diagnostic.md"

SPO2_PATTERNS = [
    "spo2",
    "sp02",
    "oxygen saturation",
    "oxygen sat",
    "peripheral oxygen",
    "pulse oxim",
    "saturation peripheral",
]

EXCLUDE_HINTS = [
    "fio2",
    "fraction inspired",
    "inspired oxygen",
    "oxygen flow",
    "flow rate",
    "po2",
    "partial pressure",
    "venous oxygen",
    "arterial oxygen pressure",
]

def norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(k).strip().lower())

def row_get(row: dict, names: list[str]) -> str:
    by_norm = {norm_key(k): v for k, v in row.items()}
    for name in names:
        v = by_norm.get(norm_key(name))
        if v is not None:
            return str(v).strip()
    return ""

def as_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def safe_open_tar_member_text(tf: tarfile.TarFile, member: tarfile.TarInfo):
    f = tf.extractfile(member)
    if f is None:
        return None
    return io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")

def find_reference_candidates() -> dict[str, dict]:
    candidates = {}

    reference_tars = list(RAW.rglob("*reference*.tar.gz")) + list(RAW.rglob("*reference*.tgz"))
    reference_csvs = list(RAW.rglob("*reference*.csv")) + list(RAW.rglob("*variable*.csv")) + list(RAW.rglob("*ordinal*.csv"))

    def inspect_csv(reader, source_name):
        for row in reader:
            joined = " | ".join(str(v) for v in row.values()).lower()
            if not any(p in joined for p in SPO2_PATTERNS):
                continue

            excluded = any(x in joined for x in EXCLUDE_HINTS)

            vid = (
                row_get(row, ["variableid", "variable_id", "Variable id", "Variable ID", "ID", "id"])
                or row_get(row, ["code"])
            )

            name = (
                row_get(row, ["Variable Name", "variable name", "name", "Name"])
                or joined[:160]
            )

            unit = row_get(row, ["Unit", "unit"])
            if vid:
                candidates[str(vid)] = {
                    "variable_id": str(vid),
                    "name": name,
                    "unit": unit,
                    "source": source_name,
                    "excluded_hint": excluded,
                    "raw_reference_text": joined[:260],
                }

    for t in reference_tars:
        try:
            with tarfile.open(t, "r:gz") as tf:
                for m in tf.getmembers():
                    if not m.isfile() or not m.name.lower().endswith(".csv"):
                        continue
                    text = safe_open_tar_member_text(tf, m)
                    if text is None:
                        continue
                    reader = csv.DictReader(text)
                    inspect_csv(reader, f"{t.name}:{m.name}")
        except Exception as e:
            print(f"Reference tar read warning: {t}: {e}")

    for c in reference_csvs:
        try:
            with c.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                inspect_csv(reader, str(c))
        except Exception as e:
            print(f"Reference csv read warning: {c}: {e}")

    return candidates

class Agg:
    def __init__(self):
        self.rows = 0
        self.numeric_rows = 0
        self.sum_raw = 0.0
        self.sum_norm = 0.0
        self.min_raw = None
        self.max_raw = None
        self.min_norm = None
        self.max_norm = None
        self.unique_patients = set()
        self.raw_buckets = defaultdict(int)
        self.norm_buckets = defaultdict(int)
        self.norm_threshold_positive = 0
        self.norm_out_of_range = 0

    def add(self, patientid: str, value):
        self.rows += 1
        if patientid:
            self.unique_patients.add(patientid)

        v = as_float(value)
        if v is None:
            return

        self.numeric_rows += 1
        self.sum_raw += v
        self.min_raw = v if self.min_raw is None else min(self.min_raw, v)
        self.max_raw = v if self.max_raw is None else max(self.max_raw, v)

        if v <= 0:
            self.raw_buckets["<=0"] += 1
        elif v <= 1.5:
            self.raw_buckets["0_to_1_5"] += 1
        elif v <= 20:
            self.raw_buckets["1_5_to_20"] += 1
        elif v < 70:
            self.raw_buckets["20_to_70"] += 1
        elif v <= 100:
            self.raw_buckets["70_to_100"] += 1
        else:
            self.raw_buckets[">100"] += 1

        # SpO2 normalization rule:
        # If value looks like a fraction, convert 0.94 -> 94.
        # Otherwise keep percentage-style values as-is.
        nv = v * 100.0 if 0 < v <= 1.5 else v

        self.sum_norm += nv
        self.min_norm = nv if self.min_norm is None else min(self.min_norm, nv)
        self.max_norm = nv if self.max_norm is None else max(self.max_norm, nv)

        if nv <= 0:
            self.norm_buckets["<=0"] += 1
        elif nv < 70:
            self.norm_buckets["0_to_70"] += 1
        elif nv <= 100:
            self.norm_buckets["70_to_100"] += 1
        else:
            self.norm_buckets[">100"] += 1

        if nv < 94:
            self.norm_threshold_positive += 1

        if nv < 50 or nv > 100:
            self.norm_out_of_range += 1

    def to_dict(self):
        raw_mean = self.sum_raw / self.numeric_rows if self.numeric_rows else None
        norm_mean = self.sum_norm / self.numeric_rows if self.numeric_rows else None
        norm_tpr = (self.norm_threshold_positive / self.numeric_rows * 100.0) if self.numeric_rows else None
        norm_out = (self.norm_out_of_range / self.numeric_rows * 100.0) if self.numeric_rows else None

        return {
            "rows": self.rows,
            "numeric_rows": self.numeric_rows,
            "unique_patients_count": len(self.unique_patients),
            "raw_min": self.min_raw,
            "raw_max": self.max_raw,
            "raw_mean": raw_mean,
            "normalized_min": self.min_norm,
            "normalized_max": self.max_norm,
            "normalized_mean": norm_mean,
            "normalized_threshold_positive_rate_percent_lt_94": norm_tpr,
            "normalized_out_of_range_rate_percent_lt50_or_gt100": norm_out,
            "raw_buckets": dict(self.raw_buckets),
            "normalized_buckets": dict(self.norm_buckets),
        }

def scan_observation_tables(candidate_ids: set[str]) -> dict[str, Agg]:
    if not candidate_ids:
        return {}

    obs_tars = list(RAW.rglob("*observation_tables_csv.tar.gz")) + list(RAW.rglob("*observation_tables*.tar.gz"))
    if not obs_tars:
        raise SystemExit("No observation_tables CSV tarball found under local_private HiRID raw folder.")

    obs_tar = obs_tars[0]
    print(f"Scanning observation table tarball for SpO2 candidate IDs only: {obs_tar}")
    print(f"Candidate IDs: {', '.join(sorted(candidate_ids))}")

    aggs = {vid: Agg() for vid in candidate_ids}

    with tarfile.open(obs_tar, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
        total_members = len(members)

        for i, m in enumerate(members, start=1):
            text = safe_open_tar_member_text(tf, m)
            if text is None:
                continue

            reader = csv.DictReader(text)
            for row in reader:
                vid = row_get(row, ["variableid", "variable_id", "Variable ID", "Variable id", "id"])
                if vid not in aggs:
                    continue

                patientid = row_get(row, ["patientid", "patient_id", "Patient ID"])
                value = row_get(row, ["value", "Value"])
                aggs[vid].add(patientid, value)

            if i % 25 == 0 or i == total_members:
                matched = sum(a.rows for a in aggs.values())
                print(f"Progress: scanned member {i}/{total_members}; matched candidate rows={matched:,}")

    return aggs

def score_candidate(summary: dict) -> int:
    score = 0
    rows = summary.get("rows") or 0
    nmean = summary.get("normalized_mean")
    ntpr = summary.get("normalized_threshold_positive_rate_percent_lt_94")
    out = summary.get("normalized_out_of_range_rate_percent_lt50_or_gt100")
    buckets = summary.get("normalized_buckets", {})
    in_range = buckets.get("70_to_100", 0)
    numeric = summary.get("numeric_rows") or 0
    in_range_rate = in_range / numeric * 100 if numeric else 0

    if rows >= 10000:
        score += 2
    if nmean is not None and 85 <= nmean <= 100:
        score += 4
    elif nmean is not None and 70 <= nmean <= 100:
        score += 2
    if ntpr is not None and ntpr <= 40:
        score += 2
    if out is not None and out <= 10:
        score += 2
    if in_range_rate >= 80:
        score += 3
    return score

def main():
    candidates = find_reference_candidates()

    if not candidates:
        raise SystemExit("No SpO2 candidate variables found in reference files. Send me the mapping audit output.")

    all_ids = set(candidates.keys())
    scan_results = scan_observation_tables(all_ids)

    final = []
    for vid, meta in candidates.items():
        summary = scan_results.get(vid, Agg()).to_dict()
        candidate_score = score_candidate(summary)
        final.append({
            **meta,
            "aggregate": summary,
            "candidate_score": candidate_score,
            "recommendation": (
                "LIKELY_SPO2_AFTER_NORMALIZATION"
                if candidate_score >= 8 and not meta.get("excluded_hint")
                else "REVIEW"
            ),
        })

    final.sort(key=lambda x: x["candidate_score"], reverse=True)

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local-only aggregate SpO2 mapping and normalization diagnostic.",
        "policy": {
            "raw_rows_printed": False,
            "public_claim_allowed": False,
            "next_step": "Review candidate mapping/normalization before running HiRID operating-point validation.",
        },
        "candidates": final,
    }

    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = []
    lines.append("# HiRID SpO2 Local-Only Mapping + Normalization Diagnostic")
    lines.append("")
    lines.append(f"Timestamp: {output['timestamp_utc']}")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Do **not** publish HiRID validation claims yet. This diagnostic only checks whether SpO2 needs remapping or normalization.")
    lines.append("")
    lines.append("## Candidate Results")
    lines.append("")

    for c in final:
        a = c["aggregate"]
        lines.append(f"### Variable ID: {c['variable_id']}")
        lines.append("")
        lines.append(f"- Name: {c.get('name', '')}")
        lines.append(f"- Unit: {c.get('unit', '')}")
        lines.append(f"- Source: {c.get('source', '')}")
        lines.append(f"- Excluded hint: {c.get('excluded_hint')}")
        lines.append(f"- Candidate score: {c['candidate_score']}")
        lines.append(f"- Recommendation: **{c['recommendation']}**")
        lines.append(f"- Rows: {a['rows']:,}")
        lines.append(f"- Numeric rows: {a['numeric_rows']:,}")
        lines.append(f"- Unique patients: {a['unique_patients_count']:,}")
        lines.append(f"- Raw mean: {a['raw_mean']}")
        lines.append(f"- Normalized mean: {a['normalized_mean']}")
        lines.append(f"- Normalized threshold-positive rate <94%: {a['normalized_threshold_positive_rate_percent_lt_94']}")
        lines.append(f"- Normalized out-of-range rate <50 or >100: {a['normalized_out_of_range_rate_percent_lt50_or_gt100']}")
        lines.append(f"- Raw buckets: `{a['raw_buckets']}`")
        lines.append(f"- Normalized buckets: `{a['normalized_buckets']}`")
        lines.append("")

    lines.append("## Guardrail")
    lines.append("")
    lines.append("This output is aggregate-only and local-only. It is not clinical validation, prospective validation, or public performance evidence.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("SPO2 DIAGNOSTIC COMPLETE")
    print("Candidates found:", len(final))
    print("Written:")
    print("-", OUT_JSON)
    print("-", OUT_MD)
    print("")
    print("Top candidates:")
    for c in final[:8]:
        a = c["aggregate"]
        print(
            f"ID={c['variable_id']} | score={c['candidate_score']} | rec={c['recommendation']} | "
            f"rows={a['rows']:,} | raw_mean={a['raw_mean']} | norm_mean={a['normalized_mean']} | "
            f"norm_lt94={a['normalized_threshold_positive_rate_percent_lt_94']} | name={c.get('name','')[:80]}"
        )

if __name__ == "__main__":
    main()
