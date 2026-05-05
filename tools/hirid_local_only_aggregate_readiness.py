#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import io
import json
import math
import re
import tarfile
from collections import defaultdict

RAW_DIR = Path("data/validation/local_private/hirid/raw")
OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
MAP_JSON = OUT_DIR / "hirid_locked_vital_mapping.json"

OUT_JSON = OUT_DIR / "hirid_local_only_aggregate_readiness_summary.json"
OUT_MD = OUT_DIR / "hirid_local_only_aggregate_readiness_summary.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "heart_rate": lambda x: x >= 120 or x <= 40,
    "spo2": lambda x: x < 94,
    "respiratory_rate": lambda x: x >= 24 or x <= 8,
    "systolic_bp": lambda x: x >= 180 or x <= 90,
    "diastolic_bp": lambda x: x >= 110 or x <= 50,
    "temperature": lambda x: x >= 38.0 or x <= 35.0,
}

CLINICAL_RANGE_SANITY = {
    "heart_rate": lambda x: 20 <= x <= 250,
    "spo2": lambda x: 50 <= x <= 100,
    "respiratory_rate": lambda x: 1 <= x <= 80,
    "systolic_bp": lambda x: 40 <= x <= 300,
    "diastolic_bp": lambda x: 20 <= x <= 200,
    "temperature": lambda x: 25 <= x <= 45,
}

def norm_key(k: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(k or "").strip().lower().replace("\ufeff", ""))

def row_get(row: dict, names: list[str]) -> str:
    wanted = {norm_key(n) for n in names}
    for k, v in row.items():
        if norm_key(k) in wanted:
            return str(v).strip() if v is not None else ""
    return ""

def parse_int(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def parse_float(x):
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None

def normalize_value(vital: str, value):
    if value is None:
        return None
    if vital == "spo2" and 0 < value <= 1.5:
        return value * 100.0
    return value

def ids_from_spec(spec):
    ids = []
    if isinstance(spec, dict):
        keys = ["primary_ids", "secondary_ids", "variable_ids", "hirid_variable_ids", "ids", "variableid", "variable_id"]
        for k in keys:
            vals = spec.get(k, [])
            if isinstance(vals, (int, float, str)):
                vals = [vals]
            for v in vals or []:
                try:
                    ids.append(int(float(v)))
                except Exception:
                    pass
    elif isinstance(spec, list):
        for v in spec:
            try:
                ids.append(int(float(v)))
            except Exception:
                pass
    elif isinstance(spec, (int, float, str)):
        for v in re.findall(r"\b\d+\b", str(spec)):
            ids.append(int(v))
    return sorted(set(ids))

def load_mapping():
    data = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    vm = data.get("vital_mapping", {})
    id_to_vitals = defaultdict(list)

    if not isinstance(vm, dict):
        raise SystemExit(f"Expected vital_mapping dict, found {type(vm).__name__}")

    for vital, spec in vm.items():
        vital_key = str(vital).lower().strip().replace(" ", "_").replace("spo₂", "spo2")
        for vid in ids_from_spec(spec):
            id_to_vitals[vid].append(vital_key)

    if not id_to_vitals:
        raise SystemExit("No variable IDs loaded from mapping.")

    return data, id_to_vitals

def find_observation_tarball():
    candidates = sorted(RAW_DIR.rglob("*observation_tables_csv*.tar.gz"))
    if not candidates:
        candidates = sorted(RAW_DIR.rglob("*observation_tables*.tar.gz"))
    if not candidates:
        raise SystemExit("ERROR: observation tables tarball not found under local_private HiRID raw folder.")
    return candidates[0]

class Agg:
    def __init__(self):
        self.rows = 0
        self.numeric_rows = 0
        self.threshold_positive_rows = 0
        self.out_of_sanity_range_rows = 0
        self.sum = 0.0
        self.min = None
        self.max = None
        self.patient_ids = set()
        self.variable_ids = defaultdict(int)

    def add(self, value, patient_id, variable_id, vital):
        self.rows += 1
        self.variable_ids[str(variable_id)] += 1

        if patient_id not in ("", "None", "nan", "NaN", "null"):
            self.patient_ids.add(str(patient_id))

        if value is None:
            return

        value = normalize_value(vital, value)
        if value is None:
            return

        self.numeric_rows += 1
        self.sum += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

        if vital in THRESHOLDS and THRESHOLDS[vital](value):
            self.threshold_positive_rows += 1

        if vital in CLINICAL_RANGE_SANITY and not CLINICAL_RANGE_SANITY[vital](value):
            self.out_of_sanity_range_rows += 1

    def to_dict(self):
        mean = self.sum / self.numeric_rows if self.numeric_rows else None
        threshold_pct = (self.threshold_positive_rows / self.numeric_rows * 100.0) if self.numeric_rows else None

        return {
            "rows": self.rows,
            "numeric_rows": self.numeric_rows,
            "unique_patients": len(self.patient_ids),
            "unique_patients_count": len(self.patient_ids),
            "min": round(self.min, 4) if self.min is not None else None,
            "max": round(self.max, 4) if self.max is not None else None,
            "mean": round(mean, 4) if mean is not None else None,
            "threshold_positive_rows": self.threshold_positive_rows,
            "threshold_positive_rate": round(threshold_pct, 4) if threshold_pct is not None else None,
            "threshold_positive_rate_percent": round(threshold_pct, 4) if threshold_pct is not None else None,
            "threshold_positive_rate_among_numeric_rows": round(threshold_pct / 100.0, 6) if threshold_pct is not None else None,
            "out_of_sanity_range_rows": self.out_of_sanity_range_rows,
            "variable_id_counts": dict(sorted(self.variable_ids.items(), key=lambda x: x[0])),
        }

def main():
    mapping, id_to_vitals = load_mapping()
    obs_tar = find_observation_tarball()

    aggs = defaultdict(Agg)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_only_aggregate_readiness_not_public_validation",
        "dataset": "HiRID v1.1.1",
        "public_wording_allowed_now": "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        "privacy_controls": {
            "raw_files_committed": False,
            "patient_level_rows_written": False,
            "timestamps_written": False,
            "aggregate_summary_only": True
        },
        "rows_scanned_total": 0,
        "vital_rows_matched_total": 0,
        "members_processed_count": 0,
        "vital_aggregate_summary": {}
    }

    print(f"Using observation tarball: {obs_tar}")
    print(f"Locked variable IDs: {sorted(id_to_vitals.keys())}")
    print("Streaming observation rows. No raw rows will be printed or exported.")
    print("")

    with tarfile.open(obs_tar, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]

        for idx, m in enumerate(members, start=1):
            f = tf.extractfile(m)
            if f is None:
                continue

            text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text)

            member_rows = 0
            member_vital_rows = 0
            member_patient_hits = 0

            for row in reader:
                member_rows += 1
                result["rows_scanned_total"] += 1

                vid = parse_int(row_get(row, ["variableid", "variable_id", "Variable ID", "varid"]))
                if vid is None or vid not in id_to_vitals:
                    continue

                value = parse_float(row_get(row, ["value", "Value", "valuenum", "numericvalue"]))
                patient_id = row_get(row, ["patientid", "patient_id", "Patient ID", "patient", "pid", "subject_id"])

                if patient_id:
                    member_patient_hits += 1

                for vital in id_to_vitals[vid]:
                    aggs[vital].add(value, patient_id, vid, vital)
                    result["vital_rows_matched_total"] += 1
                    member_vital_rows += 1

            result["members_processed_count"] += 1

            print(
                f"Processed {idx}/{len(members)}: {m.name} | "
                f"rows={member_rows:,} | vital_rows={member_vital_rows:,} | patient_id_hits={member_patient_hits:,}"
            )

    for vital, agg in sorted(aggs.items()):
        result["vital_aggregate_summary"][vital] = agg.to_dict()

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# HiRID Local-Only Aggregate Readiness Summary",
        "",
        f"Timestamp UTC: {result['timestamp_utc']}",
        "",
        "Status: local-only aggregate readiness pass. This is not public validation.",
        "",
        "Current allowed public wording:",
        "",
        "> HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.",
        "",
        "## Overall Counts",
        "",
        f"- Rows scanned total: {result['rows_scanned_total']:,}",
        f"- Matched vital rows total: {result['vital_rows_matched_total']:,}",
        "",
        "## Vital Aggregate Summary",
        ""
    ]

    for vital, s in result["vital_aggregate_summary"].items():
        lines.extend([
            f"### {vital}",
            "",
            f"- Rows: {s['rows']:,}",
            f"- Numeric rows: {s['numeric_rows']:,}",
            f"- Unique patients count: {s['unique_patients_count']:,}",
            f"- Min: {s['min']}",
            f"- Max: {s['max']}",
            f"- Mean: {s['mean']}",
            f"- Threshold-positive rows: {s['threshold_positive_rows']:,}",
            f"- Threshold-positive rate: {s['threshold_positive_rate']}%",
            f"- Out-of-sanity-range rows: {s['out_of_sanity_range_rows']:,}",
            f"- Variable ID counts: {s['variable_id_counts']}",
            ""
        ])

    lines.extend([
        "## Guardrail",
        "",
        "These are local-only aggregate readiness counts only. Do not describe them as clinical validation, prospective validation, or final HiRID performance.",
        ""
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("DONE — aggregate readiness rebuilt.")
    print(f"Wrote JSON: {OUT_JSON}")
    print(f"Wrote MD:   {OUT_MD}")
    print("")
    print("SUMMARY")
    print("=" * 80)
    print(f"Rows scanned total: {result['rows_scanned_total']:,}")
    print(f"Matched vital rows total: {result['vital_rows_matched_total']:,}")
    for vital, s in result["vital_aggregate_summary"].items():
        print(
            f"{vital}: rows={s['rows']:,}, patients={s['unique_patients_count']:,}, "
            f"mean={s['mean']}, threshold_positive_rate={s['threshold_positive_rate']}%"
        )

if __name__ == "__main__":
    main()
