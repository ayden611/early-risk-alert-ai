#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import tarfile
from collections import defaultdict

RAW_DIR = Path("data/validation/local_private/hirid/raw")
OUT_DIR = Path("data/validation/local_private/hirid/aggregate_outputs")
WORK_DIR = Path("data/validation/local_private/hirid/work")

MAPPING_JSON = OUT_DIR / "hirid_locked_vital_mapping.json"
SANITY_JSON = OUT_DIR / "hirid_local_only_vital_sanity_gate.json"

OUT_JSON = OUT_DIR / "hirid_local_only_threshold_operating_points.json"
OUT_MD = OUT_DIR / "hirid_local_only_threshold_operating_points.md"
WORK_DB = WORK_DIR / "hirid_private_threshold_work.sqlite"

KEEP_PRIVATE_WORK_DB = os.environ.get("KEEP_PRIVATE_WORK_DB", "0") == "1"

THRESHOLDS = [4.0, 5.0, 6.0]

PUBLIC_WORDING = "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."

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

RANGE_OK = {
    "heart_rate": lambda x: 20 <= x <= 250,
    "spo2": lambda x: 50 <= x <= 100,
    "respiratory_rate": lambda x: 1 <= x <= 80,
    "systolic_bp": lambda x: 40 <= x <= 300,
    "diastolic_bp": lambda x: 20 <= x <= 200,
    "temperature": lambda x: 25 <= x <= 45,
}

def score_value(vital: str, x: float) -> float:
    if vital == "heart_rate":
        if x >= 130 or x <= 40:
            return 2.0
        if x >= 120 or x <= 45:
            return 1.25
        return 0.0

    if vital == "spo2":
        if x < 90:
            return 2.25
        if x < 94:
            return 1.75
        return 0.0

    if vital == "respiratory_rate":
        if x >= 30 or x <= 8:
            return 2.0
        if x >= 24:
            return 1.5
        return 0.0

    if vital == "systolic_bp":
        if x >= 180 or x <= 80:
            return 1.75
        if x >= 160 or x <= 90:
            return 1.0
        return 0.0

    if vital == "diastolic_bp":
        if x >= 110 or x <= 45:
            return 1.25
        if x <= 50:
            return 0.75
        return 0.0

    if vital == "temperature":
        if x >= 39.0 or x <= 35.0:
            return 1.5
        if x >= 38.0:
            return 1.0
        return 0.0

    return 0.0

def normalize_value(vital: str, value: float | None) -> float | None:
    if value is None:
        return None
    if vital == "spo2" and 0 < value <= 1.5:
        return value * 100.0
    return value

def parse_float(x) -> float | None:
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None

def parse_int(x) -> int | None:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def row_get(row: dict, names: list[str]) -> str:
    lower = {str(k).strip().lower().replace("\ufeff", ""): v for k, v in row.items()}
    for name in names:
        v = lower.get(name.lower())
        if v is not None:
            return str(v).strip()
    return ""

def patient_hash(patient_id: str) -> str:
    # Local-only pseudonymous key. Not exported in outputs.
    return hashlib.sha256(("ERA_HIRID_LOCAL_ONLY:" + str(patient_id)).encode("utf-8")).hexdigest()[:20]

def time_bin_hour(raw_time: str) -> int | None:
    # HiRID raw_stage datetime is typically numeric/relative. Use hour bins only.
    v = parse_float(raw_time)
    if v is not None:
        return int(v // 3600)

    text = str(raw_time).strip()
    if not text:
        return None

    # Fallback coarse local-only bin. Never exported.
    return abs(hash(text[:13])) % 10_000_000

def ids_from_spec(spec) -> list[int]:
    if not isinstance(spec, dict):
        return []

    out = []
    for key in ["primary_ids", "secondary_ids", "variable_ids", "hirid_variable_ids", "ids", "variableid", "variable_id"]:
        vals = spec.get(key)
        if vals is None:
            continue
        if isinstance(vals, (str, int, float)):
            vals = [vals]
        for v in vals:
            try:
                out.append(int(float(v)))
            except Exception:
                pass

    excluded = set()
    vals = spec.get("excluded_ids", [])
    if isinstance(vals, (str, int, float)):
        vals = [vals]
    for v in vals or []:
        try:
            excluded.add(int(float(v)))
        except Exception:
            pass

    return sorted(set(x for x in out if x not in excluded))

def load_mapping() -> dict[int, list[str]]:
    if not MAPPING_JSON.exists():
        raise SystemExit(f"Missing mapping JSON: {MAPPING_JSON}")

    data = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
    vm = data.get("vital_mapping", {})
    out = defaultdict(list)

    if not isinstance(vm, dict):
        raise SystemExit("Mapping must be dictionary-shaped before threshold pass.")

    for vital, spec in vm.items():
        key = str(vital).lower().strip().replace(" ", "_").replace("spo₂", "spo2")
        for vid in ids_from_spec(spec):
            out[vid].append(key)

    required = {"heart_rate", "spo2", "respiratory_rate", "systolic_bp", "diastolic_bp", "temperature"}
    found = {v for vals in out.values() for v in vals}
    missing = sorted(required - found)
    if missing:
        raise SystemExit(f"Missing required vital mappings: {missing}")

    return dict(out)

def find_observation_tarball() -> Path:
    candidates = sorted(RAW_DIR.rglob("*observation_tables_csv*.tar.gz"))
    if not candidates:
        candidates = sorted(RAW_DIR.rglob("*observation_tables*.tar.gz"))
    if not candidates:
        raise SystemExit("Could not find observation_tables CSV tarball.")
    return candidates[0]

def sanity_gate_check():
    data = json.loads(SANITY_JSON.read_text(encoding="utf-8"))
    fail_count = int(data.get("fail_count", 0) or 0)
    failed = [r for r in data.get("results", []) if str(r.get("status", "")).upper() == "FAIL"]
    if fail_count > 0 or failed or data.get("overall_status") == "BLOCK_PUBLIC_VALIDATION":
        raise SystemExit("Sanity gate has actual FAIL results. Stop.")

def init_db(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vital_bin (
            pid_hash TEXT NOT NULL,
            hour_bin INTEGER NOT NULL,
            vital TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (pid_hash, hour_bin, vital)
        )
    """)
    conn.commit()

def upsert_batch(conn: sqlite3.Connection, rows: list[tuple[str, int, str, float]]):
    if not rows:
        return
    conn.executemany("""
        INSERT INTO vital_bin(pid_hash, hour_bin, vital, score)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(pid_hash, hour_bin, vital)
        DO UPDATE SET score = CASE
            WHEN excluded.score > vital_bin.score THEN excluded.score
            ELSE vital_bin.score
        END
    """, rows)
    conn.commit()

def main():
    sanity_gate_check()
    id_to_vitals = load_mapping()
    obs_tar = find_observation_tarball()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if WORK_DB.exists():
        WORK_DB.unlink()

    print("Observation tarball:", obs_tar)
    print("Variable IDs loaded:", sorted(id_to_vitals.keys()))
    print("Local private work DB:", WORK_DB)
    print("Streaming rows. Aggregate-only output. No raw rows printed.")
    print("")

    conn = sqlite3.connect(str(WORK_DB))
    init_db(conn)

    scanned_rows = 0
    matched_rows = 0
    scored_rows = 0
    skipped_bad_rows = 0
    batch = []

    vital_match_counts = defaultdict(int)
    vital_score_positive_counts = defaultdict(int)

    with tarfile.open(obs_tar, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]

        for idx, m in enumerate(members, start=1):
            f = tf.extractfile(m)
            if f is None:
                continue

            text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text)

            member_rows = 0
            member_matches = 0

            for row in reader:
                scanned_rows += 1
                member_rows += 1

                vid = parse_int(row_get(row, ["variableid", "variable_id", "Variable ID"]))
                if vid is None or vid not in id_to_vitals:
                    continue

                value = parse_float(row_get(row, ["value", "Value"]))
                pid = row_get(row, ["patientid", "patient_id", "Patient ID"])
                raw_time = row_get(row, ["datetime", "time", "charttime"])

                if not pid or not raw_time:
                    skipped_bad_rows += 1
                    continue

                hour_bin = time_bin_hour(raw_time)
                if hour_bin is None:
                    skipped_bad_rows += 1
                    continue

                ph = patient_hash(pid)

                for vital in id_to_vitals[vid]:
                    x = normalize_value(vital, value)
                    if x is None:
                        continue
                    if vital in RANGE_OK and not RANGE_OK[vital](x):
                        continue

                    score = score_value(vital, x)
                    vital_match_counts[vital] += 1
                    matched_rows += 1
                    member_matches += 1

                    if score > 0:
                        scored_rows += 1
                        vital_score_positive_counts[vital] += 1
                        batch.append((ph, hour_bin, vital, score))

                    if len(batch) >= 10000:
                        upsert_batch(conn, batch)
                        batch.clear()

            upsert_batch(conn, batch)
            batch.clear()

            print(
                f"Processed {idx}/{len(members)} | rows={member_rows:,} | "
                f"matched_vital_rows={member_matches:,} | total_matched={matched_rows:,}"
            )

    upsert_batch(conn, batch)
    batch.clear()

    print("")
    print("Building aggregate bins...")
    conn.execute("DROP TABLE IF EXISTS score_bin")
    conn.execute("""
        CREATE TABLE score_bin AS
        SELECT pid_hash, hour_bin, SUM(score) AS review_score
        FROM vital_bin
        GROUP BY pid_hash, hour_bin
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_score_bin_score ON score_bin(review_score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_score_bin_pid ON score_bin(pid_hash)")
    conn.commit()

    total_bins = conn.execute("SELECT COUNT(*) FROM score_bin").fetchone()[0]
    any_signal_bins = conn.execute("SELECT COUNT(*) FROM score_bin WHERE review_score > 0").fetchone()[0]
    unique_patients_any_signal = conn.execute("SELECT COUNT(DISTINCT pid_hash) FROM score_bin WHERE review_score > 0").fetchone()[0]

    threshold_results = []
    for t in THRESHOLDS:
        positive_bins = conn.execute(
            "SELECT COUNT(*) FROM score_bin WHERE review_score >= ?",
            (t,)
        ).fetchone()[0]

        positive_patients = conn.execute(
            "SELECT COUNT(DISTINCT pid_hash) FROM score_bin WHERE review_score >= ?",
            (t,)
        ).fetchone()[0]

        avg_score = conn.execute(
            "SELECT AVG(review_score) FROM score_bin WHERE review_score >= ?",
            (t,)
        ).fetchone()[0]

        max_score = conn.execute(
            "SELECT MAX(review_score) FROM score_bin WHERE review_score >= ?",
            (t,)
        ).fetchone()[0]

        driver_rows = conn.execute("""
            SELECT vb.vital, COUNT(*)
            FROM vital_bin vb
            JOIN score_bin sb
              ON vb.pid_hash = sb.pid_hash
             AND vb.hour_bin = sb.hour_bin
            WHERE sb.review_score >= ?
              AND vb.score > 0
            GROUP BY vb.vital
            ORDER BY COUNT(*) DESC
        """, (t,)).fetchall()

        review_rate_among_bins = (positive_bins / total_bins * 100.0) if total_bins else None
        reduction_vs_any_signal = (
            (1.0 - (positive_bins / any_signal_bins)) * 100.0
            if any_signal_bins else None
        )

        threshold_results.append({
            "threshold": t,
            "positive_review_bins": positive_bins,
            "positive_review_rate_among_scored_bins_percent": round(review_rate_among_bins, 4) if review_rate_among_bins is not None else None,
            "unique_patients_with_positive_review_bins": positive_patients,
            "average_review_score_among_positive_bins": round(avg_score, 4) if avg_score is not None else None,
            "max_review_score_among_positive_bins": round(max_score, 4) if max_score is not None else None,
            "estimated_reduction_vs_any_single_signal_bins_percent": round(reduction_vs_any_signal, 4) if reduction_vs_any_signal is not None else None,
            "driver_signal_counts_among_positive_bins": dict(driver_rows),
        })

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_only_threshold_operating_point_pass_not_public_validation",
        "dataset": "HiRID v1.1.1",
        "allowed_public_wording_now": PUBLIC_WORDING,
        "not_allowed_public_claims": NOT_ALLOWED_PUBLIC_CLAIMS,
        "privacy_policy": {
            "raw_rows_exported": False,
            "patient_level_outputs_exported": False,
            "timestamps_exported": False,
            "patient_ids_exported": False,
            "aggregate_only": True,
            "local_private_only": True,
        },
        "method_note": (
            "Private rules-based aggregate review-score pass using hour-level pseudonymous local bins. "
            "Outputs are aggregate counts only. This does not calculate FPR, detection, or lead-time."
        ),
        "overall_counts": {
            "rows_scanned_total": scanned_rows,
            "matched_vital_rows_total": matched_rows,
            "scored_signal_rows_total": scored_rows,
            "skipped_bad_rows_total": skipped_bad_rows,
            "total_scored_patient_hour_bins": total_bins,
            "any_signal_bins": any_signal_bins,
            "unique_patients_any_signal": unique_patients_any_signal,
        },
        "vital_match_counts": dict(sorted(vital_match_counts.items())),
        "vital_score_positive_counts": dict(sorted(vital_score_positive_counts.items())),
        "threshold_results": threshold_results,
    }

    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# HiRID Local-Only Threshold Operating-Point Pass",
        "",
        f"Timestamp UTC: {output['timestamp_utc']}",
        "",
        "**Status:** Local-only threshold operating-point pass. This is **not public validation**.",
        "",
        "## Allowed Public Wording Right Now",
        "",
        f"> {PUBLIC_WORDING}",
        "",
        "## Overall Aggregate Counts",
        "",
        f"- Rows scanned total: {scanned_rows:,}",
        f"- Matched vital rows total: {matched_rows:,}",
        f"- Scored signal rows total: {scored_rows:,}",
        f"- Skipped bad rows total: {skipped_bad_rows:,}",
        f"- Total scored patient-hour bins: {total_bins:,}",
        f"- Any-signal bins: {any_signal_bins:,}",
        f"- Unique patients with any signal: {unique_patients_any_signal:,}",
        "",
        "## Threshold Operating-Point Summary",
        "",
        "| Threshold | Positive review bins | Positive review rate | Unique patients | Avg score among positives | Max score | Est. reduction vs any-signal bins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in threshold_results:
        lines.append(
            f"| t={r['threshold']} | "
            f"{r['positive_review_bins']:,} | "
            f"{r['positive_review_rate_among_scored_bins_percent']}% | "
            f"{r['unique_patients_with_positive_review_bins']:,} | "
            f"{r['average_review_score_among_positive_bins']} | "
            f"{r['max_review_score_among_positive_bins']} | "
            f"{r['estimated_reduction_vs_any_single_signal_bins_percent']}% |"
        )

    lines.extend([
        "",
        "## Driver Signal Counts Among Positive Bins",
        "",
    ])

    for r in threshold_results:
        lines.append(f"### t={r['threshold']}")
        lines.append("")
        for vital, count in r["driver_signal_counts_among_positive_bins"].items():
            lines.append(f"- {vital}: {count:,}")
        lines.append("")

    lines.extend([
        "## Guardrail",
        "",
        "This is local-only aggregate threshold behavior. Do not publish it as HiRID validation.",
        "",
        "This pass does **not** calculate FPR, detection rate, lead time, prospective performance, diagnosis, treatment direction, or independent escalation.",
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
    print("DONE — local-only threshold operating-point pass complete.")
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")

    conn.close()

    if KEEP_PRIVATE_WORK_DB:
        print(f"Private work DB retained locally: {WORK_DB}")
    else:
        try:
            WORK_DB.unlink()
            print("Private work DB deleted after aggregate export.")
        except FileNotFoundError:
            pass

        for suffix in [".sqlite-wal", ".sqlite-shm"]:
            sidecar = WORK_DB.with_suffix(suffix)
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass

if __name__ == "__main__":
    main()
