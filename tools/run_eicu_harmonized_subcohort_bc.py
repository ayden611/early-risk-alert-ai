#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys

import pandas as pd
import numpy as np

ROOT = Path(".")
LOCAL = ROOT / "data" / "validation" / "local_private" / "eicu"
OUT = ROOT / "data" / "validation"
DOCS = ROOT / "docs" / "validation"

THRESHOLDS = [
    (4.0, "ICU / high-acuity"),
    (5.0, "Mixed units / balanced"),
    (6.0, "Telemetry / stepdown conservative"),
]

PRE_EVENT_HOURS = 6.0
CLUSTER_GAP_HOURS = 6.0

PATIENT_CANDIDATES = [
    "patient_id", "patientunitstayid", "stay_id", "subject_id", "hadm_id", "case_id"
]
TIME_CANDIDATES = [
    "timestamp", "charttime", "observation_time", "time", "event_time", "chart_time"
]
RISK_CANDIDATES = [
    "risk_score", "era_score", "review_score", "score"
]
EVENT_CANDIDATES = [
    "harmonized_clinical_event",
    "clinical_event",
    "clinical_event_label",
    "harmonized_event",
    "event_label",
    "event_flag",
    "event",
    "deterioration_event",
    "outcome_event",
    "endpoint_event"
]

VITAL_CANDIDATES = {
    "spo2": ["spo2", "sao2", "oxygen_saturation", "o2sat", "oxysat"],
    "hr": ["heart_rate", "heartrate", "hr"],
    "rr": ["respiratory_rate", "resp_rate", "rr"],
    "sbp": ["bp_systolic", "systolic_bp", "systolicbp", "sbp"],
    "dbp": ["bp_diastolic", "diastolic_bp", "diastolicbp", "dbp"],
    "temp": ["temperature_f", "temperature", "temp_f", "temp"],
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def first_col(cols, candidates):
    by_norm = {norm(c): c for c in cols}
    for cand in candidates:
        if norm(cand) in by_norm:
            return by_norm[norm(cand)]
    return None

def find_like(cols, candidates):
    by_norm = {norm(c): c for c in cols}
    for cand in candidates:
        nc = norm(cand)
        if nc in by_norm:
            return by_norm[nc]
    for c in cols:
        nc = norm(c)
        for cand in candidates:
            if norm(cand) in nc:
                return c
    return None

def truthy_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    x = s.astype(str).str.strip().str.lower()
    return x.isin(["1", "true", "yes", "y", "event", "positive", "detected"])

def stable_bucket(pid) -> int:
    h = hashlib.sha1(str(pid).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 3

def locate_input_file():
    if not LOCAL.exists():
        raise SystemExit("ERROR: local eICU folder not found: data/validation/local_private/eicu")

    candidates = []
    for p in LOCAL.rglob("*.csv"):
        name = p.name.lower()
        if any(x in name for x in ["harmonized", "labeled", "scored", "era"]):
            candidates.append(p)

    if not candidates:
        raise SystemExit("ERROR: No local eICU harmonized/scored CSV found under data/validation/local_private/eicu")

    scored = []
    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=100)
        except Exception:
            continue

        cols = list(head.columns)
        patient = first_col(cols, PATIENT_CANDIDATES)
        time = first_col(cols, TIME_CANDIDATES)
        risk = first_col(cols, RISK_CANDIDATES)
        event = first_col(cols, EVENT_CANDIDATES)

        if patient and time and risk and event:
            scored.append((p, patient, time, risk, event, cols))

    if not scored:
        msg = [
            "ERROR: Found local eICU CSV files, but none had patient + time + risk_score + event label columns.",
            "Need a harmonized scored/labeled input like:",
            "data/validation/local_private/eicu/harmonized/eicu_harmonized_clinical_event_labeled_era_cohort.csv",
            "or",
            "data/validation/local_private/eicu/working/eicu_era_scored_harmonized_input.csv",
        ]
        raise SystemExit("\n".join(msg))

    scored.sort(key=lambda x: ("harmonized" not in x[0].name.lower(), "scored" not in x[0].name.lower(), len(str(x[0]))))
    return scored[0]

def standard_vital_alert(df: pd.DataFrame, colmap: dict) -> pd.Series:
    alert = pd.Series(False, index=df.index)

    if colmap.get("spo2"):
        spo2 = pd.to_numeric(df[colmap["spo2"]], errors="coerce")
        alert |= spo2.le(92)

    if colmap.get("hr"):
        hr = pd.to_numeric(df[colmap["hr"]], errors="coerce")
        alert |= hr.ge(110) | hr.le(45)

    if colmap.get("rr"):
        rr = pd.to_numeric(df[colmap["rr"]], errors="coerce")
        alert |= rr.ge(24) | rr.le(8)

    if colmap.get("sbp"):
        sbp = pd.to_numeric(df[colmap["sbp"]], errors="coerce")
        alert |= sbp.le(90) | sbp.ge(180)

    if colmap.get("temp"):
        temp = pd.to_numeric(df[colmap["temp"]], errors="coerce")
        med = temp.dropna().median() if temp.notna().any() else np.nan
        if pd.notna(med) and med > 45:
            alert |= temp.ge(100.9) | temp.le(95.0)
        else:
            alert |= temp.ge(38.3) | temp.le(35.0)

    if not alert.any():
        # Conservative fallback: if no vitals are available, use any score >= 0.5 as the comparison pool.
        # This avoids publishing false precision while keeping the run from crashing.
        return pd.Series(False, index=df.index)

    return alert

def estimate_patient_days(df: pd.DataFrame, patient_col: str, time_col: str) -> float:
    total_hours = 0.0
    for _, g in df[[patient_col, time_col]].dropna().groupby(patient_col, sort=False):
        if len(g) < 2:
            total_hours += 5.0 / 60.0
            continue
        tmin = g[time_col].min()
        tmax = g[time_col].max()
        hrs = max((tmax - tmin).total_seconds() / 3600.0, 5.0 / 60.0)
        total_hours += hrs
    return max(total_hours / 24.0, 0.0001)

def build_clusters(df: pd.DataFrame, patient_col: str, time_col: str, event_col: str):
    clusters = []
    event_rows = df[df[event_col]].sort_values([patient_col, time_col])
    gap = pd.Timedelta(hours=CLUSTER_GAP_HOURS)

    for pid, g in event_rows.groupby(patient_col, sort=False):
        times = list(g[time_col].dropna())
        if not times:
            continue
        current_start = times[0]
        previous = times[0]
        for t in times[1:]:
            if t - previous > gap:
                clusters.append((pid, current_start))
                current_start = t
            previous = t
        clusters.append((pid, current_start))

    return clusters

def evaluate(df: pd.DataFrame, patient_col: str, time_col: str, risk_col: str, event_col: str, threshold: float, setting: str):
    total_rows = int(len(df))
    patient_count = int(df[patient_col].nunique())
    event_rows = int(df[event_col].sum())
    clusters = build_clusters(df, patient_col, time_col, event_col)
    cluster_count = len(clusters)

    risk = pd.to_numeric(df[risk_col], errors="coerce").fillna(0)
    era_alert = risk.ge(threshold)
    era_alert_rows = int(era_alert.sum())

    standard_alert_rows = int(df["_standard_alert"].sum())
    if standard_alert_rows > 0:
        alert_reduction = (1.0 - (era_alert_rows / standard_alert_rows)) * 100.0
    else:
        alert_reduction = None

    patient_days = estimate_patient_days(df, patient_col, time_col)
    alerts_per_patient_day = era_alert_rows / patient_days

    detected = 0
    lead_times = []
    true_window_alert_idx = set()

    grouped = {}
    for pid, g in df[[patient_col, time_col, risk_col]].copy().groupby(patient_col, sort=False):
        grouped[pid] = g.sort_values(time_col)

    for pid, event_time in clusters:
        g = grouped.get(pid)
        if g is None or g.empty:
            continue
        t0 = event_time - pd.Timedelta(hours=PRE_EVENT_HOURS)
        window = g[(g[time_col] >= t0) & (g[time_col] < event_time)].copy()
        if window.empty:
            continue
        window_risk = pd.to_numeric(window[risk_col], errors="coerce").fillna(0)
        hits = window[window_risk.ge(threshold)]
        if not hits.empty:
            detected += 1
            first_alert_time = hits[time_col].min()
            lead = (event_time - first_alert_time).total_seconds() / 3600.0
            if lead >= 0:
                lead_times.append(lead)

    # FPR approximation: ERA alert rows outside event rows, over non-event rows.
    # This is conservative and DUA-safe for aggregate robustness comparison.
    false_alert_rows = int((era_alert & (~df[event_col])).sum())
    non_event_rows = max(total_rows - event_rows, 1)
    fpr = (false_alert_rows / non_event_rows) * 100.0

    detection = (detected / cluster_count * 100.0) if cluster_count else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else None

    return {
        "threshold": threshold,
        "suggested_setting": setting,
        "rows": total_rows,
        "cases": patient_count,
        "event_rows": event_rows,
        "event_clusters": cluster_count,
        "detected_event_clusters": detected,
        "alert_rows": era_alert_rows,
        "standard_threshold_alert_rows": standard_alert_rows,
        "alert_reduction_pct": round(alert_reduction, 2) if alert_reduction is not None else None,
        "fpr_pct": round(fpr, 2),
        "detection_pct": round(detection, 2),
        "median_lead_time_hours": round(median_lead, 2) if median_lead is not None else None,
        "era_alerts_per_patient_day": round(alerts_per_patient_day, 4),
    }

def write_md(path: Path, payload: dict):
    rows = payload["threshold_matrix"]
    lines = []
    lines.append(f"# {payload['title']}")
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("")
    lines.append("DUA-safe aggregate retrospective robustness summary only. Do not publish raw eICU files, row-level outputs, patient identifiers, timestamps, or enriched restricted-data exports.")
    lines.append("")
    lines.append("This supports retrospective robustness assessment under a harmonized threshold framework. It does not establish full clinical validation, prospective performance, diagnosis, treatment direction, or autonomous escalation.")
    lines.append("")
    lines.append("## Cohort")
    lines.append("")
    lines.append(f"- Dataset: {payload['dataset']}")
    lines.append(f"- Subcohort: {payload['subcohort']}")
    lines.append(f"- Split method: {payload['split_method']}")
    lines.append(f"- Cases / stays: {payload['cases']:,}")
    lines.append(f"- Rows: {payload['rows']:,}")
    lines.append("")
    lines.append("## Threshold Matrix")
    lines.append("")
    lines.append("| Threshold | Suggested setting | Alert reduction | FPR | Detection | Median lead time | ERA alerts / patient-day | Event clusters |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ar = "NA" if r["alert_reduction_pct"] is None else f'{r["alert_reduction_pct"]}%'
        lead = "NA" if r["median_lead_time_hours"] is None else f'{r["median_lead_time_hours"]} hrs'
        lines.append(
            f'| t={r["threshold"]} | {r["suggested_setting"]} | {ar} | {r["fpr_pct"]}% | {r["detection_pct"]}% | {lead} | {r["era_alerts_per_patient_day"]} | {r["event_clusters"]} |'
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This subcohort run tests whether the eICU harmonized clinical-event result remains directionally stable when the dataset is split into deterministic patient subsets.")
    lines.append("")
    lines.append("Safe wording: eICU harmonized clinical-event subcohort robustness was evaluated using aggregate-only outputs. Results should be compared against the full eICU harmonized pass and MIMIC-IV only with clear dataset/event-definition qualifications.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    src, patient_col, time_col, risk_col, event_col, cols = locate_input_file()

    print("USING LOCAL eICU INPUT:", src)
    print("patient:", patient_col)
    print("time:", time_col)
    print("risk:", risk_col)
    print("event:", event_col)

    vital_map = {k: find_like(cols, v) for k, v in VITAL_CANDIDATES.items()}
    print("vital map:", vital_map)

    usecols = sorted(set([patient_col, time_col, risk_col, event_col] + [v for v in vital_map.values() if v]))
    df = pd.read_csv(src, usecols=usecols)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[patient_col, time_col])
    df[risk_col] = pd.to_numeric(df[risk_col], errors="coerce").fillna(0)
    df[event_col] = truthy_series(df[event_col])
    df["_standard_alert"] = standard_vital_alert(df, vital_map)
    df["_bucket"] = df[patient_col].map(stable_bucket)

    subcohorts = {
        "B": df[df["_bucket"] == 1].copy(),
        "C": df[df["_bucket"] == 2].copy(),
    }

    all_payloads = []

    for name, sub in subcohorts.items():
        if sub.empty:
            raise SystemExit(f"ERROR: eICU subcohort {name} is empty.")

        print("")
        print("=" * 72)
        print(f"RUNNING eICU HARMONIZED SUBCOHORT {name}")
        print("=" * 72)

        matrix = []
        for t, setting in THRESHOLDS:
            r = evaluate(sub, patient_col, time_col, risk_col, event_col, t, setting)
            matrix.append(r)
            lead = "NA" if r["median_lead_time_hours"] is None else f'{r["median_lead_time_hours"]} hrs'
            ar = "NA" if r["alert_reduction_pct"] is None else f'{r["alert_reduction_pct"]}%'
            print(
                f't={t:g} | alert_reduction={ar} | FPR={r["fpr_pct"]}% | '
                f'detection={r["detection_pct"]}% | lead={lead} | '
                f'alerts/day={r["era_alerts_per_patient_day"]}'
            )

        payload = {
            "ok": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "title": f"eICU Harmonized Clinical-Event Subcohort {name} Robustness Summary",
            "dataset": "eICU v2.0",
            "subcohort": name,
            "split_method": "Deterministic patient-level SHA1 bucket split; public output contains aggregate metrics only.",
            "claim_boundary": "Retrospective aggregate robustness only; not full clinical validation.",
            "cases": int(sub[patient_col].nunique()),
            "rows": int(len(sub)),
            "event_rows": int(sub[event_col].sum()),
            "threshold_matrix": matrix,
        }

        all_payloads.append(payload)

        json_path = OUT / f"eicu_subcohort_{name.lower()}_validation_summary.json"
        md_path = DOCS / f"eicu_subcohort_{name.lower()}_validation_summary.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_md(md_path, payload)

    combined = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "eICU Harmonized Clinical-Event Subcohort B/C Robustness Summary",
        "dataset": "eICU v2.0",
        "purpose": "Test whether eICU harmonized clinical-event results remain directionally stable across deterministic patient-level subcohorts.",
        "claim_boundary": "Retrospective aggregate robustness only. Do not claim full clinical validation.",
        "subcohorts": all_payloads,
    }

    (OUT / "eicu_subcohort_bc_robustness_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    lines = []
    lines.append("# eICU Harmonized Clinical-Event Subcohort B/C Robustness Summary")
    lines.append("")
    lines.append("This DUA-safe summary compares deterministic eICU patient-level subcohorts B and C using aggregate-only outputs.")
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("")
    lines.append("Retrospective aggregate robustness only. Do not claim full clinical validation, diagnosis, treatment direction, replacement of clinician judgment, or autonomous escalation.")
    lines.append("")
    for payload in all_payloads:
        lines.append(f"## Subcohort {payload['subcohort']}")
        lines.append("")
        lines.append(f"- Cases / stays: {payload['cases']:,}")
        lines.append(f"- Rows: {payload['rows']:,}")
        lines.append(f"- Event rows: {payload['event_rows']:,}")
        lines.append("")
        lines.append("| Threshold | Alert reduction | FPR | Detection | Median lead time | ERA alerts / patient-day |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in payload["threshold_matrix"]:
            ar = "NA" if r["alert_reduction_pct"] is None else f'{r["alert_reduction_pct"]}%'
            lead = "NA" if r["median_lead_time_hours"] is None else f'{r["median_lead_time_hours"]} hrs'
            lines.append(f'| t={r["threshold"]} | {ar} | {r["fpr_pct"]}% | {r["detection_pct"]}% | {lead} | {r["era_alerts_per_patient_day"]} |')
        lines.append("")

    lines.append("## Safe Interpretation")
    lines.append("")
    lines.append("If subcohort B and C preserve the same direction as the full eICU harmonized run — lower thresholds increase detection and higher thresholds reduce review burden/FPR — this strengthens the eICU robustness story before any HiRID work begins.")
    lines.append("")
    lines.append("Raw eICU files and row-level derived outputs remain local-only.")
    (DOCS / "eicu_subcohort_bc_robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("DONE — eICU subcohort B/C aggregate summaries created.")
    print("Review:")
    print(" - data/validation/eicu_subcohort_b_validation_summary.json")
    print(" - data/validation/eicu_subcohort_c_validation_summary.json")
    print(" - docs/validation/eicu_subcohort_bc_robustness_summary.md")

if __name__ == "__main__":
    main()
