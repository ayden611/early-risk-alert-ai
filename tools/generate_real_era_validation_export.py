#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median


PATIENT_COLS = ["patient_id", "subject_id", "stay_id", "hadm_id"]
TIME_COLS = ["timestamp", "charttime", "time", "reading_time", "observed_at"]
EVENT_FLAG_COLS = ["clinical_event", "event", "event_flag", "label", "deterioration_event"]
EVENT_GROUP_COLS = ["event_group", "event_label", "event_id"]

VITAL_COLS = {
    "heart_rate": ["heart_rate", "hr"],
    "spo2": ["spo2", "oxygen_saturation", "o2_sat"],
    "bp_systolic": ["bp_systolic", "systolic_bp", "sbp"],
    "bp_diastolic": ["bp_diastolic", "diastolic_bp", "dbp"],
    "respiratory_rate": ["respiratory_rate", "rr", "resp_rate"],
    "temperature_f": ["temperature_f", "temp_f", "temperature"],
}


def norm(s):
    return (s or "").strip().lower()


def find_col(fieldnames, candidates):
    lookup = {norm(c): c for c in fieldnames}
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None


def parse_float(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_bool(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t", "event", "positive", "alert"}


def parse_time(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    return None


def hour_bucket(dt):
    if not dt:
        return ""
    return dt.replace(minute=0, second=0, microsecond=0).isoformat(sep=" ")


def priority_tier(score):
    if score >= 7.0:
        return "Critical"
    if score >= 6.0:
        return "Elevated"
    if score >= 4.0:
        return "Watch"
    return "Low"


def standard_threshold_alert(row, cols):
    hr = parse_float(row.get(cols.get("heart_rate")))
    spo2 = parse_float(row.get(cols.get("spo2")))
    sbp = parse_float(row.get(cols.get("bp_systolic")))
    dbp = parse_float(row.get(cols.get("bp_diastolic")))
    rr = parse_float(row.get(cols.get("respiratory_rate")))
    temp = parse_float(row.get(cols.get("temperature_f")))

    if spo2 is not None and spo2 < 95:
        return True
    if hr is not None and (hr >= 110 or hr <= 55):
        return True
    if sbp is not None and (sbp >= 160 or sbp < 100):
        return True
    if dbp is not None and dbp >= 110:
        return True
    if rr is not None and (rr > 20 or rr < 12):
        return True
    if temp is not None and (temp > 100.4 or temp < 97.6):
        return True
    return False


def score_row(row, cols, prev_row=None):
    score = 0.0
    drivers = []

    hr = parse_float(row.get(cols.get("heart_rate")))
    spo2 = parse_float(row.get(cols.get("spo2")))
    sbp = parse_float(row.get(cols.get("bp_systolic")))
    dbp = parse_float(row.get(cols.get("bp_diastolic")))
    rr = parse_float(row.get(cols.get("respiratory_rate")))
    temp = parse_float(row.get(cols.get("temperature_f")))

    if spo2 is not None:
        if spo2 < 88:
            score += 3.5
            drivers.append("severe SpO2 decline")
        elif spo2 < 90:
            score += 3.0
            drivers.append("SpO2 below 90")
        elif spo2 < 93:
            score += 2.0
            drivers.append("SpO2 decline")
        elif spo2 < 95:
            score += 1.0
            drivers.append("borderline SpO2")

    if hr is not None:
        if hr >= 130 or hr <= 40:
            score += 2.5
            drivers.append("severe HR instability")
        elif hr >= 120 or hr <= 50:
            score += 2.0
            drivers.append("HR instability")
        elif hr >= 110 or hr <= 55:
            score += 1.0
            drivers.append("HR elevation/depression")

    if sbp is not None:
        if sbp < 85 or sbp >= 190:
            score += 2.5
            drivers.append("severe BP instability")
        elif sbp < 90 or sbp >= 180:
            score += 2.0
            drivers.append("BP instability")
        elif sbp < 100 or sbp >= 160:
            score += 1.0
            drivers.append("BP trend concern")

    if dbp is not None:
        if dbp >= 120:
            score += 1.5
            drivers.append("severe diastolic BP elevation")
        elif dbp >= 110:
            score += 1.0
            drivers.append("diastolic BP elevation")

    if rr is not None:
        if rr >= 30 or rr <= 8:
            score += 2.5
            drivers.append("severe RR instability")
        elif rr >= 24 or rr <= 10:
            score += 1.5
            drivers.append("RR instability")
        elif rr > 20:
            score += 0.5
            drivers.append("RR elevation")

    if temp is not None:
        if temp >= 102.2 or temp <= 95.9:
            score += 1.5
            drivers.append("temperature instability")
        elif temp >= 101.0 or temp <= 96.8:
            score += 1.0
            drivers.append("temperature trend concern")

    if prev_row:
        prev_spo2 = parse_float(prev_row.get(cols.get("spo2")))
        prev_hr = parse_float(prev_row.get(cols.get("heart_rate")))
        prev_sbp = parse_float(prev_row.get(cols.get("bp_systolic")))
        prev_rr = parse_float(prev_row.get(cols.get("respiratory_rate")))

        if spo2 is not None and prev_spo2 is not None:
            if prev_spo2 - spo2 >= 5:
                score += 2.0
                drivers.append("rapid SpO2 worsening")
            elif prev_spo2 - spo2 >= 3:
                score += 1.0
                drivers.append("SpO2 worsening trend")

        if hr is not None and prev_hr is not None and abs(hr - prev_hr) >= 20:
            score += 1.0
            drivers.append("rapid HR change")

        if sbp is not None and prev_sbp is not None and prev_sbp - sbp >= 20:
            score += 1.0
            drivers.append("systolic BP drop")

        if rr is not None and prev_rr is not None and rr - prev_rr >= 5:
            score += 1.0
            drivers.append("RR worsening trend")

    score = min(round(score, 2), 9.9)

    if not drivers:
        primary = "Composite multi-signal pattern"
    else:
        primary = drivers[0]

    trend_terms = " ".join(drivers).lower()
    if any(x in trend_terms for x in ["worsening", "decline", "drop", "instability", "severe"]):
        trend = "Worsening"
    elif score >= 6:
        trend = "Worsening"
    elif score >= 4:
        trend = "Elevated"
    else:
        trend = "Stable"

    return score, primary, trend


def collapse_event_times(rows, patient_col, time_col, event_col, event_group_col, event_gap_hours):
    by_patient = {}

    if event_group_col:
        grouped = {}
        for row in rows:
            if not parse_bool(row.get(event_col)):
                continue
            pid = str(row.get(patient_col, "")).strip()
            ts = row.get("_parsed_time")
            if not pid or not ts:
                continue
            group = str(row.get(event_group_col, "")).strip()
            key = (pid, group if group else ts.isoformat())
            if key not in grouped or ts < grouped[key]:
                grouped[key] = ts

        for (pid, _), ts in grouped.items():
            by_patient.setdefault(pid, []).append(ts)

        return {pid: sorted(times) for pid, times in by_patient.items()}

    gap = timedelta(hours=event_gap_hours)
    raw = {}

    for row in rows:
        if not parse_bool(row.get(event_col)):
            continue
        pid = str(row.get(patient_col, "")).strip()
        ts = row.get("_parsed_time")
        if pid and ts:
            raw.setdefault(pid, []).append(ts)

    for pid, times in raw.items():
        times = sorted(set(times))
        collapsed = []
        for t in times:
            if not collapsed or t - collapsed[-1] > gap:
                collapsed.append(t)
        by_patient[pid] = collapsed

    return by_patient


def pct(num, den):
    if not den:
        return None
    return round(100.0 * num / den, 1)


def compute_metrics(rows, patient_col, time_col, event_col, event_group_col, threshold, window_hours, event_gap_hours):
    """
    Optimized metric calculation.

    Previous version repeatedly scanned all rows for every patient/event.
    This version indexes rows by patient once, then evaluates event windows only
    within that patient's rows.
    """
    by_patient = {}
    for r in rows:
        pid = str(r.get(patient_col, "")).strip()
        if pid:
            by_patient.setdefault(pid, []).append(r)

    for prs in by_patient.values():
        prs.sort(key=lambda r: r.get("_parsed_time") or datetime.min)

    event_times_by_patient = collapse_event_times(
        rows,
        patient_col,
        time_col,
        event_col,
        event_group_col,
        event_gap_hours
    )

    event_patients = set(event_times_by_patient.keys())
    total_event_clusters = sum(len(v) for v in event_times_by_patient.values())
    window = timedelta(hours=window_hours)

    event_window_row_ids = set()
    detected_event_patients = set()
    detected_event_clusters = 0
    lead_hours = []
    examples = []

    for pid, events in event_times_by_patient.items():
        patient_rows = by_patient.get(pid, [])
        if not patient_rows:
            continue

        for ev_time in events:
            candidates = []

            for r in patient_rows:
                ts = r.get("_parsed_time")
                if not ts:
                    continue

                if ev_time - window <= ts < ev_time:
                    event_window_row_ids.add(id(r))
                    if r.get("_era_alert"):
                        candidates.append(r)

            if not candidates:
                continue

            detected_event_patients.add(pid)
            detected_event_clusters += 1

            first_alert = min(candidates, key=lambda r: r.get("_parsed_time") or ev_time)
            lead = round((ev_time - first_alert["_parsed_time"]).total_seconds() / 3600.0, 4)
            lead_hours.append(lead)

            if len(examples) < 40:
                examples.append({
                    "patient_id": pid,
                    "event_time": ev_time.isoformat(sep=" "),
                    "first_alert_time": first_alert["_parsed_time"].isoformat(sep=" "),
                    "lead_hours": round(lead, 1),
                    "priority_tier": first_alert.get("priority_tier"),
                    "primary_driver": first_alert.get("primary_driver"),
                    "trend_direction": first_alert.get("trend_direction"),
                    "score": first_alert.get("risk_score"),
                    "queue_rank": first_alert.get("queue_rank"),
                })

    era_alert_rows = [r for r in rows if r.get("_era_alert")]
    standard_alert_rows = [r for r in rows if r.get("_standard_alert")]

    event_window_count = len(event_window_row_ids)
    non_event_count = max(len(rows) - event_window_count, 0)

    era_event_alert_count = sum(1 for r in era_alert_rows if id(r) in event_window_row_ids)
    standard_event_alert_count = sum(1 for r in standard_alert_rows if id(r) in event_window_row_ids)

    era_false_alert_count = sum(1 for r in era_alert_rows if id(r) not in event_window_row_ids)
    standard_false_alert_count = sum(1 for r in standard_alert_rows if id(r) not in event_window_row_ids)

    patient_days = 0.0
    for prs in by_patient.values():
        times = sorted(r.get("_parsed_time") for r in prs if r.get("_parsed_time"))
        if not times:
            continue

        span_hours = (times[-1] - times[0]).total_seconds() / 3600.0
        patient_days += max(span_hours / 24.0, 1.0 / 24.0)

    era_total_alerts = len(era_alert_rows)
    standard_total_alerts = len(standard_alert_rows)

    return {
        "threshold": threshold,
        "rows": len(rows),
        "patients": len(by_patient),
        "events": total_event_clusters,
        "event_patients": len(event_patients),
        "era_total_alerts": era_total_alerts,
        "standard_total_alerts": standard_total_alerts,
        "alert_reduction_pct": round(100.0 * (1.0 - (era_total_alerts / standard_total_alerts)), 1) if standard_total_alerts else None,
        "era_fpr_pct": pct(era_false_alert_count, non_event_count),
        "standard_fpr_pct": pct(standard_false_alert_count, non_event_count),
        "era_patient_detection_pct": pct(len(detected_event_patients), len(event_patients)),
        "era_event_cluster_detection_pct": pct(detected_event_clusters, total_event_clusters),
        "era_reading_sensitivity_pct": pct(era_event_alert_count, event_window_count),
        "standard_reading_sensitivity_pct": pct(standard_event_alert_count, event_window_count),
        "median_lead_time_hours": round(median(lead_hours), 2) if lead_hours else None,
        "pct_detected_events_flagged_over_1h": pct(sum(1 for x in lead_hours if x >= 1), len(lead_hours)),
        "pct_detected_events_flagged_over_2h": pct(sum(1 for x in lead_hours if x >= 2), len(lead_hours)),
        "pct_detected_events_flagged_over_3h": pct(sum(1 for x in lead_hours if x >= 3), len(lead_hours)),
        "total_patient_days": round(patient_days, 3),
        "era_alerts_per_patient_day": round(era_total_alerts / patient_days, 4) if patient_days else None,
        "standard_alerts_per_patient_day": round(standard_total_alerts / patient_days, 4) if patient_days else None,
        "detected_examples": sorted(examples, key=lambda x: float(x.get("score") or 0), reverse=True)[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--event-gap-hours", type=float, default=6.0)
    ap.add_argument("--dataset-name", default="MIMIC-IV de-identified retrospective validation subset")
    ap.add_argument("--out-csv", default="data/validation/latest_era_validation_export.csv")
    ap.add_argument("--out-summary", default="data/validation/latest_era_validation_summary.json")
    ap.add_argument("--registry", default="data/validation/validation_run_registry.json")
    ap.add_argument("--milestone", default="data/validation/mimic_validation_milestone_2026_04.json")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    patient_col = find_col(fieldnames, PATIENT_COLS)
    time_col = find_col(fieldnames, TIME_COLS)
    event_col = find_col(fieldnames, EVENT_FLAG_COLS)
    event_group_col = find_col(fieldnames, EVENT_GROUP_COLS)

    missing = []
    if not patient_col:
        missing.append("patient_id / subject_id / stay_id")
    if not time_col:
        missing.append("timestamp / charttime")
    if not event_col:
        missing.append("clinical_event / event flag")

    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing) + "\nFound columns: " + ", ".join(fieldnames))

    vital_cols = {key: find_col(fieldnames, candidates) for key, candidates in VITAL_COLS.items()}

    by_patient = {}
    for row in rows:
        pid = str(row.get(patient_col, "")).strip()
        ts = parse_time(row.get(time_col))
        row["_parsed_time"] = ts
        if pid and ts:
            by_patient.setdefault(pid, []).append(row)

    scored_rows = []
    for pid, prs in by_patient.items():
        prs.sort(key=lambda r: r["_parsed_time"])
        first_crossing = None
        prev = None

        for row in prs:
            score, driver, trend = score_row(row, vital_cols, prev)
            era_alert = score >= args.threshold
            std_alert = standard_threshold_alert(row, vital_cols)

            if era_alert and first_crossing is None:
                first_crossing = row["_parsed_time"]

            row["risk_score"] = score
            row["era_alert"] = era_alert
            row["priority_tier"] = priority_tier(score)
            row["primary_driver"] = driver
            row["trend_direction"] = trend
            row["threshold_crossed_at"] = first_crossing.isoformat(sep=" ") if first_crossing else ""
            row["standard_threshold_alert"] = std_alert
            row["_era_alert"] = era_alert
            row["_standard_alert"] = std_alert
            row["_hour_bucket"] = hour_bucket(row["_parsed_time"])
            scored_rows.append(row)
            prev = row

    by_bucket = {}
    for r in scored_rows:
        by_bucket.setdefault(r["_hour_bucket"], []).append(r)

    for _, bucket_rows in by_bucket.items():
        bucket_rows.sort(key=lambda r: float(r.get("risk_score") or 0), reverse=True)
        for rank, r in enumerate(bucket_rows, start=1):
            r["queue_rank"] = rank

    thresholds = [4.0, 5.0, 6.0]
    threshold_results = {}

    for t in thresholds:
        for r in scored_rows:
            r["_era_alert"] = float(r.get("risk_score") or 0) >= t
        threshold_results[str(t)] = compute_metrics(
            scored_rows,
            patient_col,
            time_col,
            event_col,
            event_group_col,
            t,
            args.window_hours,
            args.event_gap_hours
        )

    for r in scored_rows:
        r["_era_alert"] = float(r.get("risk_score") or 0) >= args.threshold

    selected = threshold_results[str(float(args.threshold))]
    run_id = "era_mimic_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_t" + str(args.threshold).replace(".", "_")

    output_csv = Path(args.out_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    export_fields = list(fieldnames)
    for col in [
        "risk_score",
        "era_alert",
        "priority_tier",
        "primary_driver",
        "trend_direction",
        "threshold_crossed_at",
        "queue_rank",
        "standard_threshold_alert",
    ]:
        if col not in export_fields:
            export_fields.append(col)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=export_fields, extrasaction="ignore")
        writer.writeheader()
        for r in scored_rows:
            writer.writerow(r)

    summary = {
        "ok": True,
        "run_id": run_id,
        "dataset": args.dataset_name,
        "source_file": str(csv_path),
        "enriched_csv": str(output_csv),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "Generated for retrospective validation review",
        "threshold": args.threshold,
        "window_hours": args.window_hours,
        "event_gap_hours": args.event_gap_hours,
        "export_columns_added": [
            "risk_score",
            "era_alert",
            "priority_tier",
            "primary_driver",
            "trend_direction",
            "threshold_crossed_at",
            "queue_rank",
            "standard_threshold_alert",
        ],
        "selected_threshold_metrics": selected,
        "threshold_results": threshold_results,
        "pilot_safe_note": "Retrospective analysis only. Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.",
    }

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {"runs": []}
    else:
        registry = {"runs": []}

    entry = {
        "run_id": run_id,
        "dataset": args.dataset_name,
        "rows": selected.get("rows"),
        "patients": selected.get("patients"),
        "events": selected.get("events"),
        "threshold": args.threshold,
        "alert_reduction_pct": selected.get("alert_reduction_pct"),
        "fpr_pct": selected.get("era_fpr_pct"),
        "patient_detection_pct": selected.get("era_patient_detection_pct"),
        "median_lead_time_hours": selected.get("median_lead_time_hours"),
        "era_alerts_per_patient_day": selected.get("era_alerts_per_patient_day"),
        "standard_alerts_per_patient_day": selected.get("standard_alerts_per_patient_day"),
        "date_generated": summary["generated_at_utc"],
        "validation_status": summary["validation_status"],
        "summary_file": str(out_summary),
        "enriched_csv": str(output_csv),
    }

    registry.setdefault("runs", [])
    registry["runs"] = [entry] + [r for r in registry["runs"] if r.get("run_id") != run_id]
    registry["latest_run_id"] = run_id
    registry["updated_at_utc"] = summary["generated_at_utc"]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    milestone_path = Path(args.milestone)
    if milestone_path.exists():
        milestone = json.loads(milestone_path.read_text(encoding="utf-8"))
        milestone.setdefault("real_era_validation_export", {})
        milestone["real_era_validation_export"] = {
            "status": "Generated",
            "latest_run_id": run_id,
            "generated_at_utc": summary["generated_at_utc"],
            "enriched_csv": str(output_csv),
            "summary_file": str(out_summary),
            "columns_added": summary["export_columns_added"],
            "selected_threshold_metrics": selected,
        }

        milestone.setdefault("operational_alert_burden_comparison", {})
        milestone["operational_alert_burden_comparison"]["era_alerts_per_patient_day_exact"] = selected.get("era_alerts_per_patient_day")
        milestone["operational_alert_burden_comparison"]["standard_threshold_alerts_per_patient_day_exact"] = selected.get("standard_alerts_per_patient_day")
        milestone["operational_alert_burden_comparison"]["alert_reduction_pct_exact"] = selected.get("alert_reduction_pct")
        milestone["operational_alert_burden_comparison"]["pilot_safe_summary"] = (
            f"ERA generated approximately {selected.get('era_alerts_per_patient_day')} alerts per patient-day "
            f"versus approximately {selected.get('standard_alerts_per_patient_day')} alerts per patient-day under standard threshold-only alerting, "
            f"for an alert reduction of {selected.get('alert_reduction_pct')}%."
        )

        milestone.setdefault("validation_run_registry", {})
        milestone["validation_run_registry"] = {
            "status": "Available",
            "registry_file": str(registry_path),
            "latest_run_id": run_id,
        }

        milestone_path.write_text(json.dumps(milestone, indent=2), encoding="utf-8")

    docs = Path("docs/validation")
    docs.mkdir(parents=True, exist_ok=True)

    md = f"""# ERA Real Validation Export Run

## Run Summary

- Run ID: `{run_id}`
- Dataset: {args.dataset_name}
- Source file: `{csv_path}`
- Enriched CSV: `{output_csv}`
- Generated: {summary['generated_at_utc']}
- Threshold: {args.threshold}
- Rows: {selected.get('rows'):,}
- Patients: {selected.get('patients'):,}
- Events: {selected.get('events'):,}
- Alert reduction: {selected.get('alert_reduction_pct')}%
- ERA FPR: {selected.get('era_fpr_pct')}%
- Patient detection: {selected.get('era_patient_detection_pct')}%
- Median lead time: {selected.get('median_lead_time_hours')} hours
- ERA alerts per patient-day: {selected.get('era_alerts_per_patient_day')}
- Standard threshold alerts per patient-day: {selected.get('standard_alerts_per_patient_day')}

## Columns Added

- risk_score
- era_alert
- priority_tier
- primary_driver
- trend_direction
- threshold_crossed_at
- queue_rank
- standard_threshold_alert

## Pilot-Safe Note

Retrospective analysis only. Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

    (docs / "latest_real_era_validation_export.md").write_text(md, encoding="utf-8")


    # ERA_MIMIC_DUA_AUTO_SANITIZE_V1
    # Keep public validation artifacts DUA-safe after every run.
    try:
        import subprocess
        import sys
        sanitizer_path = Path(__file__).resolve().parent / "mimic_dua_public_safety.py"
        if sanitizer_path.exists():
            subprocess.run([sys.executable, str(sanitizer_path), "--quiet"], check=True)
    except Exception as e:
        print(f"WARNING: MIMIC DUA public sanitizer did not complete: {e}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
