#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median


PATIENT_COLS = ["patient_id", "subject_id", "stay_id", "hadm_id"]
TIME_COLS = ["timestamp", "charttime", "time", "reading_time", "observed_at"]
EVENT_FLAG_COLS = ["clinical_event", "event", "event_flag", "label", "deterioration_event"]
EVENT_GROUP_COLS = ["event_group", "event_label", "event_id"]
SCORE_COLS = ["risk_score", "era_score", "score", "risk"]
ALERT_COLS = ["era_alert", "alert", "alert_flag", "is_alert"]

VITAL_COLS = {
    "heart_rate": ["heart_rate", "hr"],
    "spo2": ["spo2", "oxygen_saturation", "o2_sat"],
    "bp_systolic": ["bp_systolic", "systolic_bp", "sbp"],
    "bp_diastolic": ["bp_diastolic", "diastolic_bp", "dbp"],
    "respiratory_rate": ["respiratory_rate", "rr", "resp_rate"],
    "temperature_f": ["temperature_f", "temp_f", "temperature"]
}


def norm(s):
    return (s or "").strip().lower()


def find_col(fieldnames, candidates):
    lower = {norm(c): c for c in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def parse_bool(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t", "positive", "event", "alert"}


def parse_float(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


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


def derive_score(row, cols, prev_row=None):
    score = 0.0
    drivers = []

    hr = parse_float(row.get(cols.get("heart_rate")))
    spo2 = parse_float(row.get(cols.get("spo2")))
    sbp = parse_float(row.get(cols.get("bp_systolic")))
    dbp = parse_float(row.get(cols.get("bp_diastolic")))
    rr = parse_float(row.get(cols.get("respiratory_rate")))
    temp_f = parse_float(row.get(cols.get("temperature_f")))

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

    if temp_f is not None:
        if temp_f >= 102.2 or temp_f <= 95.9:
            score += 1.5
            drivers.append("temperature instability")
        elif temp_f >= 101.0 or temp_f <= 96.8:
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

    if score >= 7:
        tier = "Critical"
    elif score >= 6:
        tier = "Elevated"
    elif score >= 4:
        tier = "Watch"
    else:
        tier = "Low"

    primary_driver = drivers[0] if drivers else "no dominant driver"

    return score, primary_driver, tier


def collapse_events_by_group(rows, patient_col, time_col, event_col, group_col):
    grouped = {}

    for row in rows:
        if not parse_bool(row.get(event_col)):
            continue

        pid = str(row.get(patient_col, "")).strip()
        ts = parse_time(row.get(time_col))
        if not pid or ts is None:
            continue

        if group_col and str(row.get(group_col, "")).strip():
            key = (pid, str(row.get(group_col)).strip())
        else:
            key = (pid, ts.isoformat())

        if key not in grouped or ts < grouped[key]:
            grouped[key] = ts

    by_patient = {}
    for (pid, _), ts in grouped.items():
        by_patient.setdefault(pid, []).append(ts)

    return {pid: sorted(times) for pid, times in by_patient.items()}


def collapse_events_by_gap(event_times, gap_hours):
    times = sorted(set(t for t in event_times if t))
    if not times:
        return []

    collapsed = []
    gap = timedelta(hours=gap_hours)

    for t in times:
        if not collapsed or t - collapsed[-1] > gap:
            collapsed.append(t)

    return collapsed


def pct(values, predicate):
    if not values:
        return None
    return round(100.0 * sum(1 for v in values if predicate(v)) / len(values), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--event-gap-hours", type=float, default=6.0)
    ap.add_argument("--out", default="data/validation/lead_time_before_event_t6.json")
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
    score_col = find_col(fieldnames, SCORE_COLS)
    alert_col = find_col(fieldnames, ALERT_COLS)

    vital_cols = {}
    for key, candidates in VITAL_COLS.items():
        vital_cols[key] = find_col(fieldnames, candidates)

    missing = []
    if not patient_col:
        missing.append("patient_id / subject_id / stay_id")
    if not time_col:
        missing.append("timestamp / charttime")
    if not event_col:
        missing.append("clinical_event / event flag")

    if missing:
        raise SystemExit(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nFound columns: "
            + ", ".join(fieldnames)
        )

    mode = "existing_era_alert_or_score"
    if not alert_col and not score_col:
        mode = "derived_vitals_proxy_score"

    by_patient_rows = {}
    for row in rows:
        pid = str(row.get(patient_col, "")).strip()
        ts = parse_time(row.get(time_col))
        if not pid or ts is None:
            continue
        row["_parsed_time"] = ts
        by_patient_rows.setdefault(pid, []).append(row)

    total_alerts = 0
    scored_rows = []
    driver_counts = {}
    tier_counts = {}

    for pid, patient_rows in by_patient_rows.items():
        patient_rows.sort(key=lambda r: r["_parsed_time"])
        prev = None

        for row in patient_rows:
            if alert_col:
                is_alert = parse_bool(row.get(alert_col))
                score = parse_float(row.get(score_col), args.threshold if is_alert else 0.0) if score_col else (args.threshold if is_alert else 0.0)
                primary_driver = "existing ERA alert flag"
                tier = "Elevated" if is_alert else "Low"
            elif score_col:
                score = parse_float(row.get(score_col), 0.0)
                is_alert = score >= args.threshold
                primary_driver = "existing ERA score"
                tier = "Critical" if score >= 7 else "Elevated" if score >= 6 else "Watch" if score >= 4 else "Low"
            else:
                score, primary_driver, tier = derive_score(row, vital_cols, prev)
                is_alert = score >= args.threshold

            row["_era_score_for_lead_time"] = score
            row["_era_alert_for_lead_time"] = is_alert
            row["_primary_driver"] = primary_driver
            row["_priority_tier"] = tier

            if is_alert:
                total_alerts += 1

            driver_counts[primary_driver] = driver_counts.get(primary_driver, 0) + 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            scored_rows.append(row)
            prev = row

    event_times_by_patient = {}

    if event_group_col:
        event_times_by_patient = collapse_events_by_group(
            scored_rows, patient_col, time_col, event_col, event_group_col
        )
    else:
        raw_events = {}
        for row in scored_rows:
            if parse_bool(row.get(event_col)):
                pid = str(row.get(patient_col, "")).strip()
                raw_events.setdefault(pid, []).append(row["_parsed_time"])

        for pid, times in raw_events.items():
            event_times_by_patient[pid] = collapse_events_by_gap(times, args.event_gap_hours)

    window = timedelta(hours=args.window_hours)

    event_patients = set(event_times_by_patient.keys())
    detected_event_patients = set()
    event_cluster_count = 0
    detected_event_cluster_count = 0

    event_level_first_leads = []
    patient_best_lead = {}
    detected_examples = []

    total_patient_days = 0.0

    for pid, patient_rows in by_patient_rows.items():
        patient_rows.sort(key=lambda r: r["_parsed_time"])
        if patient_rows:
            span_hours = (patient_rows[-1]["_parsed_time"] - patient_rows[0]["_parsed_time"]).total_seconds() / 3600.0
            total_patient_days += max(span_hours / 24.0, 1.0 / 24.0)

        alert_rows = [r for r in patient_rows if r.get("_era_alert_for_lead_time")]
        alert_times = [r["_parsed_time"] for r in alert_rows]

        for event_time in event_times_by_patient.get(pid, []):
            event_cluster_count += 1

            candidates = [
                r for r in alert_rows
                if event_time - window <= r["_parsed_time"] < event_time
            ]

            if not candidates:
                continue

            detected_event_cluster_count += 1
            detected_event_patients.add(pid)

            first_alert_row = min(candidates, key=lambda r: r["_parsed_time"])
            first_alert_time = first_alert_row["_parsed_time"]

            lead_hours = round((event_time - first_alert_time).total_seconds() / 3600.0, 4)
            event_level_first_leads.append(lead_hours)
            patient_best_lead[pid] = max(patient_best_lead.get(pid, 0.0), lead_hours)

            if len(detected_examples) < 20:
                detected_examples.append({
                    "patient_id": pid,
                    "event_time": event_time.isoformat(),
                    "first_alert_time": first_alert_time.isoformat(),
                    "lead_hours": lead_hours,
                    "score": first_alert_row.get("_era_score_for_lead_time"),
                    "priority_tier": first_alert_row.get("_priority_tier"),
                    "primary_driver": first_alert_row.get("_primary_driver")
                })

    patient_level_leads = list(patient_best_lead.values())

    median_first = round(median(event_level_first_leads), 2) if event_level_first_leads else None
    median_patient = round(median(patient_level_leads), 2) if patient_level_leads else None

    patient_detection_pct = (
        round(100.0 * len(detected_event_patients) / len(event_patients), 1)
        if event_patients else None
    )

    event_cluster_detection_pct = (
        round(100.0 * detected_event_cluster_count / event_cluster_count, 1)
        if event_cluster_count else None
    )

    alerts_per_patient_day = (
        round(total_alerts / total_patient_days, 4)
        if total_patient_days else None
    )

    alerts_per_event_patient = (
        round(total_alerts / len(event_patients), 4)
        if event_patients else None
    )

    result = {
        "ok": True,
        "source_file": str(csv_path),
        "calculation_mode": mode,
        "important_note": (
            "This CSV did not contain era_alert or risk_score columns, so the script used a transparent vitals-based proxy score."
            if mode == "derived_vitals_proxy_score"
            else "This calculation used existing ERA alert or ERA score columns from the source file."
        ),
        "threshold": args.threshold,
        "event_window_hours": args.window_hours,
        "event_gap_hours": args.event_gap_hours,
        "total_rows": len(rows),
        "total_patients": len(by_patient_rows),
        "total_alerts": total_alerts,
        "total_event_patients": len(event_patients),
        "detected_event_patients": len(detected_event_patients),
        "patient_level_detection_pct": patient_detection_pct,
        "total_event_clusters": event_cluster_count,
        "detected_event_clusters": detected_event_cluster_count,
        "event_cluster_detection_pct": event_cluster_detection_pct,
        "median_lead_time_before_event_hours": median_first,
        "median_patient_level_lead_time_hours": median_patient,
        "pct_detected_events_flagged_over_1h": pct(event_level_first_leads, lambda x: x >= 1.0),
        "pct_detected_events_flagged_over_2h": pct(event_level_first_leads, lambda x: x >= 2.0),
        "pct_detected_events_flagged_over_3h": pct(event_level_first_leads, lambda x: x >= 3.0),
        "pct_detected_patients_flagged_over_1h": pct(patient_level_leads, lambda x: x >= 1.0),
        "pct_detected_patients_flagged_over_2h": pct(patient_level_leads, lambda x: x >= 2.0),
        "pct_detected_patients_flagged_over_3h": pct(patient_level_leads, lambda x: x >= 3.0),
        "total_patient_days": round(total_patient_days, 3),
        "alerts_per_patient_day": alerts_per_patient_day,
        "alerts_per_event_patient": alerts_per_event_patient,
        "priority_tier_counts": tier_counts,
        "primary_driver_counts_top": dict(sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)[:12]),
        "detected_examples": detected_examples,
        "headline": None,
        "clinical_note": (
            "Lead time is calculated from the first ERA alert inside the configured pre-event window "
            "to the documented event timestamp. This is retrospective, de-identified analysis and "
            "requires independent clinical review."
        )
    }

    if median_first is not None:
        result["headline"] = (
            f"Among detected event clusters, ERA flags appeared a median of "
            f"{median_first} hours before documented event timestamps within the "
            f"{args.window_hours:g}-hour pre-event window."
        )
    else:
        result["headline"] = (
            "No pre-event ERA alerts were found inside the configured event window using the detected columns."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    docs_dir = Path("docs/validation")
    docs_dir.mkdir(parents=True, exist_ok=True)

    md = f"""# Lead Time Before Event — ERA Retrospective Validation

## Summary

{result['headline']}

## Calculation Mode

{result['calculation_mode']}

{result['important_note']}

## Configuration

- Source file: `{result['source_file']}`
- ERA threshold: {result['threshold']}
- Pre-event window: {result['event_window_hours']} hours
- Event clustering gap: {result['event_gap_hours']} hours

## Results

- Total rows: {result['total_rows']:,}
- Total patients: {result['total_patients']:,}
- Total event patients: {result['total_event_patients']:,}
- Detected event patients: {result['detected_event_patients']:,}
- Patient-level detection: {result['patient_level_detection_pct']}%
- Total event clusters: {result['total_event_clusters']:,}
- Detected event clusters: {result['detected_event_clusters']:,}
- Event-cluster detection: {result['event_cluster_detection_pct']}%
- Median lead time before event: {result['median_lead_time_before_event_hours']} hours
- Median patient-level lead time: {result['median_patient_level_lead_time_hours']} hours
- Detected events flagged over 1 hour before event: {result['pct_detected_events_flagged_over_1h']}%
- Detected events flagged over 2 hours before event: {result['pct_detected_events_flagged_over_2h']}%
- Detected events flagged over 3 hours before event: {result['pct_detected_events_flagged_over_3h']}%
- Alerts per patient-day: {result['alerts_per_patient_day']}
- Alerts per event patient: {result['alerts_per_event_patient']}

## Explainability Summary

Top primary drivers:

{chr(10).join('- ' + str(k) + ': ' + str(v) for k, v in result['primary_driver_counts_top'].items())}

Priority tier counts:

{chr(10).join('- ' + str(k) + ': ' + str(v) for k, v in result['priority_tier_counts'].items())}

## Pilot-Safe Interpretation

This metric should be framed as retrospective timing analysis on de-identified data. It supports pilot planning and clinical review, but does not establish prospective performance.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

    (docs_dir / "lead_time_before_event_t6.md").write_text(md, encoding="utf-8")

    milestone_path = Path("data/validation/mimic_validation_milestone_2026_04.json")
    if milestone_path.exists():
        milestone = json.loads(milestone_path.read_text(encoding="utf-8"))

        milestone.setdefault("computed_validation_metrics", {})
        milestone["computed_validation_metrics"]["lead_time_before_event"] = result

        milestone.setdefault("next_validation_metrics", {})
        milestone["next_validation_metrics"]["median_lead_time_before_event_hours"] = result["median_lead_time_before_event_hours"]
        milestone["next_validation_metrics"]["alerts_per_patient_day"] = result["alerts_per_patient_day"]
        milestone["next_validation_metrics"]["alerts_per_event_patient"] = result["alerts_per_event_patient"]
        milestone["next_validation_metrics"]["status"] = "Lead-time metrics computed from row-level retrospective file."
        milestone["lead_time_headline"] = result["headline"]
        milestone["lead_time_calculation_mode"] = result["calculation_mode"]
        milestone["lead_time_note"] = result["important_note"]

        milestone_path.write_text(json.dumps(milestone, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
