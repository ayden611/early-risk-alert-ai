#!/usr/bin/env python3
"""
Calculate validation intelligence metrics from a row-level ERA retrospective CSV.

Expected helpful columns:
- patient_id
- timestamp
- clinical_event OR event OR event_flag
- risk_score OR era_score
Optional:
- era_alert OR alert_flag

Usage:
python3 tools/calculate_validation_intelligence.py path/to/era_rows.csv --threshold 6.0 --window-hours 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--out", default="data/validation/latest_validation_intelligence_metrics.json")
    args = ap.parse_args()

    p = Path(args.csv_path)
    df = pd.read_csv(p)

    patient_col = first_existing(df, ["patient_id", "subject_id", "stay_id"])
    ts_col = first_existing(df, ["timestamp", "charttime", "time"])
    event_col = first_existing(df, ["clinical_event", "event", "event_flag", "label"])
    score_col = first_existing(df, ["risk_score", "era_score", "score"])
    alert_col = first_existing(df, ["era_alert", "alert_flag", "alert"])

    missing = [name for name, col in {
        "patient_id": patient_col,
        "timestamp": ts_col,
        "clinical_event/event": event_col,
        "risk_score": score_col,
    }.items() if col is None]

    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[patient_col, ts_col])
    df = df.sort_values([patient_col, ts_col])

    if alert_col:
        df["_era_alert"] = df[alert_col].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    else:
        df["_era_alert"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0) >= args.threshold

    df["_event"] = df[event_col].astype(str).str.lower().isin(["1", "true", "yes", "y"])

    # Alerts per patient-day
    total_alerts = int(df["_era_alert"].sum())
    patient_days = 0.0
    for _, g in df.groupby(patient_col):
        span_hours = (g[ts_col].max() - g[ts_col].min()).total_seconds() / 3600.0
        patient_days += max(span_hours / 24.0, 1 / 24.0)

    alerts_per_patient_day = total_alerts / patient_days if patient_days else None

    event_patients = df.loc[df["_event"], patient_col].nunique()
    alerts_per_event_patient = total_alerts / event_patients if event_patients else None

    # Median lead time before event:
    # for each event row, find the most recent ERA alert for same patient in the pre-event window.
    lead_hours = []
    window = pd.Timedelta(hours=args.window_hours)

    for pid, g in df.groupby(patient_col):
        alerts = g.loc[g["_era_alert"], ts_col].tolist()
        if not alerts:
            continue

        for event_time in g.loc[g["_event"], ts_col]:
            candidates = [a for a in alerts if event_time - window <= a <= event_time]
            if candidates:
                last_alert = max(candidates)
                lead_hours.append((event_time - last_alert).total_seconds() / 3600.0)

    median_lead_time = float(pd.Series(lead_hours).median()) if lead_hours else None

    result = {
        "source_file": str(p),
        "threshold": args.threshold,
        "event_window_hours": args.window_hours,
        "total_alerts": total_alerts,
        "total_patient_days": round(patient_days, 3),
        "alerts_per_patient_day": round(alerts_per_patient_day, 4) if alerts_per_patient_day is not None else None,
        "event_patients": int(event_patients),
        "alerts_per_event_patient": round(alerts_per_event_patient, 4) if alerts_per_event_patient is not None else None,
        "median_lead_time_before_event_hours": round(median_lead_time, 4) if median_lead_time is not None else None,
        "lead_time_events_with_prior_alert": len(lead_hours),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
