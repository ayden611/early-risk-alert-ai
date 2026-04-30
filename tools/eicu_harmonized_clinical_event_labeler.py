#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit("ERROR: pandas is required for this local eICU labeling script.") from exc


DEFAULT_INPUTS = [
    Path("data/validation/local_private/eicu/working/eicu_era_cohort.csv"),
    Path("data/validation/local_private/eicu/eicu_era_cohort.csv"),
    Path("data/validation/eicu_era_cohort.csv"),
]

LOCAL_DIR = Path("data/validation/local_private/eicu/harmonized")
PUBLIC_SUMMARY = Path("data/validation/eicu_harmonized_clinical_event_summary.json")
PUBLIC_DOC = Path("docs/validation/eicu_harmonized_clinical_event_summary.md")


def clean_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def find_col(df, candidates):
    normalized = {clean_col(c): c for c in df.columns}
    for cand in candidates:
        key = clean_col(cand)
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        cc = clean_col(c)
        for cand in candidates:
            if clean_col(cand) in cc:
                return c
    return None


def numeric_series(df, col):
    if col is None:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def choose_input(cli_input: str | None) -> Path:
    if cli_input:
        p = Path(cli_input)
        if not p.exists():
            raise SystemExit(f"ERROR: input file not found: {p}")
        return p

    for p in DEFAULT_INPUTS:
        if p.exists():
            return p

    raise SystemExit(
        "ERROR: No eICU ERA cohort found. Expected one of:\n"
        + "\n".join(f" - {p}" for p in DEFAULT_INPUTS)
        + "\nUse --input /path/to/eicu_era_cohort.csv if needed."
    )


def infer_minutes(df, time_col, patient_col):
    raw = df[time_col]

    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric.astype(float)

    dt = pd.to_datetime(raw, errors="coerce", utc=True)
    if dt.notna().mean() > 0.8:
        temp = pd.DataFrame({"pid": df[patient_col], "dt": dt})
        first = temp.groupby("pid")["dt"].transform("min")
        return ((dt - first).dt.total_seconds() / 60.0).astype(float)

    temp = pd.DataFrame({"pid": df[patient_col]})
    order = temp.groupby("pid").cumcount()
    return order.astype(float) * 5.0


def classify_event_type(row):
    parts = []
    if row.get("flag_spo2_extreme", False):
        parts.append("severe oxygenation decline")
    if row.get("flag_bp_extreme", False):
        parts.append("severe hemodynamic instability")
    if row.get("flag_hr_extreme", False):
        parts.append("severe heart-rate instability")
    if row.get("flag_rr_extreme", False):
        parts.append("severe respiratory-rate instability")
    if row.get("flag_temp_extreme", False):
        parts.append("severe temperature instability")
    if row.get("severe_signal_count", 0) >= 2:
        parts.append("multi-signal deterioration pattern")
    return "; ".join(parts[:3]) if parts else ""


def build_event_labels(df, patient_col, minute_col, gap_minutes):
    df = df.sort_values([patient_col, minute_col], kind="mergesort").copy()
    df["harmonized_clinical_event"] = 0
    df["event_label"] = 0
    df["actual_event"] = 0
    df["event_cluster_id"] = ""
    df["event_type"] = ""

    event_counter = 0

    for pid, g in df.groupby(patient_col, sort=False):
        last_event_time = None
        candidate = g[g["event_candidate"] == True]

        for idx, row in candidate.iterrows():
            t = row[minute_col]
            if pd.isna(t):
                continue

            if last_event_time is None or (float(t) - float(last_event_time)) >= gap_minutes:
                event_counter += 1
                cluster_id = f"EICU-HARMONIZED-{event_counter:06d}"
                etype = classify_event_type(row)

                df.loc[idx, "harmonized_clinical_event"] = 1
                df.loc[idx, "event_label"] = 1
                df.loc[idx, "actual_event"] = 1
                df.loc[idx, "event_cluster_id"] = cluster_id
                df.loc[idx, "event_type"] = etype

                last_event_time = float(t)

    return df


def compute_validation_matrix(df, patient_col, minute_col, thresholds, window_hours):
    window_minutes = float(window_hours) * 60.0

    risk_col = find_col(df, [
        "risk_score", "era_risk_score", "score", "review_score", "risk"
    ])

    if risk_col is None:
        raise SystemExit(
            "ERROR: No risk_score-like column found in eICU ERA cohort. "
            "The cohort needs real ERA scores before threshold validation."
        )

    risk = numeric_series(df, risk_col)

    standard_alert = (
        (df["spo2"].notna() & (df["spo2"] <= 90)) |
        (df["hr"].notna() & ((df["hr"] >= 120) | (df["hr"] <= 50))) |
        (df["rr"].notna() & ((df["rr"] >= 28) | (df["rr"] <= 10))) |
        (df["sbp"].notna() & (df["sbp"] <= 100)) |
        (df["map_bp"].notna() & (df["map_bp"] <= 65)) |
        (df["temp"].notna() & ((df["temp"] >= 39.0) | (df["temp"] <= 35.5)))
    )

    spans = df.groupby(patient_col)[minute_col].agg(["min", "max", "count"])
    observed_minutes = 0.0
    for _, row in spans.iterrows():
        if pd.notna(row["min"]) and pd.notna(row["max"]) and row["max"] > row["min"]:
            observed_minutes += float(row["max"] - row["min"] + 5.0)
        else:
            observed_minutes += float(row["count"]) * 5.0

    patient_days = max(observed_minutes / 1440.0, 1e-9)

    events = df[df["harmonized_clinical_event"] == 1][[patient_col, minute_col, "event_type"]].copy()
    total_events = int(len(events))

    matrix = []

    grouped = {}
    base = df[[patient_col, minute_col]].copy()
    base["_risk"] = risk
    base["_standard_alert"] = standard_alert.astype(bool)
    for pid, g in base.groupby(patient_col, sort=False):
        grouped[pid] = g.sort_values(minute_col)

    for threshold in thresholds:
        era_alert = risk >= float(threshold)
        era_alert_count = int(era_alert.sum())
        standard_alert_count = int(standard_alert.sum())

        era_alerts_per_day = era_alert_count / patient_days
        standard_alerts_per_day = standard_alert_count / patient_days
        alert_reduction = 0.0
        if standard_alerts_per_day > 0:
            alert_reduction = (1.0 - (era_alerts_per_day / standard_alerts_per_day)) * 100.0

        detected = 0
        lead_hours = []

        alert_df = df[[patient_col, minute_col]].copy()
        alert_df["_era_alert"] = era_alert.astype(bool)
        alert_groups = {
            pid: g[g["_era_alert"] == True][minute_col].dropna().astype(float).tolist()
            for pid, g in alert_df.groupby(patient_col, sort=False)
        }

        in_event_window = pd.Series(False, index=df.index)

        for event_idx, ev in events.iterrows():
            pid = ev[patient_col]
            event_time = ev[minute_col]
            if pd.isna(event_time):
                continue
            event_time = float(event_time)
            start = event_time - window_minutes

            alerts = alert_groups.get(pid, [])
            prior_alerts = [a for a in alerts if start <= a < event_time]
            if prior_alerts:
                detected += 1
                first_alert = min(prior_alerts)
                lead_hours.append((event_time - first_alert) / 60.0)

            same_pid = df[patient_col] == pid
            t = df[minute_col].astype(float)
            in_event_window = in_event_window | (same_pid & (t >= start) & (t < event_time))

        detection_pct = (detected / total_events * 100.0) if total_events else 0.0
        median_lead = float(pd.Series(lead_hours).median()) if lead_hours else None

        outside_window_rows = ~in_event_window
        false_alert_rows = era_alert & outside_window_rows
        fpr_den = int(outside_window_rows.sum())
        fpr_pct = (int(false_alert_rows.sum()) / fpr_den * 100.0) if fpr_den else 0.0

        matrix.append({
            "threshold": float(threshold),
            "alert_reduction_pct": round(alert_reduction, 2),
            "fpr_pct_outside_event_windows": round(fpr_pct, 2),
            "event_detection_pct": round(detection_pct, 2),
            "events_detected": int(detected),
            "events_total": int(total_events),
            "median_lead_time_hours": round(median_lead, 2) if median_lead is not None else None,
            "era_alerts_per_patient_day": round(era_alerts_per_day, 4),
            "standard_alerts_per_patient_day": round(standard_alerts_per_day, 4),
            "era_alert_rows": era_alert_count,
            "standard_threshold_alert_rows": standard_alert_count,
        })

    return matrix, risk_col, round(patient_days, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=str(LOCAL_DIR / "eicu_harmonized_clinical_event_labeled_era_cohort.csv"))
    ap.add_argument("--summary-json", default=str(PUBLIC_SUMMARY))
    ap.add_argument("--summary-md", default=str(PUBLIC_DOC))
    ap.add_argument("--thresholds", default="4.0,5.0,6.0")
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--event-gap-hours", type=float, default=6.0)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    input_path = choose_input(args.input)
    out_path = Path(args.output)
    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"READING eICU ERA cohort: {input_path}")
    if args.max_rows and args.max_rows > 0:
        df = pd.read_csv(input_path, nrows=args.max_rows, low_memory=False)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    if df.empty:
        raise SystemExit("ERROR: input cohort is empty.")

    patient_col = find_col(df, [
        "patientunitstayid", "patient_id", "patient", "stay_id",
        "unitstayid", "encounter_id", "case_id"
    ])

    time_col = find_col(df, [
        "chartoffset", "observationoffset", "offset", "minutes",
        "chart_time", "time", "timestamp", "hours_since_admit"
    ])

    if patient_col is None:
        raise SystemExit("ERROR: Could not find patient/stay identifier column.")
    if time_col is None:
        raise SystemExit("ERROR: Could not find time/offset column.")

    print(f"USING patient column: {patient_col}")
    print(f"USING time column: {time_col}")

    minute_col = "__era_minutes_from_start"
    df[minute_col] = infer_minutes(df, time_col, patient_col)

    spo2_col = find_col(df, ["spo2", "sao2", "oxygen_saturation", "o2sat", "oxygensaturation"])
    hr_col = find_col(df, ["heartrate", "heart_rate", "hr", "pulse"])
    rr_col = find_col(df, ["respiration", "respiratoryrate", "respiratory_rate", "rr"])
    sbp_col = find_col(df, ["systemicsystolic", "nibp_systolic", "systolic", "sbp"])
    dbp_col = find_col(df, ["systemicdiastolic", "nibp_diastolic", "diastolic", "dbp"])
    map_col = find_col(df, ["systemicmean", "nibp_mean", "meanbp", "map", "map_bp"])
    temp_col = find_col(df, ["temperature", "temp"])

    df["spo2"] = numeric_series(df, spo2_col)
    df["hr"] = numeric_series(df, hr_col)
    df["rr"] = numeric_series(df, rr_col)
    df["sbp"] = numeric_series(df, sbp_col)
    df["dbp"] = numeric_series(df, dbp_col)
    df["map_bp"] = numeric_series(df, map_col)
    df["temp"] = numeric_series(df, temp_col)

    df["flag_spo2_severe"] = df["spo2"].notna() & (df["spo2"] <= 88)
    df["flag_spo2_extreme"] = df["spo2"].notna() & (df["spo2"] <= 85)

    df["flag_hr_severe"] = df["hr"].notna() & ((df["hr"] >= 130) | (df["hr"] <= 40))
    df["flag_hr_extreme"] = df["hr"].notna() & ((df["hr"] >= 150) | (df["hr"] <= 35))

    df["flag_rr_severe"] = df["rr"].notna() & ((df["rr"] >= 30) | (df["rr"] <= 8))
    df["flag_rr_extreme"] = df["rr"].notna() & ((df["rr"] >= 35) | (df["rr"] <= 6))

    df["flag_bp_severe"] = (
        (df["sbp"].notna() & (df["sbp"] <= 90)) |
        (df["map_bp"].notna() & (df["map_bp"] <= 60))
    )
    df["flag_bp_extreme"] = (
        (df["sbp"].notna() & (df["sbp"] <= 80)) |
        (df["map_bp"].notna() & (df["map_bp"] <= 55))
    )

    df["flag_temp_severe"] = df["temp"].notna() & ((df["temp"] >= 39.5) | (df["temp"] <= 35.0))
    df["flag_temp_extreme"] = df["temp"].notna() & ((df["temp"] >= 40.0) | (df["temp"] <= 34.5))

    severe_flags = [
        "flag_spo2_severe",
        "flag_hr_severe",
        "flag_rr_severe",
        "flag_bp_severe",
        "flag_temp_severe",
    ]

    extreme_flags = [
        "flag_spo2_extreme",
        "flag_hr_extreme",
        "flag_rr_extreme",
        "flag_bp_extreme",
        "flag_temp_extreme",
    ]

    df["severe_signal_count"] = df[severe_flags].sum(axis=1)
    df["extreme_signal_count"] = df[extreme_flags].sum(axis=1)

    df["event_candidate"] = (
        (df["extreme_signal_count"] >= 1) |
        (df["severe_signal_count"] >= 2)
    )

    gap_minutes = args.event_gap_hours * 60.0
    labeled = build_event_labels(df, patient_col, minute_col, gap_minutes)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    matrix, risk_col, patient_days = compute_validation_matrix(
        labeled,
        patient_col=patient_col,
        minute_col=minute_col,
        thresholds=thresholds,
        window_hours=args.window_hours,
    )

    labeled.to_csv(out_path, index=False)
    print(f"WROTE LOCAL ROW-LEVEL LABELED COHORT: {out_path}")

    total_rows = int(len(labeled))
    total_patients = int(labeled[patient_col].nunique())
    total_events = int(labeled["harmonized_clinical_event"].sum())

    signal_counts = {
        "oxygenation_event_rows": int(labeled["flag_spo2_severe"].sum()),
        "heart_rate_event_rows": int(labeled["flag_hr_severe"].sum()),
        "respiratory_rate_event_rows": int(labeled["flag_rr_severe"].sum()),
        "blood_pressure_event_rows": int(labeled["flag_bp_severe"].sum()),
        "temperature_event_rows": int(labeled["flag_temp_severe"].sum()),
        "multi_signal_candidate_rows": int((labeled["severe_signal_count"] >= 2).sum()),
    }

    summary = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "eICU",
        "validation_type": "harmonized clinical-event labeling attempt",
        "public_safety": "Aggregate summary only. Row-level eICU files remain local-only.",
        "input_rows": total_rows,
        "patients_or_stays": total_patients,
        "harmonized_clinical_event_clusters": total_events,
        "event_gap_hours": args.event_gap_hours,
        "pre_event_window_hours": args.window_hours,
        "risk_score_column_used": str(risk_col),
        "event_definition": {
            "summary": "Clinical-event labels are generated from severe physiologic deterioration patterns rather than mortality/discharge outcome proxy.",
            "candidate_rule": "An event candidate is created when at least one extreme physiologic signal is present or at least two severe physiologic signals occur together.",
            "cluster_rule": "Candidate rows are collapsed into event clusters separated by the configured event-gap window.",
            "signals": [
                "oxygenation decline",
                "heart-rate instability",
                "respiratory-rate instability",
                "blood-pressure instability",
                "temperature instability"
            ],
        },
        "signal_counts_aggregate": signal_counts,
        "threshold_matrix": matrix,
        "interpretation_boundary": (
            "This is the first harmonized eICU clinical-event labeling pass. "
            "It is closer to the MIMIC-IV clinical-event question than the prior outcome-proxy check, "
            "but should be reviewed before using unqualified two-dataset validation language."
        ),
        "safe_public_claim": (
            "eICU now has a harmonized clinical-event labeling pass that can be compared against MIMIC-IV using the same threshold matrix and pre-event window."
        ),
        "claim_to_avoid_until_review": (
            "Do not claim fully validated on two datasets without qualification until event definitions, denominators, and metric logic are reviewed and locked."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"WROTE PUBLIC AGGREGATE SUMMARY: {summary_json}")

    lines = []
    lines.append("# eICU Harmonized Clinical-Event Labeling Summary")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at_utc']}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This run creates a harmonized eICU clinical-event labeling pass so eICU can be compared more directly to the MIMIC-IV clinical-event validation question.")
    lines.append("")
    lines.append("## DUA Safety")
    lines.append("")
    lines.append("- Public output is aggregate-only.")
    lines.append("- Row-level eICU files remain local-only.")
    lines.append("- No raw restricted files or patient-level rows should be committed.")
    lines.append("")
    lines.append("## Cohort Summary")
    lines.append("")
    lines.append(f"- Rows analyzed: {total_rows:,}")
    lines.append(f"- Patients/stays: {total_patients:,}")
    lines.append(f"- Harmonized clinical-event clusters: {total_events:,}")
    lines.append(f"- Event-gap window: {args.event_gap_hours:g} hours")
    lines.append(f"- Pre-event review window: {args.window_hours:g} hours")
    lines.append("")
    lines.append("## Event Definition")
    lines.append("")
    lines.append("An event candidate is created when at least one extreme physiologic signal is present or at least two severe physiologic signals occur together. Candidate rows are collapsed into event clusters separated by the configured event-gap window.")
    lines.append("")
    lines.append("## Threshold Matrix")
    lines.append("")
    lines.append("| Threshold | Alert reduction | FPR outside event windows | Detection | Events detected / total | Median lead time | ERA alerts / patient-day | Standard alerts / patient-day |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in matrix:
        med = "—" if r["median_lead_time_hours"] is None else f'{r["median_lead_time_hours"]} hrs'
        lines.append(
            f'| t={r["threshold"]:g} | {r["alert_reduction_pct"]}% | {r["fpr_pct_outside_event_windows"]}% | '
            f'{r["event_detection_pct"]}% | {r["events_detected"]}/{r["events_total"]} | {med} | '
            f'{r["era_alerts_per_patient_day"]} | {r["standard_alerts_per_patient_day"]} |'
        )
    lines.append("")
    lines.append("## Interpretation Boundary")
    lines.append("")
    lines.append("This is the first harmonized eICU clinical-event labeling pass. It is closer to the MIMIC-IV clinical-event question than the prior outcome-proxy check, but should be reviewed before using unqualified two-dataset validation language.")
    lines.append("")
    lines.append("## Safe Public Wording")
    lines.append("")
    lines.append("> eICU now has a harmonized clinical-event labeling pass that can be compared against MIMIC-IV using the same threshold matrix and pre-event window.")
    lines.append("")
    lines.append("## Wording to Avoid Until Review")
    lines.append("")
    lines.append("> Fully validated on two datasets.")
    lines.append("")
    lines.append("Use that only after the event definitions, denominators, and metric logic are reviewed and locked.")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE PUBLIC AGGREGATE DOC: {summary_md}")

    print("")
    print("SUMMARY")
    print("=======")
    print(f"Rows: {total_rows:,}")
    print(f"Patients/stays: {total_patients:,}")
    print(f"Harmonized clinical-event clusters: {total_events:,}")
    print("")
    print("Threshold matrix:")
    for r in matrix:
        print(
            f"t={r['threshold']:g} | alert_reduction={r['alert_reduction_pct']}% | "
            f"FPR={r['fpr_pct_outside_event_windows']}% | detection={r['event_detection_pct']}% | "
            f"lead={r['median_lead_time_hours']} hrs | "
            f"alerts/day={r['era_alerts_per_patient_day']}"
        )


if __name__ == "__main__":
    main()
