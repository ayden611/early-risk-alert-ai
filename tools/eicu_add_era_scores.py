#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = [
    Path("data/validation/local_private/eicu/working/eicu_era_cohort.csv"),
    Path("data/validation/local_private/eicu/eicu_era_cohort.csv"),
    Path("data/validation/eicu_era_cohort.csv"),
]

DEFAULT_OUTPUT = Path("data/validation/local_private/eicu/working/eicu_era_scored_harmonized_input.csv")
PUBLIC_AUDIT = Path("data/validation/eicu_local_scoring_adapter_audit_summary.json")


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


def numeric(df, col):
    # Return a true float NaN series when a signal column is missing.
    # This prevents pandas NAType casting errors when MAP or another optional signal is absent.
    if not col:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def choose_input(cli_input):
    if cli_input:
        p = Path(cli_input)
        if not p.exists():
            raise SystemExit(f"ERROR: input file not found: {p}")
        return p
    for p in DEFAULT_INPUTS:
        if p.exists():
            return p
    raise SystemExit(
        "ERROR: Could not find local eICU ERA cohort. Expected one of:\n"
        + "\n".join(f" - {p}" for p in DEFAULT_INPUTS)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--audit-json", default=str(PUBLIC_AUDIT))
    args = ap.parse_args()

    input_path = choose_input(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"READING LOCAL eICU COHORT: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    if df.empty:
        raise SystemExit("ERROR: eICU cohort is empty.")

    existing_score = find_col(df, ["risk_score", "era_risk_score", "score", "review_score"])
    if existing_score:
        print(f"Existing score column found: {existing_score}")
        if existing_score != "risk_score":
            df["risk_score"] = pd.to_numeric(df[existing_score], errors="coerce")
        df.to_csv(output_path, index=False)
        print(f"WROTE SCORED LOCAL COHORT: {output_path}")
        return

    patient_col = find_col(df, [
        "patient_id", "patientunitstayid", "unitstayid", "stay_id", "encounter_id", "case_id"
    ])
    time_col = find_col(df, [
        "timestamp", "chartoffset", "observationoffset", "offset", "minutes", "time", "chart_time"
    ])

    if not patient_col:
        raise SystemExit("ERROR: Could not identify patient/stay column.")
    if not time_col:
        raise SystemExit("ERROR: Could not identify time/timestamp column.")

    spo2_col = find_col(df, ["spo2", "sao2", "oxygen_saturation", "o2sat", "oxygensaturation"])
    hr_col = find_col(df, ["heartrate", "heart_rate", "hr", "pulse"])
    rr_col = find_col(df, ["respiration", "respiratoryrate", "respiratory_rate", "rr"])
    sbp_col = find_col(df, ["systemicsystolic", "nibp_systolic", "systolic", "sbp"])
    dbp_col = find_col(df, ["systemicdiastolic", "nibp_diastolic", "diastolic", "dbp"])
    map_col = find_col(df, ["systemicmean", "nibp_mean", "meanbp", "map", "map_bp"])
    temp_col = find_col(df, ["temperature", "temp"])

    print("Detected signal columns:")
    print(" - oxygenation:", spo2_col)
    print(" - heart rate:", hr_col)
    print(" - respiratory rate:", rr_col)
    print(" - systolic BP:", sbp_col)
    print(" - diastolic BP:", dbp_col)
    print(" - MAP:", map_col)
    print(" - temperature:", temp_col)

    if not any([spo2_col, hr_col, rr_col, sbp_col, map_col, temp_col]):
        print("Available columns:")
        for c in df.columns:
            print(" -", c)
        raise SystemExit("ERROR: No vital-sign columns found for ERA scoring.")

    df["_era_spo2"] = numeric(df, spo2_col)
    df["_era_hr"] = numeric(df, hr_col)
    df["_era_rr"] = numeric(df, rr_col)
    df["_era_sbp"] = numeric(df, sbp_col)
    df["_era_dbp"] = numeric(df, dbp_col)
    df["_era_map"] = numeric(df, map_col)

    temp_raw = numeric(df, temp_col)
    if temp_raw.notna().sum() and temp_raw.median(skipna=True) > 45:
        df["_era_temp_c"] = (temp_raw - 32.0) * (5.0 / 9.0)
    else:
        df["_era_temp_c"] = temp_raw

    missing_map = df["_era_map"].isna()
    estimated_map = (df["_era_sbp"] + (2 * df["_era_dbp"])) / 3.0
    df.loc[missing_map, "_era_map"] = estimated_map[missing_map]

    df["_original_order"] = range(len(df))
    df = df.sort_values([patient_col, time_col], kind="mergesort")

    for col in ["_era_spo2", "_era_hr", "_era_rr", "_era_sbp", "_era_map"]:
        df[col + "_prev"] = df.groupby(patient_col)[col].shift(1)
        df[col + "_delta"] = df[col] - df[col + "_prev"]

    score = pd.Series(0.5, index=df.index, dtype="float64")
    driver = pd.Series("No dominant driver", index=df.index, dtype="object")

    components = {}

    oxygen = pd.Series(0.0, index=df.index)
    oxygen += ((df["_era_spo2"].notna()) & (df["_era_spo2"] <= 92)).astype(float) * 1.0
    oxygen += ((df["_era_spo2"].notna()) & (df["_era_spo2"] <= 90)).astype(float) * 1.2
    oxygen += ((df["_era_spo2"].notna()) & (df["_era_spo2"] <= 88)).astype(float) * 1.0
    oxygen += ((df["_era_spo2"].notna()) & (df["_era_spo2"] <= 85)).astype(float) * 1.0
    oxygen += ((df["_era_spo2_delta"].notna()) & (df["_era_spo2_delta"] <= -3)).astype(float) * 0.8
    components["SpO2 decline"] = oxygen

    hr = pd.Series(0.0, index=df.index)
    hr += ((df["_era_hr"].notna()) & ((df["_era_hr"] >= 120) | (df["_era_hr"] <= 50))).astype(float) * 1.0
    hr += ((df["_era_hr"].notna()) & ((df["_era_hr"] >= 130) | (df["_era_hr"] <= 40))).astype(float) * 1.2
    hr += ((df["_era_hr"].notna()) & ((df["_era_hr"] >= 150) | (df["_era_hr"] <= 35))).astype(float) * 1.0
    hr += ((df["_era_hr_delta"].notna()) & (df["_era_hr_delta"].abs() >= 20)).astype(float) * 0.5
    components["HR instability"] = hr

    rr = pd.Series(0.0, index=df.index)
    rr += ((df["_era_rr"].notna()) & ((df["_era_rr"] >= 28) | (df["_era_rr"] <= 10))).astype(float) * 1.0
    rr += ((df["_era_rr"].notna()) & ((df["_era_rr"] >= 30) | (df["_era_rr"] <= 8))).astype(float) * 1.2
    rr += ((df["_era_rr"].notna()) & ((df["_era_rr"] >= 35) | (df["_era_rr"] <= 6))).astype(float) * 1.0
    rr += ((df["_era_rr_delta"].notna()) & (df["_era_rr_delta"].abs() >= 6)).astype(float) * 0.4
    components["RR instability"] = rr

    bp = pd.Series(0.0, index=df.index)
    bp += (
        ((df["_era_sbp"].notna()) & (df["_era_sbp"] <= 100)) |
        ((df["_era_map"].notna()) & (df["_era_map"] <= 65))
    ).astype(float) * 1.0
    bp += (
        ((df["_era_sbp"].notna()) & (df["_era_sbp"] <= 90)) |
        ((df["_era_map"].notna()) & (df["_era_map"] <= 60))
    ).astype(float) * 1.2
    bp += (
        ((df["_era_sbp"].notna()) & (df["_era_sbp"] <= 80)) |
        ((df["_era_map"].notna()) & (df["_era_map"] <= 55))
    ).astype(float) * 1.0
    bp += ((df["_era_sbp_delta"].notna()) & (df["_era_sbp_delta"] <= -15)).astype(float) * 0.7
    components["BP instability"] = bp

    temp = pd.Series(0.0, index=df.index)
    temp += ((df["_era_temp_c"].notna()) & ((df["_era_temp_c"] >= 39.0) | (df["_era_temp_c"] <= 35.5))).astype(float) * 0.7
    temp += ((df["_era_temp_c"].notna()) & ((df["_era_temp_c"] >= 40.0) | (df["_era_temp_c"] <= 34.5))).astype(float) * 0.8
    components["Temperature instability"] = temp

    comp_df = pd.DataFrame(components)
    score += comp_df.sum(axis=1)

    severe_count = (
        ((df["_era_spo2"].notna()) & (df["_era_spo2"] <= 90)).astype(int)
        + ((df["_era_hr"].notna()) & ((df["_era_hr"] >= 130) | (df["_era_hr"] <= 40))).astype(int)
        + ((df["_era_rr"].notna()) & ((df["_era_rr"] >= 30) | (df["_era_rr"] <= 8))).astype(int)
        + (
            ((df["_era_sbp"].notna()) & (df["_era_sbp"] <= 90)) |
            ((df["_era_map"].notna()) & (df["_era_map"] <= 60))
        ).astype(int)
        + ((df["_era_temp_c"].notna()) & ((df["_era_temp_c"] >= 39.5) | (df["_era_temp_c"] <= 35.0))).astype(int)
    )

    score += severe_count.clip(upper=3) * 0.45
    score = score.clip(lower=0, upper=10)

    df["risk_score"] = score.round(2)
    df["era_alert"] = df["risk_score"] >= 6.0

    df["primary_driver"] = comp_df.idxmax(axis=1)
    df.loc[comp_df.max(axis=1) <= 0, "primary_driver"] = "No dominant driver"

    df["priority_tier"] = "Low"
    df.loc[df["risk_score"] >= 4.0, "priority_tier"] = "Watch"
    df.loc[df["risk_score"] >= 6.0, "priority_tier"] = "Elevated"
    df.loc[df["risk_score"] >= 8.5, "priority_tier"] = "Critical"

    df["_score_prev"] = df.groupby(patient_col)["risk_score"].shift(1)
    df["_score_delta"] = df["risk_score"] - df["_score_prev"]

    df["trend_direction"] = "Stable"
    df.loc[df["_score_delta"] >= 0.5, "trend_direction"] = "Worsening"
    df.loc[df["_score_delta"] <= -0.5, "trend_direction"] = "Improving"
    df.loc[
        (df["_era_spo2_delta"].notna() & (df["_era_spo2_delta"] <= -3)) |
        (df["_era_sbp_delta"].notna() & (df["_era_sbp_delta"] <= -15)) |
        (df["_era_rr_delta"].notna() & (df["_era_rr_delta"].abs() >= 6)),
        "trend_direction"
    ] = "Worsening"

    df["threshold_crossed_at"] = ""
    df.loc[df["era_alert"], "threshold_crossed_at"] = "local-only-relative-time"

    df["queue_rank"] = df["risk_score"].rank(method="first", ascending=False).astype(int)

    df = df.sort_values("_original_order", kind="mergesort").drop(columns=["_original_order"])

    df.to_csv(output_path, index=False)
    print(f"WROTE LOCAL SCORED eICU COHORT: {output_path}")

    audit = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local-only eICU ERA scoring adapter before harmonized clinical-event labeling.",
        "public_safety": "Aggregate audit only. Scored row-level eICU cohort remains local-only.",
        "rows_scored": int(len(df)),
        "stays_scored": int(df[patient_col].nunique()),
        "risk_score_summary": {
            "min": round(float(df["risk_score"].min()), 3),
            "median": round(float(df["risk_score"].median()), 3),
            "mean": round(float(df["risk_score"].mean()), 3),
            "max": round(float(df["risk_score"].max()), 3),
            "era_alert_rows_at_t6": int((df["risk_score"] >= 6.0).sum()),
        },
        "explainability_columns_added": [
            "risk_score",
            "era_alert",
            "priority_tier",
            "primary_driver",
            "trend_direction",
            "queue_rank"
        ],
        "validation_boundary": (
            "This adapter creates explainable rules-based ERA scores for local eICU harmonized clinical-event testing. "
            "Do not use unqualified two-dataset validation wording until the harmonized output is reviewed."
        )
    }

    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"WROTE PUBLIC AGGREGATE AUDIT: {audit_path}")

    print("")
    print("SCORING SUMMARY")
    print("===============")
    print(f"Rows scored: {audit['rows_scored']:,}")
    print(f"Stays scored: {audit['stays_scored']:,}")
    print(f"Median risk score: {audit['risk_score_summary']['median']}")
    print(f"ERA alert rows at t=6.0: {audit['risk_score_summary']['era_alert_rows_at_t6']:,}")


if __name__ == "__main__":
    main()
