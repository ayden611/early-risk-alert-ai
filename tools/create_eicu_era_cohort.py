#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAW = Path("data/validation/local_private/eicu/raw")
WORK = Path("data/validation/local_private/eicu/working")
MANIFESTS = Path("data/validation/local_private/eicu/manifests")

REQUIRED = {
    "patient": "patient.csv.gz",
    "vital_periodic": "vitalPeriodic.csv.gz",
    "vital_aperiodic": "vitalAperiodic.csv.gz",
    "diagnosis": "diagnosis.csv.gz",
    "apache": "apachePatientResult.csv.gz",
}

ERA_COLUMNS = [
    "patient_id",
    "timestamp",
    "heart_rate",
    "spo2",
    "bp_systolic",
    "bp_diastolic",
    "respiratory_rate",
    "temperature_f",
    "clinical_event",
    "event_label",
    "event_group",
]


def open_gz_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")


def norm(s):
    return str(s or "").strip()


def lower_map(headers):
    return {str(h).lower(): h for h in headers}


def pick(headers, candidates):
    lm = lower_map(headers)
    for c in candidates:
        if c.lower() in lm:
            return lm[c.lower()]
    return None


def truthy_outcome(v):
    s = norm(v).lower()
    return s in {
        "expired",
        "death",
        "dead",
        "deceased",
        "1",
        "true",
        "yes",
        "y",
    }


def to_float(v):
    try:
        s = norm(v)
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def to_int(v):
    x = to_float(v)
    if x is None:
        return None
    return int(round(x))


def safe_case_id(stay_id):
    digest = hashlib.sha256(str(stay_id).encode("utf-8")).hexdigest()[:12]
    return f"EICU-CASE-{digest}"


def stable_bucket(value):
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100


def relative_timestamp(offset_min):
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    try:
        return (base + timedelta(minutes=float(offset_min))).isoformat()
    except Exception:
        return ""


def clean_temp_to_f(v):
    x = to_float(v)
    if x is None:
        return ""
    if 20 <= x <= 45:
        x = (x * 9 / 5) + 32
    return round(x, 2)


def clean_num(v):
    x = to_float(v)
    if x is None:
        return ""
    return round(x, 4)


def get_headers(path):
    with open_gz_csv(path) as f:
        reader = csv.reader(f)
        return next(reader, [])


def load_patient_context(patient_path: Path):
    headers = get_headers(patient_path)

    stay_col = pick(headers, ["patientunitstayid"])
    unit_discharge_offset_col = pick(headers, ["unitdischargeoffset"])
    unit_discharge_status_col = pick(headers, ["unitdischargestatus"])
    hospital_discharge_status_col = pick(headers, ["hospitaldischargestatus"])

    if not stay_col:
        raise SystemExit("ERROR: patient.csv.gz missing patientunitstayid column.")

    context = {}

    with open_gz_csv(patient_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay = norm(row.get(stay_col))
            if not stay:
                continue

            unit_offset = to_int(row.get(unit_discharge_offset_col)) if unit_discharge_offset_col else None
            unit_status = norm(row.get(unit_discharge_status_col)) if unit_discharge_status_col else ""
            hosp_status = norm(row.get(hospital_discharge_status_col)) if hospital_discharge_status_col else ""

            context[stay] = {
                "unit_discharge_offset": unit_offset,
                "unit_discharge_status": unit_status,
                "hospital_discharge_status": hosp_status,
                "event_from_patient_status": truthy_outcome(unit_status) or truthy_outcome(hosp_status),
            }

    return context, {
        "stay_col": stay_col,
        "unit_discharge_offset_col": unit_discharge_offset_col,
        "unit_discharge_status_col": unit_discharge_status_col,
        "hospital_discharge_status_col": hospital_discharge_status_col,
    }


def load_apache_events(apache_path: Path):
    headers = get_headers(apache_path)

    stay_col = pick(headers, ["patientunitstayid"])
    actual_hosp_col = pick(headers, ["actualhospitalmortality"])
    actual_icu_col = pick(headers, ["actualicumortality"])

    if not stay_col:
        raise SystemExit("ERROR: apachePatientResult.csv.gz missing patientunitstayid column.")

    apache = {}

    with open_gz_csv(apache_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay = norm(row.get(stay_col))
            if not stay:
                continue

            actual_hosp = norm(row.get(actual_hosp_col)) if actual_hosp_col else ""
            actual_icu = norm(row.get(actual_icu_col)) if actual_icu_col else ""

            apache[stay] = {
                "actual_hospital_mortality": actual_hosp,
                "actual_icu_mortality": actual_icu,
                "event_from_apache": truthy_outcome(actual_hosp) or truthy_outcome(actual_icu),
            }

    return apache, {
        "stay_col": stay_col,
        "actual_hospital_mortality_col": actual_hosp_col,
        "actual_icu_mortality_col": actual_icu_col,
    }


def select_stays(patient_context, apache_events, max_stays, max_event_stays, bucket_start, bucket_end):
    event_candidates = []
    control_candidates = []

    for stay, ctx in patient_context.items():
        bucket = stable_bucket(stay)
        if not (bucket_start <= bucket <= bucket_end):
            continue

        apache = apache_events.get(stay, {})
        is_event = bool(ctx.get("event_from_patient_status") or apache.get("event_from_apache"))

        if is_event:
            event_candidates.append(stay)
        else:
            control_candidates.append(stay)

    event_candidates = sorted(event_candidates, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    control_candidates = sorted(control_candidates, key=lambda x: hashlib.sha256(x.encode()).hexdigest())

    selected_events = event_candidates[:max_event_stays]
    remaining = max(max_stays - len(selected_events), 0)
    selected_controls = control_candidates[:remaining]

    selected = set(selected_events + selected_controls)

    return selected, {
        "selected_total": len(selected),
        "selected_event_stays": len(selected_events),
        "selected_control_stays": len(selected_controls),
        "available_event_candidates": len(event_candidates),
        "available_control_candidates": len(control_candidates),
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
    }


def load_aperiodic_bp(path: Path, selected_stays):
    headers = get_headers(path)

    stay_col = pick(headers, ["patientunitstayid"])
    offset_col = pick(headers, ["observationoffset"])
    sys_col = pick(headers, ["noninvasivesystolic", "systemicsystolic"])
    dia_col = pick(headers, ["noninvasivediastolic", "systemicdiastolic"])

    if not stay_col or not offset_col:
        return {}, {
            "available": False,
            "reason": "missing stay or offset column",
            "headers": headers,
        }

    bp = {}
    rows_scanned = 0
    rows_kept = 0

    with open_gz_csv(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_scanned += 1
            stay = norm(row.get(stay_col))
            if stay not in selected_stays:
                continue

            offset = to_int(row.get(offset_col))
            if offset is None:
                continue

            sysv = clean_num(row.get(sys_col)) if sys_col else ""
            diav = clean_num(row.get(dia_col)) if dia_col else ""

            if sysv == "" and diav == "":
                continue

            bp[(stay, offset)] = (sysv, diav)
            rows_kept += 1

    return bp, {
        "available": True,
        "stay_col": stay_col,
        "offset_col": offset_col,
        "systolic_col": sys_col,
        "diastolic_col": dia_col,
        "rows_scanned": rows_scanned,
        "rows_kept_for_selected_stays": rows_kept,
    }


def build_cohort(args):
    RAW.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    paths = {k: RAW / v for k, v in REQUIRED.items()}

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("ERROR: Missing eICU raw files in local-private raw folder:\n" + "\n".join(missing))

    patient_context, patient_cols = load_patient_context(paths["patient"])
    apache_events, apache_cols = load_apache_events(paths["apache"])

    selected_stays, selection_summary = select_stays(
        patient_context=patient_context,
        apache_events=apache_events,
        max_stays=args.max_stays,
        max_event_stays=args.max_event_stays,
        bucket_start=args.bucket_start,
        bucket_end=args.bucket_end,
    )

    if not selected_stays:
        raise SystemExit("ERROR: no eICU stays selected. Try a wider bucket range or higher max_stays.")

    bp_lookup, bp_summary = load_aperiodic_bp(paths["vital_aperiodic"], selected_stays)

    vp_headers = get_headers(paths["vital_periodic"])
    stay_col = pick(vp_headers, ["patientunitstayid"])
    offset_col = pick(vp_headers, ["observationoffset"])
    hr_col = pick(vp_headers, ["heartrate", "heart_rate"])
    spo2_col = pick(vp_headers, ["sao2", "spo2", "oxygen_saturation"])
    rr_col = pick(vp_headers, ["respiration", "respiratoryrate", "respiratory_rate"])
    temp_col = pick(vp_headers, ["temperature", "temp"])
    sys_col = pick(vp_headers, ["systemicsystolic", "noninvasivesystolic"])
    dia_col = pick(vp_headers, ["systemicdiastolic", "noninvasivediastolic"])

    if not stay_col or not offset_col:
        raise SystemExit("ERROR: vitalPeriodic.csv.gz missing patientunitstayid or observationoffset.")

    out_csv = WORK / "eicu_era_cohort.csv"
    local_map = WORK / "eicu_case_mapping_local_only.json"
    manifest_path = MANIFESTS / "eicu_era_cohort_manifest.json"

    case_map = {stay: safe_case_id(stay) for stay in selected_stays}
    last_vitals = {}
    observed_stays = set()

    rows_scanned = 0
    rows_written = 0

    with out_csv.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=ERA_COLUMNS)
        writer.writeheader()

        with open_gz_csv(paths["vital_periodic"]) as f:
            reader = csv.DictReader(f)

            for row in reader:
                rows_scanned += 1

                stay = norm(row.get(stay_col))
                if stay not in selected_stays:
                    continue

                offset = to_int(row.get(offset_col))
                if offset is None:
                    continue

                bp_sys = clean_num(row.get(sys_col)) if sys_col else ""
                bp_dia = clean_num(row.get(dia_col)) if dia_col else ""

                if (bp_sys == "" or bp_dia == "") and (stay, offset) in bp_lookup:
                    a_sys, a_dia = bp_lookup[(stay, offset)]
                    bp_sys = bp_sys or a_sys
                    bp_dia = bp_dia or a_dia

                out_row = {
                    "patient_id": case_map[stay],
                    "timestamp": relative_timestamp(offset),
                    "heart_rate": clean_num(row.get(hr_col)) if hr_col else "",
                    "spo2": clean_num(row.get(spo2_col)) if spo2_col else "",
                    "bp_systolic": bp_sys,
                    "bp_diastolic": bp_dia,
                    "respiratory_rate": clean_num(row.get(rr_col)) if rr_col else "",
                    "temperature_f": clean_temp_to_f(row.get(temp_col)) if temp_col else "",
                    "clinical_event": 0,
                    "event_label": "",
                    "event_group": "",
                }

                if not any(out_row[c] not in {"", 0} for c in ["heart_rate", "spo2", "bp_systolic", "bp_diastolic", "respiratory_rate", "temperature_f"]):
                    continue

                writer.writerow(out_row)
                rows_written += 1
                observed_stays.add(stay)
                last_vitals[stay] = (offset, out_row)

        event_proxy_rows = 0
        event_observed_stays = 0

        for stay in sorted(selected_stays):
            ctx = patient_context.get(stay, {})
            apache = apache_events.get(stay, {})
            is_event = bool(ctx.get("event_from_patient_status") or apache.get("event_from_apache"))

            if not is_event:
                continue

            event_offset = ctx.get("unit_discharge_offset")
            if event_offset is None:
                if stay in last_vitals:
                    event_offset = last_vitals[stay][0]
                else:
                    continue

            if stay not in last_vitals:
                continue

            event_observed_stays += 1
            base_row = dict(last_vitals[stay][1])
            base_row["timestamp"] = relative_timestamp(event_offset)
            base_row["clinical_event"] = 1
            base_row["event_label"] = "eicu_outcome_proxy"
            base_row["event_group"] = "apache_or_discharge_outcome_proxy"

            writer.writerow(base_row)
            rows_written += 1
            event_proxy_rows += 1

    local_map.write_text(json.dumps({
        "purpose": "LOCAL ONLY mapping from eICU patientunitstayid to synthetic ERA case ID. Do not commit or publish.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_count": len(case_map),
        "mapping": case_map,
    }, indent=2), encoding="utf-8")

    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local-only eICU to ERA-format extraction for retrospective aggregate validation readiness.",
        "output_csv_local_only": str(out_csv),
        "local_case_mapping_file": str(local_map),
        "max_stays_requested": args.max_stays,
        "max_event_stays_requested": args.max_event_stays,
        "selection_summary": selection_summary,
        "rows_scanned_in_vital_periodic": rows_scanned,
        "era_format_rows_written": rows_written,
        "selected_stays_with_vitals": len(observed_stays),
        "event_proxy_rows_written": event_proxy_rows,
        "event_stays_with_vitals": event_observed_stays,
        "patient_columns": patient_cols,
        "apache_columns": apache_cols,
        "vital_periodic_columns": {
            "stay_col": stay_col,
            "offset_col": offset_col,
            "heart_rate_col": hr_col,
            "spo2_source_col": spo2_col,
            "respiratory_rate_col": rr_col,
            "temperature_col": temp_col,
            "bp_systolic_col": sys_col,
            "bp_diastolic_col": dia_col,
        },
        "vital_aperiodic_bp_summary": bp_summary,
        "event_definition": {
            "type": "outcome proxy",
            "source": "apache actual mortality fields and/or discharge status fields",
            "event_time_proxy": "unitdischargeoffset or last observed vital offset if discharge offset unavailable",
            "claim_guardrail": "Use as eICU outcome-proxy retrospective evaluation, not direct deterioration-event validation."
        },
        "public_policy": "Row-level eICU extraction remains local-only. Public outputs may include aggregate metrics only."
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def write_public_readiness_summary(manifest):
    DATA = Path("data/validation")
    DOCS = Path("docs/validation")
    DATA.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    public = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "DUA-safe aggregate eICU extraction readiness summary.",
        "dataset": "eICU Collaborative Research Database",
        "extraction_status": "ERA-format local-only cohort created",
        "selected_stays_total": manifest["selection_summary"]["selected_total"],
        "selected_event_stays": manifest["selection_summary"]["selected_event_stays"],
        "selected_control_stays": manifest["selection_summary"]["selected_control_stays"],
        "selected_stays_with_vitals": manifest["selected_stays_with_vitals"],
        "era_format_rows_written": manifest["era_format_rows_written"],
        "event_proxy_rows_written": manifest["event_proxy_rows_written"],
        "event_stays_with_vitals": manifest["event_stays_with_vitals"],
        "vital_periodic_rows_scanned": manifest["rows_scanned_in_vital_periodic"],
        "spo2_source_column_detected": manifest["vital_periodic_columns"]["spo2_source_col"],
        "temperature_handling": "Auto-normalized to Fahrenheit when values appear Celsius; Fahrenheit-like values preserved.",
        "bp_source": "Periodic BP columns when available; aperiodic non-invasive BP used as fallback at matching offsets.",
        "event_definition_public": "Outcome-proxy event labels from mortality/discharge status fields using relative ICU offset timing.",
        "claim_guardrail": "This is extraction readiness for retrospective outcome-proxy validation, not prospective clinical validation.",
        "public_policy": "Aggregate summary only. Raw eICU files and row-level outputs remain local-only."
    }

    (DATA / "eicu_extraction_readiness_summary.json").write_text(json.dumps(public, indent=2), encoding="utf-8")

    md = f"""# eICU Extraction Readiness Summary

## Status

ERA-format local-only eICU cohort created.

## Aggregate Extraction Summary

| Item | Value |
|---|---:|
| Selected stays | {public['selected_stays_total']} |
| Selected event-proxy stays | {public['selected_event_stays']} |
| Selected control stays | {public['selected_control_stays']} |
| Selected stays with vitals | {public['selected_stays_with_vitals']} |
| ERA-format rows written | {public['era_format_rows_written']} |
| Event-proxy rows written | {public['event_proxy_rows_written']} |
| Event-proxy stays with vitals | {public['event_stays_with_vitals']} |
| VitalPeriodic rows scanned | {public['vital_periodic_rows_scanned']} |
| SpO2 source column detected | {public['spo2_source_column_detected']} |

## Event Definition

This extraction uses an eICU outcome-proxy event definition from mortality/discharge status fields and relative ICU offsets.

This should be framed as:

**Cross-dataset retrospective outcome-proxy evaluation on de-identified ICU data.**

Do not frame it as direct deterioration-event validation, prospective validation, diagnosis, treatment direction, or autonomous escalation.

## DUA-Safe Boundary

Raw eICU files and row-level ERA-format output remain local-only.

Public evidence may include aggregate metrics only.
"""

    (DOCS / "eicu_extraction_readiness_summary.md").write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stays", type=int, default=2500)
    ap.add_argument("--max-event-stays", type=int, default=800)
    ap.add_argument("--bucket-start", type=int, default=0)
    ap.add_argument("--bucket-end", type=int, default=49)
    args = ap.parse_args()

    manifest = build_cohort(args)
    write_public_readiness_summary(manifest)

    print("")
    print("eICU ERA-FORMAT EXTRACTION COMPLETE")
    print("===================================")
    print("Selected stays:", manifest["selection_summary"]["selected_total"])
    print("Selected event-proxy stays:", manifest["selection_summary"]["selected_event_stays"])
    print("Selected control stays:", manifest["selection_summary"]["selected_control_stays"])
    print("Selected stays with vitals:", manifest["selected_stays_with_vitals"])
    print("ERA-format rows written:", manifest["era_format_rows_written"])
    print("Event-proxy rows written:", manifest["event_proxy_rows_written"])
    print("Output CSV local-only:", manifest["output_csv_local_only"])
    print("Manifest local-only: data/validation/local_private/eicu/manifests/eicu_era_cohort_manifest.json")
    print("")
    print("NEXT STEP AFTER YOU REVIEW THIS OUTPUT:")
    print("Run DUA-safe validation threshold matrix on the local-only eICU ERA-format cohort.")


if __name__ == "__main__":
    main()
