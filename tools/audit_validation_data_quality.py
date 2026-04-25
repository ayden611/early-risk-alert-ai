#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


FALSE_VALUES = {"", "0", "0.0", "false", "no", "n", "none", "nan", "null", "stable", "no_event"}


def parse_time(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=None)
        except Exception:
            pass

    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def find_col(headers, candidates):
    lower = {str(h).lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def fnum(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def truthy(v):
    s = str(v or "").strip().lower()
    if s in FALSE_VALUES:
        return False
    if s in {"1", "true", "yes", "y", "event", "clinical_event", "positive", "critical", "high"}:
        return True
    try:
        return float(s) != 0
    except Exception:
        return False


def pct(n, d):
    if not d:
        return None
    return round(100.0 * n / d, 2)


def median(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.median(clean), 3)


def quantile(values, q):
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return round(clean[idx], 3)


def classify_family(driver):
    d = str(driver or "").strip().lower()
    if not d or d in {"none", "nan", "null"}:
        return "Composite / no dominant driver"
    if "spo2" in d or "sp02" in d or "oxygen" in d or "desat" in d or "saturation" in d:
        return "SpO2 / oxygenation"
    if "resp" in d or "rr" in d or "breath" in d:
        return "Respiratory rate"
    if "bp" in d or "blood pressure" in d or "systolic" in d or "diastolic" in d:
        return "Blood pressure"
    if "hr" in d or "heart" in d or "pulse" in d:
        return "Heart rate"
    if "temp" in d or "fever" in d:
        return "Temperature"
    if "composite" in d or "multi" in d or "dominant" in d:
        return "Composite / no dominant driver"
    return "Other driver"


def load_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def row_key(case_id, ts):
    if not ts:
        return None
    return (str(case_id or "").strip(), ts.strftime("%Y-%m-%d %H:%M:%S"))


def detect_event(row, clinical_col=None, label_col=None, group_col=None):
    if clinical_col:
        return truthy(row.get(clinical_col))
    if label_col:
        return truthy(row.get(label_col))
    if group_col:
        return truthy(row.get(group_col))
    return False


def collapse_event_clusters(events_by_case, gap_hours):
    gap = timedelta(hours=float(gap_hours))
    clusters = {}
    for cid, times in events_by_case.items():
        sorted_times = sorted(set(times))
        out = []
        last = None
        for ts in sorted_times:
            if last is None or ts - last > gap:
                out.append(ts)
            last = ts
        clusters[cid] = out
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--enriched-csv", default="data/validation/latest_era_validation_export.csv")
    ap.add_argument("--out-json", default="data/validation/data_quality_temporal_audit_summary.json")
    ap.add_argument("--out-md", default="docs/validation/data_quality_temporal_audit_summary.md")
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--event-gap-hours", type=float, default=6.0)
    args = ap.parse_args()

    src_headers, src_rows = load_csv(args.source_csv)
    enr_headers, enr_rows = load_csv(args.enriched_csv)

    if not src_rows:
        raise SystemExit("ERROR: source CSV has no rows.")
    if not enr_rows:
        raise SystemExit("ERROR: enriched CSV has no rows.")

    src_case_col = find_col(src_headers, ["patient_id", "case_id", "subject_id"])
    src_time_col = find_col(src_headers, ["timestamp", "charttime", "time"])
    clinical_col = find_col(src_headers, ["clinical_event", "event", "event_flag", "has_event", "clinical_event_flag"])
    label_col = find_col(src_headers, ["event_label", "label_text"])
    group_col = find_col(src_headers, ["event_group", "group", "event_cluster"])

    if not src_case_col or not src_time_col:
        raise SystemExit("ERROR: source CSV missing case/time columns.")

    enr_case_col = find_col(enr_headers, ["patient_id", "case_id", "subject_id"])
    enr_time_col = find_col(enr_headers, ["timestamp", "charttime", "time"])
    risk_col = find_col(enr_headers, ["risk_score", "era_risk_score", "score"])
    alert_col = find_col(enr_headers, ["era_alert", "alert"])
    tier_col = find_col(enr_headers, ["priority_tier", "tier"])
    driver_col = find_col(enr_headers, ["primary_driver", "driver", "dominant_driver"])
    trend_col = find_col(enr_headers, ["trend_direction", "trend"])

    if not enr_case_col or not enr_time_col:
        raise SystemExit("ERROR: enriched CSV missing case/time columns.")

    core_vitals = {
        "heart_rate": find_col(src_headers, ["heart_rate", "hr"]),
        "spo2": find_col(src_headers, ["spo2", "sp02", "oxygen_saturation"]),
        "bp_systolic": find_col(src_headers, ["bp_systolic", "systolic_bp", "sbp"]),
        "bp_diastolic": find_col(src_headers, ["bp_diastolic", "diastolic_bp", "dbp"]),
        "respiratory_rate": find_col(src_headers, ["respiratory_rate", "rr", "resp_rate"]),
        "temperature_f": find_col(src_headers, ["temperature_f", "temp_f", "temperature"]),
    }

    src_case_ids = []
    src_times = []
    src_keys = set()
    dup_keys = 0
    seen_keys = set()
    events_by_case = defaultdict(list)
    rows_per_case = Counter()
    chronology_issues = 0
    last_time_by_case = {}

    vital_missing = Counter()
    vital_present_counts = []

    for r in src_rows:
        cid = str(r.get(src_case_col, "")).strip()
        ts = parse_time(r.get(src_time_col))
        if cid:
            src_case_ids.append(cid)
            rows_per_case[cid] += 1

        if ts:
            src_times.append(ts)
            k = row_key(cid, ts)
            if k in seen_keys:
                dup_keys += 1
            else:
                seen_keys.add(k)
                src_keys.add(k)

            if cid in last_time_by_case and ts < last_time_by_case[cid]:
                chronology_issues += 1
            last_time_by_case[cid] = ts

        present_count = 0
        for name, col in core_vitals.items():
            if not col or str(r.get(col, "")).strip() == "":
                vital_missing[name] += 1
            else:
                present_count += 1
        vital_present_counts.append(present_count)

        if ts and cid and detect_event(r, clinical_col, label_col, group_col):
            events_by_case[cid].append(ts)

    clusters = collapse_event_clusters(events_by_case, args.event_gap_hours)

    enr_keys = set()
    risk_values = []
    alert_rows = []
    tier_dist = Counter()
    trend_dist = Counter()
    driver_family_dist = Counter()
    enriched_cases = set()

    for idx, r in enumerate(enr_rows):
        cid = str(r.get(enr_case_col, "")).strip()
        ts = parse_time(r.get(enr_time_col))
        if cid:
            enriched_cases.add(cid)

        if ts:
            k = row_key(cid, ts)
            if k:
                enr_keys.add(k)

        risk = fnum(r.get(risk_col)) if risk_col else None
        if risk is not None:
            risk_values.append(risk)

        if alert_col:
            is_alert = truthy(r.get(alert_col))
        else:
            is_alert = risk is not None and risk >= 6.0

        if is_alert:
            alert_rows.append((cid, ts, risk, r))
            tier_dist[str(r.get(tier_col, "") or "Unspecified").strip()] += 1
            trend_dist[str(r.get(trend_col, "") or "Unspecified").strip()] += 1
            driver_family_dist[classify_family(r.get(driver_col))] += 1

    overlap = len(src_keys & enr_keys)
    clusters_total = sum(len(v) for v in clusters.values())
    clusters_cases = len(clusters)

    by_enriched_case = defaultdict(list)
    for cid, ts, risk, raw in alert_rows:
        if cid and ts:
            by_enriched_case[cid].append((ts, risk, raw))
    for cid in by_enriched_case:
        by_enriched_case[cid].sort(key=lambda x: x[0])

    threshold_diagnostics = []
    window = timedelta(hours=float(args.window_hours))

    for threshold in [4.0, 5.0, 6.0]:
        pre_hits = 0
        post_hits = 0
        clusters_seen = 0

        for cid, event_times in clusters.items():
            rows_for_case = by_enriched_case.get(cid, [])
            if not rows_for_case:
                continue

            for ev_ts in event_times:
                clusters_seen += 1
                pre = any(
                    (risk is not None and risk >= threshold and ev_ts - window <= alert_ts < ev_ts)
                    for alert_ts, risk, raw in rows_for_case
                )
                post = any(
                    (risk is not None and risk >= threshold and ev_ts <= alert_ts <= ev_ts + window)
                    for alert_ts, risk, raw in rows_for_case
                )
                if pre:
                    pre_hits += 1
                if post:
                    post_hits += 1

        threshold_diagnostics.append({
            "threshold": threshold,
            "source_event_clusters_reviewed": clusters_seen,
            "pre_window_cluster_hits": pre_hits,
            "post_window_cluster_hits": post_hits,
            "pre_window_cluster_hit_pct": pct(pre_hits, clusters_seen),
            "post_window_cluster_hit_pct": pct(post_hits, clusters_seen),
            "interpretation": "Aggregate temporal-alignment diagnostic only; not used as official lead-time evidence."
        })

    row_counts = list(rows_per_case.values())

    time_span_days = None
    if src_times:
        time_span_days = round((max(src_times) - min(src_times)).total_seconds() / 86400.0, 3)

    summary = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "DUA-safe aggregate data-quality and temporal-alignment audit.",
        "source_schema_audit": {
            "rows": len(src_rows),
            "unique_cases": len(set(src_case_ids)),
            "valid_timestamp_rows": len(src_times),
            "valid_timestamp_pct": pct(len(src_times), len(src_rows)),
            "time_span_days": time_span_days,
            "duplicate_case_time_rows": dup_keys,
            "chronology_issues_in_original_order": chronology_issues,
            "median_rows_per_case": median(row_counts),
            "p90_rows_per_case": quantile(row_counts, 0.90),
            "event_labeled_rows": sum(len(v) for v in events_by_case.values()),
            "event_clusters_gap_6h": clusters_total,
            "cases_with_event_clusters": clusters_cases,
        },
        "vital_coverage_audit": {
            name: {
                "column_present": bool(col),
                "missing_rows": int(vital_missing[name]),
                "missing_pct": pct(vital_missing[name], len(src_rows)),
            }
            for name, col in core_vitals.items()
        },
        "vital_completeness": {
            "median_present_core_vitals_per_row": median(vital_present_counts),
            "rows_with_at_least_3_core_vitals_pct": pct(sum(1 for x in vital_present_counts if x >= 3), len(vital_present_counts)),
            "rows_with_all_detected_core_vitals_pct": pct(sum(1 for x in vital_present_counts if x == sum(1 for c in core_vitals.values() if c)), len(vital_present_counts)),
        },
        "enriched_export_audit": {
            "rows": len(enr_rows),
            "unique_cases": len(enriched_cases),
            "source_enriched_case_time_overlap_rows": overlap,
            "source_enriched_case_time_overlap_pct_of_source": pct(overlap, len(src_keys)),
            "risk_score_rows": len(risk_values),
            "risk_score_coverage_pct": pct(len(risk_values), len(enr_rows)),
            "median_risk_score": median(risk_values),
            "max_risk_score": max(risk_values) if risk_values else None,
            "era_alert_rows_t6_export_state": len(alert_rows),
            "priority_tier_distribution": dict(tier_dist),
            "trend_distribution": dict(trend_dist),
            "driver_family_distribution": dict(driver_family_dist),
        },
        "temporal_alignment_diagnostic": {
            "event_window_hours": args.window_hours,
            "event_gap_hours": args.event_gap_hours,
            "thresholds": threshold_diagnostics,
            "note": "This diagnostic explains source-event/enriched-alert timing alignment. Official lead-time evidence remains the dedicated lead-time robustness analysis."
        },
        "public_policy": "Aggregate metrics only. No row-level MIMIC data, no MIMIC IDs, no exact patient-linked timestamps.",
        "pilot_safe_claim": "A DUA-safe aggregate audit found the validation dataset and enriched ERA export can be reviewed for schema coverage, event-label integrity, timestamp quality, and temporal-alignment behavior without exposing row-level data.",
        "notice": "Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation."
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def val(x, suffix=""):
        if x is None or x == "":
            return "—"
        return f"{x}{suffix}"

    md = f"""# DUA-Safe Data Quality + Temporal Alignment Audit

## Purpose

This audit checks whether the validation dataset and enriched ERA export are suitable for continued retrospective testing.

It summarizes:

- Schema readiness
- Timestamp validity
- Duplicate case-time rows
- Event-label integrity
- Vital-sign missingness
- Enriched export alignment
- Aggregate temporal-alignment diagnostics

No row-level MIMIC-derived data, real identifiers, or exact patient-linked timestamps are included.

## Source Dataset Audit

| Metric | Value |
|---|---:|
| Rows | {summary['source_schema_audit']['rows']} |
| Unique cases | {summary['source_schema_audit']['unique_cases']} |
| Valid timestamp rows | {summary['source_schema_audit']['valid_timestamp_rows']} |
| Valid timestamp % | {val(summary['source_schema_audit']['valid_timestamp_pct'], '%')} |
| Time span | {val(summary['source_schema_audit']['time_span_days'], ' days')} |
| Duplicate case-time rows | {summary['source_schema_audit']['duplicate_case_time_rows']} |
| Chronology issues in original order | {summary['source_schema_audit']['chronology_issues_in_original_order']} |
| Median rows per case | {summary['source_schema_audit']['median_rows_per_case']} |
| Event-labeled rows | {summary['source_schema_audit']['event_labeled_rows']} |
| Event clusters, 6h gap | {summary['source_schema_audit']['event_clusters_gap_6h']} |

## Vital Coverage

| Signal | Column Present | Missing Rows | Missing % |
|---|---:|---:|---:|
"""

    for name, item in summary["vital_coverage_audit"].items():
        md += f"| {name} | {item['column_present']} | {item['missing_rows']} | {val(item['missing_pct'], '%')} |\n"

    md += f"""
## Enriched ERA Export Audit

| Metric | Value |
|---|---:|
| Rows | {summary['enriched_export_audit']['rows']} |
| Unique cases | {summary['enriched_export_audit']['unique_cases']} |
| Source/enriched case-time overlap rows | {summary['enriched_export_audit']['source_enriched_case_time_overlap_rows']} |
| Source/enriched overlap % of source | {val(summary['enriched_export_audit']['source_enriched_case_time_overlap_pct_of_source'], '%')} |
| Risk score coverage % | {val(summary['enriched_export_audit']['risk_score_coverage_pct'], '%')} |
| Median risk score | {summary['enriched_export_audit']['median_risk_score']} |
| Max risk score | {summary['enriched_export_audit']['max_risk_score']} |
| ERA alert rows in current export state | {summary['enriched_export_audit']['era_alert_rows_t6_export_state']} |

## Aggregate Temporal Alignment Diagnostic

This diagnostic explains why driver subgroups are treated as explainability distributions, while official lead-time evidence remains in the dedicated lead-time robustness test.

| Threshold | Pre-window cluster hits | Pre-window hit % | Post-window cluster hits | Post-window hit % |
|---:|---:|---:|---:|---:|
"""

    for item in threshold_diagnostics:
        md += (
            f"| t={item['threshold']} "
            f"| {item['pre_window_cluster_hits']} "
            f"| {val(item['pre_window_cluster_hit_pct'], '%')} "
            f"| {item['post_window_cluster_hits']} "
            f"| {val(item['post_window_cluster_hit_pct'], '%')} |\n"
        )

    md += """
## Pilot-Safe Interpretation

This audit strengthens the validation package by documenting dataset quality and temporal-alignment behavior before additional MIMIC testing.

Use this as quality-control evidence, not as clinical performance proof.

## Pilot-Safe Claim

A DUA-safe aggregate audit found the validation dataset and enriched ERA export can be reviewed for schema coverage, event-label integrity, timestamp quality, and temporal-alignment behavior without exposing row-level data.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    print("WROTE:", out_json)
    print("WROTE:", out_md)
    print("Rows:", len(src_rows))
    print("Unique cases:", len(set(src_case_ids)))
    print("Event clusters:", clusters_total)
    print("Overlap rows:", overlap)
    print("Risk score coverage:", summary["enriched_export_audit"]["risk_score_coverage_pct"])
    print("Temporal alignment diagnostic:")
    for item in threshold_diagnostics:
        print(" - t=", item["threshold"], "pre_hits=", item["pre_window_cluster_hits"], "post_hits=", item["post_window_cluster_hits"])


if __name__ == "__main__":
    main()
