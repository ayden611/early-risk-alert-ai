#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


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
        "%Y-%m-%dT%H:%M:%S%z",
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


def truthy(value):
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "y", "event", "clinical_event", "positive"}


def as_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def classify_family(driver):
    d = str(driver or "").strip().lower()

    if not d or d in {"nan", "none", "null"}:
        return "Composite / no dominant driver"

    if "spo2" in d or "oxygen" in d or "desat" in d or "saturation" in d:
        return "SpO2 / oxygenation"

    if "resp" in d or "rr" in d or "breath" in d:
        return "Respiratory rate"

    if "bp" in d or "blood pressure" in d or "systolic" in d or "diastolic" in d:
        return "Blood pressure"

    if "hr" in d or "heart" in d or "pulse" in d:
        return "Heart rate"

    if "temp" in d or "fever" in d:
        return "Temperature"

    if "composite" in d or "multi" in d or "no dominant" in d:
        return "Composite / no dominant driver"

    return "Other driver"


def find_col(headers, candidates):
    lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def collapse_events(rows, patient_col, time_col, event_col, event_group_col, event_gap_hours):
    by_patient = defaultdict(list)

    for r in rows:
        pid = str(r.get(patient_col, "")).strip()
        ts = r.get("_parsed_time")
        if not pid or not ts:
            continue

        is_event = truthy(r.get(event_col)) if event_col else False

        if not is_event and event_group_col:
            group_val = str(r.get(event_group_col, "")).strip()
            is_event = bool(group_val)

        if is_event:
            by_patient[pid].append(ts)

    collapsed = {}
    gap = timedelta(hours=float(event_gap_hours))

    for pid, times in by_patient.items():
        times = sorted(set(times))
        clusters = []
        last = None

        for ts in times:
            if last is None or ts - last > gap:
                clusters.append(ts)
            last = ts

        collapsed[pid] = clusters

    return collapsed


def summarize_counter(counter):
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--event-gap-hours", type=float, default=6.0)
    ap.add_argument("--out-json", default="data/validation/driver_subgroup_summary.json")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"ERROR: CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise SystemExit("ERROR: CSV has no rows.")

    patient_col = find_col(headers, ["patient_id", "case_id", "subject_id"])
    time_col = find_col(headers, ["timestamp", "charttime", "time"])
    event_col = find_col(headers, ["clinical_event", "event", "label"])
    event_group_col = find_col(headers, ["event_group", "event_label", "group"])
    risk_col = find_col(headers, ["risk_score", "era_risk_score", "score"])
    alert_col = find_col(headers, ["era_alert", "alert"])
    tier_col = find_col(headers, ["priority_tier", "tier"])
    driver_col = find_col(headers, ["primary_driver", "driver", "dominant_driver"])
    trend_col = find_col(headers, ["trend_direction", "trend"])

    required = {
        "patient": patient_col,
        "time": time_col,
        "risk": risk_col,
        "driver": driver_col,
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns for driver subgroup analysis: {missing}. Found: {headers}")

    for r in rows:
        r["_parsed_time"] = parse_time(r.get(time_col))
        risk = as_float(r.get(risk_col))
        r["_risk_score"] = risk
        if alert_col:
            r["_era_alert"] = truthy(r.get(alert_col))
        else:
            r["_era_alert"] = risk is not None and risk >= float(args.threshold)
        r["_driver_family"] = classify_family(r.get(driver_col))
        r["_priority_tier"] = str(r.get(tier_col, "")).strip() if tier_col else ""
        r["_trend_direction"] = str(r.get(trend_col, "")).strip() if trend_col else ""

    events_by_patient = collapse_events(
        rows,
        patient_col=patient_col,
        time_col=time_col,
        event_col=event_col,
        event_group_col=event_group_col,
        event_gap_hours=args.event_gap_hours,
    )

    window = timedelta(hours=float(args.window_hours))

    event_window_row_ids = set()
    detected_cluster_family_counts = Counter()
    lead_times_by_family = defaultdict(list)

    total_event_clusters = sum(len(v) for v in events_by_patient.values())

    by_patient_rows = defaultdict(list)
    for r in rows:
        pid = str(r.get(patient_col, "")).strip()
        if pid:
            by_patient_rows[pid].append(r)

    for pid, patient_rows in by_patient_rows.items():
        patient_rows.sort(key=lambda x: x.get("_parsed_time") or datetime.min)

    for pid, event_times in events_by_patient.items():
        patient_rows = by_patient_rows.get(pid, [])

        for ev_time in event_times:
            candidates = []
            for r in patient_rows:
                ts = r.get("_parsed_time")
                if not ts:
                    continue

                if ev_time - window <= ts < ev_time:
                    event_window_row_ids.add(id(r))
                    if r.get("_era_alert"):
                        candidates.append(r)

            if candidates:
                first_alert = min(candidates, key=lambda x: x.get("_parsed_time") or ev_time)
                fam = first_alert.get("_driver_family") or "Unknown"
                lead = round((ev_time - first_alert["_parsed_time"]).total_seconds() / 3600.0, 3)
                detected_cluster_family_counts[fam] += 1
                lead_times_by_family[fam].append(lead)

    alert_rows = [r for r in rows if r.get("_era_alert")]
    alerts_by_family = defaultdict(list)

    for r in alert_rows:
        alerts_by_family[r.get("_driver_family") or "Unknown"].append(r)

    family_summaries = []

    for family, fam_rows in sorted(alerts_by_family.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        risks = [r.get("_risk_score") for r in fam_rows if r.get("_risk_score") is not None]
        event_window_alerts = sum(1 for r in fam_rows if id(r) in event_window_row_ids)
        critical = sum(1 for r in fam_rows if str(r.get("_priority_tier", "")).lower() == "critical")
        elevated = sum(1 for r in fam_rows if str(r.get("_priority_tier", "")).lower() == "elevated")
        watch = sum(1 for r in fam_rows if str(r.get("_priority_tier", "")).lower() == "watch")

        leads = lead_times_by_family.get(family, [])

        family_summaries.append({
            "driver_family": family,
            "era_alert_rows": len(fam_rows),
            "share_of_era_alerts_pct": round(100.0 * len(fam_rows) / len(alert_rows), 1) if alert_rows else None,
            "event_window_alert_rows": event_window_alerts,
            "event_window_alert_share_pct": round(100.0 * event_window_alerts / len(fam_rows), 1) if fam_rows else None,
            "detected_event_clusters_as_first_driver": detected_cluster_family_counts.get(family, 0),
            "share_of_detected_clusters_pct": round(100.0 * detected_cluster_family_counts.get(family, 0) / sum(detected_cluster_family_counts.values()), 1) if detected_cluster_family_counts else None,
            "median_lead_time_hours": round(statistics.median(leads), 2) if leads else None,
            "critical_alert_rows": critical,
            "elevated_alert_rows": elevated,
            "watch_alert_rows": watch,
            "median_risk_score": round(statistics.median(risks), 3) if risks else None,
            "max_risk_score": round(max(risks), 3) if risks else None,
            "trend_distribution": summarize_counter(Counter(r.get("_trend_direction") or "Unspecified" for r in fam_rows)),
        })

    payload = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "event_window_hours": args.window_hours,
        "event_gap_hours": args.event_gap_hours,
        "rows": len(rows),
        "total_era_alert_rows": len(alert_rows),
        "total_event_clusters": total_event_clusters,
        "detected_event_clusters": sum(detected_cluster_family_counts.values()),
        "driver_family_summaries": family_summaries,
        "public_policy": "Aggregate metrics only. No row-level MIMIC data, no MIMIC IDs, no exact patient-linked timestamps.",
        "pilot_safe_claim": "Retrospective analysis on de-identified MIMIC data showed ERA can provide explainable prioritization context by grouping alerts into aggregate signal-family drivers.",
        "notice": "Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.",
    }

    out_path = Path(args.out_json)
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        existing = {
            "ok": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "DUA-safe aggregate driver / signal-family subgroup validation.",
            "runs": [],
            "public_policy": "Aggregate metrics only. No row-level MIMIC data, no MIMIC IDs, no exact patient-linked timestamps.",
        }

    existing["runs"].append(payload)
    existing["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print("Driver subgroup analysis complete:")
    print("Threshold:", args.threshold)
    print("Rows:", len(rows))
    print("ERA alert rows:", len(alert_rows))
    print("Event clusters:", total_event_clusters)
    print("Detected clusters:", sum(detected_cluster_family_counts.values()))
    for fam in family_summaries[:8]:
        print(
            f" - {fam['driver_family']}: "
            f"alerts={fam['era_alert_rows']}, "
            f"share={fam['share_of_era_alerts_pct']}%, "
            f"detected_clusters={fam['detected_event_clusters_as_first_driver']}, "
            f"median_lead={fam['median_lead_time_hours']}"
        )


if __name__ == "__main__":
    main()
