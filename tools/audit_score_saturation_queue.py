#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def find_col(headers, candidates):
    lower = {str(h).lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


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


def fnum(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def pct(n, d):
    if not d:
        return None
    return round(100.0 * n / d, 2)


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return round(statistics.median(vals), 3)


def q(values, percentile):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * percentile))))
    return round(vals[idx], 3)


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


def computed_tier(score):
    if score is None:
        return "Unscored"
    if score >= 6.0:
        return "Critical"
    if score >= 5.0:
        return "Elevated"
    if score >= 4.0:
        return "Watch"
    return "Low"


def score_bin(score):
    if score is None:
        return "Unscored"
    if score < 2:
        return "<2"
    if score < 4:
        return "2–3.9"
    if score < 5:
        return "4–4.9"
    if score < 6:
        return "5–5.9"
    if score < 7:
        return "6–6.9"
    if score < 8:
        return "7–7.9"
    if score < 9:
        return "8–8.9"
    return "9+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched-csv", default="data/validation/latest_era_validation_export.csv")
    ap.add_argument("--out-json", default="data/validation/score_saturation_queue_audit_summary.json")
    ap.add_argument("--out-md", default="docs/validation/score_saturation_queue_audit_summary.md")
    args = ap.parse_args()

    csv_path = Path(args.enriched_csv)
    if not csv_path.exists():
        raise SystemExit(f"ERROR: enriched CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise SystemExit("ERROR: enriched CSV has no rows.")

    case_col = find_col(headers, ["patient_id", "case_id", "subject_id"])
    time_col = find_col(headers, ["timestamp", "charttime", "time"])
    risk_col = find_col(headers, ["risk_score", "era_risk_score", "score"])
    tier_col = find_col(headers, ["priority_tier", "tier"])
    driver_col = find_col(headers, ["primary_driver", "driver", "dominant_driver"])
    trend_col = find_col(headers, ["trend_direction", "trend"])

    if not risk_col:
        raise SystemExit(f"ERROR: risk_score column not found. Found: {headers}")

    parsed = []
    all_scores = []

    for r in rows:
        score = fnum(r.get(risk_col))
        ts = parse_time(r.get(time_col)) if time_col else None
        case_id = str(r.get(case_col, "")).strip() if case_col else ""

        if score is not None:
            all_scores.append(score)

        parsed.append({
            "case_id": case_id,
            "timestamp": ts,
            "score": score,
            "computed_tier": computed_tier(score),
            "source_tier": str(r.get(tier_col, "")).strip() if tier_col else "",
            "driver_family": classify_family(r.get(driver_col)) if driver_col else "Driver not available",
            "trend": str(r.get(trend_col, "")).strip() if trend_col else "Trend not available",
        })

    thresholds = [4.0, 5.0, 6.0]
    threshold_results = []

    total_rows = len(parsed)
    total_cases = len(set(p["case_id"] for p in parsed if p["case_id"]))

    global_score_distribution = Counter(score_bin(p["score"]) for p in parsed)
    global_tier_distribution = Counter(p["computed_tier"] for p in parsed)

    for threshold in thresholds:
        alert_rows = [p for p in parsed if p["score"] is not None and p["score"] >= threshold]
        alert_count = len(alert_rows)
        alert_cases = len(set(p["case_id"] for p in alert_rows if p["case_id"]))

        high_score_rows = [p for p in alert_rows if p["score"] is not None and p["score"] >= 9.0]
        saturated_rows = [p for p in alert_rows if p["score"] is not None and p["score"] >= 9.9]

        queue_by_time = defaultdict(int)
        if time_col:
            for p in alert_rows:
                if p["timestamp"]:
                    queue_by_time[p["timestamp"]] += 1

        queue_sizes = list(queue_by_time.values())

        driver_dist = Counter(p["driver_family"] for p in alert_rows)
        tier_dist = Counter(p["computed_tier"] for p in alert_rows)
        trend_dist = Counter(p["trend"] or "Unspecified" for p in alert_rows)

        top_driver_share = None
        if driver_dist and alert_count:
            top_driver_share = round(100.0 * max(driver_dist.values()) / alert_count, 2)

        threshold_results.append({
            "threshold": threshold,
            "suggested_setting": (
                "ICU / high-acuity" if threshold == 4.0 else
                "Mixed units / balanced" if threshold == 5.0 else
                "Telemetry / stepdown conservative"
            ),
            "total_rows": total_rows,
            "alert_rows": alert_count,
            "alert_row_pct": pct(alert_count, total_rows),
            "unique_cases_with_alerts": alert_cases,
            "unique_case_alert_pct": pct(alert_cases, total_cases),
            "score_9_plus_rows": len(high_score_rows),
            "score_9_plus_pct_of_alert_rows": pct(len(high_score_rows), alert_count),
            "score_9_9_plus_rows": len(saturated_rows),
            "score_9_9_plus_pct_of_alert_rows": pct(len(saturated_rows), alert_count),
            "queue_timepoints": len(queue_sizes),
            "median_queue_size_per_timepoint": median(queue_sizes),
            "p90_queue_size_per_timepoint": q(queue_sizes, 0.90),
            "p95_queue_size_per_timepoint": q(queue_sizes, 0.95),
            "max_queue_size_per_timepoint": max(queue_sizes) if queue_sizes else None,
            "driver_family_distribution": dict(driver_dist),
            "top_driver_share_pct": top_driver_share,
            "tier_distribution": dict(tier_dist),
            "trend_distribution": dict(trend_dist),
            "interpretation": "Use queue rank, priority tier, primary driver, and trend context to avoid relying on raw score alone."
        })

    payload = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "DUA-safe aggregate score saturation and review-queue burden audit.",
        "rows": total_rows,
        "unique_cases": total_cases,
        "risk_score_coverage_pct": pct(len(all_scores), total_rows),
        "score_summary": {
            "median": median(all_scores),
            "p90": q(all_scores, 0.90),
            "p95": q(all_scores, 0.95),
            "p99": q(all_scores, 0.99),
            "max": max(all_scores) if all_scores else None,
            "score_bin_distribution": dict(global_score_distribution),
            "computed_tier_distribution": dict(global_tier_distribution),
        },
        "threshold_results": threshold_results,
        "command_center_implication": {
            "primary_point": "The command center should emphasize queue rank, priority tier, primary driver, trend direction, and lead-time context instead of raw score alone.",
            "recommended_display_fields": [
                "Queue Rank",
                "Priority Tier",
                "Risk Score",
                "Primary Driver",
                "Trend Direction",
                "First Threshold Crossing / Lead-Time Context"
            ],
            "pilot_safe_note": "Raw score saturation should be interpreted as a reason to show richer prioritization context, not as a standalone clinical conclusion."
        },
        "public_policy": "Aggregate metrics only. No row-level MIMIC data, no MIMIC IDs, no exact patient-linked timestamps.",
        "pilot_safe_claim": "A DUA-safe aggregate audit showed ERA review queues can be summarized by threshold, score distribution, priority tier, and driver-family context without exposing row-level data.",
        "notice": "Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation."
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def val(x, suffix=""):
        if x is None or x == "":
            return "—"
        return f"{x}{suffix}"

    md = f"""# DUA-Safe Score Saturation + Review-Queue Burden Audit

## Purpose

This audit checks whether the command-center display should rely on raw risk score alone or use richer prioritization context.

It summarizes:

- Risk score distribution
- High-score / saturation rates
- Review-queue burden by threshold
- Queue size distribution
- Driver-family distribution
- Priority-tier distribution

No row-level MIMIC-derived data, real identifiers, or exact patient-linked timestamps are included.

## Overall Score Distribution

| Metric | Value |
|---|---:|
| Rows | {payload['rows']} |
| Unique cases | {payload['unique_cases']} |
| Risk score coverage | {val(payload['risk_score_coverage_pct'], '%')} |
| Median score | {val(payload['score_summary']['median'])} |
| P90 score | {val(payload['score_summary']['p90'])} |
| P95 score | {val(payload['score_summary']['p95'])} |
| P99 score | {val(payload['score_summary']['p99'])} |
| Max score | {val(payload['score_summary']['max'])} |

## Threshold Review-Queue Summary

| Threshold | Setting | Alert Rows | Alert Row % | Cases With Alerts | Score ≥9 % of Alerts | Score ≥9.9 % of Alerts | Median Queue Size | P95 Queue Size | Max Queue Size |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""

    for r in threshold_results:
        md += (
            f"| t={r['threshold']} "
            f"| {r['suggested_setting']} "
            f"| {val(r['alert_rows'])} "
            f"| {val(r['alert_row_pct'], '%')} "
            f"| {val(r['unique_cases_with_alerts'])} "
            f"| {val(r['score_9_plus_pct_of_alert_rows'], '%')} "
            f"| {val(r['score_9_9_plus_pct_of_alert_rows'], '%')} "
            f"| {val(r['median_queue_size_per_timepoint'])} "
            f"| {val(r['p95_queue_size_per_timepoint'])} "
            f"| {val(r['max_queue_size_per_timepoint'])} |\n"
        )

    md += """
## Interpretation

This audit supports the current score-saturation fix:

- Do not rely on raw high score alone.
- Show queue rank.
- Show priority tier.
- Show primary driver.
- Show trend direction.
- Show lead-time / first-threshold context when available.

## Pilot-Safe Claim

A DUA-safe aggregate audit showed ERA review queues can be summarized by threshold, score distribution, priority tier, and driver-family context without exposing row-level data.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""

    Path(args.out_md).write_text(md, encoding="utf-8")

    print("WROTE:", args.out_json)
    print("WROTE:", args.out_md)
    print("Rows:", total_rows)
    print("Unique cases:", total_cases)
    print("Risk coverage:", payload["risk_score_coverage_pct"])
    for r in threshold_results:
        print(
            f"t={r['threshold']} | alerts={r['alert_rows']} | "
            f"score>=9={r['score_9_plus_pct_of_alert_rows']}% | "
            f"score>=9.9={r['score_9_9_plus_pct_of_alert_rows']}% | "
            f"median_queue={r['median_queue_size_per_timepoint']} | "
            f"p95_queue={r['p95_queue_size_per_timepoint']}"
        )


if __name__ == "__main__":
    main()
