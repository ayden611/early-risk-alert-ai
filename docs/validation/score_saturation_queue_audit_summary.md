# DUA-Safe Score Saturation + Review-Queue Burden Audit

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
| Rows | 456453 |
| Unique cases | 1705 |
| Risk score coverage | 100.0% |
| Median score | 1.5 |
| P90 score | 4.5 |
| P95 score | 5.5 |
| P99 score | 8.0 |
| Max score | 9.9 |

## Threshold Review-Queue Summary

| Threshold | Setting | Alert Rows | Alert Row % | Cases With Alerts | Score ≥9 % of Alerts | Score ≥9.9 % of Alerts | Median Queue Size | P95 Queue Size | Max Queue Size |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 65821 | 14.42% | 1668 | 3.53% | 1.66% | 1 | 1 | 4 |
| t=5.0 | Mixed units / balanced | 36568 | 8.01% | 1590 | 6.36% | 2.99% | 1 | 1 | 3 |
| t=6.0 | Telemetry / stepdown conservative | 19215 | 4.21% | 1429 | 12.11% | 5.68% | 1 | 1 | 3 |

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
