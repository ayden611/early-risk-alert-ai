# DUA-Safe Cross-Cohort Validation Comparison

## Purpose

This report compares ERA validation behavior across the full validation cohort and different deterministic patient/case-level subcohorts.

Public output is aggregate only.

No row-level MIMIC-derived data, real identifiers, or exact case-linked timestamps are included.

## Cohorts Compared

- Full validation cohort
- Subcohort B
- Subcohort C


## Run-Level Comparison

| Cohort | Threshold | Setting | Rows | Cases | Alert Reduction | ERA FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Full validation cohort | t=4.0 | ICU / high-acuity | 456453 | 1705 | 80.3% | 14.4% | 37.9% | 4.0 hrs | 2.2154 |
| Full validation cohort | t=5.0 | Mixed units | 456453 | 1705 | 89.1% | 8.0% | 24.6% | 4.0 hrs | 1.2308 |
| Full validation cohort | t=6.0 | Telemetry / stepdown | 456453 | 1705 | 94.3% | 4.2% | 15.3% | 4.0 hrs | 0.6467 |
| Subcohort B | t=4.0 | ICU / high-acuity | 159692 | 577 | 80.2% | 14.6% | 39.2% | 4.0 hrs | 1.6197 |
| Subcohort B | t=5.0 | Mixed units / balanced | 159692 | 577 | 88.8% | 8.2% | 25.5% | 4.0 hrs | 0.9121 |
| Subcohort B | t=6.0 | Telemetry / stepdown conservative | 159692 | 577 | 94.0% | 4.4% | 16.0% | 4.0 hrs | 0.4862 |
| Subcohort C | t=4.0 | ICU / high-acuity | 159397 | 621 | 81.3% | 13.6% | 36.4% | 4.0 hrs | 2.4812 |
| Subcohort C | t=5.0 | Mixed units / balanced | 159397 | 621 | 89.9% | 7.4% | 23.1% | 4.0 hrs | 1.3453 |
| Subcohort C | t=6.0 | Telemetry / stepdown conservative | 159397 | 621 | 94.9% | 3.7% | 14.1% | 4.0 hrs | 0.6777 |


## Threshold-Level Stability Summary

| Threshold | Cohorts Compared | Mean Alert Reduction | SD | Mean ERA FPR | SD | Mean Detection | SD | Mean Lead Time | SD | Mean ERA Alerts / Patient-Day | SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t=4.0 | 3 | 80.6% | 0.608 | 14.2% | 0.529 | 37.833% | 1.401 | 4.0 hrs | 0.0 | 2.105 | 0.441 |
| t=5.0 | 3 | 89.267% | 0.569 | 7.867% | 0.416 | 24.4% | 1.212 | 4.0 hrs | 0.0 | 1.163 | 0.224 |
| t=6.0 | 3 | 94.4% | 0.458 | 4.1% | 0.361 | 15.133% | 0.961 | 4.0 hrs | 0.0 | 0.604 | 0.103 |


## Interpretation

Use this report to show that ERA’s operating-point story can be evaluated across multiple cohort slices:

- t=4.0 supports higher-detection / high-acuity review.
- t=5.0 supports a balanced review mode.
- t=6.0 supports a conservative, low-burden review queue.

## Pilot-Safe Claim

Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
