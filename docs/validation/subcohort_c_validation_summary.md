# DUA-Safe Subcohort C Validation Summary

## Purpose

This report summarizes a validation run on a second different deterministic patient/case-level MIMIC subcohort.

Subset definition:

- Subcohort: C
- Selection method: deterministic case-level hash buckets 66 through 99
- Row-level subset file: local-only
- Public output: aggregate metrics only

No row-level MIMIC-derived data, real identifiers, or exact case-linked timestamps are included.

## Threshold Matrix on Subcohort C

| Threshold | Suggested Setting | Rows | Cases | Alert Reduction | ERA FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 159397 | 621 | 81.3% | 13.6% | 36.4% | 4.0 hrs | 2.4812 |
| t=5.0 | Mixed units / balanced | 159397 | 621 | 89.9% | 7.4% | 23.1% | 4.0 hrs | 1.3453 |
| t=6.0 | Telemetry / stepdown conservative | 159397 | 621 | 94.9% | 3.7% | 14.1% | 4.0 hrs | 0.6777 |


## Interpretation

Use this Subcohort C run to compare whether the threshold story remains directionally consistent outside the full cohort and outside Subcohort B.

Expected operating-point pattern:

- t=4.0: higher detection, more alerts
- t=5.0: balanced review burden
- t=6.0: most selective, lowest alert burden and FPR

## Pilot-Safe Claim

Retrospective DUA-safe subcohort testing supports review of ERA operating-point behavior on an additional patient/case-level subset.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
