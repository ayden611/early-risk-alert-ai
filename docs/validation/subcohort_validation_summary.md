# DUA-Safe Subcohort Validation Summary

## Purpose

This report summarizes a new validation run on a different deterministic patient/case-level MIMIC subcohort.

Subset definition:

- Subcohort: B
- Selection method: deterministic case-level hash buckets 33 through 65
- Row-level subset file: local-only
- Public output: aggregate metrics only

No row-level MIMIC-derived data, real identifiers, or exact case-linked timestamps are included.

## Threshold Matrix on Subcohort B

| Threshold | Suggested Setting | Rows | Cases | Alert Reduction | ERA FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 159692 | 577 | 80.2% | 14.6% | 39.2% | 4.0 hrs | 1.6197 |
| t=5.0 | Mixed units / balanced | 159692 | 577 | 88.8% | 8.2% | 25.5% | 4.0 hrs | 0.9121 |
| t=6.0 | Telemetry / stepdown conservative | 159692 | 577 | 94.0% | 4.4% | 16.0% | 4.0 hrs | 0.4862 |


## Interpretation

Use this subcohort result to compare whether the threshold story remains directionally consistent outside the original full-cohort summary.

The expected operating-point pattern is:

- t=4.0: higher detection, more alerts
- t=5.0: balanced review burden
- t=6.0: most selective, lowest alert burden and FPR

## Pilot-Safe Claim

Retrospective DUA-safe subcohort testing supports review of ERA operating-point behavior on a different patient/case-level subset.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
