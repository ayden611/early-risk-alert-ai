# eICU DUA-Safe Validation Summary

## Purpose

This report summarizes a DUA-safe aggregate threshold validation matrix on a local-only ERA-format eICU cohort.

This is a second-dataset retrospective outcome-proxy check.

## Important Claim Guardrail

This eICU run uses outcome-proxy event labels derived from mortality/discharge status fields and relative ICU offsets.

Use:

**Cross-dataset retrospective outcome-proxy evaluation on de-identified ICU data.**

Avoid:

- proven generalizability
- prospective validation
- direct deterioration-event validation
- diagnosis claim
- treatment direction claim
- autonomous escalation claim
- replacement of clinician judgment

## eICU Threshold Matrix

| Threshold | Suggested Setting | Rows | Cases | Event-Proxy Rows | Alert Reduction | ERA FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 2023962 | 2394 | 772 | 87.7% | 7.3% | 81.3% | 4.89 hrs | 21.7953 |
| t=5.0 | Mixed units / balanced | 2023962 | 2394 | 772 | 93.5% | 3.7% | 75.1% | 4.32 hrs | 11.4405 |
| t=6.0 | Telemetry / stepdown conservative | 2023962 | 2394 | 772 | 96.8% | 1.8% | 66.6% | 3.41 hrs | 5.7468 |


## Interpretation

Use this run to compare whether ERA’s threshold behavior remains directionally understandable on a second ICU dataset.

Expected operating-point pattern:

- t=4.0: higher detection, more alerts
- t=5.0: balanced review burden
- t=6.0: most selective, lowest alert burden and FPR

## DUA-Safe Boundary

Raw eICU files and row-level ERA-format output remain local-only.

Public evidence may include aggregate metrics only.

Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.


<!-- ERA_EICU_SECOND_DATASET_PUBLIC_V1_START -->

## eICU Second-Dataset Outcome-Proxy Check

**On a separate eICU outcome-proxy cohort, ERA preserved the same threshold pattern: lower thresholds produced higher detection, while the conservative t=6.0 setting emphasized low-burden review behavior.**

This eICU run is intentionally framed separately from the locked MIMIC-IV evidence release because it uses outcome-proxy event labels from eICU mortality/discharge status fields and relative ICU offsets.

| Threshold | Setting | Rows | Cases | Event-Proxy Rows | Alert Reduction | ERA FPR | Detection | Lead Time | Alerts / Patient-Day |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t=4 | ICU / high-acuity | 2023962 | 2394 | 772 | 87.7% | 7.3% | 81.3% | 4.89 hrs | 21.8 |
| t=5 | Mixed units / balanced | 2023962 | 2394 | 772 | 93.5% | 3.7% | 75.1% | 4.32 hrs | 11.44 |
| t=6 | Telemetry / stepdown conservative | 2023962 | 2394 | 772 | 96.8% | 1.8% | 66.6% | 3.41 hrs | 5.75 |


Safe claim: cross-dataset retrospective outcome-proxy evaluation on de-identified ICU data.

Avoid claiming proven generalizability, prospective validation, diagnosis, treatment direction, prevention, or autonomous escalation.

<!-- ERA_EICU_SECOND_DATASET_PUBLIC_V1_END -->
