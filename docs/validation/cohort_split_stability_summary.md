# DUA-Safe Patient-Level Cohort Split Stability Summary

## Purpose

This test checks whether ERA validation trends remain directionally stable across patient/case-level cohort splits.

Test setup:

- 3 deterministic patient/case-level folds
- Thresholds: t=4.0, t=5.0, t=6.0
- Event window: 6 hours
- Event-cluster gap: 6 hours
- Public output: aggregate metrics only

No row-level MIMIC-derived data, real identifiers, or exact patient-linked timestamps are included.

## Fold-Level Results

| Fold | Threshold | Setting | Rows | Cases | Alert Reduction | FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | t=4.0 | ICU / high-acuity | 147301 | 585 | 80.5% | 14.3% | 38.1% | 4.0 hrs | 2.4591 |
| 1 | t=5.0 | Mixed units / balanced | 147301 | 585 | 89.0% | 8.0% | 24.1% | 4.0 hrs | 1.3816 |
| 1 | t=6.0 | Telemetry / stepdown conservative | 147301 | 585 | 94.2% | 4.2% | 15.6% | 4.0 hrs | 0.7299 |
| 2 | t=4.0 | ICU / high-acuity | 150586 | 545 | 80.1% | 14.8% | 37.4% | 4.0 hrs | 2.0353 |
| 2 | t=5.0 | Mixed units / balanced | 150586 | 545 | 88.9% | 8.3% | 25.2% | 4.0 hrs | 1.1393 |
| 2 | t=6.0 | Telemetry / stepdown conservative | 150586 | 545 | 94.1% | 4.4% | 15.8% | 3.75 hrs | 0.6043 |
| 3 | t=4.0 | ICU / high-acuity | 158566 | 575 | 80.4% | 14.1% | 37.6% | 4.0 hrs | 2.2047 |
| 3 | t=5.0 | Mixed units / balanced | 158566 | 575 | 89.3% | 7.7% | 24.1% | 4.0 hrs | 1.2027 |
| 3 | t=6.0 | Telemetry / stepdown conservative | 158566 | 575 | 94.5% | 4.0% | 14.4% | 4.0 hrs | 0.6226 |


## Threshold Stability Summary

| Threshold | Mean Alert Reduction | SD | Mean FPR | SD | Mean Detection | SD | Mean Median Lead Time | SD | Mean ERA Alerts / Patient-Day | SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t=4.0 | 80.333% | 0.208 | 14.4% | 0.361 | 37.7% | 0.361 | 4.0 hrs | 0.0 | 2.233 | 0.213 |
| t=5.0 | 89.067% | 0.208 | 8.0% | 0.3 | 24.467% | 0.635 | 4.0 hrs | 0.0 | 1.241 | 0.126 |
| t=6.0 | 94.267% | 0.208 | 4.2% | 0.2 | 15.267% | 0.757 | 3.917 hrs | 0.144 | 0.652 | 0.068 |


## Interpretation Guidance

Use this report to answer:

- Whether ERA performance trends remain consistent across patient-level folds.
- Whether t=4.0 continues to provide higher detection.
- Whether t=6.0 continues to provide the lowest alert burden and lowest FPR.
- Whether lead-time behavior remains meaningful across cohort splits.

## Pilot-Safe Claim

Retrospective DUA-safe cohort split testing showed ERA operating-point trends can be reviewed across patient-level folds.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
