# DUA-Safe Lead-Time Robustness / Event-Window Sensitivity Summary

## Purpose

This test checks whether ERA lead-time behavior remains meaningful across different retrospective pre-event windows.

Tested event windows:

- 3 hours
- 6 hours
- 12 hours

Tested thresholds:

- t=4.0 — ICU / high-acuity
- t=5.0 — mixed-unit balanced
- t=6.0 — telemetry / stepdown conservative

No row-level MIMIC-derived data, real MIMIC identifiers, or exact patient-linked timestamps are included.

## Results

| Window | Threshold | Setting | Alert Reduction | FPR | Detection | Median Lead Time | >1h | >2h | >3h | ERA Alerts / Patient-Day |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0h | t=4.0 | ICU / high-acuity | 80.3% | 14.4% | 26.0% | 2.0 hrs | 95.8% | 68.2% | 39.4% | 2.2154 |
| 3.0h | t=5.0 | Mixed units / balanced | 89.1% | 8.0% | 16.3% | 2.0 hrs | 93.3% | 67.0% | 33.7% | 1.2308 |
| 3.0h | t=6.0 | Telemetry / stepdown conservative | 94.3% | 4.2% | 9.2% | 2.0 hrs | 92.2% | 66.0% | 34.0% | 0.6467 |
| 6.0h | t=4.0 | ICU / high-acuity | 80.3% | 14.4% | 37.7% | 4.0 hrs | 97.9% | 84.0% | 73.0% | 2.2154 |
| 6.0h | t=5.0 | Mixed units / balanced | 89.1% | 8.0% | 24.4% | 4.0 hrs | 96.0% | 83.2% | 68.9% | 1.2308 |
| 6.0h | t=6.0 | Telemetry / stepdown conservative | 94.3% | 4.2% | 15.3% | 4.0 hrs | 95.7% | 83.4% | 71.1% | 0.6467 |
| 12.0h | t=4.0 | ICU / high-acuity | 80.3% | 14.4% | 52.7% | 8.0 hrs | 98.5% | 91.9% | 86.7% | 2.2154 |
| 12.0h | t=5.0 | Mixed units / balanced | 89.1% | 8.0% | 34.7% | 7.11 hrs | 97.6% | 90.6% | 83.0% | 1.2308 |
| 12.0h | t=6.0 | Telemetry / stepdown conservative | 94.3% | 4.2% | 22.5% | 7.0 hrs | 97.3% | 90.1% | 82.8% | 0.6467 |


## Interpretation Guidance

Use this report to answer:

- Whether lead time remains meaningful when the event window changes.
- Whether t=4.0 continues to provide higher detection.
- Whether t=6.0 continues to provide the lowest alert burden and lowest FPR.
- Whether alert burden remains defensible across operating points.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and retrospective lead-time context.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
