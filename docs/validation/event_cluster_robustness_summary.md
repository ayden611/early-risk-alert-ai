# DUA-Safe Event-Cluster Definition Robustness Summary

## Purpose

This test checks whether ERA validation trends remain directionally stable when the event-cluster definition changes.

Fixed setting:

- Pre-event review window: 6 hours

Tested event-cluster gaps:

- 3 hours
- 6 hours
- 12 hours

Tested thresholds:

- t=4.0 — ICU / high-acuity
- t=5.0 — mixed-unit balanced
- t=6.0 — telemetry / stepdown conservative

No row-level MIMIC-derived data, real MIMIC identifiers, or exact patient-linked timestamps are included.

## Results

| Event Gap | Threshold | Setting | Events | Alert Reduction | FPR | Detection | Median Lead Time | ERA Alerts / Patient-Day |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 3.0h | t=4.0 | ICU / high-acuity | 1658 | 80.3% | 14.4% | 37.7% | 4.0 hrs | 2.2154 |
| 3.0h | t=5.0 | Mixed units / balanced | 1658 | 89.1% | 8.0% | 24.4% | 4.0 hrs | 1.2308 |
| 3.0h | t=6.0 | Telemetry / stepdown conservative | 1658 | 94.3% | 4.2% | 15.3% | 4.0 hrs | 0.6467 |
| 6.0h | t=4.0 | ICU / high-acuity | 1658 | 80.3% | 14.4% | 37.7% | 4.0 hrs | 2.2154 |
| 6.0h | t=5.0 | Mixed units / balanced | 1658 | 89.1% | 8.0% | 24.4% | 4.0 hrs | 1.2308 |
| 6.0h | t=6.0 | Telemetry / stepdown conservative | 1658 | 94.3% | 4.2% | 15.3% | 4.0 hrs | 0.6467 |
| 12.0h | t=4.0 | ICU / high-acuity | 1658 | 80.3% | 14.4% | 37.7% | 4.0 hrs | 2.2154 |
| 12.0h | t=5.0 | Mixed units / balanced | 1658 | 89.1% | 8.0% | 24.4% | 4.0 hrs | 1.2308 |
| 12.0h | t=6.0 | Telemetry / stepdown conservative | 1658 | 94.3% | 4.2% | 15.3% | 4.0 hrs | 0.6467 |


## Interpretation Guidance

Use this report to answer:

- Whether ERA results remain directionally stable when event-cluster definitions change.
- Whether t=4.0 continues to provide higher detection.
- Whether t=6.0 continues to provide the lowest alert burden and lowest FPR.
- Whether median lead-time behavior remains meaningful across clustering assumptions.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and retrospective lead-time context across multiple event-cluster assumptions.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
