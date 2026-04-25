# DUA-Safe ERA Threshold Test Matrix

## Purpose

This matrix compares selected ERA operating points using aggregate retrospective validation outputs only.

No row-level MIMIC-derived data, real MIMIC identifiers, or exact patient-linked timestamps are included.

## Threshold Matrix

| Threshold | Suggested setting | Framing | Alert reduction | FPR | Patient detection | Median lead time | ERA alerts/patient-day | Standard alerts/patient-day |
|---|---|---|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | Higher detection, more alerts; useful when higher sensitivity is preferred. | 80.3% | 14.4% | 37.9% | 4.0 hrs | 2.2154 | 11.2743 |
| t=5.0 | Mixed units | Balanced detection and alert burden. | 89.1% | 8.0% | 24.6% | 4.0 hrs | 1.2308 | 11.2743 |
| t=6.0 | Telemetry / stepdown | Most conservative; optimized for low false positives and alert-burden reduction. | 94.3% | 4.2% | 15.3% | 4.0 hrs | 0.6467 | 11.2743 |


## Recommended Public Framing

- **t=4.0** — ICU / high-acuity mode. Higher detection with more alerts.
- **t=5.0** — mixed-unit balanced mode.
- **t=6.0** — telemetry / stepdown conservative mode. Most selective, lowest alert burden.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and retrospective lead-time context.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
