# MIMIC Retrospective Validation Run — April 2026

## Locked Validation Milestone

- Dataset: MIMIC-IV de-identified retrospective validation subset
- Total rows: 456,453
- Total patients: 1,705
- Total clinical events: 21,396
- Event window: 6 hours
- Thresholds tested: 4.0, 5.0, 6.0

## Pilot-Safe Headline

Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable patient-level detection in a 6-hour pre-event window.

## Default Threshold Summary — t=6.0

- ERA patient-level detection: 36.6%
- ERA reading-level sensitivity: 7.8%
- ERA false-positive rate: 4.5%
- Alert reduction: 81.0%
- Standard threshold patient-level detection: 76.6%
- Standard threshold false-positive rate: 24.3%

## Unit-Specific Threshold Framing

| Threshold | Suggested setting | Framing | Patient detection | FPR | Alert reduction |
|---:|---|---|---:|---:|---:|
| 4.0 | ICU / high-acuity | Higher detection, more alerts | 57.2% | 12.4% | 48.8% |
| 5.0 | Mixed units | Balanced detection and alert burden | 46.1% | 7.4% | 69.1% |
| 6.0 | Telemetry / stepdown | Lowest alert burden, most selective | 36.6% | 4.5% | 81.0% |

## Next Validation Metrics To Add

- Median lead time before event
- Alerts per patient-day
- Alerts per event patient

## Score Saturation Presentation Fix

Do not rely on raw peak score alone. Add:

- Priority Tier: Low / Watch / Elevated / Critical
- Rank in Current Review Queue: #1, #2, #3
- Primary Driver: SpO2 decline / HR instability / BP trend / RR elevation
- Trend: Worsening / Stable / Improving
- First Crossed Threshold: hours before current review

## Approved Claim

Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable patient-level detection in a 6-hour pre-event window.

## Claims To Avoid

- ERA predicts deterioration.
- ERA detects crises early.
- ERA prevents adverse events.
- ERA outperforms standard monitoring.
- ERA replaces clinician judgment.
- ERA independently triggers escalation.

## Decision-Support Notice

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
