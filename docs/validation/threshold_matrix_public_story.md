# Threshold Matrix Public Story

## Primary Finding

Median lead time remained **4.0 hours** across all three selected ERA operating points.

| Threshold | Suggested setting | Alert reduction | FPR | Detection | ERA alerts / patient-day | Median lead time |
|---|---|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 80.3% | 14.4% | 37.9% | 2.2154 | 4.0 hrs |
| t=5.0 | Mixed units | 89.1% | 8.0% | 24.6% | 1.2308 | 4.0 hrs |
| t=6.0 | Telemetry / stepdown | 94.3% | 4.2% | 15.3% | 0.6467 | 4.0 hrs |

## Interpretation

The threshold matrix supports a configurable operating-point story:

- **t=4.0** — ICU / high-acuity mode. Higher detection with more alerts.
- **t=5.0** — mixed-unit balanced mode.
- **t=6.0** — telemetry / stepdown conservative mode. Most selective, lowest alert burden, lowest FPR.

## Operational Alert-Burden Finding

At t=6.0, ERA generated approximately **0.6467 alerts per patient-day** versus approximately **11.2743 alerts per patient-day** under standard threshold-only alerting.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and stable retrospective lead-time context.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
