# Real-Engine DUA-Safe Validation Public Framing

## Public Hero

At conservative threshold t=6.0, ERA generated approximately 94.3% fewer alerts than standard threshold-only alerting while maintaining a low 4.2% false-positive rate.

## Lead-Time Framing

Among detected event clusters, the median time from first ERA flag to documented event timestamp was approximately 4.0 hours within the 6-hour pre-event window.

## Threshold Story

| Threshold | Suggested setting | Framing |
|---|---|---|
| t=4.0 | ICU / high-acuity | Higher detection, more alerts |
| t=5.0 | Mixed units | Balanced detection and alert burden |
| t=6.0 | Telemetry / stepdown | Lowest alert burden, most selective |

## Key Point

t=6.0 is intentionally selective. It is not the best threshold for maximum detection. It is the conservative review-queue setting optimized for alert-burden reduction and low false positives.

t=4.0 is better for high-acuity review when the goal is higher detection with more alert volume.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and retrospective lead-time context.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
