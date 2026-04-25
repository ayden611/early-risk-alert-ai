# Lead-Time Robustness Public Progress Update

## What Was Completed

Early Risk Alert AI completed a DUA-safe lead-time robustness / event-window sensitivity test.

Test matrix:

- Event windows: 3 hours, 6 hours, 12 hours
- Thresholds: t=4.0, t=5.0, t=6.0
- Public output: aggregate metrics only
- Row-level MIMIC-derived data: local-only

## Primary Finding

Lead-time behavior scaled predictably across 3-hour, 6-hour, and 12-hour retrospective event windows.

## Key t=6.0 Conservative-Mode Progression

| Event Window | Detection | Median Lead Time | Alert Reduction | ERA FPR |
|---:|---:|---:|---:|---:|
| 3h | 9.2% | 2.0 hrs | 94.3% | 4.2% |
| 6h | 15.3% | 4.0 hrs | 94.3% | 4.2% |
| 12h | 22.5% | 7.0 hrs | 94.3% | 4.2% |

## High-Acuity t=4.0 Progression

| Event Window | Detection | Median Lead Time | Alert Reduction | ERA FPR |
|---:|---:|---:|---:|---:|
| 3h | 26.0% | 2.0 hrs | 80.3% | 14.4% |
| 6h | 37.7% | 4.0 hrs | 80.3% | 14.4% |
| 12h | 52.7% | 8.0 hrs | 80.3% | 14.4% |

## Interpretation

This robustness matrix supports a controlled pilot conversation by showing that ERA’s alert-burden behavior remains stable while detection and retrospective lead-time scale with the size of the pre-event review window.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and retrospective lead-time context.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
