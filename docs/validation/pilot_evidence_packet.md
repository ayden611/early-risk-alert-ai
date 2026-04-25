# Early Risk Alert AI — Pilot Evidence Packet

## MIMIC Retrospective Validation Milestone

**Prepared:** 2026-04-25 07:16 UTC

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform intended to assist authorized health care professionals in identifying monitored patients who may warrant further clinical review, support patient prioritization, and improve command-center operational awareness.

## Pilot-Safe Headline

Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable patient-level detection in a 6-hour pre-event window.

## Key Results — Default Threshold t=6.0

- Rows analyzed: 456,453
- Patients: 1,705
- Clinical events: 21,396
- Event window: 6 hours
- Alert reduction: 81%
- ERA false-positive rate: 4.5%
- Standard threshold false-positive rate: 24.3%
- ERA patient-level detection: 36.6%
- Standard threshold patient-level detection: 76.6%
- Median lead time before event: 4.0 hours among detected event clusters
- ERA alerts per patient-day: 0.6467
- Standard threshold alerts per patient-day: 3.77

## Plain-Language Interpretation

At the conservative t=6.0 operating point, ERA is configured as a selective review-prioritization layer. It reduces alert burden and false positives compared with standard threshold-only alerting, while still flagging a meaningful subset of event patients in the 6-hour pre-event window.

Among detected event clusters, the median time from first ERA flag to documented event timestamp was approximately 4.0 hours within the 6-hour pre-event window.

## Threshold Profiles

| Threshold | Suggested Setting | Framing | Patient Detection | FPR | Alert Reduction |
| --- | --- | --- | --- | --- | --- |
| t=4.0 | ICU / high-acuity | Higher detection, more alerts | 57.2% | 12.4% | 48.8% |
| t=5.0 | Mixed units | Balanced detection and alert burden | 46.1% | 7.4% | 69.1% |
| t=6.0 | Telemetry / stepdown | Lowest alert burden, most selective | 36.6% | 4.5% | 81% |


## Representative Detected Review Examples

These examples illustrate how the command center can present queue rank, lead time, priority tier, primary driver, and score together.

| Rank | Patient ID | Lead Time | Tier | Primary Driver | Score |
| --- | --- | --- | --- | --- | --- |
| #1 | MIMIC-10380296 | 5 hrs | Critical | borderline SpO2 | 9.0 |
| #2 | MIMIC-10415643 | 3.3 hrs | Critical | SpO2 decline | 8.5 |
| #3 | MIMIC-10194776 | 5 hrs | Critical | HR elevation/depression | 7.5 |
| #4 | MIMIC-10354450 | 2 hrs | Critical | borderline SpO2 | 7.5 |
| #5 | MIMIC-10137856 | 5.2 hrs | Critical | SpO2 below 90 | 7.0 |
| #6 | MIMIC-10047824 | 2 hrs | Elevated | borderline SpO2 | 6.5 |
| #7 | MIMIC-10233597 | 3 hrs | Elevated | SpO2 decline | 6.5 |
| #8 | MIMIC-10268877 | 4.3 hrs | Elevated | borderline SpO2 | 6.0 |


## Explainability Notes

ERA should not be evaluated by raw score alone. High scores can cluster near the upper cap, so the command-center display should foreground:

- Priority tier
- Queue rank
- Primary driver
- Trend direction
- Lead time before event
- First threshold-crossing context
- Workflow status

**Composite multi-signal pattern** means multiple monitored signals contributed to prioritization, but no single signal crossed the dominant-driver threshold.

## Approved Claim

Retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable patient-level detection in a 6-hour pre-event window.

## Claims To Avoid

- ERA predicts deterioration.
- ERA detects crises early.
- ERA prevents adverse events.
- ERA outperforms standard monitoring.
- ERA replaces clinician judgment.
- ERA independently triggers escalation.

## Limitations

- Retrospective analysis only.
- De-identified data only.
- Results require independent clinical review.
- This does not establish prospective clinical performance.
- ERA is decision support only and is not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
- Current lead-time computation used available row-level fields and should be repeated using exported ERA risk_score or era_alert columns when available.

## Decision-Support Notice

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
