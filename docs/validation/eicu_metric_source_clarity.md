# eICU Metric Source Clarity

## Purpose

This checkpoint prevents confusion between two separate eICU evidence tracks and prevents case-level review scores from being confused with aggregate validation percentages.

## Metric Type 1 — Case-Level Review Score

A case-level review score is used inside the Command Center review queue.

Correct framing:

- Review Score: 8.7 / 10
- Priority Tier: Critical / Elevated / Watch / Low
- Primary Driver: SpO2 decline, BP instability, HR variability, RR elevation
- Lead-Time Context: retrospective timing context only

Do not label this as a patient-risk percentage.

## Metric Type 2 — Aggregate Validation Percentages

Validation percentages are retrospective aggregate evidence metrics.

Examples:

- Alert reduction: 94.3%
- FPR: 4.2%
- Detection: 15.3%
- Median lead-time context: 4.0 hours

Do not label these as patient-level risk percentages.

## eICU Evidence Track 1 — Outcome-Proxy Check

This was the earlier second-dataset check using mortality/discharge-derived outcome-proxy event context.

Conservative t=6.0 values:

- Alert reduction: 96.8%
- FPR: 1.8%
- Detection: 66.6%
- Median lead-time context: 3.41 hours

## eICU Evidence Track 2 — Harmonized Clinical-Event Pass

This is the newer harmonized clinical-event pass intended to better align the eICU evaluation question with the MIMIC clinical-event framework.

Conservative t=6.0 values:

- Alert reduction: 94.25%
- FPR: 0.98%
- Detection: 24.66%
- Median lead-time context: 4.83 hours

## Interpretation

These are not the same validation question and should not be merged into one unlabeled number.

Correct public framing:

Early Risk Alert AI has retrospective evidence across MIMIC-IV and eICU, including an eICU outcome-proxy check and a newer harmonized eICU clinical-event labeling pass. Detection rates should be interpreted with event-definition context clearly disclosed.

## Claims Guardrail

Do not claim unqualified clinical validation or prospective performance.

Approved framing:

- Retrospective validation evidence.
- De-identified datasets.
- Aggregate metrics only.
- Decision-support and workflow-support only.
- Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
