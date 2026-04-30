# MIMIC-IV + eICU Conservative t=6.0 Aggregate Rollup

## Status

Locked aggregate rollup.

## Safe Cross-Dataset Sentence

Across MIMIC-IV strict clinical-event cohorts and eICU harmonized clinical-event cohorts, the conservative t=6.0 operating point showed directionally consistent aggregate behavior: high alert-reduction, low false-positive burden, and measurable retrospective lead-time context, while detection rates are not treated as directly equivalent because the datasets use different event-definition methods.

## Why Event Definitions Are Separated

MIMIC-IV and eICU should not be described as identical validation endpoints. MIMIC-IV results are framed as strict clinical-event retrospective validation. eICU results are framed as harmonized clinical-event retrospective robustness using fields available in eICU.

The correct public-facing phrasing is:

> Retrospectively evaluated across MIMIC-IV and eICU under a harmonized threshold framework, with event-definition differences clearly labeled.

Avoid saying:

> Validated on two datasets.

unless the same clinical-event definition and the same validation question are applied equivalently across both datasets.

## Conservative t=6.0 Aggregate Table

| Run | Dataset | Event definition | Cases | Rows | t=6 alert reduction | t=6 FPR | t=6 detection | Lead-time context | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| MIMIC-IV full cohort | MIMIC-IV v3.1 | Strict clinical-event labels / event clusters | 1,705 | 456,453 | 94.3% | 4.2% | 15.3% | 4.0 hrs | locked aggregate |
| MIMIC-IV subcohort B | MIMIC-IV v3.1 | Strict clinical-event labels / event clusters | 577 | ~159K | 94.0% | 4.4% | 16.0% | 4.0 hrs | locked aggregate |
| MIMIC-IV subcohort C | MIMIC-IV v3.1 | Strict clinical-event labels / event clusters | 621 | ~159K | 94.9% | 3.7% | 14.1% | 4.0 hrs | locked aggregate |
| eICU harmonized full pass | eICU v2.0 | Harmonized clinical-event labels derived from eICU available fields | 2,394 | 2,023,962 | 94.25% | 0.98% | 24.66% | 4.83 hrs | locked aggregate |
| eICU harmonized subcohort B | eICU v2.0 | Harmonized clinical-event labels derived from eICU available fields | — | — | 95.64% | 1.78% | 64.49% | 3.69 hrs | locked aggregate |
| eICU harmonized subcohort C | eICU v2.0 | Harmonized clinical-event labels derived from eICU available fields | — | — | 95.26% | 2.02% | 64.03% | 3.13 hrs | locked aggregate |

## Aggregate Ranges by Evidence Family

### MIMIC-IV strict clinical-event cohorts

- Alert reduction: 94.0%–94.9%
- FPR: 3.7%–4.4%
- Detection: 14.1%–16.0%
- Median lead-time context: 4.0–4.0 hours

### eICU harmonized clinical-event cohorts

- Alert reduction: 94.25%–95.64%
- FPR: 0.98%–2.02%
- Detection: 24.66%–64.49%
- Median lead-time context: 3.13–4.83 hours

## Approved Claim Language

Early Risk Alert AI has retrospective aggregate evidence across MIMIC-IV strict clinical-event cohorts and eICU harmonized clinical-event cohorts. Across both evidence families, the conservative t=6.0 operating point preserved directionally consistent behavior: high alert-reduction, low false-positive burden, and measurable retrospective lead-time context. Detection rates are reported separately because event definitions differ between datasets.

## Claims To Avoid

- Validated on two datasets.
- Clinically validated on two datasets.
- Predicts deterioration.
- Detects crises early.
- Prevents adverse events.
- Outperforms standard monitoring.
- Replaces clinician judgment.
- Independently triggers escalation.

## DUA-Safe Boundary

This document contains aggregate metrics only. Raw MIMIC/eICU data, row-level outputs, timestamps, patient identifiers, and enriched restricted-data files remain local-only.
