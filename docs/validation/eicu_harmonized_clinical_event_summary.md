# eICU Harmonized Clinical-Event Labeling Summary

Generated: 2026-04-30T14:21:29.960154+00:00

## Purpose

This run creates a harmonized eICU clinical-event labeling pass so eICU can be compared more directly to the MIMIC-IV clinical-event validation question.

## DUA Safety

- Public output is aggregate-only.
- Row-level eICU files remain local-only.
- No raw restricted files or patient-level rows should be committed.

## Cohort Summary

- Rows analyzed: 2,023,962
- Patients/stays: 2,394
- Harmonized clinical-event clusters: 12,948
- Event-gap window: 6 hours
- Pre-event review window: 6 hours

## Event Definition

An event candidate is created when at least one extreme physiologic signal is present or at least two severe physiologic signals occur together. Candidate rows are collapsed into event clusters separated by the configured event-gap window.

## Threshold Matrix

| Threshold | Alert reduction | FPR outside event windows | Detection | Events detected / total | Median lead time | ERA alerts / patient-day | Standard alerts / patient-day |
|---|---:|---:|---:|---:|---:|---:|---:|
| t=4 | 69.74% | 5.34% | 62.51% | 8094/12948 | 5.67 hrs | 29.2181 | 96.5661 |
| t=5 | 87.85% | 2.23% | 38.8% | 5024/12948 | 5.17 hrs | 11.733 | 96.5661 |
| t=6 | 94.25% | 0.98% | 24.66% | 3193/12948 | 4.83 hrs | 5.5536 | 96.5661 |

## Interpretation Boundary

This is the first harmonized eICU clinical-event labeling pass. It is closer to the MIMIC-IV clinical-event question than the prior outcome-proxy check, but should be reviewed before using unqualified two-dataset validation language.

## Safe Public Wording

> eICU now has a harmonized clinical-event labeling pass that can be compared against MIMIC-IV using the same threshold matrix and pre-event window.

## Wording to Avoid Until Review

> Fully validated on two datasets.

Use that only after the event definitions, denominators, and metric logic are reviewed and locked.
