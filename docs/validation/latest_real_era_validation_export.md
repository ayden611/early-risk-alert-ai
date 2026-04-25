# ERA Real Validation Export Run

## Run Summary

- Run ID: `era_mimic_20260425_134002_t6_0`
- Dataset: MIMIC-IV de-identified retrospective validation subset
- Source file: `/Users/andreasmith/Desktop/mimic_strict_event_labeled_era_cohort.csv`
- Enriched CSV: `data/validation/latest_era_validation_export.csv`
- Generated: 2026-04-25T13:40:04.993622+00:00
- Threshold: 6.0
- Rows: 456,453
- Patients: 1,705
- Events: 1,658
- Alert reduction: 94.3%
- ERA FPR: 4.2%
- Patient detection: 15.3%
- Median lead time: 4.0 hours
- ERA alerts per patient-day: 0.6467
- Standard threshold alerts per patient-day: 11.2743

## Columns Added

- risk_score
- era_alert
- priority_tier
- primary_driver
- trend_direction
- threshold_crossed_at
- queue_rank
- standard_threshold_alert

## Pilot-Safe Note

Retrospective analysis only. Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
