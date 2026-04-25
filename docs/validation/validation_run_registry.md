# Validation Run Registry

The validation run registry records each retrospective validation run with:

- Run ID
- Dataset
- Rows
- Patients
- Events
- Threshold
- Alert reduction
- FPR
- Patient detection
- Median lead time
- Date generated
- Validation status

## Routes

- `/validation-runs`
- `/api/validation/runs`
- `/validation-evidence/runs.json`
- `/validation-evidence/latest-enriched.csv`

## Enriched Export Columns

- risk_score
- era_alert
- priority_tier
- primary_driver
- trend_direction
- threshold_crossed_at
- queue_rank
- standard_threshold_alert

Decision support only. Retrospective analysis only.
