# eICU Harmonized Clinical-Event Subcohort B Robustness Summary

## Claim Boundary

DUA-safe aggregate retrospective robustness summary only. Do not publish raw eICU files, row-level outputs, patient identifiers, timestamps, or enriched restricted-data exports.

This supports retrospective robustness assessment under a harmonized threshold framework. It does not establish full clinical validation, prospective performance, diagnosis, treatment direction, or autonomous escalation.

## Cohort

- Dataset: eICU v2.0
- Subcohort: B
- Split method: Deterministic patient-level SHA1 bucket split; public output contains aggregate metrics only.
- Cases / stays: 795
- Rows: 668,278

## Threshold Matrix

| Threshold | Suggested setting | Alert reduction | FPR | Detection | Median lead time | ERA alerts / patient-day | Event clusters |
|---|---|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 76.32% | 9.69% | 83.27% | 5.19 hrs | 27.3079 | 245 |
| t=5.0 | Mixed units / balanced | 91.03% | 3.67% | 73.88% | 4.5 hrs | 10.3474 | 245 |
| t=6.0 | Telemetry / stepdown conservative | 95.64% | 1.78% | 64.49% | 3.69 hrs | 5.0296 | 245 |

## Interpretation

This subcohort run tests whether the eICU harmonized clinical-event result remains directionally stable when the dataset is split into deterministic patient subsets.

Safe wording: eICU harmonized clinical-event subcohort robustness was evaluated using aggregate-only outputs. Results should be compared against the full eICU harmonized pass and MIMIC-IV only with clear dataset/event-definition qualifications.
