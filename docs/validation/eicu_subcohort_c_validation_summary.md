# eICU Harmonized Clinical-Event Subcohort C Robustness Summary

## Claim Boundary

DUA-safe aggregate retrospective robustness summary only. Do not publish raw eICU files, row-level outputs, patient identifiers, timestamps, or enriched restricted-data exports.

This supports retrospective robustness assessment under a harmonized threshold framework. It does not establish full clinical validation, prospective performance, diagnosis, treatment direction, or autonomous escalation.

## Cohort

- Dataset: eICU v2.0
- Subcohort: C
- Split method: Deterministic patient-level SHA1 bucket split; public output contains aggregate metrics only.
- Cases / stays: 797
- Rows: 660,755

## Threshold Matrix

| Threshold | Suggested setting | Alert reduction | FPR | Detection | Median lead time | ERA alerts / patient-day | Event clusters |
|---|---|---:|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 73.99% | 11.1% | 82.21% | 5.0 hrs | 31.2642 | 253 |
| t=5.0 | Mixed units / balanced | 89.85% | 4.33% | 73.52% | 4.22 hrs | 12.2037 | 253 |
| t=6.0 | Telemetry / stepdown conservative | 95.26% | 2.02% | 64.03% | 3.13 hrs | 5.6949 | 253 |

## Interpretation

This subcohort run tests whether the eICU harmonized clinical-event result remains directionally stable when the dataset is split into deterministic patient subsets.

Safe wording: eICU harmonized clinical-event subcohort robustness was evaluated using aggregate-only outputs. Results should be compared against the full eICU harmonized pass and MIMIC-IV only with clear dataset/event-definition qualifications.
