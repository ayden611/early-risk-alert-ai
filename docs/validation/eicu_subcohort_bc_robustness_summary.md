# eICU Harmonized Clinical-Event Subcohort B/C Robustness Summary

This DUA-safe summary compares deterministic eICU patient-level subcohorts B and C using aggregate-only outputs.

## Claim Boundary

Retrospective aggregate robustness only. Do not claim full clinical validation, diagnosis, treatment direction, replacement of clinician judgment, or autonomous escalation.

## Subcohort B

- Cases / stays: 795
- Rows: 668,278
- Event rows: 245

| Threshold | Alert reduction | FPR | Detection | Median lead time | ERA alerts / patient-day |
|---|---:|---:|---:|---:|---:|
| t=4.0 | 76.32% | 9.69% | 83.27% | 5.19 hrs | 27.3079 |
| t=5.0 | 91.03% | 3.67% | 73.88% | 4.5 hrs | 10.3474 |
| t=6.0 | 95.64% | 1.78% | 64.49% | 3.69 hrs | 5.0296 |

## Subcohort C

- Cases / stays: 797
- Rows: 660,755
- Event rows: 253

| Threshold | Alert reduction | FPR | Detection | Median lead time | ERA alerts / patient-day |
|---|---:|---:|---:|---:|---:|
| t=4.0 | 73.99% | 11.1% | 82.21% | 5.0 hrs | 31.2642 |
| t=5.0 | 89.85% | 4.33% | 73.52% | 4.22 hrs | 12.2037 |
| t=6.0 | 95.26% | 2.02% | 64.03% | 3.13 hrs | 5.6949 |

## Safe Interpretation

If subcohort B and C preserve the same direction as the full eICU harmonized run — lower thresholds increase detection and higher thresholds reduce review burden/FPR — this strengthens the eICU robustness story before any HiRID work begins.

Raw eICU files and row-level derived outputs remain local-only.
