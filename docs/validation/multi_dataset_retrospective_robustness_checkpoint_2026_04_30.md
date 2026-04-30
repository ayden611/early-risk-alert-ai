# Multi-Dataset Retrospective Robustness Checkpoint — April 30, 2026

## Checkpoint ID

`multi-dataset-retrospective-robustness-checkpoint-2026-04-30`

## Locked Top-Line Statement

**MIMIC-IV established strict clinical-event cross-cohort retrospective stability, while eICU added a separate second-dataset outcome-proxy check; across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.**

At the conservative t=6.0 operating point, MIMIC-IV showed 4 hrs median lead-time context across locked cross-cohort evidence, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.

## Evidence Roles

| Dataset | Role |
|---|---|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort retrospective validation release |
| eICU Collaborative Research Database v2.0 | Separate second-dataset retrospective outcome-proxy check |

## MIMIC-IV Summary

| Item | Value |
|---|---|
| Dataset | MIMIC-IV v3.1 |
| Evidence role | Locked strict clinical-event cross-cohort validation release |
| Cases | 577–1,705 |
| Rows | 456,453 in full validation cohort |
| Event context | event clusters / clinical-event labels |
| Conservative threshold | t=6.0 |
| Alert reduction | 94%–94.9% |
| FPR | 3.7%–4.4% |
| Detection | 14.1%–16% |
| Lead time | 4 hrs |

## eICU Summary

| Item | Value |
|---|---|
| Dataset | eICU Collaborative Research Database v2.0 |
| Evidence role | Second-dataset retrospective outcome-proxy check |
| Cases | 2394 |
| Rows | 2023962 |
| Event context | 772 |
| Conservative threshold | t=6.0 |
| Alert reduction | 96.8% |
| FPR | 1.8% |
| Detection | 66.6% |
| Lead time | 3.41 hrs |

## Interpretation Guardrail

MIMIC-IV uses stricter clinical-event labels. eICU uses outcome-proxy labels from mortality/discharge-derived event context. Detection rates should not be treated as equivalent endpoint definitions.

## Safe Claim

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

## Avoid

- proven generalizability
- prospective validation
- diagnosis
- treatment direction
- prevention of adverse events
- replacement of clinician judgment
- autonomous escalation
- direct superiority over standard monitoring

## Public Boundary

Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.

Decision support only. Retrospective aggregate analysis only.
