# Early Risk Alert AI — Multi-Dataset Pilot Summary

## Top-Line Summary

Early Risk Alert AI now has retrospective evidence across two de-identified ICU datasets: MIMIC-IV strict clinical-event cross-cohort validation plus a separate eICU outcome-proxy check.

Across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.

At the conservative t=6.0 setting, MIMIC-IV showed 4 hrs median lead-time context across the locked cross-cohort release, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.

## Why This Matters

The evidence package no longer rests on a single dataset. MIMIC-IV supports strict clinical-event cross-cohort retrospective stability, while eICU provides a separate second-dataset outcome-proxy check.

## Conservative t=6.0 Summary

| Dataset | Event Definition | Cases | Rows | Alert Reduction | FPR | Detection | Lead-Time Context |
|---|---|---:|---:|---:|---:|---:|---:|
| MIMIC-IV v3.1 | Strict clinical-event labels / event clusters | 577–1,705 | 456,453 in full validation cohort | 94%–94.9% | 3.7%–4.4% | 14.1%–16% | 4 hrs |
| eICU v2.0 | Outcome-proxy labels from mortality/discharge-derived event context | 2394 | 2023962 | 96.8% | 1.8% | 66.6% | 3.41 hrs |

## Interpretation Guardrail

MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context.

The detection percentages are not meant to be compared directly as identical clinical endpoints. The stronger finding is that ERA preserved the same threshold-direction behavior across both datasets.

## Approved Framing

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

## Avoid

- validated on two datasets without qualification
- proven generalizability
- prospective validation
- ERA predicts deterioration
- ERA prevents adverse events
- ERA diagnoses
- ERA directs treatment
- ERA replaces clinician judgment
- ERA independently triggers escalation
- direct superiority over standard monitoring

## Decision-Support Boundary

Early Risk Alert AI is HCP-facing decision support and workflow support. It is not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
