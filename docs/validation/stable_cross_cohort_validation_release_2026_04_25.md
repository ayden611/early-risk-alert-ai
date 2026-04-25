# Stable Cross-Cohort Validation Evidence Release — April 25, 2026

## Release ID

`stable-cross-cohort-validation-release-2026-04-25`

## Locked Evidence Statement

**Across the full validation cohort and two deterministic patient-level subcohorts (577–1,705 cases each), the conservative t=6.0 ERA setting showed consistent low-burden review-queue performance, with 94.0%–94.9% alert reduction, 3.7%–4.4% ERA FPR, 14.1%–16.0% event-cluster detection, and a stable 4.0 hours median lead-time context across all cohorts.**

## What Was Locked

- Cross-cohort validation comparison
- Full validation cohort + Subcohort B + Subcohort C public story
- Conservative t=6.0 range headline
- Validation Intelligence page alignment
- Evidence Packet alignment
- Command Center evidence module alignment
- Operational impact language alignment

## t=6.0 Cross-Cohort Ranges

| Metric | Locked Range |
|---|---:|
| Cases per cohort | 577–1,705 |
| Alert reduction | 94.0%–94.9% |
| ERA FPR | 3.7%–4.4% |
| Event-cluster detection | 14.1%–16.0% |
| Median lead-time context | 4.0 hours |
| ERA alerts / patient-day | 0.5–0.7 |

## Command Center Alignment

The Command Center now reflects the stable cross-cohort release and uses operational impact framing:

- Lower-burden review queue
- Explainable prioritization workflow
- Priority tier / queue rank / driver / trend context
- Validation evidence links

## Public Routes

- `/validation-intelligence`
- `/validation-evidence`
- `/validation-runs`
- `/command-center`
- `/api/validation/cross-cohort-validation`
- `/validation-evidence/cross-cohort-validation.json`

## DUA-Safe Boundary

Public evidence remains aggregate-only.

Do not publish:

- Raw MIMIC CSV files
- Row-level enriched CSV exports
- Real restricted identifiers
- Exact case-linked timestamps
- Patient-level rows
- Any restricted row-level derived data

## Pilot-Safe Claim

Retrospective DUA-safe cross-cohort comparison supports review of ERA operating-point behavior across the full validation cohort and different deterministic case-level subcohorts.

## Claim Guardrails

Avoid:

- ERA predicts deterioration
- ERA detects crises early
- ERA prevents adverse events
- ERA diagnoses
- ERA directs treatment
- ERA outperforms standard monitoring
- ERA replaces clinician judgment
- ERA independently triggers escalation
- Financial ROI claims based only on retrospective validation

Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
