# Early Risk Alert AI — Model Card

## Intended Use

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.

It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.

## Current Evidence Status

The current evidence package includes:

- MIMIC-IV strict clinical-event cross-cohort retrospective validation.
- eICU second-dataset retrospective outcome-proxy check.
- DUA-safe aggregate-only public outputs.
- Local-only handling of restricted raw data and row-level derived outputs.

## Dataset Roles

| Dataset | Role | Event Definition |
|---|---|---|
| MIMIC-IV v3.1 | Strict clinical-event cross-cohort retrospective validation release | Clinical-event labels / event clusters |
| eICU v2.0 | Separate second-dataset retrospective outcome-proxy check | Mortality/discharge-derived outcome proxy |

## Conservative Operating Point

| Dataset | Alert Reduction | FPR | Detection | Lead-Time Context |
|---|---:|---:|---:|---:|
| MIMIC-IV t=6.0 | 94%–94.9% | 3.7%–4.4% | 14.1%–16% | 4 hrs |
| eICU t=6.0 | 96.8% | 1.8% | 66.6% | 3.41 hrs |

## Interpretation Guardrail

MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context.

## Explainability Outputs

The platform should display:

- Priority Tier
- Queue Rank
- Primary Driver
- Trend Direction
- Lead-Time Context
- Review-state workflow actions

## Limitations

- Retrospective analysis only.
- No prospective clinical validation yet.
- MIMIC-IV and eICU event definitions are not identical.
- Detection rates should not be compared as equivalent endpoints.
- The system is not autonomous and does not direct treatment.

## Governance Controls

- DUA-safe aggregate public summaries.
- Raw restricted files kept local-only.
- Row-level enriched exports kept local-only.
- Conservative claim guardrails.
- Decision-support-only disclaimers.

## Claims to Avoid

- Proven generalizability.
- Prospective validation.
- Diagnosis.
- Treatment direction.
- Prevention of adverse events.
- Replacement of clinician judgment.
- Autonomous escalation.
- Direct superiority over standard monitoring.
