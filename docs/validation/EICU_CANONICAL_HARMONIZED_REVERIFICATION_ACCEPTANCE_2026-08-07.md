# eICU Canonical Harmonized Reverification Acceptance — 2026-08-07

**Status:** ACCEPTED CANONICAL INTERNAL EVIDENCE
**Public release:** HOLD pending canonical evidence-page and Model Card reconciliation.

## Execution boundary

- Dataset: eICU v2.0
- Endpoint: harmonized clinical-event
- Rows: 2,023,962
- Patients/stays: 2,394
- Harmonized clinical-event clusters: 12,738
- Repository SHA: `941ced95c4b5a2faca1aa6ab5f9eaa75e97e2d20`
- Canonical scorer: `era.core.scoring::score_review_row`
- Scorer ID: `era-review-score-v1`
- Scorer specification: `1.0.0`
- Scorer SHA-256: `c37dc62136f13d11c7b54502a8ec1ee77facb3f7d889695f5480185a83b796da`
- Input SHA-256: `8da21e0960382260e5f4d48fb1daeade6979acee628852ff48f5d6ffd3b0a110`

## Accepted t=6.0 result

| Metric | Canonical result |
|---|---:|
| Alert-volume reduction vs. standard threshold alerting | 88.53% |
| FPR outside event windows | 2.24% |
| Event-cluster detection | 41.4% |
| Events detected | 5,274 / 12,738 |
| Median retrospective timing context | 4.75 hours |
| ERA alert rows | 78,824 |
| Standard-threshold alert rows | 687,474 |
| ERA alerts / patient-day | 10.9337 |
| Standard threshold alerts / patient-day | 95.3601 |

## Historical harmonized comparison

The previous harmonized t=6.0 values are retained only as historical
reference evidence:

- Alert-volume reduction: 94.25%
- FPR: 0.98%
- Event detection: 24.66%
- Median timing context: 4.83 hours

Canonical minus historical:

- Alert-volume reduction: -5.72 percentage points
- FPR: +1.26 percentage points
- Event-cluster detection: +16.74 percentage points
- Median timing context: -0.08 hours

These differences are accepted as the consequence of applying the
adopted canonical ERA Review Score. Historical values were not treated
as reproduction targets.

## Endpoint separation

The eICU outcome-proxy analysis is a separate historical retrospective
endpoint. It must not be merged, averaged, or represented as the same
endpoint as this harmonized clinical-event analysis.

## Interpretation

Use **alert-volume reduction**, not measured clinician burden reduction.

Use **retrospective timing context**, not proven early detection.

Event-cluster detection and FPR are retrospective characterization
metrics and are not patient-level risk probabilities.

Do not describe this result as prospective clinical validation,
diagnosis, treatment direction, or autonomous escalation.

## Immutable evidence

Canonical harmonized aggregate:

`data/validation/reverified_runs/eicu_canonical_20260807_121023_941ced95c4b5_8da21e096038_harmonized_aggregate.json`

SHA-256: `43b34d9764928a9ea39abdddec11f0db8b479867c83cdd3bc3da3baf3ace410e`

Provenance manifest:

`data/validation/reverified_runs/eicu_canonical_20260807_121023_941ced95c4b5_8da21e096038_manifest.json`

SHA-256: `a372e0cded10b7314a26e17cdbeff36a963d280d06710f1105b228687ea04ac6`

## Evidence repair status

Canonical full-cohort MIMIC reverification: complete and locked.

Canonical full-cohort eICU harmonized reverification: complete and
accepted by this record.

Next evidence-integrity step: rebuild validation evidence surfaces and
the Model Card only from canonical immutable outputs. Public deployment
remains on hold until that reconciliation is complete.
