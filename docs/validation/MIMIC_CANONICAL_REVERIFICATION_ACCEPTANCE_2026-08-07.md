# MIMIC-IV Canonical Reverification Acceptance — 2026-08-07

**Status:** ACCEPTED CANONICAL INTERNAL EVIDENCE
**Public release:** HOLD pending canonical eICU full-cohort reverification and evidence-page reconciliation.

## Canonical execution boundary

- Run ID: `mimic_canonical_20260807_155707_t6_0_e375ae5c8fe6_40edc61de435`
- Repository SHA: `e375ae5c8fe62a56a09ebcb2594ddd29e27a1065`
- Canonical scorer: `era.core.scoring::score_review_row`
- Scorer ID: `era-review-score-v1`
- Scorer specification: `1.0.0`
- Scorer file SHA-256: `c37dc62136f13d11c7b54502a8ec1ee77facb3f7d889695f5480185a83b796da`
- Reconstructed cohort SHA-256: `40edc61de4356f39521523f63139bbb573f9232099c4744703599c7a85d0485a`
- Rows: 456,453
- Patients: 1,705

## Accepted t=6.0 result

| Metric | Canonical result |
|---|---:|
| Alert-volume reduction vs. standard threshold alerting | 93.9% |
| ERA FPR | 4.5% |
| Event-cluster detection | 14.8% |
| Patient detection | 14.9% |
| Median retrospective timing context | 4.0 hours |
| ERA alerts / patient-day | 0.6859 |
| Standard threshold alerts / patient-day | 11.2743 |

## Historical comparison

The earlier MIMIC t=6.0 values of 94.3% alert reduction, 4.2% FPR,
15.3% detection, and 4.0-hour timing are retained only as historical
reference values. They are not the canonical product evidence baseline.

Canonical minus historical:

- Alert-volume reduction: -0.4 percentage points
- ERA FPR: +0.3 percentage points
- Event-cluster detection: -0.5 percentage points
- Median timing context: +0.0 hours

## Canonical-vs-legacy scorer divergence

At t=6.0:

- 15,253 of 456,453 rows changed alert classification.
- Alert-classification disagreement: 3.3416%.
- Canonical-only alert rows: 8,208.
- Legacy-only alert rows: 7,045.

The near-stability of aggregate results does not mean the scoring
implementations were equivalent. The divergence artifact demonstrates
that they were materially different.

## Interpretation

The canonical MIMIC result supports retrospective characterization of
ERA Review Score behavior on the reconstructed historical full cohort.

Use **alert-volume reduction**, not measured clinician burden reduction.

Use **retrospective timing context**, not proven early detection.

Do not describe this as prospective validation, diagnosis, treatment
direction, autonomous escalation, or patient-level risk probability.

## Evidence artifacts

- Canonical aggregate:
  `data/validation/reverified_runs/mimic_canonical_20260807_155707_t6_0_e375ae5c8fe6_40edc61de435_aggregate.json`
  SHA-256: `472cecf9b8d03a2bda29c652c3e69dcd11d04f61cc7430550b9dc6f37be1af2e`

- Canonical-vs-legacy divergence:
  `data/validation/reverified_runs/mimic_canonical_20260807_155707_t6_0_e375ae5c8fe6_40edc61de435_legacy_divergence.json`
  SHA-256: `3648e906b0b009227642ebde75ce3195795ad7a473d62432bd98721045fb2647`

## Next required evidence step

Canonical full-cohort eICU reverification through
`era.core.scoring::score_review_row`.

Public evidence pages remain under hold until that step and subsequent
evidence-page reconciliation are complete.
