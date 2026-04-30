# Harmonized MIMIC-IV + eICU Retrospective Comparison Summary

## Current Status

Early Risk Alert AI now has retrospective evidence across two de-identified ICU datasets:

1. **MIMIC-IV v3.1** — strict clinical-event retrospective validation.
2. **eICU Collaborative Research Database v2.0** — harmonized clinical-event labeling pass using severe physiologic deterioration patterns.

This comparison is DUA-safe and aggregate-only. Raw files, row-level outputs, patient identifiers, timestamps, and enriched CSVs remain local-only.

## Conservative Operating Point Comparison: t=6.0

| Dataset | Event Framing | Alert Reduction | FPR | Detection | Median Lead Time |
|---|---|---:|---:|---:|---:|
| MIMIC-IV | Strict clinical-event clusters | 94.3% | 4.2% | 15.3% | 4.0 hrs |
| eICU | Harmonized clinical-event labels | 94.25% | 0.98% | 24.66% | 4.83 hrs |

## eICU Harmonized Threshold Matrix

| Threshold | Suggested Setting | Alert Reduction | FPR | Detection | Median Lead Time | Alerts / Patient-Day |
|---|---|---:|---:|---:|---:|---:|
| t=4.0 | ICU / high-acuity | 69.74% | 5.34% | 62.51% | 5.67 hrs | 29.2181 |
| t=5.0 | Mixed units / balanced | 87.85% | 2.23% | 38.80% | 5.17 hrs | 11.733 |
| t=6.0 | Telemetry / stepdown conservative | 94.25% | 0.98% | 24.66% | 4.83 hrs | 5.5536 |

## Interpretation

The harmonized eICU clinical-event run strengthens the validation story because it no longer relies only on a mortality/discharge outcome-proxy definition. It uses physiologic deterioration patterns that are closer to the MIMIC-IV clinical-event framing.

The most important finding is not that the detection percentages match exactly. The important finding is that the **threshold behavior is consistent**:

- Lower thresholds increase detection and alert volume.
- Higher thresholds reduce alert burden and false positives.
- Conservative t=6.0 produces strong alert reduction in both datasets.
- Lead-time context remains clinically meaningful in both datasets.

## Safe Public Wording

Early Risk Alert AI has been retrospectively evaluated across MIMIC-IV and eICU using a harmonized threshold framework, with consistent conservative-threshold behavior across both datasets.

## Stronger Internal / Investor Wording

Across two de-identified ICU datasets, Early Risk Alert AI preserved the same operating-point pattern: conservative thresholds substantially reduced review burden and false-positive burden while maintaining meaningful retrospective lead-time context.

## Wording to Avoid for Now

Avoid saying:

- Fully validated on two datasets.
- Clinically proven across two datasets.
- Predicts deterioration.
- Prevents adverse events.
- Replaces standard monitoring.
- Replaces clinician judgment.
- Independently triggers escalation.

## Why Not Say “Fully Validated on Two Datasets” Yet?

The phrase should wait until the validation question, event definitions, inclusion/exclusion criteria, scoring logic, and statistical review are formally locked and reviewed. The current work supports a strong statement of **multi-dataset retrospective robustness**, not final clinical validation.

## Recommended Next Step

The next best step is not another UI redesign. The next step is a locked validation note that clearly distinguishes:

- MIMIC-IV strict clinical-event evidence.
- eICU harmonized clinical-event evidence.
- Earlier eICU outcome-proxy robustness evidence.
- The exact wording approved for pilots, investors, and public pages.
