# Multi-Dataset Robustness Summary

Generated: 2026-04-30T02:05:47.355329+00:00

## Top-Line Summary

**MIMIC-IV established strict clinical-event cross-cohort retrospective stability, while eICU added a separate second-dataset outcome-proxy check; across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.**

At the conservative t=6.0 operating point, MIMIC-IV showed 4 hrs median lead-time context across locked cross-cohort evidence, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.

## Dataset Comparison

| Dataset | Evidence Role | Cases | Rows | Event Context | Threshold | Alert Reduction | FPR | Detection | Lead Time |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort validation release | 577–1,705 | 456,453 in full validation cohort | event clusters / clinical-event labels | t=6.0 | 94%–94.9% | 3.7%–4.4% | 14.1%–16% | 4 hrs |
| eICU Collaborative Research Database v2.0 | Second-dataset retrospective outcome-proxy check | 2394 | 2023962 | 772 | t=6.0 | 96.8% | 1.8% | 66.6% | 3.41 hrs |

## Interpretation

MIMIC-IV established strict clinical-event cross-cohort retrospective stability.

eICU added a separate second-dataset outcome-proxy check.

The strongest defensible finding is not that the two detection rates are directly comparable. The strongest defensible finding is that ERA preserved the same threshold-direction behavior across both datasets.

## Citation Story

### MIMIC-IV

Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10, 1. https://doi.org/10.1038/s41597-022-01899-x

### eICU

Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2WM1R

Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. https://doi.org/10.1038/sdata.2018.178

### PhysioNet

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220. RRID:SCR_007345.

## DUA-Safe Boundary

Aggregate public evidence only.

Do not publish:

- raw restricted CSV/GZ files
- row-level enriched exports
- real restricted identifiers
- exact row-level timestamps tied to individual records
- patient-level rows
- local mapping files

## Claim Guardrails

Use:

**Cross-dataset retrospective robustness evidence across de-identified ICU datasets.**

Avoid:

- proven generalizability
- prospective validation
- diagnosis
- treatment direction
- prevention of adverse events
- replacement of clinician judgment
- autonomous escalation
- direct superiority over standard monitoring

Decision support only. Retrospective aggregate analysis only.
