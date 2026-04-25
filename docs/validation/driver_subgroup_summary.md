# DUA-Safe Driver / Signal-Family Subgroup Validation

## Purpose

This test summarizes which aggregate signal families are driving ERA alert prioritization.

The goal is to strengthen explainability without exposing row-level MIMIC-derived data.

Tested thresholds:

- t=4.0 — ICU / high-acuity
- t=5.0 — mixed-unit balanced
- t=6.0 — telemetry / stepdown conservative

Fixed validation assumptions:

- Event window: 6 hours
- Event-cluster gap: 6 hours

No row-level MIMIC-derived data, real MIMIC identifiers, or exact patient-linked timestamps are included.

## Threshold t=4.0

Rows analyzed: 456453  
ERA alert rows: 65821  
Event clusters: 2298  
Detected event clusters: 0

| Driver family | ERA alert rows | Share of ERA alerts | Event-window alert rows | Detected clusters as first driver | Share of detected clusters | Median lead time | Median risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| SpO2 / oxygenation | 41058 | 62.4% | 0 | 0 | — | — | 5.5 |
| Heart rate | 15337 | 23.3% | 0 | 0 | — | — | 5.0 |
| Blood pressure | 7460 | 11.3% | 0 | 0 | — | — | 4.5 |
| Respiratory rate | 1964 | 3.0% | 0 | 0 | — | — | 4.5 |
| Temperature | 2 | 0.0% | 0 | 0 | — | — | 4.5 |

## Threshold t=5.0

Rows analyzed: 456453  
ERA alert rows: 36568  
Event clusters: 2298  
Detected event clusters: 0

| Driver family | ERA alert rows | Share of ERA alerts | Event-window alert rows | Detected clusters as first driver | Share of detected clusters | Median lead time | Median risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| SpO2 / oxygenation | 25709 | 70.3% | 0 | 0 | — | — | 6.5 |
| Heart rate | 7783 | 21.3% | 0 | 0 | — | — | 5.5 |
| Blood pressure | 2840 | 7.8% | 0 | 0 | — | — | 5.5 |
| Respiratory rate | 236 | 0.6% | 0 | 0 | — | — | 5.5 |

## Threshold t=6.0

Rows analyzed: 456453  
ERA alert rows: 19215  
Event clusters: 2298  
Detected event clusters: 0

| Driver family | ERA alert rows | Share of ERA alerts | Event-window alert rows | Detected clusters as first driver | Share of detected clusters | Median lead time | Median risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| SpO2 / oxygenation | 14991 | 78.0% | 0 | 0 | — | — | 7.0 |
| Heart rate | 3404 | 17.7% | 0 | 0 | — | — | 6.5 |
| Blood pressure | 803 | 4.2% | 0 | 0 | — | — | 6.0 |
| Respiratory rate | 17 | 0.1% | 0 | 0 | — | — | 6.0 |

## Interpretation Guidance

Use this report to answer:

- Which signal families most often drive ERA alert prioritization.
- Whether oxygenation, respiratory rate, blood pressure, heart rate, or composite multi-signal patterns dominate review context.
- Whether detected event clusters are associated with interpretable driver families.
- Whether explainability remains meaningful across configurable thresholds.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can provide explainable prioritization context through aggregate signal-family driver summaries.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
