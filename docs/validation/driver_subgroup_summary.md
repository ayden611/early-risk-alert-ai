# DUA-Safe Driver / Signal-Family Explainability Distribution

## Purpose

This report summarizes which aggregate signal families drive ERA alert prioritization.

This is an explainability distribution report, not a detected-event-cluster attribution report.

## Why Event-Linked Driver Attribution Is Not Published

A pre-event driver-family attribution check was attempted, but source event timestamps did not align with enriched ERA alert timestamps in the pre-event window. To avoid overclaiming, event-linked driver attribution is not published.

Lead-time evidence remains supported by the separate DUA-safe lead-time robustness and event-window sensitivity analyses.

## Results

### Threshold t=4.0

Rows analyzed: 456453  
ERA alert rows: 65821  
Event-linked attribution: Not published

| Driver family | ERA alert rows | Share of ERA alerts | Median risk | Max risk |
|---|---:|---:|---:|---:|
| SpO2 / oxygenation | 41058 | 62.4% | 5.5 | 9.9 |
| Heart rate | 15337 | 23.3% | 5.0 | 9.9 |
| Blood pressure | 7460 | 11.3% | 4.5 | 9.0 |
| Respiratory rate | 1964 | 3.0% | 4.5 | 6.5 |
| Temperature | 2 | 0.0% | 4.5 | 4.5 |

### Threshold t=5.0

Rows analyzed: 456453  
ERA alert rows: 65821  
Event-linked attribution: Not published

| Driver family | ERA alert rows | Share of ERA alerts | Median risk | Max risk |
|---|---:|---:|---:|---:|
| SpO2 / oxygenation | 41058 | 62.4% | 5.5 | 9.9 |
| Heart rate | 15337 | 23.3% | 5.0 | 9.9 |
| Blood pressure | 7460 | 11.3% | 4.5 | 9.0 |
| Respiratory rate | 1964 | 3.0% | 4.5 | 6.5 |
| Temperature | 2 | 0.0% | 4.5 | 4.5 |

### Threshold t=6.0

Rows analyzed: 456453  
ERA alert rows: 65821  
Event-linked attribution: Not published

| Driver family | ERA alert rows | Share of ERA alerts | Median risk | Max risk |
|---|---:|---:|---:|---:|
| SpO2 / oxygenation | 41058 | 62.4% | 5.5 | 9.9 |
| Heart rate | 15337 | 23.3% | 5.0 | 9.9 |
| Blood pressure | 7460 | 11.3% | 4.5 | 9.0 |
| Respiratory rate | 1964 | 3.0% | 4.5 | 6.5 |
| Temperature | 2 | 0.0% | 4.5 | 4.5 |

## Pilot-Safe Interpretation

Use this report to show aggregate explainability:

- Which driver families contribute most to ERA alert prioritization
- Whether alerts are primarily oxygenation-driven, HR-driven, BP-driven, respiratory-rate-driven, or composite
- How explainability shifts across configurable thresholds

Do not use this report to claim a driver family caused or predicted a clinical event.

## Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can provide explainable prioritization context through aggregate signal-family driver summaries.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
