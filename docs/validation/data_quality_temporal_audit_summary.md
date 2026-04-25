# DUA-Safe Data Quality + Temporal Alignment Audit

## Purpose

This audit checks whether the validation dataset and enriched ERA export are suitable for continued retrospective testing.

It summarizes:

- Schema readiness
- Timestamp validity
- Duplicate case-time rows
- Event-label integrity
- Vital-sign missingness
- Enriched export alignment
- Aggregate temporal-alignment diagnostics

No row-level MIMIC-derived data, real identifiers, or exact patient-linked timestamps are included.

## Source Dataset Audit

| Metric | Value |
|---|---:|
| Rows | 456453 |
| Unique cases | 1705 |
| Valid timestamp rows | 456453 |
| Valid timestamp % | 100.0% |
| Time span | 35892.99 days |
| Duplicate case-time rows | 0 |
| Chronology issues in original order | 17783 |
| Median rows per case | 144 |
| Event-labeled rows | 12518 |
| Event clusters, 6h gap | 1672 |

## Vital Coverage

| Signal | Column Present | Missing Rows | Missing % |
|---|---:|---:|---:|
| heart_rate | True | 5958 | 1.31% |
| spo2 | True | 24196 | 5.3% |
| bp_systolic | True | 142641 | 31.25% |
| bp_diastolic | True | 142714 | 31.27% |
| respiratory_rate | True | 11743 | 2.57% |
| temperature_f | True | 330208 | 72.34% |

## Enriched ERA Export Audit

| Metric | Value |
|---|---:|
| Rows | 456453 |
| Unique cases | 1705 |
| Source/enriched case-time overlap rows | 456453 |
| Source/enriched overlap % of source | 100.0% |
| Risk score coverage % | 100.0% |
| Median risk score | 1.5 |
| Max risk score | 9.9 |
| ERA alert rows in current export state | 19215 |

## Aggregate Temporal Alignment Diagnostic

This diagnostic explains why driver subgroups are treated as explainability distributions, while official lead-time evidence remains in the dedicated lead-time robustness test.

| Threshold | Pre-window cluster hits | Pre-window hit % | Post-window cluster hits | Post-window hit % |
|---:|---:|---:|---:|---:|
| t=4.0 | 253 | 18.01% | 646 | 45.98% |
| t=5.0 | 253 | 18.01% | 646 | 45.98% |
| t=6.0 | 253 | 18.01% | 646 | 45.98% |

## Pilot-Safe Interpretation

This audit strengthens the validation package by documenting dataset quality and temporal-alignment behavior before additional MIMIC testing.

Use this as quality-control evidence, not as clinical performance proof.

## Pilot-Safe Claim

A DUA-safe aggregate audit found the validation dataset and enriched ERA export can be reviewed for schema coverage, event-label integrity, timestamp quality, and temporal-alignment behavior without exposing row-level data.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
