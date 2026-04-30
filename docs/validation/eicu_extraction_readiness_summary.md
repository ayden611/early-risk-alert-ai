# eICU Extraction Readiness Summary

## Status

ERA-format local-only eICU cohort created.

## Aggregate Extraction Summary

| Item | Value |
|---|---:|
| Selected stays | 2500 |
| Selected event-proxy stays | 800 |
| Selected control stays | 1700 |
| Selected stays with vitals | 2394 |
| ERA-format rows written | 2023962 |
| Event-proxy rows written | 772 |
| Event-proxy stays with vitals | 772 |
| VitalPeriodic rows scanned | 146671642 |
| SpO2 source column detected | sao2 |

## Event Definition

This extraction uses an eICU outcome-proxy event definition from mortality/discharge status fields and relative ICU offsets.

This should be framed as:

**Cross-dataset retrospective outcome-proxy evaluation on de-identified ICU data.**

Do not frame it as direct deterioration-event validation, prospective validation, diagnosis, treatment direction, or autonomous escalation.

## DUA-Safe Boundary

Raw eICU files and row-level ERA-format output remain local-only.

Public evidence may include aggregate metrics only.
