# eICU Harmonized Clinical-Event Subcohort B/C Internal Robustness Checkpoint

## Status

Locked internal validation checkpoint.

## Conservative Summary Sentence

Across eICU harmonized clinical-event subcohorts B and C, the conservative t=6.0 operating point showed stable aggregate behavior, with 95.26%–95.64% alert reduction, 1.78%–2.02% FPR, 64.03%–64.49% detection, and 3.13–3.69 hours of retrospective lead-time context.

## What This Supports

This supports an internal robustness conclusion that eICU harmonized clinical-event behavior remains directionally stable across deterministic patient-level subcohorts B and C at the conservative t=6.0 operating point.

## What This Does Not Claim

- It does not claim full clinical validation.
- It does not claim prospective performance.
- It does not claim diagnosis, treatment direction, clinician replacement, or autonomous escalation.
- It does not publish raw eICU data or row-level outputs.

## Locked t=6.0 Subcohort Results

| Run | Dataset | Event definition | Cases | Rows | t=6 alert reduction | t=6 FPR | t=6 detection | Lead-time context | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| eICU harmonized subcohort B | eICU v2.0 | Harmonized clinical-event labels derived from eICU available fields | — | — | 95.64% | 1.78% | 64.49% | 3.69 hrs | locked aggregate |
| eICU harmonized subcohort C | eICU v2.0 | Harmonized clinical-event labels derived from eICU available fields | — | — | 95.26% | 2.02% | 64.03% | 3.13 hrs | locked aggregate |

## Internal Interpretation

The eICU B/C subcohort results are stable enough to support a DUA-safe internal robustness checkpoint. They should be used as supporting evidence behind the broader validation narrative, not as a standalone clinical-performance claim.
