# Frontend Evidence QA Fix

Generated: 2026-04-30T12:25:29.112077+00:00

## Purpose

Fix stale frontend placeholders after the multi-dataset evidence upgrade.

## Corrected

- Removed visible `Loading...` table rows on validation pages.
- Populated locked MIMIC-IV threshold matrix values.
- Populated locked eICU outcome-proxy threshold matrix values where applicable.
- Corrected stale command-center document status for Model Card and Pilot Success Guide.
- Added a runtime frontend QA script so stale browser-rendered placeholders are corrected on load.

## Not Changed

- No validation metrics were changed.
- No raw restricted data or row-level derived outputs were exposed.
- No hard ROI claims were added.
- No proven generalizability claim was added.
- No direct equality claim between MIMIC-IV and eICU detection rates was added.

## Public Claim Boundary

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

Decision support only. Retrospective aggregate analysis only.
