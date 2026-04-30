# Stability-First Platform Polish

## Purpose

This patch stabilizes the live platform presentation while preserving the restored Command Center layout.

## Completed

- Restored/protected Command Center layout by avoiding forced frontend overrides.
- Fixed Model Card and Pilot Success Guide readiness paths.
- Fixed fragile `Loading...` states on Validation Evidence and Validation Runs with locked aggregate fallback content.
- Added minimal explainability visibility to existing Command Center HTML sections.
- Preserved conservative DUA-safe wording.

## Claim Boundary

Do not claim full clinical validation.

Approved wording:

> Early Risk Alert AI has retrospective multi-dataset evidence across MIMIC-IV and eICU using a harmonized threshold framework, with consistent conservative-threshold behavior across datasets.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
