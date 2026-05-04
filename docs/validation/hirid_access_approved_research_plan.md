# HiRID v1.1.1 Access Approved — Local-Only Retrospective Aggregate Research Plan

## Status

HiRID v1.1.1 access has been approved by PhysioNet.

This approval allows Early Risk Alert AI to begin a third-dataset retrospective aggregate evaluation workflow. It does **not** mean HiRID validation has been completed, and it should not be presented publicly as completed validation until aggregate outputs are generated, reviewed, and locked.

## Current Public Evidence Status

Early Risk Alert AI currently has locked DUA-safe aggregate retrospective evidence across:

1. MIMIC-IV retrospective clinical-event evaluation.
2. eICU retrospective harmonized clinical-event / outcome-proxy evaluation.

HiRID is now approved as the next third-dataset retrospective aggregate evaluation track.

## HiRID Guardrails

Raw HiRID files, row-level records, timestamps, patient-level outputs, identifiers, derived row-level extracts, and restricted files must remain local-only.

Do not upload restricted HiRID files to:

- GitHub
- Render
- Wix
- public websites
- public demos
- investor-facing pages
- production software
- public cloud storage

## Public Wording Until Results Are Complete

Approved wording:

> HiRID v1.1.1 access has been approved for local-only retrospective aggregate research. HiRID is not yet included in the locked public validation evidence base. Current public validation evidence remains MIMIC-IV + eICU until HiRID aggregate results are completed and reviewed.

Do not claim:

- three-dataset validation completed
- prospective validation
- clinical validation
- diagnosis
- treatment direction
- patient-level performance
- autonomous escalation
- outcome prevention
- live clinical deployment validation

## Local Folder Policy

Approved local-only workspace:

- `data/validation/local_private/hirid/raw/`
- `data/validation/local_private/hirid/working/`
- `data/validation/local_private/hirid/audit/`

Public aggregate outputs may later be written only after review to:

- `data/validation/`
- `docs/validation/`

Only aggregate summary metrics may be used publicly.
