# HiRID Access Pending Checkpoint

## Status

HiRID third-dataset validation preparation is underway.

A PhysioNet data access request has been submitted for:

**HiRID, a high time-resolution ICU dataset v1.1.1**

Submitted date: April 30, 2026  
Current status: Pending

## Purpose of Request

The requested use is retrospective multi-dataset evaluation of explainable ICU patient-risk prioritization under a harmonized clinical-event framework.

## Current Evidence Base Before HiRID

Early Risk Alert AI currently has locked DUA-safe aggregate retrospective evidence across:

1. MIMIC-IV v3.1 — strict clinical-event retrospective validation.
2. eICU v2.0 — harmonized clinical-event labeling pass.

HiRID is intended to become the third independent ICU dataset once access is approved and the extraction/validation pipeline is completed.

## Important Claim Boundary

Do not yet claim:

- three-dataset validation,
- fully clinically validated,
- validated on HiRID,
- validated across three datasets,
- clinically proven,
- predicts deterioration,
- prevents adverse events.

Safe current wording remains:

> Early Risk Alert AI has retrospective multi-dataset evidence across MIMIC-IV and eICU, with HiRID access pending for a planned third-dataset robustness evaluation.

## DUA / Access Safety

Until approval is granted:

- Do not attempt to access restricted files outside approved PhysioNet flow.
- Do not use someone else’s credentials.
- Do not share restricted access.
- Do not commit raw files, row-level extracts, patient identifiers, timestamps, or enriched restricted-data outputs.
- Keep all restricted-data work inside local-only folders.

## Next Step After Approval

Once HiRID access is approved:

1. Download approved HiRID files through PhysioNet.
2. Store raw files locally only under:
   `data/validation/local_private/hirid/raw/`
3. Run:
   `python3 tools/hirid_local_only_download_audit.py --copy-complete`
4. Build the HiRID-to-ERA cohort extractor.
5. Run local scoring.
6. Run harmonized clinical-event labeling.
7. Generate aggregate-only validation summaries.
8. Compare MIMIC-IV + eICU + HiRID under the harmonized threshold framework.
