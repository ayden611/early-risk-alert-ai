# HiRID Third-Dataset Readiness Plan

## Purpose

Prepare Early Risk Alert AI for a third retrospective ICU dataset validation pass using HiRID.

## Current Evidence Position

Early Risk Alert AI currently has locked aggregate retrospective evidence across:

1. MIMIC-IV v3.1 — strict clinical-event retrospective validation.
2. eICU v2.0 — harmonized clinical-event labeling pass.

HiRID would serve as a third independent ICU dataset from Bern University Hospital / ETH Zurich.

## Why HiRID Matters

HiRID is a high-time-resolution ICU dataset with almost 34,000 ICU admissions and bedside monitoring observations recorded at high frequency. It is clinically relevant for validating an explainable deterioration-prioritization and command-center workflow-support platform.

## Claim Boundary

Do not claim "fully clinically validated" after adding HiRID.

Safer target wording after successful HiRID completion:

> Early Risk Alert AI has three-dataset retrospective validation evidence under a harmonized evaluation framework across MIMIC-IV, eICU, and HiRID.

Even stronger wording requires formal review of:

- event definitions,
- inclusion / exclusion criteria,
- threshold logic,
- denominator definitions,
- lead-time computation,
- statistical review,
- prospective pilot evidence.

## DUA Safety

Raw HiRID files, row-level outputs, patient identifiers, timestamps, partition-level extracts, and enriched CSVs remain local-only.

Public outputs may include only:

- aggregate threshold matrices,
- alert reduction,
- false-positive burden,
- detection percentage,
- median lead-time summary,
- high-level methodology,
- limitations,
- approved wording.

## Next Technical Steps

1. Download approved HiRID files from PhysioNet after access is granted.
2. Store raw files in:
   `data/validation/local_private/hirid/raw/`
3. Run the HiRID local download audit.
4. Build a HiRID-to-ERA cohort extractor.
5. Add ERA scores locally.
6. Run harmonized clinical-event labeling.
7. Generate DUA-safe aggregate summary only.
8. Compare MIMIC-IV + eICU + HiRID under the same threshold framework.
