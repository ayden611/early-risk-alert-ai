# eICU Validation Preparation

## Purpose

Prepare a local-only eICU extraction workflow for a second-dataset retrospective check.

## Required Source Files

- patient.csv.gz
- vitalPeriodic.csv.gz
- vitalAperiodic.csv.gz
- diagnosis.csv.gz
- apachePatientResult.csv.gz

## Local-Only Rule

Raw eICU files and row-level derived outputs must remain local-only under:

`data/validation/local_private/eicu/`

Do not commit or publish:

- raw eICU CSV/GZ files
- row-level ERA-formatted eICU cohort files
- patient-level rows
- exact patient-linked timestamps
- any row-level derived restricted data

## Intended Output Later

The extraction should create a local-only ERA-format cohort with:

- patient_id
- timestamp
- heart_rate
- spo2
- bp_systolic
- bp_diastolic
- respiratory_rate
- temperature_f
- clinical_event
- event_label
- event_group

Public outputs should be aggregate-only validation summaries.

## Claim Guardrails

Use:

“Cross-dataset retrospective evaluation on de-identified ICU datasets.”

Avoid:

- proven generalizability
- prospective validation
- prediction claim
- diagnosis claim
- treatment direction claim
- autonomous escalation claim

Decision support only. Retrospective aggregate analysis only.
