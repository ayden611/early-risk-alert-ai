# Early Risk Alert AI — Model Card

## Intended Use

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.

It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.

## Evidence Status

- MIMIC-IV strict clinical-event retrospective validation.
- eICU second-dataset outcome-proxy robustness check.
- DUA-safe aggregate-only public outputs.
- Raw restricted files and row-level derived files remain local-only.

## Current Limitation

MIMIC-IV and eICU use different event definitions. Detection rates should not be treated as equivalent endpoint definitions until a harmonized event definition is run across both datasets.
